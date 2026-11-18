"""Optional test-time optimisation (TTO) on top of MPM-Reg: the anatomix + FireANTs registration, started from
MPM-Reg's displacement field.

Attribution
-----------
The registration method, its features (anatomix-dev-vit + MIND-SSC) and all of the code that performs it belong to
their authors and are used here unmodified, as an external add-on installed from their repositories
(requirements-tto.txt). MPM-Reg claims no ownership of them.
  * anatomix: N. Dey, B. Billot, H. E. Wong, C. J. Wang, M. Ren, P. E. Grant, A. V. Dalca, P. Golland, "Learning
    General-Purpose Biomedical Volume Representations using Randomized Synthesis", ICLR 2025, arXiv:2411.02372.
    Code: https://github.com/neel-dey/anatomix (MIT licence), used at commit 4c86aee.
  * FireANTs: R. Jena, P. Chaudhari, J. C. Gee, "FireANTs: Adaptive Riemannian Optimization for Multi-Scale
    Diffeomorphic Registration", Nature Communications 2024. Code: https://github.com/rohitrango/fireants (Apache-2.0);
    anatomix uses the fork https://github.com/neel-dey/FireANTs, here at commit 21ae072.
`_register` is adapted from anatomix/registration/anatomix-register.py::register_pair (anatomix commit 4c86aee): the
same steps (their CLI parser and defaults, pipeline.load_pair, pipeline.FeatureExtractor, run_registration with the
moving features re-extracted before every stage), calling their functions. `RECIPES['abdomen']` is their Learn2Reg
AbdomenMRCT example (anatomix/registration/README.md).

What MPM-Reg adds
-----------------
MPM-Reg's field u0 as the starting point: the moving volume (and mask) are pre-warped with u0 in MPM-Reg's
convention, the authors' registration runs unchanged on the pre-warped pair (fixed-to-moving grid g1), and both are
composed into one displacement field in MPM-Reg's convention: moved(p) = moving(g0(g1(p))) with g0(g1) = g1 - u0(g1)
(u0 is sampled with border padding, so the identity part stays exact where g1 leaves the field of view).

Inputs are volumes on MPM-Reg's preprocessed grid with their raw intensities (so that the authors' clipping, e.g. CT
to [-450, 450] HU, applies) and voxel spacing 1. Requires a CUDA GPU (8-9 GB for a 192x160x192 pair).
"""
import os
import tempfile
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

# extra flags for the authors' registration CLI; 'default' = their CLI defaults (one deformable stage, 6x4x2x1,
# 100 iterations per level, step 1.0, smoothing sigmas 1.0 / 0.5, anatomix-dev-vit + MIND-SSC, local CC)
RECIPES = {
    'default': [],
    'abdomen': ['--transform', 'deformable', '--step-size', '1.0', '--shrink-factors', '6x4x2x1',
                '--iterations', '100x100x100x100', '--cc-kernel-widths', '21x13x11x9'],
}


def _px_to_norm(u_px):
    d, h, w = u_px.shape[1:4]
    return u_px / torch.tensor([w / 2, h / 2, d / 2], device=u_px.device, dtype=u_px.dtype)


def _identity(shape_dhw, device):
    return F.affine_grid(torch.eye(3, 4, device=device)[None], (1, 1) + tuple(shape_dhw), align_corners=True)


def warp(vol, u_px, mode='bilinear'):
    """MPM-Reg's warp (model/mpm_reg.py): vol (1, C, D, H, W), u_px (1, D, H, W, 3) in voxels, channels (x, y, z);
    warped(p) = vol(p - u(p)), border padding."""
    grid = _identity(vol.shape[2:], vol.device) - _px_to_norm(u_px.to(vol.dtype))
    return F.grid_sample(vol, grid, align_corners=True, padding_mode='border', mode=mode)


def _write(vol_dhw, path):
    """NIfTI whose array FireANTs reads as (z, y, x) = MPM-Reg's (D, H, W); identity affine (spacing 1)."""
    arr = vol_dhw.detach().float().cpu().numpy() if torch.is_tensor(vol_dhw) else np.asarray(vol_dhw, np.float32)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(arr.transpose(2, 1, 0)), np.eye(4)), path)


def _register(argv, device, models):
    """Adapted from anatomix-register.py::register_pair (anatomix 4c86aee): one pair through the authors' pipeline.
    Returns their normalised fixed-to-moving sampling grid (1, D, H, W, 3) and the number of feature channels."""
    from anatomix.registration.registration_infrastructure.fused_ops import configure
    configure(argv)   # their fused-CUDA-ops switch (default off), as at the top of anatomix-register.py
    from anatomix.registration.registration_infrastructure import cli, pipeline
    from anatomix.registration.registration_infrastructure.features import load_backbone
    from anatomix.registration.registration_infrastructure.register import run_registration
    from anatomix.registration.registration_infrastructure.warp_io import as_batch, load_image, warp_volume

    args = cli.build_parser().parse_args(argv)
    pairs, _, stages = cli.prepare(args)
    pair = pairs[0]
    key = (args.backbone, args.features)
    if args.features in cli.MODEL_FEATURES and key not in models:
        models[key] = load_backbone(args.backbone, device, custom_arch=args.custom_arch, custom_weights=args.custom_weights,
                                    unet_kwargs=args.unet_kwargs, vit_kwargs=args.vit_kwargs)
    extract = pipeline.FeatureExtractor(args, models.get(key))
    inputs = pipeline.load_pair(pair, args, device)
    stage_specs = pipeline.resolve_stage_losses(stages, inputs.has_masks)
    masked = any(spec['loss'].startswith('masked_') for spec in stage_specs)
    fixed_batch = extract(inputs.fixed, inputs.fixed_norm, inputs.fixed_spacing, inputs.fixed_mask, masked)

    def moving_features(grid):
        with torch.no_grad():
            warped = warp_volume(inputs.moving_norm, grid, 'bilinear')
            warped_mask = warp_volume(inputs.moving_mask, grid, 'nearest') if inputs.moving_mask is not None else None
            on_fixed_grid = load_image(pair['fixed'], device)   # reloaded for its geometry
            return extract(on_fixed_grid, warped, inputs.fixed_spacing, warped_mask, masked)

    result = run_registration(fixed_batch, as_batch(inputs.moving), stage_specs, initialization=args.initialization,
                              init_images=None, initial_transform=None, reextract_moving=moving_features,
                              has_mask_channel=masked, devices=[device], verbose=args.verbose)
    return result.warped_coordinates.detach(), int(fixed_batch().shape[1])


def refine(moving_raw, fixed_raw, u0=None, moving_mask=None, fixed_mask=None, moving_clip=None, fixed_clip=None,
           recipe='default', flags=(), device='cuda', work_dir=None, models=None):
    """Test-time optimisation of one pair.

    moving_raw, fixed_raw: (D, H, W) volumes on MPM-Reg's grid, raw intensities. u0: MPM-Reg's field (1, D, H, W, 3)
    in voxels, or None to start from the identity (= the authors' registration alone). masks: optional (D, H, W)
    foreground masks (then the authors' masked CC loss is used). *_clip: optional (lo, hi) intensity clip before
    their min-max normalisation. recipe / flags: the authors' settings (RECIPES) plus further anatomix-register flags,
    e.g. ['--smooth-warp-sigma', '1.0']. models: dict that keeps the loaded backbone across calls.
    Returns the refined field (1, D, H, W, 3) in voxels (MPM-Reg's convention) and a dict with the time in seconds.
    """
    device = torch.device(device)
    flags = list(RECIPES[recipe]) + list(flags)
    assert not {'--initialization', '--initial-transform'} & set(flags), 'the start is MPM-Reg\'s field (or the identity)'
    models = {} if models is None else models
    shape = tuple(moving_raw.shape)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.time()
    moving = torch.as_tensor(moving_raw, dtype=torch.float32, device=device)[None, None]
    mmask = torch.as_tensor(moving_mask, dtype=torch.float32, device=device)[None, None] if moving_mask is not None else None
    if u0 is not None:   # pre-warp with MPM-Reg's field
        u0 = u0.float().to(device)
        moving = warp(moving, u0)
        mmask = warp(mmask, u0, mode='nearest') if mmask is not None else None
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        f = lambda name: os.path.join(tmp, name)
        _write(fixed_raw, f('fixed.nii.gz')); _write(moving[0, 0], f('moving.nii.gz'))
        argv = ['--fixed', f('fixed.nii.gz'), '--moving', f('moving.nii.gz')]
        if fixed_mask is not None:
            _write(fixed_mask, f('fixed_mask.nii.gz')); argv += ['--fixed-mask', f('fixed_mask.nii.gz')]
        if mmask is not None:
            _write(mmask[0, 0], f('moving_mask.nii.gz')); argv += ['--moving-mask', f('moving_mask.nii.gz')]
        for role, clip in (('fixed', fixed_clip), ('moving', moving_clip)):
            if clip is not None:
                argv += [f'--{role}-minclip', str(clip[0]), f'--{role}-maxclip', str(clip[1])]
        g1, channels = _register(argv + flags + ['--device', str(device)], device, models)
    identity = _identity(shape, device)
    if u0 is None:
        G = g1
    else:   # g0(g1) = g1 - u0(g1), u0 border-padded
        G = g1 - F.grid_sample(_px_to_norm(u0).permute(0, 4, 1, 2, 3), g1, align_corners=True,
                               padding_mode='border').permute(0, 2, 3, 4, 1)
    d, h, w = shape
    u = (identity - G) * torch.tensor([w / 2, h / 2, d / 2], device=device)
    torch.cuda.synchronize(device)
    return u, dict(time_s=time.time() - t0, feature_channels=channels,
                   peak_mem_gb=torch.cuda.max_memory_allocated(device) / 2 ** 30)
