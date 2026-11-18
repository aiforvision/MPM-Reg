"""Register a pair of 3D volumes with a trained MPM-Reg checkpoint (stage 2 or 3; B-spline model or MPM-Reg V2
with the dense head, detected from the checkpoint).

Warps the moving volume towards the fixed volume and writes the warped volume
and the dense displacement field (in voxels) as NIfTI files.

Example:
    python register.py --checkpoint checkpoint.pth --moving moving.nii.gz --fixed fixed.nii.gz --output_dir out/

Optional test-time optimisation (TTO) on top of MPM-Reg with the anatomix authors' FireANTs registration
(third-party code, see tto/anatomix_fireants.py and the README), e.g. for an MR (moving) to CT (fixed) pair:
    python register.py ... --tto anatomix_fireants --tto_fixed_clip -450 450
writes warped_moving_tto.nii.gz and flow_px_tto.nii.gz in addition.
"""

import argparse
import os
import shlex
import time

import numpy as np
import torch
import nibabel as nib

from model import mpm_reg
from data.data_utils import PositionalEncoding_posEmb_nd
from data.preprocessing import load_volume, preprocess_volume, grid_positions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='stage-2 or stage-3 checkpoint (.pth)')
    parser.add_argument('--moving', required=True, help='moving volume')
    parser.add_argument('--fixed', required=True, help='fixed volume')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--patch_size', type=int, nargs=3, default=[16, 16, 16])
    parser.add_argument('--projection_dim', type=int, default=1024)
    parser.add_argument('--encoder_num_heads', type=int, default=12)
    parser.add_argument('--decoder_num_heads', type=int, default=6)
    parser.add_argument('--encoder_depth', type=int, default=12)
    parser.add_argument('--decoder_depth', type=int, default=8)
    parser.add_argument('--norm_mode', default='minmax', choices=['minmax', 'meanstd'])
    parser.add_argument('--downsample_factor', type=int, default=1)
    parser.add_argument('--max_axis_size', type=int, default=None)
    parser.add_argument('--max_volume_size', type=int, default=256)
    parser.add_argument('--device', default='cuda')
    # optional test-time optimisation (tto/anatomix_fireants.py; anatomix + FireANTs by their authors, see README)
    parser.add_argument('--tto', default='none', choices=['none', 'anatomix_fireants'],
                        help='refine the MPM-Reg field with the anatomix + FireANTs registration (default: off)')
    parser.add_argument('--tto_start', default='mpmreg', choices=['mpmreg', 'identity'],
                        help="start from MPM-Reg's field (default) or from the identity (the authors' method alone)")
    parser.add_argument('--tto_recipe', default='default', choices=['default', 'abdomen'],
                        help="the anatomix authors' settings: their CLI defaults or their abdominal MR->CT example")
    parser.add_argument('--tto_fixed_clip', type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                        help='intensity clip of the fixed volume before min-max (anatomix: CT -450 450)')
    parser.add_argument('--tto_moving_clip', type=float, nargs=2, default=None, metavar=('LO', 'HI'))
    parser.add_argument('--tto_fixed_mask', default=None, help='optional foreground mask of the fixed volume')
    parser.add_argument('--tto_moving_mask', default=None, help='optional foreground mask of the moving volume')
    parser.add_argument('--tto_flags', default='',
                        help="further anatomix-register flags, e.g. '--smooth-warp-sigma 1.0' (smoother) or 0.25 (more flexible)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.backends.cudnn.conv.fp32_precision = 'ieee'   # fp32 convolutions without TF32, as in training (see main.py)

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    # MPM-Reg V2 checkpoints contain the dense head, the paper's checkpoints the B-spline model
    head = 'dense' if any(k.startswith('dense_head.') for k in checkpoint['model']) else 'spline'
    model = mpm_reg.mpm_reg_registration_3D_bspline(
        patch_size=args.patch_size,
        projection_dim=args.projection_dim,
        channels=1,
        output_parameters=3,
        training_scheme='unpaired',
        encoder_num_heads=args.encoder_num_heads,
        decoder_num_heads=args.decoder_num_heads,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        head=head,
    )
    model.load_state_dict(checkpoint['model'], strict=True)
    print(f"Loaded checkpoint {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")
    model.to(device)
    model.eval()

    def _prep(path):
        return preprocess_volume(
            load_volume(path), patch_size=args.patch_size, norm_mode=args.norm_mode,
            downsample_factor=args.downsample_factor, max_axis_size=args.max_axis_size,
            max_volume_size=args.max_volume_size, path=path)

    moving = _prep(args.moving)
    fixed = _prep(args.fixed)

    # both volumes must share one grid: crop to the common patch-aligned shape
    common = [min(a, b) for a, b in zip(moving.shape, fixed.shape)]
    moving = moving[:common[0], :common[1], :common[2]]
    fixed = fixed[:common[0], :common[1], :common[2]]

    D, H, W = moving.shape
    pos = grid_positions((D, H, W), args.patch_size)
    pe_enc = PositionalEncoding_posEmb_nd(model.encoder.embed_dim).encode_positions(pos, [D, H, W])
    pe_dec = PositionalEncoding_posEmb_nd(model.decoder.embed_dim).encode_positions(pos, [D, H, W])

    n = pos.shape[0]
    mask_target = np.ones((1, n), dtype=bool)
    mask_source = np.ones((1, n), dtype=bool)

    moving_b = moving.unsqueeze(0).to(device)
    fixed_b = fixed.unsqueeze(0).to(device)
    pe_enc = pe_enc.unsqueeze(0).to(device)
    pe_dec = pe_dec.unsqueeze(0).to(device)
    pos_b = pos.unsqueeze(0).to(device)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == 'cuda'):
            outputs, warped, flow, flow_px = model(
                x=moving_b, x_augm=fixed_b,
                pe_enc_x=pe_enc, pe_dec_x=pe_dec, pe_dec_augm=pe_dec,
                mask_target=mask_target, mask_source=mask_source,
                positions=pos_b.clone(), regularization_positions=pos_b.clone())

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    print(f'MPM-Reg: {time.time() - t0:.2f} s')

    os.makedirs(args.output_dir, exist_ok=True)
    affine = np.eye(4)
    warped_np = warped.squeeze().float().cpu().numpy()
    flow_np = flow_px.squeeze(0).float().cpu().numpy()  # [D, H, W, 3], displacement in voxels
    nib.save(nib.Nifti1Image(warped_np, affine), os.path.join(args.output_dir, 'warped_moving.nii.gz'))
    nib.save(nib.Nifti1Image(moving.numpy(), affine), os.path.join(args.output_dir, 'moving_preprocessed.nii.gz'))
    nib.save(nib.Nifti1Image(fixed.numpy(), affine), os.path.join(args.output_dir, 'fixed_preprocessed.nii.gz'))
    nib.save(nib.Nifti1Image(flow_np, affine), os.path.join(args.output_dir, 'flow_px.nii.gz'))
    print(f"Saved warped volume and displacement field to {args.output_dir}")

    if args.tto != 'none':
        from tto.anatomix_fireants import refine, warp

        def _raw(path):   # same geometry as the MPM-Reg input, raw intensities (the authors clip and normalise)
            vol = preprocess_volume(
                load_volume(path), patch_size=args.patch_size, norm_mode=None,
                downsample_factor=args.downsample_factor, max_axis_size=args.max_axis_size,
                max_volume_size=args.max_volume_size, path=path)
            return vol[:common[0], :common[1], :common[2]]

        u_tto, info = refine(
            _raw(args.moving), _raw(args.fixed), u0=flow_px.float() if args.tto_start == 'mpmreg' else None,
            moving_mask=_raw(args.tto_moving_mask) if args.tto_moving_mask else None,
            fixed_mask=_raw(args.tto_fixed_mask) if args.tto_fixed_mask else None,
            moving_clip=args.tto_moving_clip, fixed_clip=args.tto_fixed_clip, recipe=args.tto_recipe,
            flags=shlex.split(args.tto_flags), device=device, work_dir=args.output_dir)
        warped_tto = warp(moving_b.unsqueeze(1).float(), u_tto)
        nib.save(nib.Nifti1Image(warped_tto.squeeze().cpu().numpy(), affine), os.path.join(args.output_dir, 'warped_moving_tto.nii.gz'))
        nib.save(nib.Nifti1Image(u_tto.squeeze(0).cpu().numpy(), affine), os.path.join(args.output_dir, 'flow_px_tto.nii.gz'))
        print(f"TTO (anatomix + FireANTs, start: {args.tto_start}): {info['time_s']:.1f} s, peak GPU memory "
              f"{info['peak_mem_gb']:.1f} GB; saved warped_moving_tto / flow_px_tto to {args.output_dir}")


if __name__ == "__main__":
    main()
