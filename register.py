"""Register a pair of 3D volumes with a trained MPM-Reg checkpoint (stage 2 or 3).

Warps the moving volume towards the fixed volume and writes the warped volume
and the dense displacement field (in voxels) as NIfTI files.

Example:
    python register.py --checkpoint checkpoint.pth --moving moving.nii.gz --fixed fixed.nii.gz --output_dir out/
"""

import argparse
import os

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
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

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
    )
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
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

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == 'cuda'):
            outputs, warped, flow, flow_px = model(
                x=moving_b, x_augm=fixed_b,
                pe_enc_x=pe_enc, pe_dec_x=pe_dec, pe_dec_augm=pe_dec,
                mask_target=mask_target, mask_source=mask_source,
                positions=pos_b.clone(), regularization_positions=pos_b.clone())

    os.makedirs(args.output_dir, exist_ok=True)
    affine = np.eye(4)
    warped_np = warped.squeeze().float().cpu().numpy()
    flow_np = flow_px.squeeze(0).float().cpu().numpy()  # [D, H, W, 3], displacement in voxels
    nib.save(nib.Nifti1Image(warped_np, affine), os.path.join(args.output_dir, 'warped_moving.nii.gz'))
    nib.save(nib.Nifti1Image(moving.numpy(), affine), os.path.join(args.output_dir, 'moving_preprocessed.nii.gz'))
    nib.save(nib.Nifti1Image(fixed.numpy(), affine), os.path.join(args.output_dir, 'fixed_preprocessed.nii.gz'))
    nib.save(nib.Nifti1Image(flow_np, affine), os.path.join(args.output_dir, 'flow_px.nii.gz'))
    print(f"Saved warped volume and displacement field to {args.output_dir}")


if __name__ == "__main__":
    main()
