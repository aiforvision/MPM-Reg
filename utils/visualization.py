"""Compact training visualizations (mid-slice PNGs written to the log dir)."""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def _slice(img):
    """Central axial slice of [B, 1, D, H, W] or [B, D, H, W] as 2D numpy array."""
    img = img.detach().float().cpu()
    if img.dim() == 5:
        img = img[:, 0]
    return img[0, img.shape[1] // 2].numpy()


def plot_pretrain_epoch(source_img, target_img, gt_pos, pred_pos, log_dir, epoch):
    """Stage 1: masked position modeling. Scatter of ground-truth vs predicted
    patch positions (projected on the central axial plane)."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(_slice(source_img), cmap='gray')
    axs[0].set_title('input (shuffled patches)')
    axs[1].imshow(_slice(target_img), cmap='gray')
    axs[1].set_title('target')
    gt = gt_pos[0].detach().float().cpu()
    pred = pred_pos[0].detach().float().cpu()
    axs[2].scatter(gt[:, 2], gt[:, 1], s=4, c='lime', label='gt')
    axs[2].scatter(pred[:, 2], pred[:, 1], s=4, c='magenta', label='pred')
    axs[2].invert_yaxis()
    axs[2].set_aspect('equal')
    axs[2].legend()
    axs[2].set_title('patch positions (H/W plane)')
    for ax in axs[:2]:
        ax.axis('off')
    fig.suptitle(f'epoch {epoch}')
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, f'pretrain_epoch_{epoch}.png'), dpi=120)
    plt.close(fig)


def plot_registration_epoch(source_img, target_img, deformed_img, flow_px, log_dir, epoch):
    """Stages 2/3: registration. Mid slices of moving, fixed, warped image,
    green/magenta overlay and displacement-field magnitude."""
    src, tgt, wrp = _slice(source_img), _slice(target_img), _slice(deformed_img)
    flow = flow_px.detach().float().cpu()[0]  # [D, H, W, 3]
    flow_mag = flow.norm(dim=-1)[flow.shape[0] // 2].numpy()

    def _norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    overlay = torch.stack([torch.tensor(_norm(wrp)), torch.tensor(_norm(tgt)), torch.tensor(_norm(wrp))], dim=-1).numpy()

    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    for ax, img, title in zip(
            axs,
            [src, tgt, wrp, overlay, flow_mag],
            ['moving', 'fixed', 'warped moving', 'overlay (warped/fixed)', '|flow| [px]']):
        im = ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(title)
        ax.axis('off')
    fig.colorbar(im, ax=axs[-1], fraction=0.046)
    fig.suptitle(f'epoch {epoch}')
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, f'registration_epoch_{epoch}.png'), dpi=120)
    plt.close(fig)
