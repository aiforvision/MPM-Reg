"""Stage 1: single-image masked position modeling pre-training."""

import os
import sys
from typing import Iterable

import torch
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from time import time
import wandb

from utils.visualization import plot_pretrain_epoch


def pretrain_one_epoch(model: torch.nn.Module, data_loader: Iterable, data_loader_val: Iterable,
                       optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, patch_size,
                       loss_image, max_norm: float = None, start_steps=None,
                       lr_schedule_values=None, wd_schedule_values=None, log_dir=None, augmentor=None,
                       cfg=None, global_step=None):

    start_time_epoch = time()
    model.train()

    val_freq = 1
    train_stats = {}
    loss_train = []
    loss_val = []
    val_mse_losses = []

    # ratio of positional embeddings masked as MPM (1 = all positions masked)
    mt_ratio = cfg.single_image.get('mt_ratio', 1)

    for step, batch_data in enumerate(data_loader):
        start_time = time()
        it = start_steps + step
        if global_step is not None:
            current_global_step = global_step[0]
            global_step[0] += 1
        else:
            current_global_step = it

        # set lr and weight decay values
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        batch, foreground_mask, dataset = batch_data

        batch = [(item.to(device), pe_enc.to(device), pe_dec.to(device), pos.to(device)) for item, pe_enc, pe_dec, pos in batch]

        images, pe_enc_images, pe_dec_images, grid_pos = batch[0]
        augm_images, pos, pe_dec_augm, gt_pos = batch[1]

        images = augmentor(images.clone())
        augm_images = augmentor(augm_images.clone())

        B, D, H, W = images.shape

        n = D*H*W//(patch_size[0]*patch_size[1]*patch_size[2])

        mask_target = torch.ones((B, n)) > 0  # bool tensor of value true
        mask_source = torch.rand((B, n)) > mt_ratio
        mask_source = ~(mask_source.clone())
        mask_target = mask_target.to(images.device)
        mask_source = mask_source.to(images.device)

        if B > 1:
            gt_pos = [gt_pos[i][mask_source[0]] for i in range(B)]
            gt_pos = torch.stack(gt_pos, axis=0)
        else:
            gt_pos = gt_pos[mask_source]

        if B == 1:
            gt_pos = gt_pos.unsqueeze(0)

        # input to model
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(x=images,
                            x_augm=augm_images,
                            pe_enc_x=pe_enc_images,
                            pe_dec_x=pe_dec_images,
                            pe_dec_augm=pe_dec_augm,
                            mask_target=mask_target,
                            mask_source=mask_source,
                            positions=None,
                            regularization_positions=None)

            mse_value = loss_image(outputs, gt_pos)
            mse_value_scaled = mse_value * cfg.loss.imgLoss_scale_factor

            loss_value = mse_value_scaled

            if not math.isfinite(loss_value.item()):
                print(f"Loss is not finite. Stopping training.")
                sys.exit()

        optimizer.zero_grad()
        grad_norm = loss_scaler(loss_value, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=False)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        end_time = time()
        step_time = end_time - start_time

        loss_train.append(loss_value.item())

        if cfg.log_wandb:
            wandb.log({
                    "train - Unscaled and auxiliary Losses/mse_loss": mse_value.item(),
                    "Time/step_time": step_time,

                    "train - Losses/mse_loss": mse_value_scaled.item(),
                    "train - Losses/loss": loss_value.item(),

                    "train - Gradient scaling/loss_scale": loss_scale_value,
                    "train - Gradient scaling/grad_norm": grad_norm
                }, step=current_global_step)

    end_time_epoch = time()
    epoch_time = end_time_epoch - start_time_epoch
    if cfg.log_wandb:
        wandb.log({
            "Time/epoch_time": epoch_time
        }, step=current_global_step if 'current_global_step' in locals() else it)

    if epoch % val_freq == 0:
        print(f"Validation at epoch {epoch}")
        model.eval()

        for step, batch_data in enumerate(data_loader_val):

            batch, foreground_mask, dataset = batch_data

            batch = [(item.to(device), pe_enc.to(device), pe_dec.to(device), pos.to(device)) for item, pe_enc, pe_dec, pos in batch]

            images, pe_enc_images, pe_dec_images, grid_pos = batch[0]
            augm_images, pos, pe_dec_augm, gt_pos = batch[1]

            images = augmentor(images.clone())
            augm_images = augmentor(augm_images.clone())

            B, D, H, W = images.shape

            n = D*H*W//(patch_size[0]*patch_size[1]*patch_size[2])

            mask_target = torch.ones((B, n)) > 0  # bool tensor of value true
            mask_source = torch.rand((B, n)) > mt_ratio
            mask_source = ~(mask_source.clone())
            mask_target = mask_target.to(images.device)
            mask_source = mask_source.to(images.device)

            if B > 1:
                gt_pos = [gt_pos[i][mask_source[0]] for i in range(B)]
                gt_pos = torch.stack(gt_pos, axis=0)
            else:
                gt_pos = gt_pos[mask_source]

            if B == 1:
                gt_pos = gt_pos.unsqueeze(0)

            # input to model
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(x=images,
                                x_augm=augm_images,
                                pe_enc_x=pe_enc_images,
                                pe_dec_x=pe_dec_images,
                                pe_dec_augm=pe_dec_augm,
                                mask_target=mask_target,
                                mask_source=mask_source,
                                positions=None,
                                regularization_positions=None)

            mse_value_val = loss_image(outputs, gt_pos)
            mse_value_val_scaled = mse_value_val * cfg.loss.imgLoss_scale_factor

            loss_value_val = mse_value_val_scaled

            loss_val.append(loss_value_val.item())
            val_mse_losses.append(mse_value_val.item())

    train_stats['loss_train'] = np.mean(loss_train)
    train_stats['loss_val'] = np.mean(loss_val)

    # Log validation metrics after the validation loop
    if cfg.log_wandb and len(loss_val) > 0:
        val_step = current_global_step if 'current_global_step' in locals() else it
        wandb.log({
                "val - Unscaled and auxiliary Losses/mse_loss": np.mean(val_mse_losses),
                "val - Losses/mse_loss": np.mean(val_mse_losses) * cfg.loss.imgLoss_scale_factor,
                "val - Losses/loss": train_stats['loss_val'],
        }, step=val_step)

    plot_pretrain_epoch(source_img=augm_images, target_img=images,
                        gt_pos=gt_pos, pred_pos=outputs, log_dir=log_dir, epoch=epoch)

    return train_stats
