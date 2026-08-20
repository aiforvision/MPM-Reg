"""Stages 2 and 3: unpaired registration training / fine-tuning."""

import os
import sys
from typing import Iterable

import torch
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from time import time
import wandb

from utils.visualization import plot_registration_epoch


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, data_loader_val: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, patch_size,
                    loss_image, loss_smoothness, loss_mi=None, loss_lncc=None, max_norm: float = None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, log_dir=None, augmentor=None,
                    cfg=None, global_step=None):

    start_time_epoch = time()
    model.train()

    val_freq = 1
    log_freq = 10
    train_stats = {}
    loss_train = []
    loss_val = []
    val_img_losses = []
    val_smoothness_losses = []
    val_mi_losses = []
    val_lncc_losses = []

    for step, batch in enumerate(data_loader):
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

        # move data to device
        batch = [(item.to(device), pe_enc.to(device), pe_dec.to(device), pos.to(device)) for item, pe_enc, pe_dec, pos in batch]

        images, pe_enc_images, pe_dec_images, pos = batch[0]
        augm_images, _, pe_dec_augm, pos_gt = batch[1]

        images = augmentor(images.clone())
        augm_images = augmentor(augm_images.clone())

        B, D, H, W = images.shape

        n = D*H*W//(patch_size[0]*patch_size[1]*patch_size[2])

        mask_target = torch.ones((B, n), dtype=bool).to(device)
        mask_source = torch.ones((B, n), dtype=bool).to(device)

        # input to model
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs, outputs_img, flow, flow_px = model(x=images,
                                                        x_augm=augm_images,
                                                        pe_enc_x=pe_enc_images,
                                                        pe_dec_x=pe_dec_images,
                                                        pe_dec_augm=pe_dec_augm,
                                                        mask_target=mask_target,
                                                        mask_source=mask_source,
                                                        positions=pos.clone(),
                                                        regularization_positions=pos.clone())

            images = images.unsqueeze(1)
            augm_images = augm_images.unsqueeze(1)

            imgLoss_value = loss_image(outputs_img.double(), augm_images.double())
            imgLoss_value_scaled = imgLoss_value * cfg.loss.imgLoss_scale_factor

            smoothReg_value = loss_smoothness(flow_px.float().squeeze(1))  # float() for numerical stability of the smoothness term
            smoothReg_value_scaled = smoothReg_value * cfg.loss.smoothReg_scale_factor

            loss_value = imgLoss_value_scaled + smoothReg_value_scaled

        if loss_mi is not None:
            mi_value = loss_mi(outputs_img.double(), augm_images.double())
        if loss_lncc is not None:
            lncc_value = loss_lncc(outputs_img.double(), augm_images.double())

        if not math.isfinite(loss_value.item()):
            print(f"Loss is not finite. Stopping training. img {imgLoss_value_scaled} smooth {smoothReg_value_scaled}")
            sys.exit()

        optimizer.zero_grad()
        grad_norm = loss_scaler(loss_value, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=False)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        end_time = time()
        step_time = end_time - start_time

        loss_train.append(loss_value.item())

        if cfg.log_wandb and it % log_freq == 0:
            wandb.log({
                    "train - Unscaled and auxiliary Losses/image_loss": imgLoss_value.item(),
                    "train - Unscaled and auxiliary Losses/smoothness_loss": smoothReg_value.item(),
                    "train - Unscaled and auxiliary Losses/mutual_information_loss": mi_value.item() if cfg.loss.track_mi else 1,
                    "train - Unscaled and auxiliary Losses/lncc_loss": lncc_value.item() if cfg.loss.track_lncc else 1,

                    "Time/step_time": step_time,

                    "train - Losses/image_loss": imgLoss_value_scaled.item(),
                    "train - Losses/smoothness_loss": smoothReg_value_scaled.item(),
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

        for step, batch in enumerate(data_loader_val):

            batch = [(item.to(device), mask.to(device), pe_enc.to(device), pe_dec.to(device), pos.to(device)) for item, mask, pe_enc, pe_dec, pos in batch]

            images, images_mask, pe_enc_images, pe_dec_images, pos = batch[0]
            augm_images, augm_images_mask, _, pe_dec_augm, pos_gt = batch[1]

            images = augmentor(images.clone())
            augm_images = augmentor(augm_images.clone())

            B, D, H, W = images.shape

            n = D*H*W//(patch_size[0]*patch_size[1]*patch_size[2])

            mask_target = np.ones((B, n), dtype=bool)
            mask_source = np.ones((B, n), dtype=bool)

            # input to model
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs, outputs_img, flow, flow_px = model(x=images,
                                                                x_augm=augm_images,
                                                                pe_enc_x=pe_enc_images,
                                                                pe_dec_x=pe_dec_images,
                                                                pe_dec_augm=pe_dec_augm,
                                                                mask_target=mask_target,
                                                                mask_source=mask_source,
                                                                positions=pos.clone(),
                                                                regularization_positions=pos.clone())

                images = images.unsqueeze(1)
                augm_images = augm_images.unsqueeze(1)

                imgLoss_value_val = loss_image(outputs_img.double(), augm_images.double())
                imgLoss_value_val_scaled = imgLoss_value_val * cfg.loss.imgLoss_scale_factor

                smoothReg_value_val = loss_smoothness(flow_px.float().squeeze(1))
                smoothReg_value_val_scaled = smoothReg_value_val * cfg.loss.smoothReg_scale_factor

                loss_value_val = imgLoss_value_val_scaled + smoothReg_value_val_scaled

            if not math.isfinite(loss_value_val.item()):
                print(f"Loss is not finite. Stopping training. img {imgLoss_value_val_scaled} smooth {smoothReg_value_val_scaled}")
                sys.exit()

            if loss_mi is not None:
                mi_value_val = loss_mi(outputs_img.double(), augm_images.double())
            if loss_lncc is not None:
                lncc_value_val = loss_lncc(outputs_img.double(), augm_images.double())

            loss_val.append(loss_value_val.item())
            val_img_losses.append(imgLoss_value_val.item())
            val_smoothness_losses.append(smoothReg_value_val.item())
            if loss_mi is not None:
                val_mi_losses.append(mi_value_val.item())
            if loss_lncc is not None:
                val_lncc_losses.append(lncc_value_val.item())

    train_stats['loss_train'] = np.mean(loss_train)
    train_stats['loss_val'] = np.mean(loss_val)

    # Log validation metrics after the validation loop
    if cfg.log_wandb and len(loss_val) > 0:
        val_step = current_global_step if 'current_global_step' in locals() else it

        val_log_dict = {
                "val - Unscaled and auxiliary Losses/image_loss": np.mean(val_img_losses),
                "val - Unscaled and auxiliary Losses/smoothness_loss": np.mean(val_smoothness_losses),
                "val - Losses/image_loss": np.mean(val_img_losses) * cfg.loss.imgLoss_scale_factor,
                "val - Losses/smoothness_loss": np.mean(val_smoothness_losses) * cfg.loss.smoothReg_scale_factor,
                "val - Losses/loss": train_stats['loss_val'],
        }

        if len(val_mi_losses) > 0:
            val_log_dict["val - Unscaled and auxiliary Losses/mutual_information_loss"] = np.mean(val_mi_losses)
        if len(val_lncc_losses) > 0:
            val_log_dict["val - Unscaled and auxiliary Losses/lncc_loss"] = np.mean(val_lncc_losses)

        wandb.log(val_log_dict, step=val_step)

    plot_registration_epoch(source_img=images, target_img=augm_images, deformed_img=outputs_img,
                            flow_px=flow_px, log_dir=log_dir, epoch=epoch)

    return train_stats
