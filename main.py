"""MPM-Reg training entry point.

Stage 1 (masked position modeling pre-training):  ++training_scheme=single
Stage 2 (multi-dataset registration training):    ++training_scheme=unpaired ++resume=<stage-1 checkpoint>
Stage 3 (dataset-specific fine-tuning):           ++training_scheme=unpaired ++resume=<stage-2 checkpoint>
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import datetime
import numpy as np
from time import time
import torch
import torch.backends.cudnn as cudnn
import monai
import os
import sys
import wandb
import torch.nn as nn

from loss.smoothness_loss import SmoothnessLoss
from model import mpm_reg
from model.model_utils import auto_load_model, save_model
from data.dataset import RegistrationDataset, PretrainDataset
from augmentation.augmentation import brightnessAugmentor
from training.training_utils import NativeScalerWithGradNormCount as NativeScaler
from training.training_utils import create_optimizer, cosine_scheduler
from training.training_engine import train_one_epoch
from training.pretraining_engine import pretrain_one_epoch


@hydra.main(config_path="conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):

    # check if gpu is available
    if torch.cuda.is_available():
        print("GPU available")
    else:
        print("GPU not available")
        sys.exit(1)

    # note: training runs with a constant learning rate (as in the paper);
    # warmup/min lr from the config are overridden here
    cfg.training.min_lr = cfg.training.lr
    cfg.training.warmup_lr = cfg.training.lr

    if cfg.log_wandb:
        run = wandb.init(project=cfg.wandb_project, config=OmegaConf.to_container(cfg, resolve=True))

    # Create output directory
    if cfg.output_dir:
        os.makedirs(cfg.output_dir, exist_ok=True)
        cfg.log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(cfg.log_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        print("Saving to:", cfg.output_dir, " Logging to:", cfg.log_dir)
    else:
        raise NotImplementedError("Output directory not specified")
    print(OmegaConf.to_yaml(cfg))

    # Set up device and random seed
    device = torch.device(cfg.misc.device)
    cudnn.benchmark = True
    # a seed passed via ++training.seed overrides cfg.misc.seed
    training_seed = cfg.training.get('seed', None)
    if training_seed is not None:
        cfg.misc.seed = training_seed
    seed = cfg.misc.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load model
    if cfg.training_scheme == 'unpaired':
        model = mpm_reg.mpm_reg_registration_3D_bspline(
            patch_size=cfg.model.patch_size,
            projection_dim=cfg.model.projection_dim,
            channels=cfg.model.channels,
            output_parameters=cfg.model.output_parameters,
            training_scheme=cfg.training_scheme,
            encoder_num_heads=cfg.model.encoder_num_heads,
            decoder_num_heads=cfg.model.decoder_num_heads,
            encoder_depth=cfg.model.encoder_depth,
            decoder_depth=cfg.model.decoder_depth,
            give_spatial_cues=cfg.single_image.give_spatial_cues,
        )
    elif cfg.training_scheme == 'single':
        model = mpm_reg.mpm_reg_pretrain_3D(
            patch_size=cfg.model.patch_size,
            projection_dim=cfg.model.projection_dim,
            channels=cfg.model.channels,
            output_parameters=cfg.model.output_parameters,
            training_scheme=cfg.training_scheme,
            encoder_num_heads=cfg.model.encoder_num_heads,
            decoder_num_heads=cfg.model.decoder_num_heads,
            encoder_depth=cfg.model.encoder_depth,
            decoder_depth=cfg.model.decoder_depth,
            give_spatial_cues=cfg.single_image.give_spatial_cues,
        )
    else:
        raise NotImplementedError(f"Training scheme not implemented: {cfg.training_scheme}")

    enc_dim = model.encoder.embed_dim
    dec_dim = model.decoder.embed_dim

    # Get datasets
    if cfg.training_scheme == 'unpaired':
        dataset_train = RegistrationDataset(mode='train', args=cfg, embed_dim_enc=enc_dim, embed_dim_dec=dec_dim, return_mask=False, augment_heavy=cfg.augmentation.heavy)
        dataset_val = RegistrationDataset(mode='val', args=cfg, embed_dim_enc=enc_dim, embed_dim_dec=dec_dim, return_mask=True, augment_heavy=cfg.augmentation.heavy)
    else:
        dataset_train = PretrainDataset(mode='train', args=cfg, embed_dim_enc=enc_dim, embed_dim_dec=dec_dim, augment_heavy=cfg.augmentation.heavy)
        dataset_val = PretrainDataset(mode='val', args=cfg, embed_dim_enc=enc_dim, embed_dim_dec=dec_dim, augment_heavy=cfg.augmentation.heavy)

    # Set augmentors
    if cfg.augmentation.brightness:
        augmentor = brightnessAugmentor()
    else:
        augmentor = brightnessAugmentor(augment=False)

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = create_optimizer(cfg, model)
    print("Model = %s" % str(model))
    print('number of params: {} M'.format(n_parameters / 1e6))

    loss_scaler = NativeScaler()
    if cfg.training.num_training_steps_per_epoch is not None:
        num_training_steps_per_epoch = cfg.training.num_training_steps_per_epoch
        if num_training_steps_per_epoch >= len(dataset_train) // cfg.data.batch_size_train:
            num_training_steps_per_epoch = len(dataset_train) // cfg.data.batch_size_train
    else:
        num_training_steps_per_epoch = len(dataset_train) // cfg.data.batch_size_train
    print("Number of training steps per epoch: ", num_training_steps_per_epoch)

    train_sampler = torch.utils.data.RandomSampler(dataset_train, num_samples=num_training_steps_per_epoch)
    val_sampler = torch.utils.data.RandomSampler(dataset_val, num_samples=cfg.training.num_val_steps_per_epoch if cfg.training.num_val_steps_per_epoch < len(dataset_val) else len(dataset_val))

    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=cfg.data.batch_size_train,
                                                   sampler=train_sampler,
                                                   num_workers=cfg.misc.num_workers,
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   )
    dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                 batch_size=cfg.data.batch_size_val,
                                                 sampler=val_sampler,
                                                 num_workers=cfg.misc.num_workers,
                                                 pin_memory=True,
                                                 drop_last=True)
    print(f"Number of training samples: {len(dataset_train)}")
    print(f"Number of validation samples: {len(dataset_val)}")

    auto_load_model(args=cfg, model=model, optimizer=optimizer, loss_scaler=loss_scaler)

    # if model is loaded, adapt number of training epochs
    if cfg.resume != '':
        cfg.training.epochs = cfg.training.start_epoch + cfg.training.epochs

    lr_schedule_values = cosine_scheduler(
        cfg.training.lr, cfg.training.min_lr, cfg.training.epochs, num_training_steps_per_epoch,
        warmup_epochs=cfg.training.warmup_epochs, warmup_steps=cfg.training.warmup_steps, start_warmup_value=cfg.training.warmup_lr)
    if cfg.training.weight_decay_end is None:
        cfg.training.weight_decay_end = cfg.training.weight_decay
    wd_schedule_values = cosine_scheduler(
        cfg.training.weight_decay, cfg.training.weight_decay_end, cfg.training.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # setup losses
    if cfg.loss.loss == 'mse':
        loss_image = nn.MSELoss()
    elif cfg.loss.loss == 'mi':
        loss_image = monai.losses.GlobalMutualInformationLoss(kernel_type='gaussian', num_bins=32)
    elif cfg.loss.loss == 'lncc':
        loss_image = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=3, kernel_size=cfg.loss.lncc_kernel_size)
    loss_smoothness = SmoothnessLoss()

    if cfg.loss.track_mi:
        loss_mi = monai.losses.GlobalMutualInformationLoss(kernel_type='gaussian', num_bins=32)
    else:
        loss_mi = None
    if cfg.loss.track_lncc:
        loss_lncc = monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=3, kernel_size=cfg.loss.lncc_kernel_size)
    else:
        loss_lncc = None

    # Start training
    print(f"Start training for {cfg.training.epochs} epochs")
    start_time = time()
    top3_loss = []
    top3_loss_paths = []
    improvement_counter = 0

    # Initialize global step counter for wandb logging
    global_step = [cfg.training.start_epoch * num_training_steps_per_epoch]
    # Anchor the max-train-steps cap to the resume point so it counts additional
    # steps from here, not absolute steps since epoch 0.
    initial_global_step = global_step[0]

    if cfg.model.compile:
        print("Compiling model")
        model = torch.compile(model, )
    else:
        print("Not compiling model")

    for epoch in range(cfg.training.start_epoch, cfg.training.epochs):
        print(f"Epoch {epoch}/{cfg.training.epochs}")

        if cfg.training_scheme == 'unpaired':
            train_stats = train_one_epoch(
                    model=model,
                    data_loader=dataloader_train,
                    data_loader_val=dataloader_val,
                    loss_image=loss_image,
                    loss_smoothness=loss_smoothness,
                    loss_mi=loss_mi,
                    loss_lncc=loss_lncc,
                    optimizer=optimizer, device=device,
                    epoch=epoch, loss_scaler=loss_scaler,
                    start_steps=epoch * num_training_steps_per_epoch,
                    lr_schedule_values=lr_schedule_values,
                    wd_schedule_values=wd_schedule_values,
                    patch_size=cfg.model.patch_size,
                    log_dir=cfg.log_dir,
                    augmentor=augmentor,
                    cfg=cfg,
                    global_step=global_step
                )
        else:
            train_stats = pretrain_one_epoch(
                    model=model,
                    data_loader=dataloader_train,
                    data_loader_val=dataloader_val,
                    loss_image=loss_image,
                    optimizer=optimizer, device=device,
                    epoch=epoch, loss_scaler=loss_scaler,
                    start_steps=epoch * num_training_steps_per_epoch,
                    lr_schedule_values=lr_schedule_values,
                    wd_schedule_values=wd_schedule_values,
                    patch_size=cfg.model.patch_size,
                    log_dir=cfg.log_dir,
                    augmentor=augmentor,
                    cfg=cfg,
                    global_step=global_step
            )

        # save model - only save the top 3 best models
        if cfg.output_dir:
            checkp_path = os.path.join(cfg.output_dir, f"checkpoint-{epoch}.pth")
            if len(top3_loss) < 3:
                top3_loss.append(train_stats['loss_val'])
                top3_loss_paths.append(checkp_path)
                save_model(
                    args=cfg, model=model, model_without_ddp=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
                print(f"Epoch is {epoch} - Saving model to {checkp_path}")
                improvement_counter = 0
            else:
                top3_loss.append(train_stats['loss_val'])
                top3_loss_paths.append(checkp_path)
                top3_loss = np.array(top3_loss)
                top3_loss_paths = np.array(top3_loss_paths)
                idx = np.argsort(top3_loss)
                top3_loss = top3_loss[idx]
                top3_loss_paths = top3_loss_paths[idx]

                if top3_loss[-1] > train_stats['loss_val']:
                    print(f"Epoch is {epoch} - and new loss is improved - Saving model to {checkp_path}, removing {top3_loss_paths[-1]}")
                    improvement_counter = 0
                    # remove the worst checkpoint
                    os.remove(top3_loss_paths[-1])
                    top3_loss = top3_loss[:-1]
                    top3_loss_paths = top3_loss_paths[:-1]
                    top3_loss = top3_loss.tolist()
                    top3_loss_paths = top3_loss_paths.tolist()
                    save_model(args=cfg,
                               model=model,
                               model_without_ddp=model,
                               optimizer=optimizer,
                               loss_scaler=loss_scaler,
                               epoch=epoch)
                    assert len(top3_loss) == len(top3_loss_paths)
                    assert len(top3_loss) == 3
                else:
                    print(f"Epoch is {epoch} - but loss is not improved - not saving model")
                    improvement_counter += 1
                    top3_loss = top3_loss[:-1]
                    top3_loss_paths = top3_loss_paths[:-1]
                    top3_loss = top3_loss.tolist()
                    top3_loss_paths = top3_loss_paths.tolist()
                    assert len(top3_loss) == len(top3_loss_paths)
                    assert len(top3_loss) == 3

        # early stopping
        if cfg.training.early_stopping is not None and improvement_counter >= cfg.training.early_stopping:
            print(f"Early stopping at epoch {epoch} - improvement counter is {improvement_counter}")
            sys.exit(0)

        # max total training steps stop (evaluated after each epoch).
        # When resuming, the cap counts additional steps from the resume point.
        if getattr(cfg.training, 'max_total_train_steps', None) is not None:
            max_step = initial_global_step + cfg.training.max_total_train_steps
            if global_step[0] >= max_step:
                print(f"Reached max_total_train_steps ({cfg.training.max_total_train_steps}) at epoch {epoch}, global_step={global_step[0]} (started at {initial_global_step}) - stopping.")
                break

    total_time = time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()
