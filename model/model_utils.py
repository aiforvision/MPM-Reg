import numpy as np
import torch
from pathlib import Path
import os
import glob
import torch.distributed as dist


class PositionalEncoding_posEmb_nd:
    def __init__(self, d_hid, theta_max=np.pi,):
        self.d_hid = d_hid
        self.d = None
        self.h = None
        self.w = None
        self.num_parameters = None

    def get_position_angle_vec_n2(self, position):
        x, y = position
        pos_x = x / np.power(10000, 4*np.arange(self.d_hid//2)/self.d_hid)
        pos_y = y / np.power(10000, 4*np.arange(self.d_hid//2)/self.d_hid)
        return np.concatenate((pos_x, pos_y))

    def get_position_angle_vec_n3(self, position):
        x, y, theta = position
        pos_x = x / np.power(10000, 8*np.arange(self.d_hid//4)/self.d_hid)
        pos_y = y / np.power(10000, 8*np.arange(self.d_hid//4)/self.d_hid)
        pos_theta = theta / np.power(10000, 8*np.arange(self.d_hid//4)/self.d_hid)
        pos_none = np.zeros_like(pos_x)
        return np.concatenate((pos_x, pos_y, pos_theta, pos_none))

    def encode_positions(self, pos, img_size):
        B, n_patches, num_param = pos.shape
        if self.num_parameters is None:
            self.num_parameters = num_param
        pos = pos.astype(np.float32)

        # normalization to range [0, 1]
        if self.num_parameters == 2:
            self.h = img_size[0]
            self.w = img_size[1]
            pos = np.stack([pos[..., 0]/self.h, pos[..., 1]/self.w], axis=-1)
        elif self.num_parameters == 3:
            self.d = img_size[0]
            self.h = img_size[1]
            self.w = img_size[2]
            pos = np.stack([pos[..., 0]/self.d, pos[..., 1]/self.h, pos[..., 2]/self.w], axis=-1)
        else:
            raise ValueError("Only 2 or 3 parameters are supported")

        pe = np.zeros((B, n_patches, self.d_hid))

        if self.num_parameters == 2:
            for b in range(B):
                for n in range(n_patches):
                    pe[b, n] = self.get_position_angle_vec_n2(pos[b, n])
                    pe[b, n, 0::2] = np.sin(pe[b, n, 0::2])  # dim 2i
                    pe[b, n, 1::2] = np.cos(pe[b, n, 1::2])  # dim 2i+1
        elif self.num_parameters == 3:
            for b in range(B):
                for n in range(n_patches):
                    pe[b, n] = self.get_position_angle_vec_n3(pos[b, n])
                    pe[b, n, 0::2] = np.sin(pe[b, n, 0::2])
                    pe[b, n, 1::2] = np.cos(pe[b, n, 1::2])

        self.encoded_pe = pe

        return torch.FloatTensor(pe)


def auto_load_model(args, model, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)

    if args.misc.auto_resume and len(args.resume) == 0:
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
        if len(args.resume) == 0:
            print("No checkpoint found to resume - starting from scratch")
        else:
            print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
        if missing:
            print(f"Missing keys when loading checkpoint (expected for new layers): {missing}")
        if unexpected:
            print(f"Unexpected keys when loading checkpoint: {unexpected}")
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.training.start_epoch = checkpoint['epoch'] + 1
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
            except ValueError as e:
                print(f"Could not load optimizer state (model architecture changed): {e}")
                print("Starting optimizer from scratch, keeping model weights.")


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }

        save_on_master(to_save, checkpoint_path)


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def is_main_process():
    return get_rank() == 0


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
