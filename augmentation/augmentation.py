import numpy as np
from torchvision.transforms.v2 import Compose, ToTensor, Lambda
import torch
import torch.nn.functional as F
import random
# note: monai's Compose (imported below) intentionally shadows torchvision's
from monai.transforms import Compose
from monai.transforms import RandGaussianNoise, RandGaussianSmooth, RandAdjustContrast, RandScaleIntensity


class brightnessAugmentor():
    def __init__(self, min_noise=-0.2, max_noise=0.2, min_scale=0.9, max_scale=1.1, min_offset=-0.2, max_offset=0.2, augment=True):
        # ranges chosen for min-max scaled data
        if augment:
            self.transform = Compose([
                Lambda(lambda x: x * np.random.uniform(min_scale, max_scale)),
                Lambda(lambda x: x + np.random.uniform(min_offset, max_offset)),
                Lambda(lambda x: x + torch.randn_like(x) * np.random.uniform(min_noise, max_noise)),
                ToTensor(),
            ])
        else:
            self.transform = Compose([
                ToTensor()
            ])

    def __call__(self, x,):
        """ x: torch.Tensor with dimensions (B, C, H, W) """
        for n, x_n in enumerate(x):
            x[n] = self.transform(x_n)
        return x


class Heavy3DAugmentor:
    def __init__(self, augment=True):
        self.augment = augment
        # Define MONAI augmentations that support 3D
        self.monai_aug = Compose([
            RandScaleIntensity(factors=0.3, prob=0.7),
            RandAdjustContrast(prob=0.7, gamma=(0.7, 1.3)),
            RandGaussianNoise(prob=0.5, mean=0.0, std=0.05),
            RandGaussianSmooth(prob=0.2, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        ])

    @staticmethod
    def random_convolution_3d(volume, kernel_size=3):
        """Apply random 3D convolution with normalized kernel."""
        kernel = torch.randn(1, 1, kernel_size, kernel_size, kernel_size, device=volume.device)
        kernel = kernel / kernel.abs().sum()  # normalize to prevent large intensity jumps
        pad = kernel_size // 2
        volume = volume.unsqueeze(0).unsqueeze(0)
        out = F.conv3d(volume, kernel, padding=pad)
        out = out.squeeze()
        # Normalize to [0,1]
        out = (out - out.min()) / (out.max() - out.min() + 1e-8)
        return out

    @staticmethod
    def solarize(volume, threshold=0.5):
        """Invert intensities above threshold."""
        return torch.where(volume > threshold, 1.0 - volume, volume)

    @staticmethod
    def speckle_noise(volume, std=0.1):
        """Add multiplicative (speckle) noise and renormalize."""
        noise = torch.randn_like(volume) * std
        out = volume + volume * noise
        out = torch.clamp(out, 0.0, 1.0)
        return out

    @staticmethod
    def invert(volume):
        """Invert intensity values."""
        return 1.0 - volume

    def __call__(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform heavy data augmentation on 3D volumes.
        Args:
            batch_tensor: torch.Tensor (B, D, H, W), values in [0, 1]
        Returns:
            Augmented tensor of same shape with values ~ [0, 1].
        """

        if not self.augment:
            return batch_tensor

        # check if input is torch.Tensor
        if not isinstance(batch_tensor, torch.Tensor):
            # check if input is numpy array
            if isinstance(batch_tensor, np.ndarray):
                batch_tensor = torch.from_numpy(batch_tensor)
            else:
                raise ValueError(f"Input must be torch.Tensor or numpy array, got {type(batch_tensor)}")

        # handle inputs without batch dimension
        if len(batch_tensor.shape) == 3:
            batch_tensor = batch_tensor.unsqueeze(0)
            is_batch = False
        else:
            is_batch = True

        B, D, H, W = batch_tensor.shape
        augmented = []

        for i in range(B):
            vol = batch_tensor[i].clone()

            # scale intensity to [0, 1]
            vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)

            # Apply MONAI augmentations
            if random.random() < 0.9:
                vol = self.monai_aug(vol.unsqueeze(0)).squeeze(0)

            # Custom random effects
            if random.random() < 0.3:
                vol = self.speckle_noise(vol, std=random.uniform(0.05, 0.2))

            if random.random() < 0.2:
                vol = self.solarize(vol, threshold=random.uniform(0.3, 0.7))

            if random.random() < 0.2:
                vol = self.invert(vol)

            if random.random() < 0.5:
                vol = self.random_convolution_3d(vol, kernel_size=random.choice([3, 5]))

            # Ensure values remain in [0,1] for each volume
            vol = torch.clamp(vol, 0.0, 1.0)
            augmented.append(vol)

        if not is_batch:
            augmented = augmented[0]
        else:
            augmented = torch.stack(augmented)

        return augmented
