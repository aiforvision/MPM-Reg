"""Generic datasets for the three MPM-Reg training stages.

Each entry in `data.data_dirs` is a directory containing a `train.csv` and a
`val.csv` with a `filepath` column (absolute paths to 3D volumes) and an
optional `mask_filepath` column (segmentation masks, only used for
visualization/validation). Pairs for the registration stages are sampled
within the same data directory.

The per-volume preprocessing (see data/preprocessing.py) replicates the
configuration used in the paper.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.data_utils import PositionalEncoding_posEmb_nd
from data.preprocessing import load_volume, preprocess_volume, grid_positions
from augmentation.augmentation import Heavy3DAugmentor


class RegistrationDataset(Dataset):
    """Unpaired intra-dataset registration (stages 2 and 3)."""

    def __init__(self, mode, args, embed_dim_enc, embed_dim_dec, return_mask=False, augment_heavy=False):
        self.mode = mode
        self.args = args
        self.return_mask = return_mask
        self.patch_size = args.model.patch_size
        self.augment_heavy = augment_heavy
        if augment_heavy:
            self.augmentor_heavy = Heavy3DAugmentor()
        else:
            self.augmentor_heavy = Heavy3DAugmentor(augment=False)

        self.pe_enc = PositionalEncoding_posEmb_nd(embed_dim_enc)
        self.pe_dec = PositionalEncoding_posEmb_nd(embed_dim_dec)

        self.first_run = True

        self.file_paths = []
        self.mask_paths = []
        self.dataset = []
        for data_dir in args.data.data_dirs:
            csv_path = os.path.join(data_dir, f'{mode}.csv')
            df = pd.read_csv(csv_path)
            paths = df['filepath'].values
            if 'mask_filepath' in df.columns:
                masks = df['mask_filepath'].values
            else:
                masks = [None] * len(paths)
            keep = [i for i in range(len(paths)) if not pd.isna(paths[i])]
            self.file_paths += [paths[i] for i in keep]
            self.mask_paths += [masks[i] for i in keep]
            self.dataset += [data_dir] * len(keep)
        print(f'For {self.mode}: In total {len(self.file_paths)} images from {len(args.data.data_dirs)} datasets loaded.')

    def __len__(self):
        return len(self.file_paths)

    def _preprocess(self, image, path=''):
        return preprocess_volume(
            image,
            patch_size=self.patch_size,
            norm_mode=self.args.data.norm_mode,
            downsample_factor=self.args.data.downsample_factor,
            max_axis_size=self.args.data.max_axis_size,
            max_volume_size=self.args.data.max_volume_size,
            path=path,
        )

    def _load_image(self, index, dataset=None):
        """Load and preprocess one volume. If `dataset` is given, a random
        volume from the same data directory is drawn instead (the unpaired
        second image of a registration pair)."""
        if dataset is not None:
            indices = [i for i, x in enumerate(self.dataset) if x == dataset]
            index = np.random.choice(indices)

        img_path = self.file_paths[index]
        image = load_volume(img_path)

        if self.augment_heavy:
            image = self.augmentor_heavy(torch.tensor(image.astype(np.float32)).clone()).numpy()

        image = self._preprocess(image, img_path)

        if self.return_mask:
            mask_path = self.mask_paths[index]
            if mask_path is None or pd.isna(mask_path):
                mask = torch.zeros_like(image)
                if self.first_run:
                    self.first_run = False
                    print('No mask_filepath given - setting mask to zeros - if this is not desired, add the column to the csv file')
            else:
                mask = load_volume(mask_path)
                # same geometric preprocessing as the image, but no intensity normalization
                mask = preprocess_volume(
                    mask, patch_size=self.patch_size, norm_mode=None,
                    downsample_factor=self.args.data.downsample_factor,
                    max_axis_size=self.args.data.max_axis_size,
                    max_volume_size=self.args.data.max_volume_size, path=mask_path)
            assert image.shape == mask.shape, f'Image and mask shape do not match: {image.shape} vs {mask.shape}'
            return image, mask, self.dataset[index]
        return image, self.dataset[index]

    def _positions_and_pe(self, image):
        D, H, W = image.shape
        pos = grid_positions((D, H, W), self.patch_size)
        pe_enc = self.pe_enc.encode_positions(pos, [D, H, W])
        pe_dec = self.pe_dec.encode_positions(pos, [D, H, W])
        return pos, pe_enc, pe_dec

    def __getitem__(self, index):
        if self.return_mask:
            item_0, mask_0, dataset = self._load_image(index)
            item_1, mask_1, _ = self._load_image(index, dataset)
            pos_0, pe_enc_0, pe_dec_0 = self._positions_and_pe(item_0)
            pos_1, pe_enc_1, pe_dec_1 = self._positions_and_pe(item_1)
            return (item_0, mask_0, pe_enc_0, pe_dec_0, pos_0), (item_1, mask_1, pe_enc_1, pe_dec_1, pos_1)
        else:
            item_0, dataset = self._load_image(index)
            item_1, _ = self._load_image(index, dataset)
            pos_0, pe_enc_0, pe_dec_0 = self._positions_and_pe(item_0)
            pos_1, pe_enc_1, pe_dec_1 = self._positions_and_pe(item_1)
            return (item_0, pe_enc_0, pe_dec_0, pos_0), (item_1, pe_enc_1, pe_dec_1, pos_1)


class PretrainDataset(RegistrationDataset):
    """Single-image masked position modeling pre-training (stage 1)."""

    def __init__(self, mode, args, embed_dim_enc, embed_dim_dec, return_mask=False, augment_heavy=False):
        super().__init__(mode, args, embed_dim_enc, embed_dim_dec, return_mask=False)
        pertubation_factor = args.single_image.pertubation_factor
        self.pos_pertubation = [p * pertubation_factor for p in self.patch_size]
        if augment_heavy:
            self.augmentor_heavy = Heavy3DAugmentor()
        else:
            self.augmentor_heavy = Heavy3DAugmentor(augment=False)

    def __getitem__(self, index):
        image, dataset = self._load_image(index)

        foreground_mask = torch.ones_like(image)

        if self.args.augmentation.heavy:
            image = self.augmentor_heavy(image.clone())

        D, H, W = image.shape
        n_patches = D*H*W//(self.patch_size[0]*self.patch_size[1]*self.patch_size[2])

        target = image.clone()
        target = self.augmentor_heavy(target)

        # target patches stay on the regular grid
        pos_target = grid_positions((D, H, W), self.patch_size)

        # source patches are shuffled to random locations
        source = image.clone()
        pos_source = []
        pos_source_vol = [(i, j, k) for i in range(0, D-self.patch_size[0]+1, self.patch_size[0]) for j in range(0, H-self.patch_size[1]+1, self.patch_size[1]) for k in range(0, W-self.patch_size[2]+1, self.patch_size[2])]
        for p in range(n_patches):
            d, h, w = pos_source_vol[p]
            x_start = torch.randint(0, D-self.patch_size[0], (1,))
            y_start = torch.randint(0, H-self.patch_size[1], (1,))
            z_start = torch.randint(0, W-self.patch_size[2], (1,))
            x_end = x_start + self.patch_size[0]
            y_end = y_start + self.patch_size[1]
            z_end = z_start + self.patch_size[2]
            source[d:d+self.patch_size[0], h:h+self.patch_size[1], w:w+self.patch_size[2]] = image[x_start:x_end, y_start:y_end, z_start:z_end]
            pos_source.append([int((x_end-x_start)//2+x_start), int((y_end-y_start)//2+y_start), int((z_end-z_start)//2+z_start)])
        pos_source = torch.tensor(pos_source).float()

        pos_source_augm = pos_source.clone()
        pertubation = torch.stack([torch.zeros(n_patches).uniform_(-pert, pert) for pert in self.pos_pertubation], axis=-1)
        pos_source_augm = pos_source_augm + pertubation

        pe_enc = self.pe_enc.encode_positions(pos_target, [D, H, W])
        pe_dec = self.pe_dec.encode_positions(pos_target, [D, H, W])
        pe_dec_augm = self.pe_dec.encode_positions(pos_source_augm, [D, H, W])

        return ((target, pe_enc, pe_dec, pos_target), (source, pos_source_augm, pe_dec_augm, pos_source)), foreground_mask, dataset
