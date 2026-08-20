"""Preprocessing pipeline used to train MPM-Reg.

This module replicates, step by step, the preprocessing that was applied to
every training/validation volume in the paper. If you plug in your own
dataset, run your volumes through `preprocess_volume` (or apply the same
steps yourself) so that your data matches the configuration MPM-Reg was
trained with:

    1. load the volume (.nii/.nii.gz, .npy, .tif/.tiff, .h5/.hdf5)
    2. replace NaNs (all-NaN volume -> unit Gaussian noise; partial NaNs -> volume mean)
    3. optionally center-crop each axis to `max_axis_size` voxels
       (used for very large whole-body volumes; 224 in the paper)
    4. optionally subsample by strided slicing with an integer `downsample_factor`
       (paper: 2 for volumes with sub-millimetre resolution, 1 otherwise), and
       subsample further if any axis exceeds `max_volume_size` (paper: 256)
    5. zero-pad every axis that is smaller than 2 x patch_size up to 2 x patch_size
    6. crop every axis down to the nearest multiple of patch_size
    7. intensity normalization (paper: min-max to [0, 1])

Patch positions are the centers of the regular patch grid; the positional
embeddings are sinusoidal encodings of these (normalized) positions.
"""

import sys
import numpy as np
import torch
import torch.nn.functional


def load_volume(path):
    """Load a 3D volume from .npy, .tif(f), .nii(.gz), or .h5/.hdf5."""
    if path.endswith('.npy'):
        image = np.load(path)
    elif path.endswith('.tiff') or path.endswith('.tif'):
        import tifffile as tiff
        image = tiff.imread(path)
    elif path.endswith('.nii.gz') or path.endswith('.nii'):
        import nibabel as nib
        image = nib.load(path).get_fdata()
    elif path.endswith('.hdf5') or path.endswith('.h5'):
        import h5py
        image = h5py.File(path, 'r')['data'][:]
    else:
        print(f'File format not recognized: {path}')
        sys.exit(1)
    return image


def sanitize_nans(image, path=''):
    """Replace NaNs: all-NaN volumes with Gaussian noise, partial NaNs with the mean."""
    if np.all(np.isnan(image)):
        print(f"Warning: image is all NaNs — replacing with random noise. Image path: {path}")
        image = np.random.normal(loc=0.0, scale=1.0, size=image.shape)
    elif np.any(np.isnan(image)):
        print(f"Warning: image has NaNs — replacing with mean. Image path: {path}")
        mean_val = np.nanmean(image)
        image[np.isnan(image)] = mean_val
    return image


def center_crop_to_max(arr, max_size):
    """Center-crop each axis of arr down to at most max_size; leave smaller axes untouched."""
    slices = []
    for s in arr.shape:
        if s > max_size:
            start = (s - max_size) // 2
            slices.append(slice(start, start + max_size))
        else:
            slices.append(slice(None))
    return arr[tuple(slices)]


def subsample(image, downsample_factor, max_volume_size=None):
    """Strided subsampling by an integer factor; additionally cap each axis at max_volume_size."""
    if downsample_factor > 1:
        image = image[::downsample_factor, ::downsample_factor, ::downsample_factor]
    if max_volume_size is not None:
        for axis in range(3):
            if image.shape[axis] > max_volume_size:
                factor = image.shape[axis] // max_volume_size
                if factor > 1:
                    slicer = [slice(None)] * 3
                    slicer[axis] = slice(None, None, factor)
                    image = image[tuple(slicer)]
    return image


def pad_to_min_size(image, patch_size):
    """Zero-pad every axis that is smaller than 2 x patch_size up to 2 x patch_size."""
    if image.shape[0] < 2*patch_size[0]:
        image = torch.nn.functional.pad(image, (0, 0, 0, 0, 0, 2*patch_size[0]-image.shape[0]), mode='constant', value=0)
    if image.shape[1] < 2*patch_size[1]:
        image = torch.nn.functional.pad(image, (0, 0, 0, 2*patch_size[1]-image.shape[1], 0, 0), mode='constant', value=0)
    if image.shape[2] < 2*patch_size[2]:
        image = torch.nn.functional.pad(image, (0, 2*patch_size[2]-image.shape[2], 0, 0, 0, 0), mode='constant', value=0)
    return image


def crop_to_patch_multiple(image, patch_size):
    """Crop every axis down to the nearest multiple of patch_size."""
    image = image[:image.shape[0]//patch_size[0]*patch_size[0], :, :]
    image = image[:, :image.shape[1]//patch_size[1]*patch_size[1], :]
    image = image[:, :, :image.shape[2]//patch_size[2]*patch_size[2]]
    return image


def normalize(image, norm_mode='minmax'):
    """Intensity normalization. The paper uses 'minmax' (to [0, 1]). None skips normalization (label maps)."""
    if norm_mode is None:
        pass
    elif norm_mode == 'meanstd':
        image = (image - image.mean()) / (image.std() + 1e-6)
    elif norm_mode == 'minmax':
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
    else:
        print(f'Normalization mode not recognized: {norm_mode}')
        sys.exit(1)
    return image


def preprocess_volume(image, patch_size, norm_mode='minmax', downsample_factor=1,
                      max_axis_size=None, max_volume_size=None, path=''):
    """Full preprocessing pipeline (steps 2-7 above) for an already-loaded volume.

    Args:
        image: np.ndarray or torch.Tensor of shape [D, H, W]
        patch_size: [pd, ph, pw]
        norm_mode: 'minmax' (paper default) or 'meanstd'
        downsample_factor: integer strided-slicing factor (1 = no subsampling)
        max_axis_size: optional center-crop cap per axis (paper: 224 for whole-body data)
        max_volume_size: optional subsampling cap per axis (paper: 256)
        path: only used for warning messages
    Returns:
        torch.FloatTensor of shape [D', H', W'], every axis a multiple of patch_size
    """
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    image = sanitize_nans(image, path)
    image = image.astype(np.float32)
    image = torch.tensor(image)

    if max_axis_size is not None:
        image = center_crop_to_max(image, max_axis_size)

    image = subsample(image, downsample_factor, max_volume_size)

    image = pad_to_min_size(image, patch_size)
    image = crop_to_patch_multiple(image, patch_size)

    image = normalize(image, norm_mode)

    assert len(image.shape) == 3, f'Image shape not recognized: {image.shape}'
    return image


def grid_positions(shape, patch_size):
    """Patch-center positions of the regular patch grid, shape [n_patches, 3]."""
    D, H, W = shape
    pos = [(i, j, k)
           for i in range(patch_size[0]//2, D-patch_size[0]//2+1, patch_size[0])
           for j in range(patch_size[1]//2, H-patch_size[1]//2+1, patch_size[1])
           for k in range(patch_size[2]//2, W-patch_size[2]//2+1, patch_size[2])]
    return torch.tensor(pos).float()
