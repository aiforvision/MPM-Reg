# MPM-Reg: Masked Position Modeling Aligns Vision Transformers With Image Registration

Official PyTorch implementation of the paper "MPM-Reg: Masked Position Modeling Aligns Vision Transformers With Image Registration" (under review).

Training proceeds in three stages:

1. **Stage 1 — single-image pre-training (`training_scheme=single`)**:
   the positional
   embeddings of patches are masked and the decoder reconstructs each patch's position
   (with perturbed spatial cues as decoder input).
2. **Stage 2 — unpaired registration training (`training_scheme=unpaired`)**:
   the same encoder–decoder now predicts per-patch positions of a moving
   volume in the space of a fixed volume; the position grid is converted to a
   dense displacement field via cubic B-spline interpolation, trained with a mutual
   information image loss and a smoothness regularizer across several datasets.
3. **Stage 3 — dataset-specific fine-tuning**: stage 2 continued on a single
   target dataset.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.12, CUDA GPU required for training.

## Data

Each dataset is a directory containing `train.csv` and `val.csv` with a
`filepath` column (absolute paths to 3D volumes: `.nii/.nii.gz`, `.npy`,
`.tif/.tiff`, `.h5/.hdf5`) and optionally a `mask_filepath` column
(segmentation masks, used for validation visualization only).

[data/preprocessing.py](data/preprocessing.py) documents and implements the
per-volume preprocessing used in the paper (NaN handling, optional axis
cap/subsampling, padding, cropping to patch multiples, min-max normalization).
Use the same configuration when plugging in your own datasets.

## Training

```bash
bash scripts/run_stage1_pretrain.sh        # stage 1: MPM pre-training
bash scripts/run_stage2_registration.sh    # stage 2: registration training (resumes stage-1 checkpoint)
bash scripts/run_stage3_finetune.sh        # stage 3: fine-tuning (resumes stage-2 checkpoint)
```

Edit `DATA_DIRS`, `OUTPUT_DIR` and `RESUME` in the scripts. All options live in
[conf/config.yaml](conf/config.yaml) and can be overridden on the command line
(Hydra). Checkpoints (`checkpoint-<epoch>.pth`, top-3 by validation loss) and
per-epoch preview figures are written to the output directory.

## Inference

Register a pair of volumes with a stage-2/3 checkpoint:

```bash
python register.py --checkpoint checkpoint.pth \
    --moving moving.nii.gz --fixed fixed.nii.gz --output_dir out/
```

Writes the preprocessed inputs, the warped moving volume and the dense
displacement field (in voxels) as NIfTI files.

## Repository layout

```
main.py                      # training entry point (all three stages)
register.py                  # pairwise inference
conf/config.yaml             # full configuration
model/mpm_reg.py             # MPM-Reg models (stage 1 + stages 2/3)
model/blocks.py              # transformer blocks / patch embedding
model/bspline.py             # cubic B-spline interpolation to a dense displacement field
data/preprocessing.py        # preprocessing pipeline used in the paper
data/dataset.py              # generic CSV-driven datasets
augmentation/augmentation.py # brightness / heavy augmentation
training/                    # stage engines + optimizer/scheduler utilities
```

## Acknowledgements

The model implementation is based on the BEiT, timm, DINO, DeiT and
MAE-pytorch code bases (see headers in [model/](model/)). The B-spline interpolation
uses [deepali](https://github.com/BioMedIA/deepali).
