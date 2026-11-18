# MPM-Reg: Masked Position Modeling as a Pre-Training Strategy for Foundational Deformable Image Registration Models

Official PyTorch implementation of MPM-Reg, from the paper [Masked Position Modeling as a Pre-Training Strategy for Foundational Deformable Image Registration Models](https://ieeexplore.ieee.org/document/11754026).

📄 [Paper (IEEE Xplore)](https://ieeexplore.ieee.org/document/11754026) · 🎬 [Video abstract (LinkedIn)](https://www.linkedin.com/posts/new-paper-alert-%E2%84%B9-the-core-objective-ugcPost-7524487938647207936-afTT/)

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

## MPM-Reg checkpoints

The checkpoint files can be downloaded from the [releases](https://github.com/aiforvision/MPM-Reg/releases) of this
repository.

### MPM-Reg V2 (recommended)

MPM-Reg V2 replaces the fixed B-spline interpolation with a learnable, B-spline-initialised **dense head** that
predicts each voxel's displacement from the neighbouring decoder tokens, and is trained with a multi-scale local
normalised cross-correlation (MS-LNCC) loss. It is the **recommended model** to use and to
fine-tune on your data.

All checkpoints load with [conf/config.yaml](conf/config.yaml) as is; only `model.head` has to match the checkpoint
(`register.py` detects it automatically). To continue training a checkpoint as it was trained, set `loss.loss` and
`loss.smoothReg_scale_factor` (λ) to the values in the tables.

| Model | File | `model.head` / `loss.loss` | λ (smoothness) | Use |
|---|---|---|---:|---|
| **MPM-Reg V2** | `MPM-Reg-V2_stage2_dense_mslncc_lambda1.66.pth` | `dense` / `mslncc` | 1.66 | **Recommended**; default starting point for Stage 3 fine-tuning |

To adapt MPM-Reg V2 to your data, fine-tune the λ = 1.66 model (stage 3, see [Training](#training)) and tune λ for
the trade-off between registration accuracy and deformation regularity that suits your use case.

### Further checkpoints

We provide multiple stage-2 checkpoints trained with different smoothness regularisation
weights λ (a smaller λ means less regularisation and more flexible deformation fields); these can serve as different
starting points for your fine-tuning.

| Model | File | `model.head` / `loss.loss` | λ (smoothness) | Use |
|---|---|---|---:|---|
| MPM-Reg V2 | `MPM-Reg-V2_stage2_dense_mslncc_lambda0.83.pth` | `dense` / `mslncc` | 0.83 | for tasks that need more flexibility (e.g. Abdomen CT/CT) |
| MPM-Reg V2 | `MPM-Reg-V2_stage2_dense_mslncc_lambda0.5.pth` | `dense` / `mslncc` | 0.5 | for high flexibility tasks (e.g. inter-subject brain, OAI-ZIB, KiTS) |
| MPM-Reg | `MPM-Reg_stage2_spline_mslncc_lambda1.66.pth` | `spline` / `mslncc` | 1.66 | best B-spline version |
| MPM-Reg | `MPM-Reg_stage2_spline_mi_lambda2.pth` | `spline` / `mi` | 2 | closest to the paper (stage 2), trained without UKBB data* |
| MPM-Reg | `MPM-Reg_stage1.pth` | – / `mse` | – | closest to the paper (stage 1), trained without UKBB data*; load with `training_scheme=single`, or as `resume` for stage 2 |

\* Model weights trained on UK Biobank (UKBB) data may not be shared, so these models were retrained without the UKBB
data (full-body, cine MR and brain MRI). They are therefore the closest available versions of the paper's models, but
not the exact models from the paper.

## Installation

```bash
pip install -r requirements.txt          # MPM-Reg
pip install -r requirements-tto.txt      # optional: for test-time optimisation, same environment
```


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

Edit `DATA_DIRS`, `OUTPUT_DIR` and `RESUME` in the scripts. Stage 3 fine-tunes MPM-Reg V2 by default
(`HEAD=dense`, `LOSS=mslncc`, λ = `SMOOTH_REG`); for the paper's model set `HEAD=spline`. The config file is 
[conf/config.yaml](conf/config.yaml) and it can be overridden on the command line
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

### Optional: test-time optimisation (TTO) on top of MPM-Reg

Test-time optimisation (TTO) can be used additionally to refine the MPM-Reg result per pair, at the cost of longer inference time. It helps most large deformations and datasets with very little training data (e.g. abdomen MR/CT).
`--tto anatomix_fireants` runs the **anatomix + FireANTs registration of Dey et al.**, starting from the MPM-Reg field, and additionally
writes `warped_moving_tto.nii.gz` and `flow_px_tto.nii.gz`:

```bash
python register.py --checkpoint checkpoint.pth --moving moving.nii.gz --fixed fixed.nii.gz --output_dir out/ \
    --tto anatomix_fireants [--tto_fixed_clip -450 450] [--tto_fixed_mask fixed_mask.nii.gz --tto_moving_mask moving_mask.nii.gz]
```

TTO uses the anatomix authors' default settings; `python register.py -h` lists the available options.

**This is not our contribution:** the TTO method and its code are by the anatomix and FireANTs authors;
[tto/anatomix_fireants.py](tto/anatomix_fireants.py) only starts their registration from the MPM-Reg field. If you use
this option, please cite their work:

- **anatomix**: N. Dey, B. Billot, H. E. Wong, C. J. Wang, M. Ren, P. E. Grant, A. V. Dalca, P. Golland, *Learning
  General-Purpose Biomedical Volume Representations using Randomized Synthesis*, ICLR 2025,
  [arXiv:2411.02372](https://arxiv.org/abs/2411.02372). Code: https://github.com/neel-dey/anatomix (MIT licence).
- **FireANTs**: R. Jena, P. Chaudhari, J. C. Gee, *FireANTs: Adaptive Riemannian Optimization for Multi-Scale
  Diffeomorphic Registration*, Nature Communications, 2024. Code: https://github.com/rohitrango/fireants
  (Apache-2.0); anatomix uses the fork https://github.com/neel-dey/FireANTs.

## Repository layout

```
main.py                      # training entry point (all three stages)
register.py                  # pairwise inference (optional TTO: --tto)
conf/config.yaml             # full configuration
model/mpm_reg.py             # MPM-Reg models (stage 1 + stages 2/3, B-spline and V2)
model/blocks.py              # transformer blocks / patch embedding
model/bspline.py             # cubic B-spline interpolation to a dense displacement field
model/dense_head.py          # dense head of MPM-Reg V2
loss/mslncc.py               # MS-LNCC image loss of MPM-Reg V2
tto/anatomix_fireants.py     # optional TTO with anatomix + FireANTs (third-party, see Inference)
data/preprocessing.py        # preprocessing pipeline used in the paper
data/dataset.py              # generic CSV-driven datasets
augmentation/augmentation.py # brightness / heavy augmentation
training/                    # stage engines + optimizer/scheduler utilities
```

## Acknowledgements

The model implementation is based on the BEiT, timm, DINO, DeiT and
MAE-pytorch code bases (see headers in [model/](model/)). The B-spline interpolation
uses [deepali](https://github.com/BioMedIA/deepali). The optional test-time optimisation uses
[anatomix](https://github.com/neel-dey/anatomix) and [FireANTs](https://github.com/rohitrango/fireants) as unmodified external dependencies.
