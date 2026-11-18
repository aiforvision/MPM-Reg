#!/bin/bash
# ============================================================
# Stage 3: dataset-specific fine-tuning, starting from a
# stage-2 checkpoint. Defaults: MPM-Reg V2 (dense head, MS-LNCC
# loss), e.g. from the lambda 1.66 checkpoint; tune SMOOTH_REG
# (lambda) for your trade-off between accuracy and regularity.
# Paper (B-spline) model: HEAD=spline, LOSS=mi, and sweep
# smoothReg_scale_factor (1, 2, 5 in the paper) per dataset.
# ============================================================
DATA_DIRS='["/path/to/target_dataset"]'
OUTPUT_DIR="output/stage3_finetune"
RESUME="output/stage2_registration/checkpoint-XXXX.pth"
HEAD="dense"
LOSS="mslncc"
SMOOTH_REG=1.66
SEED=123

python main.py \
    ++training_scheme="unpaired" \
    ++output_dir="${OUTPUT_DIR}" \
    ++data.data_dirs="${DATA_DIRS}" \
    ++model.patch_size="[16,16,16]" \
    ++model.head="${HEAD}" \
    ++augmentation.brightness=false \
    ++augmentation.heavy=false \
    ++loss.loss="${LOSS}" \
    ++loss.smoothReg_scale_factor=${SMOOTH_REG} \
    ++training.early_stopping=50 \
    ++training.seed=${SEED} \
    ++resume="${RESUME}"
