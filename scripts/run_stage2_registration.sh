#!/bin/bash
# ============================================================
# Stage 2: unpaired registration training on multiple datasets,
# starting from the stage-1 checkpoint.
# Capped at max_total_train_steps=200000 (paper configuration).
# ============================================================
DATA_DIRS='["/path/to/dataset_a","/path/to/dataset_b"]'
OUTPUT_DIR="output/stage2_registration"
RESUME="output/stage1_pretrain/checkpoint-XXXX.pth"
SMOOTH_REG=2
SEED=123

python main.py \
    ++training_scheme="unpaired" \
    ++output_dir="${OUTPUT_DIR}" \
    ++data.data_dirs="${DATA_DIRS}" \
    ++model.patch_size="[16,16,16]" \
    ++loss.loss="mi" \
    ++loss.smoothReg_scale_factor=${SMOOTH_REG} \
    ++training.early_stopping=100 \
    ++training.max_total_train_steps=200000 \
    ++training.seed=${SEED} \
    ++augmentation.heavy=false \
    ++resume="${RESUME}"
