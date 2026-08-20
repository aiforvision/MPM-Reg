#!/bin/bash
# ============================================================
# Stage 3: dataset-specific fine-tuning, starting from the
# stage-2 checkpoint. Sweep smoothReg_scale_factor (1, 2, 5 in
# the paper) per dataset.
# ============================================================
DATA_DIRS='["/path/to/target_dataset"]'
OUTPUT_DIR="output/stage3_finetune"
RESUME="output/stage2_registration/checkpoint-XXXX.pth"
SMOOTH_REG=2
SEED=123

python main.py \
    ++training_scheme="unpaired" \
    ++output_dir="${OUTPUT_DIR}" \
    ++data.data_dirs="${DATA_DIRS}" \
    ++model.patch_size="[16,16,16]" \
    ++augmentation.brightness=false \
    ++augmentation.heavy=false \
    ++loss.loss="mi" \
    ++loss.smoothReg_scale_factor=${SMOOTH_REG} \
    ++training.early_stopping=50 \
    ++training.seed=${SEED} \
    ++resume="${RESUME}"
