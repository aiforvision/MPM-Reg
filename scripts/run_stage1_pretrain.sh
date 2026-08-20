#!/bin/bash
# ============================================================
# Stage 1: single-image masked position modeling pre-training.
# Values match the paper configuration (patch size 16, position
# cues enabled, perturbation factor 1).
# ============================================================
DATA_DIRS='["/path/to/dataset_a","/path/to/dataset_b"]'
OUTPUT_DIR="output/stage1_pretrain"
SEED=123

python main.py \
    ++training_scheme="single" \
    ++output_dir="${OUTPUT_DIR}" \
    ++data.data_dirs="${DATA_DIRS}" \
    ++augmentation.brightness=false \
    ++augmentation.heavy=false \
    ++loss.loss="mse" \
    ++training.early_stopping=200 \
    ++training.epochs=10000 \
    ++model.patch_size="[16,16,16]" \
    ++single_image.give_spatial_cues=true \
    ++single_image.pertubation_factor=1 \
    ++training.seed=${SEED}
