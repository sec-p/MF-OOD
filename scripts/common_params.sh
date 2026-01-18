#!/bin/bash
# ==============================================================================
# Common parameters for all hyperparameter search scripts
# ==============================================================================

# Project root (relative to scripts directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default training parameters
DEFAULT_EPOCHS=50
DEFAULT_BATCH_SIZE=1024
DEFAULT_LR=0.001
DEFAULT_SEED=42
DEFAULT_BACKBONE="ViT-B/16"
DEFAULT_ROOT_PATH="/data/ICML2026/clip/FA/my_dataset"
DEFAULT_SHOTS=16
DEFAULT_CLASS_NEGATIVES_PATH="/root/class_negatives.json"

# Hyperparameter search grids
LEARNING_RATES=(0.002)
BATCH_SIZES=(128)
SEEDS=(42)

# Loss function coefficients grids
LAMBDA_LLM_NEGATIVES=(5.0)
LAMBDA_MIXUP=(1.0)
MARGIN_VALUES=(0.0)

# Selector parameters grid
NUM_SELECT_VALUES=(64)

# Advanced parameters grid
SELECTOR_TEMPERATURE=(1.0)
PATCHES_PER_SLOT_ATTN=(8)

# Dimension parameters
MLP_HIDDEN_RATIO=(0.25)
SLOT_FFN_RATIO=(4.0)
FUSER_FFN_RATIO=(8.0)

# OOD score parameters
SCORE_TYPE="GL-MCM-L"
TEMPERATURE=1.0
LAMBDA_LOCAL=0.5
