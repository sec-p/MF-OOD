#!/bin/bash
# ==============================================================================
# Common parameters for all hyperparameter search scripts
# ==============================================================================

# Project root (relative to scripts directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default training parameters
DEFAULT_EPOCHS=30
DEFAULT_BATCH_SIZE=1024
DEFAULT_LR=0.001
DEFAULT_SEED=1
DEFAULT_BACKBONE="ViT-B/16"
DEFAULT_ROOT_PATH="/amax/yeliu/data"
DEFAULT_SHOTS=1
DEFAULT_CLASS_NEGATIVES_PATH="/root/huhuhu2/MF-OOD/negatives_non_photographic.json"

# Hyperparameter search grids
LEARNING_RATES=(0.00005)
BATCH_SIZES=(32)
SEEDS=(1)

# Loss function coefficients grids
LAMBDA_LLM_NEGATIVES=(5)
LAMBDA_MIXUP=(0)
MARGIN_VALUES=(0.025)
LAMBDA_INTRA_CLASS=(0)
INTRA_CLASS_TEMP=(0)

# LoCoOp OOD regularization parameters
LAMBDA_LOCOOP_OOD=(2)
LOCOOP_TOPK=(200)
LAMBDA_LOCOOP_CLS=(0.5)
LAMBDA_LOCOOP_PATCH=(0.5)

# Selector parameters grid
NUM_SELECT_VALUES=(64)

# Advanced parameters grid
SELECTOR_TEMPERATURE=(1.0)
PATCHES_PER_SLOT_ATTN=(4)

# Dimension parameters
MLP_HIDDEN_RATIO=(1)
SLOT_FFN_RATIO=(4.0)
FUSER_FFN_RATIO=(4)
ADAPTER_RATIO=(0.5)
RESIDUAL_COEF=(0.2)

# FUSER_FFN_RATIO=(8)


# OOD score parameters
SCORE_TYPE="GL-MCM"
TEMPERATURE=1.0
LAMBDA_LOCAL=0.5

# Warmup parameters
WARMUP_EPOCHS=1
WARMUP_TYPE="constant"
WARMUP_CONS_LR=1e-5
