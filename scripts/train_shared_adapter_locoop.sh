#!/bin/bash
# ==============================================================================
# Simple training script for shared_adapter + LoCoOp OOD regularization
# ==============================================================================

# Load common parameters
source "$(dirname "${BASH_SOURCE[0]}")/common_params.sh"

# Method-specific configuration
METHOD="GL_MCM_FA"
# Selector disabled - use all patches directly
# SELECTOR_TYPE="identity"
FUSER_TYPE="shared_adapter"

# LoCoOp parameters
USE_LOCOOP_OOD=true
LAMBDA_LOCOOP_OOD=0.1
LOCOOP_TOPK=200
# Note: lambda_locoop_cls and lambda_locoop_patch are no longer used
# Only patch tokens are regularized for OOD detection

# Warmup parameters
WARMUP_EPOCHS=1
WARMUP_TYPE="constant"
WARMUP_CONS_LR=1e-5

# Log directory
LOG_DIR="$PROJECT_ROOT/logs/${METHOD}_${SELECTOR_TYPE}_${FUSER_TYPE}_locoop"
mkdir -p "$LOG_DIR"

# Start time
START_TIME=$(date +%s)
echo "Starting training for ${METHOD} at $(date)"
echo "Log directory: $LOG_DIR"
echo "Method: $METHOD"
echo "Selector: $SELECTOR_TYPE"
echo "Fuser: $FUSER_TYPE"
echo "====================================="
echo "LoCoOp OOD regularization: enabled"
echo "  lambda_locoop_ood: $LAMBDA_LOCOOP_OOD"
echo "  locoop_topk: $LOCOOP_TOPK"
echo "  lambda_locoop_cls: $LAMBDA_LOCOOP_CLS"
echo "  lambda_locoop_patch: $LAMBDA_LOCOOP_PATCH"
echo "====================================="

# Run training command
if python3 "$PROJECT_ROOT/src/train_eval.py" \
    --method "$METHOD" \
    --epochs "$DEFAULT_EPOCHS" \
    --lr "$DEFAULT_LR" \
    --batch_size "$DEFAULT_BATCH_SIZE" \
    --seed "$DEFAULT_SEED" \
    --backbone "$DEFAULT_BACKBONE" \
    --root_path "$DEFAULT_ROOT_PATH" \
    --shots "$DEFAULT_SHOTS" \
    --fuser_type "$FUSER_TYPE" \
    --adapter_ratio "$ADAPTER_RATIO" \
    --use_locoop_ood \
    --lambda_locoop_ood "$LAMBDA_LOCOOP_OOD" \
    --locoop_topk "$LOCOOP_TOPK" \
    --score_type "$SCORE_TYPE" \
    --temperature "$TEMPERATURE" \
    --lambda_local "$LAMBDA_LOCAL" \
    --residual_coef "$RESIDUAL_COEF" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --warmup_type "$WARMUP_TYPE" \
    --warmup_cons_lr "$WARMUP_CONS_LR" 2>&1 | tee -a "$LOG_DIR/train.log"; then
    echo "  ✓ Training completed successfully"
else
    echo "  ✗ Training failed"
    exit 1
fi

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "====================================="
echo "Training completed at $(date)"
echo "Total duration: $((DURATION/3600))h $(((DURATION%3600)/60))m $((DURATION%60))s"
echo "Logs saved to: $LOG_DIR"
