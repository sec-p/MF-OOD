#!/bin/bash
# ==============================================================================
# Focused hyperparameter search for shared_adapter + LoCoOp
# Focus on LoCoOp parameters: lambda_cls, lambda_patch, topk
# ==============================================================================

# Load common parameters
source "$(dirname "${BASH_SOURCE[0]}")/common_params.sh"

# Method-specific configuration
METHOD="GL_MCM_FA"
SELECTOR_TYPE="identity"
FUSER_TYPE="shared_adapter"

# Fixed parameters
FIXED_LR=0.005
FIXED_BS=128
FIXED_EPOCHS=30
FIXED_ADAPTER_RATIO=0.5

# Focused LoCoOp parameter grids
LAMBDA_LOCOOP_OOD=(0.1)
LOCOOP_TOPK=(100 200 300)
LAMBDA_LOCOOP_CLS=(0.3 0.5 0.7)
LAMBDA_LOCOOP_PATCH=(0.3 0.5 0.7)

# Log directory
LOG_DIR="$PROJECT_ROOT/logs/locoop_search_${METHOD}_${FUSER_TYPE}"
mkdir -p "$LOG_DIR"

# Start time
START_TIME=$(date +%s)
echo "Starting focused LoCoOp hyperparameter search for ${METHOD} at $(date)"
echo "Log directory: $LOG_DIR"
echo "Method: $METHOD"
echo "Selector: $SELECTOR_TYPE"
echo "Fuser: $FUSER_TYPE"
echo "====================================="
echo "Fixed parameters:"
echo "  lr: $FIXED_LR"
echo "  batch_size: $FIXED_BS"
echo "  epochs: $FIXED_EPOCHS"
echo "  adapter_ratio: $FIXED_ADAPTER_RATIO"
echo "====================================="
echo "Search parameters:"
echo "  locoop_topk: ${LOCOOP_TOPK[@]}"
echo "  lambda_locoop_cls: ${LAMBDA_LOCOOP_CLS[@]}"
echo "  lambda_locoop_patch: ${LAMBDA_LOCOOP_PATCH[@]}"
echo "====================================="

# Hyperparameter combinations counter
COUNTER=0
total_combinations=$((${#LOCOOP_TOPK[@]} * ${#LAMBDA_LOCOOP_CLS[@]} * ${#LAMBDA_LOCOOP_PATCH[@]}))

# Grid search loop
for locoop_topk in "${LOCOOP_TOPK[@]}"; do
    for lambda_locoop_cls in "${LAMBDA_LOCOOP_CLS[@]}"; do
        for lambda_locoop_patch in "${LAMBDA_LOCOOP_PATCH[@]}"; do
            for seed in "${SEEDS[@]}"; do
                ((COUNTER++))
                echo -e "\n[$COUNTER/$total_combinations] Running: topk=$locoop_topk, lambda_cls=$lambda_locoop_cls, lambda_patch=$lambda_locoop_patch, seed=$seed"
                
                # Run training command
                if python3 "$PROJECT_ROOT/src/train_eval.py" \
                    --method "$METHOD" \
                    --epochs "$FIXED_EPOCHS" \
                    --lr "$FIXED_LR" \
                    --batch_size "$FIXED_BS" \
                    --seed "$seed" \
                    --backbone "$DEFAULT_BACKBONE" \
                    --root_path "$DEFAULT_ROOT_PATH" \
                    --shots "$DEFAULT_SHOTS" \
                    --selector_type "$SELECTOR_TYPE" \
                    --fuser_type "$FUSER_TYPE" \
                    --adapter_ratio "$FIXED_ADAPTER_RATIO" \
                    --residual_coef "$RESIDUAL_COEF" \
                    --use_locoop_ood \
                    --lambda_locoop_ood "$LAMBDA_LOCOOP_OOD" \
                    --locoop_topk "$locoop_topk" \
                    --lambda_locoop_cls "$lambda_locoop_cls" \
                    --lambda_locoop_patch "$lambda_locoop_patch" \
                    --score_type "$SCORE_TYPE" \
                    --temperature "$TEMPERATURE" \
                    --lambda_local "$LAMBDA_LOCAL" 2>&1 | tee -a "$LOG_DIR/search_topk_${locoop_topk}_cls_${lambda_locoop_cls}_patch_${lambda_locoop_patch}_seed_${seed}.log"; then
                    echo "  ✓ Success"
                else
                    echo "  ✗ Failed"
                fi
            done
        done
    done
done

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "====================================="
echo "Hyperparameter search completed at $(date)"
echo "Total duration: $((DURATION/3600))h $(((DURATION%3600)/60))m $((DURATION%60))s"
echo "Results saved to: $LOG_DIR"
echo "====================================="

# Summary
echo ""
echo "Summary of completed runs:"
echo "Total combinations: $total_combinations"
echo "Completed: $COUNTER"
echo ""
echo "To analyze results, check:"
echo "  $LOG_DIR/search_*.log"
