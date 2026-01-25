#!/bin/bash
# ==============================================================================
# Hyperparameter search for selector_slot_self_attn method in GL_MCM_FA
# ==============================================================================

# Load common parameters
source "$(dirname "${BASH_SOURCE[0]}")/common_params.sh"

# Method-specific configuration
METHOD="GL_MCM_FA"
SELECTOR_TYPE="mlp"
FUSER_TYPE="self_attn"

# Log directory
LOG_DIR="$PROJECT_ROOT/logs/search_${METHOD}_${SELECTOR_TYPE}_${FUSER_TYPE}"
mkdir -p "$LOG_DIR"

# Start time
START_TIME=$(date +%s)
echo "Starting hyperparameter search for ${METHOD} at $(date)"
echo "Log directory: $LOG_DIR"
echo "Method: $METHOD"
echo "Selector: $SELECTOR_TYPE"
echo "Fuser: $FUSER_TYPE"
echo "====================================="

# Hyperparameter combinations counter
COUNTER=0
total_combinations=$(( ${#LEARNING_RATES[@]} * ${#BATCH_SIZES[@]} * ${#SEEDS[@]} * ${#NUM_SELECT_VALUES[@]} * ${#LAMBDA_LLM_NEGATIVES[@]} * ${#LAMBDA_MIXUP[@]} * ${#MARGIN_VALUES[@]} * ${#SELECTOR_TEMPERATURE[@]} * ${#PATCHES_PER_SLOT_ATTN[@]} * ${#MLP_HIDDEN_RATIO[@]} * ${#SLOT_FFN_RATIO[@]} * ${#FUSER_FFN_RATIO[@]} * ${#LAMBDA_INTRA_CLASS[@]} * ${#INTRA_CLASS_TEMP[@]} ))

# Grid search loop
for lr in "${LEARNING_RATES[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for num_select in "${NUM_SELECT_VALUES[@]}"; do
                for llm_neg_lambda in "${LAMBDA_LLM_NEGATIVES[@]}"; do
                    for mixup_lambda in "${LAMBDA_MIXUP[@]}"; do
                        for margin_val in "${MARGIN_VALUES[@]}"; do
                            for sel_temp in "${SELECTOR_TEMPERATURE[@]}"; do
                                for patches_per_slot in "${PATCHES_PER_SLOT_ATTN[@]}"; do
                                    for mlp_ratio in "${MLP_HIDDEN_RATIO[@]}"; do
                                        for slot_ratio in "${SLOT_FFN_RATIO[@]}"; do
                                            for fuser_ratio in "${FUSER_FFN_RATIO[@]}"; do
                                                for intra_class_lambda in "${LAMBDA_INTRA_CLASS[@]}"; do
                                                    for intra_class_temp in "${INTRA_CLASS_TEMP[@]}"; do
                                                        ((COUNTER++))
                                                        echo -e "\n[$COUNTER/$total_combinations] Running: $METHOD - lr=$lr, bs=$bs, seed=$seed, num_select=$num_select, lambda_llm=$llm_neg_lambda, lambda_mixup=$mixup_lambda, margin=$margin_val, sel_temp=$sel_temp, patches_per_slot=$patches_per_slot, mlp_ratio=$mlp_ratio, slot_ratio=$slot_ratio, fuser_ratio=$fuser_ratio, lambda_intra_class=$intra_class_lambda, intra_class_temp=$intra_class_temp"
                                                        
                                                        # Run training command
                                                        if python3 "$PROJECT_ROOT/src/train_eval.py" \
                                                            --method "$METHOD" \
                                                            --epochs "$DEFAULT_EPOCHS" \
                                                            --lr "$lr" \
                                                            --batch_size "$bs" \
                                                            --seed "$seed" \
                                                            --backbone "$DEFAULT_BACKBONE" \
                                                            --root_path "$DEFAULT_ROOT_PATH" \
                                                            --shots "$DEFAULT_SHOTS" \
                                                            --class_negatives_path "$DEFAULT_CLASS_NEGATIVES_PATH" \
                                                            --lambda_llm_negatives "$llm_neg_lambda" \
                                                            --lambda_mixup "$mixup_lambda" \
                                                            --margin "$margin_val" \
                                                            --selector_type "$SELECTOR_TYPE" \
                                                            --fuser_type "$FUSER_TYPE" \
                                                            --num_select "$num_select" \
                                                            --selector_temperature "$sel_temp" \
                                                            --patches_per_slot_attn "$patches_per_slot" \
                                                            --mlp_hidden_ratio "$mlp_ratio" \
                                                            --slot_ffn_ratio "$slot_ratio" \
                                                            --fuser_ffn_ratio "$fuser_ratio" \
                                                            --lambda_intra_class "$intra_class_lambda" \
                                                            --intra_class_temp "$intra_class_temp" \
                                                            --score_type "$SCORE_TYPE" \
                                                            --temperature "$TEMPERATURE" \
                                                            --lambda_local "$LAMBDA_LOCAL" 2>&1 | tee -a "$LOG_DIR/grid_search.log"; then
                                                            echo "  ✓ Success"
                                                        else
                                                            echo "  ✗ Failed"
                                                        fi
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
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
