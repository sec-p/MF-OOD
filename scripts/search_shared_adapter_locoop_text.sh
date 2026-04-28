#!/bin/bash
# ==============================================================================
# Hyperparameter search for shared_adapter + LoCoOp OOD regularization
# ==============================================================================

# Load common parameters
source "$(dirname "${BASH_SOURCE[0]}")/common_params.sh"

# Method-specific configuration
METHOD="GL_MCM_FA"
SELECTOR_TYPE="identity"
FUSER_TYPE="shared_adapter"
TEXT_FEATURE_PATH="/amax/yeliu/checkpoints/locoop/ViT-B/16/16shots/seed1/text_features_locoop.pt"
# Log directory
LOG_DIR="$PROJECT_ROOT/logs/search_${METHOD}_${SELECTOR_TYPE}_${FUSER_TYPE}_locoop"
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
total_combinations=$(( ${#LEARNING_RATES[@]} * ${#BATCH_SIZES[@]} * ${#SEEDS[@]} * ${#LAMBDA_LOCOOP_OOD[@]} * ${#LOCOOP_TOPK[@]} * ${#LAMBDA_LOCOOP_CLS[@]} * ${#LAMBDA_LOCOOP_PATCH[@]} * ${#ADAPTER_RATIO[@]} * ${#LAMBDA_LLM_NEGATIVES[@]} * ${#LAMBDA_MIXUP[@]} * ${#MARGIN_VALUES[@]} * ${#LAMBDA_INTRA_CLASS[@]} * ${#INTRA_CLASS_TEMP[@]} * ${#NUM_SELECT_VALUES[@]} * ${#MLP_HIDDEN_RATIO[@]} * ${#SLOT_FFN_RATIO[@]} * ${#FUSER_FFN_RATIO[@]} * ${#SELECTOR_TEMPERATURE[@]} * ${#PATCHES_PER_SLOT_ATTN[@]} * ${#RESIDUAL_COEF[@]} ))

# Grid search loop
for lr in "${LEARNING_RATES[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for lambda_locoop_ood in "${LAMBDA_LOCOOP_OOD[@]}"; do
                for locoop_topk in "${LOCOOP_TOPK[@]}"; do
                    for lambda_locoop_cls in "${LAMBDA_LOCOOP_CLS[@]}"; do
                        for lambda_locoop_patch in "${LAMBDA_LOCOOP_PATCH[@]}"; do
                            for adapter_ratio in "${ADAPTER_RATIO[@]}"; do
                                for residual_coef in "${RESIDUAL_COEF[@]}"; do
                                    for lambda_llm_negatives in "${LAMBDA_LLM_NEGATIVES[@]}"; do
                                    for lambda_mixup in "${LAMBDA_MIXUP[@]}"; do
                                        for margin in "${MARGIN_VALUES[@]}"; do
                                            for lambda_intra_class in "${LAMBDA_INTRA_CLASS[@]}"; do
                                                for intra_class_temp in "${INTRA_CLASS_TEMP[@]}"; do
                                                    for num_select in "${NUM_SELECT_VALUES[@]}"; do
                                                        for mlp_hidden_ratio in "${MLP_HIDDEN_RATIO[@]}"; do
                                                            for slot_ffn_ratio in "${SLOT_FFN_RATIO[@]}"; do
                                                                for fuser_ffn_ratio in "${FUSER_FFN_RATIO[@]}"; do
                                                                    for selector_temperature in "${SELECTOR_TEMPERATURE[@]}"; do
                                                                        for patches_per_slot_attn in "${PATCHES_PER_SLOT_ATTN[@]}"; do
                                                                            ((COUNTER++))
                                                                            echo -e "\n[$COUNTER/$total_combinations] Running: $METHOD - lr=$lr, bs=$bs, seed=$seed, lambda_locoop_ood=$lambda_locoop_ood, locoop_topk=$locoop_topk, lambda_cls=$lambda_locoop_cls, lambda_patch=$lambda_locoop_patch, adapter_ratio=$adapter_ratio, residual_coef=$residual_coef, lambda_llm_negatives=$lambda_llm_negatives, lambda_mixup=$lambda_mixup, margin=$margin, lambda_intra_class=$lambda_intra_class, intra_class_temp=$intra_class_temp, num_select=$num_select, mlp_hidden_ratio=$mlp_hidden_ratio, slot_ffn_ratio=$slot_ffn_ratio, fuser_ffn_ratio=$fuser_ffn_ratio, selector_temperature=$selector_temperature, patches_per_slot_attn=$patches_per_slot_attn"
                                                                            
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
                                                                                                    --selector_type "$SELECTOR_TYPE" \
                                                                                                    --fuser_type "$FUSER_TYPE" \
                                                                                                    --use_locoop_ood \
                                                                                                    --lambda_locoop_ood "$lambda_locoop_ood" \
                                                                                                    --locoop_topk "$locoop_topk" \
                                                                                                    --lambda_locoop_cls "$lambda_locoop_cls" \
                                                                                                    --lambda_locoop_patch "$lambda_locoop_patch" \
                                                                                                    --adapter_ratio "$adapter_ratio" \
                                                                                                    --residual_coef "$residual_coef" \
                                                                                                    --lambda_llm_negatives "$lambda_llm_negatives" \
                                                                                                    --lambda_mixup "$lambda_mixup" \
                                                                                                    --margin "$margin" \
                                                                                                    --lambda_intra_class "$lambda_intra_class" \
                                                                                                    --intra_class_temp "$intra_class_temp" \
                                                                                                    --num_select "$num_select" \
                                                                                                    --mlp_hidden_ratio "$mlp_hidden_ratio" \
                                                                                                    --slot_ffn_ratio "$slot_ffn_ratio" \
                                                                                                    --fuser_ffn_ratio "$fuser_ffn_ratio" \
                                                                                                    --selector_temperature "$selector_temperature" \
                                                                                                    --patches_per_slot_attn "$patches_per_slot_attn" \
                                                                                                    --score_type "$SCORE_TYPE" \
                                                                                                    --temperature "$TEMPERATURE" \
                                                                                                    --lambda_local "$LAMBDA_LOCAL" \
                                                                                                    --text_features_path "$TEXT_FEATURE_PATH" 2>&1 | tee -a "$LOG_DIR/grid_search.log"; then
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
