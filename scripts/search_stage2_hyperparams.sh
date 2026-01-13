#!/bin/bash

# ============================================================================
# Stage 2 Hyperparameter Search Script
# ============================================================================

# This script searches over different hyperparameters for Stage 2 training:
# - Learning rates
# - Adapter hidden dimensions
# - Weighted pooling options
# 
# For each combination, it runs Stage 2 training and evaluates ID accuracy.
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Experiment name prefix
EXPERIMENT_NAME="stage2_hyperparam_search"

# Dataset configuration
ID_DATASET="ImageNet"
ROOT_PATH="/data/ICML2026/clip/FA/my_dataset"

# Stage 1 checkpoint (required)
STAGE1_CKPT="/data/ICML2026/GL_MCM_FA/logs/GL_MCM_FA_mlp_42_20260105_173642/checkpoints/epoch_999.pt"

# Training configuration
EPOCHS=10
LR_LIST=(0.0001 0.0005 0.001 0.005)
BATCH_SIZE=64
SEED=42
DEVICE="cuda"

# Adapter configurations
ADAPTER_HIDDEN_DIMS=(256 512 768 1024)
USE_WEIGHTED_POOL_OPTIONS=(true false)

# Output directory
OUTPUT_BASE="/data/ICML2026/GL_MCM_FA/${EXPERIMENT_NAME}"
mkdir -p ${OUTPUT_BASE}

# Log file
LOG_FILE="${OUTPUT_BASE}/search_results.log"

# ============================================================================
# Helper Functions
# ============================================================================

log_message() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] $1" | tee -a ${LOG_FILE}
}

run_single_experiment() {
    local lr=$1
    local adapter_dim=$2
    local use_weighted_pool=$3
    local exp_name="${EXPERIMENT_NAME}_lr${lr}_dim${adapter_dim}_pool${use_weighted_pool}"
    
    log_message "========================================"
    log_message "Starting experiment: ${exp_name}"
    log_message "LR: ${lr}, Adapter Dim: ${adapter_dim}, Weighted Pool: ${use_weighted_pool}"
    log_message "========================================"
    
    # Create experiment directory
    local exp_dir="${OUTPUT_BASE}/${exp_name}"
    mkdir -p ${exp_dir}
    mkdir -p ${exp_dir}/checkpoints
    
    # Run Stage 2 training
    local cmd="python src/train_eval.py \
        --train_stage 2 \
        --stage1_checkpoint ${STAGE1_CKPT} \
        --stage2_epochs ${EPOCHS} \
        --stage2_lr ${lr} \
        --batch_size ${BATCH_SIZE} \
        --seed ${SEED} \
        --device ${DEVICE} \
        --id_dataset ${ID_DATASET} \
        --root_path ${ROOT_PATH} \
        --selector_type slot \
        --fuser_type self_attn \
        --score_type GL-MCM \
        --lambda_local 1.0"
    
    # Add adapter_hidden_dim if specified
    if [ "${adapter_dim}" != "None" ] && [ "${adapter_dim}" != "" ]; then
        cmd="${cmd} --adapter_hidden_dim ${adapter_dim}"
    fi
    
    # Add use_weighted_pool flag if true
    if [ "${use_weighted_pool}" = "true" ]; then
        cmd="${cmd} --use_weighted_pool"
    fi
    
    # Run the command
    ${cmd} > ${exp_dir}/train.log 2>&1
    
    local exit_code=$?
    
    if [ ${exit_code} -eq 0 ]; then
        log_message "✓ Experiment ${exp_name} completed successfully"
        
        # Extract best metrics from results.json
        if [ -f "${exp_dir}/stage2_results.json" ]; then
            local metrics=$(python3 -c "
import json
with open('${exp_dir}/stage2_results.json', 'r') as f:
    results = json.load(f)
    # Find best id_acc
    best_id_acc = max([r.get('id_acc', 0) for r in results])
    # Find best DS-MCM avg auroc
    best_ds_auroc = max([r.get('DS-MCM_avg_auroc', 0) for r in results])
    # Find best Visual-GL-MCM avg auroc
    best_visual_auroc = max([r.get('Visual-GL-MCM_avg_auroc', 0) for r in results])
    # Find best Stage1-GL-MCM avg auroc
    best_stage1_auroc = max([r.get('Stage1-GL-MCM_avg_auroc', 0) for r in results])
    print(f'{best_id_acc:.2f},{best_ds_auroc:.2f},{best_visual_auroc:.2f},{best_stage1_auroc:.2f}')
")
            IFS=',' read -r id_acc ds_auroc visual_auroc stage1_auroc <<< "$metrics"
            log_message "Best ID Acc: ${id_acc}%, DS-MCM AUROC: ${ds_auroc}%, Visual-GL-MCM AUROC: ${visual_auroc}%, Stage1-GL-MCM AUROC: ${stage1_auroc}%"
            echo "${exp_name},${lr},${adapter_dim},${use_weighted_pool},${id_acc}%,${ds_auroc}%,${visual_auroc}%,${stage1_auroc}%" >> ${OUTPUT_BASE}/summary.csv
        else
            log_message "✗ Experiment ${exp_name} failed: results.json not found"
            echo "${exp_name},${lr},${adapter_dim},${use_weighted_pool},FAILED,FAILED,FAILED,FAILED" >> ${OUTPUT_BASE}/summary.csv
        fi
    else
        log_message "✗ Experiment ${exp_name} failed with exit code ${exit_code}"
        echo "${exp_name},${lr},${adapter_dim},${use_weighted_pool},FAILED,FAILED,FAILED,FAILED" >> ${OUTPUT_BASE}/summary.csv
    fi
    
    log_message ""
}

# ============================================================================
# Main Search Loop
# ============================================================================

log_message "Starting Stage 2 hyperparameter search"
log_message "Output directory: ${OUTPUT_BASE}"
log_message "Log file: ${LOG_FILE}"
log_message ""

# Create summary CSV header
echo "Experiment,LR,AdapterDim,WeightedPool,Best_ID_Acc,DS-MCM_AUROC,Visual-GL-MCM_AUROC,Stage1-GL-MCM_AUROC" > ${OUTPUT_BASE}/summary.csv

# Run all combinations
total_experiments=0
for lr in "${LR_LIST[@]}"; do
    for adapter_dim in "${ADAPTER_HIDDEN_DIMS[@]}"; do
        for use_weighted_pool in "${USE_WEIGHTED_POOL_OPTIONS[@]}"; do
            run_single_experiment ${lr} ${adapter_dim} ${use_weighted_pool}
            total_experiments=$((total_experiments + 1))
        done
    done
done

# ============================================================================
# Summary
# ============================================================================

log_message "========================================"
log_message "Search completed!"
log_message "Total experiments: ${total_experiments}"
log_message "Summary saved to: ${OUTPUT_BASE}/summary.csv"
log_message "========================================"

# Print summary table
log_message ""
log_message "Summary of Results:"
log_message "-------------------"
echo ""
printf "%-30s | %-10s | %-10s | %-15s | %-12s | %-12s | %-12s | %-12s\n" "Experiment" "LR" "Adapter Dim" "Weighted Pool" "ID Acc" "DS-MCM" "Visual-GL-MCM" "Stage1-GL-MCM"
echo "-------------------"

# Read and display summary
if [ -f "${OUTPUT_BASE}/summary.csv" ]; then
    tail -n +2 ${OUTPUT_BASE}/summary.csv | while IFS=',' read -r exp lr dim pool id_acc ds_auroc visual_auroc stage1_auroc; do
        printf "%-30s | %-10s | %-10s | %-15s | %-12.2f%% | %-12.2f%% | %-12.2f%% | %-12.2f%%\n" "$exp" "$lr" "$dim" "$pool" "$id_acc" "$ds_auroc" "$visual_auroc" "$stage1_auroc"
    done
    echo "-------------------"
fi

log_message ""
log_message "Check ${LOG_FILE} for detailed logs"
log_message "Check ${OUTPUT_BASE} for individual experiment results"
