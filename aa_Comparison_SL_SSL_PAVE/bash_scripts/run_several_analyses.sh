#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Path to the Python script
PYTHON_SCRIPT_PATH="/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/src/full_avg_mc_pred_sets_pipeline.py"

# Models' prediction CSV files array (add your csv paths here)
PREDICTIONS_CSV_PATHS=(
    "/scratch90/chris/Cervical_Cancer_Projects/cervical_cancer_diagnosis/experiments/SSL/M36_Pipeline/M36_Norms/DINO_DN121/inference_test2/predictions.csv"
    "/scratch90/chris/Cervical_Cancer_Projects/cervical_cancer_diagnosis/experiments/SSL/M36_Pipeline/M36_Norms/DINO_RN50/inference_test2/predictions.csv"
    # "/scratch90/chris/Cervical_Cancer_Projects/cervical_cancer_diagnosis/experiments/SSL/M36_Pipeline/M36_Norms/DINO_EffHF/inference_test2/predictions.csv"
    "/scratch90/chris/Cervical_Cancer_Projects/cervical_cancer_diagnosis/experiments/SSL/M36_Pipeline/M36_Norms/DINO_EffNetV2S/inference_test2/predictions.csv"
    # "/scratch90/chris/Cervical_Cancer_Projects/cervical_cancer_diagnosis/experiments/SSL/M36_Pipeline/M36_Norms/DINO_ViTb16/inference_test2/predictions.csv"
    # "/scratch90/chris/Cervical_Cancer_Projects/cervical_cancer_diagnosis/experiments/SSL/M36_Pipeline/M36_Norms/ViTMAE/inference_test2/predictions.csv"
    # "/scratch90/chris/Cervical_Cancer_Projects/cervical_cancer_diagnosis/experiments/SSL/M36_Pipeline/M36_Norms/EffHF/inference_test2/predictions.csv"
    "/scratch90/chris/Cervical_Cancer_Projects/cervical_cancer_diagnosis/experiments/SSL/M36_Pipeline/M36_Norms/EffNetV2S/inference_test2/predictions.csv"
    "/scratch90/chris/Cervical_Cancer_Projects/cervical_cancer_diagnosis/experiments/SSL/M36_Pipeline/M36_Norms/RN50/inference_test2/predictions.csv"
    # "/scratch90/chris/Cervical_Cancer_Projects/cervical_cancer_diagnosis/experiments/SSL/M36_Pipeline/M36_Norms/ViTb16/inference_test2/predictions.csv"
    # "/scratch90/chris/Cervical_Cancer_Projects/cervical_cancer_diagnosis/experiments/SSL/M36_Pipeline/M36_Norms/EffNetV2S_GHL_Pipeline/inference_test2/predictions.csv"
    # "/scratch90/chris/Cervical_Cancer_Projects/cervical_cancer_diagnosis/experiments/SSL/M36_Pipeline/M36_Norms/Scott_Model_Best_Overall/inference_test2/predictions.csv"
    # "/scratch90/chris/Cervical_Cancer_Projects/cervical_cancer_diagnosis/experiments/SSL/M36_Pipeline/M36_Norms/Scott_Model_Best_PC/inference_test2/predictions.csv"
    # "/scratch90/chris/Cervical_Cancer_Projects/cervical_cancer_diagnosis/experiments/SSL/M36_Pipeline/M36_Norms/model36_large_image/inference_test2/predictions.csv"
    # "/scratch90/chris/Cervical_Cancer_Projects/cervical_cancer_diagnosis/experiments/SSL/M36_Pipeline/M36_Norms/RN50_large_image/inference_test2/predictions.csv"
    # "/scratch90/chris/Cervical_Cancer_Projects/cervical_cancer_diagnosis/experiments/SSL/M36_Pipeline/M36_Norms/EffNetV2S_large_image/inference_test2/predictions.csv"

    "/scratch90/chris/Cervical_Cancer_Projects/cervical_cancer_diagnosis/experiments/SSL/M36_Pipeline/M36_Norms/Model36/inference_test2/predictions.csv"
    # Add more paths as needed
)

# Ground truth CSV path
GT_CSV_PATH="/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/csvs/SEED_three_class_ground_truths.csv"

# Dataset filter
DATASET_FILTER="test2"

# Parameters arrays
# CUQ_TYPES=("lac" "aps")
# ALPHAS=("0.05" "0.1" "0.2")
CUQ_TYPES=("lac")
ALPHAS=("0.1")
NUM_MC_COLS="50"
NUM_CLASSES="3"
LABEL_COL_NAME="CC_ST"

# Function to create a save path based on input parameters
generate_save_path() {
    local original_base_dir=$1
    local cuq_type=$2
    local alpha=$3
    local num_classes=$4
    local base_save_dir="/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/analysis/test2"
    local alpha_str=$(printf "%.2f" "$alpha" | sed 's/\.//')
    local relative_path=${original_base_dir#*/experiments/}  # Trim the path up to "experiments/"

    echo "$base_save_dir/$relative_path/$num_classes/$cuq_type/alpha_$alpha_str"
}

# Iterate over each model's predictions CSV
for PREDICTIONS_CSV_PATH in "${PREDICTIONS_CSV_PATHS[@]}"; do
    # Extract the base directory from the CSV path
    ORIGINAL_BASE_DIR=$(dirname "$PREDICTIONS_CSV_PATH")
    
    for CUQ_TYPE in "${CUQ_TYPES[@]}"; do
        for ALPHA in "${ALPHAS[@]}"; do
            SAVE_PATH=$(generate_save_path "$ORIGINAL_BASE_DIR" "$CUQ_TYPE" "$ALPHA" "$NUM_CLASSES")
            echo "Running for ORIGINAL_BASE_DIR=$ORIGINAL_BASE_DIR, CUQ_TYPE=$CUQ_TYPE, ALPHA=$ALPHA, SAVE_PATH=$SAVE_PATH"

            # Create the save directory if it doesn't exist
            mkdir -p "$SAVE_PATH"

            # Run the Python script with the current set of parameters
            python "$PYTHON_SCRIPT_PATH" \
                --predictions_csv_path "$PREDICTIONS_CSV_PATH" \
                --gt_csv_path "$GT_CSV_PATH" \
                --dataset_filter "$DATASET_FILTER" \
                --cuq_type "$CUQ_TYPE" \
                --alpha "$ALPHA" \
                --cuq_df_save_dir "$SAVE_PATH" \
                --num_mc_cols "$NUM_MC_COLS" \
                --num_classes "$NUM_CLASSES" \
                --label_col_name "$LABEL_COL_NAME" \
                --save_path "$SAVE_PATH"
        done
    done
done