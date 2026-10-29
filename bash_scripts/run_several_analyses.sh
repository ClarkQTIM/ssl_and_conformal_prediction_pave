#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# === Path to the Python script ===
PYTHON_SCRIPT_PATH="/scratch90/chris/ssl_conformal_pave/src/main.py"

# === Models' prediction CSV files ===
PREDICTIONS_CSV_PATHS_RENAMED=(
    "/scratch90/chris/ssl_conformal_pave/csvs/inference_csvs/dino_dn121_seed_test2_predictions.csv"
    "/scratch90/chris/ssl_conformal_pave/csvs/inference_csvs/dino_rn50_seed_test2_predictions.csv"
    "/scratch90/chris/ssl_conformal_pave/csvs/inference_csvs/dino_effnetv2s_seed_test2_predictions.csv"
    "/scratch90/chris/ssl_conformal_pave/csvs/inference_csvs/effnetv2s_seed_test2_predictions.csv"
    "/scratch90/chris/ssl_conformal_pave/csvs/inference_csvs/rn50_seed_test2_predictions.csv"
    "/scratch90/chris/ssl_conformal_pave/csvs/inference_csvs/dn121_seed_test2_predictions.csv"
)

# === Ground truth CSV path ===
GT_CSV_PATH="/scratch90/chris/ssl_conformal_pave/csvs/SEED_three_class_ground_truths.csv"

# === Dataset filter and experiment config ===
DATASET_FILTER="test2"
CUQ_TYPES=("lac")  # You can add "aps" etc.
ALPHAS=("0.1")      # You can add more, like "0.05" "0.2"
NUM_MC_COLS="50"
NUM_CLASSES="3"
LABEL_COL_NAME="CC_ST"

# === Function to create a save path from the filename ===
generate_save_path() {
    local predictions_csv_path=$1
    local cuq_type=$2
    local alpha=$3
    local num_classes=$4

    local base_save_dir="analysis/six_model_seed_test2"
    local alpha_str=$(printf "%.2f" "$alpha" | sed 's/\.//')

    local filename=$(basename "$predictions_csv_path")
    local model_name=${filename%%_seed_test2_predictions.csv}  # Trim at that specific pattern

    echo "$base_save_dir/$model_name/$num_classes/$cuq_type/alpha_$alpha_str"
}

# === Loop through each model CSV and run ===
for PREDICTIONS_CSV_PATH in "${PREDICTIONS_CSV_PATHS_RENAMED[@]}"; do
    for CUQ_TYPE in "${CUQ_TYPES[@]}"; do
        for ALPHA in "${ALPHAS[@]}"; do
            SAVE_PATH=$(generate_save_path "$PREDICTIONS_CSV_PATH" "$CUQ_TYPE" "$ALPHA" "$NUM_CLASSES")
            
            echo ""
            echo ">>> Running for:"
            echo "Model File:     $PREDICTIONS_CSV_PATH"
            echo "CUQ Type:       $CUQ_TYPE"
            echo "Alpha:          $ALPHA"
            echo "Save Path:      $SAVE_PATH"

            mkdir -p "$SAVE_PATH"

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
