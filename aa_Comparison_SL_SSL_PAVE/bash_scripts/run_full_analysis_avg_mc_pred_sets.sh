#!/bin/bash

# Path to the Python script
PYTHON_SCRIPT_PATH="/sddata/projects/Conformal_Uncertainty_Quantification/MICCAI_UNSURE_Workshop_Anon_Git/src/full_avg_mc_pred_sets_pipeliney.py"

# Command-line arguments for the Python script
PREDICTIONS_CSV_PATH="/sddata/projects/Cervical_Cancer_Projects/cervical_cancer_diagnosis/csvs/model_36_predictions/model_36_test2_predictions_all_cols.csv"
CUQ_TYPE="lac"
ALPHA="0.1"
NUM_MC_COLS="50"
NUM_CLASSES="3"
LABEL_COL_NAME="CC_ST"
SAVE_PATH="/sddata/projects/Conformal_Uncertainty_Quantification/MICCAI_UNSURE_Workshop/analysis/full_run"

# Run the Python script with the provided arguments
python $PYTHON_SCRIPT_PATH \
    --predictions_csv_path $PREDICTIONS_CSV_PATH \
    --cuq_type $CUQ_TYPE \
    --alpha $ALPHA \
    --cuq_df_save_dir $SAVE_PATH \
    --num_mc_cols $NUM_MC_COLS \
    --num_classes $NUM_CLASSES \
    --label_col_name $LABEL_COL_NAME \
    --save_path $SAVE_PATH
