import argparse
import os
import sys
import pandas as pd
import numpy as np
import ast

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from plotting_functions import plot_confusion_matrix, compute_roc_auc_ovr
from utility_functions import convert_strings_to_list, save_removed_data, parse_confusion_matrix, determine_proper_cutoffs, filter_data, additions_to_csvs
from analysis_functions import analyse_cuq_df_two_class, analyse_cuq_df_three_class, create_confusion_matrix_analysis
from create_cuq_csv import create_cuq_df, create_mc_df


def main(predictions_csv_path, gt_csv_path, dataset_filter, cuq_type, alpha, cuq_df_save_dir, num_mc_cols, num_classes, label_col_name, save_path):

    """
        Main pipeline for conformal uncertainty quantification (CUQ) analysis, including data merging,
        prediction quality assessment, ROC-AUC computation, filtering uncertain predictions, and visualization.

        Parameters:
        -----------
        predictions_csv_path : str
            Path to the CSV file containing model prediction outputs and associated metadata.

        gt_csv_path : str
            Path to the ground truth CSV file, containing labels and dataset source information.

        dataset_filter : str
            A string used to filter rows in the ground truth DataFrame to match the desired dataset subset.

        cuq_type : str
            The conformal uncertainty quantification method to be used (e.g., "mc" for Monte Carlo).

        alpha : float
            The miscoverage level used for CUQ calibration and prediction set construction.

        cuq_df_save_dir : str
            Directory where intermediate CUQ results will be saved.

        num_mc_cols : int
            Number of Monte Carlo dropout samples (i.e., columns of prediction probabilities).

        num_classes : int
            Number of classes in the classification task (e.g., 2 for binary, 3 for ordinal tasks).

        label_col_name : str
            The column name in the prediction DataFrame containing the ground truth labels.

        save_path : str
            Directory where all generated plots and final analysis outputs will be saved.

        Workflow Summary:
        -----------------
        1. Loads and merges prediction and ground truth data.
        2. Computes initial ROC-AUC across all predictions.
        3. Runs CUQ calibration and prediction set generation.
        4. Analyzes results by class count (2-class or 3-class).
        5. Generates confusion matrix and detailed analysis plots on unfiltered data.
        6. Applies filtering to remove uncertain predictions based on floor-average thresholds:
            - Recomputes ROC-AUC after filtering.
            - Visualizes new confusion matrices.
            - Saves CSVs with removed sample info and class-wise breakdowns.

        Outputs:
        --------
        - ROC curves (before and after filtering).
        - Confusion matrices (standard and prediction set size based).
        - CSV files of CUQ metrics, removed instances, and fine-grain label breakdowns.
    """


    # Ensure the save directories exist
    os.makedirs(cuq_df_save_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # Load prediction and ground truth data
    predictions_df = pd.read_csv(predictions_csv_path)
    gt_df = pd.read_csv(gt_csv_path)
    gt_df_subset = gt_df[gt_df['dataset'] == dataset_filter]

    # Find common images and merge
    merged_df = predictions_df.merge(gt_df_subset[['MASKED_IMG_ID', 'CC_ST']], on='MASKED_IMG_ID')
    ## Check if CC_ST_x is all NaN
    if merged_df['CC_ST_x'].isna().all():
        ### Use CC_ST_y and rename it to CC_ST
        merged_df.rename(columns={'CC_ST_y': 'CC_ST'}, inplace=True)
        ### Optionally drop CC_ST_x if it's not needed anymore
        merged_df.drop(columns=['CC_ST_x'], inplace=True)
    else:
        ### Use CC_ST_x and rename it to CC_ST
        merged_df.rename(columns={'CC_ST_x': 'CC_ST'}, inplace=True)
        ### Optionally drop CC_ST_y if it's not needed anymore
        merged_df.drop(columns=['CC_ST_y'], inplace=True)
   
    # Feed the merged dataframe into the function
    predictions_df = additions_to_csvs(merged_df)
    compute_roc_auc_ovr(predictions_df, 'pred_mc', label_col_name, 'ROC-AUC', save_path, 'roc_auc_whole_dataset.png') # Initial ROC
    
    try:
        cuq_df_eval = create_cuq_df(cuq_type, predictions_df, label_col_name, 'pred', 'predicted_class', 'MASKED_IMG_ID', 'ID',alpha, 0.2, True, num_mc_cols, num_classes)
        cuq_df = create_mc_df(cuq_type, predictions_df, cuq_df_eval, label_col_name, alpha, num_mc_cols, num_classes, cuq_df_save_dir)
    except:
        num_mc_cols=20
        cuq_df_eval = create_cuq_df(cuq_type, predictions_df, label_col_name, 'pred', 'predicted_class', 'MASKED_IMG_ID', 'ID', alpha, 0.2, True, num_mc_cols, num_classes)
        cuq_df = create_mc_df(cuq_type, predictions_df, cuq_df_eval, label_col_name, alpha, num_mc_cols, num_classes, cuq_df_save_dir)
    
    # Analysis
    if num_classes == 2:
        print('We are running a two-class analysis')
        analyse_cuq_df_two_class(cuq_df, 'mc', save_path, os.path.basename(save_path))
    elif num_classes == 3:
        print('We are running a three-class analysis')
        analyse_cuq_df_three_class(cuq_df, 'mc', save_path, os.path.basename(save_path))

    # Post-added analysis
    quality_df = pd.read_csv('/scratch90/chris/ssl_conformal_pave/csvs/IQ_test2.csv')
    fine_grain_df = pd.read_csv('/scratch90/chris/ssl_conformal_pave/csvs/testset_endpoint_20240716.csv')
    plot_confusion_matrix(cuq_df, quality_df, fine_grain_df, 'Classification Performance Before Removing any Images', save_path, 'before_removing_any_images')
    # Post-Processing by Removing Uncertain Elements
    ## Original counts of each class
    original_counts = cuq_df['gt'].value_counts()

    ## Second filtering and ROC-AUC
    cutoff_0, cutoff_1, cutoff_2 = determine_proper_cutoffs(cuq_df)

    ## Apply filtering for each class individually and then all classes combined
    for target_class in [0, 1, 2, None]:
        if target_class is not None:
            print(f"Filtering for class {target_class}")
        else:
            print("Filtering for all classes combined")
        
        filtered_data, removed_data = filter_data(cuq_df, [cutoff_0, cutoff_1, cutoff_2], target_class)
        filtered_counts = filtered_data['gt'].value_counts()
        removed_counts = original_counts - filtered_counts
        percentage_removed = (removed_counts / original_counts) * 100
        
        # Create filenames dynamically based on the target class
        if target_class is not None:
            roc_auc_title = f'ROC-AUC with Uncertain Predictions by Floor Average Excluded for Class {target_class}'
            roc_auc_filename = f'filtered_by_floor_average_roc_auc_class_{target_class}.png'
            confusion_matrix_classification_title = f'Classification Performance After Removing Uncertain Class {target_class} by Floor Average'
            confusion_matrix_classification_filename = f'after_removal_by_floor_average_class_{target_class}'
            confusion_matrix_pred_size_title = f'Average Prediction Set Sizes After Removing Uncertain Class {target_class} by Floor Average'
            confusion_matrix_pred_size_filename = f'after_removal_by_floor_average_class_{target_class}'
            removed_data_filename = f'removed_data_by_floor_average_class_{target_class}.csv'
        else:
            roc_auc_title = f'ROC-AUC with Uncertain Predictions by Floor Average Excluded for All Classes'
            roc_auc_filename = f'filtered_by_floor_average_roc_auc_all_classes.png'
            confusion_matrix_classification_title = f'Classification Performance After Removing All Uncertain Classes by Floor Average'
            confusion_matrix_classification_filename = f'after_removal_by_floor_average_all_classes'
            confusion_matrix_pred_size_title = f'Average Prediction Set Sizes After Removing All Uncertain Classes by Floor Average'
            confusion_matrix_pred_size_filename = f'after_removal_by_floor_average_all_classes'
            removed_data_filename = f'removed_data_by_floor_average_all_classes.csv'

            
        compute_roc_auc_ovr(filtered_data, 'pred_probs_mc', 'gt', roc_auc_title, save_path, roc_auc_filename, removed_counts, percentage_removed, removed_data)
        plot_confusion_matrix(filtered_data, quality_df, fine_grain_df, confusion_matrix_classification_title, save_path, confusion_matrix_classification_filename)
        create_confusion_matrix_analysis(filtered_data, 'gt', 'pred_class_mc', 'pred_set_mc_length', confusion_matrix_pred_size_title, save_path, confusion_matrix_pred_size_filename)
        save_removed_data(removed_data, confusion_matrix_classification_title, save_path, removed_data_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CUQ Analysis")
    parser.add_argument('-p', '--predictions_csv_path', type=str, required=True, help="Path to predictions CSV")
    parser.add_argument('-g', '--gt_csv_path', type=str, required=True, help="Path to ground truth CSV")
    parser.add_argument('-d', '--dataset_filter', type=str, required=True, help="Dataset filter (e.g., 'test2')")
    parser.add_argument('-t', '--cuq_type', type=str, default='lac', help="CUQ type")
    parser.add_argument('-a', '--alpha', type=float, default=0.1, help="Alpha value")
    parser.add_argument('-s', '--cuq_df_save_dir', type=str, required=True, help="Directory to save CUQ dataframe")
    parser.add_argument('-m', '--num_mc_cols', type=int, default=50, help="Number of Monte Carlo columns")
    parser.add_argument('-c', '--num_classes', type=int, default=3, help="Number of classes")
    parser.add_argument('-l', '--label_col_name', type=str, required=True, help="Label column name")
    parser.add_argument('-o', '--save_path', type=str, required=True, help="Path to save analysis results")

    args = parser.parse_args()
    print(args)

    main(args.predictions_csv_path, args.gt_csv_path, args.dataset_filter, args.cuq_type, args.alpha, args.cuq_df_save_dir, args.num_mc_cols, args.num_classes, args.label_col_name, args.save_path)
