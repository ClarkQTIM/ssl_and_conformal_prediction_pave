import argparse
import os
import sys
import pandas as pd
import numpy as np
import ast
from create_cuq_csv import additions_to_csvs, create_cuq_df, create_mc_df, compute_roc_auc_ovr
from testing_adding_conf_int_mc_cervix_analysis_cov_std_dev import analyse_cuq_df_two_class, analyse_cuq_df_three_class, create_confusion_matrix
from adding_pred_length_to_class_inference import parse_confusion_matrix, filter_data, determine_proper_cutoffs

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def convert_strings_to_list(list_of_strings):
    return [ast.literal_eval(string) for string in list_of_strings]

def plot_confusion_matrix(removed_data, quality_df=None, fine_grain_df=None, title='', save_dir='', save_name=''):
    if removed_data.empty:
        print(f"No instances removed for {title}")
        return

    # Check if quality_df is provided
    if quality_df is not None:
        try:
            quality_df = quality_df.rename(columns={'predicted_mc_class': 'pred_mc_quality'})
            quality_df = quality_df[['MASKED_IMG_ID', 'pred_mc_quality']]
            removed_data = removed_data.merge(quality_df, left_on='image', right_on='MASKED_IMG_ID', suffixes=('_removed', '_quality'))
        except Exception as e:
            print(f'Error during merging with quality_df: {e}')
            return
    else:
        # Add a default column for pred_mc_quality
        removed_data['pred_mc_quality'] = None

    # Check if fine_grain_df is provided
    if fine_grain_df is not None:
        try:
            removed_data = removed_data.merge(fine_grain_df[['ID', 'CC_ST']], left_on='id', right_on='ID', how='left')
        except Exception as e:
            print(f'Error during merging with fine_grain_df: {e}')
            return
    else:
        removed_data['CC_ST'] = 'Unknown'

    # Calculate confusion matrix and percentages
    gt_removed = removed_data['gt']
    pred_removed = removed_data['pred_class_mc']
    cm_removed = confusion_matrix(gt_removed, pred_removed, labels=[0, 1, 2])

    total_count = len(removed_data)
    cm_percentages = (cm_removed / total_count) * 100

    # Calculate confusion matrix slot for each image
    def get_conf_matrix_slot(row):
        return row['gt'] * 3 + row['pred_class_mc'] + 1

    removed_data['conf_matrix_slot'] = removed_data.apply(get_conf_matrix_slot, axis=1)

    # Save the updated removed_data DataFrame to CSV
    removed_data_csv_path = os.path.join(save_dir, f"{save_name}_removed_data_with_slots.csv")
    removed_data.to_csv(removed_data_csv_path, index=False)
    print(f"Removed data with confusion matrix slots saved to: {removed_data_csv_path}")

    fine_grain_classes = sorted(removed_data['CC_ST'].unique()) if 'CC_ST' in removed_data else []

    fine_grain_csv_data = []

    annotations = np.empty_like(cm_removed).astype(str)
    for i in range(cm_removed.shape[0]):
        for j in range(cm_removed.shape[1]):
            count = cm_removed[i, j]
            percentage = cm_percentages[i, j]
            subset = removed_data[(removed_data['gt'] == i) & (removed_data['pred_class_mc'] == j)]

            if not subset.empty:
                pred_mc_values = subset['pred_mc_quality']
                avg_quality = np.mean(pred_mc_values) if pred_mc_values.notnull().any() else None

                fine_grain_props = subset['CC_ST'].value_counts(normalize=True)
                fine_grain_counts = subset['CC_ST'].value_counts()
                fine_grain_combined = [f'{fine_grain_props.get(c, 0):.2%} ({fine_grain_counts.get(c, 0)})' for c in fine_grain_classes]

                fine_grain_csv_data.append([i, j, *fine_grain_combined])
            else:
                avg_quality = None
                fine_grain_combined = ['0.00% (0)'] * len(fine_grain_classes)
                fine_grain_csv_data.append([i, j, *fine_grain_combined])

            avg_quality_str = f'{avg_quality:.2f}' if avg_quality is not None else 'N/A'
            annotations[i, j] = f'{count}\n({percentage:.2f}%)\n{avg_quality_str}'

    if fine_grain_df is not None:
        fine_grain_df = pd.DataFrame(fine_grain_csv_data, columns=['True_Label', 'Predicted_Label'] + [f'Class_{c}_Proportion' for c in fine_grain_classes])
        fine_grain_csv_path = os.path.join(save_dir, f"{save_name}_fine_grain_props.csv")
        fine_grain_df.to_csv(fine_grain_csv_path, index=False)
        print(f"Fine-grain proportions saved to: {fine_grain_csv_path}")

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm_removed, annot=annotations, fmt='', cmap='Blues', xticklabels=['Normal', 'Gray Zone', 'Precancer+'], yticklabels=['Normal', 'Gray Zone', 'Precancer+'], cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # plt.title(title)
    
    fig.savefig(os.path.join(save_dir, f"{save_name}_confusion_matrix.png"), bbox_inches='tight')
    plt.clf()
    print(f"Confusion matrix saved to: {os.path.join(save_dir, f'{save_name}_confusion_matrix.png')}")

def save_removed_data(removed_data, title, save_dir, save_name):
    if removed_data.empty:
        print(f"No instances removed for {title}")
        return

    save_path = os.path.join(save_dir, f"{save_name}.csv")
    removed_data.to_csv(save_path, index=False)
    print(f"Removed data saved to {save_path}")

def main(predictions_csv_path, gt_csv_path, dataset_filter, cuq_type, alpha, cuq_df_save_dir, num_mc_cols, num_classes, label_col_name, save_path):

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
    quality_df = pd.read_csv('/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/csvs/IQ_test2.csv')
    fine_grain_df = pd.read_csv('/scratch90/chris/Chris_Active/Comparison_SL_SSL_PAVE/csvs/testset_endpoint_20240716.csv')
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
        create_confusion_matrix(filtered_data, 'gt', 'pred_class_mc', 'pred_set_mc_length', confusion_matrix_pred_size_title, save_path, confusion_matrix_pred_size_filename)
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
