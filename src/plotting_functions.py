"""
This module provides visualization tools for evaluating uncertainty-aware classifiers. 
Functions include ROC-AUC plotting with optional confusion matrix display, and histogram 
visualization of calibration scores alongside their quantile thresholds.

Main Functions:
---------------
- compute_roc_auc_ovr: Plots ROC curves and confusion matrices, and saves AUROC metrics.
- plot_scores: Visualizes calibration score distributions with alpha-specific quantiles.

Intended for use in analysis pipelines that leverage Monte Carlo dropout and conformal 
prediction techniques.

Dependencies:
-------------
- matplotlib
- seaborn
- sklearn.metrics
- pandas
- numpy
- os
"""

#########
# Imports
#########

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from scipy.stats import pearsonr, spearmanr
from scipy.stats import bootstrap

from utility_functions import create_accuracy_and_count_grid_by_predicted_class
from calculation_functions import calculate_accuracy_and_count_by_predicted_class

####################
# Plotting Functions
####################

def compute_roc_auc_ovr(df, pred_probs_col, gt_col, plot_title, save_dir, save_name, removed_counts=None, percentage_removed=None, removed_data=None):

    """
        Compute and plot one-vs-rest ROC curves for multi-class classification, 
        save AUROC metrics, and optionally visualize the confusion matrix of removed instances.
        Currently not plotting class 1 or the overall AUC.

        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe containing prediction probabilities and ground truth labels.

        pred_probs_col : str
            The name of the column containing prediction probabilities (as lists or strings of lists).

        gt_col : str
            The name of the ground truth label column.

        plot_title : str
            The title for the ROC curve plot (currently unused).

        save_dir : str
            Directory where plots and CSVs should be saved.

        save_name : str
            Filename (with extension `.png`) for saving the ROC curve image. Also used as the prefix for related files.

        removed_counts : dict, optional
            Dictionary mapping class indices to the number of removed instances (used for text annotation on ROC).

        percentage_removed : dict, optional
            Dictionary mapping class indices to percentage of removed instances (used for text annotation on ROC).

        removed_data : pandas.DataFrame, optional
            If provided, a confusion matrix for the removed instances will also be saved.

        Outputs:
        --------
        - ROC curve plot for each class (excluding "Indeterminate" class by default).
        - Confusion matrix image for removed data (if provided).
        - CSV file summarizing AUROC values for each class and micro-averaged overall score.
    """

    plt.rcParams.update({
        'font.size': 26,
        'axes.titlesize': 28,
        'axes.labelsize': 26,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24,
    })

    def convert_strings_to_list(input_list):
        return [eval(i) if isinstance(i, str) else i for i in input_list]

    true_label = np.array(df[gt_col])

    try:
        model_prediction = convert_strings_to_list(list(df[pred_probs_col]))
        model_prediction = np.array(model_prediction)
    except:
        model_prediction = np.array(list(df[pred_probs_col]))

    if model_prediction.ndim == 1:
        model_prediction = model_prediction.reshape(-1, 1)

    true_label_bin = label_binarize(true_label, classes=np.arange(model_prediction.shape[1]))

    fpr = {}
    tpr = {}
    auroc = {}
    n_class = model_prediction.shape[1]
    missing_classes = []
    class_labels = {0: 'Normal', 1: 'Indeterminate', 2: 'Severe'}

    for i in range(n_class):
        try:
            fpr[i], tpr[i], _ = roc_curve(true_label_bin[:, i], model_prediction[:, i])
            auroc[i] = roc_auc_score(true_label_bin[:, i], model_prediction[:, i])
        except IndexError as e:
            print(f"Class {i} index error: {e}")
            missing_classes.append(i)
            fpr[i], tpr[i], auroc[i] = [0, 1], [0, 1], None
        except ValueError as e:
            if "Only one class present in y_true" in str(e):
                missing_classes.append(i)
                fpr[i], tpr[i], auroc[i] = [0, 1], [0, 1], None

    fpr_micro, tpr_micro, _ = roc_curve(true_label_bin.ravel(), model_prediction.ravel())
    auroc_micro = roc_auc_score(true_label_bin, model_prediction, average='micro')

    # ROC Curve Plot (Exclude Class 1 and Overall)
    fig, ax = plt.subplots(figsize=(16, 12))
    colors = plt.cm.get_cmap('tab10').colors
    plot_classes = [0, 2]  # Only Normal and Severe

    for i in plot_classes:
        label = class_labels.get(i, f'Class {i}')
        if auroc[i] is not None:
            ax.plot(fpr[i], tpr[i], linestyle='--', color=colors[i], linewidth=3, label=f'{label} vs Rest (AUC = {auroc[i]:.2f})')
        else:
            ax.plot(fpr[i], tpr[i], linestyle='--', color=colors[i], linewidth=3, label=f'{label} vs Rest (AUC = N/A)')

    ax.plot([0, 1], [0, 1], linestyle='-', color='red', linewidth=2, label='Random')
    ax.plot([0, 0, 1], [0, 1, 1], linestyle='-', color='black', linewidth=2, label='Theoretical Best')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    # ax.set_title(plot_title)

    if removed_counts is not None and percentage_removed is not None:
        textstr = '\n'.join(
            [f"{class_labels.get(i, f'Class {i}')}: Removed {removed_counts[i]} ({percentage_removed[i]:.2f}%)" for i in range(n_class)]
        )
        ax.text(0.53, 0.5, textstr, fontsize=22, bbox=dict(facecolor='white', alpha=0.5), transform=ax.transAxes)

    if save_name:
        fig.savefig(os.path.join(save_dir, save_name), bbox_inches='tight')
    else:
        fig.savefig(os.path.join(save_dir, 'roc_auc.png'), bbox_inches='tight')
    plt.close()

    # Confusion Matrix for Removed Data
    if removed_data is not None and not removed_data.empty:
        fig_cm, ax_cm = plt.subplots(figsize=(10, 10))
        gt_removed = removed_data[gt_col]
        pred_removed = removed_data['pred_class_mc']
        cm_removed = confusion_matrix(gt_removed, pred_removed, labels=[0, 1, 2])

        sns.heatmap(
            cm_removed,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[class_labels.get(i, f'Class {i}') for i in [0, 1, 2]],
            yticklabels=[class_labels.get(i, f'Class {i}') for i in [0, 1, 2]],
            ax=ax_cm,
            cbar=False,
            annot_kws={"size": 18}
        )
        ax_cm.set_xlabel('Predicted Label', fontsize=22)
        ax_cm.set_ylabel('True Label', fontsize=22)
        ax_cm.tick_params(axis='both', labelsize=20)
        ax_cm.set_title('Removed Instances Confusion Matrix', fontsize=26)

        cm_save_name = save_name.replace('.png', '_conf_matrix.png') if save_name else 'conf_matrix.png'
        fig_cm.savefig(os.path.join(save_dir, cm_save_name), bbox_inches='tight')
        plt.close()

    # AUROC CSV (includes all classes + overall)
    auc_df = pd.DataFrame({
        'Class': [class_labels.get(i, f'Class {i}') for i in range(n_class)] + ['Overall'],
        'ROC-AUC': list(auroc.values()) + [auroc_micro]
    })

    auc_save_name = f'{save_name}_roc_auc_values.csv' if save_name else 'roc_auc_values.csv'
    auc_df.to_csv(os.path.join(save_dir, auc_save_name), index=False)


def plot_confusion_matrix(removed_data, quality_df=None, fine_grain_df=None, title='', save_dir='', save_name=''):

    """
        Generate a detailed annotated confusion matrix for removed instances, optionally 
        enriched with prediction quality scores and fine-grained class distribution breakdowns.

        Parameters:
        -----------
        removed_data : pandas.DataFrame
            DataFrame containing 'gt' (ground truth), 'pred_class_mc' (predicted class), and 'image' or 'id' columns.

        quality_df : pandas.DataFrame, optional
            If provided, should contain predicted class quality scores mapped to 'MASKED_IMG_ID'. 
            These are merged with `removed_data` to compute average quality per confusion matrix cell.

        fine_grain_df : pandas.DataFrame, optional
            If provided, should contain fine-grained labels (e.g., `CC_ST`) mapped to image IDs (`id` column). 
            Used to calculate per-cell class distribution proportions.

        title : str
            Optional title for the confusion matrix plot (currently unused in final plot).

        save_dir : str
            Directory where the confusion matrix image and related CSVs will be saved.

        save_name : str
            Prefix for all output filenames (e.g., images, CSVs).

        Outputs:
        --------
        - Annotated confusion matrix heatmap saved as a PNG, including:
            * Count and percentage of instances in each cell.
            * Average quality score (if available).
        - CSV of the full `removed_data` DataFrame with added `conf_matrix_slot` index.
        - CSV showing per-cell fine-grain class proportions and counts (if `fine_grain_df` is provided).

        Notes:
        ------
        - Confusion matrix is computed over classes [0, 1, 2] and labeled as:
            * 0: Normal
            * 1: Indeterminate
            * 2: Severe
        - Useful for analyzing which types of errors dominate and how they differ across subgroups or quality strata.
    """

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
    sns.heatmap(cm_removed, annot=annotations, fmt='', cmap='Blues', xticklabels=['Normal', 'Indeterminate', 'Severe'], yticklabels=['Normal', 'Indeterminate', 'Severe'], cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # plt.title(title)
    
    fig.savefig(os.path.join(save_dir, f"{save_name}_confusion_matrix.png"), bbox_inches='tight')
    plt.clf()
    print(f"Confusion matrix saved to: {os.path.join(save_dir, f'{save_name}_confusion_matrix.png')}")

# Plot calibration scores and adjusted quantile for various alpha 

def plot_scores(alphas, cal_scores, quantiles, y_max):
    """
    Plot histogram of calibration scores and vertical lines for quantile thresholds.

    Parameters:
    -----------
    alphas : list of float
        List of alpha values (miscoverage levels).

    cal_scores : np.array
        Calibration scores.

    quantiles : list of float
        List of q̂ quantiles corresponding to each alpha.

    y_max : float
        Maximum y-axis value for plotting vertical lines.
    """

    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 12})
    plt.hist(cal_scores, bins="auto")
    for i, quantile in enumerate(quantiles):
        plt.vlines(
            x=quantile,
            ymin=0,
            ymax=y_max,
            color=colors[i],
            ls="dashed",
            label=f"alpha = {alphas[i]}"
        )
    plt.title("Distribution of scores")
    plt.legend()
    plt.xlabel("Scores")
    plt.ylabel("Count")
    plt.show()

def create_histogram_of_pred_length_size(df, mc_or_eval, title, save_dir):
    """
    Create a normalized histogram of prediction set lengths grouped by ground truth values.

    Parameters:
    - df: DataFrame containing the data.
    - mc_or_eval: String indicating whether to use 'mc' or 'eval' prediction set lengths.
    - title: Title for the plot.
    - save_dir: Directory to save the resulting plot.

    Returns:
    - None: The function saves the resulting plot as a PNG file.
    """

    pred_length_col = 'pred_set_eval_length' if mc_or_eval == 'eval' else 'pred_set_mc_length'

    # Group dataframe by 'gt' and 'pred_length' and count occurrences
    grouped = df.groupby(['gt', pred_length_col]).size().unstack(fill_value=0)

    # Normalize each row to get percentages
    grouped_normalized = grouped.div(grouped.sum(axis=0), axis=1) * 100

    # Plot histogram
    grouped_normalized.T.plot(kind='bar', stacked=True, color=plt.cm.get_cmap('tab10').colors[:len(grouped_normalized.columns)])

    # Add labels and title
    plt.xlabel('Prediction Set Length')
    plt.ylabel('Percentage')
    # plt.title('Normalized Histogram of Prediction Length for ' + title)

    # Add legend
    plt.legend(title='Ground Truth')

    save_path = os.path.join(save_dir, title + '_normalized_pred_length_100_pct.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to release memory


def plot_and_save_grid_by_predicted_class(grid, save_path):

    """
        Generate and save a heatmap visualizing accuracy by predicted class and prediction set length.

        Each cell in the heatmap displays the accuracy and number of samples (n) for a specific combination
        of predicted class (x-axis) and prediction set length (y-axis).

        Parameters:
        -----------
        grid : 2D array-like
            A 3x3 array where each cell contains a tuple (accuracy_percentage, count) for a specific combination
            of predicted class and prediction set length.

        save_path : str
            Directory where the resulting heatmap image should be saved. The output will be named
            'by_pred_class_and_set_length_cm_heatmap.png'.
    """

    accuracy_grid = np.array([[grid[i, j][0] for j in range(3)] for i in range(3)])
    count_grid = np.array([[grid[i, j][1] for j in range(3)] for i in range(3)])

    # Create annotation array with both accuracy and count
    annot_array = np.empty_like(accuracy_grid, dtype=object)
    for i in range(3):
        for j in range(3):
            if not np.isnan(accuracy_grid[i, j]):
                annot_array[i, j] = f'{accuracy_grid[i, j]:.1f}%\n(n={count_grid[i, j]})'
            else:
                annot_array[i, j] = 'N/A'

    plt.rcParams.update({
        'font.size': 22,
        'axes.titlesize': 24,
        'axes.labelsize': 22,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
    })

    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(
        accuracy_grid,
        annot=annot_array,
        fmt='',
        cmap='Blues',
        cbar=False,
        annot_kws={"size": 20},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Prediction Set Length')
    ax.set_xticklabels(['Normal', 'Indeterminate', 'Severe'], rotation=0)
    ax.set_yticklabels([1, 2, 3], rotation=0)

    plt.tight_layout()
    plt.savefig(f'{save_path}/by_pred_class_and_set_length_cm_heatmap.png', bbox_inches='tight')
    plt.close()


# Main function to calculate and save the accuracy grid by predicted class
def generate_accuracy_grid_by_predicted_class(df, pred_class_col, pred_set_length_col, save_path):

    """
        Compute and visualize model accuracy based on predicted class and prediction set length.

        This function wraps the logic for:
        1. Creating an accuracy and count grid based on combinations of predicted class and set length.
        2. Generating and saving a heatmap for quick visual inspection of how accuracy varies by set length and class.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing at least the predicted class and prediction set length columns.

        pred_class_col : str
            Name of the column in `df` containing the predicted class values (e.g., 0, 1, 2).

        pred_set_length_col : str
            Name of the column in `df` containing the prediction set lengths (e.g., 1, 2, 3).

        save_path : str
            Directory where the heatmap visualization will be saved.
    """

    # Create the accuracy and count grid by predicted class
    accuracy_and_count_grid = create_accuracy_and_count_grid_by_predicted_class(df, pred_class_col, pred_set_length_col)
    
    # Plot and save the grid
    plot_and_save_grid_by_predicted_class(accuracy_and_count_grid, save_path)

# def create_confusion_matrix_analysis(df, ground_truth_col, predicted_class_col, metric_to_include, metric_name, save_dir, title):
#     """
#     Create a confusion matrix with counts, percentages, and a specified metric included, 
#     and save as both CSV and PNG.

#     Parameters:
#     - df: DataFrame containing the data.
#     - ground_truth_col: Column name for ground truth labels.
#     - predicted_class_col: Column name for predicted class labels.
#     - metric_to_include: Column name for the metric to include in the confusion matrix (e.g., average set length).
#     - metric_name: Name of the metric to include in the title.
#     - save_dir: Directory to save the resulting files.
#     - title: Title for the files.

#     Returns:
#     - None: The function saves the resulting confusion matrix as CSV and PNG files.
#     """

#     plt.rcParams.update({
#         'font.size': 22,
#         'axes.titlesize': 24,
#         'axes.labelsize': 22,
#         'xtick.labelsize': 20,
#         'ytick.labelsize': 20,
#     })

#     # Creating a copy of the dataframe
#     df_copy = df.copy()

#     # Mapping labels to their corresponding names
#     label_mapping = {0: "Normal", 1: "Indeterminate", 2: "Severe"}
#     df_copy[ground_truth_col] = df_copy[ground_truth_col].map(label_mapping)
#     df_copy[predicted_class_col] = df_copy[predicted_class_col].map(label_mapping)

#     # Specify the desired order of classes
#     class_order = ["Normal", "Indeterminate", "Severe"]

#     # Create a pivot table to calculate the average set length for each combination of ground truth and predicted class
#     pivot_table = df_copy.pivot_table(index=ground_truth_col, columns=predicted_class_col, values=metric_to_include, aggfunc=np.mean)

#     # Reorder columns and index to match the desired order
#     pivot_table = pivot_table.reindex(index=class_order, columns=class_order)

#     # Replace NaN values with a numeric placeholder (e.g., 0)
#     pivot_table = pivot_table.fillna(0)

#     # Create a confusion matrix with counts
#     confusion_matrix_counts = pd.crosstab(df_copy[ground_truth_col], df_copy[predicted_class_col], rownames=['Ground Truth'], colnames=['Predicted Class'], dropna=False)
    
#     # Calculate total count for percentages
#     total_count = confusion_matrix_counts.values.sum()
#     cm_percentages = (confusion_matrix_counts / total_count) * 100

#     # Prepare annotations with counts, percentages, and average metric
#     annotations = np.empty_like(confusion_matrix_counts, dtype=object)

#     for i in range(confusion_matrix_counts.shape[0]):
#         for j in range(confusion_matrix_counts.shape[1]):
#             count = confusion_matrix_counts.iloc[i, j]  # The count (N) of items in this cell
#             percentage = cm_percentages.iloc[i, j]  # The percentage of total
#             avg_metric = pivot_table.iloc[i, j]  # The average metric value
            
#             if avg_metric == 0:
#                 annotations[i, j] = f'N={count}\n{percentage:.2f}%\nN/A'
#             else:
#                 annotations[i, j] = f'N={count}\n{percentage:.2f}%\n{avg_metric:.2f}'

#     # Plot the confusion matrix with the annotations (counts and percentages)
#     plt.figure(figsize=(14, 12))
#     sns.heatmap(confusion_matrix_counts, annot=annotations, fmt='', cmap='Blues', xticklabels=class_order, yticklabels=class_order, cbar=False)
#     plt.xlabel('Predicted Class')
#     plt.ylabel('Ground Truth')
#     # plt.title(f'{metric_name}')
#     plt.tight_layout()

#     # Save the confusion matrix as a PNG
#     save_path_plot = os.path.join(save_dir, f"{title}_confusion_matrix.png")
#     plt.savefig(save_path_plot, bbox_inches='tight')
#     plt.close()  # Close the plot to release memory

#     # Save the confusion matrix as a CSV
#     save_path_csv = os.path.join(save_dir, f"{title}_confusion_matrix.csv")
#     confusion_matrix_counts.to_csv(save_path_csv)
#     print(f"Confusion matrix saved to: {save_path_plot}")

from scipy.stats import bootstrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def create_confusion_matrix_analysis(df, ground_truth_col, predicted_class_col, metric_to_include, metric_name, save_dir, title):
    """
    Create a confusion matrix with counts, percentages, and a specified metric 
    (with bootstrapped 95% CI) included, and save as both CSV and PNG.
    """

    plt.rcParams.update({
        'font.size': 22,
        'axes.titlesize': 24,
        'axes.labelsize': 22,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
    })

    df_copy = df.copy()
    label_mapping = {0: "Normal", 1: "Indeterminate", 2: "Severe"}
    df_copy[ground_truth_col] = df_copy[ground_truth_col].map(label_mapping)
    df_copy[predicted_class_col] = df_copy[predicted_class_col].map(label_mapping)

    class_order = ["Normal", "Indeterminate", "Severe"]

    # Create base confusion matrix
    confusion_matrix_counts = pd.crosstab(
        df_copy[ground_truth_col],
        df_copy[predicted_class_col],
        rownames=['Ground Truth'],
        colnames=['Predicted Class'],
        dropna=False
    ).reindex(index=class_order, columns=class_order, fill_value=0)

    total_count = confusion_matrix_counts.values.sum()
    cm_percentages = (confusion_matrix_counts / total_count) * 100

    # Prepare annotations with count, percent, and metric + CI
    annotations = np.empty_like(confusion_matrix_counts, dtype=object)

    for i, gt_label in enumerate(class_order):
        for j, pred_label in enumerate(class_order):
            subset = df_copy[
                (df_copy[ground_truth_col] == gt_label) &
                (df_copy[predicted_class_col] == pred_label)
            ][metric_to_include]

            count = confusion_matrix_counts.loc[gt_label, pred_label]
            percent = cm_percentages.loc[gt_label, pred_label]

            if len(subset) == 0 or subset.isnull().all():
                annotations[i, j] = f'N={count}\n{percent:.2f}%\nN/A'
            elif len(subset) == 1:
                mean = subset.iloc[0]
                annotations[i, j] = f'N={count}\n{percent:.2f}%\n{mean:.2f} [N/A]'
            else:
                try:
                    res = bootstrap(
                        (subset.values,),
                        np.mean,
                        confidence_level=0.95,
                        n_resamples=1000,
                        method='basic',
                        random_state=42
                    )
                    mean = np.mean(subset)
                    lower, upper = res.confidence_interval
                    annotations[i, j] = f'N={count}\n{percent:.2f}%\n{mean:.2f} [{lower:.2f}, {upper:.2f}]'
                except Exception:
                    mean = np.mean(subset)
                    annotations[i, j] = f'N={count}\n{percent:.2f}%\n{mean:.2f} [Err]'

    # Plot with annotations
    plt.figure(figsize=(14, 12))
    sns.heatmap(confusion_matrix_counts, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_order, yticklabels=class_order, cbar=False)
    plt.xlabel('Predicted Class')
    plt.ylabel('Ground Truth')
    plt.tight_layout()

    save_path_plot = os.path.join(save_dir, f"{title}_confusion_matrix.png")
    plt.savefig(save_path_plot, bbox_inches='tight')
    plt.close()

    save_path_csv = os.path.join(save_dir, f"{title}_confusion_matrix.csv")
    confusion_matrix_counts.to_csv(save_path_csv)
    print(f"Confusion matrix saved to: {save_path_plot}")


def distribution_of_cov_color_coded_by_set_length(df, cov_col, pred_set_length_col, save_dir, title):
    """
    Create a KDE plot of coefficient of variation (CoV), color-coded by prediction set length.

    Parameters:
    - df: DataFrame containing the data.
    - cov_col: Column name for coefficient of variation.
    - pred_set_length_col: Column name for prediction set lengths.
    - save_dir: Directory to save the resulting plot.
    - title: Title for the plot.

    Returns:
    - None: The function saves the resulting plot as a PNG file.
    """
    # Set larger fonts globally
    plt.rcParams.update({
        'font.size': 26,
        'axes.titlesize': 28,
        'axes.labelsize': 26,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24,
    })

    unique_lengths = df[pred_set_length_col].unique()
    colors = plt.cm.get_cmap('tab10').colors[:len(unique_lengths)]
    color_dict = {length: colors[i] for i, length in enumerate(unique_lengths)}

    plt.figure(figsize=(20, 16))

    # Plot KDE plot for each prediction set length
    for pred_length, group_df in df.groupby(pred_set_length_col):
        sns.kdeplot(
            data=group_df[cov_col],
            color=color_dict[pred_length],
            label=f'Conformal Pred Length = {pred_length}',
            fill=True,
            linewidth=4
        )

    plt.xlabel('Coefficient of Variation (%)')
    plt.ylabel('Density')
    plt.legend(loc='best')

    # Save plot
    save_path = os.path.join(save_dir, title + '_distribution_cov.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def distribution_of_std_dev_color_coded_by_set_length(df, std_dev_col, pred_set_length_col, save_dir, title):
    """
    Create a KDE plot of standard deviations, color-coded by prediction set length.

    Parameters:
    - df: DataFrame containing the data.
    - std_dev_col: Column name for standard deviations.
    - pred_set_length_col: Column name for prediction set lengths.
    - save_dir: Directory to save the resulting plot.
    - title: Title for the plot.

    Returns:
    - None: The function saves the resulting plot as a PNG file.
    """
    unique_lengths = df[pred_set_length_col].unique()
    colors = plt.cm.get_cmap('tab10').colors[:len(unique_lengths)]
    color_dict = {length: colors[i] for i, length in enumerate(unique_lengths)}

    # Plot KDE plot for each prediction set length
    for pred_length, group_df in df.groupby(pred_set_length_col):
        sns.kdeplot(data=group_df[std_dev_col], color=color_dict[pred_length], label=f'Conformal Pred Length = {pred_length}', fill=True)

    plt.xlabel('Standard Deviation')
    plt.ylabel('Density')
    # plt.title(title)
    plt.legend()

    # Save plot
    save_path = os.path.join(save_dir, title + '_distribution_std_dev.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to release memory

def histogram_of_cov_color_coded_by_set_length(df, cov_col, pred_set_length_col, save_dir, title):
    """
    Create a histogram of coefficient of variations, color-coded by prediction set length with density normalization.

    Parameters:
    - df: DataFrame containing the data.
    - cov_col: Column name for coefficient of variations.
    - pred_set_length_col: Column name for prediction set lengths.
    - save_dir: Directory to save the resulting plot.
    - title: Title for the plot.

    Returns:
    - None: The function saves the resulting plot as a PNG file.
    """
    unique_lengths = df[pred_set_length_col].unique()
    colors = plt.cm.get_cmap('tab10').colors[:len(unique_lengths)]
    color_dict = {length: colors[i] for i, length in enumerate(unique_lengths)}

    for pred_length, group_df in df.groupby(pred_set_length_col):
        sns.histplot(group_df[cov_col], bins=5, kde=False, stat="density", 
                     label=f'Conformal Pred Length = {pred_length}', color=color_dict[pred_length], alpha=0.5)

    plt.xlabel('Coeff of Var (%)')
    plt.ylabel('Density')
    # plt.title(title)
    plt.legend()

    # Save plot
    save_path = os.path.join(save_dir, title + '.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to release memory

def histogram_of_std_dev_color_coded_by_set_length(df, std_dev_col, pred_set_length_col, save_dir, title):
    """
    Create a histogram of standard deviations, color-coded by prediction set length with density normalization.

    Parameters:
    - df: DataFrame containing the data.
    - std_dev_col: Column name for standard deviations.
    - pred_set_length_col: Column name for prediction set lengths.
    - save_dir: Directory to save the resulting plot.
    - title: Title for the plot.

    Returns:
    - None: The function saves the resulting plot as a PNG file.
    """
    unique_lengths = df[pred_set_length_col].unique()
    colors = plt.cm.get_cmap('tab10').colors[:len(unique_lengths)]
    color_dict = {length: colors[i] for i, length in enumerate(unique_lengths)}

    for pred_length, group_df in df.groupby(pred_set_length_col):
        sns.histplot(group_df[std_dev_col], bins=5, kde=False, stat="density", 
                     label=f'Conformal Pred Length = {pred_length}', color=color_dict[pred_length], alpha=0.5)

    plt.xlabel('Standard Deviation')
    plt.ylabel('Density')
    # plt.title(title)
    plt.legend()

    # Save plot
    save_path = os.path.join(save_dir, title + '.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to release memory

def expected_value_of_prediction_vs_std_dev_by_correct_or_not(df, expected_values_col, std_dev_col, correct_or_not, save_dir, title):
    """
    Create a scatter plot of expected values of predictions versus standard deviation, colored by correctness.

    Parameters:
    - df: DataFrame containing the data.
    - expected_values_col: Column name for expected values of predictions.
    - std_dev_col: Column name for standard deviations.
    - correct_or_not: Column name for correctness of predictions (True/False).
    - save_dir: Directory to save the resulting plot.
    - title: Title for the plot.

    Returns:
    - None: The function saves the resulting plot as a PNG file.
    """

    expected_values_of_predictions = df[expected_values_col]
    std_devs = df[std_dev_col]
    correct_or_not_colors = df[correct_or_not].map({True: 'green', False: 'red'})

    plt.scatter(expected_values_of_predictions, std_devs, color=correct_or_not_colors)
    plt.xlabel('Expected Values of Predictions')
    plt.ylabel('Standard Deviations of MC Predictions')
    # plt.title(title)
    plt.tight_layout()

    # Save plot
    save_path = os.path.join(save_dir, title + '.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to release memory

def expected_value_of_prediction_vs_cov_by_correct_or_not(df, expected_values_col, cov_col, correct_or_not, save_dir, title):
    """
    Create a scatter plot of expected values of predictions versus coefficient of variation (CoV), colored by correctness.

    Parameters:
    - df: DataFrame containing the data.
    - expected_values_col: Column name for expected values of predictions.
    - cov_col: Column name for coefficient of variation.
    - correct_or_not: Column name for correctness of predictions (True/False).
    - save_dir: Directory to save the resulting plot.
    - title: Title for the plot.

    Returns:
    - None: The function saves the resulting plot as a PNG file.
    """

    expected_values_of_predictions = df[expected_values_col]
    covs = df[cov_col]
    correct_or_not_colors = df[correct_or_not].map({True: 'green', False: 'red'})

    plt.scatter(expected_values_of_predictions, covs, color=correct_or_not_colors)
    plt.xlabel('Expected Values of Predictions')
    plt.ylabel('Coeff of Var (%) of MC Predictions')
    # plt.title(title)
    plt.tight_layout()

    # Save plot
    save_path = os.path.join(save_dir, title + '.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to release memory

def expected_value_of_prediction_vs_prediction_length_colored_by_correct_or_not(df, expected_values_col, pred_length_col, correct_or_not, save_dir, title):
     
    """
    Create a scatter plot of expected values of predictions versus prediction length, colored by correctness.

    Parameters:
    - df: DataFrame containing the data.
    - expected_values_col: Column name for expected values of predictions.
    - pred_length_col: Column name for prediction lengths.
    - correct_or_not: Column name for correctness of predictions (True/False).
    - save_dir: Directory to save the resulting plot.
    - title: Title for the plot.

    Returns:
    - None: The function saves the resulting plot as a PNG file.
    """
    
    expected_values_of_predictions = df[expected_values_col]
    prediction_lengths = df[pred_length_col]
    correct_or_not_colors = df[correct_or_not].map({True: 'green', False: 'red'})

    plt.scatter(expected_values_of_predictions, prediction_lengths, color=correct_or_not_colors)
    plt.xlabel('Expected Values of Predictions')
    plt.ylabel('Prediction Length')
    # plt.title('Expected Values of Predictions vs Prediction Length')
    plt.tight_layout()

    # Save plot
    save_path = os.path.join(save_dir, title + '.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to release memory

def create_classification_plot_per_class(df, selected_class, pred_or_gt, mc_or_eval, title, save_dir):
    """
    Create pie charts showing the distribution of prediction set lengths for correct and incorrect predictions for a given class.

    Parameters:
    - df: DataFrame containing the data.
    - selected_class: The class to analyze.
    - pred_or_gt: String indicating whether to use 'gt' (ground truth) or 'pred' (predicted) as reference.
    - mc_or_eval: String indicating whether to use 'mc' or 'eval' predictions.
    - title: Title for the plot.
    - save_dir: Directory to save the resulting plot.

    Returns:
    - None: The function saves the resulting plot as a PNG file.
    """

    reference_class = 'gt' if pred_or_gt == 'gt' else ('pred_class_eval' if mc_or_eval == 'eval' else 'pred_class_mc')
    pred_length_col = 'pred_set_eval_length' if mc_or_eval == 'eval' else 'pred_set_mc_length'
    correct_pred_col = 'correct_prediction_eval' if mc_or_eval == 'eval' else 'correct_prediction_mc'

    # Filter the DataFrame for correct predictions
    correct_df = df[(df[reference_class] == selected_class) & (df[correct_pred_col] == True)]
    correct_counts = correct_df[pred_length_col].value_counts()
    correct_percentage_counts = correct_df[pred_length_col].value_counts(normalize=True) * 100

    # Filter the DataFrame for incorrect predictions
    incorrect_df = df[(df[reference_class] == selected_class) & (df[correct_pred_col] == False)]
    incorrect_counts = incorrect_df[pred_length_col].value_counts()
    incorrect_percentage_counts = incorrect_df[pred_length_col].value_counts(normalize=True) * 100

    # Define a color dictionary for each prediction set length
    unique_lengths = df[pred_length_col].unique()
    colors = plt.cm.get_cmap('tab10').colors[:len(unique_lengths)]
    color_dict = {length: colors[i] for i, length in enumerate(unique_lengths)}

    # Create subplots with one row and two columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot pie chart for correct predictions
    axes[0].pie(correct_percentage_counts, labels=[f'{index} ({count})' for index, count in zip(correct_percentage_counts.index, correct_counts)], autopct='%1.1f%%', startangle=140, colors=[color_dict.get(index, 'tab:gray') for index in correct_percentage_counts.index])
    axes[0].set_title('Correct Predictions')

    # Plot pie chart for incorrect predictions
    axes[1].pie(incorrect_percentage_counts, labels=[f'{index} ({count})' for index, count in zip(incorrect_percentage_counts.index, incorrect_counts)], autopct='%1.1f%%', startangle=140, colors=[color_dict.get(index, 'tab:gray') for index in incorrect_percentage_counts.index])
    axes[1].set_title('Incorrect Predictions')

    # Add suptitle for the overall plot
    fig.suptitle(f'Percentage of Prediction Set Length for Class {selected_class} for ' + title, fontsize=16)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    for ax in axes:
        ax.axis('equal')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save plot
    save_path = os.path.join(save_dir, 'Reference_Type_' + pred_or_gt + '_Pred_Type_' + mc_or_eval + '_Class_' + str(selected_class) + '_pie_chart.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to release memory

def create_plot_std_dev_vs_pred_set_size(df, mc_or_eval, title, save_dir):
    """
    Create a box plot showing the standard deviation versus prediction set length.

    Parameters:
    - df: DataFrame containing the data.
    - mc_or_eval: String indicating whether to use 'mc' or 'eval' prediction set lengths.
    - title: Title for the plot.
    - save_dir: Directory to save the resulting plot.

    Returns:
    - None: The function saves the resulting plot as a PNG file.
    """
    pred_length_col = 'pred_set_eval_length' if mc_or_eval == 'eval' else 'pred_set_mc_length'

    # Create the box plot
    boxplot = sns.boxplot(x=pred_length_col, y='std_avg_soft_mc_pred', data=df, showmeans=True)

    # Calculate correlation coefficient
    spearman_corr, p_value = spearmanr(df[pred_length_col], df['std_avg_soft_mc_pred'])

    # Add labels and title
    plt.xlabel('Prediction Set Length')
    plt.ylabel('Standard Deviation')
    # plt.title(f'{title} (Correlation Coefficient: {spearman_corr:.2f})')

    # Add correlation coefficient and p-value as a text box
    plt.text(0.95, 0.95, f'Spearman: {spearman_corr:.2f}\nP-value: {p_value:.3f}', 
            horizontalalignment='right', 
            verticalalignment='top', 
            transform=plt.gca().transAxes, 
            bbox=dict(facecolor='white', alpha=0.5))

    # Save plot
    save_path = os.path.join(save_dir, title + '_boxplot.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to release memory

def create_scatter_cov_vs_avg_mc_pred_lengths(df, mc_or_eval, title, save_dir):
    """
    Create a scatter plot showing the coefficient of variation versus prediction set length,
    color-coded by predicted class. Calculates overall and per-class Pearson correlations.

    Parameters:
    - df: DataFrame containing the data.
    - mc_or_eval: String indicating whether to use 'mc' or 'eval' prediction set lengths.
    - title: Title for the plot.
    - save_dir: Directory to save the resulting plot.

    Returns:
    - None: The function saves the resulting plot as a PNG file.
    """

    # Set larger font sizes
    plt.rcParams.update({
        'font.size': 26,
        'axes.titlesize': 28,
        'axes.labelsize': 26,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 20,
    })

    pred_length_col = 'pred_set_eval_length' if mc_or_eval == 'eval' else 'pred_set_mc_length'

    # Define a custom colormap excluding orange, green, and blue
    custom_colors = [
    # '#ff7f0e',  # Bright Orange
    # '#1f77b4',  # Bold Blue
    # '#2ca02c',  # Strong Green
    '#d62728',  # Vivid Red
    '#9467bd',  # Medium Purple
    '#17becf']  # Cyan-Teal

    class_labels = {0: 'Normal', 1: 'Indeterminate', 2: 'Severe'}
    unique_classes = sorted(df['pred_class_mc'].unique())
    color_dict = {cls: custom_colors[i % len(custom_colors)] for i, cls in enumerate(unique_classes)}

    # Create figure
    plt.figure(figsize=(20, 16))

    # Plot each class separately with distinct color
    for cls in unique_classes:
        class_df = df[df['pred_class_mc'] == cls]
        label_name = class_labels.get(cls, f'Class {cls}')
        sns.scatterplot(
            x='avg_mc_pred_length',
            y='cov_mc',
            data=class_df,
            label=label_name,
            color=color_dict[cls],
            s=120,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )

    # Calculate and collect overall Pearson correlation
    pearson_corr, p_value = pearsonr(df['cov_mc'], df['avg_mc_pred_length'])
    overall_corr_text = f'Overall Pearson: r={pearson_corr:.2f}'

    # Compute per-class correlations
    per_class_corr = [overall_corr_text]
    for cls in unique_classes:
        class_df = df[df['pred_class_mc'] == cls]
        label_name = class_labels.get(cls, f'Class {cls}')
        if len(class_df) > 1:
            corr, pval = pearsonr(class_df['cov_mc'], class_df['avg_mc_pred_length'])
            text = f'{label_name}: r={corr:.2f}'
        else:
            text = f'{label_name}: Insufficient data'
        per_class_corr.append(text)

    # Add combined annotation block to top left
    combined_text = '\n'.join(per_class_corr)
    plt.text(
        0.02, 0.98,
        combined_text,
        horizontalalignment='left',
        verticalalignment='top',
        transform=plt.gca().transAxes,
        fontsize=26,
        bbox=dict(facecolor='white', alpha=0.6)
    )

    # Final plot formatting
    plt.xlabel('Average MC Prediction Set Length')
    plt.ylabel('Coefficient of Variation')
    plt.legend(loc='best', title='Predicted Class')
    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, title + '_scatter.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def create_plot_cov_vs_pred_set_size(df, mc_or_eval, title, save_dir):
    """
    Create a box plot showing the coefficient of variation versus prediction set length.

    Parameters:
    - df: DataFrame containing the data.
    - mc_or_eval: String indicating whether to use 'mc' or 'eval' prediction set lengths.
    - title: Title for the plot.
    - save_dir: Directory to save the resulting plot.

    Returns:
    - None: The function saves the resulting plot as a PNG file.
    """
    pred_length_col = 'pred_set_eval_length' if mc_or_eval == 'eval' else 'pred_set_mc_length'

    # Create the box plot
    boxplot = sns.boxplot(x=pred_length_col, y='cov_mc', data=df, showmeans=True)

    # Calculate correlation coefficient
    spearman_corr, p_value = spearmanr(df[pred_length_col], df['cov_mc'])

    # Add labels and title
    plt.xlabel('Prediction Set Length')
    plt.ylabel('Coefficient of Variation (%)')
    # plt.title(f'{title} (Spearman Correlation: {spearman_corr:.2f})')

    # Add correlation coefficient and p-value as a text box
    plt.text(0.95, 0.95, f'Spearman: {spearman_corr:.2f}\nP-value: {p_value:.3f}', 
            horizontalalignment='right', 
            verticalalignment='top', 
            transform=plt.gca().transAxes, 
            bbox=dict(facecolor='white', alpha=0.5))

    # Save plot
    save_path = os.path.join(save_dir, title + '_boxplot.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to release memory

def create_histogram_of_pred_length_size(df, mc_or_eval, title, save_dir):
    """
    Create a normalized histogram of prediction set lengths grouped by ground truth values.

    Parameters:
    - df: DataFrame containing the data.
    - mc_or_eval: String indicating whether to use 'mc' or 'eval' prediction set lengths.
    - title: Title for the plot.
    - save_dir: Directory to save the resulting plot.

    Returns:
    - None: The function saves the resulting plot as a PNG file.
    """

    pred_length_col = 'pred_set_eval_length' if mc_or_eval == 'eval' else 'pred_set_mc_length'

    # Group dataframe by 'gt' and 'pred_length' and count occurrences
    grouped = df.groupby(['gt', pred_length_col]).size().unstack(fill_value=0)

    # Normalize each row to get percentages
    grouped_normalized = grouped.div(grouped.sum(axis=0), axis=1) * 100

    # Plot histogram
    grouped_normalized.T.plot(kind='bar', stacked=True, color=plt.cm.get_cmap('tab10').colors[:len(grouped_normalized.columns)])

    # Add labels and title
    plt.xlabel('Prediction Set Length')
    plt.ylabel('Percentage')
    # plt.title('Normalized Histogram of Prediction Length for ' + title)

    # Add legend
    plt.legend(title='Ground Truth')

    save_path = os.path.join(save_dir, title + '_normalized_pred_length_100_pct.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to release memory