#########
# Imports
#########

import numpy as np
import pandas as pd
import os
import sys
import torch
import ast
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import label_binarize
from cuq_classification_algorithms import *

'''
3/21/2025: Needed for paper
'''

###########
# Functions
###########

def compute_roc_auc_ovr(df, pred_probs_col, gt_col, plot_title, save_dir, save_name, removed_counts=None, percentage_removed=None, removed_data=None):

     """
    Compute and plot one-vs-rest ROC curves for multi-class classification, 
    save AUROC metrics, and optionally visualize the confusion matrix of removed instances.

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
    class_labels = {0: 'Normal', 1: 'Indeterminate', 2: 'Precancer+'}

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
    plot_classes = [0, 2]  # Only Normal and Precancer+

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

def additions_to_csvs(df):
    mc_cols = [col for col in df.columns if 'mc_' in col]
    mc_cols = [col for col in mc_cols if col not in ['pred_mc', 'predicted_mc_class', 'soft_mc_pred', 'soft_mc_prediction']]

    average_mc_expected_values = []
    rounded_average_mc_expected_values = []
    standard_deviations_mc_expected_values = []
    coefficients_of_variation_mc_expected_values = []

    for index, row in df.iterrows():
        mc_expected_values = []
        for col in mc_cols:
            mc_probs = ast.literal_eval(row[col])
            if len(mc_probs) == 3:
                mc_expected_values.append(0*mc_probs[0] + 1*mc_probs[1] + 2*mc_probs[2])
            elif len(mc_probs) == 2:
                mc_expected_values.append(0*mc_probs[0] + 1*mc_probs[1])

        average_mc_expected_value = np.average(mc_expected_values)
        standard_deviation_mc_expected_value = np.std(mc_expected_values)
        coefficient_of_variation_mc_expected_value = (standard_deviation_mc_expected_value / average_mc_expected_value) * 100

        average_mc_expected_values.append(average_mc_expected_value)
        rounded_average_mc_expected_values.append(round(average_mc_expected_value))
        standard_deviations_mc_expected_values.append(standard_deviation_mc_expected_value)
        coefficients_of_variation_mc_expected_values.append(coefficient_of_variation_mc_expected_value)

    df.insert(len(df.columns)-4, 'avg_soft_mc_pred', average_mc_expected_values)
    df.insert(len(df.columns)-4, 'rounded_avg_soft_mc_pred', rounded_average_mc_expected_values)
    df.insert(len(df.columns)-4, 'std_avg_soft_mc_pred', standard_deviations_mc_expected_values)
    df.insert(len(df.columns)-4, 'cv_avg_soft_mc_pred', coefficients_of_variation_mc_expected_values)

    average_mc_probs = []
    for index, row in df.iterrows():
        class_0 = []
        class_1 = []
        if len(mc_probs) == 3:
            class_2 = []
        for col in mc_cols:
            mc_probs = ast.literal_eval(row[col])
            class_0.append(mc_probs[0])
            class_1.append(mc_probs[1])
            if len(mc_probs) == 3:
                class_2.append(mc_probs[2])
        if len(mc_probs) == 3:
            average_mc_probs.append([np.average(class_0), np.average(class_1), np.average(class_2)])
        else:
            average_mc_probs.append([np.average(class_0), np.average(class_1)])
    df.insert(len(df.columns)-4, 'avg_mc_probs_per_class', average_mc_probs)
    df.insert(len(df.columns)-4, 'argmax_mc_probs_per_class', np.argmax(np.array(average_mc_probs), axis=1))

    return df

def calculate_coeff_of_var(df, num_mc, num_classes):
    """
    Calculate the coefficients of variation for Monte Carlo predictions.

    Parameters:
    - df: DataFrame containing the Monte Carlo predictions in string format.
    - num_mc: Number of Monte Carlo samples.
    - num_classes: Number of classes in the classification task.

    Returns:
    - df: DataFrame with an additional column 'cov' containing the coefficients of variation for each image.
    """

    mc_predictions_cols = ['mc_epistemic_' + str(i) for i in range(num_mc)]
    mc_predictions = df[mc_predictions_cols].applymap(ast.literal_eval).values  # Convert strings to lists

    # Reshape mc_predictions to have three dimensions
    num_images = mc_predictions.shape[0]
    num_mc_samples = mc_predictions.shape[1]
    mc_predictions_reshaped = np.empty((num_images, num_mc_samples, num_classes), dtype=object)

    # Populate the reshaped array with the Monte Carlo predictions
    for i in range(num_images):
        for j in range(num_mc_samples):
            mc_pred_list = mc_predictions[i, j]
            for k in range(num_classes):
                mc_predictions_reshaped[i, j, k] = mc_pred_list[k]

    # Calculate the coefficients of variation
    coefficients_of_variation = []

    for i in range(num_images):
        mc_preds_for_image = mc_predictions_reshaped[i]  # Shape: (num_MC_samples, num_classes)
        evs = []

        # Calculate expected values for each Monte Carlo sample
        for mc_sample in range(num_mc_samples):
            expected_value = np.dot(mc_preds_for_image[mc_sample], np.arange(num_classes))
            evs.append(expected_value + 1)
        
        # Compute standard deviation and mean of the expected values
        std_pred = np.std(evs)
        mean_pred = np.mean(evs)
        
        # Avoid division by zero by adding a small epsilon to the mean_pred if it's zero
        if mean_pred == 0:
            mean_pred += 1e-10
        
        # Calculate the coefficient of variation 
        coefficient_of_variation = (std_pred / mean_pred) * 100
        coefficients_of_variation.append(coefficient_of_variation)

    # Convert the list of coefficients of variation to a numpy array
    coefficients_of_variation_array = np.array(coefficients_of_variation)

    # Add the coefficients of variation as a new column in the DataFrame
    df['cov'] = coefficients_of_variation_array

    return df

def convert_strings_to_list(list_of_strings):
    return [ast.literal_eval(string) for string in list_of_strings]

def create_cuq_df(cuq_type, predictions_df, gt_col, pred_probs_col, pred_class_col, img_id_col, patient_id_col, alpha, pct_cal, print_or_not, num_mc, num_classes):
    """
    Create a DataFrame containing conformal uncertainty quantification (CUQ) information.

    Parameters:
    - cuq_type: Type of CUQ algorithm ('lac' or 'aps').
    - predictions_df: DataFrame containing predictions and other relevant data.
    - gt_col: Column name for ground truth values.
    - pred_probs_col: Column name for predicted probabilities.
    - pred_class_col: Column name for predicted classes (optional).
    - img_id_col: Column name for image IDs.
    - alpha: Significance level for prediction intervals.
    - pct_cal: Proportion of data to be used for calibration.
    - print_or_not: Boolean flag to print results.
    - num_mc: Number of Monte Carlo samples.
    - num_classes: Number of classes in the classification task.

    Returns:
    - cuq_df: DataFrame with CUQ results and additional metrics.
    """

    # Organizing our data
    gt_values = list(predictions_df[gt_col])
    pred_probs = convert_strings_to_list(list(predictions_df[pred_probs_col]))
    try:
        cov_values = list(calculate_coeff_of_var(predictions_df, num_mc, num_classes)['cov'])
    except:
        print('We were unable to calculate the coefficients of variation. Are we supposed to?')
    if pred_class_col is not None:
        pred_class = list(predictions_df[pred_class_col])
    else:
        print('We have no dedicated predicted class, so we are argmaxing over the given probability columns.')
        pred_class = [np.argmax(pred_prob) for pred_prob in pred_probs]
    img_ids = list(predictions_df[img_id_col])
    patient_ids = list(predictions_df[patient_id_col])

    # Division into calibration and testing
    test_names, cal_names, \
    test_ids, cal_ids, \
    test_pred_classes, cal_pred_classes, \
    test_probs, cal_probs, \
    test_gt, cal_gt, \
    test_cov, cal_cov = train_test_split(img_ids, patient_ids, pred_class, np.array(pred_probs), np.array(gt_values).astype(int), cov_values, test_size=pct_cal, random_state=0)

    # Getting the calibration scores
    if cuq_type == 'lac':
        cal_scores = get_calibration_scores_lac(cal_gt, cal_probs)
    elif cuq_type == 'aps':
        cal_scores = get_calibration_scores_aps(cal_gt, cal_probs)
    else:
        print('Currently, our only two options for CUQ algorithms are lac and aps. Exiting')
        sys.exit()

    # Prediction sets
    if cuq_type == 'lac':
        try:
            prediction_sets, mean_width, covered, pct_coverage, qhat = cuq_pred_set_lac(cal_scores, alpha, test_probs, test_gt)
        except:
            prediction_sets, mean_width, qhat = cuq_pred_set_lac(cal_scores, alpha, test_probs, test_gt)
    elif cuq_type == 'aps':
        try:
            prediction_sets, mean_width, covered, pct_coverage, qhat = cuq_pred_set_aps(cal_scores, alpha, test_probs, test_gt)
        except:
            prediction_sets, mean_width, qhat = cuq_pred_set_aps(cal_scores, alpha, test_probs, test_gt)

    # Printing
    if print_or_not:
        print(f'Example prediction set: {prediction_sets[0]}')
        print(f'Mean width: {mean_width}')
        try:
            print(f'Percent/Empirical Coverage {pct_coverage}')
        except:
            print('No percent coverage')
        print(f'Associated q_hat: {qhat}')

    # Convert list of lists to a list of tuples to create a Pandas Series
    prediction_sets_series = pd.Series([tuple(item) for item in prediction_sets])
    prediction_set_length = [sum(item) for item in prediction_sets_series]

    # Create a dictionary with all the columns
    cuq_dict = {
        'image': test_names,
        'id': test_ids,
        'probs': list(test_probs),
        'pred_class': test_pred_classes,
        'pred_set': prediction_sets_series,
        'pred_set_length': prediction_set_length,
        'gt': test_gt,
    }

    # Conditionally add 'covered' and 'cov' if they exist
    if 'covered' in locals():
        cuq_dict['covered'] = covered
    if 'test_cov' in locals():
        cuq_dict['cov'] = test_cov

    # Create a DataFrame from the dictionary
    cuq_df = pd.DataFrame(cuq_dict)

    # Adding soft prediction
    cuq_df['soft_prediction'] = predictions_df[predictions_df['MASKED_IMG_ID'].isin(test_names)]['soft_prediction'].tolist()
    
    # Misclassifications
    cuq_df['correct_prediction'] = cuq_df['gt'] == cuq_df['pred_class']
    ## Calculate absolute difference between ground truth and predicted class
    diff = cuq_df['gt'] - cuq_df['pred_class']
    abs_diff = abs(cuq_df['gt'] - cuq_df['pred_class'])
    cuq_df['extreme_misclassification'] = abs_diff == (num_classes - 1)  # Extreme misclassification: difference == 2 for three-class, 1 for two-class
    if num_classes == 3:
        cuq_df['normal_misclassified_as_pcplus'] = (cuq_df['gt'] == 0) & (cuq_df['pred_class'] == 2)
        cuq_df['pcplus_misclassified_as_normal'] = (cuq_df['gt'] == 2) & (cuq_df['pred_class'] == 0)
        cuq_df['normal_misclassified_as_gz'] = (cuq_df['gt'] == 0) & (cuq_df['pred_class'] == 1)
        cuq_df['gz_misclassified_as_normal'] = (cuq_df['gt'] == 1) & (cuq_df['pred_class'] == 0)
        cuq_df['pcplus_misclassified_as_gz'] = (cuq_df['gt'] == 2) & (cuq_df['pred_class'] == 1)
        cuq_df['gz_misclassified_as_pcplus'] = (cuq_df['gt'] == 1) & (cuq_df['pred_class'] == 2)
    elif num_classes == 2:
        cuq_df['misclassified_as_opposite_class'] = (cuq_df['gt'] != cuq_df['pred_class'])
    cuq_df['cov'] = cuq_df['cov']

    return cuq_df

def create_mc_df(cuq_type, predictions_df, cuq_df_eval, label_col_name, alpha, num_mc_cols, num_classes, save_dir):
    """
    Create a DataFrame for Monte Carlo (MC) predictions and evaluate their performance.

    Parameters:
    - cuq_type: Type of CUQ algorithm ('lac' or 'aps').
    - predictions_df: DataFrame containing predictions and other relevant data.
    - cuq_df_eval: Evaluation DataFrame containing CUQ evaluation results.
    - label_col_name: Column name for ground truth labels.
    - alpha: Significance level for prediction intervals.
    - num_mc_cols: Number of Monte Carlo columns.
    - num_classes: Number of classes in the classification task.
    - save_dir: Directory to save the resulting DataFrame as a CSV file.

    Note, this function will calculate a conformal prediction for each MC run, and then vote for each class on whether that 
    class is included in the final conformal prediction set for the datum. 

    Returns:
    - None: Full Conformal Prediction dataframe ready for further analysis
    """

    mc_df = pd.DataFrame()
    for i in range(num_mc_cols):
        mc_col = 'mc_epistemic_' + str(i)
        cuq_df = create_cuq_df(cuq_type, predictions_df, label_col_name, mc_col, 'rounded_avg_soft_mc_pred', 'MASKED_IMG_ID', 'ID', alpha, 0.2, False, num_mc_cols, num_classes)
        mc_df['pred_set_' + str(i)] = cuq_df['pred_set']

    mc_df.insert(0, 'image', cuq_df['image'])
    mc_df.insert(1, 'id', cuq_df['id'])
    mc_df.insert(1, 'gt', cuq_df['gt'])
    mc_df.insert(2, 'pred_class_mc', cuq_df['pred_class'])
    mc_df.insert(3, 'pred_class_eval', cuq_df_eval['pred_class'])

    mc_df.insert(4, 'soft_mc_prediction', predictions_df[predictions_df['MASKED_IMG_ID'].isin(cuq_df['image'])]['soft_mc_prediction'].tolist())
    mc_df.insert(5, 'soft_eval_prediction', cuq_df_eval['soft_prediction'])

    mc_df = pd.merge(mc_df, predictions_df[['MASKED_IMG_ID', 'std_avg_soft_mc_pred']], left_on='image', right_on='MASKED_IMG_ID', how='inner')
    mc_df.pop('MASKED_IMG_ID')
    std_avg_soft_mc_pred = mc_df['std_avg_soft_mc_pred']
    mc_df.pop('std_avg_soft_mc_pred')
    mc_df.insert(6, 'std_avg_soft_mc_pred', std_avg_soft_mc_pred)

    voting_outcomes = []
    avg_mc_pred_lengths = []
    for index, row in mc_df.iterrows():  # Iterating over the rows
        pred_lengths = []
        class_vals = [[] for _ in range(num_classes)]
        for i in range(num_mc_cols):  # Iterating over the pred_set columns
            pred_set = row['pred_set_' + str(i)]
            length = np.sum(pred_set)
            pred_lengths.append(length)
            for j in range(num_classes):  # Iterating over the pred set indices
                val = row['pred_set_' + str(i)][j]
                class_vals[j].append(val)
        final_vote = [np.average(class_vals[k]) >= .5 for k in range(len(class_vals))]
        voting_outcomes.append(final_vote)
        avg_mc_pred_lengths.append(np.mean(pred_lengths))

    mc_covered = np.array(voting_outcomes)[np.arange(len(mc_df)), list(mc_df['gt'])]
    mc_df.insert(7, 'pred_set_mc', voting_outcomes)
    mc_df.insert(7, 'avg_mc_pred_length', avg_mc_pred_lengths)
    mc_df.insert(8, 'pred_set_eval', cuq_df_eval['pred_set'])
    mc_df.insert(9, 'covered_by_pred_set_mc', mc_covered)
    mc_df.insert(10, 'covered_by_pred_set_eval', cuq_df_eval['covered'])
    mc_df.insert(11, 'pred_set_mc_length', [sum(pred_set) for pred_set in voting_outcomes])
    mc_df.insert(12, 'pred_set_eval_length', cuq_df_eval['pred_set_length'])

    # Misclassifications
    diff = mc_df['gt'] - mc_df['pred_class_mc']
    abs_diff = np.abs(diff)

    # Add new columns indicating extreme misclassifications for MC
    correct_prediction = mc_df['gt'] == mc_df['pred_class_mc']
    mc_df.insert(13, 'correct_prediction_mc', correct_prediction)
    extreme_misclassification = abs_diff == (num_classes - 1)  # Extreme misclassification: difference == 2 for three-class, 1 for two-class
    mc_df.insert(14, 'extreme_misclassification_mc', extreme_misclassification)

    if num_classes == 3:
        mc_df.insert(15, 'normal_misclassified_as_pcplus_mc', (mc_df['gt'] == 0) & (mc_df['pred_class_mc'] == 2))
        mc_df.insert(16, 'pcplus_misclassified_as_normal_mc', (mc_df['gt'] == 2) & (mc_df['pred_class_mc'] == 0))
        mc_df.insert(17, 'normal_misclassified_as_gz_mc', (mc_df['gt'] == 0) & (mc_df['pred_class_mc'] == 1))
        mc_df.insert(18, 'gz_misclassified_as_normal_mc', (mc_df['gt'] == 1) & (mc_df['pred_class_mc'] == 0))
        mc_df.insert(19, 'pcplus_misclassified_as_gz_mc', (mc_df['gt'] == 2) & (mc_df['pred_class_mc'] == 1))
        mc_df.insert(20, 'gz_misclassified_as_pcplus_mc', (mc_df['gt'] == 1) & (mc_df['pred_class_mc'] == 2))
    elif num_classes == 2:
        mc_df.insert(15, 'misclassified_as_opposite_class_mc', (mc_df['gt'] != mc_df['pred_class_mc']))

    # Add new columns indicating extreme misclassifications for eval
    mc_df.insert(21, 'correct_prediction_eval', cuq_df_eval['correct_prediction'])
    mc_df.insert(22, 'extreme_misclassification_eval', cuq_df_eval['extreme_misclassification'])

    if num_classes == 3:
        mc_df.insert(23, 'normal_misclassified_as_pcplus_eval', cuq_df_eval['normal_misclassified_as_pcplus'])
        mc_df.insert(24, 'pcplus_misclassified_as_normal_eval', cuq_df_eval['pcplus_misclassified_as_normal'])
        mc_df.insert(25, 'normal_misclassified_as_gz_eval', cuq_df_eval['normal_misclassified_as_gz'])
        mc_df.insert(26, 'gz_misclassified_as_normal_eval', cuq_df_eval['gz_misclassified_as_normal'])
        mc_df.insert(27, 'pcplus_misclassified_as_gz_eval', cuq_df_eval['pcplus_misclassified_as_gz'])
        mc_df.insert(28, 'gz_misclassified_as_pcplus_eval', cuq_df_eval['gz_misclassified_as_pcplus'])
    elif num_classes == 2:
        mc_df.insert(23, 'misclassified_as_opposite_class_eval', (cuq_df_eval['gt'] != cuq_df_eval['pred_class']))

    mc_df.insert(29, 'cov_eval', cuq_df_eval['cov'])
    mc_df.insert(30, 'cov_mc', cuq_df['cov'])

    '''
    7/10/2024 Clean this up!
    '''
    mc_df.insert(2, 'pred_probs_eval', cuq_df_eval['probs'])
    cuq_df_to_get_mc_avg_probs = create_cuq_df(cuq_type, predictions_df, label_col_name, 'pred_mc', 'rounded_avg_soft_mc_pred', 'MASKED_IMG_ID', 'ID', alpha, 0.2, False, num_mc_cols, num_classes)
    mc_df.insert(3, 'pred_probs_mc', cuq_df_to_get_mc_avg_probs['probs'])

    if cuq_type == 'lac':
        mc_df.to_csv(os.path.join(save_dir, 'mc_lac_run_alpha' + str(alpha) + '.csv'))
    elif cuq_type == 'aps':
        mc_df.to_csv(os.path.join(save_dir, 'mc_aps_run_alpha' + str(alpha) + '.csv'))

    return mc_df

if __name__ == '__main__':
    # predictions_csv_path = '/sddata/projects/Cervical_Cancer_Projects/cervical_cancer_diagnosis/csvs/misc/model_36_test2_predictions_all_cols.csv'
    predictions_csv_path = '/sddata/projects/Cervical_Cancer_Projects/cervical_cancer_diagnosis/csvs/model_36_predictions/model_36_test2_predictions_all_cols.csv'
    # predictions_csv_path = '/sddata/projects/Cervical_Cancer_Projects/cervical_cancer_diagnosis/csvs/model_36_predictions/model_36_test2_binary.csv'
    cuq_type = 'lac'
    alpha = 0.1
    save_dir = '/sddata/projects/Conformal_Uncertainty_Quantification/MICCAI_UNSURE_Workshop/analysis/testing'
    num_mc_cols = 50
    num_classes = 3  # Change to 2 for two-class model
    label_col_name = 'CC_ST'

    predictions_df = pd.read_csv(predictions_csv_path)  # Load the predictions DataFrame
    cuq_df_eval = create_cuq_df(cuq_type, predictions_df, label_col_name, 'pred', 'predicted_class', 'MASKED_IMG_ID', alpha, 0.2, True, num_mc_cols, num_classes)
    mc_df = create_mc_df(cuq_type, predictions_df, cuq_df_eval, label_col_name, alpha, num_mc_cols, num_classes, save_dir)