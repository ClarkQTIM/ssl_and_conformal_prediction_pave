import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def convert_strings_to_list(strings):
    """
    Converts a list of strings to a list of lists of floats.
    Example input: ['[0.1, 0.2, 0.7]', '[0.3, 0.5, 0.2]']
    Example output: [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]]
    """
    # Replace spaces with commas
    '''
    7/10/2024
    Clean this up! Probably has to do with mc_cervix_analysis_cov_std_dev.py and how you're creating this.
    '''
    strings = [string.replace(' ', ',') for string in strings]
    strings = [string.replace(',,,', ',') for string in strings]
    strings = [string.replace(',,', ',') for string in strings]
    return [eval(string) for string in strings]

def parse_confusion_matrix(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    diagonals = []

    # Extract diagonal elements
    for class_index in range(len(df)):
        row = df.iloc[class_index]
        gt_class = row.name
        pred_class_value = row[f'{gt_class}']
        avg_set_length = float(pred_class_value.split('(')[1].strip(')'))
        diagonals.append(avg_set_length)

    return diagonals

def compute_roc_auc_ovr(df, pred_probs_col, gt_col, plot_title, save_dir, file_name):
    true_label = np.array(df[gt_col])
    model_prediction = np.array(convert_strings_to_list(list(df[pred_probs_col])))

    fpr = {}
    tpr = {}
    auroc = {}

    n_class = 3

    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(true_label, model_prediction[:, i], pos_label=i)
        auroc[i] = roc_auc_score(true_label == i, model_prediction[:, i])

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    
    colors = ['purple', 'gold', 'pink']
    class_names = ['Gray Zone', 'Normal', 'Pre-Cancer Plus']
    for i in range(n_class):
        plt.plot(fpr[i], tpr[i], linestyle='--', color=colors[i], label=f'Class {class_names[i]} vs Rest (AUC = {auroc[i]:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='-', color='red', label='Random')
    plt.plot([0, 0, 1], [0, 1, 1], linestyle='-', color='black', label='Theoretical Best')

    if plot_title:
        plt.title(plot_title)
    else:
        plt.title('')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')

    fig.savefig(os.path.join(save_dir, file_name), bbox_inches='tight')
    plt.clf()

def filter_data(data, max_set_sizes, target_class=None):
    """
    Filters the data based on prediction set size criteria for each class.
    
    Args:
    - data (pd.DataFrame): The dataset containing 'ground_truth', 'prediction', and 'prediction_set_size'.
    - max_set_sizes (list of float): The maximum prediction set size for each class.
    - target_class (int, optional): The class to apply the filter to. If None, applies to all classes.
    
    Returns:
    - pd.DataFrame: The filtered dataset.
    - pd.DataFrame: The removed instances.
    """
    def should_keep(row):
        pred = row['pred_class_mc']
        set_size = row['pred_set_mc_length']
        max_size = max_set_sizes[pred]
        
        if target_class is not None and pred != target_class:
            return True  # Keep all instances not of the target class
        return set_size <= max_size if max_size != 3 else set_size <= 2

    filtered_data = data[data.apply(should_keep, axis=1)]
    removed_data = data[~data.apply(should_keep, axis=1)]
    
    return filtered_data, removed_data

def determine_proper_cutoffs(data):
    cutoff_0 = np.floor(np.average(data[data['pred_class_mc'] == 0]['pred_set_mc_length']))
    cutoff_1 = np.floor(np.average(data[data['pred_class_mc'] == 1]['pred_set_mc_length']))
    cutoff_2 = np.floor(np.average(data[data['pred_class_mc'] == 2]['pred_set_mc_length']))

    return cutoff_0, cutoff_1, cutoff_2

# if __name__ == "__main__":

#     # Example dataset
#     data = pd.read_csv('/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL/analysis/SSL/M36_Pipeline/M36_Norms/DINO_EffNetV2S/inference_test2/3/lac/alpha_010/mc_lac_run_alpha0.1.csv')

#     cutoff_0, cutoff_1, cutoff_2 = determine_proper_cutoffs(data)
#     print(cutoff_0, cutoff_1, cutoff_2)
#     # Define the maximum set sizes for each class
#     parsed_diagonals = parse_confusion_matrix(avg_pred_sizes_path)
#     max_set_sizes_diagonal = [np.floor(parsed_diagonals[1]), np.floor(parsed_diagonals[0]), np.floor(parsed_diagonals[2])]

#     # Filtering and ROC-AUC
#     filtered_data = filter_data(data, max_set_sizes_diagonal)
#     print(f'We have removed {len(data)-len(filtered_data)} data points of each class {np.unique(data["gt"], return_counts=True)[0]}')
#     compute_roc_auc_ovr(filtered_data, 'pred_probs_mc', 'gt', 'ROC-AUC with Uncertain Predictions Excluded', '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL/analysis/SSL/M36_Pipeline/M36_Norms/DINO_EffNetV2S/inference_test2/3/lac/alpha_010', 'floor_cutoff_roc_auc.png')

#     max_set_sizes = [2, 2, 2]
#     filtered_data = filter_data(data, max_set_sizes)
#     compute_roc_auc_ovr(filtered_data, 'pred_probs_mc', 'gt', 'ROC-AUC with Uncertain Predictions Excluded', '/sddata/projects/Conformal_Uncertainty_Quantification/Comparison_SL_SSL/analysis/SSL/M36_Pipeline/M36_Norms/DINO_EffNetV2S/inference_test2/3/lac/alpha_010', 'no_length_three_roc_auc.png')
