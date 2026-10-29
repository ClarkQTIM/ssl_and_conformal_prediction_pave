import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
from sklearn.utils import resample

from calculation_functions import calculate_accuracy_and_count_by_predicted_class
from plotting_functions import generate_accuracy_grid_by_predicted_class, plot_and_save_grid_by_predicted_class, create_confusion_matrix_analysis, distribution_of_cov_color_coded_by_set_length, distribution_of_std_dev_color_coded_by_set_length
from plotting_functions import histogram_of_cov_color_coded_by_set_length, histogram_of_std_dev_color_coded_by_set_length, expected_value_of_prediction_vs_std_dev_by_correct_or_not, expected_value_of_prediction_vs_cov_by_correct_or_not, expected_value_of_prediction_vs_prediction_length_colored_by_correct_or_not
from plotting_functions import create_classification_plot_per_class, create_plot_std_dev_vs_pred_set_size, create_scatter_cov_vs_avg_mc_pred_lengths, create_plot_cov_vs_pred_set_size, create_histogram_of_pred_length_size
from calculation_functions import independent_ttest


##########
# Analysis
##########

def analyse_cuq_df_three_class(cuq_df, eval_or_mc, save_path, save_name):

    """
    Analyze the CUQ DataFrame and save various analysis results to CSV and PNG files.

    Parameters:
    - cuq_df: DataFrame containing the CUQ results.
    - eval_or_mc: String indicating whether the analysis is for 'eval' or 'mc'.
    - save_path: Directory to save the resulting files.
    - save_name: Base name for the saved files.

    Returns:
    - None: The function saves the resulting analysis as CSV and PNG files.
    """

    def calculate_avg_median_ci(data, confidence=0.95):
        """
        Calculate the average, median, and confidence interval for the given data.
        """
        mean = np.mean(data)
        median = np.median(data)
        sem = stats.sem(data)  # Standard error of the mean
        margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., len(data)-1)
        return mean, median, margin_of_error

    def calculate_misclassification_stats(cuq_df, misclass_col, pred_set_length_col, cov_col, std_col, confidence=0.95):
        """
        Calculate statistics (average, median, confidence interval) for misclassifications, 
        including prediction set length, coefficient of variation, and standard deviation.
        """
        misclass_df = cuq_df[cuq_df[misclass_col] == True]
        
        avg_pred_set_length, median_pred_set_length, ci_pred_set_length = calculate_avg_median_ci(misclass_df[pred_set_length_col])
        avg_cov_mc, _, ci_cov_mc = calculate_avg_median_ci(misclass_df[cov_col])
        avg_std_mc, _, ci_std_mc = calculate_avg_median_ci(misclass_df[std_col])
        
        return (avg_pred_set_length, median_pred_set_length, ci_pred_set_length, 
                avg_cov_mc, ci_cov_mc, avg_std_mc, ci_std_mc)

    if eval_or_mc == 'eval':
        pred_col = 'pred_class_eval'
        corr_col = 'correct_prediction_eval'
        pred_set_length_col = 'pred_set_eval_length'
        extreme_misclassification_col = 'extreme_misclassification_eval'
        soft_col = 'soft_eval_prediction'
        normal_misclassified_as_pcplus_col = 'normal_misclassified_as_pcplus_eval'
        pcplus_misclassified_as_normal_col = 'pcplus_misclassified_as_normal_eval'
        normal_misclassified_as_gz_col = 'normal_misclassified_as_gz_eval'
        gz_misclassified_as_normal_col = 'gz_misclassified_as_normal_eval'
        pcplus_misclassified_as_gz_col = 'pcplus_misclassified_as_gz_eval'
        gz_misclassified_as_pcplus_col = 'gz_misclassified_as_pcplus_eval'
        covered_by_pred_set_col = 'covered_by_pred_set_eval'
        cov_col = 'cov_eval'
        std_col = 'std_avg_soft_mc_pred'

        # Per-class correct/incorrect pie charts
        for class_id in range(3):
            create_classification_plot_per_class(cuq_df, class_id, 'gt', eval_or_mc, 'Eval', save_path)
            create_classification_plot_per_class(cuq_df, class_id, 'pred', eval_or_mc, 'Eval', save_path)
        
    elif eval_or_mc == 'mc':
        pred_col = 'pred_class_mc'
        corr_col = 'correct_prediction_mc'
        pred_set_length_col = 'pred_set_mc_length'
        extreme_misclassification_col = 'extreme_misclassification_mc'
        soft_col = 'soft_mc_prediction'
        normal_misclassified_as_pcplus_col = 'normal_misclassified_as_pcplus_mc'
        pcplus_misclassified_as_normal_col = 'pcplus_misclassified_as_normal_mc'
        normal_misclassified_as_gz_col = 'normal_misclassified_as_gz_mc'
        gz_misclassified_as_normal_col = 'gz_misclassified_as_normal_mc'
        pcplus_misclassified_as_gz_col = 'pcplus_misclassified_as_gz_mc'
        gz_misclassified_as_pcplus_col = 'gz_misclassified_as_pcplus_mc'
        covered_by_pred_set_col = 'covered_by_pred_set_mc'
        cov_col = 'cov_mc'
        std_col = 'std_avg_soft_mc_pred'

        for class_id in range(3):
            create_classification_plot_per_class(cuq_df, class_id, 'gt', eval_or_mc, 'MC', save_path)
            create_classification_plot_per_class(cuq_df, class_id, 'pred', eval_or_mc, 'MC', save_path)

    generate_accuracy_grid_by_predicted_class(cuq_df, pred_col, pred_set_length_col, save_path)

    # Average prediction set sizes
    avg_pred_set_length_whole_df, median_pred_set_length_whole_df, ci_avg_pred_set_length_whole_df = calculate_avg_median_ci(cuq_df[pred_set_length_col])
    avg_pred_set_length_normal_only, median_pred_set_length_normal_only, ci_avg_pred_set_length_normal_only = calculate_avg_median_ci(cuq_df[cuq_df['gt'] == 0][pred_set_length_col])
    avg_pred_set_length_gz_only, median_pred_set_length_gz_only, ci_avg_pred_set_length_gz_only = calculate_avg_median_ci(cuq_df[cuq_df['gt'] == 1][pred_set_length_col])
    avg_pred_set_length_pcplus_only, median_pred_set_length_pcplus_only, ci_avg_pred_set_length_pcplus_only = calculate_avg_median_ci(cuq_df[cuq_df['gt'] == 2][pred_set_length_col])

    correct_classification_subset = cuq_df[cuq_df[corr_col] == True]
    misclassification_overall_subset = cuq_df[cuq_df[corr_col] == False]

    # Misclassification subsets
    misclassification_not_extreme_subset = misclassification_overall_subset[misclassification_overall_subset[extreme_misclassification_col] == False]
    misclassification_extreme_subset = misclassification_overall_subset[misclassification_overall_subset[extreme_misclassification_col] == True]

    # Misclassification percentages
    pct_misclassifications = (len(misclassification_overall_subset)/len(cuq_df))
    pct_misclassifications_not_extreme = (len(misclassification_not_extreme_subset)/len(cuq_df))
    pct_misclassifications_extreme = (len(misclassification_extreme_subset)/len(cuq_df))

    # Extreme misclassifications by class with confidence intervals
    (avg_pred_set_length_ext_miscl_normal_to_pcplus, median_pred_set_length_ext_miscl_normal_to_pcplus, ci_pred_set_length_ext_miscl_normal_to_pcplus, 
     cov_mc_for_normal_to_pcplus, ci_cov_mc_for_normal_to_pcplus, std_mc_for_normal_to_pcplus, ci_std_mc_for_normal_to_pcplus) = calculate_misclassification_stats(cuq_df, normal_misclassified_as_pcplus_col, pred_set_length_col, cov_col, std_col)

    (avg_pred_set_length_ext_miscl_pcplus_to_normal, median_pred_set_length_ext_miscl_pcplus_to_normal, ci_pred_set_length_ext_miscl_pcplus_to_normal, 
     cov_mc_for_pcplus_to_normal, ci_cov_mc_for_pcplus_to_normal, std_mc_for_pcplus_to_normal, ci_std_mc_for_pcplus_to_normal) = calculate_misclassification_stats(cuq_df, pcplus_misclassified_as_normal_col, pred_set_length_col, cov_col, std_col)

    (avg_pred_set_length_miscl_normal_to_gz, median_pred_set_length_miscl_normal_to_gz, ci_pred_set_length_miscl_normal_to_gz, 
     cov_mc_for_normal_to_gz, ci_cov_mc_for_normal_to_gz, std_mc_for_normal_to_gz, ci_std_mc_for_normal_to_gz) = calculate_misclassification_stats(cuq_df, normal_misclassified_as_gz_col, pred_set_length_col, cov_col, std_col)

    (avg_pred_set_length_miscl_gz_to_normal, median_pred_set_length_miscl_gz_to_normal, ci_pred_set_length_miscl_gz_to_normal, 
     cov_mc_for_gz_to_normal, ci_cov_mc_for_gz_to_normal, std_mc_for_gz_to_normal, ci_std_mc_for_gz_to_normal) = calculate_misclassification_stats(cuq_df, gz_misclassified_as_normal_col, pred_set_length_col, cov_col, std_col)

    (avg_pred_set_length_miscl_pcplus_to_gz, median_pred_set_length_miscl_pcplus_to_gz, ci_pred_set_length_miscl_pcplus_to_gz, 
     cov_mc_for_pcplus_to_gz, ci_cov_mc_for_pcplus_to_gz, std_mc_for_pcplus_to_gz, ci_std_mc_for_pcplus_to_gz) = calculate_misclassification_stats(cuq_df, pcplus_misclassified_as_gz_col, pred_set_length_col, cov_col, std_col)

    (avg_pred_set_length_miscl_gz_to_pcplus, median_pred_set_length_miscl_gz_to_pcplus, ci_pred_set_length_miscl_gz_to_pcplus, 
     cov_mc_for_gz_to_pcplus, ci_cov_mc_for_gz_to_pcplus, std_mc_for_gz_to_pcplus, ci_std_mc_for_gz_to_pcplus) = calculate_misclassification_stats(cuq_df, gz_misclassified_as_pcplus_col, pred_set_length_col, cov_col, std_col)

    ## Hypothesis 1: Expected value of prediction versus prediction length color-coded by correct or not
    expected_value_of_prediction_vs_prediction_length_colored_by_correct_or_not(cuq_df, soft_col, pred_set_length_col, corr_col, save_path, 'ev_preds_vs_pred_length')

    ## Hypothesis 2: Plot the coefficient of variations on the y-axis and the expected value on the x
    expected_value_of_prediction_vs_cov_by_correct_or_not(cuq_df, soft_col, cov_col, corr_col, save_path, 'ev_preds_vs_cov')
    expected_value_of_prediction_vs_std_dev_by_correct_or_not(cuq_df, soft_col, std_col, corr_col, save_path, 'ev_preds_vs_std_devs')

    ## Hypothesis 2: Plot histogram and distribution of the  color-coded by prediction set length
    histogram_of_cov_color_coded_by_set_length(cuq_df, cov_col, pred_set_length_col, save_path, 'cov_hist')
    distribution_of_cov_color_coded_by_set_length(cuq_df, cov_col, pred_set_length_col, save_path, 'cov_dist')
    histogram_of_std_dev_color_coded_by_set_length(cuq_df, std_col, pred_set_length_col, save_path, 'std_dev_hist')
    distribution_of_std_dev_color_coded_by_set_length(cuq_df, std_col, pred_set_length_col, save_path, 'std_dev_dist')

    ## Hypothesis 2: Confusion matrix with average standard deviations
    create_confusion_matrix_analysis(cuq_df, 'gt', pred_col, cov_col, 'Average Coefficient of Variation', save_path, 'gt_vs_pred_class_w_avg_cov')

    ## Hypothesis 3: Confusion matrix with average prediction set length
    create_confusion_matrix_analysis(cuq_df, 'gt', pred_col, pred_set_length_col, 'Average Prediction Set Length', save_path, 'gt_vs_pred_class_w_avg_pred_set_length')
    

    ## Percent Coverage
    pct_coverage = np.average(list(cuq_df[covered_by_pred_set_col]))

    ## Correlation and box plotting between Coeff of Var and Pred Set Size
    create_plot_cov_vs_pred_set_size(cuq_df, eval_or_mc, 'Pred Size Vs CoV', save_path)
    create_scatter_cov_vs_avg_mc_pred_lengths(cuq_df, eval_or_mc, 'Pred Size vs Avg MC Pred Length', save_path)
    create_plot_std_dev_vs_pred_set_size(cuq_df, eval_or_mc, 'Pred Size Vs Std Dev Eval', save_path)

    metrics_dict_eval = {
    'avg_pred_set_length_whole_df': avg_pred_set_length_whole_df,
    'median_pred_set_length_whole_df': median_pred_set_length_whole_df,
    'ci_avg_pred_set_length_whole_df': ci_avg_pred_set_length_whole_df,

    'avg_pred_set_length_normal_only': avg_pred_set_length_normal_only,
    'median_pred_set_length_normal_only': median_pred_set_length_normal_only,
    'ci_avg_pred_set_length_normal_only': ci_avg_pred_set_length_normal_only,

    'avg_pred_set_length_gz_only': avg_pred_set_length_gz_only,
    'median_pred_set_length_hz_only': median_pred_set_length_gz_only,
    'ci_avg_pred_set_length_gz_only': ci_avg_pred_set_length_gz_only,

    'avg_pred_set_length_pcplus_only': avg_pred_set_length_pcplus_only,
    'median_pred_set_length_pcplus_only': median_pred_set_length_pcplus_only,
    'ci_avg_pred_set_length_pcplus_only': ci_avg_pred_set_length_pcplus_only,

    'pct_coverage': pct_coverage,

    'avg_pred_set_length_ext_miscl_normal_to_pcplus': avg_pred_set_length_ext_miscl_normal_to_pcplus,
    'median_pred_set_length_ext_miscl_normal_to_pcplus': median_pred_set_length_ext_miscl_normal_to_pcplus,
    'ci_pred_set_length_ext_miscl_normal_to_pcplus': ci_pred_set_length_ext_miscl_normal_to_pcplus,
    'cov_mc_for_normal_to_pcplus': cov_mc_for_normal_to_pcplus,
    'ci_cov_mc_for_normal_to_pcplus': ci_cov_mc_for_normal_to_pcplus,
    'std_mc_for_normal_to_pcplus': std_mc_for_normal_to_pcplus,
    'ci_std_mc_for_normal_to_pcplus': ci_std_mc_for_normal_to_pcplus,

    'avg_pred_set_length_ext_miscl_pcplus_to_normal': avg_pred_set_length_ext_miscl_pcplus_to_normal,
    'median_pred_set_length_ext_miscl_pcplus_to_normal': median_pred_set_length_ext_miscl_pcplus_to_normal,
    'ci_pred_set_length_ext_miscl_pcplus_to_normal': ci_pred_set_length_ext_miscl_pcplus_to_normal,
    'cov_mc_for_pcplus_to_normal': cov_mc_for_pcplus_to_normal,
    'ci_cov_mc_for_pcplus_to_normal': ci_cov_mc_for_pcplus_to_normal,
    'std_mc_for_pcplus_to_normal': std_mc_for_pcplus_to_normal,
    'ci_std_mc_for_pcplus_to_normal': ci_std_mc_for_pcplus_to_normal,

    'avg_pred_set_length_miscl_normal_to_gz': avg_pred_set_length_miscl_normal_to_gz,
    'median_pred_set_length_miscl_normal_to_gz': median_pred_set_length_miscl_normal_to_gz,
    'ci_pred_set_length_miscl_normal_to_gz': ci_pred_set_length_miscl_normal_to_gz,
    'cov_mc_for_normal_to_gz': cov_mc_for_normal_to_gz,
    'ci_cov_mc_for_normal_to_gz': ci_cov_mc_for_normal_to_gz,
    'std_mc_for_normal_to_gz': std_mc_for_normal_to_gz,
    'ci_std_mc_for_normal_to_gz': ci_std_mc_for_normal_to_gz,

    'avg_pred_set_length_miscl_gz_to_normal': avg_pred_set_length_miscl_gz_to_normal,
    'median_pred_set_length_miscl_gz_to_normal': median_pred_set_length_miscl_gz_to_normal,
    'ci_pred_set_length_miscl_gz_to_normal': ci_pred_set_length_miscl_gz_to_normal,
    'cov_mc_for_gz_to_normal': cov_mc_for_gz_to_normal,
    'ci_cov_mc_for_gz_to_normal': ci_cov_mc_for_gz_to_normal,
    'std_mc_for_gz_to_normal': std_mc_for_gz_to_normal,
    'ci_std_mc_for_gz_to_normal': ci_std_mc_for_gz_to_normal,

    'avg_pred_set_length_miscl_pcplus_to_gz': avg_pred_set_length_miscl_pcplus_to_gz,
    'median_pred_set_length_miscl_pcplus_to_gz': median_pred_set_length_miscl_pcplus_to_gz,
    'ci_pred_set_length_miscl_pcplus_to_gz': ci_pred_set_length_miscl_pcplus_to_gz,
    'cov_mc_for_pcplus_to_gz': cov_mc_for_pcplus_to_gz,
    'ci_cov_mc_for_pcplus_to_gz': ci_cov_mc_for_pcplus_to_gz,
    'std_mc_for_pcplus_to_gz': std_mc_for_pcplus_to_gz,
    'ci_std_mc_for_pcplus_to_gz': ci_std_mc_for_pcplus_to_gz,

    'avg_pred_set_length_miscl_gz_to_pcplus': avg_pred_set_length_miscl_gz_to_pcplus,
    'median_pred_set_length_miscl_gz_to_pcplus': median_pred_set_length_miscl_gz_to_pcplus,
    'ci_pred_set_length_miscl_gz_to_pcplus': ci_pred_set_length_miscl_gz_to_pcplus,
    'cov_mc_for_gz_to_pcplus': cov_mc_for_gz_to_pcplus,
    'ci_cov_mc_for_gz_to_pcplus': ci_cov_mc_for_gz_to_pcplus,
    'std_mc_for_gz_to_pcplus': std_mc_for_gz_to_pcplus,
    'ci_std_mc_for_gz_to_pcplus': ci_std_mc_for_gz_to_pcplus,
}

    df = pd.DataFrame([metrics_dict_eval])
    df.to_csv(os.path.join(save_path, 'analysis_' + save_name))


def analyse_cuq_df_two_class(cuq_df, eval_or_mc, save_path, save_name):
    """
    Analyze the CUQ DataFrame and save various analysis results to CSV and PNG files.

    Parameters:
    - cuq_df: DataFrame containing the CUQ results.
    - eval_or_mc: String indicating whether the analysis is for 'eval' or 'mc'.
    - save_path: Directory to save the resulting files.
    - save_name: Base name for the saved files.

    Returns:
    - None: The function saves the resulting analysis as CSV and PNG files.
    """

    if eval_or_mc == 'eval':
        pred_col = 'pred_class_eval'
        corr_col = 'correct_prediction_eval'
        pred_set_length_col = 'pred_set_eval_length'
        extreme_misclassification_col = 'extreme_misclassification_eval'
        soft_col = 'soft_eval_prediction'
        normal_misclassified_as_pcplus_col = 'normal_misclassified_as_pcplus_eval'
        pcplus_misclassified_as_normal_col = 'pcplus_misclassified_as_normal_eval'
        normal_misclassified_as_gz_col = 'normal_misclassified_as_gz_eval'
        gz_misclassified_as_normal_col = 'gz_misclassified_as_normal_eval'
        pcplus_misclassified_as_gz_col = 'pcplus_misclassified_as_gz_eval'
        gz_misclassified_as_pcplus_col = 'gz_misclassified_as_pcplus_eval'
        covered_by_pred_set_col = 'covered_by_pred_set_eval'
        cov_col = 'cov_eval'
        std_col = 'std_avg_soft_mc_pred'

        # Per-class correct/incorrect pie charts
        for class_id in range(3):
            create_classification_plot_per_class(cuq_df, class_id, 'gt', eval_or_mc, 'Eval', save_path)
            create_classification_plot_per_class(cuq_df, class_id, 'pred', eval_or_mc, 'Eval', save_path)
        
    elif eval_or_mc == 'mc':
        pred_col = 'pred_class_mc'
        corr_col = 'correct_prediction_mc'
        pred_set_length_col = 'pred_set_mc_length'
        extreme_misclassification_col = 'extreme_misclassification_mc'
        soft_col = 'soft_mc_prediction'
        normal_misclassified_as_pcplus_col = 'normal_misclassified_as_pcplus_mc'
        pcplus_misclassified_as_normal_col = 'pcplus_misclassified_as_normal_mc'
        normal_misclassified_as_gz_col = 'normal_misclassified_as_gz_mc'
        gz_misclassified_as_normal_col = 'gz_misclassified_as_normal_mc'
        pcplus_misclassified_as_gz_col = 'pcplus_misclassified_as_gz_mc'
        gz_misclassified_as_pcplus_col = 'gz_misclassified_as_pcplus_mc'
        covered_by_pred_set_col = 'covered_by_pred_set_mc'
        cov_col = 'cov_mc'
        std_col = 'std_avg_soft_mc_pred'

        for class_id in range(3):
            create_classification_plot_per_class(cuq_df, class_id, 'gt', eval_or_mc, 'MC', save_path)
            create_classification_plot_per_class(cuq_df, class_id, 'pred', eval_or_mc, 'MC', save_path)

    # Average prediction set sizes
    correct_classification_subset = cuq_df[cuq_df[corr_col] == True]
    avg_pred_set_length_whole_df = np.average(list(cuq_df[pred_set_length_col]))
    median_pred_set_length_whole_df = np.median(list(cuq_df[pred_set_length_col]))

    ## Misclassifications
    misclassification_overall_subset = cuq_df[cuq_df[corr_col] == False]
    # create_histogram_of_pred_length_size(misclassification_overall_subset, 'eval', 'Misclassifications Eval', save_path)
    pct_misclassifications = (len(misclassification_overall_subset)/len(cuq_df))
    
    misclassification_not_extreme_subset = misclassification_overall_subset[misclassification_overall_subset[extreme_misclassification_col] == False]
    # create_histogram_of_pred_length_size(misclassification_not_extreme_subset, 'eval', 'Misclassifications Not Extreme Eval', save_path)
    pct_misclassifications_not_extreme = (len(misclassification_not_extreme_subset)/len(cuq_df))
    
    misclassification_extreme_subset = misclassification_overall_subset[misclassification_overall_subset[extreme_misclassification_col] == True]
    # create_histogram_of_pred_length_size(misclassification_extreme_subset, eval_or_mc, 'Misclassifications Extreme', save_path)
    pct_misclassifications_extreme = (len(misclassification_extreme_subset)/len(cuq_df))

    ## Hypothesis 1: Statistical Analysis of Correct vs Incorrect
    names = ['Correct vs Incorrect', 'Correct vs Single-Class Misclass', 'Correct vs Extreme Misclass', 'Single-Class vs Extreme Misclass']
    data1_means, data2_means, t_statistics, p_values = [], [], [], []
    ci_lowers1, ci_uppers1, ci_lowers2, ci_uppers2 = [], [], [], []

    comparisons = [
        (correct_classification_subset[pred_set_length_col], misclassification_overall_subset[pred_set_length_col]),
        (correct_classification_subset[pred_set_length_col], misclassification_not_extreme_subset[pred_set_length_col]),
        (correct_classification_subset[pred_set_length_col], misclassification_extreme_subset[pred_set_length_col]),
        (misclassification_not_extreme_subset[pred_set_length_col], misclassification_extreme_subset[pred_set_length_col])
    ]

    for data1, data2 in comparisons:
        t_statistic, p_value, data1_mean, data2_mean, ci_lower1, ci_upper1, ci_lower2, ci_upper2 = independent_ttest(data1, data2)
        t_statistics.append(t_statistic)
        p_values.append(p_value)
        data1_means.append(data1_mean)
        data2_means.append(data2_mean)
        ci_lowers1.append(ci_lower1)
        ci_uppers1.append(ci_upper1)
        ci_lowers2.append(ci_lower2)
        ci_uppers2.append(ci_upper2)

    misclassification_t_test_dict = {
        'Type': names,
        'Data1 Mean': data1_means,
        'Data2 Mean': data2_means,
        'ci_lower1': ci_lowers1,
        'ci_upper1': ci_uppers1,
        'ci_lower2': ci_lowers2,
        'ci_upper2': ci_uppers2,
        't_statistic': t_statistics,
        'p_value': p_values
    }

    misclassification_df = pd.DataFrame(misclassification_t_test_dict)
    misclassification_df.to_csv(os.path.join(save_path, 'misclassification_t_test_' + save_name))

    ## Hypothesis 1: Expected value of prediction versus prediction length color-coded by correct or not
    expected_value_of_prediction_vs_prediction_length_colored_by_correct_or_not(cuq_df, soft_col, pred_set_length_col, corr_col, save_path, 'ev_preds_vs_pred_length')

    ## Hypothesis 2: Plot the coefficient of variations on the y-axis and the expected value on the x
    expected_value_of_prediction_vs_cov_by_correct_or_not(cuq_df, soft_col, cov_col, corr_col, save_path, 'ev_preds_vs_cov')
    expected_value_of_prediction_vs_std_dev_by_correct_or_not(cuq_df, soft_col, std_col, corr_col, save_path, 'ev_preds_vs_std_devs')

    ## Hypothesis 2: Plot histogram and distribution of the color-coded by prediction set length
    histogram_of_cov_color_coded_by_set_length(cuq_df, cov_col, pred_set_length_col, save_path, 'cov_hist')
    distribution_of_cov_color_coded_by_set_length(cuq_df, cov_col, pred_set_length_col, save_path, 'cov_dist')
    histogram_of_std_dev_color_coded_by_set_length(cuq_df, std_col, pred_set_length_col, save_path, 'std_dev_hist')
    distribution_of_std_dev_color_coded_by_set_length(cuq_df, std_col, pred_set_length_col, save_path, 'std_dev_dist')

    ## Hypothesis 2: Confusion matrix with average standard deviations
    create_confusion_matrix_analysis(cuq_df, 'gt', pred_col, cov_col, 'Average Coefficient of Variation', save_path, 'gt_vs_pred_class_w_avg_cov')

    ## Hypothesis 3: Confusion matrix with average prediction set length
    create_confusion_matrix_analysis(cuq_df, 'gt', pred_col, pred_set_length_col, 'Average Prediction Set Length', save_path, 'gt_vs_pred_class_w_avg_pred_set_length')
    ### Per type of misclassification
    avg_pred_set_length_correct_only = np.average(list(correct_classification_subset[pred_set_length_col]))
    median_pred_set_length_correct_only = np.median(list(correct_classification_subset[pred_set_length_col]))
    avg_pred_set_length_all_misclassification = np.average(list(misclassification_overall_subset[pred_set_length_col]))
    median_pred_set_length_all_misclassification = np.median(list(misclassification_overall_subset[pred_set_length_col]))

    # Per-class analysis
    def calculate_class_stats(cuq_df, pred_set_length_col, class_id):
        class_df = cuq_df[cuq_df['gt'] == class_id]
        avg_pred_sets_length = np.average(class_df[pred_set_length_col])
        median_pred_sets_length = np.median(class_df[pred_set_length_col])
        return avg_pred_sets_length, median_pred_sets_length

    avg_pred_sets_length_normal_only, median_pred_sets_length_normal_only = calculate_class_stats(cuq_df, pred_set_length_col, 0)
    avg_pred_sets_length_pcplus_only, median_pred_sets_length_pcplus_only = calculate_class_stats(cuq_df, pred_set_length_col, 1)

    # Hypothesis 4: T-test to see if the average prediction length is different for each class overall
    names = ['0 vs 1']
    data1_means, data2_means, t_statistics, p_values = [], [], [], []
    ci_lowers1, ci_uppers1, ci_lowers2, ci_uppers2 = [], [], [], []

    class_comparisons = [
        (cuq_df[cuq_df['gt'] == 0][pred_set_length_col], cuq_df[cuq_df['gt'] == 1][pred_set_length_col])
    ]

    for data1, data2 in class_comparisons:
        t_statistic, p_value, data1_mean, data2_mean, ci_lower1, ci_upper1, ci_lower2, ci_upper2 = independent_ttest(data1, data2)
        t_statistics.append(t_statistic)
        p_values.append(p_value)
        data1_means.append(data1_mean)
        data2_means.append(data2_mean)
        ci_lowers1.append(ci_lower1)
        ci_uppers1.append(ci_upper1)
        ci_lowers2.append(ci_lower2)
        ci_uppers2.append(ci_upper2)

    by_class_t_test_dict = {
        'Type': names,
        'Data1 Mean': data1_means,
        'Data2 Mean': data2_means,
        'ci_lower1': ci_lowers1,
        'ci_upper1': ci_uppers1,
        'ci_lower2': ci_lowers2,
        'ci_upper2': ci_uppers2,
        't_statistic': t_statistics,
        'p_value': p_values
    }

    by_class_t_test_df = pd.DataFrame(by_class_t_test_dict)
    by_class_t_test_df.to_csv(os.path.join(save_path, 'by_class_t_test_df_' + save_name))

    # CoV and Standard Dev by prediction set length
    cov_mc_for_length_1_pred_sets = np.average(cuq_df[cuq_df[pred_set_length_col] == 1][cov_col])
    cov_mc_for_length_2_pred_sets = np.average(cuq_df[cuq_df[pred_set_length_col] == 2][cov_col])
    std_mc_for_length_1_pred_sets = np.average(cuq_df[cuq_df[pred_set_length_col] == 1][std_col])
    std_mc_for_length_2_pred_sets = np.average(cuq_df[cuq_df[pred_set_length_col] == 2][std_col])

    ## Percent Coverage
    pct_coverage = np.average(list(cuq_df[covered_by_pred_set_col]))

    ## Correlation and box plotting between Coeff of Var and Pred Set Size
    create_plot_cov_vs_pred_set_size(cuq_df, eval_or_mc, 'Pred Size Vs CoV', save_path)
    create_scatter_cov_vs_avg_mc_pred_lengths(cuq_df, eval_or_mc, 'Pred Size vs Avg MC Pred Length', save_path)
    create_plot_std_dev_vs_pred_set_size(cuq_df, eval_or_mc, 'Pred Size Vs Std Dev Eval', save_path)

    metrics_dict_eval = {
        'avg_pred_set_length_whole_df': avg_pred_set_length_whole_df,
        'median_pred_set_length_whole_df': median_pred_set_length_whole_df,

        'avg_pred_set_length_correct_only': avg_pred_set_length_correct_only,
        'median_pred_set_length_correct_only': median_pred_set_length_correct_only,
        'avg_pred_set_length_all_misclassification': avg_pred_set_length_all_misclassification,
        'median_pred_set_length_all_misclassification': median_pred_set_length_all_misclassification,

        'pct_coverage': pct_coverage,
        
        'avg_pred_sets_length_normal_only': avg_pred_sets_length_normal_only,
        'median_pred_sets_length_normal_only': median_pred_sets_length_normal_only,
        'avg_pred_sets_length_pcplus_only': avg_pred_sets_length_pcplus_only,
        'median_pred_sets_length_pcplus_only': median_pred_sets_length_pcplus_only,

        'pct_misclassifications': pct_misclassifications,

        'cov_mc_for_length_1_pred_sets': cov_mc_for_length_1_pred_sets,
        'cov_mc_for_length_2_pred_sets': cov_mc_for_length_2_pred_sets,
        'std_mc_for_length_1_pred_sets': std_mc_for_length_1_pred_sets,
        'std_mc_for_length_2_pred_sets': std_mc_for_length_2_pred_sets
    }

    df = pd.DataFrame([metrics_dict_eval])
    df.to_csv(os.path.join(save_path, 'analysis_' + save_name))

if __name__ == '__main__':

    # Args
    # save_path = '/sddata/projects/Conformal_Uncertainty_Quantification/csvs/MICCAI_2024_avg_pred_set_bin/analysis/lar_0.1'
    save_path = '/sddata/projects/Conformal_Uncertainty_Quantification/MICCAI_UNSURE_Workshop/analysis/trouble_shooting'
    csv_path = '/sddata/projects/Conformal_Uncertainty_Quantification/MICCAI_UNSURE_Workshop/analysis/testing_cov/mc_lar_run_alpha0.1.csv'
    cuq_df = pd.read_csv(csv_path)

    # Check if save path exists, if not, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    analyse_cuq_df_two_class(cuq_df, 'mc', save_path, csv_path.split('/')[-1])