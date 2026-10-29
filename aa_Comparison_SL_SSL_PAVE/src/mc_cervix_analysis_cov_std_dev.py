#########
# Imports
#########

## Standard
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

## Stats Testing
from scipy import stats
from scipy.stats import spearmanr
from scipy.stats import pearsonr

###########
# Functions
###########

def independent_ttest(data1, data2, alpha=0.05):
    """
    Perform an independent samples t-test.

    Parameters:
    - data1: list or array-like, containing the data for the first group
    - data2: list or array-like, containing the data for the second group
    - alpha: significance level for the confidence interval

    Returns:
    - t_statistic: The calculated t-statistic
    - p_value: The two-tailed p-value
    - ci_lower1: Lower bound of the confidence interval for the mean of data1
    - ci_upper1: Upper bound of the confidence interval for the mean of data1
    - ci_lower2: Lower bound of the confidence interval for the mean of data2
    - ci_upper2: Upper bound of the confidence interval for the mean of data2
    """
    t_statistic, p_value = stats.ttest_ind(data1, data2)
    
    # Degrees of freedom
    dof = len(data1) + len(data2) - 2
    
    # Standard error of the mean for each group
    se1 = np.std(data1, ddof=1) / np.sqrt(len(data1))
    se2 = np.std(data2, ddof=1) / np.sqrt(len(data2))
    
    # Critical value
    t_critical = stats.t.ppf(1 - alpha / 2, dof)
    
    # Confidence intervals
    mean_diff = np.mean(data1) - np.mean(data2)
    ci_lower1 = np.mean(data1) - t_critical * se1
    ci_upper1 = np.mean(data1) + t_critical * se1
    ci_lower2 = np.mean(data2) - t_critical * se2
    ci_upper2 = np.mean(data2) + t_critical * se2

    return t_statistic, p_value, data1.mean(), data2.mean(), ci_lower1, ci_upper1, ci_lower2, ci_upper2

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
    plt.title('Normalized Histogram of Prediction Length for ' + title)

    # Add legend
    plt.legend(title='Ground Truth')

    save_path = os.path.join(save_dir, title + '_normalized_pred_length_100_pct.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to release memory

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

def create_scatter_cov_vs_avg_mc_pred_lengths(df, mc_or_eval, title, save_dir):
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
    scatterplot = sns.scatterplot(x='avg_mc_pred_length', y='cov_mc', data=df)

    # Calculate correlation coefficient
    pearson_corr, p_value = pearsonr(df['cov_mc'], df['avg_mc_pred_length'])

    # Add labels and title
    plt.xlabel('Average MC Pred Set Length')
    plt.ylabel('Coefficient of Variation')
    # plt.title(f'{title} (Spearman Correlation: {spearman_corr:.2f})')

    # Add correlation coefficient and p-value as a text box
    plt.text(0.95, 0.95, f'Pearson: {pearson_corr:.2f}\nP-value: {p_value:.3f}', 
            horizontalalignment='right', 
            verticalalignment='top', 
            transform=plt.gca().transAxes, 
            bbox=dict(facecolor='white', alpha=0.5))

    # Save plot
    save_path = os.path.join(save_dir, title + '_scatter.png')
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
    plt.title(f'{title} (Correlation Coefficient: {spearman_corr:.2f})')

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
    plt.title(title)
    plt.tight_layout()

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
    plt.title(title)
    plt.tight_layout()

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
    plt.title(title)
    plt.legend()

    # Save plot
    save_path = os.path.join(save_dir, title + '.png')
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
    plt.title(title)
    plt.legend()

    # Save plot
    save_path = os.path.join(save_dir, title + '.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to release memory

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
    plt.title(title)
    plt.legend()

    # Save plot
    save_path = os.path.join(save_dir, title + '_distribution_std_dev.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to release memory

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
    unique_lengths = df[pred_set_length_col].unique()
    colors = plt.cm.get_cmap('tab10').colors[:len(unique_lengths)]
    color_dict = {length: colors[i] for i, length in enumerate(unique_lengths)}

    # Plot KDE plot for each prediction set length
    for pred_length, group_df in df.groupby(pred_set_length_col):
        sns.kdeplot(data=group_df[cov_col], color=color_dict[pred_length], label=f'Conformal Pred Length = {pred_length}', fill=True)

    plt.xlabel('Coefficient of Variation (%)')
    plt.ylabel('Density')
    # plt.title(title)
    plt.legend()

    # Save plot
    save_path = os.path.join(save_dir, title + '_distribution_cov.png')
    plt.savefig(save_path)
    plt.close()  # Close the plot to release memory

def create_confusion_matrix(df, ground_truth_col, predicted_class_col, metric_to_include, metric_name, save_dir, title):
    """
    Create a confusion matrix with a specified metric included and save as both CSV and PNG.

    Parameters:
    - df: DataFrame containing the data.
    - ground_truth_col: Column name for ground truth labels.
    - predicted_class_col: Column name for predicted class labels.
    - metric_to_include: Column name for the metric to include in the confusion matrix (e.g., average set length).
    - metric_name: Name of the metric to include in the title.
    - save_dir: Directory to save the resulting files.
    - title: Title for the files.

    Returns:
    - None: The function saves the resulting confusion matrix as CSV and PNG files.
    """

    # Creating a copy of the dataframe
    df_copy = df.copy()

    # Mapping labels to their corresponding names
    label_mapping = {0: "Normal", 1: "Gray Zone", 2: "Pre-Cancer Plus"}
    df_copy[ground_truth_col] = df_copy[ground_truth_col].map(label_mapping)
    df_copy[predicted_class_col] = df_copy[predicted_class_col].map(label_mapping)

    # Specify the desired order of classes
    class_order = ["Normal", "Gray Zone", "Pre-Cancer Plus"]

    # Create a pivot table to calculate the average set length for each combination of ground truth and predicted class
    pivot_table = df_copy.pivot_table(index=ground_truth_col, columns=predicted_class_col, values=metric_to_include, aggfunc=np.mean)

    # Reorder columns and index to match the desired order
    pivot_table = pivot_table.reindex(index=class_order, columns=class_order)

    # Replace NaN values with a numeric placeholder (e.g., 0)
    pivot_table = pivot_table.fillna(0)

    # Create a confusion matrix with counts
    confusion_matrix = pd.crosstab(df_copy[ground_truth_col], df_copy[predicted_class_col], rownames=['Ground Truth'], colnames=['Predicted Class'], dropna=False)

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='Blues', cbar=False, annot_kws={"fontsize": 15}, linewidths=0.5, linecolor='grey')
    plt.title(f'{metric_name}')
    plt.xlabel('Predicted Class')
    plt.ylabel('Ground Truth')
    plt.tight_layout()

    # Insert average set length into each cell of the confusion matrix
    for row in confusion_matrix.index:
        for col in confusion_matrix.columns:
            avg_set_length = pivot_table.loc[row, col]
            if avg_set_length == 0:
                confusion_matrix.loc[row, col] = f'{confusion_matrix.loc[row, col]} (N/A)'
            else:
                confusion_matrix.loc[row, col] = f'{confusion_matrix.loc[row, col]} ({avg_set_length:.2f})'

    # Save confusion matrix to a CSV file with row and column labels
    save_path = os.path.join(save_dir, title + '.csv')
    confusion_matrix.to_csv(save_path)

    # Save confusion matrix plot
    save_path_plot = os.path.join(save_dir, title + '_confusion_matrix.png')
    plt.savefig(save_path_plot)
    plt.close()  # Close the plot to release memory

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

    # Eval set
    ## Average prediction set sizes
    correct_classification_subset = cuq_df[cuq_df[corr_col] == True]
    avg_pred_set_length_whole_df = np.average(list(cuq_df[pred_set_length_col]))
    median_pred_set_length_whole_df = np.median(list(cuq_df[pred_set_length_col]))

    # create_histogram_of_pred_length_size(correct_classification_subset, 'eval', 'Correct Classification Eval', save_path)

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

    ## Per-class correct vs incorrect prediction set lengths
    per_class_names = []
    per_class_data1_means, per_class_data2_means, per_class_t_statistics, per_class_p_values = [], [], [], []
    per_class_ci_lowers1, per_class_ci_uppers1, per_class_ci_lowers2, per_class_ci_uppers2 = [], [], [], []

    for gt_class in range(3):
        correct_subset = cuq_df[(cuq_df['gt'] == gt_class) & (cuq_df[corr_col] == True)]
        incorrect_subset = cuq_df[(cuq_df['gt'] == gt_class) & (cuq_df[corr_col] == False)]

        if not correct_subset.empty and not incorrect_subset.empty:
            data1 = correct_subset[pred_set_length_col]
            data2 = incorrect_subset[pred_set_length_col]
            t_statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False)
            data1_mean, data2_mean = data1.mean(), data2.mean()
            ci_lower1, ci_upper1 = stats.norm.interval(0.95, loc=data1_mean, scale=stats.sem(data1))
            ci_lower2, ci_upper2 = stats.norm.interval(0.95, loc=data2_mean, scale=stats.sem(data2))

            per_class_names.append(f'GT Class {gt_class}: Correct vs Incorrect')
            per_class_data1_means.append(data1_mean)
            per_class_data2_means.append(data2_mean)
            per_class_ci_lowers1.append(ci_lower1)
            per_class_ci_uppers1.append(ci_upper1)
            per_class_ci_lowers2.append(ci_lower2)
            per_class_ci_uppers2.append(ci_upper2)
            per_class_t_statistics.append(t_statistic)
            per_class_p_values.append(p_value)

    per_class_t_test_dict = {
        'Type': per_class_names,
        'Data1 Mean': per_class_data1_means,
        'Data2 Mean': per_class_data2_means,
        'ci_lower1': per_class_ci_lowers1,
        'ci_upper1': per_class_ci_uppers1,
        'ci_lower2': per_class_ci_lowers2,
        'ci_upper2': per_class_ci_uppers2,
        't_statistic': per_class_t_statistics,
        'p_value': per_class_p_values
    }

    per_class_df = pd.DataFrame(per_class_t_test_dict)
    per_class_df.to_csv(os.path.join(save_path, 'per_class_t_test_' + save_name))

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
    create_confusion_matrix(cuq_df, 'gt', pred_col, cov_col, 'Average Coefficient of Variation', save_path, 'gt_vs_pred_class_w_avg_cov')

    ## Hypothesis 3: Confusion matrix with average prediction set length
    create_confusion_matrix(cuq_df, 'gt', pred_col, pred_set_length_col, 'Average Prediction Set Length', save_path, 'gt_vs_pred_class_w_avg_pred_set_length')
    
    ### Per type of misclassification
    avg_pred_set_length_correct_only = np.average(list(correct_classification_subset[pred_set_length_col]))
    median_pred_set_length_correct_only = np.median(list(correct_classification_subset[pred_set_length_col]))
    avg_pred_set_length_all_misclassification = np.average(list(misclassification_overall_subset[pred_set_length_col]))
    median_pred_set_length_all_misclassification = np.median(list(misclassification_overall_subset[pred_set_length_col]))
    avg_pred_set_length_single_class_misclassification = np.average(list(misclassification_not_extreme_subset[pred_set_length_col]))
    median_pred_set_length_single_class_misclassification = np.median(list(misclassification_not_extreme_subset[pred_set_length_col]))
    avg_pred_set_length_extreme_class_misclassification = np.average(list(misclassification_extreme_subset[pred_set_length_col]))
    median_pred_set_length_extreme_class_misclassification = np.median(list(misclassification_extreme_subset[pred_set_length_col]))

    # Per-class analysis
    def calculate_class_stats(cuq_df, pred_set_length_col, class_id):
        class_df = cuq_df[cuq_df['gt'] == class_id]
        avg_pred_sets_length = np.average(class_df[pred_set_length_col])
        median_pred_sets_length = np.median(class_df[pred_set_length_col])
        return avg_pred_sets_length, median_pred_sets_length

    avg_pred_sets_length_normal_only, median_pred_sets_length_normal_only = calculate_class_stats(cuq_df, pred_set_length_col, 0)
    avg_pred_sets_length_gz_only, median_pred_sets_length_gz_only = calculate_class_stats(cuq_df, pred_set_length_col, 1)
    avg_pred_sets_length_pcplus_only, median_pred_sets_length_pcplus_only = calculate_class_stats(cuq_df, pred_set_length_col, 2)

    # Hypothesis 4: T-test to see if the average prediction length is different for each class overall
    names = ['0 vs 1', '0 vs 2', '1 vs 2']
    data1_means, data2_means, t_statistics, p_values = [], [], [], []
    ci_lowers1, ci_uppers1, ci_lowers2, ci_uppers2 = [], [], [], []

    class_comparisons = [
        (cuq_df[cuq_df['gt'] == 0][pred_set_length_col], cuq_df[cuq_df['gt'] == 1][pred_set_length_col]),
        (cuq_df[cuq_df['gt'] == 0][pred_set_length_col], cuq_df[cuq_df['gt'] == 2][pred_set_length_col]),
        (cuq_df[cuq_df['gt'] == 1][pred_set_length_col], cuq_df[cuq_df['gt'] == 2][pred_set_length_col])
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

    # Extreme misclassifications by class
    def calculate_misclassification_stats(cuq_df, misclass_col, pred_set_length_col, cov_col, std_col):
        misclass_df = cuq_df[cuq_df[misclass_col] == True]
        avg_pred_set_length = np.average(misclass_df[pred_set_length_col])
        median_pred_set_length = np.median(misclass_df[pred_set_length_col])
        avg_cov_mc = np.average(misclass_df[cov_col])
        avg_std_mc = np.average(misclass_df[std_col])
        return avg_pred_set_length, median_pred_set_length, avg_cov_mc, avg_std_mc

    avg_pred_set_length_ext_miscl_normal_to_pcplus, median_pred_set_length_ext_miscl_normal_to_pcplus, cov_mc_for_normal_to_pcplus, std_mc_for_normal_to_pcplus = calculate_misclassification_stats(cuq_df, normal_misclassified_as_pcplus_col, pred_set_length_col, cov_col, std_col)
    avg_pred_set_length_ext_miscl_pcplus_to_normal, median_pred_set_length_ext_miscl_pcplus_to_normal, cov_mc_for_pcplus_to_normal, std_mc_for_pcplus_to_normal = calculate_misclassification_stats(cuq_df, pcplus_misclassified_as_normal_col, pred_set_length_col, cov_col, std_col)

    avg_pred_set_length_miscl_normal_to_gz, median_pred_set_length_miscl_normal_to_gz, cov_mc_for_normal_to_gz, std_mc_for_normal_to_gz = calculate_misclassification_stats(cuq_df, normal_misclassified_as_gz_col, pred_set_length_col, cov_col, std_col)
    avg_pred_set_length_miscl_gz_to_normal, median_pred_set_length_miscl_gz_to_normal, cov_mc_for_gz_to_normal, std_mc_for_gz_to_normal = calculate_misclassification_stats(cuq_df, gz_misclassified_as_normal_col, pred_set_length_col, cov_col, std_col)
    avg_pred_set_length_miscl_pcplus_to_gz, median_pred_set_length_miscl_pcplus_to_gz, cov_mc_for_pcplus_to_gz, std_mc_for_pcplus_to_gz = calculate_misclassification_stats(cuq_df, pcplus_misclassified_as_gz_col, pred_set_length_col, cov_col, std_col)
    avg_pred_set_length_miscl_gz_to_pcplus, median_pred_set_length_miscl_gz_to_pcplus, cov_mc_for_gz_to_pcplus, std_mc_for_gz_to_pcplus = calculate_misclassification_stats(cuq_df, gz_misclassified_as_pcplus_col, pred_set_length_col, cov_col, std_col)

    # CoV and Standard Dev by prediction set length
    cov_mc_for_length_1_pred_sets = np.average(cuq_df[cuq_df[pred_set_length_col] == 1][cov_col])
    cov_mc_for_length_2_pred_sets = np.average(cuq_df[cuq_df[pred_set_length_col] == 2][cov_col])
    cov_mc_for_length_3_pred_sets = np.average(cuq_df[cuq_df[pred_set_length_col] == 3][cov_col])
    std_mc_for_length_1_pred_sets = np.average(cuq_df[cuq_df[pred_set_length_col] == 1][std_col])
    std_mc_for_length_2_pred_sets = np.average(cuq_df[cuq_df[pred_set_length_col] == 2][std_col])
    std_mc_for_length_3_pred_sets = np.average(cuq_df[cuq_df[pred_set_length_col] == 3][std_col])

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
        'avg_pred_set_length_single_class_misclassification': avg_pred_set_length_single_class_misclassification,
        'median_pred_set_length_single_class_misclassification': median_pred_set_length_single_class_misclassification,
        'avg_pred_set_length_extreme_class_misclassification': avg_pred_set_length_extreme_class_misclassification,
        'median_pred_set_length_extreme_class_misclassification': median_pred_set_length_extreme_class_misclassification,

        'pct_coverage': pct_coverage,
        
        'avg_pred_sets_length_normal_only': avg_pred_sets_length_normal_only,
        'median_pred_sets_length_normal_only': median_pred_sets_length_normal_only,
        'avg_pred_sets_length_gz_only': avg_pred_sets_length_gz_only,
        'median_pred_sets_length_gz_only': median_pred_sets_length_gz_only,
        'avg_pred_sets_length_pcplus_only': avg_pred_sets_length_pcplus_only,
        'median_pred_sets_length_pcplus_only': median_pred_sets_length_pcplus_only,

        'pct_misclassifications': pct_misclassifications,
        'pct_misclassifications_not_extreme': pct_misclassifications_not_extreme,
        'pct_misclassifications_extreme': pct_misclassifications_extreme,

        'avg_pred_set_length_ext_miscl_normal_to_pcplus': avg_pred_set_length_ext_miscl_normal_to_pcplus,
        'median_pred_set_length_ext_miscl_normal_to_pcplus': median_pred_set_length_ext_miscl_normal_to_pcplus,
        'avg_pred_set_length_ext_miscl_pcplus_to_normal': avg_pred_set_length_ext_miscl_pcplus_to_normal,
        'median_pred_set_length_ext_miscl_pcplus_to_normal': median_pred_set_length_ext_miscl_pcplus_to_normal,

        'avg_pred_set_length_miscl_normal_to_gz': avg_pred_set_length_miscl_normal_to_gz,
        'median_pred_set_length_miscl_normal_to_gz': median_pred_set_length_miscl_normal_to_gz,
        'avg_pred_set_length_miscl_gz_to_normal': avg_pred_set_length_miscl_gz_to_normal,
        'median_pred_set_length_miscl_gz_to_normal': median_pred_set_length_miscl_gz_to_normal,
        'avg_pred_set_length_miscl_pcplus_to_gz': avg_pred_set_length_miscl_pcplus_to_gz,
        'median_pred_set_length_miscl_pcplus_to_gz': median_pred_set_length_miscl_pcplus_to_gz,
        'avg_pred_set_length_miscl_gz_to_pcplus': avg_pred_set_length_miscl_gz_to_pcplus,
        'median_pred_set_length_miscl_gz_to_pcplus': median_pred_set_length_miscl_gz_to_pcplus,

        'cov_mc_for_normal_to_pcplus': cov_mc_for_normal_to_pcplus,
        'cov_mc_for_pcplus_to_normal': cov_mc_for_pcplus_to_normal,
        'cov_mc_for_normal_to_gz': cov_mc_for_normal_to_gz,
        'cov_mc_for_gz_to_normal': cov_mc_for_gz_to_normal,
        'cov_mc_for_pcplus_to_gz': cov_mc_for_pcplus_to_gz,
        'cov_mc_for_gz_to_pcplus': cov_mc_for_gz_to_pcplus,

        'std_mc_for_normal_to_pcplus': std_mc_for_normal_to_pcplus,
        'std_mc_for_pcplus_to_normal': std_mc_for_pcplus_to_normal,
        'std_mc_for_normal_to_gz': std_mc_for_normal_to_gz,
        'std_mc_for_gz_to_normal': std_mc_for_gz_to_normal,
        'std_mc_for_pcplus_to_gz': std_mc_for_pcplus_to_gz,
        'std_mc_for_gz_to_pcplus': std_mc_for_gz_to_pcplus,

        'cov_mc_for_length_1_pred_sets': cov_mc_for_length_1_pred_sets,
        'cov_mc_for_length_2_pred_sets': cov_mc_for_length_2_pred_sets,
        'cov_mc_for_length_3_pred_sets': cov_mc_for_length_3_pred_sets,
        'std_mc_for_length_1_pred_sets': std_mc_for_length_1_pred_sets,
        'std_mc_for_length_2_pred_sets': std_mc_for_length_2_pred_sets,
        'std_mc_for_length_3_pred_sets': std_mc_for_length_3_pred_sets,
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
    create_confusion_matrix(cuq_df, 'gt', pred_col, cov_col, 'Average Coefficient of Variation', save_path, 'gt_vs_pred_class_w_avg_cov')

    ## Hypothesis 3: Confusion matrix with average prediction set length
    create_confusion_matrix(cuq_df, 'gt', pred_col, pred_set_length_col, 'Average Prediction Set Length', save_path, 'gt_vs_pred_class_w_avg_pred_set_length')
    
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