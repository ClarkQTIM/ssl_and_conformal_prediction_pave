"""

This module provides numerical utilities to support uncertainty quantification 
in Monte Carlo (MC) dropout inference pipelines.

Currently Included:
- `calculate_coeff_of_var`: Computes the coefficient of variation (CoV) from MC dropout 
  predictions, serving as a proxy for model uncertainty. The CoV is calculated from 
  the expected values (EVs) of the class probabilities across multiple MC runs.

This function is especially useful when paired with conformal prediction techniques 
to evaluate the spread of prediction confidence across classes.

Dependencies:
- numpy
- pandas
- ast
"""


#########
# Imports
#########

import numpy as np
import pandas as pd
import ast

###########
# Functions
###########

def calculate_accuracy_and_count_by_predicted_class(df, pred_value, pred_class_col, pred_set_length_col, pred_set_length_value):

    """
        Calculate the accuracy and count of samples for a specific predicted class and prediction set length.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing prediction results, including predicted class, ground truth (`gt`), 
            and prediction set length.

        pred_value : int
            The predicted class value to filter on (e.g., 0, 1, or 2).

        pred_class_col : str
            Name of the column in `df` that contains the predicted class values.

        pred_set_length_col : str
            Name of the column in `df` that contains the prediction set length values.

        pred_set_length_value : int
            The prediction set length value to filter on (e.g., 1, 2, or 3).

        Returns:
        --------
        accuracy : float or np.nan
            The percentage of correct predictions for the filtered group, or NaN if no samples matched.

        count : int
            The number of samples in the filtered group.

        Notes:
        ------
        Accuracy is computed as the proportion of instances where the predicted class matches the ground truth (`gt`)
        within the specified subset. This function is useful for evaluating performance conditioned on both 
        model confidence (via prediction set length) and predicted class.
    """

    # Filter the dataframe based on the predicted class and prediction set length
    subset_df = df[(df[pred_class_col] == pred_value) & (df[pred_set_length_col] == pred_set_length_value)]
    
    if len(subset_df) == 0:
        return np.nan, 0  # Return NaN and 0 if there's no data for this combination

    # Calculate accuracy: Correct if the ground truth (df['gt']) matches the predicted class
    correct_predictions = (subset_df['gt'] == subset_df[pred_class_col]).sum()
    total_predictions = len(subset_df)
    
    accuracy = (correct_predictions / total_predictions) * 100

    return accuracy, total_predictions  # Return both accuracy and count


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

def calculate_accuracy_and_count_by_predicted_class(df, pred_value, pred_class_col, pred_set_length_col, pred_set_length_value):
    # Filter the dataframe based on the predicted class and prediction set length
    subset_df = df[(df[pred_class_col] == pred_value) & (df[pred_set_length_col] == pred_set_length_value)]
    
    if len(subset_df) == 0:
        return np.nan, 0  # Return NaN and 0 if there's no data for this combination

    # Calculate accuracy: Correct if the ground truth (df['gt']) matches the predicted class
    correct_predictions = (subset_df['gt'] == subset_df[pred_class_col]).sum()
    total_predictions = len(subset_df)
    
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy, total_predictions  # Return both accuracy and count