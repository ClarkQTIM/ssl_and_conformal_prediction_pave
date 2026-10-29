"""
This module provides utility functions to support uncertainty-aware deep learning 
experiments, particularly in the context of Monte Carlo (MC) dropout inference 
and conformal prediction.

Key Functionalities:
- `convert_strings_to_list`: Converts stringified Python lists (e.g., "[0.1, 0.8, 0.1]") 
  into actual list objects.
- `additions_to_csvs`: Enriches a prediction CSV with uncertainty metrics derived from 
  MC dropout outputs, including expected value, standard deviation, coefficient of variation, 
  and per-class average probabilities.

These tools are designed to augment model prediction CSVs with interpretable and 
quantitative indicators of predictive uncertainty for further downstream analysis.

Dependencies:
- NumPy
- pandas
- ast
"""


#########
# Imports
#########

import ast
import numpy as np
import pandas as pd
import os

from calculation_functions import calculate_accuracy_and_count_by_predicted_class

###################
# Utility Functions
###################

def create_accuracy_and_count_grid_by_predicted_class(df, pred_class_col, pred_set_length_col):

    """
        Compute a 3x3 grid of (accuracy, count) tuples stratified by predicted class and prediction set length.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing prediction results with columns for predicted class and prediction set length.

        pred_class_col : str
            Name of the column in `df` that contains the predicted class values (assumed to be 0, 1, or 2).

        pred_set_length_col : str
            Name of the column in `df` that contains the length of the prediction set (assumed to be 1, 2, or 3).

        Returns:
        --------
        grid : numpy.ndarray (3x3, dtype=object)
            A 3x3 grid where each cell at [i, j] contains a tuple (accuracy, count):
                - i corresponds to prediction set lengths 1, 2, or 3 (row-wise).
                - j corresponds to predicted classes 0, 1, or 2 (column-wise).
            Accuracy is the proportion of correct predictions among rows matching that combination.
            Count is the number of samples in that group.

        Notes:
        ------
        This function allows detailed inspection of model performance across both prediction set size
        (e.g., 1-label confident vs. 3-label uncertain) and predicted class identity.
        Useful for analyzing trade-offs between confidence and classification accuracy.
    """

    # Initialize a 3x3 grid to store accuracy values and another for counts
    grid = np.zeros((3, 3), dtype=object)  # Store (accuracy, count) tuples
    
    # Iterate over predicted classes (0, 1, 2) and pred_set_mc_length (1, 2, 3)
    for pred_value in range(3):  # Predicted class values
        for pred_set_length_value in range(1, 4):  # pred_set_mc_length values (1, 2, 3)
            accuracy, count = calculate_accuracy_and_count_by_predicted_class(df, pred_value, pred_class_col, pred_set_length_col, pred_set_length_value)
            grid[pred_set_length_value - 1, pred_value] = (accuracy, count)  # Store both accuracy and count in the grid, flipped axes
    
    return grid

def save_removed_data(removed_data, title, save_dir, save_name):

    """
        Save removed instances DataFrame to CSV if not empty.

        Parameters:
        -----------
        removed_data : pandas.DataFrame
            The DataFrame containing removed instances to be saved.

        title : str
            A descriptor used for printing status messages (e.g., the name of the model or experiment).

        save_dir : str
            Directory where the CSV file should be saved.

        save_name : str
            Filename (without extension) to use when saving the CSV.

        Outputs:
        --------
        - Saves `removed_data` to a CSV at the specified location if the DataFrame is not empty.
        - Prints confirmation message with the save path, or a message if no data was saved.
    """

    if removed_data.empty:
        print(f"No instances removed for {title}")
        return

    save_path = os.path.join(save_dir, f"{save_name}.csv")
    removed_data.to_csv(save_path, index=False)
    print(f"Removed data saved to {save_path}")

def convert_strings_to_list(list_of_strings):

    '''
    Converts a string into a list.
    '''
    return [ast.literal_eval(string) for string in list_of_strings]

def additions_to_csvs(df):
    """
    Adds several uncertainty-related metrics to a dataframe containing 
    Monte Carlo (MC) dropout predictions.

    This function computes the following for each row:
    - Expected value of predicted class across all MC runs.
    - Standard deviation and coefficient of variation of the expected values.
    - Rounded expected class (as a pseudo hard prediction).
    - Per-class average probabilities across all MC runs.
    - Argmax of the averaged per-class probabilities (final prediction).

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing MC prediction columns with probability vectors 
        stored as strings. These columns must be labeled with "mc_" in their name 
        but not include 'pred_mc', 'predicted_mc_class', etc.

    Returns:
    --------
    df : pandas.DataFrame
        The input DataFrame with the following additional columns:
        - 'avg_soft_mc_pred': Expected class value averaged over all MC runs.
        - 'rounded_avg_soft_mc_pred': Rounded version of avg_soft_mc_pred.
        - 'std_avg_soft_mc_pred': Standard deviation of expected values.
        - 'cv_avg_soft_mc_pred': Coefficient of variation (%).
        - 'avg_mc_probs_per_class': Mean probability for each class over MC runs.
        - 'argmax_mc_probs_per_class': Final predicted class from averaged probabilities.

    Notes:
    ------
    - This function assumes MC prediction columns contain stringified lists 
      (e.g., '[0.1, 0.8, 0.1]').
    - It handles both binary and ternary classification setups.
    - The newly inserted columns appear near the end of the DataFrame.
    """

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

def parse_confusion_matrix(csv_path):

    """
        Parses a saved confusion matrix CSV with annotated entries (e.g., "12 (1.80)") and extracts the diagonal values.

        Assumes each cell contains both a count and an average prediction set size in the format: "count (avg)".

        Parameters:
        -----------
        csv_path : str
            Path to the CSV file containing the annotated confusion matrix.

        Returns:
        --------
        diagonals : list of float
            List of average prediction set sizes from the diagonal (i.e., correct predictions) of the confusion matrix.
    """

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


def filter_data(data, max_set_sizes, target_class=None):
        
    """
        Filters instances from the dataset based on prediction set size thresholds for each predicted class.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing model predictions, predicted class labels, and prediction set lengths.

        max_set_sizes : list of float
            A list specifying the maximum allowed prediction set size for each class (by predicted class index).

        target_class : int, optional
            If specified, filtering is applied only to this class. All other classes are retained without filtering.
            If None, filtering is applied to all predicted classes.

        Returns:
        --------
        filtered_data : pd.DataFrame
            Subset of the input data where prediction set size does not exceed the class-specific threshold.

        removed_data : pd.DataFrame
            Subset of the input data that was removed due to exceeding the class-specific threshold.
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

    """
        Computes the average prediction set size for each predicted class and applies floor rounding to use as cutoff thresholds.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing predicted class labels and prediction set sizes.

        Returns:
        --------
        cutoff_0 : float
            Floor of the average prediction set size for class 0.

        cutoff_1 : float
            Floor of the average prediction set size for class 1.

        cutoff_2 : float
            Floor of the average prediction set size for class 2.
    """

    cutoff_0 = np.floor(np.average(data[data['pred_class_mc'] == 0]['pred_set_mc_length']))
    cutoff_1 = np.floor(np.average(data[data['pred_class_mc'] == 1]['pred_set_mc_length']))
    cutoff_2 = np.floor(np.average(data[data['pred_class_mc'] == 2]['pred_set_mc_length']))

    return cutoff_0, cutoff_1, cutoff_2