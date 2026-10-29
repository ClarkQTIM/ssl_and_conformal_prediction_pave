"""
Conformal Prediction Utilities
------------------------------

This module implements calibration score functions, quantile finding, and prediction set construction
for conformal prediction using the LAC (Least Ambiguous Classifier) and APS (Adaptive Prediction Set) methods.

The code is based on methods presented in:
"A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"
Angelopoulos & Bates (2021)
https://arxiv.org/abs/2107.07511
"""

#########
# Imports
#########

import numpy as np

#################################
# Conformal Prediction Algorithms
#################################

# Calibration Scores

## LAC
def get_calibration_scores_lac(y_cal, y_cal_prob):
    """
    Compute Least Ambiguous Classifier (LAC) calibration scores.

    Parameters:
        y_cal (np.array): Ground truth labels for the calibration set.
        y_cal_prob (np.array): Model-predicted probabilities for each class.

    Returns:
        np.array: Calibration scores as 1 - predicted probability of the true class.
    """

    n_cal = y_cal_prob.shape[0]
    cal_scores = 1 - y_cal_prob[np.arange(n_cal), y_cal]

    return cal_scores

## APS
def get_calibration_scores_aps(y_cal, y_cal_prob):
    """
    Compute Adaptive Prediction Set (APS) calibration scores.

    APS scores are the cumulative probability assigned to the true class
    in a sorted (descending) cumulative probability distribution.

    Parameters:
        y_cal (np.array): Ground truth labels for the calibration set.
        y_cal_prob (np.array): Model-predicted probabilities for each class.

    Returns:
        np.array: Calibration scores using APS method.
    """

    # Setup
    n_cal=y_cal_prob.shape[0] # number of calibration points

    # Calibration Scores
    cal_pi = y_cal_prob.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(y_cal_prob, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n_cal), y_cal
    ]

    return cal_scores

# Quantile

def find_quantile(cal_scores, alpha):
    """
    Compute quantile threshold q̂ based on calibration scores and target miscoverage level α.

    Parameters:
        cal_scores (np.array): Calibration scores.
        alpha (float): Target miscoverage level (e.g., 0.1 for 90% coverage).

    Returns:
        float: Quantile q̂ used to threshold prediction sets.
    """

    n_cal = cal_scores.shape[0]

    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal

    qhat = np.quantile(cal_scores, q_level, method='higher')
    
    return qhat 

# Plot calibration scores and adjusted quantile for various alpha 

def plot_scores(alphas, cal_scores, quantiles, y_max):
    """
    Plot histogram of calibration scores and vertical lines for quantile thresholds.

    Parameters:
        alphas (list): List of alpha values (miscoverage levels).
        cal_scores (np.array): Calibration scores.
        quantiles (list): List of q̂ quantiles corresponding to alphas.
        y_max (float): Maximum y-axis value for plotting vertical lines.
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

# Prediction Sets

## LAC
def cuq_pred_set_lac(cal_scores, alpha, y_test_probs, y_test):
    """
    Construct LAC-based conformal prediction sets using test probabilities and q̂.

    Parameters:
        cal_scores (np.array): Calibration scores from LAC.
        alpha (float): Target miscoverage level.
        y_test_probs (np.array): Probabilities predicted on the test set.
        y_test (np.array): Ground truth test labels.

    Returns:
        prediction_sets (np.array): Boolean mask of conformal prediction sets.
        mean_width (float): Average size of prediction sets.
        covered (np.array): Boolean array indicating if true label is in the set.
        pct_coverage (float): Proportion of true labels covered.
        qhat (float): Threshold used for prediction set construction.
    """

    qhat = find_quantile(cal_scores, alpha)
    prediction_sets = y_test_probs >= (1-qhat) 

    # Mean width of prediction set
    mean_width = prediction_sets.sum(axis=1).mean()

    ## Percent coverage if we have y_test
    if y_test is not None:
        covered = prediction_sets[np.arange(y_test_probs.shape[0]), y_test]
        pct_coverage = covered.mean()
        return prediction_sets, mean_width, covered, pct_coverage, qhat

    else:
        return prediction_sets, mean_width, qhat

## APS
def cuq_pred_set_aps(cal_scores, alpha, y_test_probs, y_test):
    """
    Construct APS-based conformal prediction sets using test probabilities and q̂.

    Parameters:
        cal_scores (np.array): Calibration scores from APS.
        alpha (float): Target miscoverage level.
        y_test_probs (np.array): Probabilities predicted on the test set.
        y_test (np.array): Ground truth test labels.

    Returns:
        prediction_sets (np.array): Boolean mask of conformal prediction sets.
        mean_width (float): Average size of prediction sets.
        covered (np.array): Boolean array indicating if true label is in the set.
        pct_coverage (float): Proportion of true labels covered.
        qhat (float): Threshold used for prediction set construction.
    """

    # Get the score quantile
    qhat = find_quantile(cal_scores, alpha)

    # Deploy
    val_pi = y_test_probs.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(y_test_probs, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)

    # Mean width of prediction set
    mean_width = prediction_sets.sum(axis=1).mean()

    ## Percent coverage if we have y_test
    if y_test is not None:
        covered = prediction_sets[np.arange(y_test_probs.shape[0]), y_test]
        pct_coverage = covered.mean()
        return prediction_sets, mean_width, covered, pct_coverage, qhat
    else:
        return prediction_sets, mean_width, qhat
