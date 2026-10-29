#########
# Imports
#########

import numpy as np


############
# Algorithms
############

# Calibration Scores

## LAC
def get_calibration_scores_lac(y_cal, y_cal_prob):

    n_cal = y_cal_prob.shape[0]
    cal_scores = 1 - y_cal_prob[np.arange(n_cal), y_cal]

    return cal_scores

## APS
def get_calibration_scores_aps(y_cal, y_cal_prob):

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

    n_cal = cal_scores.shape[0]

    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal

    qhat = np.quantile(cal_scores, q_level, method='higher')
    
    return qhat 

# Plot calibration scores and adjusted quantile for various alpha 

def plot_scores(alphas, cal_scores, quantiles, y_max):
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
