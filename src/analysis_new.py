import pandas as pd
import numpy as np
from scipy import stats
from scipy import stats
import pandas as pd
import itertools

from scipy import stats
from scipy.stats import ttest_ind, levene
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
import re
from MLstatkit.stats import Delong_test
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def parse_array_string(s):
    s = s.strip("[]").replace(",", " ")
    parts = re.split(r'\s+', s.strip())
    try:
        return np.array([float(p) for p in parts if p])
    except ValueError:
        raise ValueError(f"Could not parse string to float array: {s}")

def run_delong_test(
    df1, df2,
    model1_name, model2_name,  # SL first, SSL second
    output_path,
    probs_col='pred_probs_eval',
    gt_col='gt',
    class_indices=[0, 2],
    class_names=["Normal", "Severe"],
    num_tests=6
):
    # Parse stringified arrays into actual arrays
    df1[probs_col] = df1[probs_col].apply(parse_array_string)
    df2[probs_col] = df2[probs_col].apply(parse_array_string)

    # Sanity checks
    assert len(df1) == len(df2), "Mismatch in row counts"
    assert all(df1[gt_col] == df2[gt_col]), "Ground truth labels do not match"

    y_true = df1[gt_col].values
    preds_sl = np.vstack(df1[probs_col])   # model1 = SL
    preds_ssl = np.vstack(df2[probs_col])  # model2 = SSL

    results = []

    for class_idx, class_name in zip(class_indices, class_names):
        y_binary = (y_true == class_idx).astype(int)
        prob_sl = preds_sl[:, class_idx]
        prob_ssl = preds_ssl[:, class_idx]

        auc_sl = roc_auc_score(y_binary, prob_sl)
        auc_ssl = roc_auc_score(y_binary, prob_ssl)

        z_score, p_value = Delong_test(y_binary, prob_ssl, prob_sl)  # still SSL vs SL (test order)

        if p_value < 0.05 / num_tests:
            significance = f"Significant (Bonferroni p<{0.05 / num_tests:.4f})"
        elif p_value < 0.05:
            significance = "Significant (p<0.05)"
        else:
            significance = "Not significant"

        results.append({
            "Class": class_name,
            "Class_Index": class_idx,
            "SL_AUC": auc_sl,
            "SSL_AUC": auc_ssl,
            "Z_Score": z_score,
            "P_Value": p_value,
            "Significance": significance
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_path, index=False)
    print(f"[🧪] DeLong results saved to: {output_path}")




def run_two_sided_ttest(
    df1, df2, 
    model1_name, model2_name, 
    output_path, 
    length_col='pred_set_eval_length', 
    subset_col='gt', 
    class_indices=[0, 1, 2], 
    class_names=['Normal', 'Indeterminate', 'Severe']
):
    # os.makedirs(output_dir, exist_ok=True)

    assert len(df1) == len(df2), f"[❌] Mismatch in row counts between {model1_name} and {model2_name}"
    assert all(df1['image'] == df2['image']), f"[❌] Mismatched image ordering between {model1_name} and {model2_name}"

    results = []
    alpha_uncorrected = 0.05
    alpha_bonferroni = alpha_uncorrected / 9 # 3 classes * 3 comparisons

    for class_idx, class_name in zip(class_indices, class_names):
        subset1 = df1[df1[subset_col] == class_idx][length_col].dropna()
        subset2 = df2[df2[subset_col] == class_idx][length_col].dropna()

        lev_stat, lev_p = levene(subset1, subset2)
        equal_var = lev_p > 0.05
        t_stat, p_val = ttest_ind(subset1, subset2, equal_var=equal_var)

        results.append({
            "Class": class_name,
            "Class_Index": class_idx,
            "SSL_Mean": np.mean(subset1),
            "SL_Mean": np.mean(subset2),
            "Levene_p": lev_p,
            "Equal_Var": equal_var,
            "T_Stat": t_stat,
            "P_Value": p_val,
            "Significant_p<0.05": p_val < alpha_uncorrected,
            "Significant_Bonferroni": p_val < alpha_bonferroni
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"[📏] T-test results saved: {output_path}")


def plot_gt_distribution_by_pred_and_set_length_stacked(cuq_df, gt_col, pred_col, pred_set_length_col, filename):
    """
    For each predicted class (0, 1, 2), plots a stacked bar chart of ground truth class proportions
    across different prediction set lengths. One subplot per predicted class.
    Adds annotations showing total sample count (N) for each bar.
    """

    # Human-readable GT labels
    gt_label_map = {0: "Normal", 1: "Indeterminate", 2: "Severe"}

    pred_classes = sorted(cuq_df[pred_col].unique())
    gt_classes = sorted(cuq_df[gt_col].unique())

    # Prepare the figure
    n_cols = len(pred_classes)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6), sharey=True)

    if n_cols == 1:
        axes = [axes]

    for i, pred_val in enumerate(pred_classes):
        ax = axes[i]
        pred_subset = cuq_df[cuq_df[pred_col] == pred_val]

        # Group by set length and GT
        group = (
            pred_subset
            .groupby([pred_set_length_col, gt_col])
            .size()
            .reset_index(name='count')
        )

        # Total count per bar (for annotation)
        total_per_length = group.groupby(pred_set_length_col)['count'].sum().rename("total")
        group = group.merge(total_per_length, on=pred_set_length_col)

        # Normalize within each bar
        group['proportion'] = group['count'] / group['total']

        # Pivot for plotting
        pivoted = group.pivot_table(
            index=pred_set_length_col,
            columns=gt_col,
            values='proportion',
            fill_value=0
        ).sort_index()

        totals = total_per_length.sort_index()

        # Plot
        bottom = None
        for gt_val in gt_classes:
            values = pivoted[gt_val] if gt_val in pivoted.columns else [0] * len(pivoted)
            ax.bar(
                pivoted.index,
                values,
                bottom=bottom,
                label=gt_label_map[gt_val],
            )
            if bottom is None:
                bottom = values
            else:
                bottom = [b + v for b, v in zip(bottom, values)]

        # Annotate with N
        for set_len in pivoted.index:
            total_n = totals.loc[set_len]
            ax.text(
                x=set_len,
                y=1.02,
                s=f"N={total_n}",
                ha='center',
                va='bottom',
                fontsize=14,
                fontweight='bold'
            )

        ax.set_title(f"Predicted Class = {gt_label_map.get(pred_val, pred_val)}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Prediction Set Length", fontsize=12)
        ax.set_ylabel("Proportion of GT Classes", fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.set_xticks([1, 2, 3])  # Only show valid lengths
        ax.tick_params(axis='both', labelsize=11)
        ax.legend(title="GT Class", fontsize=14, title_fontsize=14)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"[✅] Stacked bar plot with N annotations saved to: {filename}")

def t_test_correctly_predicted_gt_classwise(cuq_df, pred_set_length_col, gt_col='gt', pred_col='pred'):
    """
    Performs t-tests comparing prediction set lengths between correctly predicted GT classes.
    Specifically tests: GT=1 vs GT=0, GT=1 vs GT=2, GT=0 vs GT=2.
    Returns a DataFrame with mean set lengths, t-test stats, and Bonferroni-corrected significance flags.
    """

    # Filter to correctly predicted examples
    correct_df = cuq_df[cuq_df[gt_col] == cuq_df[pred_col]]

    gt_groups = {
        0: correct_df[correct_df[gt_col] == 0][pred_set_length_col],
        1: correct_df[correct_df[gt_col] == 1][pred_set_length_col],
        2: correct_df[correct_df[gt_col] == 2][pred_set_length_col],
    }

    comparisons = [
        (1, 0, "GT=1 vs GT=0"),
        (1, 2, "GT=1 vs GT=2"),
        (0, 2, "GT=0 vs GT=2"),
    ]

    bonferroni_threshold = 0.05 / 18

    results = []

    for a, b, label in comparisons:
        group_a = gt_groups[a]
        group_b = gt_groups[b]

        # Levene's test for equal variance
        levene_p = stats.levene(group_a, group_b).pvalue
        equal_var = levene_p > 0.05

        # t-test
        t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=equal_var)

        sig = p_val < 0.05
        sig_bonf = p_val < bonferroni_threshold
        sig_symbol = '**' if sig_bonf else ('*' if sig else '')

        results.append({
            'GT_Comparison': label,
            'GT_A': a,
            'GT_B': b,
            'Mean_A': group_a.mean(),
            'Mean_B': group_b.mean(),
            'Levene_p': levene_p,
            'Equal_Var': equal_var,
            'T_Stat': t_stat,
            'P_Value': p_val,
            'N_A': len(group_a),
            'N_B': len(group_b),
            'Significant_p<0.05': sig,
            'Significant_Bonferroni': sig_bonf,
            'Significance': sig_symbol,
        })

    return pd.DataFrame(results)


def t_test_by_predicted_class_only(df, pred_col='pred', length_col='pred_set_eval_length'):

    """
    Performs pairwise t-tests comparing conformal prediction set lengths across predicted classes,
    regardless of ground truth.

    Bonferroni correction is applied assuming 3 pairwise tests across 6 models (18 tests total).

    For example, compares the average set lengths for all predictions where pred=0 vs pred=1, etc.

    This is useful to evaluate whether the model is systematically more or less confident 
    depending on which class it predicts — independent of correctness.

    Returns a DataFrame with t-statistics, p-values, and significance markers.
    """

    results = []
    groups = {p: df[df[pred_col] == p][length_col].dropna() for p in [0, 1, 2]}
    bonferroni_alpha = 0.05 / 18  # 3 pairwise tests

    for (a, b) in itertools.combinations([0, 1, 2], 2):
        g1 = groups[a]
        g2 = groups[b]

        levene_p = levene(g1, g2).pvalue
        equal_var = levene_p > 0.05
        t_stat, p_val = ttest_ind(g1, g2, equal_var=equal_var)

        results.append({
            'Comparison': f'Pred={a} vs Pred={b}',
            'Pred_A': a,
            'Pred_B': b,
            'Mean_A': g1.mean(),
            'Mean_B': g2.mean(),
            'N_A': len(g1),
            'N_B': len(g2),
            'T_Stat': t_stat,
            'P_Value': p_val,
            'Levene_p': levene_p,
            'Equal_Var': equal_var,
            'Significant_p<0.05': p_val < 0.05,
            'Significant_Bonferroni': p_val < bonferroni_alpha,
            'Sig': '**' if p_val < bonferroni_alpha else ('*' if p_val < 0.05 else ''),
        })

    return pd.DataFrame(results)


def t_test_prediction_length_by_specific_misclass(cuq_df, pred_set_length_col, gt_col='gt', pred_col='pred'):
    """
    Performs t-tests comparing prediction set lengths between correctly and incorrectly predicted subsets
    for GT=0 (Normal) and GT=2 (Severe), specifically split by what class they were misclassified into.
    Returns a DataFrame including Bonferroni-corrected significance flags.
    """

    comparisons = [
        # GT = 0 (Normal): correct vs misclassified as Indeterminate
        {'gt': 0, 'pred_correct': 0, 'pred_incorrect': 1, 'label': 'Normal: Corr vs GZ'},
        {'gt': 0, 'pred_correct': 0, 'pred_incorrect': 2, 'label': 'Normal: Corr vs Severe'},

        # GT = 2 (Severe): correct vs misclassified as Indeterminate
        {'gt': 2, 'pred_correct': 2, 'pred_incorrect': 1, 'label': 'Severe: Corr vs GZ'},
        {'gt': 2, 'pred_correct': 2, 'pred_incorrect': 0, 'label': 'Severe: Corr vs Normal'},
    ]

    bonferroni_threshold = 0.05 / 24  # 24 total tests

    results = []

    for comp in comparisons:
        gt_val = comp['gt']
        pred_corr = comp['pred_correct']
        pred_incorr = comp['pred_incorrect']
        label = comp['label']

        correct = cuq_df[(cuq_df[gt_col] == gt_val) & (cuq_df[pred_col] == pred_corr)][pred_set_length_col]
        incorrect = cuq_df[(cuq_df[gt_col] == gt_val) & (cuq_df[pred_col] == pred_incorr)][pred_set_length_col]

        # Levene's test for equal variance
        levene_p = stats.levene(correct, incorrect).pvalue
        equal_var = levene_p > 0.05

        # Perform t-test
        t_stat, p_val = stats.ttest_ind(correct, incorrect, equal_var=equal_var)

        # Determine significance
        sig = p_val < 0.05
        sig_bonf = p_val < bonferroni_threshold
        sig_symbol = '**' if sig_bonf else ('*' if sig else '')

        results.append({
            'GT_Class': gt_val,
            'Comparison': label,
            'Correct_Mean': correct.mean(),
            'Incorrect_Mean': incorrect.mean(),
            'Levene_p': levene_p,
            'Equal_Var': equal_var,
            'T_Stat': t_stat,
            'P_Value': p_val,
            'N_Correct': len(correct),
            'N_Incorrect': len(incorrect),
            'Significant_p<0.05': sig,
            'Significant_Bonferroni': sig_bonf,
            'Significance': sig_symbol,
        })

    return pd.DataFrame(results)

def t_test_by_prediction_class(df, pred_col='pred', gt_col='gt', length_col='pred_set_eval_length'):
    """
    Performs pairwise t-tests across ground truth classes for each predicted class.
    This is a column-wise analysis: how uncertain is the model when it predicts a given class,
    depending on what the actual ground truth was.
    """
    results = []
    bonferroni_alpha = 0.05 / (3 * 2 * 6)  # 3 predicted classes × 2 GT comparisons × 6 models

    for pred_class in [0, 2]:
        for gt_pair in [(0,1), (0,2), (1,2)]:
            g1, g2 = gt_pair
            group1 = df[(df[pred_col] == pred_class) & (df[gt_col] == g1)][length_col].dropna()
            group2 = df[(df[pred_col] == pred_class) & (df[gt_col] == g2)][length_col].dropna()

            if len(group1) < 5 or len(group2) < 5:
                continue  # skip low-N cases

            levene_p = stats.levene(group1, group2).pvalue
            equal_var = levene_p > 0.05
            t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=equal_var)
            sig = p_val < 0.05
            sig_bonf = p_val < bonferroni_alpha
            sig_symbol = '**' if sig_bonf else ('*' if sig else '')

            results.append({
                'Predicted_Class': pred_class,
                'GT_Comparison': f"{g1} vs {g2}",
                'Mean1': group1.mean(),
                'Mean2': group2.mean(),
                'Levene_p': levene_p,
                'T_Stat': t_stat,
                'P_Value': p_val,
                'N1': len(group1),
                'N2': len(group2),
                'Significance': sig_symbol,
            })

    return pd.DataFrame(results)



def combine_stat_tests(base_dir, model_names):
    """
    Combines t-test CSV outputs from all models into two unified DataFrames:
    - One for correctly predicted GT comparisons
    - One for subset misclassification comparisons

    Args:
        base_dir (str): Path to the analysis_outputs folder.
        model_names (list of str): List of model folder names to look in.

    Returns:
        tuple: (combined_correct_gt_df, combined_subset_misclass_df)
    """
    correct_gt_dfs = []
    subset_misclass_dfs = []

    for model in model_names:
        model_dir = os.path.join(base_dir, model)

        correct_path = os.path.join(model_dir, f"{model}__correctly_predicted_gt_comparisons.csv")
        subset_path = os.path.join(model_dir, f"{model}__subset_misclassification_t_tests.csv")

        if os.path.exists(correct_path):
            df_correct = pd.read_csv(correct_path)
            df_correct["Model"] = model
            correct_gt_dfs.append(df_correct)
        else:
            print(f"[⚠️] Missing: {correct_path}")

        if os.path.exists(subset_path):
            df_subset = pd.read_csv(subset_path)
            df_subset["Model"] = model
            subset_misclass_dfs.append(df_subset)
        else:
            print(f"[⚠️] Missing: {subset_path}")

    combined_correct = pd.concat(correct_gt_dfs, ignore_index=True) if correct_gt_dfs else pd.DataFrame()
    combined_subset = pd.concat(subset_misclass_dfs, ignore_index=True) if subset_misclass_dfs else pd.DataFrame()

    print(f"[✅] Combined: {len(combined_correct)} correct-GT rows, {len(combined_subset)} subset-misclass rows")
    return combined_correct, combined_subset




def plot_accuracy_by_pred_length_for_predicted_classes(
    cuq_df, gt_col, pred_col, pred_set_length_col, model_name, save_path
):
    """
    Plots accuracy vs. prediction set length for predicted classes 0 and 2,
    labeling SL vs SSL models and showing 'N=' annotations above each point.
    """

    pred_focus = [0, 2]  # Only plot for predicted Normal and Severe
    acc_results = []

    for pred_val in pred_focus:
        pred_subset = cuq_df[cuq_df[pred_col] == pred_val]

        for length in sorted(pred_subset[pred_set_length_col].unique()):
            if length not in [1, 2, 3]:
                continue  # Skip irrelevant lengths

            subset = pred_subset[pred_subset[pred_set_length_col] == length]
            if len(subset) == 0:
                continue

            accuracy = (subset[gt_col] == pred_val).mean()
            acc_results.append({
                'Model_Type': 'SSL' if 'dino' in model_name.lower() else 'SL',
                'Predicted_Class': pred_val,
                'Pred_Set_Length': length,
                'Accuracy': accuracy,
                'N': len(subset)
            })

    acc_df = pd.DataFrame(acc_results)

    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=acc_df,
        x='Pred_Set_Length',
        y='Accuracy',
        hue='Model_Type',
        marker='o',
        style='Predicted_Class',
        palette={'SSL': 'darkgreen', 'SL': 'navy'},
        markers=True
    )

    plt.xlabel("Prediction Set Length", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks([1, 2, 3], fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(0, 1)
    plt.legend(title="Training Type", fontsize=11, title_fontsize=12)
    plt.tight_layout()

    # Annotate N with spacing depending on predicted class
    for _, row in acc_df.iterrows():
        offset = 0.035 if row['Predicted_Class'] == 0 else 0.06  # separate Normal vs Severe
        ax.text(
            row['Pred_Set_Length'],
            row['Accuracy'] + offset,
            f"N={row['N']}",
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold'
        )

    # Save figure
    filename = f"accuracy_by_pred_length_{model_name.replace('/', '_')}.png"
    plt.savefig(os.path.join(save_path, filename))
    plt.close()

    print(f"[✅] Accuracy plot with N labels saved to: {os.path.join(save_path, 'accuracy_by_pred_length_pred0_pred2_with_N.png')}")

def compare_accuracy_by_pred_length_for_predicted_classes_between_models(
    df1, df2,
    model_name_1, model_name_2,
    pred_col, gt_col, pred_set_length_col,
    filename,  # for PNG
    csv_filename=None  # optional CSV output
):
    """
    For predicted classes 0 and 2, compares accuracy vs. prediction set length
    between two models. Adds N_{SL} / N_{SSL} annotations and uses solid/dotted lines for distinction.
    Optionally saves a CSV of the underlying values.
    """

    def compute(df, model_name):
        results = []
        for pred_class in [0, 2]:
            pred_subset = df[df[pred_col] == pred_class]
            grouped = pred_subset.groupby(pred_set_length_col)
            for pred_len, group in grouped:
                acc = (group[gt_col] == group[pred_col]).mean()
                results.append({
                    'Predicted_Class': pred_class,
                    'Pred_Set_Length': pred_len,
                    'Accuracy': acc,
                    'Model': model_name,
                    'Model_Type': 'SSL' if 'dino' in model_name.lower() else 'SL',
                    'N': len(group)
                })
        return pd.DataFrame(results)

    df1_acc = compute(df1, model_name_1)
    df2_acc = compute(df2, model_name_2)
    df_all = pd.concat([df1_acc, df2_acc], ignore_index=True)

    # Save CSV if requested
    if csv_filename:
        df_all.to_csv(csv_filename, index=False)
        print(f"[💾] Accuracy results saved to: {csv_filename}")

    # Human-readable label mapping
    class_names = {0: "Normal", 2: "Severe"}
    model_label = lambda name: "SSL" if "dino" in name.lower() else "SL"
    color_map = {0: 'blue', 2: 'darkred'}

    line_styles = {
        model_name_1: 'dotted',
        model_name_2: 'solid'
    }

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    for pred_class in [0, 2]:
        for model in [model_name_1, model_name_2]:
            sub = df_all[(df_all['Predicted_Class'] == pred_class) & (df_all['Model'] == model)]
            if sub.empty:
                print(f"[⚠️] Skipping: No data for Predicted_Class={pred_class} and Model='{model}'")
                continue

            label = f"{class_names[pred_class]} ({model_label(model)})"
            linestyle = line_styles[model]
            color = color_map[pred_class]

            ax.plot(
                sub['Pred_Set_Length'],
                sub['Accuracy'],
                marker='o',
                label=label,
                linestyle=linestyle,
                color=color
            )

            for _, row in sub.iterrows():
                offset = 0.035 if row['Model_Type'] == 'SSL' else -0.045
                va = 'bottom' if row['Model_Type'] == 'SSL' else 'top'
                ax.text(
                    row['Pred_Set_Length'],
                    row['Accuracy'] + offset,
                    f"$N_{{\\mathrm{{{row['Model_Type']}}}}}={row['N']}$",
                    ha='center',
                    va=va,
                    fontsize=14,
                    fontweight='bold'
                )

    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Prediction Set Length", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_xticks([1, 2, 3])
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(title="Predicted / Model Type", fontsize=12, title_fontsize=13)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"[✅] Accuracy-by-prediction plot saved to: {filename}")


from scipy.stats import chi2_contingency

def compute_chi_squared_for_gt_distribution(cuq_df, gt_col, pred_col, pred_set_length_col):
    """
    For each predicted class (0 and 2), runs a chi-squared test comparing GT class
    distributions between prediction set lengths 1 and 3.

    Returns:
        DataFrame with chi-squared stats and significance flags.
    """

    results = []
    pred_classes = [0, 2]  # Only compare Normal and Severe
    gt_classes = sorted(cuq_df[gt_col].unique())

    alpha_bonf = 0.05 / 12  # Bonferroni correction for 12 comparisons

    for pred_class in pred_classes:
        subset = cuq_df[cuq_df[pred_col] == pred_class]

        # Only use lengths 1 and 3
        subset = subset[subset[pred_set_length_col].isin([1, 3])]
        if subset.empty:
            continue

        # Create contingency table: rows = pred_set_length (1, 3), cols = GT class
        contingency = (
            subset.groupby([pred_set_length_col, gt_col])
            .size()
            .unstack(fill_value=0)
            .reindex(index=[1, 3], fill_value=0)
            .reindex(columns=gt_classes, fill_value=0)
        )

        # Perform chi-squared test
        chi2, p, dof, expected = chi2_contingency(contingency)

        results.append({
            "Predicted_Class": pred_class,
            "Chi2": chi2,
            "p_value": p,
            "dof": dof,
            "N_len1": contingency.loc[1].sum(),
            "N_len3": contingency.loc[3].sum(),
            "Significant_p<0.05": p < 0.05,
            "Significant_Bonferroni": p < alpha_bonf
        })

    return pd.DataFrame(results)


def compare_gt_distributions_between_models(
    df1, df2,
    model_name_1, model_name_2,
    gt_col, pred_col, pred_set_length_col,
    save_dir
):
    """
    For each predicted class (0 and 2) and each prediction set length (1 and 3),
    compares the distribution of GT classes between two models using a chi-squared test.
    
    Saves a CSV with test results to: save_dir/{model_name_1}_vs_{model_name_2}_gtdist_comparison.csv
    """

    # Parameters
    pred_classes = [0, 2]
    set_lengths = [1, 3]
    gt_classes = sorted(set(df1[gt_col].unique()) | set(df2[gt_col].unique()))

    results = []

    for pred_class in pred_classes:
        for set_len in set_lengths:
            # Filter data for each model
            sl_subset = df1[(df1[pred_col] == pred_class) & (df1[pred_set_length_col] == set_len)]
            ssl_subset = df2[(df2[pred_col] == pred_class) & (df2[pred_set_length_col] == set_len)]

            if sl_subset.empty or ssl_subset.empty:
                print(f"[⚠️] Skipping: No data for Pred={pred_class}, Len={set_len} in one of the models.")
                continue

            # Contingency table: rows = [SL, SSL], columns = GT class counts
            sl_counts = sl_subset[gt_col].value_counts().reindex(gt_classes, fill_value=0)
            ssl_counts = ssl_subset[gt_col].value_counts().reindex(gt_classes, fill_value=0)
            contingency = pd.DataFrame([sl_counts, ssl_counts], index=["SL", "SSL"])

            # Run chi-squared test
            chi2, p, dof, expected = chi2_contingency(contingency)

            results.append({
                "Predicted_Class": pred_class,
                "Set_Length": set_len,
                "SL_Model": model_name_1,
                "SSL_Model": model_name_2,
                "Chi2": chi2,
                "p_value": p,
                "dof": dof,
                "N_SL": sl_counts.sum(),
                "N_SSL": ssl_counts.sum()
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Bonferroni correction (3 model pairs × 4 tests = 12)
    alpha = 0.05
    corrected_alpha = alpha / 12
    results_df["Significant (p<0.05)"] = results_df["p_value"] < 0.05
    results_df["Significant (Bonferroni)"] = results_df["p_value"] < corrected_alpha
    results_df["Bonferroni_alpha"] = corrected_alpha

    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name_1}_vs_{model_name_2}_gtdist_comparison.csv")
    results_df.to_csv(save_path, index=False)

    print(f"[✅] Chi-squared GT distribution comparison saved to:\n{save_path}")

    return results_df

def run_all_analyses(cuq_df, pred_col, pred_set_length_col, save_path, model_name):
    os.makedirs(save_path, exist_ok=True)

    def label(name):  # Helper for consistent file naming
        return f"{model_name}__{name}"

    # Subset misclassification t-tests
    df_t_test = t_test_prediction_length_by_specific_misclass(
        cuq_df, pred_set_length_col, pred_col=pred_col
    )
    df_t_test["Model"] = model_name
    df_t_test.to_csv(os.path.join(save_path, label("subset_misclassification_t_tests.csv")), index=False)

    # Correct GT classwise t-tests
    df_t_test_correct_gt = t_test_correctly_predicted_gt_classwise(
        cuq_df, pred_set_length_col, gt_col='gt', pred_col=pred_col
    )
    df_t_test_correct_gt["Model"] = model_name
    df_t_test_correct_gt.to_csv(os.path.join(save_path, label("correctly_predicted_gt_comparisons.csv")), index=False)

    # New: Prediction-only classwise t-tests
    df_t_test_pred_only = t_test_by_predicted_class_only(
        cuq_df, pred_col=pred_col, length_col=pred_set_length_col
    )
    df_t_test_pred_only["Model"] = model_name
    df_t_test_pred_only.to_csv(os.path.join(save_path, label("predicted_class_comparisons.csv")), index=False)

    plot_accuracy_by_pred_length_for_predicted_classes(
        cuq_df, 'gt', pred_col, pred_set_length_col, model_name=model_name, save_path=save_path
    )

    plot_gt_distribution_by_pred_and_set_length_stacked(
        cuq_df, 'gt', pred_col, pred_set_length_col,
        os.path.join(save_path, label("stacked_gt_dist.png"))
    )

    results_df = compute_chi_squared_for_gt_distribution(
        cuq_df, gt_col="gt", pred_col="pred_class_eval", pred_set_length_col="pred_set_eval_length"
    )
    results_df.to_csv(
        os.path.join(save_path, "chi_squared_gt_distribution.csv"),
        index=False
    )

    df_t_test_columnwise = t_test_by_prediction_class(
        cuq_df, pred_col=pred_col, length_col=pred_set_length_col
    )
    df_t_test_columnwise["Model"] = model_name
    df_t_test_columnwise.to_csv(
        os.path.join(save_path, label("predicted_class_columnwise_t_tests.csv")),
        index=False
    )



def plot_predset_heatmap_from_csv(save_dir, csv_path, model_name):
    """
    Plots a 3x3 heatmap of average prediction set lengths for eval predictions.
    Rows = ground truth, Columns = predicted class.

    Parameters:
        csv_path (str): Path to model's mc_lac_run_alpha0.1.csv file.
        model_name (str): Name of the model (for title and saving).
        save_dir (str): Directory to save heatmap. If None, saves alongside CSV.
    """
    df = pd.read_csv(csv_path)

    # Sanity check: ensure necessary columns exist
    required_cols = ["gt", "pred_class_eval", "pred_set_eval_length"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Aggregate mean prediction set length and count for each (gt, pred) pair
    grouped = df.groupby(["gt", "pred_class_eval"])["pred_set_eval_length"].agg(["mean", "count"]).reset_index()

    # Initialize heatmap matrices
    heatmap = np.full((3, 3), np.nan)
    annot = np.full((3, 3), "", dtype=object)

    # Fill in values
    for _, row in grouped.iterrows():
        gt = int(row["gt"])
        pred = int(row["pred_class_eval"])
        mean_val = row["mean"]
        count = int(row["count"])
        heatmap[gt, pred] = mean_val
        annot[gt, pred] = f"{mean_val:.2f}\nN={count}"

    # Plotting
    plt.figure(figsize=(6, 5))
    sns.heatmap(heatmap, annot=annot, fmt="", cmap="Blues", cbar=False, annot_kws={"fontsize": 13})
    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth")
    # plt.title(f"{model_name} – Eval Prediction Set Lengths")
    plt.xticks(ticks=[0.5, 1.5, 2.5], labels=["Normal", "Indeterminate", "Severe"])
    plt.yticks(ticks=[0.5, 1.5, 2.5], labels=["Normal", "Indeterminate", "Severe"], rotation=0)
    plt.tight_layout()

    # Save
    out_path = os.path.join(save_dir, f"{model_name}_eval_setlength_heatmap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()



def plot_auc(save_path, csv_path, model_name, probs_col="pred_probs_eval", gt_col="gt"):
    # Load and parse
    df = pd.read_csv(csv_path)
    df[probs_col] = df[probs_col].apply(parse_array_string)
    probs = np.vstack(df[probs_col])
    y_true = df[gt_col].values

    class_names = ["Normal", "Indeterminate", "Severe"]
    colors = ['blue', 'orange', 'green']

    plt.figure(figsize=(8, 6))

    for class_idx, class_name in enumerate(class_names):
        binary_labels = (y_true == class_idx).astype(int)
        try:
            fpr, tpr, _ = roc_curve(binary_labels, probs[:, class_idx])
            auc = roc_auc_score(binary_labels, probs[:, class_idx])
            plt.plot(fpr, tpr, label=f"{class_name} (AUC = {auc:.3f})", color=colors[class_idx])
        except ValueError:
            # Skip if class is not present
            continue

    # Plot random baseline
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')

    # Format plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curves – {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    # Save
    outpath = os.path.join(save_path, f"{model_name}_roc_curves.png")
    plt.savefig(outpath)
    plt.close()
    print(f"[📈] ROC plot saved to {outpath}")

def main():
    '''
    Runs prediction set length and GT/pred agreement analysis for each model file.
    Outputs CSVs and plots into a dedicated directory per model with clearly labeled filenames.
    Also runs two-sided t-tests comparing SSL vs SL models by GT class.
    '''

    # === Paths to model prediction CSVs ===
    dn121_path = "/scratch90/chris/ssl_conformal_pave/analysis/six_model_seed_test2/dn121/3/lac/alpha_010/mc_lac_run_alpha0.1.csv"
    dino_dn121_path = "/scratch90/chris/ssl_conformal_pave/analysis/six_model_seed_test2/dino_dn121/3/lac/alpha_010/mc_lac_run_alpha0.1.csv"
    rn_50_path = "/scratch90/chris/ssl_conformal_pave/analysis/six_model_seed_test2/rn50/3/lac/alpha_010/mc_lac_run_alpha0.1.csv"
    dino_rn_50_path = "/scratch90/chris/ssl_conformal_pave/analysis/six_model_seed_test2/dino_rn50/3/lac/alpha_010/mc_lac_run_alpha0.1.csv"
    effnetv2s_path = "/scratch90/chris/ssl_conformal_pave/analysis/six_model_seed_test2/effnetv2s/3/lac/alpha_010/mc_lac_run_alpha0.1.csv"
    dino_effnetv2s_path = "/scratch90/chris/ssl_conformal_pave/analysis/six_model_seed_test2/dino_effnetv2s/3/lac/alpha_010/mc_lac_run_alpha0.1.csv"

    model_configs = {
        "dn121": dn121_path,
        "dino_dn121": dino_dn121_path,
        "resnet50": rn_50_path,
        "dino_resnet50": dino_rn_50_path,
        "effnetv2s": effnetv2s_path,
        "dino_effnetv2s": dino_effnetv2s_path,
    }

    # === Run individual model analyses ===
    for model_name, csv_path in model_configs.items():
        print(f"[🔍] Running analysis for: {model_name}")
        save_path = os.path.join("analysis_outputs_eval", model_name)

        cuq_df = pd.read_csv(csv_path)
        pred_col = 'pred_class_eval'
        pred_set_length_col = 'pred_set_eval_length'

        run_all_analyses(cuq_df, pred_col, pred_set_length_col, save_path, model_name)

        # Prediction and Ground-Truth Heatmaps
        plot_predset_heatmap_from_csv(save_path, csv_path, model_name=model_name)

        # AUC
        plot_auc(save_path, csv_path, model_name=model_name)

    # === Compare SSL vs SL accuracy curves ===
    print("[📊] Generating accuracy comparison plots between baseline and DINO models...")

    pairs = [
        ("dn121", "dino_dn121"),
        ("resnet50", "dino_resnet50"),
        ("effnetv2s", "dino_effnetv2s")
    ]

    model_dfs = {name: pd.read_csv(path) for name, path in model_configs.items()}

    for baseline, dino in pairs:
        df1 = model_dfs[baseline]
        df2 = model_dfs[dino]

        compare_accuracy_by_pred_length_for_predicted_classes_between_models(
            df1=df1,
            df2=df2,
            model_name_1=baseline,
            model_name_2=dino,
            pred_col='pred_class_eval',
            gt_col='gt',
            pred_set_length_col='pred_set_eval_length',
            filename=os.path.join("analysis_outputs_eval", f"{baseline}_vs_{dino}_accuracy_by_pred.png"),
            csv_filename=os.path.join("analysis_outputs_eval", f"{baseline}_vs_{dino}_accuracy_by_pred.csv")
        )

    # === Combine per-model t-tests ===
    print("[📦] Combining all statistical test outputs...")
    model_names = list(model_configs.keys())
    base_dir = "analysis_outputs_eval"

    correct_gt_df, subset_misclass_df = combine_stat_tests(base_dir, model_names)
    correct_gt_df.to_csv(os.path.join(base_dir, "combined_correct_gt_t_tests.csv"), index=False)
    subset_misclass_df.to_csv(os.path.join(base_dir, "combined_subset_misclassification_t_tests.csv"), index=False)

    print("[🧪] Creating prediction versus ground truth average conformal prediction length confusion matrices...")

    model_name_map = {
    "dn121": "DenseNet121 SL",
    "dino_dn121": "DenseNet121 SSL",
    "resnet50": "ResNet50 SL",
    "dino_resnet50": "ResNet50 SSL",
    "effnetv2s": "EffNetV2S SL",
    "dino_effnetv2s": "EffNetV2S SSL"
    }

    print("[🧪] Running two-sided t-tests comparing SSL vs SL models...")

    # Run classwise t-tests between SSL vs SL
    for sl, ssl in pairs:
        sl_df = model_dfs[sl]
        ssl_df = model_dfs[ssl]

        # Build a descriptive filename
        outname = f"{ssl}_vs_{sl}__t_test_pred_set_length_by_gt.csv"
        outpath = os.path.join(base_dir, outname)

        run_two_sided_ttest(
            df1=ssl_df,
            df2=sl_df,
            model1_name=sl,
            model2_name=ssl,
            output_path=outpath,
            length_col='pred_set_eval_length',
            subset_col='pred_class_eval',
            class_indices=[0, 1, 2],
            class_names=["Normal", "Indeterminate", "Severe"]
        )

    print("[📐] Running DeLong AUC tests for SSL vs SL model pairs...")
    
    for ssl, sl in pairs:
        ssl_df = model_dfs[ssl]  # SSL first
        sl_df = model_dfs[sl]    # SL second
        outname = f"{ssl}_vs_{sl}_delong_auc_comparison.csv"
        outpath = os.path.join(base_dir, outname)

        run_delong_test(
            df1=ssl_df,               # SSL
            df2=sl_df,                # SL
            model1_name=ssl,          # SSL
            model2_name=sl,           # SL
            output_path=outpath,
            probs_col='pred_probs_eval',
            gt_col='gt',
            class_indices=[0, 2],
            class_names=["Normal", "Severe"],
            num_tests=6
        )

    for baseline, dino in pairs:
        df1 = model_dfs[baseline]
        df2 = model_dfs[dino]

        compare_gt_distributions_between_models(
            df1=df1,
            df2=df2,
            model_name_1=baseline,
            model_name_2=dino,
            gt_col='gt',
            pred_col='pred_class_eval',
            pred_set_length_col='pred_set_eval_length',
            save_dir="analysis_outputs_eval"
        )


    print("[✅] All analyses complete!")


if __name__ == "__main__":
    main()