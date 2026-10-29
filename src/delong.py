import pandas as pd
import numpy as np
import os
import argparse
import re
from MLstatkit.stats import Delong_test

def parse_array_string(s):
    s = s.strip("[]").replace(",", " ")
    parts = re.split(r'\s+', s.strip())
    try:
        return np.array([float(p) for p in parts if p])
    except ValueError:
        raise ValueError(f"Could not parse string to float array: {s}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run DeLong test between two models for class-wise AUC.")
    parser.add_argument('--model1_csv', type=str, required=True, help='Path to SSL CSV (Model 1)')
    parser.add_argument('--model2_csv', type=str, required=True, help='Path to SL CSV (Model 2)')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save CSV results')
    parser.add_argument('--probs_col', type=str, default='pred_probs_mc', help='Column containing prediction probabilities')
    parser.add_argument('--gt_col', type=str, default='gt', help='Ground truth label column')
    parser.add_argument('--class_indices', type=int, nargs='+', default=[0, 2], help='Class indices to compare (e.g., 0 2)')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Normal', 'Severe'], help='Names of classes to label results')
    parser.add_argument('--num_tests', type=int, default=6, help='Total number of hypothesis tests for Bonferroni correction')
    return parser.parse_args()

def get_significance(p, alpha=0.05, num_tests=6):
    bonf_alpha = alpha / num_tests
    if p < bonf_alpha:
        return f"Significant (Bonferroni p<{bonf_alpha:.4f})"
    elif p < alpha:
        return "Significant (p<0.05)"
    else:
        return "Not significant"

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and align
    df1 = pd.read_csv(args.model1_csv)
    df2 = pd.read_csv(args.model2_csv)
    df1[args.probs_col] = df1[args.probs_col].apply(parse_array_string)
    df2[args.probs_col] = df2[args.probs_col].apply(parse_array_string)
    assert len(df1) == len(df2), "Mismatch in row counts"
    assert all(df1[args.gt_col] == df2[args.gt_col]), "Ground truth labels do not match"

    y_true = df1[args.gt_col].values
    preds1 = np.vstack(df1[args.probs_col])
    preds2 = np.vstack(df2[args.probs_col])

    rows = []
    for class_idx, class_name in zip(args.class_indices, args.class_names):
        y_binary = (y_true == class_idx).astype(int)
        prob_ssl = preds1[:, class_idx]
        prob_sl = preds2[:, class_idx]

        z_score, p_value = Delong_test(y_binary, prob_ssl, prob_sl)
        sig = get_significance(p_value, alpha=0.05, num_tests=args.num_tests)

        print(f"{class_name}: Z = {z_score:.3f}, p = {p_value:.5f} → {sig}")
        rows.append({
            "Class": class_name,
            "Class_Index": class_idx,
            "Z_Score": z_score,
            "P_Value": p_value,
            "Significance": sig
        })

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(args.output_dir, 'delong_results.csv')
    df_out.to_csv(out_path, index=False)
    print(f"[✅] DeLong test results saved to {out_path}")

if __name__ == "__main__":
    main()
