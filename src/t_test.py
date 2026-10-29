import argparse
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, levene

def parse_args():
    parser = argparse.ArgumentParser(description="Run two-sided t-test on prediction set lengths between SSL and SL.")
    parser.add_argument('--model1_csv', type=str, required=True, help='Path to SSL model CSV (Model 1)')
    parser.add_argument('--model2_csv', type=str, required=True, help='Path to SL model CSV (Model 2)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save t-test results')
    parser.add_argument('--length_col', type=str, default='pred_set_mc_length', help='Column for prediction set length')
    parser.add_argument('--subset_col', type=str, default='gt', help='Column to subset by class label')
    parser.add_argument('--class_indices', type=int, nargs='+', default=[0, 1, 2], help='Class indices to compare')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Normal', 'Indeterminate', 'Severe'], help='Class names')
    return parser.parse_args()

def two_sided_ttest(a, b, equal_var):
    t_stat, p_val = ttest_ind(a, b, equal_var=equal_var)
    return t_stat, p_val

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df1 = pd.read_csv(args.model1_csv)
    df2 = pd.read_csv(args.model2_csv)

    assert len(df1) == len(df2), "Mismatch in number of rows."
    assert all(df1['image'] == df2['image']), "Image ordering mismatch between files."

    results = []
    alpha_uncorrected = 0.05
    alpha_bonferroni = alpha_uncorrected / 9  # 3 models × 3 classes = 9 tests

    for class_idx, class_name in zip(args.class_indices, args.class_names):
        subset1 = df1[df1[args.subset_col] == class_idx][args.length_col].dropna()
        subset2 = df2[df2[args.subset_col] == class_idx][args.length_col].dropna()

        # Levene’s test for equal variances
        lev_stat, lev_p = levene(subset1, subset2)
        equal_var = lev_p > 0.05

        t_stat, p_val = two_sided_ttest(subset1, subset2, equal_var=equal_var)

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

    # Optional: include "All" class if needed
    # ...

    out_df = pd.DataFrame(results)
    out_path = os.path.join(args.output_dir, 't_test_results.csv')
    out_df.to_csv(out_path, index=False)

    print(f"[✅] Two-sided t-test results saved to {out_path}")
    print(f"Bonferroni-corrected threshold: {alpha_bonferroni:.4f}")

if __name__ == "__main__":
    main()
