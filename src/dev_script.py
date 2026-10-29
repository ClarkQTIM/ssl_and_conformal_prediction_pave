import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import re

# === HARDCODED PATHS ===
csv1_path = "/scratch90/chris/ssl_conformal_pave/analysis/six_model_seed_test2/dn121/3/lac/alpha_010/mc_lac_run_alpha0.1.csv"
csv2_path = "/scratch90/chris/ssl_conformal_pave/analysis/six_model_seed_test2/dino_dn121/3/lac/alpha_010/mc_lac_run_alpha0.1.csv"


# === CONFIG ===
probs_col = 'pred_probs_eval'
gt_col = 'gt'
num_classes = 3

def parse_array_string(s):
    """Parses string like '[0.1, 0.2, 0.7]' into a float array."""
    s = s.strip("[]").replace(",", " ")
    parts = re.split(r'\s+', s.strip())
    return np.array([float(p) for p in parts if p])

def load_and_parse(csv_path):
    df = pd.read_csv(csv_path)
    df[probs_col] = df[probs_col].apply(parse_array_string)
    return df

def compute_aucs(df, model_name):
    print(f"\n📊 AUCs for: {model_name}")
    y_true = df[gt_col].values
    y_probs = np.vstack(df[probs_col].values)

    for class_idx in range(num_classes):
        y_binary = (y_true == class_idx).astype(int)
        auc = roc_auc_score(y_binary, y_probs[:, class_idx])
        print(f"  Class {class_idx} AUC: {auc:.4f}")

# === Run ===
df1 = load_and_parse(csv1_path)
df2 = load_and_parse(csv2_path)

compute_aucs(df1, model_name="Model 1 (SL)")
compute_aucs(df2, model_name="Model 2 (SSL)")
