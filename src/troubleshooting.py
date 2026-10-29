import pandas as pd
import numpy as np

dn121_df = pd.read_csv('/scratch90/chris/ssl_conformal_pave/analysis/six_model_seed_test2/dn121/3/lac/alpha_010/mc_lac_run_alpha0.1.csv')
dino_dn121_df = pd.read_csv('/scratch90/chris/ssl_conformal_pave/analysis/six_model_seed_test2/dino_dn121/3/lac/alpha_010/mc_lac_run_alpha0.1.csv')

print(dn121_df.columns)
print(dino_dn121_df.columns)

#  Group by ground truth and compute average prediction set length (EVAL)
dn121_avg_eval_lengths = dn121_df.groupby('gt')['pred_set_eval_length'].mean()
dino_dn121_avg_eval_lengths = dino_dn121_df.groupby('gt')['pred_set_eval_length'].mean()

print("DenseNet121 (SL) - Avg Eval Set Length by GT:")
print(dn121_avg_eval_lengths)

print("\nDINO_DN121 (SSL) - Avg Eval Set Length by GT:")
print(dino_dn121_avg_eval_lengths)