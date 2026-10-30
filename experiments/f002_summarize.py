"""Summaries behind AUDIT.md F-002 / F-021 / F-022. Reads experiments/outputs/f002_*.csv (from f002_split_sensitivity.py)
and recomputes the full-set (all 1,686 images) DeLong and the legacy-split patient overlap. Aggregates only.

Run: PYENV_VERSION=dr_ssl_sl python experiments/f002_summarize.py
"""
import os
import sys

import numpy as np
import pandas as pd
from MLstatkit.stats import Delong_test
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import f002_split_sensitivity as f  # noqa: E402

O = HERE + "/outputs/"
R, T4, T1 = (pd.read_csv(O + n) for n in ("f002_per_model.csv", "f002_table4_ttests.csv", "f002_table1_delong.csv"))
CN = {0: "Normal", 1: "Indet", 2: "Severe"}
pd.set_option("display.width", 250)

d = f.load()
n = len(d["dn121"]["gt"])
cal, test = f.split("image", 0, n, None)
g = d["dn121"]["group"]
print(f"== legacy split: n={n}, n_cal={len(cal)}, n_test={len(test)}; cal women={len(set(g[cal]))}, "
      f"of which also in test={len(set(g[cal]) & set(g[test]))}; test images sharing a woman with cal={np.isin(g[test], g[cal]).mean():.3f}")
print(f"   images per woman over all rows = {n / len(set(g)):.2f}")

print("\n== coverage over seeds (mean, sd, 5th pct, min) [F-002/F-022]")
print(R.groupby(["scheme", "model"]).coverage.agg(mean="mean", sd="std", p5=lambda x: x.quantile(.05), min="min").round(4))
print("\n== class-conditional coverage, seed 0 image split (the paper's split) [F-022]")
print(R[(R.scheme == "image") & (R.seed == 0)].set_index("model")[["coverage", "cov_gt0", "cov_gt1", "cov_gt2"]].round(3))
print("\n== class-conditional coverage, mean over seeds")
print(R.groupby(["scheme", "model"])[["cov_gt0", "cov_gt1", "cov_gt2"]].mean().round(3))
print("\n== max empty-set fraction over all runs:", R.frac_empty.max())

print("\n== Table 4 stability [F-021]: SL-SSL CPSL gap; % splits significant uncorrected / Bonferroni(/9)")
for sch in ("image", "patient"):
    print("--", sch)
    for (a, c), x in T4[T4.scheme == sch].groupby(["arch", "pred_class"]):
        gap = x.sl_mean - x.ssl_mean
        s0 = gap[x.seed == 0].item()
        print(f" {a:10s}{CN[c]:7s} seed0 gap={s0:+.3f} mean={gap.mean():+.3f} [{gap.quantile(.05):+.3f},{gap.quantile(.95):+.3f}] "
              f"p<.05: {100 * (x.p < .05).mean():5.1f}%  Bonf: {100 * (x.p < .05 / 9).mean():5.1f}%  seed0 pct-rank: {100 * (gap < s0).mean():.0f}")

print("\n== Table 1 stability [F-021]: AUC(SSL)-AUC(SL); % splits significant uncorrected / Bonferroni(/6)")
for sch in ("image", "patient"):
    print("--", sch)
    for (a, c), x in T1[T1.scheme == sch].groupby(["arch", "cls"]):
        dd = x.auc_ssl - x.auc_sl
        print(f" {a:10s}{CN[c]:7s} seed0 dAUC={dd[x.seed == 0].item():+.4f} mean={dd.mean():+.4f} "
              f"p<.05: {100 * (x.p < .05).mean():5.1f}%  Bonf: {100 * (x.p < .05 / 6).mean():5.1f}%")

print(f"\n== DeLong on ALL {n} images (no split needed for AUC) [F-021]")
for sl, ssl in f.PAIRS:
    for c in (0, 2):
        y = (d[sl]["gt"] == c).astype(int)
        a, b = d[ssl]["probs"][:, c], d[sl]["probs"][:, c]
        _, p = Delong_test(y, a, b)
        print(f" {sl:10s}{CN[c]:7s} AUC sl={roc_auc_score(y, b):.4f} ssl={roc_auc_score(y, a):.4f} p={p:.4f} p<.05:{p < .05} Bonf(/6):{p < .05 / 6}")
