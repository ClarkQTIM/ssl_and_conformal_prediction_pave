"""Recompute every number in paper Tables 1-4 and the Sec. 4.x text from the SAVED legacy per-image outputs
(`analysis/six_model_seed_test2/<model>/3/lac/alpha_010/mc_lac_run_alpha0.1.csv`), compare to the value printed in
the paper, and write `ledger/claims_ledger.yaml` (the ledger is generated here so it cannot drift from the check).

Scope/honesty: this verifies "paper == saved legacy outputs, independently recomputed with pandas/scipy/sklearn".
It does NOT re-execute the legacy pipeline (Phase 4). Only aggregates are written; no patient/image IDs.

Run: PYENV_VERSION=dr_ssl_sl python verification/reproduce_paper_tables.py
"""
import hashlib
import os

import numpy as np
import pandas as pd
import yaml
from scipy.stats import levene, ttest_ind
from sklearn.metrics import roc_auc_score
from MLstatkit.stats import Delong_test

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV = ROOT + "/analysis/six_model_seed_test2/{m}/3/lac/alpha_010/mc_lac_run_alpha0.1.csv"
ARCH = {"DenseNet121": "dn121", "ResNet50": "rn50", "EffNetV2S": "effnetv2s"}
CLS = {"Normal": 0, "Indeterminate": 1, "Severe": 2}
TEX = "paper/main_els.tex"
SCRIPT = "verification/reproduce_paper_tables.py"
LEGACY = "src/analysis_new.py"

_cache = {}


def load(m):
    if m not in _cache:
        d = pd.read_csv(CSV.format(m=m))
        d["P"] = [np.array(s.strip("[]").split(), dtype=float) for s in d["pred_probs_eval"]]
        _cache[m] = d
    return _cache[m]


def sha(m):
    return hashlib.sha256(open(CSV.format(m=m), "rb").read()).hexdigest()


def sig(x, y):
    """Repo logic: Levene -> Student (equal var) or Welch; returns p."""
    return ttest_ind(x, y, equal_var=levene(x, y)[1] > 0.05)[1]


rows = []


def add(cid, loc, claim, reported, computed, kind, models, fn, note="", tol=None):
    """kind: 'mean2' (2 dp), 'int', 'pct' (rounded integer %), 'star' (bool/str)."""
    if kind == "mean2":
        ok, diff = round(computed, 2) == reported, abs(computed - reported)
    elif kind == "pct":
        ok, diff = round(100 * computed) == reported, abs(100 * computed - reported)
    elif kind == "int":
        ok, diff = int(computed) == reported, abs(int(computed) - reported)
    else:
        ok, diff = computed == reported, 0.0 if computed == reported else 1.0
    rows.append(dict(
        id=cid, paper_location=loc, claim_text=claim, reported_value=reported,
        script=SCRIPT, function=fn, legacy_script=LEGACY, config_hparams=dict(cp="LAC", single_pass=True, num_mc_cols_unused=50),
        alpha=0.1, seeds=dict(calibration_test_split_random_state=0), split="test2: 338 calibration / 1,348 test (image-level, 20%)",
        input_csv=[dict(path=CSV.format(m=m).replace(ROOT + "/", ""), sha256=sha(m)) for m in models],
        model_checkpoint="UNKNOWN (upstream, Phase 8)", reproduced_legacy=(computed if isinstance(computed, str) else float(100 * computed if kind == 'pct' else computed)),
        reproduced_refactor=None, abs_diff=float(diff),
        evidence="saved legacy outputs, independently recomputed; legacy code NOT re-executed",
        status="verified" if ok else "discrepancy", notes=note))


def star(p):
    return "*" if p < 0.05 else "Not Sig."


# ---- Table 1 (tex 288-306): DeLong, AUC SL vs SSL, per class (one-vs-rest on probs column) ----
T1 = {("DenseNet121", "Normal"): (0.87, 0.87, "Not Sig."), ("DenseNet121", "Severe"): (0.82, 0.84, "*"),
      ("ResNet50", "Normal"): (0.87, 0.88, "Not Sig."), ("ResNet50", "Severe"): (0.83, 0.84, "Not Sig."),
      ("EffNetV2S", "Normal"): (0.87, 0.88, "Not Sig."), ("EffNetV2S", "Severe"): (0.81, 0.84, "*")}
for (a, c), (sl_r, ssl_r, st_r) in T1.items():
    sl, ssl = load(ARCH[a]), load("dino_" + ARCH[a])
    y = (sl["gt"].values == CLS[c]).astype(int)
    ps, pd_ = np.vstack(sl["P"])[:, CLS[c]], np.vstack(ssl["P"])[:, CLS[c]]
    _, p = Delong_test(y, pd_, ps)
    m = [ARCH[a], "dino_" + ARCH[a]]
    add(f"T1-{a}-{c}-AUC_SL", f"Table 1, {a} {c}, SL AUC", "OvR AUC, SL", sl_r, roc_auc_score(y, ps), "mean2", m, "sklearn roc_auc_score")
    add(f"T1-{a}-{c}-AUC_SSL", f"Table 1, {a} {c}, SSL AUC", "OvR AUC, SSL", ssl_r, roc_auc_score(y, pd_), "mean2", m, "sklearn roc_auc_score")
    add(f"T1-{a}-{c}-sig", f"Table 1, {a} {c}, Sig.", "DeLong p<0.05 (uncorrected)", st_r, star(p), "star", m, "MLstatkit Delong_test (cross-checked, F-015)",
        note=f"p={p:.4f}; Bonferroni(/6) significant: {p < 0.05 / 6} (F-014)")

# ---- Table 2 (tex 327-333): mean CPSL (N) by predicted class, DN121; pairwise t-tests ----
T2 = {"dn121": {"Normal": (1.44, 739), "Indeterminate": (2.61, 394), "Severe": (2.05, 215)},
      "dino_dn121": {"Normal": (1.22, 817), "Indeterminate": (2.57, 302), "Severe": (1.75, 229)}}
for m, cells in T2.items():
    d = load(m)
    for c, (mean_r, n_r) in cells.items():
        s = d.loc[d["pred_class_eval"] == CLS[c], "pred_set_eval_length"]
        add(f"T2-{m}-{c}-mean", f"Table 2, DenseNet121 {'SSL' if 'dino' in m else 'SL'} predicted {c}, mean CPSL", "mean CPSL", mean_r, s.mean(), "mean2", [m], "pandas groupby mean")
        add(f"T2-{m}-{c}-N", f"Table 2, ... predicted {c}, N", "N images", n_r, len(s), "int", [m], "pandas count")
    for a_, b_ in [("Normal", "Indeterminate"), ("Normal", "Severe"), ("Indeterminate", "Severe")]:
        x = d.loc[d["pred_class_eval"] == CLS[a_], "pred_set_eval_length"]
        y = d.loc[d["pred_class_eval"] == CLS[b_], "pred_set_eval_length"]
        p = sig(x, y)
        add(f"T2-{m}-{a_}v{b_}-sig", f"Table 2, {m}, {a_} vs {b_}", "t-test p<0.05", "*", star(p), "star", [m], "scipy levene+ttest_ind", note=f"p={p:.2e}")

# ---- Table 3 (tex 389-401): mean CPSL (N) by predicted x ground truth, DN121; stars vs the 'Ref' (correct) cell ----
T3 = {"dn121": [("Normal", "Normal", 1.41, 693, "Ref"), ("Normal", "Indeterminate", 1.89, 35, "*"), ("Normal", "Severe", 1.91, 11, "*"),
                ("Severe", "Severe", 1.90, 68, "Ref"), ("Severe", "Normal", 2.20, 45, "*"), ("Severe", "Indeterminate", 2.08, 102, "Not Sig.")],
      "dino_dn121": [("Normal", "Normal", 1.16, 736, "Ref"), ("Normal", "Indeterminate", 1.73, 64, "*"), ("Normal", "Severe", 1.59, 17, "*"),
                     ("Severe", "Severe", 1.54, 72, "Ref"), ("Severe", "Normal", 1.93, 57, "*"), ("Severe", "Indeterminate", 1.79, 100, "*")]}
for m, cells in T3.items():
    d = load(m)
    for pc, gc, mean_r, n_r, st_r in cells:
        sel = lambda p_, g_: d.loc[(d["pred_class_eval"] == CLS[p_]) & (d["gt"] == CLS[g_]), "pred_set_eval_length"]
        s = sel(pc, gc)
        add(f"T3-{m}-{pc}x{gc}-mean", f"Table 3, {m}, pred {pc}, GT {gc}, mean CPSL", "mean CPSL", mean_r, s.mean(), "mean2", [m], "pandas groupby mean")
        add(f"T3-{m}-{pc}x{gc}-N", f"Table 3, {m}, pred {pc}, GT {gc}, N", "N images", n_r, len(s), "int", [m], "pandas count")
        if st_r != "Ref":
            p = sig(s, sel(pc, pc))
            add(f"T3-{m}-{pc}x{gc}-sig", f"Table 3, {m}, pred {pc}, GT {gc}, Sig. vs Ref", "t-test p<0.05 vs correct group", st_r, star(p), "star", [m],
                "scipy levene+ttest_ind", note=f"p={p:.4f}")

# ---- Table 4 (tex 448-458): SL vs SSL mean CPSL by predicted class, all architectures ----
T4 = {("DenseNet121", "Normal"): (1.44, 1.22, "*"), ("DenseNet121", "Indeterminate"): (2.61, 2.57, "Not Sig."), ("DenseNet121", "Severe"): (2.05, 1.75, "*"),
      ("ResNet50", "Normal"): (1.46, 1.21, "*"), ("ResNet50", "Indeterminate"): (2.87, 2.41, "*"), ("ResNet50", "Severe"): (1.77, 1.70, "Not Sig."),
      ("EffNetV2S", "Normal"): (1.32, 1.21, "*"), ("EffNetV2S", "Indeterminate"): (2.71, 2.74, "Not Sig."), ("EffNetV2S", "Severe"): (1.85, 2.04, "*")}
for (a, c), (sl_r, ssl_r, st_r) in T4.items():
    sl, ssl = load(ARCH[a]), load("dino_" + ARCH[a])
    x = ssl.loc[ssl["pred_class_eval"] == CLS[c], "pred_set_eval_length"]
    y = sl.loc[sl["pred_class_eval"] == CLS[c], "pred_set_eval_length"]
    p = sig(x, y)
    m = [ARCH[a], "dino_" + ARCH[a]]
    add(f"T4-{a}-{c}-SL", f"Table 4, {a} {c}, SL mean CPSL", "mean CPSL", sl_r, y.mean(), "mean2", m, "pandas groupby mean")
    add(f"T4-{a}-{c}-SSL", f"Table 4, {a} {c}, SSL mean CPSL", "mean CPSL", ssl_r, x.mean(), "mean2", m, "pandas groupby mean")
    add(f"T4-{a}-{c}-sig", f"Table 4, {a} {c}, Sig.", "t-test p<0.05 (uncorrected)", st_r, star(p), "star", m, "scipy levene+ttest_ind",
        note=f"p={p:.4f}; Bonferroni(/9) significant: {p < 0.05 / 9}; single-split result, see F-021")

# ---- Text numbers (tex 430-432): DN121 CPSL=1 counts and accuracy by CPSL ----
def acc(m, pc, L):
    d = load(m)
    s = d[(d["pred_class_eval"] == CLS[pc]) & (d["pred_set_eval_length"] == L)]
    return (s["gt"] == CLS[pc]).mean(), len(s)

add("TXT-430-N_SL", "Sec. 4.x text l.430, SL", "confident Normal predictions (CPSL=1), SL", 481, acc("dn121", "Normal", 1)[1], "int", ["dn121"], "pandas")
add("TXT-430-N_SSL", "Sec. 4.x text l.430, SSL", "confident Normal predictions (CPSL=1), SSL", 669, acc("dino_dn121", "Normal", 1)[1], "int", ["dino_dn121"], "pandas")
TXT = [("dn121", "Normal", 1, 97), ("dn121", "Normal", 2, 92), ("dn121", "Normal", 3, 81), ("dn121", "Severe", 1, 47), ("dn121", "Severe", 2, 30), ("dn121", "Severe", 3, 22),
       ("dino_dn121", "Normal", 1, 95), ("dino_dn121", "Normal", 2, 71), ("dino_dn121", "Normal", 3, 62), ("dino_dn121", "Severe", 1, 45), ("dino_dn121", "Severe", 2, 23)]
for m, pc, L, rep in TXT:
    a, n = acc(m, pc, L)
    add(f"TXT-432-{m}-{pc}-CPSL{L}", f"text l.432, {m}, predicted {pc}, CPSL={L}, accuracy %", "accuracy (%) by CPSL", rep, a, "pct", [m], "pandas",
        note=f"computed {100 * a:.2f}% (N={n})" + ("; see F-019" if round(100 * a) != rep else ""))

for i, r in enumerate(rows, 1):
    r["id"] = f"C-{i:03d}:{r['id']}"
out = dict(generated_by=SCRIPT, note="GENERATED FILE. Do not edit by hand; rerun the script.", claims=rows)
with open(ROOT + "/ledger/claims_ledger.yaml", "w") as f:
    yaml.safe_dump(out, f, sort_keys=False, width=140)

df = pd.DataFrame(rows)
print(df["status"].value_counts().to_string())
print("\nDISCREPANCIES:")
print(df[df["status"] != "verified"][["id", "reported_value", "reproduced_legacy", "notes"]].to_string(index=False))
