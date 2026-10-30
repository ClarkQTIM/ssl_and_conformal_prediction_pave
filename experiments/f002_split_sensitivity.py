"""F-002: sensitivity of the paper's CPSL / AUC results to how calibration vs test is split.

Schemes (calibration = 20% in every case, alpha = 0.1, LAC, single-pass `pred` column, as in the paper):
  image   : random image-level split (the legacy behaviour; seed 0 == the paper's split)
  patient : random patient-level split on STUDY+ID (no woman in both calibration and test)

Gate: the legacy split (image, seed 0) must reproduce the saved `pred_set_eval_length` per image for all six
models exactly, otherwise the script aborts. Only aggregates are printed (no IDs). Uses the unmodified src/ code.

Run: PYENV_VERSION=dr_ssl_sl python experiments/f002_split_sensitivity.py [n_seeds]
"""
import ast
import os
import sys

import numpy as np
import pandas as pd
from MLstatkit.stats import Delong_test
from scipy.stats import levene, ttest_ind
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from cuq_classification_algorithms import cuq_pred_set_lac, get_calibration_scores_lac  # noqa: E402

ALPHA, PCT_CAL = 0.1, 0.2
MODELS = ["dn121", "dino_dn121", "rn50", "dino_rn50", "effnetv2s", "dino_effnetv2s"]
PAIRS = [("dn121", "dino_dn121"), ("rn50", "dino_rn50"), ("effnetv2s", "dino_effnetv2s")]
PRED_CSV = ROOT + "/csvs/inference_csvs/{m}_seed_test2_predictions.csv"
GT_CSV = ROOT + "/csvs/SEED_three_class_ground_truths.csv"
SAVED = ROOT + "/analysis/six_model_seed_test2/{m}/3/lac/alpha_010/mc_lac_run_alpha0.1.csv"
OUT = ROOT + "/experiments/outputs"


def load():
    gt = pd.read_csv(GT_CSV)
    gt = gt[gt["dataset"] == "test2"][["MASKED_IMG_ID", "CC_ST"]]
    data = {}
    for m in MODELS:
        p = pd.read_csv(PRED_CSV.format(m=m), usecols=["MASKED_IMG_ID", "STUDY", "ID", "pred", "predicted_class"])
        p = p.merge(gt, on="MASKED_IMG_ID")  # same merge as main.py
        data[m] = dict(
            img=p["MASKED_IMG_ID"].values,
            probs=np.array([ast.literal_eval(s) for s in p["pred"]]),
            pred_class=p["predicted_class"].values.astype(int),
            gt=p["CC_ST"].values.astype(int),
            group=(p["STUDY"].astype(str) + "_" + p["ID"].astype(str)).values,
        )
    ref = data[MODELS[0]]
    for m in MODELS:
        assert (data[m]["img"] == ref["img"]).all() and (data[m]["gt"] == ref["gt"]).all() and (data[m]["group"] == ref["group"]).all()
    return data


def lac(d, cal, test):
    scores = get_calibration_scores_lac(d["gt"][cal], d["probs"][cal])
    sets, _, covered, _, qhat = cuq_pred_set_lac(scores, ALPHA, d["probs"][test], d["gt"][test])
    return sets.sum(1), covered, qhat


def split(scheme, seed, n, group):
    idx = np.arange(n)
    if scheme == "image":
        test, cal = train_test_split(idx, test_size=PCT_CAL, random_state=seed)
    else:
        test, cal = next(GroupShuffleSplit(n_splits=1, test_size=PCT_CAL, random_state=seed).split(idx, groups=group))
    return cal, test


def gate(data):
    n = len(data[MODELS[0]]["img"])
    cal, test = split("image", 0, n, None)
    for m in MODELS:
        d = data[m]
        length, _, _ = lac(d, cal, test)
        s = pd.read_csv(SAVED.format(m=m), usecols=["image", "pred_set_eval_length", "pred_class_eval"]).set_index("image")
        mine = pd.Series(length, index=d["img"][test])
        assert set(s.index) == set(mine.index), f"{m}: test image set differs from saved output"
        assert (s.loc[mine.index, "pred_set_eval_length"].values == mine.values).all(), f"{m}: CPSL differs from saved output"
        assert (s.loc[mine.index, "pred_class_eval"].values == d["pred_class"][test]).all(), f"{m}: predicted class differs"
    print(f"[gate] legacy split reproduces saved CPSL for all {len(MODELS)} models exactly (n_cal={len(cal)}, n_test={len(test)})")


def run(data, scheme, seed):
    n = len(data[MODELS[0]]["img"])
    group = data[MODELS[0]]["group"]
    cal, test = split(scheme, seed, n, group)
    overlap = float(np.isin(group[test], group[cal]).mean())
    per_model, rows = {}, []
    for m in MODELS:
        d = data[m]
        length, covered, qhat = lac(d, cal, test)
        pc, g = d["pred_class"][test], d["gt"][test]
        per_model[m] = dict(length=length, pc=pc, g=g, probs=d["probs"][test])
        r = dict(scheme=scheme, seed=seed, model=m, n_cal=len(cal), n_test=len(test), test_img_with_cal_woman=overlap,
                 qhat=qhat, coverage=covered.mean(), mean_cpsl=length.mean(), frac_empty=(length == 0).mean(),
                 acc=(pc == g).mean())
        for c in range(3):
            r[f"cov_gt{c}"] = covered[g == c].mean() if (g == c).any() else np.nan
            r[f"cpsl_pred{c}"] = length[pc == c].mean() if (pc == c).any() else np.nan
            r[f"n_pred{c}"] = int((pc == c).sum())
        for c in (0, 2):
            sel = pc == c
            r[f"n_cpsl1_pred{c}"] = int((length[sel] == 1).sum())
            r[f"acc_cpsl1_pred{c}"] = (g[sel & (length == 1)] == c).mean() if (sel & (length == 1)).any() else np.nan
        rows.append(r)
    # Table-4 style SL vs SSL CPSL t-tests (repo logic: Levene -> Student/Welch) and Table-1 style DeLong
    t4, t1 = [], []
    for sl, ssl in PAIRS:
        a, b = per_model[ssl], per_model[sl]
        for c in range(3):
            x, y = a["length"][a["pc"] == c], b["length"][b["pc"] == c]
            ok = len(x) > 1 and len(y) > 1
            p = ttest_ind(x, y, equal_var=levene(x, y)[1] > 0.05)[1] if ok else np.nan
            t4.append(dict(scheme=scheme, seed=seed, arch=sl, pred_class=c, ssl_mean=x.mean(), sl_mean=y.mean(), p=p))
        for c in (0, 2):
            yb = (a["g"] == c).astype(int)
            _, p = Delong_test(yb, a["probs"][:, c], b["probs"][:, c])
            t1.append(dict(scheme=scheme, seed=seed, arch=sl, cls=c, auc_sl=roc_auc_score(yb, b["probs"][:, c]),
                           auc_ssl=roc_auc_score(yb, a["probs"][:, c]), p=p))
    return rows, t4, t1


def main(n_seeds):
    os.makedirs(OUT, exist_ok=True)
    data = load()
    gate(data)
    R, T4, T1 = [], [], []
    for scheme in ("image", "patient"):
        for seed in range(n_seeds):
            r, t4, t1 = run(data, scheme, seed)
            R += r; T4 += t4; T1 += t1
    R = pd.DataFrame(R)
    T4, T1 = pd.DataFrame(T4), pd.DataFrame(T1)
    R.to_csv(OUT + "/f002_per_model.csv", index=False)
    T4.to_csv(OUT + "/f002_table4_ttests.csv", index=False)
    T1.to_csv(OUT + "/f002_table1_delong.csv", index=False)
    print(f"[done] {n_seeds} seeds x 2 schemes; outputs in {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
