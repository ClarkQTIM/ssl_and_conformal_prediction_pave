"""Executable evidence for AUDIT.md findings, run against the UNMODIFIED legacy code in src/.

xfail(strict=True) = documents a confirmed bug. When the bug is fixed the test XPASSes, strict mode
turns that into a failure, and the marker must then be removed. Run with the dr_ssl_sl environment:
    PYENV_VERSION=dr_ssl_sl pytest tests -q
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cuq_classification_algorithms import find_quantile  # noqa: E402


def _textbook_qhat(scores, alpha):
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return np.sort(scores)[k - 1]


@pytest.mark.xfail(strict=True, reason="F-010: find_quantile returns rank k+1, one above the textbook k-th smallest")
@pytest.mark.parametrize("n,alpha", [(10, 0.2), (20, 0.1), (50, 0.1), (100, 0.1), (337, 0.1)])
def test_find_quantile_is_textbook_kth_order_statistic(n, alpha):
    s = np.random.default_rng(n).random(n)
    assert find_quantile(s, alpha) == _textbook_qhat(s, alpha)


def test_find_quantile_gives_valid_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    n_cal, n_test, alpha, covs = 337, 2000, 0.10, []
    for _ in range(150):
        p = np.exp(rng.normal(size=(n_cal + n_test, 3)) * 2)
        p /= p.sum(1, keepdims=True)
        y = (p.cumsum(1) > rng.random((len(p), 1))).argmax(1)
        q = find_quantile(1 - p[np.arange(n_cal), y[:n_cal]], alpha)
        covs.append(((p[n_cal:] >= 1 - q)[np.arange(n_test), y[n_cal:]]).mean())
    assert np.mean(covs) >= 1 - alpha - 0.005


def _synthetic_predictions(n=400, per_patient=2):
    r = np.random.default_rng(1)
    P = r.dirichlet([1, 1, 1], size=n)
    df = pd.DataFrame({
        "MASKED_IMG_ID": [f"img{i}" for i in range(n)],
        "ID": [i // per_patient for i in range(n)],
        "gt": r.integers(0, 3, n),
        "pred": [str(list(p)) for p in P],
        "predicted_class": P.argmax(1),
        "soft_prediction": np.arange(n),  # marker equal to the row index
    })
    for m in range(3):
        df[f"mc_epistemic_{m}"] = [str(list(r.dirichlet([1, 1, 1]))) for _ in range(n)]
    return df


def _run_create_cuq_df(df):
    from create_cuq_csv import create_cuq_df
    return create_cuq_df("lac", df, "gt", "pred", "predicted_class", "MASKED_IMG_ID", "ID", 0.1, 0.2, False, 3, 3)


@pytest.mark.xfail(strict=True, reason="F-009: soft_prediction is attached in file order but rows are shuffled by train_test_split")
def test_create_cuq_df_soft_prediction_stays_with_its_image():
    out = _run_create_cuq_df(_synthetic_predictions())
    expected_marker = out["image"].str[3:].astype(int).values
    assert (out["soft_prediction"].values == expected_marker).all()


@pytest.mark.xfail(strict=True, reason="F-002: calibration/test split is random at image level, so patients appear in both")
def test_create_cuq_df_calibration_and_test_share_no_patients():
    df = _synthetic_predictions()
    out = _run_create_cuq_df(df)
    cal_images = set(df["MASKED_IMG_ID"]) - set(out["image"])
    cal_patients = set(df[df["MASKED_IMG_ID"].isin(cal_images)]["ID"])
    assert not (cal_patients & set(out["id"]))
