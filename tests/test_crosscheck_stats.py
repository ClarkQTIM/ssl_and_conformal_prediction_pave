"""Cross-implementation checks on synthetic data (no patient data). Evidence for F-015."""
import numpy as np
import pytest
from scipy.stats import norm
from MLstatkit.stats import Delong_test


def _psi(pos, neg):
    d = pos[:, None] - neg[None, :]
    return (d > 0) + 0.5 * (d == 0)


def _independent_delong(y, a, b):
    """Textbook O(mn) DeLong: placement values from pairwise comparisons."""
    pa, na, pb, nb = a[y == 1], a[y == 0], b[y == 1], b[y == 0]
    m, n = len(pa), len(na)
    Ma, Mb = _psi(pa, na), _psi(pb, nb)
    A = np.array([Ma.mean(), Mb.mean()])
    S = np.cov(np.vstack([Ma.mean(1), Mb.mean(1)])) / m + np.cov(np.vstack([Ma.mean(0), Mb.mean(0)])) / n
    z = (A[0] - A[1]) / np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    return z, 2 * norm.sf(abs(z))


def _scores(seed, n=300):
    r = np.random.default_rng(seed)
    y = (r.random(n) < 0.3).astype(int)
    a = np.round(r.normal(size=n) + 1.0 * y, 1)  # rounded so ties occur
    b = np.round(0.5 * a + r.normal(size=n) + 0.5 * y, 1)
    return y, a, b


@pytest.mark.parametrize("seed", range(5))
def test_mlstatkit_delong_p_value_matches_independent_implementation(seed):
    y, a, b = _scores(seed)
    z_i, p_i = _independent_delong(y, a, b)
    z_l, p_l = Delong_test(y, a, b)
    assert p_l == pytest.approx(p_i, rel=1e-8)
    assert abs(z_l) == pytest.approx(abs(z_i), rel=1e-8)


@pytest.mark.parametrize("seed", range(5))
def test_mlstatkit_delong_z_is_never_positive_so_it_carries_no_direction(seed):
    y, a, b = _scores(seed)
    assert Delong_test(y, a, b)[0] <= 0 and Delong_test(y, b, a)[0] <= 0
