"""
Tests for the response diagnostics get_response_concentration and get_r2_rank_corr.

Both must describe the components with the largest eigenvalues. Eigenvalues are
stored in ascending order, so the expected values are recomputed here from an
independent eigendecomposition, selecting the largest components explicitly.
"""
import numpy as np
import pytest
from scipy.stats import spearmanr

from RMT.linalg.eig import EigenSystem
from RMT.rmt.resolvent import Resolvent
from RMT.rmt.ZStar import ZStar
from RMT.rmt.grs import GRS


def _make_grs(T: int, P: int, seed: int = 0):
    X = np.random.default_rng(seed).normal(0.1, 1.0, size=(T, P))
    eigsys = EigenSystem(X)
    resolvent = Resolvent(eigsys, z_grid=np.array([0.1, 0.5]))
    return X, GRS(eig_system=eigsys, resolvent=resolvent, Zstar=ZStar(resolvent))


@pytest.mark.parametrize("T, P", [
    pytest.param(40, 10, id="under"),
    pytest.param(10, 40, id="over"),
])
def test_w1_is_weight_of_largest_component(T, P):
    X, grs = _make_grs(T, P)
    ones = np.ones(T)

    if P <= T:
        # w_i = v_i^2 / ||v||^2 with v = Q' mu_hat (eigenvectors of X'X / T)
        evals, evecs = np.linalg.eigh(X.T @ X / T)
        v = evecs.T @ (X.T @ ones / T)
        w = v ** 2 / np.sum(v ** 2)
    else:
        # w_i = u_i^2 / T with u = Q' 1 (eigenvectors of XX' / T)
        evals, evecs = np.linalg.eigh(X @ X.T / T)
        w = (evecs.T @ ones) ** 2 / T

    expected = w[np.argmax(evals)]
    assert grs.get_response_concentration()['w1'] == pytest.approx(expected, rel=1e-10)


# With very few components the rank correlation takes few distinct values,
# so small top_k could match the smallest components by chance.
@pytest.mark.parametrize("top_k", [10, 20])
def test_r2_rank_corr_uses_largest_components(top_k):
    T, P = 30, 60
    X, grs = _make_grs(T, P)

    evals, evecs = np.linalg.eigh(X @ X.T / T)
    u = evecs.T @ np.ones(T)
    top = np.argsort(evals)[::-1][:top_k]          # the top_k largest eigenvalues
    expected = spearmanr(evals[top], u[top] ** 2 / evals[top])[0]

    assert grs.get_r2_rank_corr(top_k=top_k) == pytest.approx(expected, rel=1e-10)
