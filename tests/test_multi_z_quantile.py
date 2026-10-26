"""
Check the Monte Carlo equicoordinate quantile used by the multi-z bounds.

The library computes ``q`` with ``P(max_i Z_i <= q) = Phi(n_sigma)`` for
``Z ~ N(0, R)`` by Monte Carlo (1,000,000 draws). We compare it with the exact
value obtained by root-finding on the multivariate normal CDF, for the lower
and upper bounds, both orders (1 and 2), on under- and over-parameterised GRS
objects built from synthetic data. The tolerance atol=0.005 leaves a wide
margin over the Monte Carlo error.
"""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import norm, multivariate_normal

from RMT.linalg.eig import EigenSystem
from RMT.rmt.resolvent import Resolvent
from RMT.rmt.ZStar import ZStar
from RMT.rmt.grs import GRS


# ── Exact reference (brentq on the multivariate normal CDF) ────────────────

def _brentq_equicoordinate_quantile(R: np.ndarray, n_sigma: float) -> float:
    """Solve P(Z_1 <= c, ..., Z_K <= c) = Phi(n_sigma) for Z ~ N(0, R)."""
    K = R.shape[0]
    target = norm.cdf(n_sigma)
    mean = np.zeros(K)

    def cdf_at(c):
        return multivariate_normal.cdf(np.full(K, c), mean=mean, cov=R)

    # The root lies between the perfectly correlated and the independent cases
    c_lo = norm.ppf(target)
    c_hi = norm.ppf(target ** (1 / K)) + 1e-6
    return brentq(lambda c: cdf_at(c) - target, c_lo, c_hi)


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_grs(T: int, P: int, seed: int = 0) -> GRS:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, P))
    eigsys = EigenSystem(X)
    z_grid = np.array([0.1, 0.3, 0.5])
    resolvent = Resolvent(eigsys, z_grid=z_grid)
    zstar = ZStar(resolvent)
    return GRS(eig_system=eigsys, resolvent=resolvent, Zstar=zstar)


@pytest.fixture(params=[
    pytest.param((10, 5),  id="under"),
    pytest.param((5, 10),  id="over"),
    pytest.param((30, 20), id="under_larger"),
])
def grs(request):
    T, P = request.param
    return _make_grs(T, P)


# ── Tests ───────────────────────────────────────────────────────────────────

ATOL = 0.005  # MC accuracy ~0.001, safety factor ~5x


@pytest.mark.parametrize("order", [1, 2])
def test_lb_quantile_matches_brentq(grs, order):
    R = grs._R_sigma_H_lb[order]
    mc_val = grs._compute_equicoordinate_quantile_lb(order)
    ref_val = _brentq_equicoordinate_quantile(R, grs.n_sigma)
    assert abs(mc_val - ref_val) < ATOL, (
        f"LB quantile mismatch (order={order}): MC={mc_val:.6f}, brentq={ref_val:.6f}, "
        f"diff={abs(mc_val - ref_val):.6f}"
    )


@pytest.mark.parametrize("order", [1, 2])
def test_ub_quantile_matches_brentq(grs, order):
    R = grs._R_sigma_H_ub[order]
    mc_val = grs._compute_equicoordinate_quantile_ub(order)
    ref_val = _brentq_equicoordinate_quantile(R, grs.n_sigma)
    assert abs(mc_val - ref_val) < ATOL, (
        f"UB quantile mismatch (order={order}): MC={mc_val:.6f}, brentq={ref_val:.6f}, "
        f"diff={abs(mc_val - ref_val):.6f}"
    )


@pytest.mark.parametrize("order", [1, 2])
def test_sigma_J_multi_lb_matches_loop(grs, order):
    """Verify the einsum version of sigma_J_multi_lb matches the old loop."""
    grad_lb = grs._lower_bound_grad[order]
    A_Xi_A = grs._A_Xi_A_multi[order]
    M = grs.z_grid.shape[0]

    # Reference: explicit double loop
    ref = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            ref[i, j] = grad_lb[i] @ A_Xi_A[i, j] @ grad_lb[j].T

    result = grs._sigma_J_multi_lb[order]
    # _sigma_J_multi_lb has clip_eigenvalues applied; compare the raw einsum part
    raw = np.einsum('ia,ijab,jb->ij', grad_lb, A_Xi_A, grad_lb)
    np.testing.assert_allclose(raw, ref, rtol=1e-10,
                               err_msg=f"sigma_J_multi_lb einsum mismatch (order={order})")


@pytest.mark.parametrize("order", [1, 2])
def test_sigma_J_multi_ub_matches_loop(grs, order):
    """Verify the einsum version of sigma_J_multi_ub matches the old loop."""
    grad_ub = grs._upper_bound_data[order]["u_J_drv"]
    A_Xi_A = grs._A_Xi_A_multi[order]
    M = grs.z_grid.shape[0]

    ref = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            ref[i, j] = grad_ub[i] @ A_Xi_A[i, j] @ grad_ub[j].T

    raw = np.einsum('ia,ijab,jb->ij', grad_ub, A_Xi_A, grad_ub)
    np.testing.assert_allclose(raw, ref, rtol=1e-10,
                               err_msg=f"sigma_J_multi_ub einsum mismatch (order={order})")


@pytest.mark.parametrize("xi_fn, sigma_fn", [
    ("_compute_Xi_multi_00", "_compute_sigma00"),
    ("_compute_Xi_multi_10", "_compute_sigma01"),
    ("_compute_Xi_multi_01", "_compute_sigma01"),
    ("_compute_Xi_multi_11", "_compute_sigma11"),
    ("_compute_Xi_multi_20", "_compute_sigma02"),
    ("_compute_Xi_multi_02", "_compute_sigma02"),
    ("_compute_Xi_multi_21", "_compute_sigma12"),
    ("_compute_Xi_multi_12", "_compute_sigma12"),
    ("_compute_Xi_multi_22", "_compute_sigma22"),
])
def test_xi_diagonal_matches_sigma(grs, xi_fn, sigma_fn):
    """Verify that np.fill_diagonal fills the same values as the old per-element loop."""
    Xi = getattr(grs, xi_fn)()
    sigma_vals = getattr(grs, sigma_fn)()[0, :]  # shape (M,)
    np.testing.assert_allclose(
        np.diag(Xi), sigma_vals, rtol=1e-10,
        err_msg=f"Diagonal of {xi_fn} doesn't match {sigma_fn}"
    )
