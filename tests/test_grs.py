import numpy as np
import pytest

from RMT.linalg.eig import EigenSystem
from RMT.rmt.resolvent import Resolvent
from RMT.rmt.ZStar import ZStar
from RMT.rmt.grs import GRS


# ----------------------------------------------------------------------
# Fixtures: under- and over-parameterized settings
# ----------------------------------------------------------------------

@pytest.fixture
def eigensystem_under():
    """
    Under-parameterized case: P <= T.
    """
    rng = np.random.default_rng(7)
    T, P = 10, 5
    X = rng.normal(size=(T, P))
    return EigenSystem(X)


@pytest.fixture
def eigensystem_over():
    """
    Over-parameterized case: P > T.
    """
    rng = np.random.default_rng(315)
    T, P = 5, 10
    X = rng.normal(size=(T, P))
    return EigenSystem(X)


@pytest.fixture
def grs_under(eigensystem_under):
    """
    Full GRS stack in the under-parameterized regime.
    """
    # Positive z-grid for the theory
    z_grid = np.linspace(0.1, 0.5, 6)
    resolvent = Resolvent(eigensystem_under, z_grid=z_grid)
    zstar = ZStar(resolvent)
    return GRS(eig_system=eigensystem_under, resolvent=resolvent, Zstar=zstar)


@pytest.fixture
def grs_over(eigensystem_over):
    """
    Full GRS stack in the over-parameterized regime.
    """
    z_grid = np.linspace(0.1, 0.5, 6)
    resolvent = Resolvent(eigensystem_over, z_grid=z_grid)
    zstar = ZStar(resolvent)
    return GRS(eig_system=eigensystem_over, resolvent=resolvent, Zstar=zstar)


# ----------------------------------------------------------------------
# Helpers: scalar θ(z) and finite differences
# ----------------------------------------------------------------------

def theta_scalar(z: float, grs: GRS) -> float:
    """
    Scalar definition of θ(z) matching the formulas inside GRS.

    Under-parameterized (P <= T):
        θ_before(z) = Σ v_i^2 / (z + d_i)

    Over-parameterized (P > T):
        θ_before(z) = (1/T) Σ u_i^2 s_i / (z + s_i)

    θ(z) = θ_before(z) / (1 - θ_before(z))
    """
    eigsys = grs._eigsys  # use the same EigenSystem as GRS

    if eigsys._regime == "under":
        # eigenvalues of X^T X / T, shape (P,)
        eigvals = eigsys.get_eigenvalues()
        v = grs._v.flatten()
        theta_before = np.sum(v**2 / (z + eigvals))
    else:
        # non-zero eigenvalues of X X^T / T, shape (T,)
        s = eigsys.get_non_zero_eigenvalues()
        u = grs._u.flatten()
        theta_before = np.sum(u**2 * s / (z + s)) / eigsys.t_

    return theta_before / (1.0 - theta_before)


def finite_diff(f, z: float, order: int, h: float = 1e-4) -> float:
    """
    Central finite-difference approximation for first/second/third derivatives
    of a scalar function f at point z.
    """
    if order == 1:
        return (f(z + h) - f(z - h)) / (2.0 * h)
    elif order == 2:
        return (f(z + h) - 2.0 * f(z) + f(z - h)) / (h ** 2)
    elif order == 3:
        # 5-point stencil for third derivative
        return (
            -f(z - 2 * h)
            + 2.0 * f(z - h)
            - 2.0 * f(z + h)
            + f(z + 2.0 * h)
        ) / (2.0 * h ** 3)
    else:
        raise ValueError("Only orders 1, 2, 3 are supported.")


# ----------------------------------------------------------------------
# Tests: θ(z) matches scalar definition
# ----------------------------------------------------------------------

@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_theta_matches_scalar_definition(grs_fixture_name, request):
    """
    Check that _theta agrees with the scalar θ(z) definition
    for all points in the grid, in both regimes.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    z_grid = grs.z_grid  # positive z
    theta_vec = grs._theta.reshape(-1)  # (M,)

    theta_scalar_vals = np.array([theta_scalar(z, grs) for z in z_grid])

    np.testing.assert_allclose(theta_vec, theta_scalar_vals, rtol=1e-6, atol=1e-8)


# ----------------------------------------------------------------------
# Tests: θ'(z) and θ''(z) via finite differences
# ----------------------------------------------------------------------

@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_theta_prime_matches_finite_difference(grs_fixture_name, request):
    """
    Check that _theta_drv1 matches a finite-difference
    approximation of dθ/dz on the grid.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    z_grid = grs.z_grid
    theta_prime_vec = grs._theta_drv1.reshape(-1)

    approx_prime = np.array([
        finite_diff(lambda zz: theta_scalar(zz, grs), z, order=1)
        for z in z_grid
    ])

    np.testing.assert_allclose(theta_prime_vec, approx_prime, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_theta_second_matches_finite_difference(grs_fixture_name, request):
    """
    Check that _theta_drv2 matches a finite-difference
    approximation of d²θ/dz² on the grid.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    z_grid = grs.z_grid
    theta_second_vec = grs._theta_drv2.reshape(-1)

    approx_second = np.array([
        finite_diff(lambda zz: theta_scalar(zz, grs), z, order=2)
        for z in z_grid
    ])

    # Second derivatives are more sensitive; use a slightly looser tolerance.
    np.testing.assert_allclose(theta_second_vec, approx_second, rtol=1e-3, atol=1e-3)
    
@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_theta_third_matches_finite_difference(grs_fixture_name, request):
    """
    Check that _theta_drv3 matches a finite-difference
    approximation of d³θ/dz³ on the grid.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    z_grid = grs.z_grid
    theta_third_vec = grs._theta_drv3.reshape(-1)

    approx_third = np.array([
        finite_diff(lambda zz: theta_scalar(zz, grs), z, order=3)
        for z in z_grid
    ])

    # Third derivatives are more sensitive; use a slightly looser tolerance.
    np.testing.assert_allclose(theta_third_vec, approx_third, rtol=1e-3, atol=1e-3)


# ----------------------------------------------------------------------
# Tests: a_min and u_J gradients via finite differences
# ----------------------------------------------------------------------

def a_min_from_B(B_vals: np.ndarray, Zhat: np.ndarray, c: float, J: int = 2, eps: float = 1e-14) -> np.ndarray:
    """
    Compute a_min given B values directly (for finite-difference testing).

    Parameters
    ----------
    B_vals : ndarray of shape (n_z, J+1)
        Values of B_0, ..., B_J on the grid.
    Zhat : ndarray of shape (n_z,)
        Z-star values on the grid.
    c : float
        tr(Sigma) / P.
    J : int
        Order (default 2).
    eps : float
        Numerical stabilizer.

    Returns
    -------
    a_min : ndarray of shape (n_z,)
    """
    D = np.sum(B_vals[:, :J+1], axis=1)
    BJ = B_vals[:, J]
    D_safe = np.where(np.abs(D) < eps, np.sign(D) * eps + eps, D)
    Q = (BJ * Zhat) ** 2 + (BJ * Zhat) * D * c
    sqrtQ = np.sqrt(np.maximum(Q, eps))
    a_min = (-(BJ * Zhat) + sqrtQ) / D_safe
    return a_min


def u_J_from_B(B_vals: np.ndarray, Zhat: np.ndarray, c: float, J: int = 2, eps: float = 1e-14) -> np.ndarray:
    """
    Compute u_J given B values directly (for finite-difference testing).

    Parameters
    ----------
    B_vals : ndarray of shape (n_z, J+1)
        Values of B_0, ..., B_J on the grid.
    Zhat : ndarray of shape (n_z,)
        Z-star values on the grid.
    c : float
        tr(Sigma) / P.
    J : int
        Order (default 2).
    eps : float
        Numerical stabilizer.

    Returns
    -------
    u_J : ndarray of shape (n_z,)
    """
    BJ = B_vals[:, J]
    if J == 0:
        D_Jm1 = np.zeros(B_vals.shape[0], dtype=float)
    else:
        D_Jm1 = np.sum(B_vals[:, :J], axis=1)

    a = a_min_from_B(B_vals, Zhat, c, J, eps)
    a_safe = np.where(np.abs(a) < eps, np.sign(a) * eps + eps, a)

    numer = D_Jm1 + BJ * ((a_safe + Zhat) / a_safe)
    denom = 1.0 - (a_safe / c)
    denom_safe = np.where(np.abs(denom) < eps, np.sign(denom) * eps + eps, denom)

    return numer / denom_safe


def finite_diff_gradient(f, B_vals: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """
    Compute finite-difference gradient of f with respect to each B component.

    Parameters
    ----------
    f : callable
        Function taking B_vals and returning (n_z,) array.
    B_vals : ndarray of shape (n_z, J+1)
        Current B values.
    h : float
        Step size for finite differences.

    Returns
    -------
    grad : ndarray of shape (n_z, J+1)
        Finite-difference gradient.
    """
    n_z, n_B = B_vals.shape
    grad = np.zeros((n_z, n_B), dtype=float)

    for k in range(n_B):
        B_plus = B_vals.copy()
        B_minus = B_vals.copy()
        B_plus[:, k] += h
        B_minus[:, k] -= h
        grad[:, k] = (f(B_plus) - f(B_minus)) / (2.0 * h)

    return grad


def a_star_from_B(B_vals: np.ndarray, Zhat: np.ndarray, J: int, k: int, eps: float = 1e-14) -> np.ndarray:
    """
    Compute a_star^J(B; k) given B values directly (for finite-difference testing).

    Parameters
    ----------
    B_vals : ndarray of shape (n_z, J+1)
        Values of B_0, ..., B_J on the grid.
    Zhat : ndarray of shape (n_z,)
        Z-star values on the grid.
    J : int
        Order (1 or 2).
    k : int
        Power parameter.
    eps : float
        Numerical stabilizer.

    Returns
    -------
    a_star : ndarray of shape (n_z,)
    """
    BJ = B_vals[:, J]
    BJ_m1 = B_vals[:, J - 1]
    BJ_m1_safe = np.where(np.abs(BJ_m1) < eps, np.sign(BJ_m1) * eps + eps, BJ_m1)
    r = BJ / BJ_m1_safe
    numer = Zhat * (1.0 - r)
    denom = r - r ** (k + 1)
    denom_safe = np.where(np.abs(denom) < eps, np.sign(denom) * eps + eps, denom)
    return numer / denom_safe


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
@pytest.mark.parametrize("J", [1, 2])
@pytest.mark.parametrize("k", [2, 4, 8, 10])
def test_a_star_drv_matches_finite_difference(grs_fixture_name, J, k, request):
    """
    Check that the analytical gradient of a_star^J(B; k) matches
    finite-difference approximation with respect to B_0, ..., B_J.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)

    grad_analytical = grs._a_star_drv_data[J][k]  # (n_z, J+1)

    B_vals = np.column_stack([grs._cb0, grs._cb1, grs._cb2])[:, :J + 1]  # (n_z, J+1)
    Zhat = grs._Zstar.get_z_star().reshape(-1)

    def f(B):
        return a_star_from_B(B, Zhat, J=J, k=k)

    grad_fd = finite_diff_gradient(f, B_vals, h=1e-6)

    np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4, atol=1e-6)


def u_star_from_B(B_vals: np.ndarray, Zhat: np.ndarray, c: float, J: int, k: int, eps: float = 1e-14) -> np.ndarray:
    """
    Compute u_star^J(B; k) given B values directly (for finite-difference testing).

    Parameters
    ----------
    B_vals : ndarray of shape (n_z, J+1)
        Values of B_0, ..., B_J on the grid.
    Zhat : ndarray of shape (n_z,)
        Z-star values on the grid.
    c : float
        tr(Sigma) / P.
    J : int
        Order (1 or 2).
    k : int
        Power parameter.
    eps : float
        Numerical stabilizer.

    Returns
    -------
    u_star : ndarray of shape (n_z,)
    """
    a = a_star_from_B(B_vals, Zhat, J=J, k=k, eps=eps)
    a_safe = np.where(np.abs(a) < eps, np.sign(a) * eps + eps, a)

    BJ = B_vals[:, J]
    D_Jm1 = np.sum(B_vals[:, :J], axis=1)

    numer = D_Jm1 + BJ * ((a_safe + Zhat) / a_safe)
    denom = 1.0 - (a_safe / c)
    denom_safe = np.where(np.abs(denom) < eps, np.sign(denom) * eps + eps, denom)
    return numer / denom_safe


@pytest.mark.parametrize("grs_fixture_name", ["grs_under"])
@pytest.mark.parametrize("J", [1])
@pytest.mark.parametrize("k", [2, 4, 8, 10])
def test_u_star_drv_matches_finite_difference(grs_fixture_name, J, k, request):
    """
    Check that the analytical gradient of u_star^J(B; k) matches
    finite-difference approximation with respect to B_0, ..., B_J.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)

    grad_analytical = grs._u_J_star_drv[J][k]  # (n_z, J+1)

    B_vals = np.column_stack([grs._cb0, grs._cb1, grs._cb2])[:, :J + 1]  # (n_z, J+1)
    Zhat = grs._Zstar.get_z_star().reshape(-1)
    trSigma = np.sum(grs._eigsys.get_eigenvalues())
    P = grs._p
    c = trSigma / float(P)

    def f(B):
        return u_star_from_B(B, Zhat, c, J=J, k=k)

    grad_fd = finite_diff_gradient(f, B_vals, h=1e-6)

    np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_a_min_gradient_matches_finite_difference(grs_fixture_name, request):
    """
    Check that the analytical gradient of a_min matches finite-difference
    approximation with respect to B_0, B_1, B_2.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)

    # Get computed gradient
    grad_analytical = grs.get_a_min_drv()  # (n_z, 3)

    # Get the B values and parameters needed for finite-diff
    B_vals = np.column_stack([grs._cb0, grs._cb1, grs._cb2])  # (n_z, 3)
    Zhat = grs._Zstar.get_z_star().reshape(-1)
    trSigma = np.sum(grs._eigsys.get_eigenvalues())
    P = grs._p
    c = trSigma / float(P)

    # Compute finite-difference gradient
    def f(B):
        return a_min_from_B(B, Zhat, c, J=2)

    grad_fd = finite_diff_gradient(f, B_vals, h=1e-6)

    np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_u_J_gradient_matches_finite_difference(grs_fixture_name, request):
    """
    Check that the analytical gradient of u_J matches finite-difference
    approximation with respect to B_0, B_1, B_2.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)

    # Get computed gradient
    grad_analytical = grs.get_u_J_drv()  # (n_z, 3)

    # Get the B values and parameters needed for finite-diff
    B_vals = np.column_stack([grs._cb0, grs._cb1, grs._cb2])  # (n_z, 3)
    Zhat = grs._Zstar.get_z_star().reshape(-1)
    trSigma = np.sum(grs._eigsys.get_eigenvalues())
    P = grs._p
    c = trSigma / float(P)

    # Compute finite-difference gradient
    def f(B):
        return u_J_from_B(B, Zhat, c, J=2)

    grad_fd = finite_diff_gradient(f, B_vals, h=1e-6)

    np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4, atol=1e-6)


# ----------------------------------------------------------------------
# Helpers: scalar A(z), B(z), C(z) and off-diagonal Xi_00 formula
# ----------------------------------------------------------------------

def m_scalar(z: float, eigvals: np.ndarray) -> float:
    """Stieltjes transform at positive z: mean(1/(eigvals + z))."""
    return np.mean(1.0 / (eigvals + z))


def xi_scalar_fn(z: float, eigvals: np.ndarray, c: float) -> float:
    m = m_scalar(z, eigvals)
    return (1.0 - z * m) / (1.0 / c - 1.0 + z * m)


def zstar_scalar_fn(z: float, eigvals: np.ndarray, c: float) -> float:
    m = m_scalar(z, eigvals)
    return z / (1.0 - c + c * z * m)


def A_scalar(z: float, grs: GRS) -> float:
    eigvals = grs._eigsys.get_eigenvalues()
    c = grs._p / grs._t
    return 4.0 * z * (theta_scalar(z, grs) - xi_scalar_fn(z, eigvals, c))


def B_scalar(z: float, grs: GRS) -> float:
    eigvals = grs._eigsys.get_eigenvalues()
    c = grs._p / grs._t
    return 2.0 * z * xi_scalar_fn(z, eigvals, c)


def C_scalar(z: float, grs: GRS) -> float:
    eigvals = grs._eigsys.get_eigenvalues()
    c = grs._p / grs._t
    return 16.0 * zstar_scalar_fn(z, eigvals, c)


def xi_00_offdiag(z1: float, z2: float, grs: GRS) -> float:
    """Closed-form off-diagonal Xi_00(z1, z2) for z1 != z2."""
    delta_z = z2 - z1
    delta_A = A_scalar(z2, grs) - A_scalar(z1, grs)
    delta_B = B_scalar(z2, grs) - B_scalar(z1, grs)
    delta_C = C_scalar(z2, grs) - C_scalar(z1, grs)
    return (
        2.0 * delta_A ** 2 / (delta_C * delta_z)
        + delta_A / delta_z
        + delta_B / delta_z
    )


# ----------------------------------------------------------------------
# Tests: Xi_multi_00, Xi_multi_10, Xi_multi_01
# ----------------------------------------------------------------------

@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_00_offdiag_matches_formula(grs_fixture_name, request):
    """
    Off-diagonal elements of Xi_multi_00 should match the closed-form
    scalar formula evaluated at each pair (z_i, z_j).
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_00 = grs._compute_Xi_multi_00()
    z_grid = grs.z_grid

    for i in range(len(z_grid)):
        for j in range(len(z_grid)):
            if i == j:
                continue
            expected = xi_00_offdiag(z_grid[i], z_grid[j], grs)
            np.testing.assert_allclose(
                xi_00[i, j], expected, rtol=1e-5, atol=1e-8,
                err_msg=f"Xi_multi_00[{i},{j}] mismatch"
            )


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_10_offdiag_matches_finite_difference(grs_fixture_name, request):
    """
    Off-diagonal elements of Xi_multi_10 should match the finite-difference
    derivative of Xi_00(z1, z2) w.r.t. z1, holding z2 fixed.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_10 = grs._compute_Xi_multi_10()
    z_grid = grs.z_grid
    h = 1e-5

    for i in range(len(z_grid)):
        for j in range(len(z_grid)):
            if i == j:
                continue
            z1, z2 = z_grid[i], z_grid[j]
            fd = (xi_00_offdiag(z1 + h, z2, grs) - xi_00_offdiag(z1 - h, z2, grs)) / (2.0 * h)
            np.testing.assert_allclose(
                xi_10[i, j], fd, rtol=1e-4, atol=1e-6,
                err_msg=f"Xi_multi_10[{i},{j}] mismatch"
            )


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_01_offdiag_matches_finite_difference(grs_fixture_name, request):
    """
    Off-diagonal elements of Xi_multi_01 should match the finite-difference
    derivative of Xi_00(z1, z2) w.r.t. z2, holding z1 fixed.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_01 = grs._compute_Xi_multi_01()
    z_grid = grs.z_grid
    h = 1e-5

    for i in range(len(z_grid)):
        for j in range(len(z_grid)):
            if i == j:
                continue
            z1, z2 = z_grid[i], z_grid[j]
            fd = (xi_00_offdiag(z1, z2 + h, grs) - xi_00_offdiag(z1, z2 - h, grs)) / (2.0 * h)
            np.testing.assert_allclose(
                xi_01[i, j], fd, rtol=1e-4, atol=1e-6,
                err_msg=f"Xi_multi_01[{i},{j}] mismatch"
            )


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_01_is_transpose_of_10(grs_fixture_name, request):
    """
    Xi_multi_01 should equal the transpose of Xi_multi_10.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_10 = grs._compute_Xi_multi_10()
    xi_01 = grs._compute_Xi_multi_01()
    np.testing.assert_allclose(xi_01, xi_10.T, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_00_diagonal_matches_sigma00(grs_fixture_name, request):
    """
    Diagonal of Xi_multi_00 should equal sigma00(z), the z1=z2 limit.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_00 = grs._compute_Xi_multi_00()
    sigma00 = grs._compute_sigma00().ravel()
    np.testing.assert_allclose(np.diag(xi_00), sigma00, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_10_diagonal_matches_sigma01(grs_fixture_name, request):
    """
    Diagonal of Xi_multi_10 should equal sigma01(z), the z1=z2 limit
    of the derivative w.r.t. z1.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_10 = grs._compute_Xi_multi_10()
    sigma01 = grs._compute_sigma01().ravel()
    np.testing.assert_allclose(np.diag(xi_10), sigma01, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_01_diagonal_matches_sigma01(grs_fixture_name, request):
    """
    Diagonal of Xi_multi_01 should equal sigma01(z), the z1=z2 limit
    of the derivative w.r.t. z2.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_01 = grs._compute_Xi_multi_01()
    sigma01 = grs._compute_sigma01().ravel()
    np.testing.assert_allclose(np.diag(xi_01), sigma01, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_11_diagonal_matches_sigma11(grs_fixture_name, request):
    """
    Diagonal of Xi_multi_11 should equal sigma11(z), the z1=z2 limit
    of the mixed derivative w.r.t. z1 and z2.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_11 = grs._compute_Xi_multi_11()
    sigma11 = grs._compute_sigma11().ravel()
    np.testing.assert_allclose(np.diag(xi_11), sigma11, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_20_diagonal_matches_sigma02(grs_fixture_name, request):
    """
    Diagonal of Xi_multi_20 should equal sigma02(z), the z1=z2 limit
    of the second derivative w.r.t. z1.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_20 = grs._compute_Xi_multi_20()
    sigma02 = grs._compute_sigma02().ravel()
    np.testing.assert_allclose(np.diag(xi_20), sigma02, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_20_offdiag_matches_finite_difference(grs_fixture_name, request):
    """
    Off-diagonal elements of Xi_multi_20 should match the finite-difference
    second derivative of Xi_00(z1, z2) w.r.t. z1.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_20 = grs._compute_Xi_multi_20()
    z_grid = grs.z_grid
    h = 1e-4

    for i in range(len(z_grid)):
        for j in range(len(z_grid)):
            if i == j:
                continue
            z1, z2 = z_grid[i], z_grid[j]
            fd = (
                xi_00_offdiag(z1 + h, z2, grs)
                - 2.0 * xi_00_offdiag(z1, z2, grs)
                + xi_00_offdiag(z1 - h, z2, grs)
            ) / h ** 2
            np.testing.assert_allclose(
                xi_20[i, j], fd, rtol=1e-3, atol=1e-5,
                err_msg=f"Xi_multi_20[{i},{j}] mismatch"
            )


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_02_diagonal_matches_sigma02(grs_fixture_name, request):
    """
    Diagonal of Xi_multi_02 should equal sigma02(z), the z1=z2 limit
    of the second derivative w.r.t. z2.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_02 = grs._compute_Xi_multi_02()
    sigma02 = grs._compute_sigma02().ravel()
    np.testing.assert_allclose(np.diag(xi_02), sigma02, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_02_offdiag_matches_finite_difference(grs_fixture_name, request):
    """
    Off-diagonal elements of Xi_multi_02 should match the finite-difference
    second derivative of Xi_00(z1, z2) w.r.t. z2.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_02 = grs._compute_Xi_multi_02()
    z_grid = grs.z_grid
    h = 1e-4

    for i in range(len(z_grid)):
        for j in range(len(z_grid)):
            if i == j:
                continue
            z1, z2 = z_grid[i], z_grid[j]
            fd = (
                xi_00_offdiag(z1, z2 + h, grs)
                - 2.0 * xi_00_offdiag(z1, z2, grs)
                + xi_00_offdiag(z1, z2 - h, grs)
            ) / h ** 2
            np.testing.assert_allclose(
                xi_02[i, j], fd, rtol=1e-3, atol=1e-5,
                err_msg=f"Xi_multi_02[{i},{j}] mismatch"
            )


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_02_is_transpose_of_20(grs_fixture_name, request):
    """
    Xi_multi_02 should equal the transpose of Xi_multi_20.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_20 = grs._compute_Xi_multi_20()
    xi_02 = grs._compute_Xi_multi_02()
    np.testing.assert_allclose(xi_02, xi_20.T, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_21_diagonal_matches_sigma12(grs_fixture_name, request):
    """
    Diagonal of Xi_multi_21 should equal sigma12(z), the z1=z2 limit
    of d³Xi_00/dz1²dz2.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_21 = grs._compute_Xi_multi_21()
    sigma12 = grs._compute_sigma12().ravel()
    np.testing.assert_allclose(np.diag(xi_21), sigma12, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_21_offdiag_matches_finite_difference(grs_fixture_name, request):
    """
    Off-diagonal elements of Xi_multi_21 should match d³Xi_00(z1,z2)/dz1²dz2
    computed via finite differences.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_21 = grs._compute_Xi_multi_21()
    z_grid = grs.z_grid
    h = 1e-3

    def d2_dz1(z1, z2):
        return (
            xi_00_offdiag(z1 + h, z2, grs)
            - 2.0 * xi_00_offdiag(z1, z2, grs)
            + xi_00_offdiag(z1 - h, z2, grs)
        ) / h ** 2

    for i in range(len(z_grid)):
        for j in range(len(z_grid)):
            if i == j:
                continue
            z1, z2 = z_grid[i], z_grid[j]
            fd = (d2_dz1(z1, z2 + h) - d2_dz1(z1, z2 - h)) / (2.0 * h)
            np.testing.assert_allclose(
                xi_21[i, j], fd, rtol=1e-2, atol=1e-4,
                err_msg=f"Xi_multi_21[{i},{j}] mismatch"
            )


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_12_diagonal_matches_sigma12(grs_fixture_name, request):
    """
    Diagonal of Xi_multi_12 should equal sigma12(z), the z1=z2 limit
    of d³Xi_00/dz1dz2².
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_12 = grs._compute_Xi_multi_12()
    sigma12 = grs._compute_sigma12().ravel()
    np.testing.assert_allclose(np.diag(xi_12), sigma12, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_12_offdiag_matches_finite_difference(grs_fixture_name, request):
    """
    Off-diagonal elements of Xi_multi_12 should match d³Xi_00(z1,z2)/dz1dz2²
    computed via finite differences.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_12 = grs._compute_Xi_multi_12()
    z_grid = grs.z_grid
    h = 1e-3

    def d2_dz2(z1, z2):
        return (
            xi_00_offdiag(z1, z2 + h, grs)
            - 2.0 * xi_00_offdiag(z1, z2, grs)
            + xi_00_offdiag(z1, z2 - h, grs)
        ) / h ** 2

    for i in range(len(z_grid)):
        for j in range(len(z_grid)):
            if i == j:
                continue
            z1, z2 = z_grid[i], z_grid[j]
            fd = (d2_dz2(z1 + h, z2) - d2_dz2(z1 - h, z2)) / (2.0 * h)
            np.testing.assert_allclose(
                xi_12[i, j], fd, rtol=1e-2, atol=1e-4,
                err_msg=f"Xi_multi_12[{i},{j}] mismatch"
            )


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_12_is_transpose_of_21(grs_fixture_name, request):
    """
    Xi_multi_12 should equal the transpose of Xi_multi_21.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_21 = grs._compute_Xi_multi_21()
    xi_12 = grs._compute_Xi_multi_12()
    np.testing.assert_allclose(xi_12, xi_21.T, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_22_diagonal_matches_sigma22(grs_fixture_name, request):
    """
    Diagonal of Xi_multi_22 should equal sigma22(z), the z1=z2 limit
    of d⁴Xi_00/dz1²dz2².
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_22 = grs._compute_Xi_multi_22()
    sigma22 = grs._compute_sigma22().ravel()
    np.testing.assert_allclose(np.diag(xi_22), sigma22, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_22_offdiag_matches_finite_difference(grs_fixture_name, request):
    """
    Off-diagonal elements of Xi_multi_22 should match d⁴Xi_00(z1,z2)/dz1²dz2²
    computed via finite differences.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_22 = grs._compute_Xi_multi_22()
    z_grid = grs.z_grid
    h = 1e-3

    def d2_dz1(z1, z2):
        return (
            xi_00_offdiag(z1 + h, z2, grs)
            - 2.0 * xi_00_offdiag(z1, z2, grs)
            + xi_00_offdiag(z1 - h, z2, grs)
        ) / h ** 2

    def d2_dz1_d2_dz2(z1, z2):
        return (
            d2_dz1(z1, z2 + h)
            - 2.0 * d2_dz1(z1, z2)
            + d2_dz1(z1, z2 - h)
        ) / h ** 2

    for i in range(len(z_grid)):
        for j in range(len(z_grid)):
            if i == j:
                continue
            z1, z2 = z_grid[i], z_grid[j]
            fd = d2_dz1_d2_dz2(z1, z2)
            np.testing.assert_allclose(
                xi_22[i, j], fd, rtol=1e-2, atol=1e-3,
                err_msg=f"Xi_multi_22[{i},{j}] mismatch"
            )


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_22_is_symmetric(grs_fixture_name, request):
    """
    Xi_multi_22 should be symmetric (transpose of itself).
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_22 = grs._compute_Xi_multi_22()
    np.testing.assert_allclose(xi_22, xi_22.T, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("grs_fixture_name", ["grs_under", "grs_over"])
def test_xi_multi_11_offdiag_matches_finite_difference(grs_fixture_name, request):
    """
    Off-diagonal elements of Xi_multi_11 should match the mixed finite-difference
    derivative ∂²Xi_00(z1,z2)/∂z1∂z2, holding z2 and z1 respectively fixed.
    """
    grs: GRS = request.getfixturevalue(grs_fixture_name)
    xi_11 = grs._compute_Xi_multi_11()
    z_grid = grs.z_grid
    h = 1e-4

    for i in range(len(z_grid)):
        for j in range(len(z_grid)):
            if i == j:
                continue
            z1, z2 = z_grid[i], z_grid[j]
            # Central mixed partial: [f(z1+h,z2+h) - f(z1+h,z2-h) - f(z1-h,z2+h) + f(z1-h,z2-h)] / (4h²)
            fd = (
                xi_00_offdiag(z1 + h, z2 + h, grs)
                - xi_00_offdiag(z1 + h, z2 - h, grs)
                - xi_00_offdiag(z1 - h, z2 + h, grs)
                + xi_00_offdiag(z1 - h, z2 - h, grs)
            ) / (4.0 * h ** 2)
            np.testing.assert_allclose(
                xi_11[i, j], fd, rtol=1e-4, atol=1e-6,
                err_msg=f"Xi_multi_11[{i},{j}] mismatch"
            )
