# test_resolvent.py

import numpy as np
import pytest

from RMT.linalg.eig import EigenSystem
from RMT.rmt.resolvent import Resolvent


# -------------------------------------------------------------------
# Helpers: scalar definitions and finite differences
# -------------------------------------------------------------------

def m_hat_scalar(z: float, eigvals: np.ndarray) -> float:
    """Direct definition of m_hat(z) = mean(1 / (λ_i - z))."""
    return np.mean(1.0 / (eigvals - z))


def m_tilde_scalar(z: float, eigvals: np.ndarray, c: float) -> float:
    """Direct definition of m_tilde(z) = (c - 1)/z + c * m_hat(z)."""
    return (c - 1.0) / z + c * m_hat_scalar(z, eigvals)

def xi_scalar(z: float, eigvals: np.ndarray, c: float) -> float:
    """
    Direct definition of xi(z) in terms of m_hat and c:

        xi(z) = (1 - z m_hat(z)) / (1/c - 1 + z m_hat(z)).
    """
    m = m_hat_scalar(z, eigvals)
    z = -z
    numerator = 1
    denominator = 1-c+c*z*m
    return numerator / denominator - 1


def finite_diff(f, z, order, h=1e-2):
    if order == 1:
        # 2nd-order central difference for first derivative
        return (f(z + h) - f(z - h)) / (2.0 * h)

    elif order == 2:
        # 2nd-order central difference for second derivative
        return (f(z + h) - 2.0 * f(z) + f(z - h)) / (h ** 2)

    elif order == 3:
        # 5-point stencil (O(h^4)) for third derivative
        return (
            -f(z - 2 * h)
            + 2.0 * f(z - h)
            - 2.0 * f(z + h)
            + f(z + 2.0 * h)
        ) / (2.0 * h ** 3)

    elif order == 4:
        # 5-point stencil (O(h^4)) for fourth derivative
        return (
            f(z - 2 * h)
            - 4.0 * f(z - h)
            + 6.0 * f(z)
            - 4.0 * f(z + h)
            + f(z + 2.0 * h)
        ) / (h ** 4)

    else:
        raise ValueError("Only orders 1, 2, 3, 4 are supported.")



# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def eigensystem_under():
    """
    Simple under-parameterized EigenSystem for testing.

    T > P so that EigenSystem is in the 'under' regime.
    """
    rng = np.random.default_rng(0)
    T, P = 10, 5
    X = rng.standard_normal((T, P))
    return EigenSystem(X)


@pytest.fixture
def resolvent(eigensystem_under):
    """
    Build a Resolvent instance on a small grid of negative z-values.

    We use negative z so the internal flipping logic (if any) is consistent,
    but for all comparisons we use resolvent.z_grid (the internal grid),
    not the raw input.
    """
    z_input = -np.linspace(1, 10.0, 6)  # all negative, away from eigenvalues
    return Resolvent(eigensystem_under, z_input)


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Tests for m, m_tilde and its derivatives
# -------------------------------------------------------------------
def test_m_hat_matches_definition(resolvent, eigensystem_under):
    """
    Check that m_hat computed by Resolvent matches the direct definition:
        m_hat(z) = mean( 1 / (λ_i - z) ).
    """
    eigvals = eigensystem_under.get_eigenvalues()
    z_grid_internal = resolvent.z_grid  # use the internal z (after any sign flips)

    m_direct = np.array(
        [m_hat_scalar(z, eigvals) for z in z_grid_internal]
    )
    m_res = resolvent.get_m_hat()

    np.testing.assert_allclose(m_res, m_direct, rtol=1e-10, atol=1e-12)


def test_m_tilde_matches_definition(resolvent, eigensystem_under):
    """
    Check that m_tilde computed by Resolvent matches:
        m_tilde(z) = (c - 1)/z + c * m_hat(z).
    """
    eigvals = eigensystem_under.get_eigenvalues()
    z_grid_internal = resolvent.z_grid
    c = resolvent.c_

    mtilde_direct = np.array(
        [m_tilde_scalar(z, eigvals, c) for z in z_grid_internal]
    )
    mtilde_res = resolvent.get_m_tilde()

    np.testing.assert_allclose(mtilde_res, mtilde_direct, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_m_hat_derivatives_match_finite_difference(resolvent, eigensystem_under, order):
    """
    Compare m_hat^{(k)}(z) from Resolvent with finite-difference approximations
    of m_hat(z) for k = 1, 2, 3.
    """
    eigvals = eigensystem_under.get_eigenvalues()
    z_grid_internal = resolvent.z_grid
    m_hat_drv_res = resolvent.get_m_hat_drv(order)

    approx = np.array([
        finite_diff(lambda zz: m_hat_scalar(zz, eigvals), z, order=order)
        for z in z_grid_internal
    ])

    # Derivatives can be numerically sensitive → slightly looser tolerance.
    np.testing.assert_allclose(m_hat_drv_res, approx, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_m_tilde_derivatives_match_finite_difference(resolvent, eigensystem_under, order):
    """
    Compare m_tilde^{(k)}(z) from Resolvent with finite-difference approximations
    of m_tilde(z) for k = 1, 2, 3.
    """
    eigvals = eigensystem_under.get_eigenvalues()
    z_grid_internal = resolvent.z_grid
    c = resolvent.c_
    mtilde_drv_res = resolvent.get_m_tilde_drv(order)

    approx = np.array([
        finite_diff(lambda zz: m_tilde_scalar(zz, eigvals, c), z, order=order)
        for z in z_grid_internal
    ])

    np.testing.assert_allclose(mtilde_drv_res, approx, rtol=1e-3, atol=1e-3)

# -------------------------------------------------------------------
# Tests for xi and its derivatives
# -------------------------------------------------------------------
def test_xi_matches_definition(resolvent, eigensystem_under):
    """
    Check that xi(z) computed by Resolvent matches its direct definition
    in terms of m_hat(z) and c.
    """
    eigvals = eigensystem_under.get_eigenvalues()
    z_grid_internal = resolvent.z_grid
    c = resolvent.c_

    # Direct scalar xi(z) using the same z and eigenvalues
    xi_direct = np.array([
        xi_scalar(z, eigvals, c) for z in z_grid_internal
    ])

    xi_res = resolvent.get_xi()

    np.testing.assert_allclose(xi_res, xi_direct, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_xi_derivatives_match_finite_difference(resolvent, eigensystem_under, order):
    """
    Compare xi^{(k)}(z) from Resolvent with finite-difference approximations
    of xi(z) for k = 1, 2, 3.
    """
    eigvals = eigensystem_under.get_eigenvalues()
    z_grid_internal = resolvent.z_grid
    c = resolvent.c_

    xi_drv_res = resolvent.get_xi_drv(order=order)

    # Finite-difference approximation of derivative of scalar xi(z)
    approx = np.array([
        finite_diff(lambda zz: xi_scalar(zz, eigvals, c), z, order=order)
        for z in z_grid_internal
    ])

    # Third derivatives are numerically delicate; allow slightly looser tol
    np.testing.assert_allclose((-1)**order*xi_drv_res, approx, rtol=1e-3, atol=1e-3)
