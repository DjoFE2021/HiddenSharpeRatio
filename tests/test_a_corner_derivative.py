"""
Tests for _compute_grad_H_u_a_corner.

By construction, a_corner(z) solves H_1^U(B(z); a) = H_1^L(B(z_0)), so
differentiating the identity in B gives a result that depends only on B at
the first grid point z_0 and is therefore z-independent. The function
broadcasts ``self._lower_bound_grad[1][0, :2]`` to shape (n_z, 2).
"""

import numpy as np
from RMT.rmt.final_stat import setup_grs
from RMT.utils.constants import Z_EFF_GRID


def _setup(seed):
    np.random.seed(seed)
    T, P = 60, 120
    R = np.random.randn(T, P) * 0.01
    z_grid = (P / T) * Z_EFF_GRID
    return setup_grs(R, z_grid)


def test_grad_H_u_a_corner_shape_and_caching():
    """Shape is (n_z, 2), result is cached on self, every row is identical."""
    grs = _setup(0)
    grad = grs._compute_grad_H_u_a_corner()
    n_z = grs.z_grid.shape[0]

    assert grad.shape == (n_z, 2), f"Expected ({n_z}, 2), got {grad.shape}"
    assert np.all(np.isfinite(grad)), f"Non-finite entries: {grad}"
    assert grs._grad_H_u_a_corner is grad, "Result not cached on self"

    # Every row should be identical (z-independent)
    assert np.allclose(grad, grad[0:1, :], rtol=0, atol=0), (
        "Rows differ — function should be z-independent by construction"
    )


def test_grad_H_u_a_corner_matches_lower_bound_grad():
    """The broadcast row equals self._lower_bound_grad[1][0, :2]."""
    grs = _setup(0)
    grad = grs._compute_grad_H_u_a_corner()

    expected = grs._lower_bound_grad[1][0, :2]
    assert np.array_equal(grad[0], expected), (
        f"row[0]={grad[0]} differs from lower_bound_grad[1][0, :2]={expected}"
    )
    # And it's broadcast to every z
    assert np.array_equal(grad, np.tile(expected, (grs.z_grid.shape[0], 1)))


def test_grad_H_u_a_corner_seed42():
    """Same invariants on a different seed."""
    grs = _setup(42)
    grad = grs._compute_grad_H_u_a_corner()
    expected = grs._lower_bound_grad[1][0, :2]
    n_z = grs.z_grid.shape[0]

    assert grad.shape == (n_z, 2)
    assert np.all(np.isfinite(grad))
    assert np.array_equal(grad, np.tile(expected, (n_z, 1)))
