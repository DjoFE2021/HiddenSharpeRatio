import numpy as np
import pytest
from numpy.testing import assert_allclose

from RMT.rmt.ZStar import ZStar


class DummyResolvent:
    """
    Minimal stub for Resolvent, only implements what ZStar needs:
      - z_grid
      - c_
      - get_m_hat()
      - get_xi()
      - get_xi_drv(order)
    """
    def __init__(self, z_grid, c_, m_hat, xi, xi_drvs):
        self.z_grid = np.asarray(z_grid, dtype=float)
        self.c_ = float(c_)
        self._m_hat = np.asarray(m_hat, dtype=float)
        self._xi = np.asarray(xi, dtype=float)
        # xi_drvs: dict[int, array]
        self._xi_drvs = {k: np.asarray(v, dtype=float) for k, v in xi_drvs.items()}

    def get_m_hat(self):
        return self._m_hat

    def get_xi(self):
        return self._xi

    def get_xi_drv(self, order: int):
        return self._xi_drvs[order]


@pytest.fixture
def simple_resolvent():
    # Two-point grid of NEGATIVE values, as expected by ZStar
    z_grid = np.array([-1.0, -2.0])
    c_ = 0.5

    # Arbitrary but fixed values for tests
    m_hat = np.array([0.2, 0.4])

    xi = np.array([10.0, 20.0])
    xi1 = np.array([1.0, 2.0])      # xi'(z)
    xi2 = np.array([5.0, 6.0])      # xi''(z)
    xi3 = np.array([7.0, 8.0])      # xi^{(3)}(z)
    xi4 = np.array([9.0, 10.0])     # xi^{(4)}(z)

    xi_drvs = {
        1: xi1,
        2: xi2,
        3: xi3,
        4: xi4,
    }

    return DummyResolvent(
        z_grid=z_grid,
        c_=c_,
        m_hat=m_hat,
        xi=xi,
        xi_drvs=xi_drvs,
    )


@pytest.fixture
def zstar(simple_resolvent):
    return ZStar(simple_resolvent)


def test_compute_z_star(simple_resolvent, zstar):
    """
    Check that ZStar.get_z_star implements:

        Z*(z) = z / (1 - c + c z m(-z))

    with z = -z_grid.
    """
    res = simple_resolvent
    z = -res.z_grid
    c = res.c_
    m = res.get_m_hat()

    expected = z / (1.0 - c + c * z * m)

    actual = zstar.get_z_star()
    assert_allclose(actual, expected)


def test_z_star_drv1(simple_resolvent, zstar):
    """
    (Z*)'(z) = 1 + xi(z) + z * xi'(z)
    """
    res = simple_resolvent
    z = -res.z_grid
    xi = res.get_xi()
    xi1 = res.get_xi_drv(order=1)

    expected = 1.0 + xi + z * xi1
    actual = zstar.get_z_star_drv(order=1)

    assert_allclose(actual, expected)


def test_z_star_drv2(simple_resolvent, zstar):
    """
    (Z*)''(z) = 2 * xi'(z) + z * xi''(z)
    """
    res = simple_resolvent
    z = -res.z_grid
    xi1 = res.get_xi_drv(order=1)
    xi2 = res.get_xi_drv(order=2)

    expected = 2.0 * xi1 + z * xi2
    actual = zstar.get_z_star_drv(order=2)

    assert_allclose(actual, expected)


def test_z_star_drv3(simple_resolvent, zstar):
    """
    (Z*)^{(3)}(z) = 3 * xi''(z) + z * xi^{(3)}(z)
    """
    res = simple_resolvent
    z = -res.z_grid
    xi2 = res.get_xi_drv(order=2)
    xi3 = res.get_xi_drv(order=3)

    expected = 3.0 * xi2 + z * xi3
    actual = zstar.get_z_star_drv(order=3)

    assert_allclose(actual, expected)


def test_z_star_drv4(simple_resolvent, zstar):
    """
    (Z*)^{(4)}(z) = 4 * xi^{(3)}(z) + z * xi^{(4)}(z)

    Note: ZStar actually precomputes derivative of order 4 as well
    and stores it in _z_star_drv[4], so we test that behavior.
    """
    res = simple_resolvent
    z = -res.z_grid
    xi3 = res.get_xi_drv(order=3)
    xi4 = res.get_xi_drv(order=4)

    expected = 4.0 * xi3 + z * xi4
    actual = zstar.get_z_star_drv(order=4)

    assert_allclose(actual, expected)


def test_z_star_drv_invalid_order_raises(zstar):
    """
    get_z_star_drv should raise ValueError for unsupported orders.
    """
    with pytest.raises(ValueError, match=r"order must be in \{1, 2, 3, 4\}"):
        zstar.get_z_star_drv(order=0)

    with pytest.raises(ValueError, match=r"order must be in \{1, 2, 3, 4\}"):
        zstar.get_z_star_drv(order=5)
