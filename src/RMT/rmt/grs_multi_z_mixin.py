"""
Joint CLT across grid points and simultaneous (multi-z) bounds (mixin of :class:`GRS`).
"""

import numpy as np
from scipy.stats import norm
from RMT.linalg.eig import clip_eigenvalues


class _MultiZMixin:
    """
    Joint covariance of the statistics across the z-grid and the multi-z critical values.

    ``Xi_multi[i, j]`` is the 3 x 3 asymptotic cross-covariance between
    ``sqrt(T) (Theta_d, Theta_d', Theta_d'')`` at ``z_i`` and at ``z_j``. Off the
    diagonal it follows from the kernel

        K(z1, z2) = Cov(sqrt(T) Theta_d(z1), sqrt(T) Theta_d(z2))
                  = (2 Delta_a^2 / Delta_c + Delta_a + Delta_b) / Delta_z,

    with ``Delta_f = f(z2) - f(z1)`` for the coefficient functions ``a``, ``b``,
    ``c`` (see :class:`_SigmaMixin`): block ``(p, q)`` is
    ``d^p/dz1^p d^q/dz2^q K(z1, z2)``. Diagonal blocks are the pointwise ``Xi_clt``.

    Projecting on the gradient of a bound gives its ``M x M`` covariance across
    the grid and its correlation ``R``. The critical value ``q`` solves
    ``P(max_i Z_i <= q) = Phi(n_sigma)`` for ``Z ~ N(0, R)``, so that, with
    probability ``Phi(n_sigma)``, the bounds hold at all grid points at once:

        lower:  max_i (H_i - q sd_i / sqrt(T)),
        upper:  min_i (U_i + q sd_i / sqrt(T)).

    In the block functions below, row ``i`` refers to ``z1 = z_i`` and column
    ``j`` to ``z2 = z_j``; ``a1_z1`` is ``a'(z_i)``, ``c2_z2`` is ``c''(z_j)``, etc.
    The diagonal of each block is filled with the pointwise value and the
    closed form is used off the diagonal (where ``Delta_z != 0``).
    """
    def _compute_Xi_multi_00(self) -> np.ndarray:
        """Block (0, 0): ``K(z_i, z_j)``, shape ``(M, M)``."""
        xi_00 = np.zeros((self.z_grid.shape[0], self.z_grid.shape[0]))
        np.fill_diagonal(xi_00, self._compute_sigma00()[0, :])
        delta_z = self.z_grid - self.z_grid[:,None]
        delta_a = self._a.ravel()[None, :] - self._a.ravel()[:, None]
        delta_b = self._b[None,:] - self._b[None,:].T
        delta_c = self._c[None,:] - self._c[None,:].T
        with np.errstate(invalid='ignore', divide='ignore'):
            temp_xi = 2*(delta_a)**2/(delta_c*delta_z) + delta_a/delta_z + delta_b/delta_z
        np.fill_diagonal(temp_xi, 0.0)
        xi_00 += temp_xi
        return xi_00

    def _compute_Xi_multi_10(self) -> np.ndarray:
        """Block (1, 0): ``dK/dz1``, the covariance of ``Theta_d'(z_i)`` and ``Theta_d(z_j)``."""
        xi_10 = np.zeros((self.z_grid.shape[0], self.z_grid.shape[0]))
        np.fill_diagonal(xi_10, self._compute_sigma01()[0, :])
        delta_z = self.z_grid - self.z_grid[:, None]
        delta_a = self._a.ravel()[None, :] - self._a.ravel()[:, None]
        delta_b = self._b[None, :] - self._b[:, None]
        delta_c = self._c[None, :] - self._c[:, None]
        a1_z1 = self._a_drv1.ravel()[:, None]
        b1_z1 = self._b_drv1[:, None]
        c1_z1 = self._c_drv1[:, None]
        numer = (delta_c ** 2 * (delta_b - delta_z * (a1_z1 + b1_z1))
                + delta_a * delta_c * (delta_c - 4 * a1_z1 * delta_z)
                + 2 * delta_a ** 2 * (c1_z1 * delta_z + delta_c))
        with np.errstate(invalid='ignore', divide='ignore'):
            temp_xi = numer / (delta_c ** 2 * delta_z ** 2)
        np.fill_diagonal(temp_xi, 0.0)
        xi_10 += temp_xi
        return xi_10

    def _compute_Xi_multi_01(self) -> np.ndarray:
        """Block (0, 1): ``dK/dz2``, the covariance of ``Theta_d(z_i)`` and ``Theta_d'(z_j)``."""
        xi_01 = np.zeros((self.z_grid.shape[0], self.z_grid.shape[0]))
        np.fill_diagonal(xi_01, self._compute_sigma01()[0, :])
        delta_z = self.z_grid - self.z_grid[:, None]
        delta_a = self._a.ravel()[None, :] - self._a.ravel()[:, None]
        delta_b = self._b[None, :] - self._b[:, None]
        delta_c = self._c[None, :] - self._c[:, None]
        a1_z2 = self._a_drv1.ravel()[None, :]
        b1_z2 = self._b_drv1[None, :]
        c1_z2 = self._c_drv1[None, :]
        numer = (delta_c ** 2 * (delta_z * (a1_z2 + b1_z2) - delta_b)
                - delta_a * delta_c * (delta_c - 4 * a1_z2 * delta_z)
                - 2 * delta_a ** 2 * (c1_z2 * delta_z + delta_c))
        with np.errstate(invalid='ignore', divide='ignore'):
            temp_xi = numer / (delta_c ** 2 * delta_z ** 2)
        np.fill_diagonal(temp_xi, 0.0)
        xi_01 += temp_xi
        return xi_01

    def _compute_Xi_multi_11(self) -> np.ndarray:
        """Block (1, 1): ``d^2K/dz1 dz2``, the covariance of ``Theta_d'(z_i)`` and ``Theta_d'(z_j)``."""
        xi_11 = np.zeros((self.z_grid.shape[0], self.z_grid.shape[0]))
        np.fill_diagonal(xi_11, self._compute_sigma11()[0, :])
        delta_z = self.z_grid - self.z_grid[:, None]
        delta_a = self._a.ravel()[None, :] - self._a.ravel()[:, None]
        delta_b = self._b[None, :] - self._b[:, None]
        delta_c = self._c[None, :] - self._c[:, None]
        a1_z1 = self._a_drv1.ravel()[:, None]
        a1_z2 = self._a_drv1.ravel()[None, :]
        b1_z1 = self._b_drv1[:, None]
        b1_z2 = self._b_drv1[None, :]
        c1_z1 = self._c_drv1[:, None]
        c1_z2 = self._c_drv1[None, :]
        numer = (delta_c * delta_z * (delta_c ** 2 * (a1_z1 + a1_z2 + b1_z1 + b1_z2)
                + 4 * delta_a * delta_c * (a1_z1 + a1_z2)
                - 2 * delta_a ** 2 * (c1_z1 + c1_z2))
                - 4 * delta_z ** 2 * (c1_z1 * delta_a - a1_z1 * delta_c) * (c1_z2 * delta_a - a1_z2 * delta_c)
                - 2 * delta_c ** 2 * (2 * delta_a ** 2 + delta_c * (delta_a + delta_b)))
        with np.errstate(invalid='ignore', divide='ignore'):
            temp_xi = numer / (delta_c ** 3 * delta_z ** 3)
        np.fill_diagonal(temp_xi, 0.0)
        xi_11 += temp_xi
        return xi_11

    def _compute_Xi_multi_20(self) -> np.ndarray:
        """Block (2, 0): ``d^2K/dz1^2``, the covariance of ``Theta_d''(z_i)`` and ``Theta_d(z_j)``."""
        xi_20 = np.zeros((self.z_grid.shape[0], self.z_grid.shape[0]))
        np.fill_diagonal(xi_20, self._compute_sigma02()[0, :])
        delta_z = self.z_grid - self.z_grid[:, None]
        delta_a = self._a.ravel()[None, :] - self._a.ravel()[:, None]
        delta_b = self._b[None, :] - self._b[:, None]
        delta_c = self._c[None, :] - self._c[:, None]
        a1_z1 = self._a_drv1.ravel()[:, None]
        a2_z1 = self._a_drv2.ravel()[:, None]
        b1_z1 = self._b_drv1[:, None]
        b2_z1 = self._b_drv2[:, None]
        c1_z1 = self._c_drv1[:, None]
        c2_z1 = self._c_drv2[:, None]
        numer = (delta_c ** 2 * (4 * a1_z1 ** 2 * delta_z ** 2
                - 2 * delta_c * delta_z * (a1_z1 + b1_z1)
                - delta_c * delta_z ** 2 * (a2_z1 + b2_z1)
                + 2 * delta_b * delta_c)
                + 2 * delta_a * delta_c * (
                - 2 * delta_c * delta_z * (2 * a1_z1 + a2_z1 * delta_z)
                - 4 * a1_z1 * c1_z1 * delta_z ** 2
                + delta_c ** 2)
                + 2 * delta_a ** 2 * (
                2 * c1_z1 ** 2 * delta_z ** 2
                + delta_c * delta_z * (2 * c1_z1 + c2_z1 * delta_z)
                + 2 * delta_c ** 2))
        with np.errstate(invalid='ignore', divide='ignore'):
            temp_xi = numer / (delta_c ** 3 * delta_z ** 3)
        np.fill_diagonal(temp_xi, 0.0)
        xi_20 += temp_xi
        return xi_20

    def _compute_Xi_multi_02(self) -> np.ndarray:
        """Block (0, 2): ``d^2K/dz2^2``, the covariance of ``Theta_d(z_i)`` and ``Theta_d''(z_j)``."""
        xi_02 = np.zeros((self.z_grid.shape[0], self.z_grid.shape[0]))
        np.fill_diagonal(xi_02, self._compute_sigma02()[0, :])
        delta_z = self.z_grid - self.z_grid[:, None]
        delta_a = self._a.ravel()[None, :] - self._a.ravel()[:, None]
        delta_b = self._b[None, :] - self._b[:, None]
        delta_c = self._c[None, :] - self._c[:, None]
        a1_z2 = self._a_drv1.ravel()[None, :]
        a2_z2 = self._a_drv2.ravel()[None, :]
        b2_z2 = self._b_drv2[None, :]
        c1_z2 = self._c_drv1[None, :]
        c2_z2 = self._c_drv2[None, :]
        numer = (delta_z ** 2 * (4 * delta_c ** 2 * (a1_z2 ** 2 + a2_z2 * delta_a)
                - 2 * delta_a * delta_c * (4 * a1_z2 * c1_z2 + c2_z2 * delta_a)
                + delta_c ** 3 * (a2_z2 + b2_z2)
                + 4 * c1_z2 ** 2 * delta_a ** 2)
                - 2 * delta_c * delta_z * (delta_c * (
                delta_c * (a1_z2 + self._b_drv1[None, :])
                + 4 * a1_z2 * delta_a)
                - 2 * c1_z2 * delta_a ** 2)
                + 2 * delta_c ** 2 * (2 * delta_a ** 2
                + delta_c * (delta_a + delta_b)))
        with np.errstate(invalid='ignore', divide='ignore'):
            temp_xi = numer / (delta_c ** 3 * delta_z ** 3)
        np.fill_diagonal(temp_xi, 0.0)
        xi_02 += temp_xi
        return xi_02

    def _compute_Xi_multi_21(self) -> np.ndarray:
        """Block (2, 1): ``d^3K/dz1^2 dz2``, the covariance of ``Theta_d''(z_i)`` and ``Theta_d'(z_j)``."""
        xi_21 = np.zeros((self.z_grid.shape[0], self.z_grid.shape[0]))
        np.fill_diagonal(xi_21, self._compute_sigma12()[0, :])
        delta_z = self.z_grid - self.z_grid[:, None]
        delta_a = self._a.ravel()[None, :] - self._a.ravel()[:, None]
        delta_b = self._b[None, :] - self._b[:, None]
        delta_c = self._c[None, :] - self._c[:, None]
        a1_z1 = self._a_drv1.ravel()[:, None]
        a2_z1 = self._a_drv2.ravel()[:, None]
        b1_z1 = self._b_drv1[:, None]
        b2_z1 = self._b_drv2[:, None]
        c1_z1 = self._c_drv1[:, None]
        c2_z1 = self._c_drv2[:, None]
        a1_z2 = self._a_drv1.ravel()[None, :]
        b1_z2 = self._b_drv1[None, :]
        c1_z2 = self._c_drv1[None, :]
        numer = (delta_c ** 2 * delta_z * (- 4 * delta_z ** 2 * (a1_z1 ** 2 * c1_z2 + 2 * a1_z1 * a1_z2 * c1_z1 + a2_z1 * a1_z2 * delta_c)
                + delta_c * delta_z * (delta_c * (a2_z1 + b2_z1) - 4 * a1_z1 * (a1_z1 + 2 * a1_z2))
                + 2 * delta_c ** 2 * (2 * a1_z1 + a1_z2 + 2 * b1_z1 + b1_z2))
                + 2 * delta_a * delta_c * (2 * delta_c * delta_z ** 2 * (2 * a1_z1 * (c1_z1 + c1_z2) + a2_z1 * c1_z2 * delta_z + 2 * a1_z2 * c1_z1 + a1_z2 * c2_z1 * delta_z)
                + 2 * delta_c ** 2 * delta_z * (4 * a1_z1 + a2_z1 * delta_z + 2 * a1_z2)
                + 4 * c1_z1 * delta_z ** 3 * (2 * a1_z1 * c1_z2 + a1_z2 * c1_z1)
                - 3 * delta_c ** 3)
                - 2 * delta_a ** 2 * (2 * delta_c * delta_z ** 2 * (c1_z1 ** 2 + 2 * c1_z1 * c1_z2 + c2_z1 * c1_z2 * delta_z)
                + 6 * c1_z1 ** 2 * c1_z2 * delta_z ** 3
                + delta_c ** 2 * delta_z * (4 * c1_z1 + c2_z1 * delta_z + 2 * c1_z2)
                + 6 * delta_c ** 3)
                - 6 * delta_b * delta_c ** 4)
        with np.errstate(invalid='ignore', divide='ignore'):
            temp_xi = numer / (delta_c ** 4 * delta_z ** 4)
        np.fill_diagonal(temp_xi, 0.0)
        xi_21 += temp_xi
        return xi_21

    def _compute_Xi_multi_12(self) -> np.ndarray:
        """Block (1, 2): ``d^3K/dz1 dz2^2``, the covariance of ``Theta_d'(z_i)`` and ``Theta_d''(z_j)``."""
        xi_12 = np.zeros((self.z_grid.shape[0], self.z_grid.shape[0]))
        np.fill_diagonal(xi_12, self._compute_sigma12()[0, :])
        delta_z = self.z_grid - self.z_grid[:, None]
        delta_a = self._a.ravel()[None, :] - self._a.ravel()[:, None]
        delta_b = self._b[None, :] - self._b[:, None]
        delta_c = self._c[None, :] - self._c[:, None]
        a1_z1 = self._a_drv1.ravel()[:, None]
        b1_z1 = self._b_drv1[:, None]
        c1_z1 = self._c_drv1[:, None]
        a1_z2 = self._a_drv1.ravel()[None, :]
        a2_z2 = self._a_drv2.ravel()[None, :]
        b1_z2 = self._b_drv1[None, :]
        b2_z2 = self._b_drv2[None, :]
        c1_z2 = self._c_drv1[None, :]
        c2_z2 = self._c_drv2[None, :]
        numer = (delta_c * delta_z ** 2 * (4 * delta_c ** 2 * (2 * a1_z1 * a1_z2 + a1_z2 ** 2 + a2_z2 * delta_a)
                - 2 * delta_a * delta_c * (4 * a1_z1 * c1_z2 + 4 * a1_z2 * (c1_z1 + c1_z2) + c2_z2 * delta_a)
                + delta_c ** 3 * (a2_z2 + b2_z2)
                + 4 * c1_z2 * delta_a ** 2 * (2 * c1_z1 + c1_z2))
                + 4 * delta_z ** 3 * (delta_c ** 2 * (2 * a1_z1 * a1_z2 * c1_z2 + a1_z1 * c2_z2 * delta_a + a1_z2 ** 2 * c1_z1 + a2_z2 * c1_z1 * delta_a)
                - delta_a * delta_c * (2 * a1_z1 * c1_z2 ** 2 + 4 * a1_z2 * c1_z1 * c1_z2 + c1_z1 * c2_z2 * delta_a)
                - a1_z1 * a2_z2 * delta_c ** 3
                + 3 * c1_z1 * c1_z2 ** 2 * delta_a ** 2)
                - 2 * delta_c ** 2 * delta_z * (delta_c ** 2 * (a1_z1 + 2 * a1_z2 + b1_z1 + 2 * b1_z2)
                + 4 * delta_a * delta_c * (a1_z1 + 2 * a1_z2)
                - 2 * delta_a ** 2 * (c1_z1 + 2 * c1_z2))
                + 6 * delta_c ** 3 * (2 * delta_a ** 2 + delta_c * (delta_a + delta_b)))
        with np.errstate(invalid='ignore', divide='ignore'):
            temp_xi = numer / (delta_c ** 4 * delta_z ** 4)
        np.fill_diagonal(temp_xi, 0.0)
        xi_12 += temp_xi
        return xi_12

    def _compute_Xi_multi_22(self) -> np.ndarray:
        """Block (2, 2): ``d^4K/dz1^2 dz2^2``, the covariance of ``Theta_d''(z_i)`` and ``Theta_d''(z_j)``."""
        xi_22 = np.zeros((self.z_grid.shape[0], self.z_grid.shape[0]))
        np.fill_diagonal(xi_22, self._compute_sigma22()[0, :])
        delta_z = self.z_grid - self.z_grid[:, None]
        delta_a = self._a.ravel()[None, :] - self._a.ravel()[:, None]
        delta_b = self._b[None, :] - self._b[:, None]
        delta_c = self._c[None, :] - self._c[:, None]
        a1_z1 = self._a_drv1.ravel()[:, None]
        a2_z1 = self._a_drv2.ravel()[:, None]
        b1_z1 = self._b_drv1[:, None]
        b2_z1 = self._b_drv2[:, None]
        c1_z1 = self._c_drv1[:, None]
        c2_z1 = self._c_drv2[:, None]
        a1_z2 = self._a_drv1.ravel()[None, :]
        a2_z2 = self._a_drv2.ravel()[None, :]
        b1_z2 = self._b_drv1[None, :]
        b2_z2 = self._b_drv2[None, :]
        c1_z2 = self._c_drv1[None, :]
        c2_z2 = self._c_drv2[None, :]
        N1 = 2 * delta_c ** 2 * delta_z ** 2 * (4 * delta_c ** 2 * (a1_z1 ** 2 + 4 * a1_z1 * a1_z2 + a1_z2 ** 2 + delta_a * (a2_z2 - a2_z1))
                - 2 * delta_a * delta_c * (4 * a1_z1 * (c1_z1 + 2 * c1_z2) + 4 * a1_z2 * (2 * c1_z1 + c1_z2) + delta_a * (c2_z2 - c2_z1))
                + delta_c ** 3 * (- a2_z1 + a2_z2 - b2_z1 + b2_z2)
                + 4 * delta_a ** 2 * (c1_z1 ** 2 + 4 * c1_z1 * c1_z2 + c1_z2 ** 2))
        N2 = 4 * delta_z ** 4 * (delta_c * (delta_c ** 2 * (- a1_z1 ** 2 * c2_z2 + 2 * a2_z1 * a1_z2 * c1_z2 + a2_z1 * c2_z2 * delta_a + a1_z2 ** 2 * c2_z1 + a2_z2 * c2_z1 * delta_a)
                + 2 * a1_z1 ** 2 * c1_z2 ** 2 * delta_c
                - delta_a * delta_c * (2 * a2_z1 * c1_z2 ** 2 + 4 * a1_z2 * c2_z1 * c1_z2 + c2_z1 * c2_z2 * delta_a)
                - a2_z1 * a2_z2 * delta_c ** 3
                + 3 * c2_z1 * c1_z2 ** 2 * delta_a ** 2)
                - 2 * a1_z1 * c1_z1 * delta_c * (- 4 * a1_z2 * c1_z2 * delta_c + delta_c * (a2_z2 * delta_c - 2 * c2_z2 * delta_a) + 6 * c1_z2 ** 2 * delta_a)
                + c1_z1 ** 2 * (2 * delta_c ** 2 * (a1_z2 ** 2 + a2_z2 * delta_a) - 3 * delta_a * delta_c * (4 * a1_z2 * c1_z2 + c2_z2 * delta_a) + 12 * c1_z2 ** 2 * delta_a ** 2))
        N3 = 8 * delta_c * delta_z ** 3 * (delta_c ** 2 * (a1_z1 ** 2 * c1_z2 + 2 * a1_z1 * a1_z2 * (c1_z1 + c1_z2) + a1_z1 * c2_z2 * delta_a - a2_z1 * c1_z2 * delta_a + a1_z2 ** 2 * c1_z1 - a1_z2 * c2_z1 * delta_a + a2_z2 * c1_z1 * delta_a)
                + delta_c ** 3 * (a2_z1 * a1_z2 - a1_z1 * a2_z2)
                - delta_a * delta_c * (2 * a1_z1 * c1_z2 * (2 * c1_z1 + c1_z2) + 2 * a1_z2 * c1_z1 * (c1_z1 + 2 * c1_z2) + c1_z1 * c2_z2 * delta_a - c2_z1 * c1_z2 * delta_a)
                + 3 * c1_z1 * c1_z2 * delta_a ** 2 * (c1_z1 + c1_z2))
        N4 = -12 * delta_c ** 3 * delta_z * (delta_c ** 2 * (a1_z1 + a1_z2 + b1_z1 + b1_z2) + 4 * delta_a * delta_c * (a1_z1 + a1_z2) - 2 * delta_a ** 2 * (c1_z1 + c1_z2))
        N5 = 24 * delta_c ** 4 * (2 * delta_a ** 2 + delta_c * (delta_a + delta_b))
        numer = N1 + N2 + N3 + N4 + N5
        with np.errstate(invalid='ignore', divide='ignore'):
            temp_xi = numer / (delta_c ** 5 * delta_z ** 5)
        np.fill_diagonal(temp_xi, 0.0)
        xi_22 += temp_xi
        return xi_22

    def _compute_Xi_multi(self) -> np.ndarray:
        """
        Assemble the joint covariance tensor ``Xi_multi``.

        ``Xi_multi[i, j, p, q]`` is the asymptotic covariance between
        coordinate ``p`` at ``z_i`` and coordinate ``q`` at ``z_j`` of
        ``sqrt(T) (Theta_d, Theta_d', Theta_d'')``. The diagonal blocks
        ``Xi_multi[i, i]`` are set to the pointwise ``Xi_clt[i]``.

        Returns
        -------
        ndarray of shape (M, M, 3, 3)
        """
        Xi_multi = np.zeros((self.z_grid.shape[0], self.z_grid.shape[0], 3, 3))
        Xi_multi[:, :, 0, 0] = self._compute_Xi_multi_00()
        Xi_multi[:, :, 1, 0] = self._compute_Xi_multi_10()
        Xi_multi[:, :, 0, 1] = self._compute_Xi_multi_01()
        Xi_multi[:, :, 1, 1] = self._compute_Xi_multi_11()
        Xi_multi[:, :, 2, 0] = self._compute_Xi_multi_20()
        Xi_multi[:, :, 0, 2] = self._compute_Xi_multi_02()
        Xi_multi[:, :, 2, 1] = self._compute_Xi_multi_21()
        Xi_multi[:, :, 1, 2] = self._compute_Xi_multi_12()
        Xi_multi[:, :, 2, 2] = self._compute_Xi_multi_22()
        for i in range(self.z_grid.shape[0]):
            Xi_multi[i, i, :, :] = self._Xi_clt[i, :, :]
        return Xi_multi

    def _compute_A_Xi_A_multi(self, order = 1) -> np.ndarray:
        """
        Joint covariance of ``sqrt(T) (B_0, ..., B_order)`` across grid points.

        Block ``[i, j]`` is ``tilde_A_i Xi_multi[i, j] tilde_A_j'``, restricted
        to the first ``order + 1`` coordinates; each block is then passed
        through :func:`~RMT.linalg.eig.clip_eigenvalues` (symmetrised,
        eigenvalues floored at 1e-10).

        Parameters
        ----------
        order : {1, 2}, default=1
            Bound order ``J``.

        Returns
        -------
        None
            Fills ``self._A_Xi_A_multi[order]``, of shape ``(M, M, order + 1, order + 1)``.
        """
        tilde_A = self._tilde_A_T_clt[:, :order+1, :order+1]
        Xi_multi = self._Xi_multi[:, :, :order+1, :order+1]
        for i in range(self.z_grid.shape[0]):
            for j in range(self.z_grid.shape[0]):
                term1 = tilde_A[i]
                term3 = tilde_A[j].T
                term2 = Xi_multi[i, j]
                self._A_Xi_A_multi[order][i, j] =  clip_eigenvalues(term1 @ term2 @ term3)

    def _compute_sigma_J_multi_lb(self, order = 1) -> np.ndarray:
        """
        Covariance of the infinite-series lower bound across the grid (delta method).

        ``Sigma[i, j] = g_i' C[i, j] g_j``, with ``g`` the gradient of the lower
        bound of order ``J = order`` (``_lower_bound_grad``) and ``C`` from
        :meth:`_compute_A_Xi_A_multi`. Eigenvalues are floored at 1e-10.

        Returns
        -------
        ndarray of shape (M, M)
            Asymptotic covariance of ``sqrt(T) H_J`` across grid points.
        """
        grad_lb = self._lower_bound_grad[order].astype(np.float64)
        A_Xi_A_multi = self._A_Xi_A_multi[order].astype(np.float64)
        sigma_J_multi = np.einsum('ia,ijab,jb->ij', grad_lb, A_Xi_A_multi, grad_lb)
        return clip_eigenvalues(sigma_J_multi)

    def _compute_sigma_J_multi_ub(self, order = 1) -> np.ndarray:
        """
        Covariance of the upper bound at ``a_min`` across the grid (delta method).

        Same as :meth:`_compute_sigma_J_multi_lb` with the gradient of ``u_J``.

        Returns
        -------
        ndarray of shape (M, M)
            Asymptotic covariance of ``sqrt(T) u_J`` across grid points.
        """
        grad_ub = self._upper_bound_data[order]["u_J_drv"].astype(np.float64)
        A_Xi_A_multi = self._A_Xi_A_multi[order].astype(np.float64)
        sigma_J_multi_ub = np.einsum('ia,ijab,jb->ij', grad_ub, A_Xi_A_multi, grad_ub)
        return clip_eigenvalues(sigma_J_multi_ub)

    # -------------------------------------------------------------------------
    # Corner branch of the simultaneous upper bound (J = 1)
    #
    # When the a_min upper bound falls below the lower bound H_1^L at the first
    # grid point z_0, the upper bound is replaced by that lower bound (see
    # GRS.get_all_ubs). The corner cut-off a_corner(z) solves
    # H_1^U(B(z); a) = H_1^L(B(z_0)).
    # -------------------------------------------------------------------------
    def _compute_grad_H_u_a_corner(self) -> np.ndarray:
        """
        Gradient of the corner-branch upper bound with respect to ``(B_0, B_1)``.

        Differentiating ``H_1^U(B(z); a_corner(z)) = H_1^L(B(z_0))`` shows that
        the gradient equals that of ``H_1^L`` at the first grid point, for
        every ``z``. It is therefore the row ``_lower_bound_grad[1][0, :2]``
        broadcast to all grid points.

        Returns
        -------
        ndarray of shape (M, 2)
            Also stored in ``self._grad_H_u_a_corner``.
        """
        n_z = self.z_grid.shape[0]
        lb_grad_z2 = self._lower_bound_grad[1][0, :2]  # [dH/dB_0, dH/dB_1] at z_0
        self._grad_H_u_a_corner = np.broadcast_to(lb_grad_z2, (n_z, 2)).copy()
        return self._grad_H_u_a_corner

    def _compute_sigma_J_multi_ub_corner(self) -> np.ndarray:
        """
        Covariance of the corner-branch upper bound across the grid.

        The bound is ``H_1^L`` at the first grid point for every ``z``, so all
        entries equal its asymptotic variance (eigenvalues then floored at 1e-10).

        Returns
        -------
        ndarray of shape (M, M)
            Also stored in ``self._sigma_J_multi_ub_corner``.
        """
        order = 1
        var_index = 1
        z_2_index = 0
        # Asymptotic variance of the J = 1 infinite lower bound at the first grid point
        sigma_2_z2 = self._lower_bound_clt_infinity[order][var_index][z_2_index]

        sigma_J_multi_ub_corner = sigma_2_z2 * np.ones(
            (self.z_grid.shape[0], self.z_grid.shape[0])
        )

        self._sigma_J_multi_ub_corner = clip_eigenvalues(sigma_J_multi_ub_corner)
        return self._sigma_J_multi_ub_corner

    def _compute_R_sigma_H_ub_corner(self) -> np.ndarray:
        """Correlation matrix of the corner-branch upper bound (see :meth:`_compute_R_sigma_H_lb`)."""
        sigma = self._sigma_J_multi_ub_corner
        diag_term = np.diag(1/np.sqrt(np.diag(sigma)))
        R = clip_eigenvalues(diag_term@sigma@diag_term, eps=1e-6)
        d = np.sqrt(np.diag(R))
        return R / np.outer(d, d)

    def _compute_equicoordinate_quantile_ub_corner(self) -> float:
        """Critical value ``q`` of the corner-branch upper bound (see :meth:`_compute_equicoordinate_quantile_ub`)."""
        R = self._R_sigma_H_ub_corner
        alpha = 1 - norm.cdf(self.n_sigma)
        return self._mc_equicoordinate_quantile(R, target=1 - alpha)

    # -------------------------------------------------------------------------
    # Correlation matrices and critical values
    # -------------------------------------------------------------------------
    def _compute_R_sigma_H_lb(self, order= 2):
        """
        Correlation matrix of the lower bound of order ``J = order`` across the grid.

        ``D^{-1/2} Sigma D^{-1/2}`` with ``D = diag(Sigma)``, eigenvalues floored
        at 1e-6 and rescaled to a unit diagonal.

        Returns
        -------
        ndarray of shape (M, M)
        """
        sigma = self._sigma_J_multi_lb[order]
        diag_term = np.diag(1/np.sqrt(np.diag(sigma)))
        R = clip_eigenvalues(diag_term@sigma@diag_term, eps=1e-6)
        d = np.sqrt(np.diag(R))
        return R / np.outer(d, d)

    @staticmethod
    def _mc_equicoordinate_quantile(R: np.ndarray, target: float,
                                    n_samples: int = 1_000_000,
                                    seed: int = 0) -> float:
        """
        Monte Carlo ``target``-quantile of ``max_i Z_i`` for ``Z ~ N(0, R)``.

        Parameters
        ----------
        R : ndarray of shape (K, K)
            Correlation matrix.
        target : float
            Quantile level in ``(0, 1)``.
        n_samples : int, default=1_000_000
            Number of draws.
        seed : int, default=0
            Seed of the random generator (results are reproducible).

        Returns
        -------
        float
            ``q`` such that ``P(max_i Z_i <= q) ~= target``. For ``K = 1`` this
            is the normal quantile; it grows with ``K`` for weakly correlated ``Z``.
        """
        K = R.shape[0]
        rng = np.random.default_rng(seed)
        samples = rng.multivariate_normal(np.zeros(K), R, size=n_samples)
        return float(np.quantile(np.max(samples, axis=1), target))

    def _compute_equicoordinate_quantile_lb(self, order: int) -> float:
        """
        Critical value ``q`` of the simultaneous lower bound of order ``J = order``.

        Solves ``P(max_i Z_i <= q) = Phi(n_sigma)`` for ``Z ~ N(0, R_lb)``,
        i.e. one-sided level ``alpha = 1 - Phi(n_sigma)`` (0.025 for the
        default ``n_sigma = 1.96``).
        """
        R = self._R_sigma_H_lb[order]
        alpha = 1 - norm.cdf(self.n_sigma)
        return self._mc_equicoordinate_quantile(R, target=1 - alpha)

    def _compute_equicoordinate_quantile_ub(self, order: int) -> float:
        """Critical value ``q`` of the simultaneous upper bound of order ``J = order`` (same level as the lower bound)."""
        R = self._R_sigma_H_ub[order]
        alpha = 1 - norm.cdf(self.n_sigma)
        return self._mc_equicoordinate_quantile(R, target=1 - alpha)

    def _compute_R_sigma_H_ub(self, order = 2):
        """Correlation matrix of the upper bound at ``a_min`` across the grid (see :meth:`_compute_R_sigma_H_lb`)."""
        sigma = self._sigma_J_multi_ub[order]
        diag_term = np.diag(1/np.sqrt(np.diag(sigma)))
        R = clip_eigenvalues(diag_term@sigma@diag_term, eps=1e-6)
        d = np.sqrt(np.diag(R))
        return R / np.outer(d, d)
