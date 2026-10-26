"""
Pointwise CLT for the GRS statistics (mixin of :class:`GRS`).
"""

import numpy as np
from RMT.linalg.eig import minimal_shift_for_pd


class _SigmaMixin:
    """
    Asymptotic covariance of the GRS statistics at each grid point.

    ``Xi_clt`` is the 3 x 3 asymptotic covariance of
    ``sqrt(T) (Theta_d, Theta_d', Theta_d'')``, the debiased statistic and its
    first two z-derivatives. Its entries are closed-form expressions in the
    derivatives of the coefficient functions ``a(z) = 4 z (theta - xi)``,
    ``b(z) = 2 z xi`` and ``c(z) = 16 Z*(z)``, written below as ``A1 = a'``,
    ``B3 = b'''``, ``C2 = c''``, etc. The delta method maps ``Xi_clt`` to
    ``sigma_theta``, the asymptotic covariance of ``sqrt(T) (B_0, B_1, B_2)``.

    Entry functions return arrays of shape ``(1, M)``.
    """

    def _compute_sigma00(self) -> np.ndarray:
        """``Xi_clt[0, 0]``: asymptotic variance of ``sqrt(T) Theta_d``."""
        A1 = self._a_drv1
        B1 = self._b_drv1
        C1 = self._c_drv1
        return 2 * A1 ** 2 / C1 + A1 + B1

    def _compute_sigma01(self) -> np.ndarray:
        """``Xi_clt[0, 1]``: asymptotic covariance of ``Theta_d`` and ``Theta_d'``."""
        A1 = self._a_drv1
        A2 = self._a_drv2
        B2 = self._b_drv2
        C1 = self._c_drv1
        C2 = self._c_drv2
        term1 = - (A1 ** 2 * C2) / (C1 ** 2)
        term2 = 2 * A1 * A2 / C1
        term3 = 0.5 * (A2 + B2)
        return term1 + term2 + term3

    def _compute_sigma02(self) -> np.ndarray:
        """``Xi_clt[0, 2]``: asymptotic covariance of ``Theta_d`` and ``Theta_d''``."""
        A1 = self._a_drv1
        A2 = self._a_drv2
        A3 = self._a_drv3
        B3 = self._b_drv3
        C1 = self._c_drv1
        C2 = self._c_drv2
        C3 = self._c_drv3
        num1 = 3 * A1 ** 2 * C2 ** 2 - 2 * A1 * C1 * (A1 * C3 + 3 * A2 * C2)
        num2 = C1 ** 3 * (A3 + B3) + C1 ** 2 * (4 * A1 * A3 + 3 * A2 ** 2)
        den = 3 * C1 ** 3
        return (num1 + num2) / den

    def _compute_sigma11(self):
        """``Xi_clt[1, 1]``: asymptotic variance of ``sqrt(T) Theta_d'``."""
        A1 = self._a_drv1
        A2 = self._a_drv2
        A3 = self._a_drv3
        B3 = self._b_drv3
        C1 = self._c_drv1
        C2 = self._c_drv2
        C3 = self._c_drv3
        term1 = - (A1 ** 2 * C3) / (3 * C1 ** 2)
        term2 = (A1 ** 2 * C2 ** 2) / (C1 ** 3)
        term3 = - (2 * A1 * A2 * C2) / (C1 ** 2)
        term4 = (2 * A1 * A3) / (3 * C1)
        term5 = (A2 ** 2) / C1
        term6 = (A3 + B3) / 6.0
        return term1 + term2 + term3 + term4 + term5 + term6

    def _compute_sigma12(self) -> np.ndarray:
        """``Xi_clt[1, 2]``: asymptotic covariance of ``Theta_d'`` and ``Theta_d''``."""
        A1 = self._a_drv1
        A2 = self._a_drv2
        A3 = self._a_drv3
        A4 = self._a_drv4
        B4 = self._b_drv4
        C1 = self._c_drv1
        C2 = self._c_drv2
        C3 = self._c_drv3
        C4 = self._c_drv4
        term1 = - A1 ** 2 * C4 / (6 * C1 ** 2)
        term2 = 4 * A1 ** 2 * C2 * C3 / (3 * C1 ** 3)
        term3 = - 3 * A1 ** 2 * C2 ** 3 / (2 * C1 ** 4)
        term4 = - 4 * A1 * A2 * C3 / (3 * C1 ** 2)
        term5 = 3 * A1 * A2 * C2 ** 2 / (C1 ** 3)
        term6 = - 4 * A1 * A3 * C2 / (3 * C1 ** 2)
        term7 = A1 * A4 / (3 * C1)
        term8 = - 3 * A2 ** 2 * C2 / (2 * C1 ** 2)
        term9 = 4 * A2 * A3 / (3 * C1)
        term10 = A4 / 12
        term11 = B4 / 12
        return (term1 + term2 + term3 + term4 + term5 + term6 + term7 +
                term8 + term9 + term10 + term11)

    def _compute_sigma22(self) -> np.ndarray:
        """``Xi_clt[2, 2]``: asymptotic variance of ``sqrt(T) Theta_d''``."""
        A1 = self._a_drv1
        A2 = self._a_drv2
        A3 = self._a_drv3
        A4 = self._a_drv4
        C1 = self._c_drv1
        C2 = self._c_drv2
        C3 = self._c_drv3
        C4 = self._c_drv4
        term1 = 2 * A1 ** 2 * C2 * C4 / (3 * C1 ** 3)
        term2 = 2 * A1 ** 2 * C3 ** 2 / (3 * C1 ** 3)
        term3 = -4 * A1 ** 2 * C2 ** 2 * C3 / (C1 ** 4)
        term4 = 3 * A1 ** 2 * C2 ** 4 / (C1 ** 5)
        term5 = -2 * A1 * A2 * C4 / (3 * C1 ** 2)
        term6 = 16 * A1 * A2 * C2 * C3 / (3 * C1 ** 3)
        term7 = -6 * A1 * A2 * C2 ** 3 / (C1 ** 4)
        term8 = -4 * A1 * A3 * C3 / (3 * C1 ** 2)
        term9 = 8 * A1 * A3 * C2 ** 2 / (3 * C1 ** 3)
        term10_14 = (2 * A4 / (3 * C1)) * (A2 - A1 * C2 / C1)
        term11 = -4 * A2 ** 2 * C3 / (3 * C1 ** 2)
        term12 = 3 * A2 ** 2 * C2 ** 2 / (C1 ** 3)
        term13 = -8 * A2 * A3 * C2 / (3 * C1 ** 2)
        term15 = 2 * A3 ** 2 / (3 * C1)
        return (term1 + term2 + term3 + term4 + term5 + term6 + term7 +
                term8 + term9 + term10_14 + term11 + term12 + term13 + term15)

    def _compute_Xi_clt(self) -> np.ndarray:
        """
        Assemble ``Xi_clt``, the covariance of ``sqrt(T) (Theta_d, Theta_d', Theta_d'')``.

        Each 3 x 3 block is made positive definite with
        :func:`~RMT.linalg.eig.minimal_shift_for_pd`, which raises its ``(2, 2)``
        entry by at least 1e-3.

        Returns
        -------
        ndarray of shape (M, 3, 3)
        """
        xi_clt = np.zeros((self.z_grid.shape[0], 3, 3))
        xi_clt[:, 0, 0] = self._compute_sigma00()
        xi_clt[:, 0, 1], xi_clt[:, 1, 0] = self._compute_sigma01(), self._compute_sigma01()
        xi_clt[:, 0, 2], xi_clt[:, 2, 0] = self._compute_sigma02(), self._compute_sigma02()
        xi_clt[:, 1, 1] = self._compute_sigma11()
        xi_clt[:, 1, 2], xi_clt[:, 2, 1] = self._compute_sigma12(), self._compute_sigma12()
        xi_clt[:, 2, 2] = self._compute_sigma22()
        for i in range(len(self.z_grid)):
            tmp = xi_clt[i, :, :]
            tmp = minimal_shift_for_pd(tmp)
            xi_clt[i, :, :] = tmp
        return xi_clt

    def _compute_A_T_clt(self) -> np.ndarray:
        """
        Jacobian of ``(Theta_d, Theta_d', Theta_d'')`` to ``(B_0, B_1 / Z*, B_2 / Z*^2)``.

        With ``Z*' = c' / 16`` and ``Z*'' = c'' / 16``::

            A_T = [[1,  0,                    0               ],
                   [0, -1 / Z*',              0               ],
                   [0, -Z*'' / (2 Z*'^3),     1 / (2 Z*'^2)   ]]

        Returns
        -------
        ndarray of shape (M, 3, 3)
        """
        a_t = np.zeros((self.z_grid.shape[0], 3, 3))
        a_t[:, 0, 0] = 1.0
        a_t[:, 1, 1] = -1.0 / (self._c_drv1/16)
        a_t[:, 2, 2] = 1.0 / (2 * (self._c_drv1/16) ** 2)
        a_t[:, 2, 1] = -(self._c_drv2/16) / (2 * (self._c_drv1/16) ** 3)
        return a_t

    def _compute_tilde_A_T_clt(self) -> np.ndarray:
        """
        Full Jacobian ``diag(1, Z*, Z*^2) A_T`` of ``(Theta_d, Theta_d', Theta_d'')`` to ``(B_0, B_1, B_2)``.

        Returns
        -------
        ndarray of shape (M, 3, 3)
        """
        # Z* = c / 16
        diag_Zstar = np.zeros((self.z_grid.shape[0], 3, 3))
        diag_Zstar[:, 0, 0] = 1.0
        diag_Zstar[:, 1, 1] = (self._c/16)
        diag_Zstar[:, 2, 2] = (self._c/16) ** 2
        tilde_a_t = np.matmul(diag_Zstar, self._A_T_clt)
        return tilde_a_t

    def _compute_sigma_theta(self) -> np.ndarray:
        """
        Asymptotic covariance of ``sqrt(T) (B_0, B_1, B_2)``: ``tilde_A_T Xi_clt tilde_A_T'``.

        The standard error of ``B_j`` at grid point ``i`` is
        ``sqrt(sigma_theta[i, j, j] / T)``.

        Returns
        -------
        ndarray of shape (M, 3, 3)
        """
        sigma_theta = np.matmul(np.matmul(self._tilde_A_T_clt, self._Xi_clt),
                               self._tilde_A_T_clt.transpose(0, 2, 1))
        return sigma_theta

    def _compute_sigma_rj(self) -> list:
        """
        Asymptotic variances of the partial sums ``sqrt(T) (B_0 + ... + B_j)``.

        Computed as ``1' sigma_theta[:j+1, :j+1] 1`` for ``j = 0, 1, 2``.

        Returns
        -------
        list of three ndarrays of shape (M,)
        """
        variances = []
        for j in range(3):
            one_vec = np.ones((self.z_grid.shape[0], j + 1, 1))
            tmp = np.matmul(np.matmul(one_vec.transpose(0, 2, 1), self.sigma_theta[:, :(j + 1), :(j + 1)]),
                            one_vec).squeeze()
            variances.append(tmp)
        return variances

    def _compute_true_B(self, sigma: np.ndarray, mu: np.ndarray, order: int = 2) -> np.ndarray:
        """
        Population values ``B_j = Z*^j mu' (Sigma + Z* I)^{-(j+1)} mu`` for a known ``(mu, Sigma)``.

        Evaluated at the effective ridge ``Z*(z)`` of this object, so the
        estimates can be compared with their targets in simulations.

        Parameters
        ----------
        sigma : ndarray of shape (P,) or (P, P)
            Covariance matrix, or its eigenvalues for a diagonal covariance
            (``mu`` is then expressed in the same basis).
        mu : ndarray of shape (P,)
            Mean vector.
        order : int, default=2
            Highest ``j`` computed.

        Returns
        -------
        ndarray of shape (order + 1, M)
            Row ``j`` holds ``B_j`` on the grid.
        """
        zstar_grid = self._Zstar.get_z_star()
        zstar_grid = np.atleast_1d(zstar_grid)
        mu = np.asarray(mu)
        if len(sigma.shape) == 1:
            # Diagonal covariance: sigma holds the eigenvalues
            sigma = np.asarray(sigma)
            mu2 = mu ** 2
            true_B = []
            Z = zstar_grid[:, None]
            S = sigma[None, :]
            for j in range(order + 1):
                denom = (Z + S) ** (1 + j)
                vals = (zstar_grid ** j) * np.sum(mu2[None, :] / denom, axis=1)
                true_B.append(vals)
            return np.stack(true_B, axis=0)
        else:
            # Full covariance: rotate mu to the eigenbasis of sigma
            sigma = np.asarray(sigma)
            evals, evecs = np.linalg.eigh(sigma)
            mu_tilde = evecs.T @ mu
            mu_tilde2 = mu_tilde ** 2
            true_B = []
            Z = zstar_grid[:, None]
            L = evals[None, :]
            for j in range(order + 1):
                denom = (Z + L) ** (1 + j)
                vals = (zstar_grid ** j) * np.sum(mu_tilde2[None, :] / denom, axis=1)
                true_B.append(vals)
            return np.stack(true_B, axis=0)

    def get_sigma_rj(self) -> np.ndarray:
        """
        Asymptotic variances of ``sqrt(T) B_0``, ``sqrt(T) (B_0 + B_1)`` and ``sqrt(T) (B_0 + B_1 + B_2)``.

        Returns
        -------
        list of three ndarrays of shape (M,)
        """
        return self._sigma_rj
