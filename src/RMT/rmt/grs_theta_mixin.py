"""
theta(z), its derivatives and the debiased statistics B_0, B_1, B_2 (mixin of :class:`GRS`).
"""

import numpy as np
from typing import Dict


class _ThetaMixin:
    """
    Ridge statistic ``theta(z)``, its z-derivatives and the GRS statistics ``B_j``.

    ``theta(z) = mu_hat' (Sigma_hat + z I)^{-1} mu_hat`` is the ridge-regularised
    in-sample squared Sharpe ratio, with ``mu_hat`` and ``Sigma_hat`` the sample
    mean and covariance of the returns. It is computed from the eigendecomposition
    of the second-moment matrix ``X'X / T`` in two steps:

    1. ``theta_before(z) = mu_hat' (X'X / T + z I)^{-1} mu_hat``;
    2. ``theta = theta_before / (1 - theta_before)`` (Sherman-Morrison), which
       replaces the second-moment matrix by the covariance ``X'X / T - mu_hat mu_hat'``.

    Debiasing with ``xi(z)`` and differentiating along ``Z*(z)`` then gives the
    estimates of ``B_j(z) = Z*^j mu' (Sigma + Z* I)^{-(j+1)} mu`` for j = 0, 1, 2.
    """

    @staticmethod
    def the_tbs(
            a: np.ndarray,
            denom: np.ndarray,
            max_order: int,
    ) -> np.ndarray:
        """
        Sums ``tb_k = sum_i (-1)^k k! a_i / denom_i^(k+1)`` for ``k = 0, ..., max_order``.

        These are ``sum_i a_i / (z + lambda_i)`` and its z-derivatives when
        ``denom = z + lambda``.

        Parameters
        ----------
        a : ndarray of shape (N, 1)
            Weights ``a_i``.
        denom : ndarray of shape (N, M)
            Denominators ``z_j + lambda_i``.
        max_order : int
            Highest derivative order (0 to 4).

        Returns
        -------
        list of ndarray
            ``[tb_0, ..., tb_max_order]``, each of shape ``(1, M)``.
        """
        out: list[np.ndarray] = []

        tb0 = (a / denom).sum(axis=0, keepdims=True)  # (1, M)
        out.append(tb0)

        if max_order >= 1:
            tb1 = (-a / denom ** 2).sum(axis=0, keepdims=True)
            out.append(tb1)
        if max_order >= 2:
            tb2 = (2.0 * a / denom ** 3).sum(axis=0, keepdims=True)
            out.append(tb2)
        if max_order >= 3:
            tb3 = (-6.0 * a / denom ** 4).sum(axis=0, keepdims=True)
            out.append(tb3)
        if max_order >= 4:
            tb4 = (24.0 * a / denom ** 5).sum(axis=0, keepdims=True)
            out.append(tb4)
        return out

    def _theta_before_and_derivs(
            self,
            max_order: int = 0,
    ) -> list[np.ndarray]:
        """
        ``theta_before(z) = mu_hat' (X'X / T + z I)^{-1} mu_hat`` and its z-derivatives.

        In the eigenbasis:

        - ``P <= T``: ``theta_before(z) = sum_i v_i^2 / (z + d_i)``, with ``d_i``
          the eigenvalues of ``X'X / T`` and ``v = Q' mu_hat``;
        - ``P > T``: ``theta_before(z) = (1/T) sum_i u_i^2 s_i / (z + s_i)``, with
          ``s_i`` the eigenvalues of ``XX' / T`` and ``u = Q' 1``.

        Parameters
        ----------
        max_order : int, default=0
            Highest derivative order (0 to 4).

        Returns
        -------
        list of ndarray
            ``[theta_before, theta_before', ...]``, each of shape ``(1, M)``.
        """
        if not (0 <= max_order <= 4):
            raise ValueError("max_order must be between 0 and 4.")

        z = self.z_grid.reshape(1, -1)  # (1, M)

        if self._eigsys.get_regime == "under":
            # a_i = v_i^2, denominators z + d_i
            eig = self._eigsys.get_eigenvalues().reshape(-1, 1)  # (P, 1)
            v = self._v.reshape(-1, 1)  # (P, 1)
            a = v ** 2  # (P, 1)
            denom = z + eig  # (P, M)
        else:
            # a_i = u_i^2 s_i / T, denominators z + s_i
            s = self._eigsys.get_non_zero_eigenvalues().reshape(-1, 1)  # (T, 1)
            u = self._u.reshape(-1, 1)  # (T, 1)
            a = (u ** 2 * s) / self._eigsys.t_  # (T, 1)
            denom = z + s  # (T, M)

        out = self.the_tbs(a, denom, max_order)
        return out

    def _compute_debiased_theta_and_its_derivatives(self,
                                                    theta: np.ndarray,
                                                    theta_drv1: np.ndarray,
                                                    theta_drv2: np.ndarray,
                                                    z_star: np.ndarray,
                                                    z_star_drv1: np.ndarray,
                                                    z_star_drv2: np.ndarray,
                                                    xi: np.ndarray,
                                                    xi_drv1: np.ndarray,
                                                    xi_drv2: np.ndarray
                                                    ) -> None:
        """
        Debias ``theta`` and compute the GRS statistics ``B_0``, ``B_1``, ``B_2``.

        The debiased statistic ``Theta_d(z) = (theta(z) - xi(z)) / (1 + xi(z))``
        estimates ``g(Z*(z))`` with ``g(Z) = mu' (Sigma + Z I)^{-1} mu``. Since
        ``B_1 = -Z g'(Z)`` and ``B_2 = Z^2 g''(Z) / 2``, the chain rule along
        ``Z*(z)`` gives

            B_0 = Theta_d
            B_1 = -Z* Theta_d' / Z*'
            B_2 = Z*^2 / (2 Z*'^2) * (Theta_d'' - (Z*'' / Z*') Theta_d')

        (primes are derivatives in ``z``). The estimates are then clipped so
        that ``0 < B_2 < B_1 < B_0``, as required by the bounds:
        ``B_0 >= 1.1e-8``, ``B_1`` in ``[0.001 B_0, 0.99 B_0]`` and ``B_2`` in
        ``[0.0001 B_0, 0.99 B_1]``.

        All inputs are arrays of shape ``(M, 1)`` on the grid.

        Stores ``_cb0``, ``_cb1``, ``_cb2`` and, for :meth:`get_b1_diagnostics`,
        ``_cb1_raw`` (``B_1`` before clipping), ``_diag_z_star``,
        ``_diag_z_star_drv1`` and ``_diag_theta_d_drv1``.
        """
        # Theta_d = (theta - xi) / (1 + xi) = f / g
        f       = theta - xi
        f_drv1  = theta_drv1 - xi_drv1
        f_drv2  = theta_drv2 - xi_drv2
        g       = 1 + xi
        g_drv1  = xi_drv1
        g_drv2  = xi_drv2

        # B_0 = Theta_d, floored to stay positive
        eps = 1e-8
        self._cb0 = np.maximum(f / g, eps * 1.1)

        # Theta_d' = (f'g - fg') / g^2
        theta_d_drv1 = (f_drv1 * g - f * g_drv1) / g ** 2

        # Theta_d'' = [(f''g - fg'')g - 2g'(f'g - fg')] / g^3
        theta_d_drv2 = ((f_drv2 * g - f * g_drv2) * g - 2 * g_drv1 * (f_drv1 * g - f * g_drv1)) / g ** 3

        # Components kept for get_b1_diagnostics()
        self._diag_z_star       = z_star
        self._diag_z_star_drv1  = z_star_drv1
        self._diag_theta_d_drv1 = theta_d_drv1

        # B_1 = -Z* Theta_d' / Z*'
        self._cb1_raw = -z_star * theta_d_drv1 / z_star_drv1
        self._cb1 = np.clip(self._cb1_raw, 0.001*self._cb0, 0.99 * self._cb0)

        # B_2 = Z*^2 / (2 Z*'^2) [Theta_d'' - (Z*'' / Z*') Theta_d']
        self._cb2 = z_star ** 2 / (2 * z_star_drv1 ** 2) * (
                theta_d_drv2 - z_star_drv2 / z_star_drv1 * theta_d_drv1
        )
        self._cb2 = np.clip(self._cb2, 0.0001*self._cb0, 0.99 * self._cb1)

    # theta = 1 / (1 - x) - 1 with x = theta_before, so its derivatives follow
    # from the chain rule applied to 1 / (1 - x).
    def _compute_theta_hat(self) -> np.ndarray:
        """``theta(z) = theta_before(z) / (1 - theta_before(z))`` on the grid, shape ``(1, M)``."""
        theta_before = self._theta_before_and_derivs(max_order=0)[0]
        theta = theta_before / (1.0 - theta_before)
        return theta

    def _compute_theta_hat_prime(self) -> np.ndarray:
        """``theta'(z) = x' / (1 - x)^2`` with ``x = theta_before``, shape ``(1, M)``."""
        tb0, tb1 = self._theta_before_and_derivs(max_order=1)

        d = 1.0 - tb0
        return tb1 / d ** 2

    def _compute_theta_hat_second(self) -> np.ndarray:
        """``theta''(z) = x'' / (1 - x)^2 + 2 x'^2 / (1 - x)^3``, shape ``(1, M)``."""
        tb0, tb1, tb2 = self._theta_before_and_derivs(max_order=2)
        d = 1.0 - tb0
        return tb2 / d ** 2 + 2.0 * tb1 ** 2 / d ** 3

    def _compute_theta_hat_third(self) -> np.ndarray:
        """``theta'''(z) = x''' / (1 - x)^2 + 6 x' x'' / (1 - x)^3 + 6 x'^3 / (1 - x)^4``."""
        tb0, tb1, tb2, tb3 = self._theta_before_and_derivs(max_order=3)
        d = 1.0 - tb0
        return (
                6.0 * tb1 ** 3 / d ** 4
                + 6.0 * tb1 * tb2 / d ** 3
                + tb3 / d ** 2
        )

    def _compute_theta_hat_fourth(self) -> np.ndarray:
        """
        Fourth derivative of ``theta`` in ``z``, shape ``(1, M)``.

        ``theta'''' = x'''' / (1-x)^2 + (8 x' x''' + 6 x''^2) / (1-x)^3
        + 36 x'^2 x'' / (1-x)^4 + 24 x'^4 / (1-x)^5`` with ``x = theta_before``.
        """
        tb0, tb1, tb2, tb3, tb4 = self._theta_before_and_derivs(max_order=4)
        d = 1.0 - tb0
        return (
                24.0 * tb1 ** 4 / d ** 5
                + 36.0 * tb1 ** 2 * tb2 / d ** 4
                + 6.0 * tb2 ** 2 / d ** 3
                + 8.0 * tb1 * tb3 / d ** 3
                + tb4 / d ** 2
        )

    def _compute_grs_statistics(self) -> None:
        """Compute ``B_0``, ``B_1``, ``B_2`` on the grid (stored in ``_cb0``, ``_cb1``, ``_cb2``)."""
        # Reshape all inputs to column vectors (M, 1)
        xi = self._resolvent.get_xi().reshape(-1, 1)
        xi_drv1 = self._resolvent.get_xi_drv(order=1).reshape(-1, 1)
        xi_drv2 = self._resolvent.get_xi_drv(order=2).reshape(-1, 1)
        z_star = self._Zstar.get_z_star().reshape(-1, 1)
        z_star_drv1 = self._Zstar.get_z_star_drv(order=1).reshape(-1, 1)
        z_star_drv2 = self._Zstar.get_z_star_drv(order=2).reshape(-1, 1)

        theta = self._theta.reshape(-1, 1)
        theta_drv1 = self._theta_drv1.reshape(-1, 1)
        theta_drv2 = self._theta_drv2.reshape(-1, 1)

        self._compute_debiased_theta_and_its_derivatives(theta,
                                                         theta_drv1,
                                                         theta_drv2,
                                                         z_star,
                                                         z_star_drv1,
                                                         z_star_drv2,
                                                         xi,
                                                         xi_drv1,
                                                         xi_drv2)

    def get_grs_statistics(
            self,
    ) -> Dict[int, np.ndarray]:
        """
        The GRS statistics on the grid, per period.

        Returns
        -------
        dict
            ``{0: B_0, 1: B_1, 2: B_2}``, each an array of shape ``(M,)``
            (after clipping, see :meth:`_compute_debiased_theta_and_its_derivatives`).
        """
        out = {
            0: self._cb0.flatten(),
            1: self._cb1.flatten(),
            2: self._cb2.flatten(),
        }

        return out

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    def get_b1_diagnostics(self) -> dict:
        """
        Components of ``B_1 = -Z* Theta_d' / Z*'``, before and after clipping.

        Useful to see where on the grid ``B_1`` hits its clipping bounds
        ``[0.001 B_0, 0.99 B_0]``.

        Returns
        -------
        dict of ndarray of shape (M,)
            ``'z_star'`` (``Z*``), ``'z_star_drv1'`` (``Z*'``), ``'theta_d_drv1'``
            (``Theta_d'``), ``'b1_raw'`` (``B_1`` before clipping), ``'b0'``
            and ``'b1'`` (the reported ``B_0`` and ``B_1``).

        Examples
        --------
        >>> diags = grs.get_b1_diagnostics()
        >>> n_clipped = int(np.sum(np.abs(diags['b1_raw'] - diags['b1']) > 1e-10))
        """
        return {
            'z_star':       np.asarray(self._diag_z_star).flatten(),
            'z_star_drv1':  np.asarray(self._diag_z_star_drv1).flatten(),
            'theta_d_drv1': np.asarray(self._diag_theta_d_drv1).flatten(),
            'b1_raw':       np.asarray(self._cb1_raw).flatten(),
            'b0':           np.asarray(self._cb0).flatten(),
            'b1':           np.asarray(self._cb1).flatten(),
        }

    def get_response_concentration(self) -> dict:
        """
        How concentrated the mean-return signal is across principal components.

        The response weights are ``w_i = u_i^2 / T`` with ``u = Q' 1`` when
        ``P > T`` (they sum to one), and ``w_i = v_i^2 / ||v||^2`` with
        ``v = Q' mu_hat`` when ``P <= T``.

        Returns
        -------
        dict
            ``'n_eff'``: effective number of components ``1 / sum_i w_i^2``
            (inverse Herfindahl index, between 1 and the number of components);
            ``'w1'``: weight of the dominant component (largest eigenvalue);
            ``'T'``: number of observations.
        """
        T = int(self._eigsys.t_)
        if self._eigsys.get_regime == "under":
            v = self._v.reshape(-1)
            norm = float(np.sum(v ** 2))
            w = (v ** 2) / norm if norm > 0 else np.ones(len(v)) / len(v)
        else:
            u = self._u.reshape(-1)
            w = (u ** 2) / T   # sums to 1 since ||Q' 1||^2 = T

        hhi = float(np.sum(w ** 2))
        n_eff = 1.0 / hhi if hhi > 0 else float(len(w))
        w1 = float(w[-1])  # dominant component: eigenvalues are ascending, so it is last
        return {'n_eff': n_eff, 'w1': w1, 'T': T}

    def get_f_tilt_ratio(self) -> np.ndarray:
        """
        Tilt of the response weights towards small eigenvalues, after ridge weighting.

        With kernel ``f(lambda) = 1 / (lambda + z)``, observable
        ``phi(lambda) = lambda / (lambda + z)`` and response weights
        ``w_i = u_i^2 / T``, the ratio is

            E_W[phi] / E_U[phi],
            E_U[phi] = mean_i(phi_i f_i) / mean_i(f_i),
            E_W[phi] = sum_i w_i phi_i f_i / sum_i w_i f_i.

        A ratio below 1 means that the response loads more on small-eigenvalue
        components than uniform weights would. Only available when ``P > T``.

        Returns
        -------
        ndarray of shape (M,)
            The ratio at each grid point.
        """
        z = self.z_grid.reshape(1, -1)                              # (1, M)
        eigs = self._eigsys.get_non_zero_eigenvalues().reshape(-1, 1)  # (T, 1)
        u = self._u.reshape(-1)                                     # (T,)
        T = float(self._eigsys.t_)
        w = (u ** 2) / T                                            # (T,) sums to 1

        f   = 1.0 / (eigs + z)          # (T, M)
        phi = eigs / (eigs + z)         # (T, M)

        # Uniform moments
        M0  = f.mean(axis=0)            # (M,)  (1/T) sum 1/(lambda+z)
        Mla = (phi * f).mean(axis=0)    # (M,)  (1/T) sum lambda/(lambda+z)^2

        # Response-weighted moments
        M0w  = w @ f                    # (M,)  sum w_i/(lambda+z)
        Mlaw = w @ (phi * f)            # (M,)  sum w_i lambda/(lambda+z)^2

        E_U = Mla / np.where(M0  > 0, M0,  1e-30)
        E_W = Mlaw / np.where(M0w > 0, M0w, 1e-30)

        return E_W / np.where(E_U > 0, E_U, 1e-30)

    def get_r2_rank_corr(self, top_k: int = 20) -> float:
        """
        Rank correlation between eigenvalues and signal per unit variance.

        For each component ``i`` let ``R2_i = u_i^2 / lambda_i`` (``u = Q' 1``),
        the signal per unit of variance. Returns the correlation between the
        ranks of ``lambda_i`` and of ``R2_i`` (Spearman correlation, ties not
        averaged) over the ``top_k`` components with the largest eigenvalues.
        A positive value means that, among the leading components, those with
        more variance also carry more signal per unit of variance. Only
        available when ``P > T``.

        Parameters
        ----------
        top_k : int, default=20
            Number of leading components used.

        Returns
        -------
        float
            Correlation in ``[-1, 1]``, or NaN if fewer than 3 components are used.
        """
        eigs = self._eigsys.get_non_zero_eigenvalues()   # (T,) ascending
        u    = self._u.reshape(-1)                       # (T,)
        k    = min(top_k, len(eigs))
        if k < 3:
            return float('nan')
        # The k largest eigenvalues are the last k entries
        eigs_k = eigs[-k:]
        r2_k   = u[-k:] ** 2 / np.where(eigs_k > 0, eigs_k, 1e-30)
        eig_ranks = np.argsort(np.argsort(-eigs_k))
        r2_ranks  = np.argsort(np.argsort(-r2_k))
        return float(np.corrcoef(eig_ranks, r2_ranks)[0, 1])
