"""
Lower and upper bounds on the maximal squared Sharpe ratio (mixin of :class:`GRS`).
"""

import numpy as np
import pandas as pd
from typing import Dict
from RMT.rmt.grs_utils import f_and_grad


class _BoundsMixin:
    """
    Lower and upper bounds on ``SR^2`` built from the GRS statistics.

    Notation: ``B_j`` are the GRS statistics, ``D_J = B_0 + ... + B_J``
    (``D_{-1} = 0``), ``r = B_J / B_{J-1}``, ``Z = Z*(z)`` and
    ``lambda_bar = tr(X'X / T) / P`` (``av_trace`` in the code).

    Lower bounds (order ``J = 1, 2``)
        The ratios ``B_{j+1} / B_j`` increase with ``j``, so extending the
        series geometrically from ``B_J`` underestimates ``SR^2``::

            H_{J,k}   = D_J + B_J (r + r^2 + ... + r^k)
            H_{J,inf} = D_{J-1} + B_J / (1 - r)

    Upper bounds (order ``J = 0, 1, 2``)
        Under the no-good-deal condition (see :mod:`RMT.rmt`), for every
        cut-off ``0 < a < lambda_bar``::

            SR^2 <= U_J(a) = (D_J + Z B_J / a) / (1 - a / lambda_bar)

        The bound is evaluated at

        - ``a_min``, the minimiser of ``U_J`` (columns ``u_J``);
        - ``a_*(k) = Z / (r + ... + r^k)``, for which
          ``U_J(a_*(k)) = H_{J,k} / (1 - a_*(k) / lambda_bar)`` (columns ``full_ubJ_k``);
        - its limit ``a_inf = Z (1 - r) / r``, for which
          ``U_J(a_inf) = H_{J,inf} / (1 - a_inf / lambda_bar)`` (columns ``full_ubJ_inf``);
        - a fixed, user-supplied ``a`` (:meth:`get_all_ubs_fixed_a`).

    Confidence bounds
        Each bound is a smooth function of ``(B_0, ..., B_J)``. Its asymptotic
        variance is ``grad' sigma_theta grad`` (delta method) and the one-sided
        confidence bounds are ``estimate -/+ n_sigma * sqrt(variance / T)``.
        Simultaneous bounds over the grid use the critical values of
        :class:`_MultiZMixin`.

    All estimates and variances are arrays over the z-grid (length ``M``).
    Variances refer to ``sqrt(T)`` times the estimate.
    """

    # -------------------------------------------------------------------------
    # Lower bounds
    # -------------------------------------------------------------------------
    def _compute_lower_bound_estimator_infinite(
            self,
            order: int = 2,
    ):
        """
        Infinite-series lower bound ``H_{J,inf} = D_{J-1} + B_J / (1 - r)`` and its variance.

        Parameters
        ----------
        order : {1, 2}, default=2
            Bound order ``J``.

        Returns
        -------
        lower_bound_est : ndarray of shape (M,)
            ``H_{J,inf}`` on the grid.
        lower_bound_var : ndarray of shape (M,)
            Asymptotic variance of ``sqrt(T) H_{J,inf}``.
        """

        cbs = [self._cb0, self._cb1, self._cb2]

        lower_bound_est = (np.sum(cbs[:order], axis=0) + cbs[order] / (1 - cbs[order] / cbs[order - 1])).flatten()

        grad_h = self._compute_lower_bound_grad(order)
        lower_bound_std = np.einsum('ni,nij,nj->n', grad_h, self.sigma_theta[:, :(order + 1), :(order + 1)], grad_h)
        return lower_bound_est, lower_bound_std

    def _compute_lower_bound_grad(self, order: int):
        """
        Gradient of ``H_{J,inf}`` with respect to ``(B_0, ..., B_J)``.

        Writing ``H_{J,inf} = D_{J-2} + B_{J-1} + B_J B_{J-1} / (B_{J-1} - B_J)``::

            dH/dB_j     = 1                                   for j < J - 1
            dH/dB_{J-1} = 1 - B_J^2 / (B_{J-1} - B_J)^2
            dH/dB_J     = B_{J-1}^2 / (B_{J-1} - B_J)^2

        Parameters
        ----------
        order : {1, 2}
            Bound order ``J``.

        Returns
        -------
        ndarray of shape (M, J + 1)
            Also stored in ``self._lower_bound_grad[order]``.
        """
        cbs = [self._cb0, self._cb1, self._cb2]
        # B_J <= 0.99 B_{J-1} after clipping, so B_{J-1} - B_J > 0
        ones = np.ones((self.z_grid.shape[0], order - 1)).tolist()
        h_j1 = 1 - cbs[order] ** 2 / (cbs[order - 1] - cbs[order]) ** 2
        h_j2 = cbs[order - 1] ** 2 / (cbs[order - 1] - cbs[order]) ** 2
        grad_h = np.concat([ones, h_j1, h_j2], axis=1)
        self._lower_bound_grad[order] = grad_h
        return grad_h

    def lower_b_for_one_power(self, order: int, power: int):
        """
        Truncated lower bound ``H_{J,k} = D_J + B_J (r + ... + r^k)`` and its variance.

        The series is computed with :func:`~RMT.rmt.grs_utils.f_and_grad` as
        ``D_{J-2} + f(B_J, B_{J-1}, k)``, which also gives the gradient.

        Parameters
        ----------
        order : {1, 2}
            Bound order ``J``.
        power : int
            Number ``k`` of extra geometric terms.

        Returns
        -------
        lower_bound_est : ndarray of shape (M, 1)
            ``H_{J,k}`` on the grid.
        lower_bound_var : ndarray of shape (M,)
            Asymptotic variance of ``sqrt(T) H_{J,k}``.
        """
        cbs = [self._cb0, self._cb1, self._cb2]

        lower_bound_est, (h_j1, h_j2) = f_and_grad(cbs[order], cbs[order - 1], power)
        if order >= 2:
           lower_bound_est += np.sum(cbs[:(order - 1)], axis=0)

        # Gradient: 1 for B_0, ..., B_{J-2}, then (df/dB_{J-1}, df/dB_J)
        ones = np.ones((self.z_grid.shape[0], order - 1)).tolist()
        grad_h = np.concat([ones, h_j1, h_j2], axis=1)
        lower_bound_std = np.einsum('ni,nij,nj->n', grad_h, self.sigma_theta[:, :(order + 1), :(order + 1)], grad_h)
        return lower_bound_est, lower_bound_std

    def _compute_lower_bound_estimator(
            self,
            order: int,
    ) -> Dict[int, tuple]:
        """
        Truncated lower bounds of order ``J = order`` for every ``k`` in ``self.k_values``.

        Returns
        -------
        dict
            ``{k: (estimate, variance)}`` as returned by :meth:`lower_b_for_one_power`.
        """
        lower_bound_dict = {k_value: self.lower_b_for_one_power(order, k_value) for k_value in self.k_values}
        return lower_bound_dict

    def get_lower_bound_clt(self) -> Dict[int, Dict[int, tuple]]:
        """
        Truncated lower bounds and their variances.

        Returns
        -------
        dict
            ``{J: {k: (estimate, variance)}}`` for ``J = 1, 2`` and ``k`` in
            ``self.k_values``.
        """
        return self._lower_bound_clt

    def get_all_lbs(self, T: int, annualizer: int = 12, debug: bool = False) -> pd.DataFrame:
        """
        Table of the GRS statistics and all lower bounds on ``SR^2``.

        Parameters
        ----------
        T : int
            Number of observations, used to scale the standard errors.
        annualizer : int, default=12
            Number of periods per year (12 for monthly data, 1 to keep
            per-period units). Squared Sharpe ratios scale linearly with it.
        debug : bool, default=False
            Print ``sqrt(annualizer * bound)`` of the simultaneous bound at
            each grid point.

        Returns
        -------
        pd.DataFrame
            One row per grid point. Every column is multiplied by
            ``annualizer`` (including ``zstar`` and ``zstar_drv1``). Columns
            ending in ``_clt`` subtract ``n_sigma`` standard errors.

            - ``grs_0``, ``grs_1``, ``grs_2`` and ``grs_j_clt``: ``B_j``.
            - ``zstar``, ``zstar_drv1``: ``Z*(z)`` and ``Z*'(z)``.
            - ``full_simple1``, ``full_simple2`` and ``_clt``: partial sums
              ``D_1`` and ``D_2``.
            - ``full_lb{J}_{k}`` and ``_clt``: ``H_{J,k}`` for ``J = 1, 2`` and
              ``k`` in ``self.k_values``.
            - ``full_lb{J}_inf`` and ``_clt``: ``H_{J,inf}``.
            - ``muti_lb{J}``: simultaneous lower bound
              ``max_i (H_{J,inf}(z_i) - q se_i)``, the same in every row;
              ``muti_lb{J}_pt``: ``H_{J,inf}`` at the maximising grid point.
        """
        grs_stats = self.get_grs_statistics()
        full_matr = self.sigma_theta
        zstar = self._Zstar.get_z_star()
        zstar_drv1 = self._Zstar.get_z_star_drv(order=1)
        with np.errstate(invalid='ignore'):
            # Pointwise bounds on each B_j
            errors = [np.sqrt(full_matr[:, i, i].flatten()) / np.sqrt(T) for i in [0, 1, 2]]
            lower_bounds = [(grs_stats[i] - self.n_sigma * errors[i]).reshape(-1, 1) for i in [0, 1, 2]]

            all_grs = np.concatenate([grs_stats[i].reshape(-1, 1) for i in [0, 1, 2]], axis=1)
            all_lb = np.concatenate(lower_bounds, axis=1)
            all = np.concatenate([all_grs, all_lb], axis=1)
            df = pd.DataFrame(all)
            df.columns = ['grs_0', 'grs_1', 'grs_2', 'grs_0_clt', 'grs_1_clt', 'grs_2_clt']
            df['zstar'] = zstar
            df['zstar_drv1'] = zstar_drv1

            # Partial sums D_1, D_2
            sigma_rj = self.get_sigma_rj()
            full_errors = [np.sqrt(sigma_rj[i].flatten()) / np.sqrt(T) for i in [0, 1, 2]]
            full_lower_bound1 = grs_stats[0] + grs_stats[1] - self.n_sigma * full_errors[1]
            full_lower_bound2 = grs_stats[0] + grs_stats[1] + grs_stats[2] - self.n_sigma * full_errors[2]
            df['full_simple1'] = (grs_stats[0] + grs_stats[1]).reshape(-1, 1)
            df['full_simple2'] = (grs_stats[0] + grs_stats[1] + grs_stats[2]).reshape(-1, 1)
            df['full_simple1_clt'] = full_lower_bound1.reshape(-1, 1)
            df['full_simple2_clt'] = full_lower_bound2.reshape(-1, 1)

            # Geometric extension truncated after k terms
            tmp = self._lower_bound_clt
            for order in [1, 2]:
                for k_value in self.k_values:
                    df[f'full_lb{order}_{k_value}'] = tmp[order][k_value][0].reshape(-1, 1)
                    df[f'full_lb{order}_{k_value}_clt'] = tmp[order][k_value][0].reshape(-1, 1) - self.n_sigma * np.sqrt(tmp[order][k_value][1].reshape(-1, 1) / T)

            # Infinite geometric extension
            tmp_inf = self._lower_bound_clt_infinity
            for order in [1, 2]:
                est, var = tmp_inf[order]
                df[f'full_lb{order}_inf'] = est.reshape(-1, 1)
                df[f'full_lb{order}_inf_clt'] = est.reshape(-1, 1) - self.n_sigma * np.sqrt(var.reshape(-1, 1) / T)

            # Simultaneous bound over the grid: best of the multi-z lower bounds
            for order in[1,2]:
                sigma_ii = np.sqrt(self._sigma_J_multi_lb[order].diagonal().reshape(-1, 1))/np.sqrt(T)
                H_i = self._lower_bound_clt_infinity[order][0].reshape(-1, 1)
                clt_vals = (H_i - self._equicoordinate_quantile_lb[order] * sigma_ii).flatten()
                if debug:
                    print(np.sqrt(annualizer*clt_vals))
                i_star = int(np.argmax(clt_vals))
                df[f'muti_lb{order}'] = float(clt_vals[i_star])
                df[f'muti_lb{order}_pt'] = float(H_i.flatten()[i_star])

        return annualizer * df

    def get_all_ubs(self, T: int, annualizer: int = 12, debug = False) -> pd.DataFrame:
        """
        Table of the GRS statistics and all upper bounds on ``SR^2``.

        Parameters
        ----------
        T : int
            Number of observations, used to scale the standard errors.
        annualizer : int, default=12
            Number of periods per year (12 for monthly data, 1 to keep
            per-period units).
        debug : bool, default=False
            Print ``sqrt(annualizer * bound)`` of the simultaneous bound at
            each grid point.

        Returns
        -------
        pd.DataFrame
            One row per grid point. Every column is multiplied by
            ``annualizer`` (including ``zstar``, ``av_trace`` and ``a_star_*``).
            Columns ending in ``_clt`` add ``n_sigma`` standard errors.

            - ``grs_0``, ``grs_1``, ``grs_2``: ``B_j``.
            - ``zstar``, ``zstar_drv1``: ``Z*(z)`` and ``Z*'(z)``.
            - ``av_trace``: ``lambda_bar``.
            - ``u_{J}`` and ``_clt``: ``U_J(a_min)`` for ``J = 0, 1, 2``.
            - ``full_ub{J}_{k}`` and ``_clt``: ``U_J(a_*(k))``;
              ``a_star_{J}_{k}``: the cut-off ``a_*(k)``.
            - ``full_ub{J}_inf`` and ``_clt``: ``U_J(a_inf)``.
            - ``muti_ub{J}``: simultaneous upper bound
              ``min_i (U_J(a_min)(z_i) + q se_i)``, the same in every row;
              ``muti_ub{J}_pt``: ``U_J(a_min)`` at the minimising grid point.
            - ``multi_ub_diagcorr`` and ``_pt``: equal to ``muti_ub1`` unless
              ``U_1(a_min)`` is non-finite or below ``H_{1,inf}`` at the first
              grid point ``z_0`` somewhere on the grid. In that case (the
              "corner" branch) the upper bound becomes ``H_{1,inf}(z_0)`` plus
              ``q`` of its standard errors.

        Notes
        -----
        The cut-offs behind the simultaneous bounds are stored, not annualised:
        ``_muti_ub_i_star[J]`` and ``_muti_ub_a_used[J]`` (selected grid index
        and its ``a_min``), ``_multi_ub_diagcorr_used_corner``,
        ``_multi_ub_diagcorr_i_star`` and ``_multi_ub_diagcorr_a_used`` (``a_min``,
        or ``a_corner`` in the corner branch; NaN if it has no real solution).
        """
        grs_stats = self.get_grs_statistics()
        zstar = self._Zstar.get_z_star()
        zstar_drv1 = self._Zstar.get_z_star_drv(order=1)

        all_grs = np.concatenate([grs_stats[i].reshape(-1, 1) for i in [0, 1, 2]], axis=1)
        df = pd.DataFrame(all_grs, columns=['grs_0', 'grs_1', 'grs_2'])
        df['zstar'] = zstar
        df['zstar_drv1'] = zstar_drv1

        P = int(self._p)
        trSigma = np.sum(self._eigsys.get_eigenvalues())
        av_trace = trSigma / float(P)
        df['av_trace'] = av_trace

        with np.errstate(invalid='ignore'):
            extra = {}
            # Upper bounds at the optimal cut-off a_min
            for J in [0, 1, 2]:
                data = self._upper_bound_data[J]
                u_J = data['u_J']
                sigma_sq = data['sigma_squared']
                extra[f'u_{J}'] = u_J.ravel()
                extra[f'u_{J}_clt'] = (u_J + self.n_sigma * np.sqrt(sigma_sq / T)).ravel()

            # Upper bounds at the implied cut-offs a_*(k)
            for J in [1, 2]:
                for k_value in self.k_values:
                    u_star = self._u_J_star[J][k_value]
                    sigma_sq_star = self._sigma_squared_UB_star[J][k_value]
                    extra[f'full_ub{J}_{k_value}'] = u_star.ravel()
                    extra[f'full_ub{J}_{k_value}_clt'] = (u_star + self.n_sigma * np.sqrt(sigma_sq_star / T)).ravel()
                    extra[f'a_star_{J}_{k_value}'] = self._a_star_data[J][k_value].ravel()

            # Upper bounds at a_inf
            for J in [1, 2]:
                u_inf = self._u_J_infinity[J]
                sigma_sq_inf = self._sigma_squared_UB_infinity[J]
                extra[f'full_ub{J}_inf'] = u_inf.ravel()
                extra[f'full_ub{J}_inf_clt'] = (u_inf + self.n_sigma * np.sqrt(sigma_sq_inf / T)).ravel()

            # Simultaneous bound over the grid: best of the multi-z upper bounds.
            # The cut-off a used at the selected grid point is stored on self
            # (not in df) so that it is not multiplied by the annualizer.
            self._muti_ub_a_used = {}
            self._muti_ub_i_star = {}
            for order in[1,2]:
                sigma_ii = np.sqrt(self._sigma_J_multi_ub[order].diagonal().reshape(-1, 1))/np.sqrt(T)
                H_i = self._upper_bound_data[order]['u_J'].reshape(-1, 1)
                clt_vals = (H_i + self._equicoordinate_quantile_ub[order] * sigma_ii).flatten()
                if debug:
                    print(np.sqrt(clt_vals*annualizer))
                i_star = int(np.argmin(clt_vals))
                df[f'muti_ub{order}'] = float(clt_vals[i_star])
                df[f'muti_ub{order}_pt'] = float(H_i.flatten()[i_star])
                self._muti_ub_i_star[order] = i_star
                self._muti_ub_a_used[order] = float(
                    self._upper_bound_data[order]['a_min'][i_star]
                )

            df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)

            # Simultaneous J = 1 upper bound, with the corner branch
            order = 1
            # Point estimate of the J = 1 infinite lower bound at the first grid point
            lb = self._lower_bound_clt_infinity[order][0][0]

            ub_a_min = self._upper_bound_data[order]['u_J'].reshape(-1, 1)

            # Switch to the corner branch if the a_min upper bound is non-finite
            # or below the lower bound anywhere on the grid (a NaN would make the
            # comparison False and silently skip the branch, hence the isfinite).
            use_a_corner = (~np.isfinite(ub_a_min) | (ub_a_min < lb)).any()
            self._multi_ub_diagcorr_used_corner = bool(use_a_corner)

            # Grid point selected by the simultaneous a_min upper bound; the
            # reported cut-off is evaluated there in both branches.
            i_star_param = self._muti_ub_i_star[order]
            self._multi_ub_diagcorr_i_star = i_star_param

            if use_a_corner:
                # Corner bound: constant in z and equal to the lower bound at
                # z_0, whose covariance matrix has all entries Var(LB(z_0)).
                se = float(np.sqrt(self._sigma_J_multi_ub_corner[0, 0] / T))
                df[f'multi_ub_diagcorr'] = float(
                    lb + self._equicoordinate_quantile_ub_corner * se
                )
                df[f'multi_ub_diagcorr_pt'] = float(lb)
                # a_corner solves U_1(a) = lb at z = z[i_star_param]:
                # (lb / lambda_bar) a^2 + (B_0 + B_1 - lb) a + Z B_1 = 0 (smaller root)
                cb_0_iz   = float(np.asarray(self._cb0).reshape(-1)[i_star_param])
                cb_1_iz   = float(np.asarray(self._cb1).reshape(-1)[i_star_param])
                z_star_iz = float(np.asarray(self._Zstar.get_z_star()).reshape(-1)[i_star_param])
                disc = (cb_0_iz + cb_1_iz - lb) ** 2 \
                       - 4 * cb_1_iz * lb * z_star_iz / self._barlam
                if disc >= 0:
                    a_corner_iz = (self._barlam / (2 * lb)) * (
                        (lb - cb_0_iz - cb_1_iz) - np.sqrt(disc)
                    )
                    self._multi_ub_diagcorr_a_used = float(a_corner_iz)
                else:
                    # No real root at z[i_star]
                    self._multi_ub_diagcorr_a_used = float('nan')

            else:
                # Regular branch: keep the a_min bound
                df[f'multi_ub_diagcorr'] = df[f'muti_ub{order}']
                df[f'multi_ub_diagcorr_pt'] = df[f'muti_ub{order}_pt']
                self._multi_ub_diagcorr_a_used = self._muti_ub_a_used[order]

        return annualizer * df

    # -------------------------------------------------------------------------
    # Upper bound at the optimal cut-off a_min
    # -------------------------------------------------------------------------
    def _compute_a_min_J(self, J: int, eps: float = 1e-8) -> np.ndarray:
        """
        Cut-off ``a_min`` that minimises ``U_J(a) = (D_J + Z B_J / a) / (1 - a / lambda_bar)``.

        Setting ``dU_J/da = 0`` gives ``D_J a^2 + 2 Z B_J a - Z B_J lambda_bar = 0``,
        whose positive root is::

            a_min = (-Z B_J + sqrt((Z B_J)^2 + Z B_J D_J lambda_bar)) / D_J

        Parameters
        ----------
        J : {0, 1, 2}
            Bound order.
        eps : float, default=1e-8
            Unused.

        Returns
        -------
        ndarray of shape (M,)
        """
        Zhat = np.asarray(self._Zstar.get_z_star()).reshape(-1)
        P = int(self._p)
        trSigma = np.sum(self._eigsys.get_eigenvalues())
        av_trace = trSigma / float(P)

        B_list = [self._cb0, self._cb1, self._cb2]
        if J < 0:
            raise ValueError("J must be >= 0.")
        if len(B_list) < J + 1:
            raise ValueError(
                f"Not enough B's available for J={J}. "
                f"Have {len(B_list)} (0..{len(B_list)-1}). "
                "Currently class computes up to cb2; extend _get_B_list() if needed."
            )

        Bmat = np.column_stack(B_list[:J+1])
        n = Bmat.shape[0]
        if Zhat.shape[0] != n:
            raise ValueError(f"Zhat length {Zhat.shape[0]} != B length {n}.")

        D_safe = np.sum(Bmat[:, :J+1], axis=1)

        BJ = Bmat[:, J]

        Q = (BJ * Zhat) ** 2 + (BJ * Zhat) * D_safe * av_trace
        sqrtQ = np.sqrt(Q)

        a_min = (-(BJ * Zhat) + sqrtQ) / D_safe
        return a_min

    def _compute_a_min_J_grad(self, J: int, eps: float = 1e-8) -> np.ndarray:
        """
        Gradient of ``a_min`` with respect to ``(B_0, ..., B_J)``.

        ``a_min`` depends on ``B_0, ..., B_{J-1}`` only through ``D_J``, and on
        ``B_J`` through both ``D_J`` and ``Z B_J``.

        Parameters
        ----------
        J : {0, 1, 2}
            Bound order.
        eps : float, default=1e-8
            Raise a ``ValueError`` if ``D_J < eps`` somewhere on the grid.

        Returns
        -------
        ndarray of shape (M, J + 1)
            Column ``k`` is ``d a_min / d B_k``.
        """
        Zhat = np.asarray(self._Zstar.get_z_star()).reshape(-1)
        P = int(self._p)
        trSigma = np.sum(self._eigsys.get_eigenvalues())
        av_trace = trSigma / float(P)

        B_list = [self._cb0, self._cb1, self._cb2]
        if J < 0:
            raise ValueError("J must be >= 0.")
        if len(B_list) < J + 1:
            raise ValueError(
                f"Not enough B's available for J={J}. "
                f"Have {len(B_list)} (0..{len(B_list)-1}). "
                "Currently class computes up to cb2; extend _get_B_list() if needed."
            )

        Bmat = np.column_stack(B_list[:J+1])
        n = Bmat.shape[0]
        if Zhat.shape[0] != n:
            raise ValueError(f"Zhat length {Zhat.shape[0]} != B length {n}.")

        D_safe = np.sum(Bmat[:, :J+1], axis=1)
        if D_safe.min() < eps:
            raise ValueError("D_safe < eps.")
        BJ = Bmat[:, J]

        # With u = sqrt(B_J Z), v = sqrt(B_J Z + D c) and c = lambda_bar,
        # a_min = c u / (u + v). This form has no subtraction, so the
        # derivatives are free of cancellation:
        #   d a / dD   = -c^2 u / (2 v (u + v)^2)
        #   d a / dB_J =  c^2 Z (D - B_J) / (2 u v (u + v)^2)   (total, B_J is also in D)
        # For J = 0, D = B_J and d a / dB_0 = 0 (a_min does not depend on B_0).
        x = BJ * Zhat                              # u^2
        u = np.sqrt(x)
        v = np.sqrt(x + D_safe * av_trace)
        S = u + v

        grad = np.zeros((n, J+1), dtype=float)

        # k = 0, ..., J-1: only D changes
        if J > 0:
            d_a_dD = -(av_trace ** 2) * u / (2.0 * v * S ** 2)
            grad[:, :J] = d_a_dD[:, None]

        # k = J
        d_a_dBJ = (av_trace ** 2) * Zhat * (D_safe - BJ) / (2.0 * u * v * S ** 2)
        grad[:, J] = d_a_dBJ

        return grad

    def _compute_u_J(self, J: int, eps: float = 1e-8) -> np.ndarray:
        """
        Upper bound ``U_J(a_min) = (D_{J-1} + B_J (a + Z) / a) / (1 - a / lambda_bar)`` at ``a = a_min``.

        ``D_{J-1} + B_J (a + Z) / a`` equals ``D_J + Z B_J / a``.

        Parameters
        ----------
        J : {0, 1, 2}
            Bound order.
        eps : float, default=1e-8
            Unused.

        Returns
        -------
        ndarray of shape (M,)
        """
        if J < 0:
            raise ValueError("J must be >= 0.")

        Zhat = np.asarray(self._Zstar.get_z_star()).reshape(-1)  # (n_z,)
        trSigma = np.sum(self._eigsys.get_eigenvalues())
        P = int(self._p)
        av_trace = trSigma / float(P)

        B_list = [self._cb0, self._cb1, self._cb2]
        if len(B_list) < J + 1:
            raise ValueError(
                f"Not enough B's available for J={J}. "
                f"Have {len(B_list)} (0..{len(B_list)-1})."
            )
        Bmat = np.column_stack(B_list[:J+1])  # (n_z, J+1)
        n = Bmat.shape[0]
        if Zhat.shape[0] != n:
            raise ValueError(f"Zhat length {Zhat.shape[0]} != B length {n}.")

        BJ = Bmat[:, J]

        # D_{J-1} (0 if J = 0)
        if J == 0:
            D_Jm1 = np.zeros(n, dtype=float)
        else:
            D_Jm1 = np.sum(Bmat[:, :J], axis=1)

        a = self._compute_a_min_J(J)

        numer = D_Jm1 + BJ * ((a+ Zhat) / a)
        denom = 1.0 - (a / av_trace)
        return numer / denom

    def _compute_u_J_drv(self, J: int):
        """
        Gradient of ``U_J(a_min)`` with respect to ``(B_0, ..., B_J)``.

        By the envelope theorem (``a_min`` minimises ``U_J``), ``a`` can be held
        fixed at ``a_min``::

            dU/dB_k = 1 / (1 - a / lambda_bar)              for k < J
            dU/dB_J = (1 + Z / a) / (1 - a / lambda_bar)

        Uses ``self._a_min``, which must hold ``a_min`` for the same ``J``.

        Parameters
        ----------
        J : {0, 1, 2}
            Bound order.

        Returns
        -------
        ndarray of shape (M, J + 1)
        """
        Zhat = np.asarray(self._Zstar.get_z_star(), dtype=float).reshape(-1)
        trSigma = float(np.sum(self._eigsys.get_eigenvalues()))
        P = int(self._p)
        av_trace = trSigma / float(P)

        B_list = [self._cb0, self._cb1, self._cb2]
        if J < 0:
            raise ValueError("J must be >= 0.")
        if len(B_list) < J + 1:
            raise ValueError(
                f"Not enough B's available for J={J}. "
                f"Have {len(B_list)} (0..{len(B_list)-1})."
            )

        Bmat = np.column_stack([np.asarray(B, dtype=float).reshape(-1) for B in B_list[:J+1]])
        n = Bmat.shape[0]
        if Zhat.shape[0] != n:
            raise ValueError(f"Zhat length {Zhat.shape[0]} != B length {n}.")

        a = np.asarray(self._a_min, dtype=float).reshape(-1)
        if a.shape[0] != n:
            raise ValueError(f"a length {a.shape[0]} != expected length {n}.")

        denom = 1.0 - a / av_trace

        grad_u = np.zeros((n, J + 1), dtype=float)

        # k = 0, ..., J-1
        if J > 0:
            grad_u[:, :J] = (1.0 / denom)[:, None]

        # k = J
        grad_u[:, J] = (1.0 + Zhat / a) / denom

        return grad_u

    # -------------------------------------------------------------------------
    # Upper bound at the cut-off a_*(k) implied by the truncated series
    # -------------------------------------------------------------------------
    def  _compute_a_star_J(self, J: int, k: int, eps: float = 1e-8) -> np.ndarray:
        """
        Cut-off ``a_*(k) = Z / (r + r^2 + ... + r^k)``, with ``r = B_J / B_{J-1}``.

        For this ``a``, ``Z B_J / a = B_J (r + ... + r^k)`` is the geometric
        extension used by the lower bound ``H_{J,k}``. The powers are summed
        directly (rather than via ``Z (1 - r) / (r (1 - r^k))``), which is
        accurate when ``r`` is close to 1. As ``k -> inf``, ``a_*(k) -> a_inf``.

        Parameters
        ----------
        J : {1, 2}
            Bound order.
        k : int
            Number of geometric terms.
        eps : float, default=1e-8
            Unused.

        Returns
        -------
        ndarray of shape (M,)

        Raises
        ------
        ValueError
            If ``r >= 1`` at some grid point.
        """
        if J < 1:
            raise ValueError("J must be >= 1 for a_star (need B_{J-1}).")

        Zhat = np.asarray(self._Zstar.get_z_star()).reshape(-1)

        B_list = [self._cb0, self._cb1, self._cb2]
        BJ = np.asarray(B_list[J]).reshape(-1)
        BJ_m1_safe = np.asarray(B_list[J - 1]).reshape(-1)
        r = BJ / BJ_m1_safe
        if r.max() >= 1:
            raise ValueError("a_star requires r = B_J / B_{J-1} < 1 on the whole grid.")

        # S(k-1) = sum_{j=0}^{k-1} r^j
        r_col = r.reshape(-1, 1)
        S_km1 = np.sum(np.concatenate([r_col ** j for j in range(k)], axis=1), axis=1)

        # a_star = Z / (r S(k-1))
        rS = r * S_km1

        return Zhat / rS

    def _compute_a_star_J_drv(self, J: int, k: int, eps: float = 1e-8) -> np.ndarray:
        """
        Gradient of ``a_*(k)`` with respect to ``(B_0, ..., B_J)``.

        ``a_*`` depends only on ``r = B_J / B_{J-1}``; the derivative in ``r``
        is chained through ``dr/dB_{J-1} = -r / B_{J-1}`` and ``dr/dB_J = 1 / B_{J-1}``.

        Parameters
        ----------
        J : {1, 2}
            Bound order.
        k : int
            Number of geometric terms.
        eps : float, default=1e-8
            Unused.

        Returns
        -------
        ndarray of shape (M, J + 1)
            Column ``l`` is ``d a_* / d B_l`` (zero for ``l < J - 1``).
        """
        if J < 1:
            raise ValueError("J must be >= 1 for a_star gradient (need B_{J-1}).")

        Zhat = np.asarray(self._Zstar.get_z_star()).reshape(-1)

        B_list = [self._cb0, self._cb1, self._cb2]
        BJ = np.asarray(B_list[J]).reshape(-1)
        BJ_m1_safe = np.asarray(B_list[J - 1]).reshape(-1)

        r = BJ / BJ_m1_safe

        # S(k-1) and its derivative S'(k-1), summed term by term
        r_col = r.reshape(-1, 1)
        S_km1 = np.sum(np.concatenate([r_col ** j for j in range(k)], axis=1), axis=1)
        if k > 1:
            S_km1_prime = np.sum(
                np.concatenate([j * r_col ** (j - 1) for j in range(1, k)], axis=1), axis=1
            )
        else:
            S_km1_prime = np.zeros_like(r)

        rS_safe = r * S_km1

        # a_star = Z / (r S), so d a_star / dr = -Z (S + r S') / (r S)^2
        da_star_dr = -Zhat * (S_km1 + r * S_km1_prime) / rS_safe ** 2

        n = Zhat.shape[0]
        grad = np.zeros((n, J + 1), dtype=float)

        # dr/dB_{J-1} = -r / B_{J-1}
        grad[:, J - 1] = da_star_dr * (-r / BJ_m1_safe)

        # dr/dB_J = 1 / B_{J-1}
        grad[:, J] = da_star_dr * (1.0 / BJ_m1_safe)

        return grad

    def _compute_u_star_J_k(self, J: int, k: int, eps: float = 1e-8) -> np.ndarray:
        """
        Upper bound at the implied cut-off: ``U_J(a_*(k)) = H_{J,k} / (1 - a_*(k) / lambda_bar)``.

        The numerator ``H_{J,k} = D_{J-2} + f(B_J, B_{J-1}, k)`` comes from
        :func:`~RMT.rmt.grs_utils.f_and_grad`.

        Parameters
        ----------
        J : {1, 2}
            Bound order.
        k : int
            Number of geometric terms.
        eps : float, default=1e-8
            ``a_*(k)`` is clipped to ``[eps, 0.99 lambda_bar]`` in the denominator.

        Returns
        -------
        ndarray of shape (M,)
        """
        if J < 1:
            raise ValueError("J must be >= 1 for u_star (need B_{J-1}).")

        trSigma = np.sum(self._eigsys.get_eigenvalues())
        P = int(self._p)
        av_trace = trSigma / float(P)

        B_list = [self._cb0, self._cb1, self._cb2]

        # f(B_J, B_{J-1}, k) = B_{J-1} + B_J (1 + r + ... + r^k)
        f_val, _ = f_and_grad(
            np.asarray(B_list[J]).reshape(-1),
            np.asarray(B_list[J - 1]).reshape(-1), k
        )
        f_val = f_val.flatten()

        # D_{J-2} (0 if J = 1)
        if J >= 2:
            D_Jm2 = np.sum(np.column_stack(B_list[:J - 1]), axis=1)
        else:
            D_Jm2 = 0.0

        numer = D_Jm2 + f_val

        # Keep a inside (0, lambda_bar) so the denominator stays positive
        a = self._a_star_data[J][k]
        a_safe = np.where(a < eps, eps, a)
        a_safe = np.where(a_safe > 0.99 * av_trace, 0.99 * av_trace, a_safe)
        denom_safe = 1.0 - (a_safe / av_trace)

        return numer / denom_safe

    def _compute_u_star_J_k_drv(self, J: int, k: int, eps: float = 1e-8) -> np.ndarray:
        """
        Gradient of ``U_J(a_*(k))`` with respect to ``(B_0, ..., B_J)``.

        Combines the gradient of the numerator (from ``f_and_grad``) with that
        of ``a_*(k)`` in the denominator (quotient rule).

        Parameters
        ----------
        J : {1, 2}
            Bound order.
        k : int
            Number of geometric terms.
        eps : float, default=1e-8
            ``a_*(k)`` is clipped to ``[eps, 0.99 lambda_bar]``.

        Returns
        -------
        ndarray of shape (M, J + 1)
        """
        if J < 1:
            raise ValueError("J must be >= 1 for u_star gradient (need B_{J-1}).")

        trSigma = np.sum(self._eigsys.get_eigenvalues())
        P = int(self._p)
        av_trace = trSigma / float(P)

        B_list = [self._cb0, self._cb1, self._cb2]

        f_val, (df_dx, df_dy) = f_and_grad(
            np.asarray(B_list[J]).reshape(-1),
            np.asarray(B_list[J - 1]).reshape(-1), k
        )
        f_val = f_val.flatten()
        df_dBJ = df_dx.flatten()      # df/dB_J
        df_dBJm1 = df_dy.flatten()    # df/dB_{J-1}

        # D_{J-2} (0 if J = 1)
        if J >= 2:
            D_Jm2 = np.sum(np.column_stack(B_list[:J - 1]), axis=1)
        else:
            D_Jm2 = 0.0

        numer = D_Jm2 + f_val

        a = self._a_star_data[J][k]
        a_safe = np.where(a < eps, eps, a)
        a_safe = np.where(a > 0.99 * av_trace, 0.99 * av_trace, a_safe)
        grad_a = self._a_star_drv_data[J][k]  # (n_z, J+1)

        one_minus = 1.0 - (a_safe / av_trace)

        n = f_val.shape[0]
        grad_u = np.zeros((n, J + 1), dtype=float)

        # u = numer / (1 - a/c):
        # du/dB_l = dnumer/dB_l / (1 - a/c) + numer * (da/dB_l / c) / (1 - a/c)^2

        # l = 0, ..., J-2: dnumer/dB_l = 1, da/dB_l = 0
        if J > 1:
            grad_u[:, :J - 1] = (1.0 / one_minus)[:, None]

        # l = J-1
        da_Jm1 = grad_a[:, J - 1]
        grad_u[:, J - 1] = df_dBJm1 / one_minus + numer * (da_Jm1 / av_trace) / (one_minus ** 2)

        # l = J
        da_J = grad_a[:, J]
        grad_u[:, J] = df_dBJ / one_minus + numer * (da_J / av_trace) / (one_minus ** 2)

        return grad_u

    def _compute_sigma_squared_UB_star_J_k(self, J: int, k: int) -> np.ndarray:
        """Asymptotic variance ``grad' sigma_theta grad`` of ``sqrt(T) U_J(a_*(k))``, shape ``(M,)``."""
        grad = self._u_J_star_drv[J][k]  # (n_z, J+1)
        d = J + 1
        SigmaJ = self.sigma_theta[:, :d, :d]
        return np.einsum("ni,nij,nj->n", grad, SigmaJ, grad)

    # -------------------------------------------------------------------------
    # Upper bound at the limit cut-off a_inf
    # -------------------------------------------------------------------------
    def _compute_a_infinity_J(self, J: int, eps: float = 1e-8) -> np.ndarray:
        """
        Cut-off ``a_inf = lim_{k -> inf} a_*(k) = Z (B_{J-1} - B_J) / B_J = Z (1 - r) / r``.

        Parameters
        ----------
        J : {1, 2}
            Bound order.
        eps : float, default=1e-8
            Unused.

        Returns
        -------
        ndarray of shape (M,)
        """
        if J < 1:
            raise ValueError("J must be >= 1 for a_infinity (need hat_Theta_{J-1}).")

        Zhat = np.asarray(self._Zstar.get_z_star()).reshape(-1)

        B_list = [self._cb0, self._cb1, self._cb2]
        BJ_safe = np.asarray(B_list[J]).reshape(-1)
        BJ_m1 = np.asarray(B_list[J - 1]).reshape(-1)

        return Zhat * (BJ_m1 - BJ_safe) / BJ_safe

    def _compute_a_infinity_J_drv(self, J: int, eps: float = 1e-8) -> np.ndarray:
        """
        Gradient of ``a_inf`` with respect to ``(B_0, ..., B_J)``.

        ``d a_inf / d B_{J-1} = Z / B_J`` and ``d a_inf / d B_J = -Z B_{J-1} / B_J^2``;
        the other entries are zero.

        Returns
        -------
        ndarray of shape (M, J + 1)
        """
        if J < 1:
            raise ValueError("J must be >= 1 for a_infinity gradient (need hat_Theta_{J-1}).")

        Zhat = np.asarray(self._Zstar.get_z_star()).reshape(-1)
        n = Zhat.shape[0]

        B_list = [self._cb0, self._cb1, self._cb2]
        BJ_safe = np.asarray(B_list[J]).reshape(-1)
        BJ_m1 = np.asarray(B_list[J - 1]).reshape(-1)

        grad = np.zeros((n, J + 1), dtype=float)

        # l = J-1
        grad[:, J - 1] = Zhat / BJ_safe

        # l = J
        grad[:, J] = -Zhat * BJ_m1 / BJ_safe ** 2

        return grad

    def _compute_u_infinity_J(self, J: int, eps: float = 1e-8) -> np.ndarray:
        """
        Upper bound at the limit cut-off: ``U_J(a_inf) = H_{J,inf} / (1 - a_inf / lambda_bar)``.

        The numerator is written ``D_{J-2} + B_{J-1}^2 / (B_{J-1} - B_J)``,
        which equals ``H_{J,inf}``.

        Parameters
        ----------
        J : {1, 2}
            Bound order.
        eps : float, default=1e-8
            ``a_inf`` is clipped to ``[eps, 0.99 lambda_bar]``; a ``ValueError``
            is raised if ``B_{J-1} / (B_{J-1} - B_J) > 1 / eps``.

        Returns
        -------
        ndarray of shape (M,)
        """
        if J < 1:
            raise ValueError("J must be >= 1 for u_infinity (need hat_Theta_{J-1}).")

        trSigma = np.sum(self._eigsys.get_eigenvalues())
        P = int(self._p)
        av_trace = trSigma / float(P)

        B_list = [self._cb0, self._cb1, self._cb2]
        BJ = np.asarray(B_list[J]).reshape(-1)
        BJ_m1 = np.asarray(B_list[J - 1]).reshape(-1)

        # f_inf = B_{J-1}^2 / (B_{J-1} - B_J)
        diff = BJ_m1 - BJ
        qq = BJ_m1 / diff
        if qq.max() > 1 / eps:
            raise ValueError("B_{J-1} / (B_{J-1} - B_J) exceeds 1 / eps in u_infinity.")
        f_inf = BJ_m1 * qq

        # D_{J-2} (0 if J = 1)
        if J >= 2:
            D_Jm2 = np.sum(np.column_stack(B_list[:J - 1]), axis=1)
        else:
            D_Jm2 = 0.0

        numer = D_Jm2 + f_inf

        # Keep a inside (0, lambda_bar) so the denominator stays positive
        a = self._a_infinity_data[J]
        a_safe = np.where(a < eps, eps, a)
        a_safe = np.where(a_safe > 0.99 * av_trace, 0.99 * av_trace, a_safe)

        denom_safe = 1.0 - (a_safe / av_trace)

        return numer / denom_safe

    def _compute_u_infinity_J_drv(self, J: int, eps: float = 1e-8) -> np.ndarray:
        """
        Gradient of ``U_J(a_inf)`` with respect to ``(B_0, ..., B_J)``.

        With ``y = B_{J-1}``, ``x = B_J`` and ``f_inf = y^2 / (y - x)``:
        ``df_inf/dy = y (y - 2x) / (y - x)^2`` and ``df_inf/dx = y^2 / (y - x)^2``.
        These are combined with the gradient of ``a_inf`` in the denominator
        (quotient rule).

        Parameters
        ----------
        J : {1, 2}
            Bound order.
        eps : float, default=1e-8
            Same role as in :meth:`_compute_u_infinity_J`.

        Returns
        -------
        ndarray of shape (M, J + 1)
        """
        if J < 1:
            raise ValueError("J must be >= 1 for u_infinity gradient (need hat_Theta_{J-1}).")

        trSigma = np.sum(self._eigsys.get_eigenvalues())
        P = int(self._p)
        av_trace = trSigma / float(P)

        B_list = [self._cb0, self._cb1, self._cb2]
        BJ = np.asarray(B_list[J]).reshape(-1)        # x
        BJ_m1 = np.asarray(B_list[J - 1]).reshape(-1) # y
        n = BJ.shape[0]

        # f_inf = B_{J-1}^2 / (B_{J-1} - B_J)
        diff = BJ_m1 - BJ
        qq = BJ_m1 / diff
        if qq.max() > 1 / eps:
            raise ValueError("B_{J-1} / (B_{J-1} - B_J) exceeds 1 / eps in u_infinity.")
        f_inf = BJ_m1 * qq

        # Partial derivatives of f_inf
        df_dBJm1 = (qq ** 2) * (1 - 2 * BJ / BJ_m1)  # df/dy
        df_dBJ   = qq ** 2                # df/dx

        if J >= 2:
            D_Jm2 = np.sum(np.column_stack(B_list[:J - 1]), axis=1)
        else:
            D_Jm2 = 0.0

        numer = D_Jm2 + f_inf

        a = self._a_infinity_data[J]
        a_safe = np.where(a < eps, eps, a)
        a_safe = np.where(a_safe > 0.99 * av_trace, 0.99 * av_trace, a_safe)

        grad_a = self._a_infinity_drv_data[J]  # (n_z, J+1)

        denom = 1.0 - (a_safe / av_trace)
        denom_safe = np.where(np.abs(denom) < eps, np.sign(denom) * eps + eps, denom)

        grad_u = np.zeros((n, J + 1), dtype=float)

        # l = 0, ..., J-2: dnumer/dB_l = 1, da/dB_l = 0
        if J > 1:
            grad_u[:, :J - 1] = (1.0 / denom_safe)[:, None]

        # l = J-1
        da_Jm1 = grad_a[:, J - 1]
        grad_u[:, J - 1] = df_dBJm1 / denom_safe + numer * (da_Jm1 / av_trace) / denom_safe ** 2

        # l = J
        da_J = grad_a[:, J]
        grad_u[:, J] = df_dBJ / denom_safe + numer * (da_J / av_trace) / denom_safe ** 2

        return grad_u

    def _compute_sigma_squared_UB_infinity_J(self, J: int) -> np.ndarray:
        """Asymptotic variance ``grad' sigma_theta grad`` of ``sqrt(T) U_J(a_inf)``, shape ``(M,)``."""
        grad = self._u_J_infinity_drv[J]  # (n_z, J+1)
        d = J + 1
        SigmaJ = self.sigma_theta[:, :d, :d]
        return np.einsum("ni,nij,nj->n", grad, SigmaJ, grad)

    def get_a_star_data(self) -> Dict[int, Dict[int, np.ndarray]]:
        """Cut-offs ``a_*(k)``: ``{J: {k: ndarray of shape (M,)}}`` for ``J = 1, 2``."""
        return self._a_star_data

    def get_a_infinity_data(self) -> Dict[int, np.ndarray]:
        """Cut-offs ``a_inf``: ``{J: ndarray of shape (M,)}`` for ``J = 1, 2``."""
        return self._a_infinity_data

    # -------------------------------------------------------------------------
    # Variance of U_J(a_min) and accessors
    # -------------------------------------------------------------------------
    def _compute_sigma_squared_UB_J(self, J: int = 2) -> np.ndarray:
        """
        Asymptotic variance ``grad' sigma_theta grad`` of ``sqrt(T) U_J(a_min)``.

        Uses the gradient stored in ``self._u_J_drv``, which must correspond to
        the same ``J``.

        Parameters
        ----------
        J : int, default=2
            Bound order; the top-left ``(J + 1) x (J + 1)`` block of
            ``sigma_theta`` is used.

        Returns
        -------
        ndarray of shape (M,)
        """
        if not hasattr(self, "sigma_theta"):
            raise AttributeError("sigma_theta not found. Expected self.sigma_theta to be computed.")
        if not hasattr(self, "_u_J_drv"):
            raise AttributeError("u_J gradient not found. Expected self._u_J_drv to be computed.")

        grad = np.asarray(self._u_J_drv)
        Sigma = np.asarray(self.sigma_theta)

        if grad.ndim != 2:
            raise ValueError(f"self._u_J_drv must be 2D (n_z, J+1). Got shape {grad.shape}.")
        if Sigma.ndim != 3:
            raise ValueError(f"self.sigma_theta must be 3D (n_z, d, d). Got shape {Sigma.shape}.")

        n_z = Sigma.shape[0]
        if grad.shape[0] != n_z:
            raise ValueError(f"Row mismatch: grad has {grad.shape[0]} rows, Sigma has {n_z}.")

        d = J + 1
        if Sigma.shape[1] < d or Sigma.shape[2] < d:
            raise ValueError(f"sigma_theta is too small for J={J}. Need at least ({d},{d}). Got {Sigma.shape[1:]}.")

        if grad.shape[1] != d:
            raise ValueError(f"Gradient width mismatch: expected {d} columns for J={J}, got {grad.shape[1]}.")

        SigmaJ = Sigma[:, :d, :d]

        # g' Sigma g at each grid point
        sigma_ub_sq = np.einsum("ni,nij,nj->n", grad, SigmaJ, grad)

        return sigma_ub_sq

    # The accessors below return the values for J = 2 (the last order computed
    # in GRS.__init__); use self._upper_bound_data[J] for J = 0, 1.
    def get_a_min(self) -> np.ndarray:
        """Cut-off ``a_min`` for ``J = 2``, shape ``(M,)``."""
        return self._a_min

    def get_a_min_drv(self) -> np.ndarray:
        """Gradient of ``a_min`` with respect to ``(B_0, B_1, B_2)`` for ``J = 2``, shape ``(M, 3)``."""
        return self._a_min_drv

    def get_u_J(self) -> np.ndarray:
        """Upper bound ``U_2(a_min)``, shape ``(M,)``."""
        return self._u_J

    def get_u_J_drv(self) -> np.ndarray:
        """Gradient of ``U_2(a_min)`` with respect to ``(B_0, B_1, B_2)``, shape ``(M, 3)``."""
        return self._u_J_drv

    def get_sigma_squared_UB_J(self) -> np.ndarray:
        """Asymptotic variance of ``sqrt(T) U_2(a_min)``, shape ``(M,)``."""
        return self._sigma_squared_UB_J

    # -------------------------------------------------------------------------
    # Upper bound at a fixed, user-supplied cut-off a
    # -------------------------------------------------------------------------
    def _compute_u_J_fixed_a(self, J: int, a: float, eps: float = 1e-8) -> np.ndarray:
        """
        Upper bound ``U_J(a) = (D_{J-1} + B_J (a + Z) / a) / (1 - a / lambda_bar)`` at a fixed ``a``.

        Parameters
        ----------
        J : {0, 1, 2}
            Bound order.
        a : float
            Cut-off, clipped to ``[eps, 0.99 lambda_bar]``.
        eps : float, default=1e-8
            Lower clipping value for ``a``.

        Returns
        -------
        ndarray of shape (M,)
        """
        if J < 0:
            raise ValueError("J must be >= 0.")

        Zhat = np.asarray(self._Zstar.get_z_star()).reshape(-1)
        trSigma = np.sum(self._eigsys.get_eigenvalues())
        P = int(self._p)
        av_trace = trSigma / float(P)

        B_list = [self._cb0, self._cb1, self._cb2]
        if len(B_list) < J + 1:
            raise ValueError(
                f"Not enough B's available for J={J}. "
                f"Have {len(B_list)} (0..{len(B_list)-1})."
            )
        Bmat = np.column_stack(B_list[:J+1])
        n = Bmat.shape[0]
        if Zhat.shape[0] != n:
            raise ValueError(f"Zhat length {Zhat.shape[0]} != B length {n}.")

        BJ = Bmat[:, J]

        if J == 0:
            D_Jm1 = np.zeros(n, dtype=float)
        else:
            D_Jm1 = np.sum(Bmat[:, :J], axis=1)

        a_safe = max(a, eps)
        a_safe = min(a_safe, 0.99 * av_trace)

        numer = D_Jm1 + BJ * ((a_safe + Zhat) / a_safe)
        denom = 1.0 - (a_safe / av_trace)
        return numer / denom

    def _compute_u_J_fixed_a_drv(self, J: int, a: float, eps: float = 1e-8) -> np.ndarray:
        """
        Gradient of ``U_J(a)`` with respect to ``(B_0, ..., B_J)`` for a fixed ``a``.

        ``dU/dB_k = 1 / (1 - a / lambda_bar)`` for ``k < J`` and
        ``dU/dB_J = (1 + Z / a) / (1 - a / lambda_bar)``.

        Parameters
        ----------
        J : {0, 1, 2}
            Bound order.
        a : float
            Cut-off, clipped to ``[eps, 0.99 lambda_bar]``.
        eps : float, default=1e-8
            Lower clipping value for ``a``.

        Returns
        -------
        ndarray of shape (M, J + 1)
        """
        Zhat = np.asarray(self._Zstar.get_z_star()).reshape(-1)
        trSigma = np.sum(self._eigsys.get_eigenvalues())
        P = int(self._p)
        av_trace = trSigma / float(P)

        B_list = [self._cb0, self._cb1, self._cb2]
        if J < 0:
            raise ValueError("J must be >= 0.")
        if len(B_list) < J + 1:
            raise ValueError(
                f"Not enough B's available for J={J}. "
                f"Have {len(B_list)} (0..{len(B_list)-1})."
            )
        Bmat = np.column_stack(B_list[:J+1])
        n = Bmat.shape[0]
        if Zhat.shape[0] != n:
            raise ValueError(f"Zhat length {Zhat.shape[0]} != B length {n}.")

        a_safe = max(a, eps)
        a_safe = min(a_safe, 0.99 * av_trace)

        one_minus = 1.0 - (a_safe / av_trace)
        Z_over_a = Zhat / a_safe

        grad_u = np.zeros((n, J+1), dtype=float)

        # k = 0, ..., J-1
        if J > 0:
            grad_u[:, :J] = 1.0 / one_minus

        # k = J
        grad_u[:, J] = (1.0 + Z_over_a) / one_minus

        return grad_u

    def get_all_ubs_fixed_a(self, a: float, T: int, annualizer: int = 12) -> pd.DataFrame:
        """
        Table of upper bounds ``U_J(a)`` at a fixed cut-off ``a``, for ``J = 1, 2``.

        Parameters
        ----------
        a : float
            Cut-off, clipped to ``[1e-8, 0.99 lambda_bar]``.
        T : int
            Number of observations, used to scale the standard errors.
        annualizer : int, default=12
            Number of periods per year.

        Returns
        -------
        pd.DataFrame
            One row per grid point, every column multiplied by ``annualizer``:
            ``grs_0``, ``grs_1``, ``grs_2``, ``zstar``, ``zstar_drv1``,
            ``av_trace``, and ``u_{J}``, ``u_{J}_clt`` for ``J = 1, 2``. The
            asymptotic variance in ``u_{J}_clt`` is floored at 1e-3.
        """
        grs_stats = self.get_grs_statistics()
        zstar = self._Zstar.get_z_star()
        zstar_drv1 = self._Zstar.get_z_star_drv(order=1)

        all_grs = np.concatenate([grs_stats[i].reshape(-1, 1) for i in [0, 1, 2]], axis=1)
        df = pd.DataFrame(all_grs, columns=['grs_0', 'grs_1', 'grs_2'])
        df['zstar'] = zstar
        df['zstar_drv1'] = zstar_drv1

        P = int(self._p)
        trSigma = np.sum(self._eigsys.get_eigenvalues())
        av_trace = trSigma / float(P)
        df['av_trace'] = av_trace

        with np.errstate(invalid='ignore'):
            for J in [1, 2]:
                u_J = self._compute_u_J_fixed_a(J=J, a=a)
                grad_u = self._compute_u_J_fixed_a_drv(J=J, a=a)
                # Delta-method variance, floored for stability
                d = J + 1
                SigmaJ = self.sigma_theta[:, :d, :d]
                sigma_sq = np.einsum("ni,nij,nj->n", grad_u, SigmaJ, grad_u)
                sigma_sq = np.where(sigma_sq < 1e-3, 1e-3, sigma_sq)

                df[f'u_{J}'] = u_J.reshape(-1, 1)
                ub_clt = (u_J + self.n_sigma * np.sqrt(sigma_sq / T)).reshape(-1, 1)
                df[f'u_{J}_clt'] = ub_clt

        return annualizer * df
