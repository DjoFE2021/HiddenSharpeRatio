"""
The :class:`GRS` class: GRS statistics, their CLT and the Sharpe-ratio bounds.
"""

import numpy as np
from RMT.linalg.eig import EigenSystem
from RMT.rmt.resolvent import Resolvent
from RMT.rmt.ZStar import ZStar
from RMT.rmt.grs_theta_mixin import _ThetaMixin
from RMT.rmt.grs_sigma_mixin import _SigmaMixin
from RMT.rmt.grs_multi_z_mixin import _MultiZMixin
from RMT.rmt.grs_bounds_mixin import _BoundsMixin


class GRS(_ThetaMixin, _SigmaMixin, _MultiZMixin, _BoundsMixin):
    """
    GRS statistics, their CLT and the Sharpe-ratio bounds on a grid of ridge penalties.

    Everything is computed at construction; afterwards the object is only
    queried. For each penalty ``z`` on the grid the object holds:

    1. ``theta(z) = mu_hat' (Sigma_hat + z I)^{-1} mu_hat`` and its first four
       z-derivatives, where ``mu_hat`` and ``Sigma_hat`` are the sample mean
       and covariance of the rows of ``X`` (:class:`_ThetaMixin`);
    2. the debiased estimates ``B_0``, ``B_1``, ``B_2`` (see :mod:`RMT.rmt`);
    3. their asymptotic covariance (:class:`_SigmaMixin`) and the joint
       covariance across grid points (:class:`_MultiZMixin`);
    4. lower and upper bounds on the maximal squared Sharpe ratio with their
       delta-method variances (:class:`_BoundsMixin`).

    Results are read with :meth:`get_all_lbs`, :meth:`get_all_ubs` and
    :meth:`get_all_ubs_fixed_a`. :func:`RMT.rmt.final_stat.setup_grs` builds
    all inputs from a return matrix.

    Parameters
    ----------
    eig_system : EigenSystem
        Eigendecomposition of the ``T x P`` return matrix ``X``.
    resolvent : Resolvent
        Resolvent quantities on the ridge grid, built from ``eig_system``.
    Zstar : ZStar
        Effective ridge ``Z*(z)`` on the same grid.
    n_sigma : float, default=1.960
        Critical value of the one-sided confidence bounds. The simultaneous
        (multi-z) bounds use the matching confidence level ``Phi(n_sigma)``.

    Attributes
    ----------
    z_grid : ndarray of shape (M,)
        Positive ridge penalties (``-resolvent.z_grid``).
    n_sigma : float
        Critical value of the confidence bounds.
    k_values : list of int
        Truncation orders ``k`` of the geometric-series bounds.
    sigma_theta : ndarray of shape (M, 3, 3)
        Asymptotic covariance of ``sqrt(T) (B_0, B_1, B_2)`` at each grid point.

    Notes
    -----
    The response is a vector of ones: the ridge regression of ``y = 1`` on
    ``X`` gives the ridge maximum-Sharpe-ratio portfolio (Britten-Jones, 1999).
    All statistics are per period; multiplying by the number of periods per
    year (``annualizer``) annualises squared Sharpe ratios.

    The simultaneous bounds draw 1,000,000 normal vectors of dimension ``M``,
    so keep the grid small (a handful of points).

    Examples
    --------
    >>> import numpy as np
    >>> from RMT.linalg.eig import EigenSystem
    >>> from RMT.rmt.resolvent import Resolvent
    >>> from RMT.rmt.ZStar import ZStar
    >>> from RMT.rmt.grs import GRS
    >>> X = np.random.default_rng(0).normal(0.05, 1.0, size=(240, 100))
    >>> eig_system = EigenSystem(X)
    >>> resolvent = Resolvent(eig_system, z_grid=np.array([0.01, 0.1, 1.0]))
    >>> grs = GRS(eig_system, resolvent, ZStar(resolvent))
    >>> lbs = grs.get_all_lbs(T=240, annualizer=12)
    >>> ubs = grs.get_all_ubs(T=240, annualizer=12)
    """

    def __init__(
            self,
            eig_system: EigenSystem,
            resolvent: Resolvent,
            Zstar: ZStar,
            n_sigma: float = 1.960,
    ) -> None:
        self._eigsys = eig_system
        self._resolvent = resolvent
        self._Zstar = Zstar
        self.n_sigma = n_sigma
        self.k_values = [2, 4, 6, 8, 10, 12, 14, 15, 16, 18, 20, 30, 40, 50, 70, 100, 1000]

        self._p = self._eigsys.p_
        self._t = self._eigsys.t_

        # Average eigenvalue tr(X'X / T) / P (constant over the grid)
        self._barlam = np.sum(self._eigsys.get_eigenvalues()) / self._p

        # Response y = 1 (T x 1)
        self._y = np.ones([self._eigsys.t_, 1])

        # The resolvent stores the grid with a negative sign; GRS works with z > 0
        self.z_grid = -resolvent.z_grid

        # Coordinates of the response in the eigenbasis
        if self._eigsys.get_regime == "under":
            # v = Q' mu_hat with mu_hat = X'y / T (feature space)
            self._bf = (self._eigsys.get_x.T @ self._y) / self._t
            self._v = self._eigsys.get_q.T @ self._bf
        else:
            # u = Q' y (sample space)
            self._u = self._eigsys.get_q.T @ self._y

        # theta(z) and its first four derivatives
        self._theta = self._compute_theta_hat()
        self._theta_drv1 = self._compute_theta_hat_prime()
        self._theta_drv2 = self._compute_theta_hat_second()
        self._theta_drv3 = self._compute_theta_hat_third()
        self._theta_drv4 = self._compute_theta_hat_fourth()

        # xi(z) and Z*(z) with their first four derivatives
        xi = self._resolvent.get_xi()
        xi_drv1 = self._resolvent.get_xi_drv(order=1)
        xi_drv2 = self._resolvent.get_xi_drv(order=2)
        xi_drv3 = self._resolvent.get_xi_drv(order=3)
        xi_drv4 = self._resolvent.get_xi_drv(order=4)

        zstar = self._Zstar.get_z_star()
        zstar_drv1 = self._Zstar.get_z_star_drv(order=1)
        zstar_drv2 = self._Zstar.get_z_star_drv(order=2)
        zstar_drv3 = self._Zstar.get_z_star_drv(order=3)
        zstar_drv4 = self._Zstar.get_z_star_drv(order=4)

        # Coefficient functions of the CLT covariance and their derivatives:
        #   a(z) = 4 z (theta - xi),  b(z) = 2 z xi,  c(z) = 16 Z*(z).
        # Derivatives of z * g(z) use the Leibniz rule (z g)^(k) = k g^(k-1) + z g^(k).
        dtype = np.longdouble

        z = np.asarray(self.z_grid, dtype=dtype)

        # g_k = k-th derivative of theta - xi
        g0 = np.asarray(self._theta,      dtype=dtype) - np.asarray(xi,      dtype=dtype)
        g1 = np.asarray(self._theta_drv1, dtype=dtype) - np.asarray(xi_drv1, dtype=dtype)
        g2 = np.asarray(self._theta_drv2, dtype=dtype) - np.asarray(xi_drv2, dtype=dtype)
        g3 = np.asarray(self._theta_drv3, dtype=dtype) - np.asarray(xi_drv3, dtype=dtype)
        g4 = np.asarray(self._theta_drv4, dtype=dtype) - np.asarray(xi_drv4, dtype=dtype)

        self._a      = 4*(z * g0)
        self._a_drv1 = 4*(g0 + z * g1)
        self._a_drv2 = 4*(2.0 * g1 + z * g2)
        self._a_drv3 = 4*(3.0 * g2 + z * g3)
        self._a_drv4 = 4*(4.0 * g3 + z * g4)

        self._b = 2*self.z_grid * xi
        self._b_drv1 = 2*(xi + self.z_grid * xi_drv1)
        self._b_drv2 = 2*(2 * xi_drv1 + self.z_grid * xi_drv2)
        self._b_drv3 = 2*(3 * xi_drv2 + self.z_grid * xi_drv3)
        self._b_drv4 = 2*(4 * xi_drv3 + self.z_grid * xi_drv4)

        self._c = 16*zstar
        self._c_drv1 = 16*zstar_drv1
        self._c_drv2 = 16*zstar_drv2
        self._c_drv3 = 16*zstar_drv3
        self._c_drv4 = 16*zstar_drv4

        # Pointwise CLT and the statistics B_0, B_1, B_2
        self._Xi_clt = self._compute_Xi_clt()
        self._A_T_clt = self._compute_A_T_clt()
        self._tilde_A_T_clt = self._compute_tilde_A_T_clt()
        self.sigma_theta = self._compute_sigma_theta()
        self._sigma_rj = self._compute_sigma_rj()
        self._compute_grs_statistics()

        # Lower bounds for J = 1, 2: truncated after k terms, and k = infinity.
        # _lower_bound_grad is filled by the k = infinity computation.
        self._lower_bound_clt = {
            1: self._compute_lower_bound_estimator(order=1),
            2: self._compute_lower_bound_estimator(order=2)
        }
        self._lower_bound_grad = {}
        self._lower_bound_clt_infinity = {
            1: self._compute_lower_bound_estimator_infinite(order=1),
            2: self._compute_lower_bound_estimator_infinite(order=2)
        }

        # Upper bounds at the optimal cut-off a_min for J = 0, 1, 2. The
        # attributes _a_min, _u_J, ... keep the values of the last loop (J = 2).
        self._upper_bound_data = {}

        for J in [0, 1, 2]:
            self._a_min = self._compute_a_min_J(J=J)
            self._a_min_drv = self._compute_a_min_J_grad(J=J)
            self._u_J = self._compute_u_J(J=J)
            self._u_J_drv = self._compute_u_J_drv(J=J)
            self._sigma_squared_UB_J = self._compute_sigma_squared_UB_J(J=J)
            self._upper_bound_data[J] = {
                'a_min': self._a_min,
                'a_min_drv': self._a_min_drv,
                'u_J': self._u_J,
                'u_J_drv': self._u_J_drv,
                'sigma_squared': self._sigma_squared_UB_J,
            }

        # Upper bounds at the cut-off a_*(k) implied by the truncated series (J = 1, 2)
        self._a_star_data = {}
        self._a_star_drv_data = {}
        self._u_J_star = {}
        self._u_J_star_drv = {}
        self._sigma_squared_UB_star = {}
        for J in [1, 2]:
            self._a_star_data[J] = {
                k: self._compute_a_star_J(J=J, k=k) for k in self.k_values
            }
            self._a_star_drv_data[J] = {
                k: self._compute_a_star_J_drv(J=J, k=k) for k in self.k_values
            }
            self._u_J_star[J] = {
                k: self._compute_u_star_J_k(J=J, k=k) for k in self.k_values
            }
            self._u_J_star_drv[J] = {
                k: self._compute_u_star_J_k_drv(J=J, k=k) for k in self.k_values
            }
            self._sigma_squared_UB_star[J] = {
                k: self._compute_sigma_squared_UB_star_J_k(J=J, k=k) for k in self.k_values
            }

        # ... and at its limit a_inf = lim_{k -> inf} a_*(k)
        self._a_infinity_data = {
            J: self._compute_a_infinity_J(J=J) for J in [1, 2]
        }
        self._a_infinity_drv_data = {
            J: self._compute_a_infinity_J_drv(J=J) for J in [1, 2]
        }
        self._u_J_infinity = {
            J: self._compute_u_infinity_J(J=J) for J in [1, 2]
        }
        self._u_J_infinity_drv = {
            J: self._compute_u_infinity_J_drv(J=J) for J in [1, 2]
        }
        self._sigma_squared_UB_infinity = {
            J: self._compute_sigma_squared_UB_infinity_J(J=J) for J in [1, 2]
        }

        # Joint CLT across the grid, then the critical values of the
        # simultaneous (multi-z) bounds for J = 1, 2
        self._Xi_multi = self._compute_Xi_multi()
        self._A_Xi_A_multi = {order : np.zeros((self.z_grid.shape[0], self.z_grid.shape[0], order+1, order+1), dtype=np.longdouble) for order in range(1,3)}
        for order in range(1, 3):
            self._compute_A_Xi_A_multi(order = order)

        self._sigma_J_multi_lb = {order: self._compute_sigma_J_multi_lb(order=order) for order in range(1,3)}
        self._sigma_J_multi_ub = {order: self._compute_sigma_J_multi_ub(order=order) for order in range(1,3)}

        self._R_sigma_H_lb = {order: self._compute_R_sigma_H_lb(order = order) for order in range(1,3)}
        self._R_sigma_H_ub = {order:self._compute_R_sigma_H_ub(order = order) for order in range(1,3)}

        self._equicoordinate_quantile_lb = {order: self._compute_equicoordinate_quantile_lb(order) for order in range(1, 3)}
        self._equicoordinate_quantile_ub = {order: self._compute_equicoordinate_quantile_ub(order) for order in range(1, 3)}

        # Corner branch of the simultaneous upper bound (J = 1 only):
        # gradient, covariance, correlation and critical value
        self._compute_grad_H_u_a_corner()
        self._sigma_J_multi_ub_corner = self._compute_sigma_J_multi_ub_corner()
        self._R_sigma_H_ub_corner = self._compute_R_sigma_H_ub_corner()
        self._equicoordinate_quantile_ub_corner = self._compute_equicoordinate_quantile_ub_corner()
