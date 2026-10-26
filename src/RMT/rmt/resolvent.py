"""
Stieltjes-transform quantities of the sample spectrum on a grid of ridge penalties.

For the eigenvalues ``lambda_1, ..., lambda_P`` of ``X'X / T`` and the aspect
ratio ``c = P / T``, :class:`Resolvent` evaluates, with their first four
derivatives:

- the Stieltjes transform ``m(w) = (1/P) sum_i 1 / (lambda_i - w)``;
- the companion transform ``m_tilde(w) = (c - 1) / w + c m(w)``;
- the bias term ``xi(z) = (1 - z m(-z)) / (1/c - 1 + z m(-z))``, which
  measures how much estimation noise inflates the in-sample ridge statistics.
  Equivalently ``1 + xi(z) = 1 / D(z)`` with ``D(z) = 1 - c + c z m(-z)``.

Sign convention: the grid is stored as non-positive points ``w = -z``, so
``m`` and ``m_tilde`` (and their derivatives) are functions of ``w``, while
``xi`` and its derivatives are functions of the penalty ``z > 0``.
"""

import numpy as np
from typing import Dict
from scipy.special import factorial

from RMT.linalg.eig import EigenSystem


class Resolvent:
    """
    Stieltjes transform, companion transform and bias term ``xi`` on a grid.

    All quantities are computed at construction (see the module docstring for
    the definitions and the sign convention).

    Parameters
    ----------
    eigensystem : EigenSystem
        Provides the eigenvalues of ``X'X / T`` and the dimensions ``T``, ``P``.
    z_grid : ndarray of shape (M,)
        Ridge penalties, either all non-negative or all non-positive; only
        their absolute values matter.

    Attributes
    ----------
    z_grid : ndarray of shape (M,)
        The grid stored as non-positive points ``w = -|z|``.
    t_ : int
        Number of observations ``T``.
    p_ : int
        Number of features ``P``.
    c_ : float
        Aspect ratio ``P / T``.
    """

    def __init__(
        self,
        eigensystem: EigenSystem,
        z_grid: np.ndarray,
    ) -> None:
        assert (np.any(~np.isnan(z_grid))),(
            "z_grid contains nan values."
        )

        assert (np.all(z_grid >= 0) or np.all(z_grid <= 0)), (
            "z_grid must be all non-negative or all non-positive"
        )

        # Store the grid as w = -|z| <= 0
        self.z_grid: np.ndarray = -z_grid if np.all(z_grid >= 0) else z_grid

        self.t_: int = eigensystem.t_
        self.p_: int = eigensystem.p_
        self.c_: float = self.p_ / self.t_
        self._regime: str = eigensystem._regime
        self._eigensystem: EigenSystem = eigensystem

        # lambda_i - w_j for every grid point j (rows) and eigenvalue i (columns)
        self._eigvals_minus_z: np.ndarray = (
            self._eigensystem.get_eigenvalues()[np.newaxis, :] - self.z_grid[:, np.newaxis]
        )

        # Stieltjes transform m(w) and its derivatives
        self._m_hat: np.ndarray = self._compute_m_hat()
        self._m_hat_drv: Dict[int, np.ndarray] = {
            order: self._compute_m_hat_drv(order) for order in range(1, 5)
        }

        # Companion transform m_tilde(w) and its derivatives
        self._m_tilde: np.ndarray = self._compute_m_tilde()
        self._m_tilde_drv: Dict[int, np.ndarray] = {
            order: self._compute_m_tilde_drv(order) for order in range(1, 5)
        }

        # Bias term xi(z) and its derivatives in z
        self._xi : np.ndarray = self._compute_xi()
        self._xi_drv : Dict[int, np.ndarray] = {
            1: self._compute_xi_drv1(),
            2: self._compute_xi_drv2(),
            3: self._compute_xi_drv3(),
            4: self._compute_xi_drv4()
        }

    # -------------------------------------------------------------------------
    # Stieltjes and companion transforms (functions of w = -z)
    # -------------------------------------------------------------------------
    def _compute_m_hat(self) -> np.ndarray:
        """``m(w) = (1/P) sum_i 1 / (lambda_i - w)`` on the grid, shape ``(M,)``."""
        m_hat: np.ndarray = np.mean(1.0 / self._eigvals_minus_z, axis=1)
        return m_hat

    def _compute_m_hat_drv(self, order: int = 1) -> np.ndarray:
        """``m^(k)(w) = k! (1/P) sum_i (lambda_i - w)^-(k+1)`` for ``k = order``, shape ``(M,)``."""
        power = -order - 1
        vals = self._eigvals_minus_z ** power
        return factorial(order, exact=False) * np.mean(vals, axis=1)

    def _compute_m_tilde(self) -> np.ndarray:
        """``m_tilde(w) = (c - 1) / w + c m(w)`` on the grid, shape ``(M,)``."""
        return (self.c_ - 1.0) / self.z_grid + self.c_ * self._m_hat

    def _compute_m_tilde_drv(self, order: int = 1) -> np.ndarray:
        """``m_tilde^(k)(w) = (-1)^k k! (c - 1) w^-(k+1) + c m^(k)(w)`` for ``k = order``."""
        # k-th derivative of (c - 1) / w
        term1 = (
            ((-1) ** order)
            * factorial(order, exact=False)
            * (self.c_ - 1.0)
            * (self.z_grid ** (-order - 1))
        )

        # k-th derivative of c m(w)
        term2 = self.c_ * self._m_hat_drv[order]

        return term1 + term2

    # -------------------------------------------------------------------------
    # Bias term xi(z) = 1 / D(z) - 1 and its derivatives in z
    # (m, m_z, m_zz, ... below denote m and its w-derivatives evaluated at w = -z)
    # -------------------------------------------------------------------------
    def _compute_xi(
        self
        ) -> np.ndarray:
        """``xi(z) = (1 - z m(-z)) / (1/c - 1 + z m(-z))`` on the grid, shape ``(M,)``."""
        z = -self.z_grid
        m = self._m_hat

        nom = 1.0 - z * m
        den = (1.0 / self.c_) - 1.0 + z * m
        return nom / den

    def _compute_xi_drv1(
        self
        ) -> np.ndarray:
        """
        First derivative ``xi'(z) = -D'(z) / D(z)^2``.

        Here ``D(z) = 1 - c + c z m(-z)`` and ``D'(z) = c A(z)`` with
        ``A(z) = m(-z) - z m'(-z)``.
        """
        c = self.c_
        z = -self.z_grid
        m = self._m_hat
        m_z = self._m_hat_drv[1]

        d = 1.0 - c + c * z * m                 # D(z)
        a = m - z * m_z                         # A(z) = D'(z) / c

        return -c * a / (d ** 2)

    def _compute_xi_drv2(
        self
        ) -> np.ndarray:
        """
        Second derivative ``xi''(z) = (2 D'^2 - D D'') / D^3``.

        Uses ``D' = c A`` and ``D'' = c (z m''(-z) - 2 m'(-z))``.
        """
        c = self.c_
        z = -self.z_grid
        m = self._m_hat
        m_z = self._m_hat_drv[1]
        m_zz = self._m_hat_drv[2]

        d = 1.0 - c + c * z * m
        a = m - z * m_z

        # c (2 m' - z m'') = -D'',  2 c^2 A^2 = 2 D'^2
        numerator = c * (2.0 * m_z - z * m_zz) * d + 2.0 * (c ** 2) * (a ** 2)
        denominator = d ** 3
        return numerator / denominator

    def _compute_xi_drv3(
        self
        ) -> np.ndarray:
        """
        Third derivative ``xi'''(z) = (N' D - 3 N D') / D^4``.

        Obtained by differentiating ``xi'' = N / D^3`` with
        ``N = c (2 c A^2 - A' D)``, where ``A(z) = m(-z) - z m'(-z)``.
        """
        c = self.c_
        z = -self.z_grid

        m = self._m_hat
        m_z = self._m_hat_drv[1]
        m_zz = self._m_hat_drv[2]
        m_zzz = self._m_hat_drv[3]

        d = 1.0 - c + c * z * m              # D(z)
        a = m - z * m_z                      # A(z)
        a_prime = 2.0 * m_z - z * m_zz       # -A'(z)
        a_dash = z * m_zz - 2.0 * m_z        # A'(z)
        a_prime_prime = -3.0 * m_zz + z * m_zzz   # -A''(z)

        d_prime = c * a                      # D'(z)

        # N = c (2 c A^2 - A' D)
        n_val = c * (a_prime * d + 2.0 * c * a * a)

        # N' = c (-A'' D - A' D') + 4 c^2 A A'
        n_prime = c * (a_prime_prime * d + a_prime * d_prime) + 4.0 * (c ** 2) * a * a_dash

        return (n_prime * d - 3.0 * n_val * d_prime) / (d ** 4)

    def _compute_xi_drv4(
        self
        ) -> np.ndarray:
        """
        Fourth derivative of ``xi = 1 / D - 1``.

        ``(1/D)'''' = (24 D'^4 - 36 D D'^2 D'' + 6 D^2 D''^2 + 8 D^2 D' D''' - D^3 D'''') / D^5``.
        """
        c = self.c_
        z = -self.z_grid

        m      = self._m_hat
        m_z    = self._m_hat_drv[1]
        m_zz   = self._m_hat_drv[2]
        m_zzz  = self._m_hat_drv[3]
        m_zzzz = self._m_hat_drv[4]

        # D(z) = 1 - c + c z m(-z)
        d = 1.0 - c + c * z * m

        # Derivatives of D in z
        d_prime   = c * (m - z * m_z)
        d_second  = c * (-2.0 * m_z + z * m_zz)
        d_third   = c * (3.0 * m_zz - z * m_zzz)
        d_fourth  = c * (-4.0 * m_zzz + z * m_zzzz)

        num = (
            24.0 * d_prime**4
            - 36.0 * d * d_prime**2 * d_second
            + 6.0 * d**2 * d_second**2
            + 8.0 * d**2 * d_prime * d_third
            - d**3 * d_fourth
        )

        return num / (d**5)

    # -------------------------------------------------------------------------
    # Getters (all arrays have shape (M,))
    # -------------------------------------------------------------------------
    def get_m_hat(
        self
        ) -> np.ndarray:
        """Stieltjes transform ``m(w)`` on the stored grid ``w = -z``."""
        return self._m_hat

    def get_m_tilde(
        self
        ) -> np.ndarray:
        """Companion transform ``m_tilde(w) = (c - 1) / w + c m(w)`` on the stored grid."""
        return self._m_tilde

    def get_m_hat_drv(
        self,
        order: int = 1
        ) -> np.ndarray:
        """Derivative of ``m`` with respect to ``w`` of the given order (1 to 4)."""
        return self._m_hat_drv[order]

    def get_m_tilde_drv(
        self,
        order: int = 1
        ) -> np.ndarray:
        """Derivative of ``m_tilde`` with respect to ``w`` of the given order (1 to 4)."""
        return self._m_tilde_drv[order]

    def get_xi(
        self
        ) -> np.ndarray:
        """Bias term ``xi(z) = (1 - z m(-z)) / (1/c - 1 + z m(-z))``."""
        return self._xi

    def get_xi_drv(
        self,
        order: int = 1,
        ) -> np.ndarray:
        """Derivative of ``xi`` with respect to ``z`` of the given order (1 to 4)."""
        return self._xi_drv[order]
