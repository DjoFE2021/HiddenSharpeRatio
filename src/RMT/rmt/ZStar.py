"""
Effective ridge parameter ``Z*(z)`` and its derivatives.
"""

import numpy as np

from typing import Dict

from RMT.rmt.resolvent import Resolvent

class ZStar:
    """
    Effective ridge ``Z*(z) = z / (1 - c + c z m(-z)) = z (1 + xi(z))``.

    The debiased ridge statistic computed at penalty ``z`` estimates the
    corresponding population quantity at the larger effective penalty
    ``Z*(z)`` (see :mod:`RMT.rmt`). ``Z*`` and its first four derivatives in
    ``z`` are computed at construction.

    Parameters
    ----------
    resolvent : Resolvent
        Provides the grid, the aspect ratio ``c``, ``m(-z)`` and ``xi(z)``
        with its derivatives.

    Attributes
    ----------
    resolvent : Resolvent
        The resolvent the quantities were computed from.
    """

    def __init__(self,
                 resolvent: Resolvent,
                 ) -> None:
        self.resolvent: Resolvent = resolvent

        self._z_star : np.ndarray = self._compute_z_star()
        self._z_star_drv: Dict[int, np.ndarray] = {
            1: self._compute_z_star_drv1(),
            2: self._compute_z_star_drv2(),
            3: self._compute_z_star_drv3(),
            4: self._compute_z_star_drv4(),
        }

    # Derivatives follow from Z* = z (1 + xi): (Z*)^(k) = k xi^(k-1) + z xi^(k) for k >= 2.
    def _compute_z_star(self) -> np.ndarray:
        """``Z*(z) = z / (1 - c + c z m(-z))`` on the grid, shape ``(M,)``."""
        # The resolvent stores w = -z; recover the positive penalty
        z = -self.resolvent.z_grid
        c = self.resolvent.c_
        m = self.resolvent.get_m_hat()

        out = z / (1.0 - c + c * z * m)

        return out

    def _compute_z_star_drv1(
        self
        ) -> np.ndarray:
        """``(Z*)'(z) = 1 + xi(z) + z xi'(z)``."""
        z = -self.resolvent.z_grid
        return 1.0 + self.resolvent.get_xi() + z * self.resolvent.get_xi_drv(order=1)

    def _compute_z_star_drv2(
        self
        ) -> np.ndarray:
        """``(Z*)''(z) = 2 xi'(z) + z xi''(z)``."""
        z = -self.resolvent.z_grid
        return 2.0 * self.resolvent.get_xi_drv(order=1) + z * self.resolvent.get_xi_drv(order=2)

    def _compute_z_star_drv3(
        self
        ) -> np.ndarray:
        """``(Z*)'''(z) = 3 xi''(z) + z xi'''(z)``."""
        z = -self.resolvent.z_grid
        return 3.0 * self.resolvent.get_xi_drv(order=2) + z * self.resolvent.get_xi_drv(order=3)

    def _compute_z_star_drv4(
        self
        ) -> np.ndarray:
        """``(Z*)''''(z) = 4 xi'''(z) + z xi''''(z)``."""
        z = -self.resolvent.z_grid
        return 4.0 * self.resolvent.get_xi_drv(order=3) + z * self.resolvent.get_xi_drv(order=4)

    def get_z_star(
        self
        ) -> np.ndarray:
        """Effective ridge ``Z*(z)`` on the grid, shape ``(M,)``."""
        return self._z_star

    def get_z_star_drv(
        self,
        order: int
        ) -> np.ndarray:
        """
        Derivative of ``Z*`` with respect to ``z``.

        Parameters
        ----------
        order : {1, 2, 3, 4}
            Derivative order.

        Returns
        -------
        ndarray of shape (M,)

        Raises
        ------
        ValueError
            If ``order`` is not 1, 2, 3 or 4.
        """
        if order not in self._z_star_drv:
            raise ValueError(f"order must be in {{1, 2, 3, 4}}, got {order}.")

        return self._z_star_drv[order]

    @property
    def z_star(self) -> np.ndarray:
        """Effective ridge ``Z*(z)`` on the grid (same as :meth:`get_z_star`)."""
        return self._z_star
