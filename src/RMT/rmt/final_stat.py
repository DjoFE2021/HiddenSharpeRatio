"""
Entry points: build the GRS pipeline from a return matrix and get the bound tables.
"""

from RMT.linalg.eig import EigenSystem
from RMT.rmt.ZStar import ZStar
from RMT.rmt.grs import GRS
from RMT.rmt.resolvent import Resolvent


def setup_grs(R, z_grid):
    """
    Build the full GRS pipeline for a return matrix.

    Runs :class:`~RMT.linalg.eig.EigenSystem`, :class:`~RMT.rmt.resolvent.Resolvent`,
    :class:`~RMT.rmt.ZStar.ZStar` and :class:`~RMT.rmt.grs.GRS` in sequence.

    Parameters
    ----------
    R : ndarray of shape (T, P)
        Returns, one row per period. Used as is (no demeaning).
    z_grid : ndarray of shape (M,)
        Positive ridge penalties in units of the average eigenvalue
        ``tr(R'R / T) / P``; the penalties actually used are ``z_grid`` times
        that average, which makes the results invariant to the scale of
        ``R``. A common choice is ``(P / T) * RMT.utils.constants.Z_EFF_GRID``.

    Returns
    -------
    GRS
        Fully computed object; query it with ``get_all_lbs``, ``get_all_ubs``
        or ``get_all_ubs_fixed_a``.
    """
    eigensys = EigenSystem(R)
    # Express the ridge grid in units of the average eigenvalue
    trace = eigensys._eigvals.mean()
    resolvent = Resolvent(eigensys, z_grid * trace)
    zstar = ZStar(resolvent)
    grs = GRS(eigensys, resolvent, zstar)
    return grs


def final_stat_lb(R, z_grid):
    """
    Lower-bound table for a return matrix, in per-period units.

    Equivalent to ``setup_grs(R, z_grid).get_all_lbs(T, annualizer=1)``; see
    :meth:`GRS.get_all_lbs` for the columns.
    """
    grs = setup_grs(R, z_grid)
    T = R.shape[0]
    df = grs.get_all_lbs(T, annualizer=1)
    return df


def final_stat_ub(R, z_grid):
    """
    Upper-bound table for a return matrix, in per-period units.

    Equivalent to ``setup_grs(R, z_grid).get_all_ubs(T, annualizer=1)``; see
    :meth:`GRS.get_all_ubs` for the columns.
    """
    grs = setup_grs(R, z_grid)
    T = R.shape[0]
    df = grs.get_all_ubs(T, annualizer=1)
    return df


def get_final_statistics(R, z_grid):
    """
    Lower- and upper-bound tables from a single GRS pipeline, in per-period units.

    Parameters
    ----------
    R : ndarray of shape (T, P)
        Returns, one row per period.
    z_grid : ndarray of shape (M,)
        Ridge penalties in units of the average eigenvalue (see :func:`setup_grs`).

    Returns
    -------
    dict
        ``{'lb': get_all_lbs(...), 'ub': get_all_ubs(...)}``, both with
        ``annualizer=1``.
    """
    grs = setup_grs(R, z_grid)

    lb_df = grs.get_all_lbs(R.shape[0], annualizer=1)
    ub_df = grs.get_all_ubs(R.shape[0], annualizer=1)

    final_df = {
        'lb': lb_df,
        'ub': ub_df
    }

    return final_df
