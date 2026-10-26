"""
GRS statistics and random-matrix-theory bounds on the maximal Sharpe ratio.

Setting
-------
Let ``X`` be a ``T x P`` matrix of (excess) returns with mean ``mu`` and
covariance ``Sigma``. The target is the maximal squared Sharpe ratio
``SR^2 = mu' Sigma^{-1} mu``. Its in-sample estimate (the GRS statistic of
Gibbons, Ross and Shanken) is strongly biased upwards when ``P`` is comparable
to ``T``. The package works with ridge-regularised versions on a grid of
penalties ``z`` and removes the bias with random matrix theory.

For an effective ridge ``Z = Z*(z)`` define

    B_j(z) = Z^j mu' (Sigma + Z I)^{-(j+1)} mu,    j = 0, 1, 2, ...

These terms add up to ``SR^2`` and their ratios ``B_{j+1} / B_j`` increase
with ``j``. The package estimates ``B_0``, ``B_1`` and ``B_2`` and derives:

- lower bounds on ``SR^2``, by extending the series geometrically with ratio
  ``r = B_J / B_{J-1}`` for ``k`` more terms (or ``k = inf``). Because the
  true ratios increase, this underestimates the tail;
- upper bounds on ``SR^2`` under a no-good-deal condition: eigen-directions
  with eigenvalue below a cut-off ``a`` carry at most a share
  ``a / lambda_bar`` of ``SR^2``, where ``lambda_bar = tr(Sigma) / P``. Then

      SR^2 <= (B_0 + ... + B_J + Z B_J / a) / (1 - a / lambda_bar)

  for every ``a`` in ``(0, lambda_bar)``. The bound is evaluated at the
  optimal ``a`` or at the ``a`` implied by the geometric extension;
- confidence bounds from a CLT for the estimated ``B_j`` (delta method),
  both pointwise in ``z`` and simultaneously over the grid.

Modules
-------
final_stat
    :func:`~RMT.rmt.final_stat.setup_grs` builds the whole pipeline from a
    return matrix; small wrappers return the bound tables directly.
grs
    :class:`~RMT.rmt.grs.GRS` computes everything at construction. Its methods
    are split across mixins: ``grs_theta_mixin`` (theta(z) and the B_j),
    ``grs_sigma_mixin`` (pointwise CLT), ``grs_multi_z_mixin`` (joint CLT over
    the grid) and ``grs_bounds_mixin`` (bounds and result tables).
resolvent
    :class:`~RMT.rmt.resolvent.Resolvent`: Stieltjes transform of the sample
    spectrum and the bias term xi(z).
ZStar
    :class:`~RMT.rmt.ZStar.ZStar`: effective ridge parameter Z*(z).
grs_utils
    Truncated geometric series with its gradient.
"""
