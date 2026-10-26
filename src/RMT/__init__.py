"""
RMT: random-matrix-theory inference on the maximal Sharpe ratio of many assets.

Given a ``T x P`` matrix of returns, the library estimates ridge-type GRS
statistics, removes their in-sample bias using random matrix theory, and turns
them into lower and upper confidence bounds on the maximal squared Sharpe ratio
``mu' Sigma^{-1} mu`` attainable from the ``P`` assets.

Subpackages
-----------
RMT.rmt
    GRS statistics, their CLT and the Sharpe-ratio bounds. The usual entry
    point is :func:`RMT.rmt.final_stat.setup_grs`.
RMT.linalg
    Eigendecomposition of the data matrix and small linear-algebra helpers.
RMT.portfolios
    Building managed (random-feature) portfolios from stock-level panel data.
RMT.utils
    Shared constants.
"""
