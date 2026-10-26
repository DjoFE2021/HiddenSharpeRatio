"""
Managed portfolios built from stock-level panel data.

Panel inputs (characteristics, returns) are row-aligned and indexed by a
MultiIndex with a level named ``"date"`` (the period) and an asset identifier.
A signal ``s`` defines a managed portfolio whose return at date ``t`` is the
cross-sectional mean ``(1 / N_t) sum_i s_{i,t} r_{i,t}``, where ``r`` is the
return aligned with the signal (typically the next-period return). The
resulting ``T x P`` matrix of managed returns is the input of
:func:`RMT.rmt.final_stat.setup_grs`.

random_feature_portfolios
    Random-feature signals (random projections of the characteristics followed
    by a non-linearity), managed returns, ridge maximum-Sharpe-ratio portfolios
    and their rolling out-of-sample returns. Requires PyTorch.
residualize
    Per-date residualisation and whitening of signals or returns. Requires PyTorch.
preprocess
    Cross-sectional rank transform of characteristics.
"""
