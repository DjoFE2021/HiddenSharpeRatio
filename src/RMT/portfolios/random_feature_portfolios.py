"""
Random-feature managed portfolios and ridge maximum-Sharpe-ratio portfolios.

Panel inputs (signals, returns) are row-aligned and indexed by a MultiIndex
with a level named ``"date"``. Random features map the ``d`` characteristics of
each asset to ``P`` signals ``act(X omega')``, with Gaussian weights ``omega``
of variance ``2 / d`` (He initialisation). Each signal defines a managed
portfolio whose return is the cross-sectional mean of signal times return at
each date.

Computations run in PyTorch, on the GPU when one is available.
"""

import torch
import pandas as pd
from RMT.utils.constants import Z_EFF_GRID
import numpy as np

def get_device(device=None):
    """
    Return ``torch.device(device)``, or CUDA if available (else CPU) when ``device`` is None.

    Parameters
    ----------
    device : str or torch.device, optional
        Device specification such as ``'cpu'``, ``'cuda'`` or ``'cuda:0'``.

    Returns
    -------
    torch.device
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def average_trace(X):
    """
    Mean of the squared entries of ``X``, i.e. the average eigenvalue of ``X'X / n``.

    Used to express ridge penalties in units of the data scale.

    Parameters
    ----------
    X : torch.Tensor, pd.DataFrame or np.ndarray
        Matrix of shape ``(n, p)``.

    Returns
    -------
    float
    """
    if isinstance(X, torch.Tensor):
        return (X ** 2).mean().item()
    elif isinstance(X, pd.DataFrame):
        return (X.values ** 2).mean()
    else:
        return (X ** 2).mean()

def compute_random_features(
    signals,
    P,
    num_seeds=1,
    scale=1.0,
    activation="relu",
    base_seed=0,
    device=None,
    dtype=torch.float32,
):
    """
    Random features ``act(X omega')`` of the input signals.

    For each seed ``s``, ``omega`` is a ``P x d`` matrix with i.i.d.
    ``N(0, scale^2 * 2 / d)`` entries drawn from a generator seeded with
    ``base_seed + s``. The blocks of all seeds are concatenated.

    Parameters
    ----------
    signals : pd.DataFrame or np.ndarray
        Characteristics of shape ``(NT, d)`` (one row per asset-date).
    P : int
        Number of random features per seed.
    num_seeds : int, default=1
        Number of independent weight draws.
    scale : float, default=1.0
        Multiplier of the weight standard deviation.
    activation : {'relu', 'gelu', 'sincos'}, default='relu'
        Non-linearity. ``'sincos'`` returns ``sqrt(2) sin`` and ``sqrt(2) cos``
        features, doubling the number of features.
    base_seed : int, default=0
        Seed of the first draw.
    device : str or torch.device, optional
        Device; auto-detected if None.
    dtype : torch.dtype, default=torch.float32
        Computation dtype.

    Returns
    -------
    torch.Tensor
        Features of shape ``(NT, P * num_seeds)``, or ``(NT, 2 * P * num_seeds)``
        for ``'sincos'``.

    Examples
    --------
    >>> signals = np.random.randn(1000, 10)
    >>> compute_random_features(signals, P=50, num_seeds=2).shape
    torch.Size([1000, 100])
    """
    device = get_device(device)

    X = torch.as_tensor(
        signals.values if isinstance(signals, pd.DataFrame) else signals,
        device=device,
        dtype=dtype,
    )
    NT, d = X.shape

    Z_blocks = []

    for seed in range(num_seeds):
        g = torch.Generator(device=device)
        g.manual_seed(base_seed + seed)

        # Random weights with He scaling: Var = scale^2 * 2 / d
        omega = (
            scale
            * torch.randn((P, d), generator=g, device=device, dtype=dtype)
            * (2.0 / d) ** 0.5
        )

        Z = X @ omega.T

        if activation == "relu":
            Z = torch.relu(Z)
        elif activation == "gelu":
            Z = torch.nn.functional.gelu(Z)
        elif activation == "sincos":
            Z = torch.cat(
                [torch.sqrt(torch.tensor(2.0, device=device)) * torch.sin(Z),
                 torch.sqrt(torch.tensor(2.0, device=device)) * torch.cos(Z)],
                dim=1,
            )
        else:
            raise ValueError(f"Unknown activation {activation}")

        Z_blocks.append(Z)

    return torch.cat(Z_blocks, dim=1)

def build_managed_returns(returns, signals):
    """
    Managed-portfolio returns: cross-sectional mean of signal times return at each date.

    ``managed[t, p] = mean_i signals[(t, i), p] * returns[(t, i)]``.

    Parameters
    ----------
    returns : pd.Series
        Asset returns of length ``NT``, indexed by a MultiIndex with a
        ``"date"`` level.
    signals : torch.Tensor, pd.DataFrame or np.ndarray
        Portfolio weights of shape ``(NT, P)``, row-aligned with ``returns``.

    Returns
    -------
    pd.DataFrame
        Managed returns of shape ``(T, P)``, indexed by date.

    Examples
    --------
    >>> idx = pd.MultiIndex.from_product([pd.date_range('2020-01-01', periods=100), range(50)],
    ...                                  names=['date', 'id'])
    >>> returns = pd.Series(np.random.randn(5000), index=idx)
    >>> build_managed_returns(returns, torch.randn(5000, 10)).shape
    (100, 10)
    """
    if not isinstance(signals, torch.Tensor):
        if isinstance(signals, pd.DataFrame):
            signals = signals.values
        signals = torch.as_tensor(signals)

    dates = returns.index.get_level_values("date")
    r = torch.as_tensor(returns.values, device=signals.device).view(-1, 1)

    weighted = signals * r
    df = pd.DataFrame(
        weighted.cpu().numpy(),
        index=returns.index,
    )

    return df.groupby(dates).mean()

def ridge_regr(
    signals: torch.Tensor,
    labels: torch.Tensor,
    future_signals: torch.Tensor,
    shrinkage_list,
):
    """
    Ridge regression for several penalties at once, via one eigendecomposition.

    For each penalty ``lambda``:
    ``beta(lambda) = (S'S / t + lambda I)^{-1} S'y / t``. The ``p x p`` matrix
    ``S'S / t`` is decomposed when ``p <= t``, and the ``t x t`` matrix
    ``SS' / t`` otherwise (using ``(S'S/t + lambda)^{-1} S' = S' (SS'/t + lambda)^{-1}``).

    With ``labels`` a vector of ones and ``signals`` a matrix of returns, the
    coefficients are the ridge maximum-Sharpe-ratio portfolio weights
    (Britten-Jones, 1999).

    Parameters
    ----------
    signals : torch.Tensor
        Training regressors ``S`` of shape ``(t, p)``.
    labels : torch.Tensor
        Training targets ``y`` of shape ``(t, 1)``.
    future_signals : torch.Tensor
        Regressors to predict, of shape ``(t_test, p)`` (or ``(p,)``).
    shrinkage_list : array-like of shape (L,)
        Positive penalties ``lambda``.

    Returns
    -------
    betas : torch.Tensor
        Coefficients of shape ``(p, L)``, one column per penalty.
    preds : torch.Tensor
        ``future_signals @ betas``, of shape ``(t_test, L)``.
    """
    device = signals.device
    dtype = signals.dtype

    t_, p_ = signals.shape

    # Penalties on the same device and dtype as the data
    shrinkage = torch.as_tensor(
        shrinkage_list,
        device=device,
        dtype=dtype,
    )

    if p_ <= t_:
        # Feature space: eigendecomposition of S'S / t
        G = (signals.T @ signals) / t_
        eigvals, eigvecs = torch.linalg.eigh(G)

        mu = (signals.T @ labels) / t_
        U_mu = eigvecs.T @ mu

        inv = 1.0 / (eigvals[:, None] + shrinkage[None, :])
        betas = eigvecs @ (inv * U_mu)

    else:
        # Sample space: eigendecomposition of SS' / t, then map back with S'
        G = (signals @ signals.T) / t_
        eigvals, eigvecs = torch.linalg.eigh(G)

        mu = labels / t_
        U_mu = eigvecs.T @ mu

        inv = 1.0 / (eigvals[:, None] + shrinkage[None, :])

        tmp = eigvecs.T @ signals
        betas = tmp.T @ (inv * U_mu)

    preds = future_signals @ betas
    return betas, preds


def rolling_msrr(
    rets: pd.DataFrame,
    window=120,
    device=None,
):
    """
    Rolling out-of-sample returns of ridge maximum-Sharpe-ratio portfolios.

    At each date ``i >= window``, a vector of ones is ridge-regressed on the
    previous ``window`` rows of ``rets`` (missing values set to 0), with
    penalties ``z * average_trace(R)`` for ``z`` in ``(P / window) * Z_EFF_GRID``.
    The resulting portfolios are applied to the returns of date ``i``.

    Parameters
    ----------
    rets : pd.DataFrame
        Returns of shape ``(T, P)``, indexed by date.
    window : int, default=120
        Length of the estimation window.
    device : str or torch.device, optional
        Device; auto-detected if None.

    Returns
    -------
    pd.DataFrame
        Out-of-sample portfolio returns of shape ``(T - window, len(Z_EFF_GRID))``,
        indexed by ``rets.index[window:]``, with one column per value of
        ``Z_EFF_GRID``. Missing returns at the out-of-sample date give NaN.
    """
    device = get_device(device)

    T = window
    P = rets.shape[1]
    z_grid = (P / T) * Z_EFF_GRID
    dates = rets.index
    results = []

    for i in range(window, len(dates)):
        # Estimation window and out-of-sample row
        R = torch.tensor(
            rets.iloc[i - window:i].fillna(0).values,
            device=device,
            dtype=torch.float32,
        )
        oos = torch.tensor(
            rets.iloc[i].values,
            device=device,
            dtype=torch.float32,
        )

        _, optimal = ridge_regr(
            signals=R,
            labels=torch.ones((R.shape[0], 1), device=device),
            future_signals=oos,
            shrinkage_list=z_grid *average_trace(R),
        )

        results.append(optimal.squeeze().cpu().numpy())

    return pd.DataFrame(results, index=dates[window:], columns=Z_EFF_GRID)

def produce_random_feature_managed_returns(
    P,
    r1,
    signals,
    num_seeds=2,
    scale=1.0,
    activation="relu",
    return_features=False,
    device=None,
):
    """
    Managed returns of random-feature portfolios (all features held in memory).

    Combines :func:`compute_random_features` (with ``base_seed=0``) and
    :func:`build_managed_returns`.

    Parameters
    ----------
    P : int
        Number of random features per seed.
    r1 : pd.Series
        Asset returns of length ``NT``, indexed by a MultiIndex with a ``"date"`` level.
    signals : pd.DataFrame or np.ndarray
        Characteristics of shape ``(NT, d)``, row-aligned with ``r1``.
    num_seeds : int, default=2
        Number of independent weight draws.
    scale : float, default=1.0
        Multiplier of the weight standard deviation.
    activation : {'relu', 'gelu', 'sincos'}, default='relu'
        Non-linearity.
    return_features : bool, default=False
        Also return the features.
    device : str or torch.device, optional
        Device; auto-detected if None.

    Returns
    -------
    managed : pd.DataFrame
        Managed returns of shape ``(T, P * num_seeds)`` (twice as many columns
        for ``'sincos'``).
    features : pd.DataFrame
        Only if ``return_features``: the features, of shape ``(NT, P * num_seeds)``.
    """
    device = get_device(device)

    Z = compute_random_features(
        signals,
        P=P,
        num_seeds=num_seeds,
        scale=scale,
        activation=activation,
        device=device,
    )

    managed = build_managed_returns(r1, Z)

    if return_features:
        Z_df = pd.DataFrame(Z.cpu().numpy())
        return managed, Z_df

    return managed

def produce_random_feature_managed_returns_chunked(
    P,
    r1,
    signals,
    num_seeds=2,
    scale=1.0,
    activation="relu",
    base_seed=0,
    device=None,
    dtype=torch.float32,
):
    """
    Managed returns of random-feature portfolios, one seed at a time.

    Builds the same portfolios as :func:`produce_random_feature_managed_returns`
    (same seeds and weights) but collapses each ``NT x P`` block of features to
    ``T x P`` managed returns before drawing the next seed, so that the full
    ``NT x (P * num_seeds)`` feature matrix is never held in memory. Returns
    are cast to ``dtype``.

    Parameters
    ----------
    P : int
        Number of random features per seed.
    r1 : pd.Series
        Asset returns of length ``NT``, indexed by a MultiIndex with a ``"date"`` level.
    signals : pd.DataFrame or np.ndarray
        Characteristics of shape ``(NT, d)``, row-aligned with ``r1``.
    num_seeds : int, default=2
        Number of independent weight draws.
    scale : float, default=1.0
        Multiplier of the weight standard deviation.
    activation : {'relu', 'gelu', 'sincos'}, default='relu'
        Non-linearity.
    base_seed : int, default=0
        Seed of the first draw.
    device : str or torch.device, optional
        Device; auto-detected if None.
    dtype : torch.dtype, default=torch.float32
        Computation dtype.

    Returns
    -------
    pd.DataFrame
        Managed returns of shape ``(T, P * num_seeds)`` (twice as many columns
        for ``'sincos'``), seeds concatenated along the columns.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Inputs are moved to the device once
    X = torch.as_tensor(
        np.array(signals.values if isinstance(signals, pd.DataFrame) else signals),
        device=device,
        dtype=dtype,
    )
    r = torch.as_tensor(
        np.array(r1.values),
        device=device,
        dtype=dtype,
    ).view(-1, 1)

    dates = r1.index.get_level_values("date")

    NT, d = X.shape
    sqrt2_over_d = (2.0 / d) ** 0.5

    managed_blocks = []

    for seed in range(num_seeds):
        g = torch.Generator(device=device)
        g.manual_seed(base_seed + seed)

        # Random weights (P x d), He scaling
        omega = (
            scale
            * torch.randn((P, d), generator=g, device=device, dtype=dtype)
            * sqrt2_over_d
        )

        # Features (NT x P)
        Z = X @ omega.T

        if activation == "relu":
            Z = torch.relu(Z)
        elif activation == "gelu":
            Z = torch.nn.functional.gelu(Z)
        elif activation == "sincos":
            Z = torch.cat(
                [torch.sqrt(torch.tensor(2.0, device=device)) * torch.sin(Z),
                 torch.sqrt(torch.tensor(2.0, device=device)) * torch.cos(Z)],
                dim=1,
            )
        else:
            raise ValueError(f"Unknown activation '{activation}'")

        # Collapse to managed returns (T x P) right away
        weighted = Z * r
        df = pd.DataFrame(
            weighted.cpu().numpy(),
            index=r1.index,
        )

        managed = df.groupby(dates).mean()
        managed_blocks.append(managed)

        # Free the large intermediates before the next seed (matters on GPU)
        del omega, Z, weighted, df
        torch.cuda.empty_cache()

    return pd.concat(managed_blocks, axis=1)


def sharpe_ratio(rets):
    """
    Annualised Sharpe ratio of monthly returns: ``sqrt(12) * mean / std``.

    No risk-free rate is subtracted. The standard deviation uses ``ddof=1``
    for pandas inputs and ``ddof=0`` for NumPy arrays.

    Parameters
    ----------
    rets : pd.DataFrame, pd.Series or np.ndarray
        Monthly returns of shape ``(T,)`` or ``(T, P)``.

    Returns
    -------
    float or pd.Series
        One Sharpe ratio per column.
    """
    return np.sqrt(12) * rets.mean() / rets.std()

def produce_random_feature_portfolios(
    P,
    r1,
    signals,
    num_seeds=2,
    scale=1.0,
    activation="relu",
    device=None,
):
    """
    Random-feature managed returns, switching to the chunked builder for many features.

    Uses :func:`produce_random_feature_managed_returns` (and returns the
    features) when ``P * num_seeds < 500``, and
    :func:`produce_random_feature_managed_returns_chunked` otherwise.

    Parameters
    ----------
    P : int
        Number of random features per seed.
    r1 : pd.Series
        Asset returns of length ``NT``, indexed by a MultiIndex with a ``"date"`` level.
    signals : pd.DataFrame
        Characteristics of shape ``(NT, d)``, row-aligned with ``r1``.
    num_seeds : int, default=2
        Number of independent weight draws.
    scale : float, default=1.0
        Multiplier of the weight standard deviation.
    activation : {'relu', 'gelu', 'sincos'}, default='relu'
        Non-linearity.
    device : str or torch.device, optional
        Device; auto-detected if None.

    Returns
    -------
    managed_returns : pd.DataFrame
        Managed returns of shape ``(T, P * num_seeds)``.
    features : pd.DataFrame or None
        The features if ``P * num_seeds < 500``, else None.
    """
    full_number_features = P * num_seeds
    if full_number_features < 500:
        rets_rf, rf_features = produce_random_feature_managed_returns(
            P,
            r1,
            signals=signals.values,
            num_seeds=num_seeds,
            scale=scale,
            activation=activation,
            return_features=True,
            device=device
        )
    else:
        rf_features = None
        rets_rf = produce_random_feature_managed_returns_chunked(
            P,
            r1,
            signals=signals.values,
            num_seeds=num_seeds,
            scale=scale,
            activation=activation,
            base_seed=0,
            device=device,
            dtype=torch.float32
        )
    return rets_rf, rf_features

def build_random_features_portfolios(
    data: pd.DataFrame,
    ret_col: str = "r_1",
    size_group: str = "all",
    activation: str = "relu",
    scale: float = 0.5,
    num_seeds: int = 10,
    signal_group: list | None = None,
    n_random_feat_per_seed : int = 100,
    device=None,
) -> pd.DataFrame:
    """
    Random-feature managed returns from a stock-level panel.

    Optionally filters the panel on a size group, uses every remaining column
    other than ``ret_col``, ``'size_grp'`` and ``'signal_group'`` as a
    characteristic (or only the columns in ``signal_group``), and builds the
    managed returns with :func:`produce_random_feature_portfolios`.

    Parameters
    ----------
    data : pd.DataFrame
        Panel indexed by a MultiIndex with a ``"date"`` level, containing the
        return column, the characteristics and, if ``size_group`` is used, a
        ``'size_grp'`` column.
    ret_col : str, default='r_1'
        Return column.
    size_group : str, default='all'
        ``'all'`` keeps every stock, ``'all_no_micro'`` drops
        ``size_grp == 'micro'``, any other value keeps ``size_grp == size_group``.
    activation : {'relu', 'gelu', 'sincos'}, default='relu'
        Non-linearity.
    scale : float, default=0.5
        Multiplier of the weight standard deviation.
    num_seeds : int, default=10
        Number of independent weight draws.
    signal_group : list of str, optional
        Characteristics to use (default: all of them).
    n_random_feat_per_seed : int, default=100
        Number of random features per seed.
    device : str or torch.device, optional
        Device; auto-detected if None.

    Returns
    -------
    pd.DataFrame
        Managed returns of shape ``(T, n_random_feat_per_seed * num_seeds)``,
        indexed by date.
    """
    # Restrict the universe to the requested size group
    df_filtered = data.copy()
    if size_group == "all_no_micro":
        df_filtered = df_filtered[df_filtered['size_grp'] != 'micro']
    elif size_group != "all":
        df_filtered = df_filtered[df_filtered['size_grp'] == size_group]

    signals = df_filtered.drop(columns=[ret_col, 'size_grp', 'signal_group'], errors='ignore')
    returns = df_filtered[ret_col]

    if signal_group is not None:
        signals = signals[[x for x in signal_group]]

    managed_returns = produce_random_feature_portfolios(
        P=n_random_feat_per_seed,
        r1=returns,
        signals=signals,
        num_seeds=num_seeds,
        scale=scale,
        activation=activation,
        device=device,
    )

    return managed_returns[0]

def build_pairwise_features_portfolios(
    data : pd.DataFrame,
    ret_col : str = "r_1",
    chunk_size : int = 100,
) -> pd.DataFrame:
    """
    Managed returns from all pairwise products of the characteristics.

    A constant column is appended to the ``n`` characteristics and every
    product of two distinct columns is used as a signal: the ``n (n - 1) / 2``
    interactions plus the ``n`` characteristics themselves (squares are not
    included). Products are built ``chunk_size`` at a time to limit memory.

    Parameters
    ----------
    data : pd.DataFrame
        Panel indexed by a MultiIndex with a ``"date"`` level, containing
        ``ret_col`` and the characteristics (all other columns).
    ret_col : str, default='r_1'
        Return column.
    chunk_size : int, default=100
        Number of products computed at once.

    Returns
    -------
    pd.DataFrame
        Managed returns of shape ``(T, n (n + 1) / 2)``, indexed by date.
        Columns are labelled by position within each chunk, so labels repeat
        across chunks.
    """

    signals = data.drop(columns = [ret_col])
    signals['ones'] = 1.0
    returns = data[ret_col]

    cols = signals.columns.tolist()
    n_cols = len(cols)

    # All pairs (i, j) with i < j
    pairs = [(i, j) for i in range(n_cols) for j in range(i + 1, n_cols)]

    managed_blocks = []
    dates = returns.index.get_level_values("date")
    r = returns.values

    for chunk_start in range(0, len(pairs), chunk_size):

        chunk_pairs = pairs[chunk_start:chunk_start + chunk_size]

        # Products for this chunk (rows: asset-dates, columns: pairs)
        chunk_features = []
        signals_array = signals.values
        for i, j in chunk_pairs:
            product = signals_array[:, i] * signals_array[:, j]
            chunk_features.append(product)

        chunk_df = pd.DataFrame(
            np.column_stack(chunk_features),
            index=signals.index
        )

        # Managed returns: mean over assets of signal times return, per date
        weighted = chunk_df.multiply(r, axis=0)
        managed = weighted.groupby(dates).mean()
        managed_blocks.append(managed)

        del chunk_df, weighted

    return pd.concat(managed_blocks, axis=1)
