"""
Per-date residualisation and whitening of panel data, plus random rotations.

Panel inputs are indexed by a MultiIndex whose date level is given by
``date_level``. Each date is processed independently, in PyTorch (on the GPU
when available).
"""

import warnings

import numpy as np
import pandas as pd
import torch

def random_rotation(d, seed=0):
    """
    Uniformly distributed random rotation matrix (Haar measure on SO(d)).

    Takes the QR decomposition of a standard Gaussian matrix, fixes the signs
    so that ``Q`` is Haar-distributed on O(d), and flips the first column if
    needed to get determinant +1.

    Parameters
    ----------
    d : int
        Dimension.
    seed : int, default=0
        Seed of the random generator.

    Returns
    -------
    ndarray of shape (d, d)
        Orthogonal matrix with determinant +1.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def whiten_by_day_df_cuda(
    df: pd.DataFrame,
    S_cols,
    *,
    q: float = 0.5,
    eps: float = 1e-8,
    date_level=0,
    device=None,
    dtype=torch.float32,
    keep_original=False,
):
    """
    Whiten a block of columns separately at each date.

    For the ``n x p`` matrix ``S`` of a given date, returns
    ``(SS' + eps I)^{-q} S`` when ``n <= p`` and ``S (S'S + eps I)^{-q}``
    otherwise (eigenvalues floored at ``eps``). Both expressions agree up to
    the regularisation. With ``q = 0.5`` the columns of the result are
    orthonormal when ``n > p``.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    S_cols : list of str
        Columns to whiten.
    q : float, default=0.5
        Power of the inverse Gram matrix.
    eps : float, default=1e-8
        Ridge added to the Gram matrix and floor for its eigenvalues.
    date_level : int or str, default=0
        Index level holding the date.
    device : str or torch.device, optional
        Device; auto-detected if None.
    dtype : torch.dtype, default=torch.float32
        Computation dtype.
    keep_original : bool, default=False
        If True, return all columns of ``df`` followed by the whitened columns
        with the suffix ``"_white"``.

    Returns
    -------
    pd.DataFrame
        Whitened ``S_cols`` (or the extended frame), in the row order of ``df``.
    """

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Locate the contiguous block of rows of each date
    df_sorted = df.sort_index(level=date_level)
    idx = df_sorted.index
    dates = idx.get_level_values(date_level).to_numpy()

    change = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(df_sorted)]

    S_np = df_sorted[S_cols].to_numpy()
    out = np.empty_like(S_np, dtype=np.float64)

    for a, b in zip(starts, ends):
        S = torch.as_tensor(S_np[a:b], device=device, dtype=dtype)
        n, p = S.shape

        if n <= p:
            # Row space: (S S')^{-q} S
            G = S @ S.T
            if eps > 0:
                G = G + eps * torch.eye(n, device=device, dtype=dtype)

            evals, U = torch.linalg.eigh(G)
            w = evals.clamp_min(eps).pow(-q)      # (n,)
            US = U.T @ S                          # (n x p)
            S_white = U @ (w[:, None] * US)

        else:
            # Column space: S (S' S)^{-q}
            G = S.T @ S
            if eps > 0:
                G = G + eps * torch.eye(p, device=device, dtype=dtype)

            evals, V = torch.linalg.eigh(G)
            w = evals.clamp_min(eps).pow(-q)      # (p,)
            SV = S @ V                            # (n x p)
            S_white = (SV * w[None, :]) @ V.T

        out[a:b] = S_white.cpu().numpy()

    res = pd.DataFrame(out, index=df_sorted.index, columns=S_cols)

    if keep_original:
        res = pd.concat([df_sorted, res.add_suffix("_white")], axis=1)

    return res.reindex(df.index)


def resid_I_minus_S_ridge_torch(E, S, eps=1e-8, *, prefer=None):
    """
    Residuals of a ridge regression of ``E`` on ``S``: ``E - S (S'S + eps I)^{-1} S'E``.

    Two equivalent forms are available: ``'p'`` solves the ``p x p`` system
    ``(S'S + eps I) alpha = S'E`` by Cholesky, and ``'n'`` uses the identity
    ``E - S (S'S + eps I)^{-1} S'E = eps (SS' + eps I)^{-1} E`` with an
    ``n x n`` Cholesky solve.

    Parameters
    ----------
    E : torch.Tensor
        Targets of shape ``(n,)`` or ``(n, m)``; cast to the dtype and device of ``S``.
    S : torch.Tensor
        Regressors of shape ``(n, p)``.
    eps : float, default=1e-8
        Ridge penalty.
    prefer : {'p', 'n'}, optional
        Form to use; by default ``'p'`` if ``p <= n``, else ``'n'``.

    Returns
    -------
    torch.Tensor
        Residuals, same shape as ``E``. In the ``'p'`` form, if the Cholesky
        factorisation fails, a warning is issued and ``E`` is returned unchanged.
    """
    dtype = S.dtype
    device = S.device

    E = E.to(dtype=dtype, device=device)

    n, p = S.shape
    squeeze = False
    if E.ndim == 1:
        E = E[:, None]
        squeeze = True

    if prefer is None:
        mode = "p" if p <= n else "n"
    else:
        mode = prefer
    if mode == "p":
        # (S'S + eps I) alpha = S'E, then R = E - S alpha
        G = S.T @ S
        if eps > 0:
            G = G + eps * torch.eye(p, device=device, dtype=dtype)

        B = S.T @ E
        try:
            L = torch.linalg.cholesky(G)
            alpha = torch.cholesky_solve(B, L)
            R = E - S @ alpha
        except torch.linalg.LinAlgError:
            warnings.warn(
                f"Cholesky failed for E {tuple(E.shape)} and S {tuple(S.shape)}; "
                "returning E unchanged."
            )
            R = E

    else:
        # R = eps (SS' + eps I)^{-1} E
        K = S @ S.T
        if eps > 0:
            K = K + eps * torch.eye(n, device=device, dtype=dtype)

        L = torch.linalg.cholesky(K)
        X = torch.cholesky_solve(E, L)
        R = eps * X

    return R[:, 0] if squeeze else R

def resid_by_day_df_cuda(
    df: pd.DataFrame,
    E_cols,
    S_cols,
    eps: float = 1e-8,
    *,
    date_level=0,
    keep_original=False,
    device=None,
    dtype=torch.float32,
):
    """
    Residualise columns on other columns separately at each date.

    For each date, ``E_cols`` are regressed on ``S_cols`` across assets with a
    small ridge penalty (:func:`resid_I_minus_S_ridge_torch`) and replaced by
    the residuals, e.g. to remove the exposure of returns to a set of
    characteristics.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    E_cols : str or list of str
        Columns to residualise.
    S_cols : list of str
        Regressors.
    eps : float, default=1e-8
        Ridge penalty.
    date_level : int or str, default=0
        Index level holding the date.
    keep_original : bool, default=False
        If True, return all columns of ``df`` followed by the residual columns
        (which keep the names of ``E_cols``).
    device : str or torch.device, optional
        Device; auto-detected if None.
    dtype : torch.dtype, default=torch.float32
        Computation dtype.

    Returns
    -------
    pd.DataFrame
        Residuals with columns ``E_cols`` (or the extended frame), in the row
        order of ``df``.

    Examples
    --------
    >>> idx = pd.MultiIndex.from_product([pd.date_range('2020-01-01', periods=100), range(50)],
    ...                                  names=['date', 'id'])
    >>> df = pd.DataFrame(np.random.randn(5000, 4), index=idx,
    ...                   columns=['ret', 'mkt', 'value', 'size'])
    >>> resid_by_day_df_cuda(df, E_cols='ret', S_cols=['mkt', 'value', 'size']).shape
    (5000, 1)
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if isinstance(E_cols, str):
        E_cols = [E_cols]

    # Locate the contiguous block of rows of each date
    df_sorted = df.sort_index(level=date_level)
    idx = df_sorted.index
    dates = idx.get_level_values(date_level).to_numpy()

    change = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(df_sorted)]

    E_np = df_sorted[E_cols].to_numpy()
    S_np = df_sorted[S_cols].to_numpy()

    out = np.empty_like(E_np, dtype=np.float64)

    for a, b in zip(starts, ends):
        # Same dtype for targets and regressors
        E_t = torch.as_tensor(E_np[a:b], device=device, dtype=dtype)
        S_t = torch.as_tensor(S_np[a:b], device=device, dtype=dtype)

        R_t = resid_I_minus_S_ridge_torch(E_t, S_t, eps=eps)
        out[a:b] = R_t.cpu().numpy()

    res = pd.DataFrame(out, index=df_sorted.index, columns=E_cols)

    if keep_original:
        res = pd.concat([df_sorted, res], axis=1)

    return res.reindex(df.index)
