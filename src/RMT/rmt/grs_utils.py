"""
Helper for the geometric-series bounds.
"""

import numpy as np


def f_and_grad(x, y, n):
    """
    Truncated geometric series ``f(x, y) = y + x * sum_{j=0}^{n} (x / y)^j`` and its gradient.

    With ``x = B_J``, ``y = B_{J-1}`` and ``r = x / y``,
    ``f = B_{J-1} + B_J (1 + r + ... + r^n)`` extends the series
    ``B_{J-1} + B_J + B_{J+1} + ...`` geometrically for ``n`` more terms.
    The powers are summed directly instead of using ``(1 - r^{n+1}) / (1 - r)``,
    which loses accuracy when ``r`` is close to 1.

    Parameters
    ----------
    x, y : ndarray of shape (N,) or (N, 1)
        Arrays of equal length; ``y`` must be non-zero.
    n : int
        Highest power of ``r`` in the sum.

    Returns
    -------
    f : ndarray of shape (N, 1)
        Function values.
    grad : tuple of two ndarrays of shape (N, 1)
        ``(df/dx, df/dy)``, where ``df/dx = sum_{j=0}^{n} r^j + r S'`` and
        ``df/dy = 1 - r^2 S'``, with ``S' = sum_{j=1}^{n} j r^{j-1}``.
    """
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    r = x / y

    # S = sum_{j=0}^n r^j and S' = dS/dr, summed term by term (r is often close to 1)
    tmp = np.sum(np.concatenate([r **j for j in range(n + 1)], axis=1), axis=1).reshape(-1, 1)
    if n > 0:
        tmp_der = np.sum(np.concatenate([j * r ** (j - 1) for j in range(1, n + 1)], axis=1), axis=1).reshape(-1, 1)
    else:
        tmp_der = 0
    f = y + x * tmp
    df_dx = tmp + x * tmp_der / y
    df_dy = 1 - (x * r / y) * tmp_der
    return f, (df_dx, df_dy)
