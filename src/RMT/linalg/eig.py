"""
Eigendecomposition of the second-moment matrix of a data matrix.

For a data matrix ``X`` of shape ``(T, P)``, :class:`EigenSystem` diagonalises
the smaller of the two Gram matrices:

- ``P <= T`` ("under-parameterized"): ``X'X / T``, of size ``P x P``;
- ``P > T`` ("over-parameterized"): ``XX' / T``, of size ``T x T``. Both
  matrices share their non-zero eigenvalues, so the spectrum of ``X'X / T`` is
  recovered by padding with ``P - T`` zeros.

The module also provides two helpers that repair estimated covariance
matrices: :func:`clip_eigenvalues` and :func:`minimal_shift_for_pd`.
"""

import numpy as np


def minimal_shift_for_pd(A):
    """
    Make a symmetric matrix positive definite by raising its last diagonal entry.

    Writing ``A = [[B, b], [b', c]]``, ``A`` is positive definite iff ``B`` is
    and the Schur complement ``c - b' B^{-1} b`` is positive. The last diagonal
    entry is increased by ``max(1e-3, b' B^{-1} b - c + 1e-10)``, which makes
    the Schur complement positive. Note that a shift of at least ``1e-3`` is
    always applied, even if ``A`` is already positive definite.

    Parameters
    ----------
    A : ndarray of shape (n, n)
        Symmetric matrix whose top-left ``(n-1) x (n-1)`` block is positive
        definite. Modified in place if it is already a float ndarray.

    Returns
    -------
    ndarray of shape (n, n)
        ``A`` with its last diagonal entry shifted.

    Raises
    ------
    ValueError
        If the top-left block is singular.
    """
    A = np.asarray(A)
    n = A.shape[0]

    # Split A = [[B, b], [b', c]]
    B = A[:-1, :-1]
    b = A[:-1, -1]
    c = A[-1, -1]

    # x = B^{-1} b, without forming the inverse
    try:
        x = np.linalg.solve(B, b)
    except np.linalg.LinAlgError:
        raise ValueError("Top-left block B is not invertible / PD.")

    val = b @ x  # b' B^{-1} b

    # Shift c so that the Schur complement c - b' B^{-1} b becomes positive
    a = max(1e-3, val - c + 1e-10)
    A[-1, -1] = a + c
    return A


def clip_eigenvalues(A, eps=1e-10):
    """
    Symmetrise a matrix and floor its eigenvalues at ``eps``.

    Parameters
    ----------
    A : ndarray of shape (n, n)
        Square matrix to repair; it is replaced by ``(A + A') / 2`` first.
    eps : float, default=1e-10
        Floor applied to the eigenvalues.

    Returns
    -------
    ndarray of shape (n, n)
        Symmetric matrix ``V diag(max(w, eps)) V'``, where ``V diag(w) V'`` is
        the eigendecomposition of the symmetrised input.
    """
    A = (A + A.T) / 2
    w, V = np.linalg.eigh(A)
    w_clipped = np.maximum(w, eps)
    A_clipped = V @ np.diag(w_clipped) @ V.T
    # Re-symmetrise to remove round-off from the reconstruction
    return (A_clipped + A_clipped.T) / 2


class EigenSystem:
    """
    Eigendecomposition of ``X'X / T`` for a data matrix ``X`` of shape ``(T, P)``.

    The decomposition is computed once, at construction: in feature space when
    ``P <= T`` and in sample space when ``P > T`` (see the module docstring).
    Eigenvalues are in ascending order, as returned by :func:`numpy.linalg.eigh`;
    when ``P > T`` the ``P - T`` zero eigenvalues come first.

    Parameters
    ----------
    x : ndarray of shape (T, P)
        Data matrix, used as is (no demeaning or scaling).

    Attributes
    ----------
    t_ : int
        Number of observations ``T``.
    p_ : int
        Number of features ``P``.
    _regime : {"under", "over"}
        ``"under"`` if ``P <= T``, ``"over"`` otherwise.
    _xtx : ndarray of shape (P, P) or None
        ``X'X / T`` (under-parameterized case only).
    _xxt : ndarray of shape (T, T) or None
        ``XX' / T`` (over-parameterized case only).
    _eigvals : ndarray of shape (P,)
        Eigenvalues of ``X'X / T``, zero-padded when ``P > T``.
    _non_zero_eigvals : ndarray of shape (T,) or None
        Eigenvalues of ``XX' / T`` (over-parameterized case only).
    _q : ndarray of shape (P, P) or (T, T)
        Eigenvectors of the matrix that was decomposed.
    _xtq : ndarray of shape (P, T) or None
        ``X'Q``, computed and cached by :meth:`get_eigenvectors`.
    """

    def __init__(self,
                 x: np.ndarray,
                 ) -> None:
        self._x: np.ndarray = x

        self.t_: int = self._x.shape[0]
        self.p_: int = self._x.shape[1]

        self._regime: str = None
        self._xxt: np.ndarray = None
        self._xtx: np.ndarray = None
        self._eigvals: np.ndarray = None
        self._non_zero_eigvals: np.ndarray = None
        self._q: np.ndarray = None
        self._xtq: np.ndarray = None

        self._compute_eigendecomposition()

    def _compute_eigendecomposition(self) -> None:
        """Diagonalise ``X'X / T`` (``P <= T``) or ``XX' / T`` (``P > T``)."""
        if self.p_ <= self.t_:
            self._regime = 'under'

            # P x P second-moment matrix in feature space
            self._xtx = self._x.T @ self._x / self.t_
            self._eigvals, self._q = np.linalg.eigh(self._xtx)

        else:
            self._regime = 'over'

            # T x T Gram matrix in sample space (same non-zero spectrum as X'X / T)
            self._xxt = self._x @ self._x.T / self.t_
            eigvals_t, self._q = np.linalg.eigh(self._xxt)

            # Prepend the P - T zero eigenvalues of X'X / T
            num_zeros = self.p_ - self.t_
            self._non_zero_eigvals = eigvals_t.copy()
            self._eigvals = np.insert(eigvals_t, 0, np.zeros(num_zeros))

            # X'Q is only needed to map eigenvectors back to feature space, so
            # it is computed lazily in get_eigenvectors() (O(P T^2)).

    @property
    def get_regime(self) -> str:
        """``"under"`` if ``P <= T``, ``"over"`` otherwise."""
        return self._regime

    def get_eigenvalues(self) -> np.ndarray:
        """
        Eigenvalues of ``X'X / T``.

        Returns
        -------
        ndarray of shape (P,)
            Eigenvalues in ascending order, zero-padded when ``P > T``.
        """
        return self._eigvals

    def get_non_zero_eigenvalues(self) -> np.ndarray:
        """
        Eigenvalues without the zero padding.

        Returns
        -------
        ndarray of shape (P,) or (T,)
            All ``P`` eigenvalues when ``P <= T``; the ``T`` eigenvalues of
            ``XX' / T`` when ``P > T``. Ascending order.
        """
        if self._regime == 'under':
            return self._eigvals
        else:
            return self._non_zero_eigvals

    def get_eigenvectors(self) -> np.ndarray:
        """
        Eigenvectors of ``X'X / T`` expressed in feature space.

        Returns
        -------
        ndarray of shape (P, P) or (P, T)
            When ``P <= T``: the orthonormal eigenvectors ``Q`` of ``X'X / T``.
            When ``P > T``: ``X'Q``, where ``Q`` are the eigenvectors of
            ``XX' / T``. Its columns are eigenvectors of ``X'X / T`` for the
            non-zero eigenvalues but are not normalised (squared norm
            ``T * lambda_i``). It is computed on the first call and cached.
        """
        if self._regime == 'under':
            return self._q
        else:
            if self._xtq is None:
                self._xtq = self._x.T @ self._q
            return self._xtq

    def get_eigendecomposition(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return ``(get_eigenvalues(), get_eigenvectors())``.

        When ``P > T`` the eigenvalues have length ``P`` while the eigenvector
        matrix has ``T`` columns (one per non-zero eigenvalue).
        """
        return self.get_eigenvalues(), self.get_eigenvectors()

    @property
    def get_x(self) -> np.ndarray:
        """The data matrix ``X`` of shape ``(T, P)``."""
        return self._x

    @property
    def get_q(self) -> np.ndarray:
        """
        Eigenvectors of the matrix that was decomposed.

        ``(P, P)`` eigenvectors of ``X'X / T`` when ``P <= T``, ``(T, T)``
        eigenvectors of ``XX' / T`` otherwise. Use :meth:`get_eigenvectors`
        for feature-space eigenvectors.
        """
        return self._q
