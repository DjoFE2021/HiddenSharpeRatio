import numpy as np
from RMT.linalg.eig import EigenSystem

def test_under_parameterized():
    # Under-parameterized case (P ≤ T)
    T : int = 5
    P : int = 3
    X : np.ndarray = np.random.randn(T, P)
    eig_sys : EigenSystem = EigenSystem(X)
    
    eigenvalues, eigenvectors = np.linalg.eigh(X.T @ X / T)
    np.testing.assert_allclose(eig_sys.get_eigenvalues(), eigenvalues)
    np.testing.assert_allclose(eig_sys.get_eigenvectors(), eigenvectors)

    assert eig_sys.t_ == T
    assert eig_sys.p_ == P
    assert eig_sys._regime == "under"
    assert eig_sys._xtx.shape == (P, P)
    assert eig_sys._eigvals.shape == (P,)
    assert eig_sys._q.shape == (P, P)
    assert eig_sys._xtq is None
    
def test_over_parameterized():
    # Over-parameterized case (P > T)
    T : int = 3
    P : int = 5
    X : np.ndarray = np.random.randn(T, P)
    eig_sys : EigenSystem = EigenSystem(X)
    
    eigenvalues, eigenvectors = np.linalg.eigh(X.T @ X / T)
    np.testing.assert_allclose(eig_sys.get_eigenvalues()[:P], eigenvalues, rtol = 1e-3, atol = 1e-3)

    assert eig_sys.t_ == T
    assert eig_sys.p_ == P
    assert eig_sys._regime == "over"
    assert eig_sys._xxt.shape == (T, T)
    assert eig_sys._eigvals.shape == (P,)
    assert eig_sys._q.shape == (T, T)