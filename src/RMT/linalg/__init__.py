"""
Linear-algebra building blocks.

eig
    :class:`~RMT.linalg.eig.EigenSystem` eigendecomposes the second-moment
    matrix of a ``T x P`` data matrix, working in whichever of the ``P x P`` or
    ``T x T`` spaces is smaller. Also provides :func:`~RMT.linalg.eig.clip_eigenvalues`
    and :func:`~RMT.linalg.eig.minimal_shift_for_pd` to repair covariance
    matrices that are not positive (semi-)definite.
"""
