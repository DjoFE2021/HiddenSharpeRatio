"""
Constants shared across the package.
"""

import numpy as np

#: Default grid of effective shrinkage levels ``z_eff``.
#:
#: The grid is expressed for an aspect ratio ``c = P / T = 1``; for other
#: shapes it is rescaled as ``z_grid = (P / T) * Z_EFF_GRID``. The resulting
#: values are ridge penalties in units of the average eigenvalue, as expected
#: by :func:`RMT.rmt.final_stat.setup_grs`. Small values mean little
#: shrinkage, large values strong regularisation.
Z_EFF_GRID = np.array([1e-4, 1e-3, 1e-2, 1e-1])


#: Display labels for size-group keys (keys are used in file names, labels in plots).
SIZE_GROUP_LABEL = {
    'mega':         'mega',
    'large':        'large',
    'small':        'small',
    'micro':        'micro',
    'all':          'all',
    'all_no_micro': 'All ex. micro',
}


def size_group_label(key: str) -> str:
    """Return the display label of a size-group key, or the key itself if it has none."""
    return SIZE_GROUP_LABEL.get(key, key)


#: Ridge penalties (e.g. for ridge regressions).
RIDGES = np.array([0.1, 1, 10])
