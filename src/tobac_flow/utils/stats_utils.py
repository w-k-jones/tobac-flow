import numpy as np
from scipy import stats

def find_overlap_mode(x):
    if np.any(x):
        overlap_mode = stats.mode(x[x != 0], keepdims=False)[0]
    else:
        overlap_mode = 0
    return overlap_mode

def n_unique_along_axis(a, axis=0):
    b = np.sort(np.moveaxis(a, axis, 0), axis=0)
    return (b[1:] != b[:-1]).sum(axis=0) + (
        np.count_nonzero(a, axis=axis) == a.shape[axis]
    ).astype(int)
