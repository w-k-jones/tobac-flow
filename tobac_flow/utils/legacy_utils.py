import numpy as np
from typing import Callable


def apply_func_to_labels(
    labels: np.ndarray[int],
    field: np.ndarray,
    func: Callable,
    default: None | float = None,
) -> np.ndarray:
    """
    Apply a given function to the regions of an array given by an array of
        labels. Functions similar to ndi.labeled_comprehension, but may be more
        adaptable in some circumstances
    """
    if labels.shape != field.shape:
        raise ValueError("Input labels and field do not have the same shape")
    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    return np.array(
        [
            func(field.ravel()[args[bins[i] : bins[i + 1]]])
            if bins[i + 1] > bins[i]
            else default
            for i in range(bins.size - 1)
        ]
    )


def apply_weighted_func_to_labels(
    labels: np.ndarray[int],
    field: np.ndarray,
    weights: np.ndarray,
    func: Callable,
    default: None | float = None,
) -> np.ndarray:
    """
    Apply a given weighted function to the regions of an array given by an array
        of labels. The weights provided by the weights array in the labelled
        region will also be provided to the function.
    """
    if labels.shape != field.shape:
        raise ValueError("Input labels and field do not have the same shape")
    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    return np.array(
        [
            func(
                field.ravel()[args[bins[i] : bins[i + 1]]],
                weights.ravel()[args[bins[i] : bins[i + 1]]],
            )
            if bins[i + 1] > bins[i]
            else default
            for i in range(bins.size - 1)
        ]
    )
