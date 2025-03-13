"""
Functions for performing normalisation over a range of data
"""

from typing import Callable
import numpy as np
import scipy.ndimage as ndi


def to_8bit(
    array: np.ndarray[float], vmin: float = None, vmax: float = None, fill_value=127
) -> np.ndarray[np.uint8]:
    """
    Converts an array to an 8-bit range between 0 and 255
    """
    if vmin is None:
        vmin = np.nanmin(array)
    if vmax is None:
        vmax = np.nanmax(array)
    if vmin == vmax:
        factor = 0
    else:
        factor = 255 / (vmax - vmin)
    array_out = (array - vmin) * factor

    # Replace non-finite values before converting to uint8
    wh_finite = np.isfinite(array_out)
    array_out[np.logical_not(wh_finite)] = fill_value
    # Large changes in the field values when one frame is NaN and the next is not cause problems for optical flow. In these cases, we just replace those values with those from the other frame
    array_out[0][~wh_finite[0]] = array_out[1][~wh_finite[0]]
    array_out[1][~wh_finite[1]] = array_out[0][~wh_finite[1]]

    return array_out.astype("uint8")


def linearise_field(
    field: np.ndarray[float], lower_threshold: float, upper_threshold: float
) -> np.ndarray[float]:
    if lower_threshold == upper_threshold:
        raise ValueError("lower and upper thresholds must have different values")
    if lower_threshold > upper_threshold:
        upper_threshold, lower_threshold = lower_threshold, upper_threshold
        linearised_field = 1 - np.maximum(
            np.minimum(
                (field - lower_threshold) / (upper_threshold - lower_threshold), 1
            ),
            0,
        )
    else:
        linearised_field = np.maximum(
            np.minimum(
                (field - lower_threshold) / (upper_threshold - lower_threshold), 1
            ),
            0,
        )
    return linearised_field


def linear_norm(
    array: np.ndarray[float], vmin: float = None, vmax: float = None
) -> np.ndarray[float]:
    if vmin is None:
        vmin = np.nanmin(array)
    if vmax is None:
        vmax = np.nanmax(array)
    if vmax > vmin:
        factor = 1 / (vmax - vmin)
    else:
        factor = 0
    array_out = (array - vmin) * factor
    array_out = np.maximum(np.minimum(array_out, 1), 0)
    return array_out


def log_norm(
    array: np.ndarray[float], vmin: float = None, vmax: float = None
) -> np.ndarray[float]:
    vmin = np.nanmin(array)
    norm = np.log(array - vmin + 1)
    return linear_norm(norm, vmin=vmin, vmax=vmax)


def inverse_log_norm(
    array: np.ndarray[float], vmin: float = None, vmax: float = None
) -> np.ndarray[float]:
    vmax = np.nanmax(array)
    inv_log_norm = np.log(vmax - array + 1)
    return linear_norm(inv_log_norm, vmin=vmin, vmax=vmax)


def z_norm(array: np.ndarray[float], max_std: float = 3) -> np.ndarray[float]:
    mean = np.nanmean(array)
    std = np.nanstd(array)
    norm = (array - mean) / std
    return linear_norm(norm, vmin=-max_std, vmax=max_std)


def uniform_norm(array: np.ndarray[float], quantiles: int = 256) -> np.ndarray[float]:
    bin_edges = np.quantile(array, np.linspace(0, 1, quantiles + 1))
    bin_edges[-1] = bin_edges[-1] + 1
    norm = np.digitize(array, bin_edges)
    return linear_norm(norm)


def local_linear_norm(data: np.ndarray[float], size: int = 100) -> np.ndarray[float]:
    if not np.all(np.isfinite(data)):
        data = np.copy(data)
        data[np.isnan(data)] = np.nanmean(data)
    vmax = ndi.maximum_filter(data, size)
    vmin = ndi.minimum_filter(data, size)
    factor = vmax - vmin
    wh_zero = factor == 0
    factor[wh_zero] = 1
    factor = 1 / factor
    factor[wh_zero] = 0
    return (data - vmin) * factor


def select_normalisation_method(method: str) -> Callable:
    norm_methods = {
        "linear": linear_norm,
        "log": log_norm,
        "inverse_log": inverse_log_norm,
        "z_score": z_norm,
        "uniform": uniform_norm,
        "local_linear": local_linear_norm,
    }
    if method in norm_methods:
        return norm_methods[method]
    else:
        raise ValueError(
            f"{method} not an acceptable normalisation method, method must be one of {list(norm_methods.keys())}"
        )


__all__ = (
    "to_8bit",
    "linearise_field",
    "linear_norm",
    "log_norm",
    "inverse_log_norm",
    "z_norm",
    "uniform_norm",
    "local_linear_norm",
    "select_normalisation_method",
)
