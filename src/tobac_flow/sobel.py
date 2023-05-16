import numpy as np
import scipy.ndimage as ndi

from tobac_flow.convolve import convolve


def _sobel_matrix(ndims):
    """
    Calculate the sobel coefficient matrix for a given number of dimensions

    Parameters
    ----------
    ndims : int
        The number of dimensions of the required sobel matrix

    Returns
    -------
    sobel_matrix : numpy.ndarray
        An array of the sobel coefficients for the reuqested dimensionality.
            Result is an array with ndims dimensions and a length of 3 in each
            dimension
    """
    sobel_matrix = np.array([-1, 0, 1])
    for i in range(ndims - 1):
        sobel_matrix = np.multiply.outer(np.array([1, 2, 1]), sobel_matrix)
    return sobel_matrix


sobel_matrix = _sobel_matrix(3)


def _sobel_func_uphill(x):
    x = np.fmax(x - x[13], 0)
    out_array = np.nansum(x * sobel_matrix.ravel()[:, np.newaxis, np.newaxis], 0) ** 2
    out_array += (
        np.nansum(
            x * sobel_matrix.transpose([1, 2, 0]).ravel()[:, np.newaxis, np.newaxis], 0
        )
        ** 2
    )
    out_array += (
        np.nansum(
            x * sobel_matrix.transpose([2, 0, 1]).ravel()[:, np.newaxis, np.newaxis], 0
        )
        ** 2
    )

    return out_array**0.5


def _sobel_func_downhill(x):
    x = np.fmin(x - x[13], 0)
    out_array = np.nansum(x * sobel_matrix.ravel()[:, np.newaxis, np.newaxis], 0) ** 2
    out_array += (
        np.nansum(
            x * sobel_matrix.transpose([1, 2, 0]).ravel()[:, np.newaxis, np.newaxis], 0
        )
        ** 2
    )
    out_array += (
        np.nansum(
            x * sobel_matrix.transpose([2, 0, 1]).ravel()[:, np.newaxis, np.newaxis], 0
        )
        ** 2
    )

    return out_array**0.5


def _sobel_func(x):
    x -= x[13]
    out_array = np.nansum(x * sobel_matrix.ravel()[:, np.newaxis, np.newaxis], 0) ** 2
    out_array += (
        np.nansum(
            x * sobel_matrix.transpose([1, 2, 0]).ravel()[:, np.newaxis, np.newaxis], 0
        )
        ** 2
    )
    out_array += (
        np.nansum(
            x * sobel_matrix.transpose([2, 0, 1]).ravel()[:, np.newaxis, np.newaxis], 0
        )
        ** 2
    )

    return out_array**0.5


def sobel(
    data,
    forward_flow,
    backward_flow,
    method="linear",
    dtype=np.float32,
    fill_value=np.nan,
    direction=None,
):
    """
    Sobel edge detection algorithm in a semi-Lagrangian space

    Parameters
    ----------
    data : numpy.ndarray
        Data in which to detect edges
    forward_flow : numpy.ndarray
        The flow vectors acting forward along the leading dimension of data
    backward_flow : numpy.ndarray
        The flow vectors acting backward along the leading dimension of data
    method : string
        The interpolation method to use of the offset pixel locations by the
            flow vectors. Must be one of 'linear' or 'nearest'
    dtype : type, optional (default : np.float32)
        The dtype of the output data
    fill_value : scalar, optional (default : np.nan)
        Value used to fill locations that are invalid
    direction : string, optional (default : None)
        If 'downhill' or 'uphill' only calculate edges where the surrounding
            pixels are less than or greater than the centre pixel respecitively

    Returns
    -------
    res : numpy.ndarray
        The magnitude of the edges detected using the sobel method
    """
    if direction == "uphill":
        func = _sobel_func_uphill
    elif direction == "downhill":
        func = _sobel_func_downhill
    else:
        func = _sobel_func

    res = convolve(
        data,
        forward_flow,
        backward_flow,
        structure=ndi.generate_binary_structure(3, 3),
        method=method,
        dtype=dtype,
        fill_value=fill_value,
        func=func,
    )

    return res
