import numpy as np
import scipy.ndimage as ndi
import cv2
import xarray as xr

def warp_flow(img, flow, method="linear",
                 fill_value=np.nan, offsets=np.array([[0,0]])):
    """
    Warp an image according to a set of optical flow vectors. Can be provided
        with an array of offsets to warp the image to set of adjacent locations
        at once.

    Parameters
    ----------
    img : numpy.ndarray
        The image to be warped
    flow : numpy.ndarray
        The flow vectors to warp the image by. Must have shape of img.shape, 2
    method : string, optional (default : "linear")
        Interpolation method to use when warping. Either 'linear' or 'nearest'
    fill_value : scalar, optional (default : np.nan)
        Value used to fill locations that are warped outside of the image
    offsets : numpy.ndarray, optional (default [0,0])
        Offset locations for the x and y dimensions that offset the locations
            that the array is warped to

    Returns
    -------
    res : numpy.ndarray
        The warped image according the the provided flow vectors and offsets
    """
    if method == "linear":
        method = cv2.INTER_LINEAR
    elif method == "nearest":
        method = cv2.INTER_NEAREST
    else:
        raise ValueError("method must be either 'linear' or 'nearest'")
    h, w = flow.shape[:2]
    locs = flow[np.newaxis,...] + np.atleast_2d(offsets)[:,np.newaxis,np.newaxis,:].astype(np.float32)
    locs[...,0] += np.arange(w)
    locs[...,1] += np.arange(h)[...,np.newaxis]
    res = cv2.remap(img,
                    locs.reshape([ -1, locs.shape[-2], locs.shape[-1]]),
                    None,
                    method,
                    None,
                    cv2.BORDER_CONSTANT,
                    fill_value).reshape(locs.shape[:-1])

    return res

def convolve_same_step(img, offsets, fill_value=np.nan):
    """
    Convolve an image according to a set of offsets provided by a structuring
        element

    Parameters
    ----------
    img : numpy.ndarray
        The image to be warped
    offsets : numpy.ndarray
        Offset locations for the x and y dimensions that offset the locations
            that the array is warped to
    fill_value : scalar, optional (default : np.nan)
        Value used to fill locations that are warped outside of the image

    Returns
    -------
    res : numpy.ndarray
        The convolved image according the the provided offsets
    """
    h, w = img.shape
    locs = (np.stack(np.meshgrid(range(w), range(h)),-1)
            + np.atleast_2d(offsets)[:,np.newaxis,np.newaxis,:]).astype(int)

    wh_nan = np.logical_or.reduce([locs[...,0] < 0,
                                   locs[...,1] < 0,
                                   locs[...,0] >= w,
                                   locs[...,1] >= h])

    locs[...,0][wh_nan] = 0
    locs[...,1][wh_nan] = 0

    res = img[locs[...,1], locs[...,0]]

    res[wh_nan] = fill_value

    return res

def convolve_step(prev_step, same_step, next_step, forward_flow, backward_flow,
                  structure=ndi.generate_binary_structure(3,1),
                  method="linear",
                  dtype=np.float32,
                  fill_value=np.nan):
    """
    Convolve a sequence of images using optical flow vectors to offset adjacent
        elements in the leading dimensions at a single time step

    Parameters
    ----------
    prev_step : numpy.ndarray
        The data to convolved at the previous time step
    same_step : numpy.ndarray
        The data to convolved at the previous current step
    next_step : numpy.ndarray
        The data to convolved at the previous next step
    forward_flow : numpy.ndarray
        The flow vectors from the current step to the next step
    backward_flow : numpy.ndarray
        The flow vectors from the current step to the previous step
    structure : numpy.ndarray, optional
        The structuring element used to find adjacent pixles for the
            convolution. Default is a 1 connectivity structure produced by
            ndi.generate_binary_structure(3,1)
    dtype : type, optional (default : np.float32)
        The dtype of the output data
    fill_value : scalar, optional (default : np.nan)
        Value used to fill locations that are warped outside of the image

    Returns
    -------
    res : numpy.ndarray
        The convolved image according the the provided structure
    """
    n_struct = np.count_nonzero(structure)
    res_shape = (n_struct,) + same_step.shape
    res = np.full(res_shape, fill_value, dtype=dtype)

    n_backward = np.count_nonzero(structure[0])
    n_same = np.count_nonzero(structure[1])
    n_forward = np.count_nonzero(structure[-1])

    if n_backward:
        offsets = np.stack(np.where(structure[0]), -1)[...,::-1]-1
        res[:n_backward] = warp_flow(prev_step, backward_flow,
                                        method=method, fill_value=fill_value,
                                        offsets=offsets)

    if n_same:
        offsets = np.stack(np.where(structure[1]), -1)[...,::-1]-1
        res[n_backward:-n_forward] = convolve_same_step(same_step, offsets, fill_value=fill_value)

    if n_forward:
        offsets = np.stack(np.where(structure[-1]), -1)[...,::-1]-1
        res[-n_forward:] = warp_flow(next_step, forward_flow,
                                        method=method, fill_value=fill_value,
                                        offsets=offsets)

    return res

def convolve(data, forward_flow, backward_flow,
             structure=ndi.generate_binary_structure(3,1),
             method="linear",
             dtype=np.float32,
             fill_value=np.nan,
             func=None):
    """
    Convolve a sequence of images using optical flow vectors to offset adjacent
        elements in the leading dimensions

    Parameters
    ----------
    data : numpy.ndarray
        The dataset to be convolved
    forward_flow : numpy.ndarray
        The flow vectors acting forward along the leading dimension of data
    backward_flow : numpy.ndarray
        The flow vectors acting backward along the leading dimension of data
    structure : numpy.ndarray, optional
        The structuring element used to find adjacent pixles for the
            convolution. Default is a 1 connectivity structure produced by
            ndi.generate_binary_structure(3,1)
    method : string
        The interpolation method to use of the offset pixel locations by the
            flow vectors. Must be one of 'linear' or 'nearest'
    dtype : type, optional (default : np.float32)
        The dtype of the output data
    fill_value : scalar, optional (default : np.nan)
        Value used to fill locations that are warped outside of the image
    func : function, optional (default : np.nan)
        The function to be applied to the convolved data at each time step. If
            None, the convolution will return all the convolved pixel locations.

    Returns
    -------
    res : numpy.ndarray
        The convolved data according the the provided structure
    """
    assert structure.shape == (3,3,3), "Structure input must be a 3x3x3 array"

    if isinstance(data, xr.DataArray):
        data = data.compute().data

    if func is not None:
        res = np.full(data.shape, fill_value, dtype=dtype)
    else:
        res = np.full((np.count_nonzero(structure),) + data.shape, fill_value, dtype=dtype)

    for i in range(data.shape[0]):
        step_frame = data[i]
        if i == 0 :
            prev_frame = np.full(step_frame.shape, fill_value, dtype=dtype)
        else:
            prev_frame = data[i-1]
        if i == data.shape[0]-1:
            next_frame = np.full(step_frame.shape, fill_value, dtype=dtype)
        else:
            next_frame = data[i+1]

        if func is not None:
            res[i] = func(convolve_step(prev_frame, step_frame, next_frame,
                                        forward_flow[i], backward_flow[i],
                                        structure=structure,
                                        method=method,
                                        dtype=dtype,
                                        fill_value=fill_value))
        else:
            res[:,i] = convolve_step(prev_frame, step_frame, next_frame,
                                     forward_flow[i], backward_flow[i],
                                     structure=structure,
                                     method=method,
                                     dtype=dtype,
                                     fill_value=fill_value)
    if func is not None:
        res[np.isnan(data)] = fill_value
    return res
