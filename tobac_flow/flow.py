import numpy as np
import xarray as xr
import cv2
from scipy import ndimage as ndi
import warnings

from .convolve import warp_flow, convolve
from .label import flow_label
from .sobel import sobel
from .watershed import watershed

class Flow:
    """
    Class to perform semi-lagrangian operations using optical flow
    """
    def __init__(self, data, model="DIS", vr_steps=0, smoothing_passes=0):
        self.shape = data.shape
        self.get_flow(data, model=model, vr_steps=vr_steps,
                      smoothing_passes=smoothing_passes)

    def get_flow(self, data, model="DIS", vr_steps=0, smoothing_passes=0):
        """
        Calculates forward and backward optical flow vectors for a given set of data

        Parameters
        ----------
        data : numpy.ndarray
            Array of data to calculate optical flow for. Flow vectors are calculated
                along the leading dimension
        model : string, optional (default : 'DIS')
            opencv optical flow model to use
        vr_steps : int, optional (default : 0)
            Number of variational refinement operations to perform
        smoothing_step : int, optional (default : 0)
            Number of smoothing operation between the forward and backward flow to
                perform
        """
        self.flow_for, self.flow_back = get_flow(data, model=model,
                                                 vr_steps=vr_steps,
                                                 smoothing_passes=smoothing_passes)

    def convolve(self, data, structure=ndi.generate_binary_structure(3,1),
                 method="linear", fill_value=np.nan, dtype=np.float32,
                 func=None):
        """
        Convolve a sequence of images using optical flow vectors to offset adjacent
            elements in the leading dimensions

        Parameters
        ----------
        data : numpy.ndarray
            The dataset to be convolved
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
        assert data.shape == self.shape, "Data input must have the same shape as the Flow object"

        output = convolve(data, self.flow_for, self.flow_back,
                          structure=structure, method=method,
                          dtype=dtype, fill_value=fill_value,
                          func=func)

        return output

    def diff(self, data, dtype=np.float32):
        """
        Calculate the gradient of a dataset along the leading dimension in a
            semi-Lagrangian framework

        Parameters
        ----------
        data : numpy.ndarray
            The data to find the gradient of
        dtype : type, optional (default : np.float32)
            The dtype of the output data

        Returns
        -------
        diff : numpy.ndarray
            The gradient of the data along the leading dimension
        """
        diff_struct = np.zeros([3,3,3])
        diff_struct[:,1,1] = 1
        diff_func = lambda x:np.nansum([x[2]-x[1], x[1]-x[0]], axis=0) \
                                      * 1/np.maximum(np.sum([np.isfinite(x[2]),
                                                             np.isfinite(x[0])], 0), 1)
        diff = self.convolve(data, structure=diff_struct,
                             func=diff_func,
                             dtype=dtype)
        return diff

    def sobel(self, data, method='linear', dtype=None,
              fill_value=np.nan, direction=None,):
        """
        Sobel edge detection algorithm in a semi-Lagrangian space

        Parameters
        ----------
        data : numpy.ndarray
            Data in which to detect edges
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
        res = sobel(data, self.flow_for, self.flow_back,
                    method=method, dtype=dtype,
                    fill_value=fill_value, direction=direction)

        return res

    def watershed(self, field, markers, mask=None,
                  structure=ndi.generate_binary_structure(3,1)):
        """
        Watershed segmentation of a sequence of images in a Semi-Lagrangian
            framework.

        Parameters
        ----------
        field : numpy.ndarray
            Array like data to be segmented using the watershed algorithm

        markers : numpy.ndarray
            Array of markers to seed segmentation

        mask : numpy.ndarray, optional (default : None)
            Array of locations to mask during segmentation. These areas will not be
                included in any of the segments

        structure : numpy.ndarray, optional (default 1 conectivity)
            Structuring array for the watershed segmentation. Defaults to a
                connectivity of 1, provided by ndi.generate_binary_structure

        Returns
        -------
        output : numpy.ndarray
            The resulting labelled regions of the input data corresponding to each
                marker
        """
        output = watershed(self, field, markers, mask=mask,
                           structure=structure)

        return output

    def label(self, data, structure=ndi.generate_binary_structure(3,1),
              dtype=np.int32, overlap=0, subsegment_shrink=0):
        """
        Label 3d connected objects in a semi-Lagrangian reference frame

        Parameters
        ----------
        mask : numpy.ndarray
            A 3d array of in which non-zero values are treated as regions to be
            labelled
        structure : numpy.ndarray - optional
            A (3,3,3) boolean array defining the connectivity between each point
            and its neighbours. Defaults to square connectivity
        dtype : dtype - optional
            Dtype for the returned labelled array. Defaults to np.int32
        overlap : float - optional
            The required minimum overlap between subsequent labels (when accounting
                for Lagrangian motion) to consider them a continous object. Defaults
                to 0.
        subsegment_shrink : float - optional
            The proportion of each regions approximate radius to shrink it by when
                performing subsegmentation. If 0 subsegmentation will not be
                performed. Defaults to 0.
        peak_min_distance : int - optional
            The minimum distance between maxima allowed when performing
                subsegmentation. Defaults to 5
        """
        return flow_label(self, data, structure=structure, dtype=dtype,
                          overlap=overlap, subsegment_shrink=subsegment_shrink)


"""
Dicts to convert keyword inputs to opencv flags for flow keywords
"""
flow_flags = {'default':0, 'gaussian':cv2.OPTFLOW_FARNEBACK_GAUSSIAN}

"""
Dicts to convert keyword inputs to opencv flags for remap keywords
"""
border_modes = {'constant':cv2.BORDER_CONSTANT,
                'nearest':cv2.BORDER_REPLICATE,
                'reflect':cv2.BORDER_REFLECT,
                'mirror':cv2.BORDER_REFLECT_101,
                'wrap':cv2.BORDER_WRAP,
                'isolated':cv2.BORDER_ISOLATED,
                'transparent':cv2.BORDER_TRANSPARENT}

interp_modes = {'nearest':cv2.INTER_NEAREST,
                'linear':cv2.INTER_LINEAR,
                'cubic':cv2.INTER_CUBIC,
                'lanczos':cv2.INTER_LANCZOS4}

# Let's create a new optical_flow derivation function
vr_model = cv2.VariationalRefinement.create()

def get_of_model(model):
    """
    Initiates an opencv optical flow model

    Parameters
    ----------
    model : string
        The model to initatite. Must be one of 'Farneback', 'DeepFlow', 'PCA',
            'SimpleFlow', 'SparseToDense', 'DIS', 'DenseRLOF', 'DualTVL1'

    Returns
    -------
    of_model : cv2.DenseOpticalFlow
        opencv optical flow model
    """
    if model == "Farneback":
        of_model = cv2.optflow.createOptFlow_Farneback()
    elif model == "DeepFlow":
        of_model = cv2.optflow.createOptFlow_DeepFlow()
    elif model == "PCA":
        of_model = cv2.optflow.createOptFlow_PCAFlow()
    elif model == "SimpleFlow":
        of_model = cv2.optflow.createOptFlow_SimpleFlow()
    elif model == "SparseToDense":
        of_model = cv2.optflow.createOptFlow_SparseToDense()
    elif model == "DIS":
        of_model = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        of_model.setUseSpatialPropagation(True)
    elif model == "DenseRLOF":
        of_model = cv2.optflow.createOptFlow_DenseRLOF()
    elif model == "DualTVL1":
        of_model = cv2.optflow.createOptFlow_DualTVL1()
    else:
        raise ValueError("'model' parameter must be one of: 'Farneback', 'DeepFlow', 'PCA', 'SimpleFlow', 'SparseToDense', 'DIS', 'DenseRLOF', 'DualTVL1'")

    return of_model

def get_flow(data, model="DIS", vr_steps=0, smoothing_passes=0):
    """
    Calculates forward and backward optical flow vectors for a given set of data

    Parameters
    ----------
    data : numpy.ndarray
        Array of data to calculate optical flow for. Flow vectors are calculated
            along the leading dimension
    model : string, optional (default : 'DIS')
        opencv optical flow model to use
    vr_steps : int, optional (default : 0)
        Number of variational refinement operations to perform
    smoothing_passes : int, optional (default : 0)
        Number of smoothing operation between the forward and backward flow to
            perform

    Returns
    -------
    forward_flow : numpy.ndarray
        Array of optical flow vectors acting forward along the leading dimension
            of data
    backward_flow : numpy.ndarray
        Array of optical flow vectors acting backwards along the leading
            dimension of data
    """
    if isinstance(data, xr.DataArray):
        data = data.compute().data

    vmax = np.nanmax(data)
    vmin = np.nanmin(data)

    forward_flow = np.full(data.shape+(2,), np.nan, dtype=np.float32)
    backward_flow = np.full(data.shape+(2,), np.nan, dtype=np.float32)

    of_model = get_of_model(model)

    for i in range(data.shape[0]-1):
        prev_frame = to_8bit(data[i], vmin, vmax)
        next_frame = to_8bit(data[i+1], vmin, vmax)

        forward_flow[i], backward_flow[i+1] = get_flow_frame(prev_frame,
                                                             next_frame,
                                                             of_model,
                                                             vr_steps=vr_steps,
                                                             smoothing_steps=smoothing_passes)

    forward_flow[-1] = -backward_flow[-1]
    backward_flow[0] = -forward_flow[0]

    return forward_flow, backward_flow

def to_8bit(array, vmin=None, vmax=None):
    """
    Converts an array to an 8-bit range between 0 and 255
    """
    if vmin is None:
        vmin = np.nanmin(array)
    if vmax is None:
        vmax = np.nanmax(array)
    if vmin==vmax:
        factor = 0
    else:
        factor = 255 / (vmax-vmin)
    array_out = (array-vmin) * factor
    return array_out.astype('uint8')

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    locs = flow.copy()
    locs[:,:,0] += np.arange(w)
    locs[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, locs, None, cv2.INTER_LINEAR)
    return res

def get_flow_frame(prev_frame,
                   next_frame,
                   of_model,
                   vr_steps=0,
                   smoothing_steps=0):

    forward_flow = of_model.calc(prev_frame, next_frame, None)

    if vr_steps > 0:
        forward_flow = vr_model.calc(prev_frame, next_frame, forward_flow)

    backward_flow = of_model.calc(next_frame, prev_frame, None)

    if vr_steps > 0:
        backward_flow = vr_model.calc(next_frame, prev_frame, backward_flow)

    if smoothing_steps > 0:
        for i in range(smoothing_steps):
            forward_flow, backward_flow = smooth_flow_step(forward_flow, backward_flow)

    return forward_flow, backward_flow

def smooth_flow_step(forward_flow, backward_flow):
    forward_flow, backward_flow = (np.nanmean([forward_flow,
                                               np.stack([-warp_flow(backward_flow[...,0], forward_flow),
                                                         -warp_flow(backward_flow[...,1], forward_flow)], -1)], 0),
                                   np.nanmean([backward_flow,
                                               np.stack([-warp_flow(forward_flow[...,0], backward_flow),
                                                         -warp_flow(forward_flow[...,1], backward_flow)], -1)], 0))

    return forward_flow, backward_flow
