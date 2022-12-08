import numpy as np
from numpy import ma
import pandas as pd
import xarray as xr
import cv2 as cv
from scipy import ndimage as ndi
import warnings

from .convolve import warp_flow_cv, convolve
from .label import flow_label
from .sobel import sobel
from .watershed import watershed

class Flow:
    """
    Class to perform semi-lagrangian operations using optical flow
    """
    def __init__(self, dataset, smoothing_passes=1, flow_kwargs={}):
        self.get_flow(dataset, smoothing_passes, flow_kwargs)

    def get_flow(self, data, smoothing_passes, flow_kwargs):
        self.shape = data.shape
        self.flow_for = np.full(self.shape+(2,), np.nan, dtype=np.float32)
        self.flow_back = np.full(self.shape+(2,), np.nan, dtype=np.float32)

        for i in range(self.shape[0]-1):
            print(i, end='\r')
            a, b = data[i].compute().data, data[i+1].compute().data

            self.flow_for[i] = self.cv_flow(a, b, **flow_kwargs)
            self.flow_back[i+1] = self.cv_flow(b, a, **flow_kwargs)
            if smoothing_passes > 0:
                for j in range(smoothing_passes):
                    self._smooth_flow_step(i)

        self.flow_back[0] = -self.flow_for[0]
        self.flow_for[-1] = -self.flow_back[-1]

    def to_8bit(self, array, vmin=None, vmax=None):
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

    def cv_flow(self, a, b, pyr_scale=0.5, levels=5, winsize=16, iterations=3,
                poly_n=5, poly_sigma=1.1, flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN):
        """
        Wrapper function for cv.calcOpticalFlowFarneback
        """
        flow = cv.calcOpticalFlowFarneback(self.to_8bit(a), self.to_8bit(b), None,
                                           pyr_scale, levels, winsize, iterations,
                                           poly_n, poly_sigma, flags)
        return flow

    def _warp_flow_step(self, img, step, method='linear', direction='forward',
                        stencil=ndi.generate_binary_structure(3,1)[0]):
        if img.shape != self.shape[1:]:
            raise ValueError("Image shape does not match flow shape")
        if method == 'linear':
            method = cv.INTER_LINEAR
        elif method =='nearest':
            method = cv.INTER_NEAREST
        else:
            raise ValueError("method must be either 'linear' or 'nearest'")

        h, w = self.shape[1:]
        n = np.sum(stencil!=0)
        offsets = (np.stack(np.where(stencil!=0), -1)[:,np.newaxis,np.newaxis,::-1]-1).astype(np.float32)
        locations = np.tile(offsets, [1,h,w,1])
        locations += np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1)
        if direction=='forward':
            locations += self.flow_for[step]
        elif direction=='backward':
            locations += self.flow_back[step]

        locations = locations.reshape([n*h,w,2])

        if isinstance(img, xr.DataArray):
            out_image = cv.remap(img.data.astype(np.float32),
                                 locations.reshape([n*h,w,2]),
                                 None, method, None, cv.BORDER_CONSTANT,
                                 np.nan).reshape([n,h,w])
        else:
            out_image = cv.remap(img.astype(np.float32),
                                 locations.reshape([n*h,w,2]),
                                 None, method, None, cv.BORDER_CONSTANT,
                                 np.nan).reshape([n,h,w])

        return out_image

    def _smooth_flow_step(self, step):
        flow_for_warp = np.full_like(self.flow_for[step], np.nan)
        flow_back_warp = np.full_like(self.flow_back[step+1], np.nan)

        flow_for_warp[...,0] = -self._warp_flow_step(self.flow_back[step+1,...,0], step)
        flow_for_warp[...,1] = -self._warp_flow_step(self.flow_back[step+1,...,1], step)
        flow_back_warp[...,0] = -self._warp_flow_step(self.flow_for[step,...,0], step+1, direction='backward')
        flow_back_warp[...,1] = -self._warp_flow_step(self.flow_for[step,...,1], step+1, direction='backward')

        self.flow_for[step] = np.nanmean([self.flow_for[step],
                                          flow_for_warp], 0)
        self.flow_back[step+1] = np.nanmean([self.flow_back[step+1],
                                             flow_back_warp], 0)

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


def to_8bit(array, vmin=None, vmax=None):
    """
    Converts an array to an 8-bit range between 0 and 255 with dtype uint8
    """
    if vmin is None:
        vmin = np.nanmin(array)
    if vmax is None:
        vmax = np.nanmax(array)
    array_out = (array-vmin) * 255 / (vmax-vmin)
    return array_out.astype('uint8')

"""
Dicts to convert keyword inputs to opencv flags for flow keywords
"""
flow_flags = {'default':0, 'gaussian':cv.OPTFLOW_FARNEBACK_GAUSSIAN}

def cv_flow(a, b, pyr_scale=0.5, levels=5, winsize=16, iterations=3,
            poly_n=5, poly_sigma=1.1, flags='gaussian'):
    """
    Wrapper function for cv.calcOpticalFlowFarneback
    """
    assert flags in flow_flags, \
        f"{flags} not a valid input for flags keyword, input must be one of {list(flow_flags.keys())}"

    flow = cv.calcOpticalFlowFarneback(to_8bit(a), to_8bit(b), None,
                                       pyr_scale, levels, winsize, iterations,
                                       poly_n, poly_sigma, flow_flags[flags])
    return flow

"""
Dicts to convert keyword inputs to opencv flags for remap keywords
"""
border_modes = {'constant':cv.BORDER_CONSTANT,
                'nearest':cv.BORDER_REPLICATE,
                'reflect':cv.BORDER_REFLECT,
                'mirror':cv.BORDER_REFLECT_101,
                'wrap':cv.BORDER_WRAP,
                'isolated':cv.BORDER_ISOLATED,
                'transparent':cv.BORDER_TRANSPARENT}

interp_modes = {'nearest':cv.INTER_NEAREST,
                'linear':cv.INTER_LINEAR,
                'cubic':cv.INTER_CUBIC,
                'lanczos':cv.INTER_LANCZOS4}
