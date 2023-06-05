from typing import Callable
from datetime import datetime
import numpy as np
import xarray as xr
import cv2
from scipy import ndimage as ndi

from tobac_flow.convolve import convolve
from tobac_flow.label import flow_label, flow_link_overlap
from tobac_flow.sobel import sobel
from tobac_flow.watershed import watershed

from tobac_flow.core import Abstract_Flow
from tobac_flow.utils import (
    to_8bit,
    select_normalisation_method,
    select_of_model,
    warp_flow,
    mse,
)


def create_flow(
    data: np.ndarray,
    model: str = "DIS",
    vr_steps: int = 0,
    smoothing_passes: int = 0,
    interp_method: str = "linear",
    max_value=20,
) -> "Flow":
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

    Returns
    -------
    flow : Flow
        A Flow object with optical flow vectors calculated from the input data
    """
    forward_flow, backward_flow = calculate_flow(
        data,
        model=model,
        vr_steps=vr_steps,
        smoothing_passes=smoothing_passes,
        interp_method=interp_method,
    )

    forward_flow = np.minimum(np.maximum(forward_flow, -max_value), max_value)
    backward_flow = np.minimum(np.maximum(backward_flow, -max_value), max_value)

    flow = Flow(forward_flow, backward_flow)

    return flow


class Flow(Abstract_Flow):
    """
    Class to perform semi-lagrangian operations using optical flow
    """

    def __init__(
        self, forward_flow: np.ndarray[float], backward_flow: np.ndarray[float]
    ) -> None:
        """
        Initialise the flow object with a data array and calculate the optical
            flow vectors for that array
        """
        if forward_flow.shape != backward_flow.shape:
            raise ValueError(
                "Forward and backward flow vector arrays must have the same shape"
            )
        if forward_flow.shape[-1] != 2:
            raise ValueError(
                "Flow vectors must have a size of 2 in the trailing dimension"
            )
        self.shape = forward_flow.shape[:-1]
        self.forward_flow = forward_flow
        self.backward_flow = backward_flow

    @property
    def flow(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Return the flow vectors
        """
        return self.forward_flow, self.backward_flow

    def __getitem__(self, items: tuple) -> "Flow":
        """
        Return a subset of the flow object
        """
        return Flow(self.forward_flow[items], self.backward_flow[items])

    def convolve(
        self,
        data: np.ndarray[float],
        structure: np.ndarray[bool] = ndi.generate_binary_structure(3, 1),
        method: str = "linear",
        fill_value: float = np.nan,
        dtype: type = np.float32,
        func: Callable | None = None,
    ) -> np.ndarray:
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
        assert (
            data.shape == self.shape
        ), "Data input must have the same shape as the Flow object"

        output = convolve(
            data,
            self.forward_flow,
            self.backward_flow,
            structure=structure,
            method=method,
            dtype=dtype,
            fill_value=fill_value,
            func=func,
        )

        return output

    def diff(
        self, data: np.ndarray[float], method: str = "linear", dtype: type = np.float32
    ) -> np.ndarray[float]:
        """
        Calculate the gradient of a dataset along the leading dimension in a
            semi-Lagrangian framework

        Parameters
        ----------
        data : numpy.ndarray
            The data to find the gradient of
        method : str, optional (default : linear)
            The interpolation method used for calculating the difference
        dtype : type, optional (default : np.float32)
            The dtype of the output data

        Returns
        -------
        diff : numpy.ndarray
            The gradient of the data along the leading dimension
        """
        diff_struct = np.zeros([3, 3, 3])
        diff_struct[:, 1, 1] = 1
        diff_func = (
            lambda x: np.nansum([x[2] - x[1], x[1] - x[0]], axis=0)
            * 1
            / np.maximum(np.sum([np.isfinite(x[2]), np.isfinite(x[0])], 0), 1)
        )
        diff = self.convolve(
            data, structure=diff_struct, func=diff_func, method=method, dtype=dtype
        )

        return diff

    def sobel(
        self,
        data: np.ndarray[float],
        method: str = "linear",
        dtype: type = None,
        fill_value: float = np.nan,
        direction: str | None = None,
    ) -> np.ndarray:
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
        res = sobel(
            data,
            self.forward_flow,
            self.backward_flow,
            method=method,
            dtype=dtype,
            fill_value=fill_value,
            direction=direction,
        )

        return res

    def watershed(
        self,
        field: np.ndarray[float],
        markers: np.ndarray[int],
        mask: np.ndarray[bool] | None = None,
        structure: np.ndarray[bool] = ndi.generate_binary_structure(3, 1),
    ) -> np.ndarray[int]:
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
        output = watershed(
            self.forward_flow,
            self.backward_flow,
            field,
            markers,
            mask=mask,
            structure=structure,
        )

        return output

    def label(
        self,
        data: np.ndarray[bool],
        structure: np.ndarray[bool] = ndi.generate_binary_structure(3, 1),
        dtype: type = np.int32,
        overlap: float = 0,
        subsegment_shrink: float = 0,
    ) -> np.ndarray[int]:
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

        Returns
        -------
        labels : numpy.ndarray[int]
            The labelled array of the regions in mask
        """
        labels = flow_label(
            self,
            data,
            structure=structure,
            dtype=dtype,
            overlap=overlap,
            subsegment_shrink=subsegment_shrink,
        )

        return labels

    def link_overlap(
        self,
        data: np.ndarray[bool],
        structure: np.ndarray[bool] = ndi.generate_binary_structure(3, 1),
        dtype: type = np.int32,
        overlap: float = 0,
    ) -> np.ndarray[int]:
        """
        Link existing labels to form new, contiguous labels
        """
        labels = flow_link_overlap(
            self,
            data,
            structure=structure,
            dtype=dtype,
            overlap=overlap,
        )

        return labels


# Let's create a new optical_flow derivation function
vr_model = cv2.VariationalRefinement.create()


def calculate_flow(
    data: np.ndarray | xr.DataArray,
    model: str = "DIS",
    vr_steps: int = 0,
    smoothing_passes: int = 0,
    interp_method: str = "linear",
    normalisation_method: str = "linear",
    **normalisation_kwargs: None | dict
) -> tuple[np.ndarray, np.ndarray]:
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
    normalisation_method : str, optional (default : linear)
        Normalisation method to apply to each pair of frames
    **normalisation_kwargs : dict, optional
        Dictionary of keyword parameters to pass to the selected normalisation
            method

    Returns
    -------
    forward_flow : numpy.ndarray
        Array of optical flow vectors acting forward along the leading dimension
            of data
    backward_flow : numpy.ndarray
        Array of optical flow vectors acting backwards along the leading
            dimension of data
    """
    of_model = select_of_model(model)

    norm_method = select_normalisation_method(normalisation_method)

    if isinstance(data, xr.DataArray):
        data = data.compute().data

    forward_flow = np.full(data.shape + (2,), np.nan, dtype=np.float32)
    backward_flow = np.full(data.shape + (2,), np.nan, dtype=np.float32)

    for i in range(data.shape[0] - 1):
        prev_frame, next_frame = to_8bit(
            norm_method(data[i : i + 2], **normalisation_kwargs), 0, 1
        )

        forward_flow[i], backward_flow[i + 1] = calculate_flow_frame(
            prev_frame,
            next_frame,
            of_model,
            vr_steps=vr_steps,
            smoothing_steps=smoothing_passes,
            interp_method=interp_method,
        )

    forward_flow[-1] = -backward_flow[-1]
    backward_flow[0] = -forward_flow[0]

    return forward_flow, backward_flow


def calculate_flow_2(
    a: np.ndarray | xr.DataArray,
    b: np.ndarray | xr.DataArray,
    model: str = "DIS",
    vr_steps: int = 0,
    smoothing_passes: int = 0,
    normalisation_method: str = "linear",
    **normalisation_kwargs: None | dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates forward and backward optical flow vectors from two datasets

    Parameters
    ----------
    a : numpy.ndarray
        Array of data to calculate optical flow for. Flow vectors are calculated
            along the leading dimension
    b : numpy.ndarray
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
    of_model = select_of_model(model)

    norm_method = select_normalisation_method(normalisation_method)

    if isinstance(a, xr.DataArray):
        a = a.compute().data
    if isinstance(b, xr.DataArray):
        b = b.compute().data

    forward_flow = np.full(a.shape + (2,), np.nan, dtype=np.float32)
    backward_flow = np.full(a.shape + (2,), np.nan, dtype=np.float32)

    for i in range(a.shape[0] - 1):
        prev_frame, next_frame = to_8bit(
            norm_method(np.stack([a[i], b[i]], 0), **normalisation_kwargs), 0, 1
        )

        forward_flow[i], backward_flow[i + 1] = calculate_flow_frame(
            prev_frame,
            next_frame,
            of_model,
            vr_steps=vr_steps,
            smoothing_steps=smoothing_passes,
        )

    forward_flow[-1] = -backward_flow[-1]
    backward_flow[0] = -forward_flow[0]

    return forward_flow, backward_flow


def calculate_flow_frame(
    prev_frame: np.ndarray[np.uint8],
    next_frame: np.ndarray[np.uint8],
    of_model: cv2.DenseOpticalFlow,
    vr_steps: int = 0,
    smoothing_steps: int = 0,
    interp_method: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the forward and backward optical flow vectors between two
        subsequent images
    """
    forward_flow = of_model.calc(prev_frame, next_frame, None)

    if vr_steps > 0:
        forward_flow = vr_model.calc(prev_frame, next_frame, forward_flow)

    backward_flow = of_model.calc(next_frame, prev_frame, None)

    if vr_steps > 0:
        backward_flow = vr_model.calc(next_frame, prev_frame, backward_flow)

    if smoothing_steps > 0:
        for i in range(smoothing_steps):
            forward_flow, backward_flow = smooth_flow_step(
                forward_flow, backward_flow, method=interp_method
            )

    return forward_flow, backward_flow


def smooth_flow_step(
    forward_flow: np.ndarray, backward_flow: np.ndarray, method: str = "linear"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooth a set of flow vectors by warping and averaging the corresponding
        forward and backward vectors
    """
    # This is a complete mess, we should clean it up
    # For now I've expanded it to see what's actually happening
    forward_flow, backward_flow = (
        np.nanmean(
            [
                forward_flow,
                np.stack(
                    [
                        -warp_flow(backward_flow[..., 0], forward_flow, method=method),
                        -warp_flow(backward_flow[..., 1], forward_flow, method=method),
                    ],
                    -1,
                ),
            ],
            0,
        ),
        np.nanmean(
            [
                backward_flow,
                np.stack(
                    [
                        -warp_flow(forward_flow[..., 0], backward_flow, method=method),
                        -warp_flow(forward_flow[..., 1], backward_flow, method=method),
                    ],
                    -1,
                ),
            ],
            0,
        ),
    )

    return forward_flow, backward_flow


def combine_flow(*args: tuple[Flow]) -> Flow:
    """
    Combine multiple flow objects into one
    """
    forward_magnitudes = [
        ((flow.forward_flow[..., 0] ** 2 + flow.forward_flow[..., 1] ** 2) ** 0.5)[
            ..., np.newaxis
        ]
        for flow in args
    ]

    combined_flow_forward = sum(
        [
            flow.forward_flow * magnitude
            for flow, magnitude in zip(args, forward_magnitudes)
        ]
    ) / sum(forward_magnitudes)

    backward_magnitudes = [
        ((flow.backward_flow[..., 0] ** 2 + flow.backward_flow[..., 1] ** 2) ** 0.5)[
            ..., np.newaxis
        ]
        for flow in args
    ]

    combined_flow_backward = sum(
        [
            flow.backward_flow * magnitude
            for flow, magnitude in zip(args, backward_magnitudes)
        ]
    ) / sum(backward_magnitudes)

    return Flow(combined_flow_forward, combined_flow_backward)


def get_forward_warp(da, flow):
    forward_struct = np.zeros([3, 3, 3], dtype=bool)
    forward_struct[2, 1, 1] = True
    return flow.convolve(da.data, forward_struct)[0]


def flow_diff_mse_estimate(da, flow):
    forward_warp = get_forward_warp(da, flow)
    all_mse = mse(forward_warp, da.data)
    wh = da.data < 273
    cold_mse = mse(forward_warp[wh], da.data[wh])
    return all_mse, cold_mse


def get_flow_residual(da, flow, model="Farneback", vr_steps=1, smoothing_passes=1):
    forward_warp = get_forward_warp(da, flow)
    new_flow, _ = calculate_flow_2(
        da.data,
        forward_warp,
        model=model,
        vr_steps=vr_steps,
        smoothing_passes=smoothing_passes,
    )
    return new_flow


def flow_magnitude(flow, direction="forward") -> np.ndarray[float]:
    if direction == "forward":
        magnitude = (
            flow.forward_flow[..., 0] ** 2 + flow.forward_flow[..., 1] ** 2
        ) ** 0.5
    elif direction == "backward":
        magnitude = (
            flow.backward_flow[..., 0] ** 2 + flow.backward_flow[..., 1] ** 2
        ) ** 0.5
    else:
        raise ValueError("Direction must be one of 'forward', 'backward'")
    return magnitude


def flow_residual_cold_mse_estimate(
    da, flow, model="Farneback", vr_steps=1, smoothing_passes=1
):
    new_flow = get_flow_residual(
        da, flow, model=model, vr_steps=vr_steps, smoothing_passes=smoothing_passes
    )
    magnitude = (new_flow[..., 0] ** 2 + new_flow[..., 1] ** 2) ** 0.5
    magnitude = magnitude[:, 20:-20, 20:-20]
    all_mse = mse(magnitude, np.zeros_like(magnitude))
    wh_cold = da.data[:, 20:-20, 20:-20] < 273
    cold_mse = mse(magnitude[wh_cold], np.zeros_like(magnitude[wh_cold]))
    return all_mse, cold_mse


def time_flow(da, model="Farneback", vr_steps=1, smoothing_passes=1):
    start_date = datetime.now()
    _ = create_flow(
        da, model=model, vr_steps=vr_steps, smoothing_passes=smoothing_passes
    )
    total_time = (datetime.now() - start_date).total_seconds()
    return total_time
