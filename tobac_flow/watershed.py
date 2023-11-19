"""
Watershed segmentation operations in a semi-Lagrangian framework.
"""
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology._util import (
    _validate_connectivity,
    _offsets_to_raveled_neighbors,
)
from skimage.util import crop
from skimage.segmentation._watershed import _validate_inputs

from tobac_flow._watershed import watershed_raveled


def watershed(
    forward_flow: np.ndarray[float],
    backward_flow: np.ndarray[float],
    field: np.ndarray[float],
    markers: np.ndarray[int],
    mask: np.ndarray[bool] | None = None,
    connectivity: int | np.ndarray[bool] = 1,
) -> np.ndarray[int]:
    """
    Watershed segmentation of a sequence of images in a Semi-Lagrangian
        framework.

    Parameters
    ----------
    forward_flow : numpy.ndarray
        Array of optical flow vectors acting forward along the leading dimension
            of data
    backward_flow : numpy.ndarray
        Array of optical flow vectors acting backwards along the leading
            dimension of data
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

    See Also
    --------
    skimage.segmentation.watershed : the original function that this is adapted
        from
    """
    offset = None
    compactness = 0
    watershed_line = False

    # image, markers, mask = _validate_inputs(field, markers, mask, connectivity)
    if field.dtype != np.float32:
        field = field.astype(np.float32)
    if markers.shape != field.shape:
        message = (
            f"`markers` (shape {markers.shape}) must have same "
            f"shape as `image` (shape {field.shape})"
        )
        raise ValueError(message)
    if markers.dtype != np.int32:
        markers = markers.astype(np.int32)
    if mask is None:
        # Use a complete `True` mask if none is provided
        mask = np.ones(field.shape, np.int8)
    else:
        if mask.dtype != np.int8:
            mask = mask.astype(np.int8)
        if mask.shape != field.shape:
            message = (
                f"`mask` (shape {mask.shape}) must have same shape "
                f"as `image` (shape {field.shape})"
            )
            raise ValueError(message)

    connectivity, offset = _validate_connectivity(field.ndim, connectivity, offset)

    # pad the image, markers, and mask so that we can use the mask to
    # keep from running off the edges
    # pad_width = [(p, p) for p in offset]
    # Modify padding by maximum of flow vectors
    pad_offset = offset.copy()

    y_flow_maximum = np.maximum(
        np.max(np.round(np.abs(forward_flow[..., 1]))),
        np.max(np.round(np.abs(backward_flow[..., 1]))),
    ).astype(np.int32)

    pad_offset[1] += y_flow_maximum

    x_flow_maximum = np.maximum(
        np.max(np.round(np.abs(forward_flow[..., 0]))),
        np.max(np.round(np.abs(backward_flow[..., 0]))),
    ).astype(np.int32)

    pad_offset[2] += x_flow_maximum

    pad_width = [(p, p) for p in pad_offset]

    field = np.pad(field, pad_width, mode="constant")
    mask = np.pad(mask, pad_width, mode="constant").ravel()
    output = np.pad(markers, pad_width, mode="constant")
    flat_neighborhood = _offsets_to_raveled_neighbors(
        field.shape, connectivity, center=offset
    )
    marker_locations = np.flatnonzero(output)
    image_strides = np.array(field.strides, dtype=np.int32) // field.itemsize

    # Calculate ravelled offsets for flow field
    forward_offset = (
        np.pad(
            np.round(forward_flow[..., 0]).astype(np.int32), pad_width, mode="constant"
        ).ravel()
        * image_strides[2]
        + np.pad(
            np.round(forward_flow[..., 1]).astype(np.int32), pad_width, mode="constant"
        ).ravel()
        * image_strides[1]
    )

    backward_offset = (
        np.pad(
            np.round(backward_flow[..., 0]).astype(np.int32), pad_width, mode="constant"
        ).ravel()
        * image_strides[2]
        + np.pad(
            np.round(backward_flow[..., 1]).astype(np.int32), pad_width, mode="constant"
        ).ravel()
        * image_strides[1]
    )

    forward_offset_locations = (
        np.round(flat_neighborhood / image_strides[0]) == 1
    ).astype(np.int32)

    backward_offset_locations = (
        np.round(flat_neighborhood / image_strides[0]) == -1
    ).astype(np.int32)

    watershed_raveled(
        field.ravel(),
        marker_locations,
        flat_neighborhood,
        forward_offset,
        backward_offset,
        forward_offset_locations,
        backward_offset_locations,
        mask,
        image_strides,
        compactness,
        output.ravel(),
        watershed_line,
    )

    output = crop(output, pad_width, copy=True)

    return output
