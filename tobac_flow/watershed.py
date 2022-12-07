"""
Watershed segmentation operations in a semi-Lagrangian framework.
"""
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology._util import (_validate_connectivity,
    _offsets_to_raveled_neighbors)
from skimage.util import crop, regular_seeds
from skimage.segmentation._watershed import _validate_inputs

from ._watershed import watershed_raveled

def watershed(flow, field, markers, mask=None,
              structure=ndi.generate_binary_structure(3,1)):
    """
    Watershed segmentation of a sequence of images in a Semi-Lagrangian
        framework.

    Parameters
    ----------
    flow : tobac_flow.Flow
        Flow object containing the forward and backward flow vectors for the
            data to be segmented

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
    skimage_markers = markers.copy().astype(np.intp)
    skimage_markers[mask] = -1
    connectivity=1
    offset=None
    mask=None
    compactness=0
    watershed_line=False

    image, markers, mask = _validate_inputs(field,
                                            skimage_markers,
                                            mask,
                                            connectivity)

    connectivity, offset = _validate_connectivity(image.ndim,
                                                  connectivity,
                                                  offset)

    # pad the image, markers, and mask so that we can use the mask to
    # keep from running off the edges
    # pad_width = [(p, p) for p in offset]
    # Modify padding by maximum of flow vectors
    pad_offset = offset.copy()
    pad_offset[1] += np.max(np.round(np.abs(flow.flow_for[...,1])).astype(np.intp))
    pad_offset[2] += np.max(np.round(np.abs(flow.flow_for[...,0])).astype(np.intp))
    pad_width = [(p, p) for p in pad_offset]
    image = np.pad(image, pad_width, mode='constant')
    mask = np.pad(mask, pad_width, mode='constant').ravel()
    output = np.pad(markers, pad_width, mode='constant')
    flat_neighborhood = _offsets_to_raveled_neighbors(image.shape,
                                                      connectivity,
                                                      center=offset)
    marker_locations = np.flatnonzero(output)
    image_strides = np.array(image.strides, dtype=np.intp) // image.itemsize

    # Calculate ravelled offsets for flow field
    forward_offset = (np.pad(np.round(flow.flow_for[...,0]).astype(np.intp),
                             pad_width,
                             mode='constant').ravel() * image_strides[2]
                      + np.pad(np.round(flow.flow_for[...,1]).astype(np.intp),
                             pad_width,
                             mode='constant').ravel() * image_strides[1])

    backward_offset = (np.pad(np.round(flow.flow_back[...,0]).astype(np.intp),
                             pad_width,
                             mode='constant').ravel() * image_strides[2]
                      + np.pad(np.round(flow.flow_back[...,1]).astype(np.intp),
                             pad_width,
                             mode='constant').ravel() * image_strides[1])

    forward_offset_locations = (np.round(flat_neighborhood/image_strides[0])==1).astype(np.intp)
    backward_offset_locations = (np.round(flat_neighborhood/image_strides[0])==-1).astype(np.intp)

    watershed_raveled(image.ravel(),
                      marker_locations, flat_neighborhood,
                      forward_offset, backward_offset,
                      forward_offset_locations, backward_offset_locations,
                      mask, image_strides, compactness,
                      output.ravel(),
                      watershed_line)

    output = crop(output, pad_width, copy=True)

    output[output==-1] = 0

    return output
