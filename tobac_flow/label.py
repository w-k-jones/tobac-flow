import warnings
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from tobac_flow.utils.label_utils import (
    flat_label,
    relabel_objects,
    find_overlapping_labels,
)


def subsegment_labels(
    input_mask: np.ndarray[bool], shrink_factor: float = 0.1, peak_min_distance: int = 5
) -> np.ndarray[int]:
    """
    Takes a 3d array (t,y,x) of regions and splits each label at each time step
        into multiple labels based on morphology.

    Parameters
    ----------
    input_mask : numpy.ndarray
        A 3d array of in which non-zero values are treated as regions of
            interest
    shrink_factor : float - optional
        The proportion of the approximate radius to shrink each object. Defaults
            to 0.1
    peak_min_distance : int - optional
        The minimum distance between neighbouring peaks for additional maxima
            added to the subsegment markers. Defaults to 5

    Returns
    -------
    subseg_labels : numpy.ndarray
        A 3d array of labels, where each subsegment label at each time step has
            an individual integer value. Same size as input_mask

    The splitting is performed by approximating each region as a circle,
        calculating an approximate radius from the area, and then shrinking each
        region by the approximate radius times the 'shrink_factor'. Objects that
        are more irregular in shape will be shrunk more than those that have
        more regular shapes. Each separate section of the region is given its
        own label, and the original region is segmented between them by
        watershedding. To ensure that smaller objects are not eroded by the
        shrinking process, maxima in the distance from the field to the edge of
        each region are also included as seeds to the watershed.
    """
    # Individually label regions at each time step
    labels = flat_label(input_mask != 0)

    # Calculate the distance from the edge of each region for pixels inside each labe;
    dist_mask = ndi.morphology.distance_transform_edt(labels, [1e9, 1, 1])
    pixel_counts = np.bincount(labels.ravel())
    dist_mask /= ((pixel_counts / np.pi) ** 0.5)[labels]

    shrunk_markers = dist_mask > shrink_factor

    # Find local maxima to see if we've missed any labels by "over-shrinking"
    local_maxima = np.zeros_like(shrunk_markers)
    for i in range(local_maxima.shape[0]):
        maxima = peak_local_max(
            dist_mask[i], min_distance=peak_min_distance, threshold_abs=1e-8
        )
        local_maxima[i][tuple(maxima.T)] = True

    shrunk_markers = flat_label(np.logical_or(shrunk_markers, local_maxima))
    shrunk_markers[labels == 0] = -1

    struct = ndi.generate_binary_structure(3, 1)
    struct[0] = 0
    struct[-1] = 0

    subseg_labels = np.zeros_like(labels)

    for i in range(subseg_labels.shape[0]):
        subseg_labels[i] = watershed(
            -dist_mask[i], shrunk_markers[i], mask=labels[i] != 0
        )

    return subseg_labels


# implement minimum overlap for flow_label function
def flow_label(
    flow,
    mask: np.ndarray[bool],
    structure: np.ndarray[bool] = ndi.generate_binary_structure(3, 1),
    dtype: type = np.int32,
    overlap: float = 0.0,
    absolute_overlap: int = 0,
    subsegment_shrink: float = 0.0,
    peak_min_distance: int = 10,
) -> np.ndarray[int]:
    """
    Label 3d connected objects in a semi-Lagrangian reference frame

    Parameters
    ----------
    flow : tobac_flow.Flow object
        The flow-field object corresponding to the mask being labelled
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
    absolute_overlap : int, optional (default: 1)
        The required minimum overlap in pixels
    subsegment_shrink : float - optional
        The proportion of each regions approximate radius to shrink it by when
            performing subsegmentation. If 0 subsegmentation will not be
            performed. Defaults to 0.
    peak_min_distance : int - optional
        The minimum distance between maxima allowed when performing
            subsegmentation. Defaults to 5
    """
    #     Get flat (2d) labels
    if subsegment_shrink == 0:
        flat_labels = flat_label(mask != 0, structure=structure).astype(dtype)
    else:
        flat_labels = subsegment_labels(
            mask != 0,
            shrink_factor=subsegment_shrink,
            peak_min_distance=peak_min_distance,
        )

    label_struct = structure * np.array([1, 0, 1])[:, np.newaxis, np.newaxis]

    back_labels, forward_labels = flow.convolve(
        flat_labels, method="nearest", dtype=dtype, structure=label_struct, fill_value=0
    )

    bins = np.cumsum(np.bincount(flat_labels.ravel()))
    args = np.argsort(flat_labels.ravel())

    processed_labels = np.zeros(bins.size, dtype=bool)
    label_map = {}

    for label in range(1, bins.size):
        if not processed_labels[label]:
            label_map[label] = [label]
            processed_labels[label] = True

            i = 0
            while i < len(label_map[label]):
                find_neighbour_labels(
                    label_map[label][i],
                    label_map[label],
                    bins,
                    args,
                    processed_labels,
                    forward_labels,
                    back_labels,
                    overlap=overlap,
                    absolute_overlap=absolute_overlap,
                )
                i += 1

    new_labels = np.zeros(mask.shape, dtype=dtype)

    for ik, k in enumerate(label_map):
        for i in label_map[k]:
            if bins[i] > bins[i - 1]:
                new_labels.ravel()[args[bins[i - 1] : bins[i]]] = ik + 1

    if not np.all((new_labels != 0) == (mask != 0)):
        # This may occur if subsegmentation is over zealous
        warnings.warn("Not all regions present in labeled array", RuntimeWarning)
    return new_labels


def find_neighbour_labels(
    label: int,
    label_stack: list[int],
    bins: np.ndarray[int],
    args: np.ndarray[int],
    processed_labels: np.ndarray[int],
    forward_labels: np.ndarray[int],
    back_labels: np.ndarray[int],
    overlap: float = 0,
    absolute_overlap: int = 1,
):
    """
    Find the neighbouring labels at the previous and next time steps to a given
        label

    Parameters
    ----------
    label : int
        The value of the label to find neighbours for

    label_stack : list
        The list of labels in the stack to be processed

    bins : numpy.ndarray
        Cumulative bins counting the number of each label

    args : numpy.ndarray
        The ravelled locations for each label using argsort

    processed_labels : numpy.ndarray
        An array to lookup whether a label has been added to the stack to
            process

    forward_labels : numpy.ndarray
        The labelled regions for the next time step forward

    backward_labels : numpy.ndarray
        The labelled regions for the previous time step

    overlap : float, optional (default : 0)
        The proportion of the area of each label overlapping its neighbours to
            be considered linked. If zero, any amount of overlap will count.

    absolute_overlap : int, optional (default: 1)
        The required minimum overlap in pixels
    """
    if bins[label] > bins[label - 1]:  # check that there are any pixels in this label
        for new_label in find_overlapping_labels(
            forward_labels,
            args[bins[label - 1] : bins[label]],
            bins,
            overlap=overlap,
            absolute_overlap=absolute_overlap,
        ):
            if not processed_labels[new_label]:
                label_stack.append(new_label)
                processed_labels[new_label] = True

        for new_label in find_overlapping_labels(
            back_labels,
            args[bins[label - 1] : bins[label]],
            bins,
            overlap=overlap,
            absolute_overlap=absolute_overlap,
        ):
            if not processed_labels[new_label]:
                label_stack.append(new_label)
                processed_labels[new_label] = True


# implement minimum overlap for flow_label function
def flow_link_overlap(
    flow,
    flat_labels: np.ndarray[int],
    structure: np.ndarray[bool] = ndi.generate_binary_structure(3, 1),
    dtype: type = np.int32,
    overlap: float = 0.0,
    absolute_overlap: int = 0,
) -> np.ndarray[int]:
    """
    Label 3d connected objects in a semi-Lagrangian reference frame

    Parameters
    ----------
    flow : tobac_flow.Flow object
        The flow-field object corresponding to the mask being labelled
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
    absolute_overlap : int, optional (default: 1)
        The required minimum overlap in pixels
    """
    label_struct = structure * np.array([1, 0, 1])[:, np.newaxis, np.newaxis]

    back_labels, forward_labels = flow.convolve(
        flat_labels, method="nearest", dtype=dtype, structure=label_struct, fill_value=0
    )

    bins = np.cumsum(np.bincount(flat_labels.ravel()))
    args = np.argsort(flat_labels.ravel())

    processed_labels = np.zeros(bins.size, dtype=bool)
    label_map = {}

    for label in range(1, bins.size):
        if not processed_labels[label]:
            label_map[label] = [label]
            processed_labels[label] = True

            i = 0
            while i < len(label_map[label]):
                find_neighbour_labels(
                    label_map[label][i],
                    label_map[label],
                    bins,
                    args,
                    processed_labels,
                    forward_labels,
                    back_labels,
                    overlap=overlap,
                    absolute_overlap=absolute_overlap,
                )
                i += 1

    new_labels = np.zeros(flat_labels.shape, dtype=dtype)

    for ik, k in enumerate(label_map):
        for i in label_map[k]:
            if bins[i] > bins[i - 1]:
                new_labels.ravel()[args[bins[i - 1] : bins[i]]] = ik + 1

    if not np.all(new_labels.astype(bool) == flat_labels.astype(bool)):
        # This may occur if subsegmentation is over zealous
        warnings.warn("Not all regions present in labeled array", RuntimeWarning)
    return new_labels
