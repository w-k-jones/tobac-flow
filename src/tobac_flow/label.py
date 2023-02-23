import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def flat_label(mask, structure=ndi.generate_binary_structure(3, 1), dtype=np.int32):
    """
    For a 3d+ field, return connected component labels that do not connect
        across the highest order dimension. In general this is used for finding
        labels at individual time steps.

    Parameters
    ----------
    mask : numpy.ndarray
        The (boolean) array of masked values to label into separate regions
    structure : numpy.ndarray, optional
        The structuring element to connect labels. This defaults to 1
            connectivity using ndi.generate_binary_structure(3,1).
    dtype : type, optional (default : np.int32)
        The dtype of the ouput labels

    Returns
    -------
    Output : numpy.ndarray
        The array of connected labelled regions

    See Also
    --------
    scipy.ndimage.label : this is the labelling routine used, with the
        connectivity of the highest dimension set to 0
    get_step_labels_for_label : find which step labels correspond to each input
        label
    """
    label_struct = structure.copy()
    label_struct[0] = 0
    label_struct[-1] = 0

    output = ndi.label(mask, structure=label_struct, output=dtype)[0]
    return output


def get_step_labels_for_label(
    labels, step_labels, structure=ndi.generate_binary_structure(3, 1), dtype=np.int32
):
    """
    Given the output from flat_label, and the original label array, find which
        step labels correspond to each original label

    Parameters
    ----------
    labels : numpy.ndarray
        The orginal array of labels
    step_labels : numpy.ndarray
        The array of labels split into separate steps
    structure : numpy.ndarray, optional
        The structuring element to connect labels. This defaults to 1
            connectivity using ndi.generate_binary_structure(3,1).
    dtype : type, optional (default : np.int32)
        The dtype of the ouput labels

    Returns
    -------
    Output : numpy.ndarray
        The array of labelled regions at each time step

    See Also
    --------
    scipy.ndimage.label : this is the labelling routine used, with the
        connectivity of the highest dimension set to 0
    """
    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    return [
        np.unique(step_labels.ravel()[args[bins[i] : bins[i + 1]]])
        if bins[i + 1] > bins[i]
        else None
        for i in range(bins.size - 1)
    ]


def relabel_objects(labels):
    """
    Given an array of labelled regions, renumber the labels so that they are
        contiguous integers

    Parameters
    ----------
    labels : numpy.ndarray
        The array of labels to renumber

    Returns
    -------
    new_labels : numpy.ndarray
        The regions labelled with contiguous integers
    """
    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    new_labels = np.zeros_like(labels)
    counter = 1
    for i in range(bins.size - 1):
        if bins[i + 1] > bins[i]:
            new_labels.ravel()[args[bins[i] : bins[i + 1]]] = counter
            counter += 1
    return new_labels


def slice_labels(labels):
    """
    Given an array of labelled regions, will split these regions into separate
        labels along the leading dimension. In general this is used for finding
        the section of each label at each individual time step. Note that unlike
        flat_label the regions for each label at each time step will remain as a
        single label, even if it is not all connected at that step.

    Parameters
    ----------
    labels : numpy.ndarray
        The array of labels to split into individual time steps

    Returns
    -------
    step_labels : numpy.ndarray
        An array of labels corresponding the regions associated with the input
            labels at individual time steps
    """
    step_labels = np.zeros_like(labels)
    max_label = 0
    for i in range(labels.shape[0]):
        step_labels[i] = relabel_objects(labels[i])
        step_labels[i][np.nonzero(step_labels[i])] += max_label
        max_label = step_labels.max()
    return step_labels


def slice_label_da(label_da):
    """
    Slice labels into individual time steps for a dataarray of input labels, and
        set attributes of the returned dataarray appropriately

    Parameters
    ----------
    labels : xarray.DataArray
        The array of labels to split into individual time steps

    Returns
    -------
    step_labels : xarray.DataArray
        An array of labels corresponding the regions associated with the input
            labels at individual time steps
    """
    label_name = label_da.name.split("_label")[0]
    step_labels = create_dataarray(
        slice_labels(label_da.data),
        label_da.dims,
        f"{label_name}_step_label",
        long_name=f"{label_da.long_name} at each time step",
        units="",
        dtype=np.int32,
    )
    return step_labels


def subsegment_labels(input_mask, shrink_factor=0.1, peak_min_distance=5):
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
    mask,
    structure=ndi.generate_binary_structure(3, 1),
    dtype=np.int32,
    overlap=0.0,
    subsegment_shrink=0.0,
    peak_min_distance=10,
):
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
        flat_labels = flat_label(mask.astype(bool), structure=structure).astype(dtype)
    else:
        flat_labels = subsegment_labels(
            mask.astype(bool), shrink_factor=subsegment_shrink
        )

    label_struct = structure * np.array([1, 0, 1])[:, np.newaxis, np.newaxis]

    back_labels, forward_labels = flow.convolve(
        flat_labels, method="nearest", dtype=dtype, structure=label_struct
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
                )
                i += 1

    new_labels = np.zeros(mask.shape, dtype=dtype)

    for ik, k in enumerate(label_map):
        for i in label_map[k]:
            if bins[i] > bins[i - 1]:
                new_labels.ravel()[args[bins[i - 1] : bins[i]]] = ik + 1

    if not np.all(new_labels.astype(bool) == mask.astype(bool)):
        # This may occur if subsegmentation is over zealous
        warnings.warn("Not all regions present in labeled array", RuntimeWarning)
    return new_labels


def find_neighbour_labels(
    label,
    label_stack,
    bins,
    args,
    processed_labels,
    forward_labels,
    back_labels,
    overlap=0,
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
    """
    if bins[label] > bins[label - 1]:  # check that there are any pixels in this label
        forward_lap = forward_labels.ravel()[args[bins[label - 1] : bins[label]]]
        forward_bins = np.bincount(np.maximum(forward_lap, 0))
        for new_label in np.unique(forward_lap):
            if (
                new_label > 0
                and not processed_labels[new_label]
                and forward_bins[new_label]
                >= overlap
                * np.minimum(
                    bins[label] - bins[label - 1], bins[new_label] - bins[new_label - 1]
                )
            ):

                label_stack.append(new_label)
                processed_labels[new_label] = True

        backward_lap = back_labels.ravel()[args[bins[label - 1] : bins[label]]]
        backward_bins = np.bincount(np.maximum(backward_lap, 0))
        for new_label in np.unique(backward_lap):
            if (
                new_label > 0
                and not processed_labels[new_label]
                and backward_bins[new_label]
                >= overlap
                * np.minimum(
                    bins[label] - bins[label - 1], bins[new_label] - bins[new_label - 1]
                )
            ):

                label_stack.append(new_label)
                processed_labels[new_label] = True
