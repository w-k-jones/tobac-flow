from multiprocessing import Value
import numpy as np
import scipy.ndimage as ndi
from typing import Callable


def labeled_comprehension(
    field: np.ndarray,
    labels: np.ndarray[int],
    func: Callable,
    index: list[int] | None = None,
    dtype: type | None = None,
    default: float | None = None,
    pass_positions: bool = False,
) -> np.ndarray:
    """
    Wrapper for the scipy.ndimage.labeled_comprehension function

    Parameters
    ----------
    field : numpy.ndarray
        Data to apply the function overlap
    labels : numpy.ndarray
        The array of labeled regions
    func : function
        The function to apply to each labelled region
    index : list, optional (default : None)
        The label values to apply the comprehension over. Default value of None
            will apply the comprehension to all non-zero labels
    dtype : type, optional (default : None)
        The dtype of the output. Defaults to that of the input field
    default : scalar, optional (default : None)
        The value to return if a label does not exist
    pass_positions : bool, optional (default : False)
        If true, will pass the indexes of the label locations to the funtion in
            labeled_comprehension

    Returns
    -------
    comp : numpy.ndarray
        An array of the result of func applied to each labelled region included
            in index
    """
    if not dtype:
        dtype = field.dtype

    if index is None:
        index = np.unique(labels[labels != 0])

    comp = ndi.labeled_comprehension(
        field, labels, index, func, dtype, default, pass_positions
    )

    return comp


def apply_func_to_labels(
    labels: np.ndarray[int],
    *fields: tuple[np.ndarray],
    func: Callable = np.mean,
    index: None | list[int] = None,
    default: None | float = None,
) -> np.ndarray:
    """
    Apply a given function to the regions of an array given by an array of
        labels. Functions similar to ndi.labeled_comprehension, but may be more
        adaptable in some circumstances

    Parameters
    ----------
    labels : np.ndarray[int]
        labels of each region to apply function to
    *fields : tuple[np.ndarray]
        fields to give as arguments to each function call. Must have the same
            shape as labels
    func: Callable, optional (default: np.mean)
        function to apply over each region
    index: None | list[int], optional (default: None)
        list of indexes of regions in labels to apply function to. If None, will
            default to all integers between 1 and the maximum value in labels
    default: None | float, optional (default: None)
        default value to return in a region has no values
    """
    broadcast_fields = np.broadcast_arrays(labels, *fields)
    broadcast_labels = broadcast_fields[0]
    broadcast_fields = broadcast_fields[1:]

    if index is None:
        min_label = np.minimum(np.min(labels), 0)
        n_bins = np.max(labels) - min_label + 1
        index = range(1, n_bins)
    else:
        min_label = np.minimum.reduce([np.min(index) - 1, np.min(labels), 0])
        n_bins = np.maximum(np.max(index), np.max(labels)) - min_label + 1

    bins = np.cumsum(
        np.bincount(broadcast_labels.ravel() - min_label, minlength=n_bins)
    )
    args = np.argsort(broadcast_labels.ravel())
    # Format the default value in case func has multiple return values
    try:
        _ = iter(default)
        assert not isinstance(default, str)
    except (TypeError, AssertionError):
        i = np.where(np.diff(bins))[0][0] + 1
        return_vals = func(
            *[field.ravel()[args[bins[i - 1] : bins[i]]] for field in broadcast_fields]
        )
        try:
            assert not isinstance(return_vals, str)
            n_return_vals = len(return_vals)
        except (AssertionError, TypeError):
            default_vals = default
        else:
            default_vals = [default] * n_return_vals
    else:
        if len(default) == 1 and not isinstance(default, str):
            default_vals = default[0]
        else:
            default_vals = default

    return np.stack(
        [
            func(
                *[
                    field.ravel()[args[bins[i - min_label - 1] : bins[i - min_label]]]
                    for field in broadcast_fields
                ]
            )
            if bins[i - min_label] > bins[i - min_label - 1]
            else default_vals
            for i in index
        ],
        -1,
    ).squeeze()


def flat_label(
    mask: np.ndarray[bool],
    structure: np.ndarray[int] = ndi.generate_binary_structure(3, 1),
    dtype=np.int32,
) -> np.ndarray[int]:
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


def make_step_labels(labels):
    step_labels = flat_label(labels != 0)
    step_labels += labels * step_labels.max()
    return relabel_objects(step_labels)


def get_step_labels_for_label(
    labels: np.ndarray[int], step_labels: np.ndarray[int]
) -> list[int]:
    """
    Given the output from flat_label, and the original label array, find which
        step labels correspond to each original label

    Parameters
    ----------
    labels : numpy.ndarray
        The orginal array of labels
    step_labels : numpy.ndarray
        The array of labels split into separate steps

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


def relabel_objects(labels: np.ndarray[int], inplace=False) -> np.ndarray[int]:
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
    if not inplace:
        labels = np.zeros_like(labels)
    counter = 1
    for i in range(bins.size - 1):
        if bins[i + 1] > bins[i]:
            labels.ravel()[args[bins[i] : bins[i + 1]]] = counter
            counter += 1
    return labels


def slice_labels(labels: np.ndarray[int]) -> np.ndarray[int]:
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
    # step_labels = np.zeros_like(labels)
    # max_label = 0
    # for i in range(labels.shape[0]):
    #     step_labels[i] = relabel_objects(labels[i])
    #     step_labels[i][np.nonzero(step_labels[i])] += max_label
    #     max_label = step_labels.max()
    max_step_label = np.cumsum(
        np.max(labels, axis=tuple(range(1, len(labels.shape))))
    ).reshape([-1] + [1] * (len(labels.shape) - 1))
    step_labels = relabel_objects(
        (labels + max_step_label) * (labels != 0).astype(int), inplace=True
    )
    return step_labels


def find_overlapping_labels(
    labels: np.ndarray[int],
    locs: np.ndarray[int],
    bins: np.ndarray[int],
    overlap: float = 0,
    absolute_overlap: int = 0,
) -> list[int]:
    """
    Find which labels overlap at the locations given by locs, accounting for
    (proportional) overlap and absolute overlap requirements
    """
    n_locs = len(locs)
    if n_locs > 0:
        overlap_labels = labels.ravel()[locs]
        overlap_bins = np.bincount(np.maximum(overlap_labels, 0))
        return [
            new_label
            for new_label in np.unique(overlap_labels)
            if new_label != 0
            and overlap_bins[new_label] > absolute_overlap
            and overlap_bins[new_label]
            >= overlap * np.minimum(n_locs, bins[new_label] - bins[new_label - 1])
        ]
    else:
        return []


__all__ = (
    "labeled_comprehension",
    "apply_func_to_labels",
    "flat_label",
    "make_step_labels",
    "get_step_labels_for_label",
    "relabel_objects",
    "slice_labels",
    "find_overlapping_labels",
)
