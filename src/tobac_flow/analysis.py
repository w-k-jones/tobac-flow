import numpy as np
from scipy import ndimage as ndi
from tobac_flow.dataset import (
    add_dataarray_to_ds,
    create_dataarray,
    n_unique_along_axis,
)
from tobac_flow.utils import apply_func_to_labels, apply_weighted_func_to_labels


def find_object_lengths(labels: np.ndarray[int], axis: int = 0) -> np.ndarray[int]:
    """
    Find the length of each label in the leading dimension (usually time)

    Parameters
    ----------
    labels : numpy.ndarray
        Array of labelled regions
    axis : int, optional (default : 0)
        The axis to find the object length along

    Returns
    -------
    object_lengths : numpy.ndarray
        The length of each label in the leading dimension
    """
    object_lengths = np.array(
        [o[axis].stop - o[axis].start for o in ndi.find_objects(labels)]
    )

    return object_lengths


def mask_labels(labels: np.ndarray[int], mask: np.ndarray[bool]) -> np.ndarray[bool]:
    """
    Apply a mask to an array of labelled regions to find which labels overlap
        the mask array

    Parameters
    ----------
    labels : numpy.ndarray
        The array of labeled regions to apply the mask to
    mask : numpy.ndarray
        The mask to apply to the labeled array. Must have the same shape as
            labels

    Returns
    -------
    output : numpy.ndarray
        A 1d array of length (labels.max()) which is True for each label which
            overlaps the masked regions, and False for those which do not
    """
    assert (
        labels.shape == mask.shape
    ), "Labels and mask parameters must have the same shape"
    masked_labels = np.unique(labels[mask])
    output = np.zeros(labels.max() + 1, dtype=bool)
    output[masked_labels] = True
    return output[1:]


def remap_labels(
    labels: np.ndarray[int], locations: np.ndarray[bool] | np.ndarray[int]
) -> np.ndarray[int]:
    """
    Remap a label array to a new array of contiguous values for the labels that
        are True in locations

    Parameters
    ----------
    labels : numpy.ndarray
        The array of labeled regions to remap
    locations : numpy.ndarray
        An array of length (labels.max()) with values of True for labels that
            are to be retained and renumbered, and False for labels to be
            removed

    Returns
    -------
    remapped_labels : numpy.ndarray
        A label array of the same shape as the labels parameter, with the
            regions corresponding to the True values in 'locations' retained
            and renumbered with contiguous integer values
    """
    assert (
        locations.size == labels.max()
    ), "The size of the locations parameter must be equal to the maximum label in the labels parameter"
    remapper = np.zeros(np.nanmax(labels) + 1, labels.dtype)
    if locations.dtype == bool:
        remapper[1:][locations] = np.arange(1, np.sum(locations) + 1)
    else:
        remapper[locations] = np.arange(1, np.sum(locations) + 1)
    remapped_labels = remapper[labels]

    return remapped_labels


def filter_labels_by_length(labels, min_length):
    wh = (
        np.array([o[0].stop - o[0].start for o in ndi.find_objects(labels)])
        >= min_length
    )

    remap = np.zeros([np.nanmax(labels) + 1], labels.dtype)
    remap[1:] = np.cumsum(wh) * wh

    return remap[labels]


def filter_labels_by_mask(labels, mask):
    wh = ndi.labeled_comprehension(
        mask, labels, range(1, np.nanmax(labels) + 1), np.any, None, None
    )

    remap = np.zeros([np.nanmax(labels) + 1], labels.dtype)
    remap[1:] = np.cumsum(wh) * wh

    return remap[labels]


def filter_labels_by_length_and_mask(labels, mask, min_length):
    wh = np.logical_and(
        np.array([o[0].stop - o[0].start for o in ndi.find_objects(labels)])
        >= min_length,
        ndi.labeled_comprehension(
            mask, labels, range(1, np.nanmax(labels) + 1), np.any, None, None
        ),
    )

    remap = np.zeros([np.nanmax(labels) + 1], labels.dtype)
    remap[1:] = np.cumsum(wh) * wh

    return remap[labels]


def filter_labels_by_multimask(labels, masks):
    if type(masks) is not type(list()):
        raise ValueError("masks input must be a list of masks to process")

    wh = np.logical_and.reduce(
        [
            ndi.labeled_comprehension(
                m, labels, range(1, np.nanmax(labels) + 1), np.any, np.bool8, 0
            )
            for m in masks
        ]
    )

    remap = np.zeros([np.nanmax(labels) + 1], labels.dtype)
    remap[1:] = np.cumsum(wh) * wh

    return remap[labels]


def filter_labels_by_length_and_multimask(labels, masks, min_length):
    if type(masks) is not type(list()):
        raise ValueError("masks input must be a list of masks to process")

    wh = np.logical_and(
        np.array([o[0].stop - o[0].start for o in ndi.find_objects(labels)])
        >= min_length,
        np.logical_and.reduce(
            [
                ndi.labeled_comprehension(
                    m, labels, range(1, np.nanmax(labels) + 1), np.any, np.bool8, 0
                )
                for m in masks
            ]
        ),
    )

    remap = np.zeros([np.nanmax(labels) + 1], labels.dtype)
    remap[1:] = np.cumsum(wh) * wh

    return remap[labels]


def filter_labels_by_length_legacy(labels, min_length):
    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    object_lengths = np.array(
        [o[0].stop - o[0].start for o in ndi.find_objects(labels)]
    )
    counter = 1
    for i in range(bins.size - 1):
        if bins[i + 1] > bins[i]:
            if object_lengths[i] < min_length:
                labels.ravel()[args[bins[i] : bins[i + 1]]] = 0
            else:
                labels.ravel()[args[bins[i] : bins[i + 1]]] = counter
                counter += 1
    return labels


def filter_labels_by_length_and_mask_legacy(labels, mask, min_length):
    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    object_lengths = np.array(
        [o[0].stop - o[0].start for o in ndi.find_objects(labels)]
    )
    counter = 1
    for i in range(bins.size - 1):
        if bins[i + 1] > bins[i]:
            if object_lengths[i] >= min_length and np.any(
                mask.ravel()[args[bins[i] : bins[i + 1]]]
            ):
                labels.ravel()[args[bins[i] : bins[i + 1]]] = counter
                counter += 1
            else:
                labels.ravel()[args[bins[i] : bins[i + 1]]] = 0
    return labels


def filter_labels_by_length_and_multimask_legacy(labels, masks, min_length):
    if type(masks) is not type(list()):
        raise ValueError("masks input must be a list of masks to process")

    bins = np.cumsum(np.bincount(labels.ravel()))
    args = np.argsort(labels.ravel())
    object_lengths = np.array(
        [o[0].stop - o[0].start for o in ndi.find_objects(labels)]
    )
    counter = 1
    for i in range(bins.size - 1):
        if bins[i + 1] > bins[i]:
            if object_lengths[i] >= min_length and np.all(
                [np.any(m.ravel()[args[bins[i] : bins[i + 1]]]) for m in masks]
            ):
                labels.ravel()[args[bins[i] : bins[i + 1]]] = counter
                counter += 1
            else:
                labels.ravel()[args[bins[i] : bins[i + 1]]] = 0
    return labels


def get_stats_for_labels(labels, da, dim=None, dtype=None):
    if not dim:
        dim = labels.name.split("_label")[0]
    if dtype == None:
        dtype = da.dtype
    mean_da = create_dataarray(
        apply_func_to_labels(labels.data, da.data, np.nanmean),
        (dim,),
        f"{dim}_{da.name}_mean",
        long_name=f"Mean of {da.long_name} for each {dim}",
        units=da.units,
        dtype=dtype,
    )
    std_da = create_dataarray(
        apply_func_to_labels(labels.data, da.data, np.nanstd),
        (dim,),
        f"{dim}_{da.name}_std",
        long_name=f"Standard deviation of {da.long_name} for each {dim}",
        units=da.units,
        dtype=dtype,
    )
    max_da = create_dataarray(
        apply_func_to_labels(labels.data, da.data, np.nanmax),
        (dim,),
        f"{dim}_{da.name}_max",
        long_name=f"Maximum of {da.long_name} for each {dim}",
        units=da.units,
        dtype=dtype,
    )
    min_da = create_dataarray(
        apply_func_to_labels(labels.data, da.data, np.nanmin),
        (dim,),
        f"{dim}_{da.name}_min",
        long_name=f"Minimum of {da.long_name} for each {dim}",
        units=da.units,
        dtype=dtype,
    )

    return mean_da, std_da, max_da, min_da


def get_label_stats(da, ds):
    add_dataarray_to_ds(
        create_dataarray(
            np.count_nonzero(da, 0) / da.t.size,
            ("y", "x"),
            f"{da.name}_fraction",
            long_name=f"Fractional coverage of {da.long_name}",
            units="",
            dtype=np.float32,
        ),
        ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            n_unique_along_axis(da.data, 0),
            ("y", "x"),
            f"{da.name}_unique_count",
            long_name=f"Number of unique {da.long_name}",
            units="",
            dtype=np.int32,
        ),
        ds,
    )

    add_dataarray_to_ds(
        create_dataarray(
            np.count_nonzero(da, (1, 2)) / (da.x.size * da.y.size),
            ("t",),
            f"{da.name}_temporal_fraction",
            long_name=f"Fractional coverage of {da.long_name} over time",
            units="",
            dtype=np.float32,
        ),
        ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            n_unique_along_axis(da.data.reshape([da.t.size, -1]), 1),
            ("t",),
            f"{da.name}_temporal_unique_count",
            long_name=f"Number of unique {da.long_name} over time",
            units="",
            dtype=np.int32,
        ),
        ds,
    )


def weighted_statistics_on_labels(labels, da, weights, name=None, dim=None, dtype=None):
    if not dim:
        dim = labels.name.split("_label")[0]
    if dtype == None:
        dtype = da.dtype

    try:
        long_name = da.long_name
    except AttributeError:
        long_name = da.name

    try:
        units = da.units
    except AttributeError:
        units = ""

    def weighted_average(values, weights, ignore_nan=True):
        if ignore_nan:
            wh_nan = np.isnan(values)
            values = values[~wh_nan]
            weights = weights[~wh_nan]

        if np.nansum(weights) == 0:
            return np.nan

        return np.average(values, weights=weights)

    weighted_std = (
        lambda x, w: weighted_average((x - weighted_average(x, w)) ** 2, w) ** 0.5
    )
    weighted_stats = (
        lambda x, w: [
            weighted_average(x, w),
            weighted_std(x, w),
            np.nanmax(x[w > 0]),
            np.nanmin(x[w > 0]),
        ]
        if np.nansum(w) > 0
        else [np.nan, np.nan, np.nan, np.nan]
    )

    stats_array = apply_weighted_func_to_labels(
        labels.data, da.data, weights, weighted_stats
    )

    mean_da = create_dataarray(
        stats_array[..., 0],
        (dim,),
        f"{name}_{da.name}_mean",
        long_name=f"Mean of {long_name} for each {dim}",
        units=units,
        dtype=dtype,
    )

    std_da = create_dataarray(
        stats_array[..., 1],
        (dim,),
        f"{name}_{da.name}_std",
        long_name=f"Standard deviation of {long_name} for each {dim}",
        units=units,
        dtype=dtype,
    )
    max_da = create_dataarray(
        stats_array[..., 2],
        (dim,),
        f"{name}_{da.name}_max",
        long_name=f"Maximum of {long_name} for each {dim}",
        units=units,
        dtype=dtype,
    )
    min_da = create_dataarray(
        stats_array[..., 3],
        (dim,),
        f"{name}_{da.name}_min",
        long_name=f"Minimum of {long_name} for each {dim}",
        units=units,
        dtype=dtype,
    )

    return mean_da, std_da, max_da, min_da
