import warnings
from functools import partial
import numpy as np
import xarray as xr
from scipy import ndimage as ndi, stats
from skimage.feature import peak_local_max
from tobac_flow.analysis import (
    filter_labels_by_length,
    filter_labels_by_mask,
    filter_labels_by_length_and_multimask_legacy,
    find_object_lengths,
    mask_labels,
    remap_labels,
)
from tobac_flow.utils import (
    get_time_diff_from_coord,
    linearise_field,
    slice_labels,
    labeled_comprehension,
    make_step_labels,
)
from tobac_flow.flow import Flow


# Filtering of the growth metric occurs in three steps:
# 1. The mean of the growth is taken over 3 time periods (15 minutes)
# 2. The max value of the mean is extended to cover the adjacent time steps
# 3. An opening filter is applied, to remove any regions less than 3x3 pixels in size


def filtered_tdiff(flow, raw_diff):
    """Filtered a time-derivative by taking a moving average in a semi-
    Lagrangian framework

    Parameters
    ----------
    flow : tobac_flow.flow.Flow
        flow object
    raw_diff : np.ndarray
        Unfiltered time differential

    Returns
    -------
    np.ndarray
        Filtered time differential
    """
    t_struct = np.zeros([3, 3, 3])
    t_struct[:, 1, 1] = 1
    # s_struct = ndi.generate_binary_structure(2, 1)[np.newaxis, ...]

    filtered_diff = flow.convolve(
        raw_diff, structure=t_struct, func=lambda x: np.nanmean(x, 0)
    )
    # filtered_diff = flow.convolve(
    #     filtered_diff, structure=t_struct, func=lambda x: np.nanmax(x, 0)
    # )

    return filtered_diff


# Get a mask which only picks up where the curvature field is positive or negative
def get_curvature_filter(field, sigma=2, threshold=0, direction="negative"):
    smoothed_field = ndi.gaussian_filter(field, (0, sigma, sigma))
    x_diff = np.zeros(field.shape)
    x_diff[:, :, 1:-1] = np.diff(smoothed_field, n=2, axis=2)

    y_diff = np.zeros(field.shape)
    y_diff[:, 1:-1] = np.diff(smoothed_field, n=2, axis=1)

    s_struct = ndi.generate_binary_structure(3, 1)
    s_struct[0] = 0
    s_struct[2] = 0

    if direction == "negative":
        curvature_filter = ndi.binary_opening(
            ndi.binary_fill_holes(
                np.logical_and(x_diff < -threshold, y_diff < -threshold),
                structure=s_struct,
            ),
            structure=s_struct,
        )
    elif direction == "positive":
        curvature_filter = ndi.binary_opening(
            ndi.binary_fill_holes(
                np.logical_and(x_diff > threshold, y_diff > threshold),
                structure=s_struct,
            ),
            structure=s_struct,
        )
    else:
        raise ValueError("Direction must be either positive or negative")
    return curvature_filter


# Detect regions of growth in the the wvd field
def detect_growth_markers(flow, wvd):
    wvd_diff_raw = (
        flow.diff(wvd) / get_time_diff_from_coord(wvd.t)[:, np.newaxis, np.newaxis]
    )

    wvd_diff_smoothed = filtered_tdiff(flow, wvd_diff_raw)

    s_struct = ndi.generate_binary_structure(2, 1)[np.newaxis, ...]
    wvd_diff_filtered = ndi.grey_opening(
        wvd_diff_smoothed, footprint=s_struct
    ) * get_curvature_filter(wvd)

    marker_labels = flow.label(
        ndi.binary_opening(wvd_diff_filtered >= 0.25, structure=s_struct)
    )

    marker_labels = filter_labels_by_length(marker_labels, 3)
    marker_labels = filter_labels_by_mask(marker_labels, wvd_diff_filtered >= 0.5)
    if isinstance(wvd, xr.DataArray):
        marker_labels = filter_labels_by_mask(marker_labels, wvd.data >= -5)
    else:
        marker_labels = filter_labels_by_mask(marker_labels, wvd >= -5)

    if isinstance(wvd, xr.DataArray):
        wvd_diff_raw = xr.DataArray(wvd_diff_raw, wvd.coords, wvd.dims)
        marker_labels = xr.DataArray(marker_labels, wvd.coords, wvd.dims)

    return wvd_diff_smoothed, marker_labels


def nan_gaussian_filter(a, *args, propagate_nan=True, **kwargs):
    wh_nan = np.isnan(a)

    a0 = a.copy()
    a0[wh_nan] = 0

    c = np.ones_like(a)
    c[wh_nan] = 0

    a0_gaussian = ndi.gaussian_filter(a0, *args, **kwargs)
    c_gaussian = ndi.gaussian_filter(c, *args, **kwargs)
    c_gaussian[c_gaussian == 0] = np.nan

    result = a0_gaussian / c_gaussian

    if propagate_nan:
        result[wh_nan] = np.nan

    return result


def get_peak_filter(field, sigma=2, min_distance=10, direction="negative"):
    smoothed_field = ndi.gaussian_filter(field, (0, sigma, sigma))
    peak_filter = np.zeros(field.shape, dtype=np.int32)
    if direction == "negative":
        for i in range(field.shape[0]):
            peak_locs = peak_local_max(smoothed_field[i], min_distance=10).T
            peak_filter[i][(peak_locs[0], peak_locs[1])] = 1
            peak_filter[i] = (
                ndi.distance_transform_edt(np.logical_not(peak_filter[i])) < 5
            )
    elif direction == "positive":
        for i in range(field.shape[0]):
            peak_locs = peak_local_max(-smoothed_field[i], min_distance=10).T
            peak_filter[i][(peak_locs[0], peak_locs[1])] = 1
            peak_filter[i] = (
                ndi.distance_transform_edt(np.logical_not(peak_filter[i])) < 5
            )
    else:
        raise ValueError("Direction must be either positive or negative")
    return peak_filter


def get_growth_rate(flow, field, method: str = "linear"):
    """
    Detect the growth/cooling rate of a field
    """
    growth_rate = (
        flow.diff(field, method=method)
        / get_time_diff_from_coord(field.t)[:, np.newaxis, np.newaxis]
    )

    s_struct = ndi.generate_binary_structure(3, 1)
    s_struct[0] = 0
    s_struct[2] = 0

    t_struct = np.zeros([3, 3, 3])
    t_struct[:, 1, 1] = 1

    growth_rate = flow.convolve(
        growth_rate,
        structure=s_struct,
        func=lambda x: np.nanmean(x, 0),
        method=method,
    )
    # growth_rate = flow.convolve(
    #     growth_rate,
    #     structure=ndi.generate_binary_structure(3, 1),
    #     func=lambda x: np.nanmax(x, 0),
    #     method=method,
    # )

    return growth_rate


def detect_growth_markers_multichannel(
    flow,
    wvd,
    bt,
    t_sigma=1,
    overlap=0.5,
    subsegment_shrink=0,
    min_length=4,
    lower_threshold=0.25,
    upper_threshold=0.5,
):
    wvd_diff_smoothed = filtered_tdiff(
        flow,
        flow.diff(wvd) / get_time_diff_from_coord(wvd.t)[:, np.newaxis, np.newaxis],
    )
    bt_diff_smoothed = filtered_tdiff(
        flow, flow.diff(bt) / get_time_diff_from_coord(bt.t)[:, np.newaxis, np.newaxis]
    )

    markers = np.logical_or(
        (wvd_diff_smoothed * get_curvature_filter(wvd)) >= lower_threshold,
        (bt_diff_smoothed * get_curvature_filter(bt, direction="positive"))
        <= -lower_threshold,
    )
    markers = flow.label(
        ndi.binary_opening(
            markers, structure=ndi.generate_binary_structure(2, 1)[np.newaxis, ...]
        ),
        overlap=overlap,
        subsegment_shrink=subsegment_shrink,
    )

    # markers = filter_labels_by_length(markers, min_length)
    if np.count_nonzero(markers) > 0:
        markers = filter_labels_by_length_and_multimask_legacy(
            markers,
            [
                wvd_diff_smoothed >= upper_threshold,
                bt_diff_smoothed <= -upper_threshold,
                wvd.data > -5,
            ],
            min_length,
        )
    else:
        warnings.warn("No regions detected in labeled array", RuntimeWarning)

    if isinstance(wvd, xr.DataArray):
        wvd_diff_smoothed = xr.DataArray(wvd_diff_smoothed, wvd.coords, wvd.dims)
        bt_diff_smoothed = xr.DataArray(bt_diff_smoothed, bt.coords, bt.dims)
        markers = xr.DataArray(markers, wvd.coords, wvd.dims)

    return wvd_diff_smoothed, bt_diff_smoothed, markers


def edge_watershed(
    flow,
    field,
    markers,
    upper_threshold,
    lower_threshold,
    structure=ndi.generate_binary_structure(3, 1),
    erode_distance=5,
    verbose=False,
):
    if isinstance(field, xr.DataArray):
        field = np.maximum(np.minimum(field.data, upper_threshold), lower_threshold)
    else:
        field = np.maximum(np.minimum(field, upper_threshold), lower_threshold)

    if isinstance(markers, xr.DataArray):
        markers = markers.data

    field[markers != 0] = upper_threshold

    s_struct = np.ones([1, 3, 3])
    mask = ndi.binary_erosion(
        field == lower_threshold,
        structure=s_struct,
        iterations=erode_distance,
        border_value=1,
    )

    # edges = flow.sobel(field, direction='uphill', method='nearest')
    edges = flow.sobel(field, method="nearest")

    watershed = flow.watershed(
        edges, markers, mask=mask, structure=structure, debug_mode=verbose
    )

    s_struct = ndi.generate_binary_structure(2, 1)[np.newaxis]
    watershed = watershed * ndi.binary_opening(watershed != 0, structure=s_struct)

    if isinstance(field, xr.DataArray):
        watershed = xr.DataArray(watershed, field.coords, field.dims)

    return watershed


def get_combined_filters(flow, bt, wvd, swd, use_wvd=True):
    """
    Get combined cloud top filter from bt, wvd and swd fields
    """
    t_struct = np.zeros([3, 3, 3], dtype=bool)
    t_struct[:, 1, 1] = True
    s_struct = ndi.generate_binary_structure(3, 1)
    s_struct[0] = 0
    s_struct[2] = 0

    bt_curvature_filter = get_curvature_filter(bt, direction="positive")
    bt_peak_filter = get_peak_filter(bt, sigma=0.5, direction="positive")
    bt_filter = flow.convolve(
        np.logical_or(bt_curvature_filter, bt_peak_filter).astype(int),
        structure=t_struct,
        method="nearest",
        fill_value=False,
        dtype=np.int32,
        func=partial(np.any, axis=0),
    )

    if use_wvd:
        wvd_curvature_filter = get_curvature_filter(wvd, direction="negative")
        wvd_peak_filter = get_peak_filter(wvd, sigma=0.5, direction="negative")
        wvd_filter = flow.convolve(
            np.logical_or(wvd_curvature_filter, wvd_peak_filter).astype(int),
            structure=t_struct,
            method="nearest",
            fill_value=False,
            dtype=np.int32,
            func=partial(np.any, axis=0),
        )
        combined_filter = ndi.binary_opening(
            ndi.binary_fill_holes(
                np.logical_or(bt_filter, wvd_filter),
                structure=s_struct,
            ),
            structure=s_struct,
        )

    else:
        combined_filter = ndi.binary_opening(
            ndi.binary_fill_holes(
                bt_filter,
                structure=s_struct,
            ),
            structure=s_struct,
        )

    swd_filter = 1 - linearise_field(swd.to_numpy(), 2.5, 7.5)

    combined_filter = combined_filter.astype(float) * swd_filter

    return combined_filter


def detect_cores(
    flow,
    bt,
    wvd,
    swd,
    wvd_threshold=0.25,
    bt_threshold=0.5,
    overlap=0.5,
    absolute_overlap=4,
    subsegment_shrink=0.0,
    min_length=3,
    use_wvd=True,
):
    """
    Detect growing cores using BT, WVD and SWD channels
    """
    combined_filter = get_combined_filters(flow, bt, wvd, swd, use_wvd=use_wvd)

    s_struct = ndi.generate_binary_structure(3, 1)
    s_struct *= np.array([0, 1, 0])[:, np.newaxis, np.newaxis].astype(bool)

    bt_growth = get_growth_rate(flow, -bt, method="cubic")
    bt_markers = (bt_growth * combined_filter) > bt_threshold

    if use_wvd:
        wvd_growth = get_growth_rate(flow, wvd, method="cubic")
        wvd_markers = (wvd_growth * combined_filter) > wvd_threshold

        combined_markers = ndi.binary_opening(
            np.logical_or.reduce([wvd_markers, bt_markers]), structure=s_struct
        )
        print("WVD growth above threshold: area =", np.sum(wvd_markers))
    else:
        combined_markers = ndi.binary_opening(bt_markers, structure=s_struct)

    print("BT growth above threshold: area =", np.sum(bt_markers))
    print("Detected markers: area =", np.sum(combined_markers))

    core_labels = flow.label(
        combined_markers,
        overlap=overlap,
        absolute_overlap=absolute_overlap,
        subsegment_shrink=subsegment_shrink,
    )

    print("Initial core count:", np.max(core_labels))

    # Filter labels by length and wvd growth threshold
    core_label_lengths = find_object_lengths(core_labels)

    print(
        "Core labels meeting length threshold:", np.sum(core_label_lengths > min_length)
    )

    core_label_wvd_mask = mask_labels(core_labels, wvd > -5)

    print("Core labels meeting WVD threshold:", np.sum(core_label_wvd_mask))

    combined_mask = np.logical_and(core_label_lengths > min_length, core_label_wvd_mask)

    core_labels = remap_labels(core_labels, combined_mask)

    core_step_labels = slice_labels(core_labels)

    mode = lambda x: stats.mode(x, keepdims=False)[0]
    core_step_core_index = labeled_comprehension(
        core_labels, core_step_labels, mode, default=0
    )

    core_step_bt_mean = labeled_comprehension(
        bt, core_step_labels, np.nanmean, default=np.nan
    )

    core_step_t = labeled_comprehension(
        bt.t.data[:, np.newaxis, np.newaxis], core_step_labels, np.nanmin, default=0
    )

    def bt_diff_func(step_bt, pos):
        step_t = core_step_t[pos]
        args = np.argsort(step_t)

        step_bt = step_bt[args]
        step_t = step_t[args]

        step_bt_diff = (step_bt[:-min_length] - step_bt[min_length:]) / (
            (step_t[min_length:] - step_t[:-min_length])
            .astype("timedelta64[s]")
            .astype("int")
            / 60
        )

        return np.nanmax(step_bt_diff)

    core_bt_diff_mean = labeled_comprehension(
        core_step_bt_mean,
        core_step_core_index,
        bt_diff_func,
        default=0,
        pass_positions=True,
    )

    wh_valid_core = core_bt_diff_mean >= 0.5

    print("Core labels meeting cooling rate threshold:", np.sum(wh_valid_core))

    core_labels = remap_labels(core_labels, wh_valid_core)

    return core_labels


def get_anvil_markers(
    flow,
    field,
    threshold=-5,
    overlap=0.5,
    absolute_overlap=5,
    subsegment_shrink=0,
    min_length=3,
):
    structure = ndi.generate_binary_structure(3, 1)
    s_struct = structure * np.array([0, 1, 0])[:, np.newaxis, np.newaxis].astype(bool)
    mask = ndi.binary_opening(field > threshold, structure=s_struct)
    marker_labels = flow.label(
        mask,
        overlap=overlap,
        absolute_overlap=absolute_overlap,
        subsegment_shrink=subsegment_shrink,
    )
    marker_label_lengths = find_object_lengths(marker_labels)
    marker_labels = remap_labels(marker_labels, marker_label_lengths > min_length)
    return marker_labels


def detect_anvils(
    flow: Flow,
    field: np.ndarray[float],
    markers=None,
    upper_threshold=-5,
    lower_threshold=-15,
    erode_distance=1,
    min_length=3,
):
    if isinstance(field, xr.DataArray):
        field = linearise_field(field.to_numpy(), lower_threshold, upper_threshold)
    else:
        field = linearise_field(field, lower_threshold, upper_threshold)
    structure = ndi.generate_binary_structure(3, 1)
    s_struct = structure * np.array([0, 1, 0])[:, np.newaxis, np.newaxis].astype(bool)
    if markers is None:
        markers = field >= 1
    # else:
    #     field[markers!=0] = 1
    markers *= ndi.binary_erosion(markers != 0, structure=s_struct).astype(int)
    mask = ndi.binary_erosion(
        field <= 0,
        structure=np.ones([3, 3, 3]),
        iterations=erode_distance,
        border_value=1,
    )
    # edges = flow.sobel(field, direction="uphill", method="cubic")
    # # edges[markers != 0] = 0
    # edges[edges > 0] += 1
    anvil_labels = flow.watershed(
        get_combined_edge_field(flow, field),
        markers,
        mask=mask,
        structure=ndi.generate_binary_structure(3, 1),
    )
    anvil_labels *= ndi.binary_opening(anvil_labels != 0, structure=s_struct).astype(
        int
    )

    anvil_labels[markers != 0] = markers[markers != 0]

    marker_label_lengths = find_object_lengths(anvil_labels)
    marker_label_threshold = mask_labels(anvil_labels, markers != 0)

    anvil_labels = remap_labels(
        anvil_labels,
        np.logical_and(marker_label_lengths > min_length, marker_label_threshold),
    )

    return anvil_labels


def get_combined_edge_field(
    flow: Flow, field: np.ndarray[float], **kwargs
) -> np.ndarray[float]:
    """Apply sobel edge filter to a field, and then subtract the field from
    regions where the edges == 0 to use with watershed segmentation

    Parameters
    ----------
    flow : Flow
        flow object
    field : np.ndarray[float]
        the field to calculate edges from

    Returns
    -------
    np.ndarray[float]
        combined edges field for segmentation
    """
    edges = flow.sobel(field, direction="uphill", method="cubic")
    edges[edges > 0] += 1
    edges = edges - field
    edges[np.isnan(field)] = np.inf
    return edges


def relabel_anvils(
    flow: Flow,
    anvil_labels: np.ndarray[int],
    markers: np.ndarray[bool] = None,
    overlap: float = 0.5,
    absolute_overlap: int = 5,
    min_length: int = 3,
):
    anvil_labels = flow.link_overlap(
        make_step_labels(anvil_labels),
        overlap=overlap,
        absolute_overlap=absolute_overlap,
    )

    marker_label_lengths = find_object_lengths(anvil_labels)
    if markers is not None:
        marker_label_threshold = mask_labels(anvil_labels, markers != 0)
        anvil_labels = remap_labels(
            anvil_labels,
            np.logical_and(marker_label_lengths > min_length, marker_label_threshold),
        )
    else:
        anvil_labels = remap_labels(
            anvil_labels,
            marker_label_lengths > min_length,
        )

    return anvil_labels
