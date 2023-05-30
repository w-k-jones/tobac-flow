import numpy as np
import xarray as xr
from scipy import ndimage as ndi
from tobac_flow.utils import apply_func_to_labels


def get_min_dist_for_objects(distance_array, labels, index=None):
    if isinstance(labels, xr.DataArray):
        labels = labels.to_numpy()
    if isinstance(distance_array, xr.DataArray):
        distance_array = distance_array.to_numpy()

    return apply_func_to_labels(
        labels, distance_array, func=np.nanmin, index=index, default=np.nan
    )


def get_marker_distance(labels, time_range=1):
    marker_distance = np.zeros(labels.shape)
    for i in range(marker_distance.shape[0]):
        if np.any(labels[i] != 0):
            marker_distance[i] = ndi.morphology.distance_transform_edt(labels[i] == 0)
        else:
            marker_distance[i] = np.inf

    for i in range(1, time_range + 1):
        marker_distance[i:] = np.fmin(marker_distance[:-i], marker_distance[i:])
        marker_distance[:-i] = np.fmin(marker_distance[:-i], marker_distance[i:])

    return marker_distance


def validate_markers(
    labels, glm_grid, glm_distance, edge_filter, coord=None, margin=10, time_margin=3
):
    marker_distance = get_marker_distance(labels, time_range=time_margin)
    flash_distance_to_marker = np.repeat(
        marker_distance.ravel(),
        (glm_grid.data.astype(int) * edge_filter.astype(int)).ravel(),
    )
    n_glm_in_margin = np.nansum(glm_grid.data * edge_filter.astype(int))
    if n_glm_in_margin > 0:
        pod = np.nansum(flash_distance_to_marker <= 10) / n_glm_in_margin
    else:
        pod = np.nan
    margin_flag = apply_func_to_labels(
        labels, edge_filter, func=np.nanmin, index=coord, default=0
    ).astype("bool")
    n_marker_in_margin = np.nansum(margin_flag)
    marker_distance_to_flash = get_min_dist_for_objects(
        glm_distance, labels, index=coord
    )
    if n_marker_in_margin > 0:
        far = (
            np.nansum(marker_distance_to_flash[margin_flag] > margin)
            / n_marker_in_margin
        )
    else:
        far = np.nan

    return flash_distance_to_marker, marker_distance_to_flash, pod, far, n_marker_in_margin, n_glm_in_margin, margin_flag
