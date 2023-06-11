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


def get_marker_distance_ellipse(markers, time_margin, margin):
    """
    Get distances to markers and nearest marker for each point in an ellipse
    with semi-major axis of length time_margin in the leading dimension and
    margin in the other dimensions
    """
    distances, indices = ndi.distance_transform_edt(
        markers == 0, return_indices=True, sampling=(margin / time_margin, 1, 1)
    )
    closest_marker = markers[indices[0], indices[1], indices[2]]
    return distances, closest_marker


def get_marker_distance_cylinder(markers, time_margin):
    """
    Get distances to markers and nearest marker for each point in a cylinder
    with height time_margin in the leading dimension and radius of margin in
    the other dimensions
    """
    distances = np.full(markers.shape, np.inf, dtype=float)
    closest_markers = np.full(markers.shape, 0, dtype=int)

    for i in range(markers.shape[0]):
        if np.any(markers[i]):
            step_distances, indices = ndi.distance_transform_edt(
                markers[i] == 0, return_indices=True
            )
            distances[i] = step_distances
            closest_markers[i] = markers[i][indices[0], indices[1]]

    distances2 = np.full(markers.shape, np.inf, dtype=float)
    closest_markers2 = np.full(markers.shape, 0, dtype=int)

    for i in range(markers.shape[0]):
        i_slice = slice(
            np.maximum(i - time_margin, 0),
            np.minimum(i + time_margin + 1, markers.shape[0]),
        )
        argmin = np.expand_dims(np.nanargmin(distances[i_slice], axis=0), axis=0)
        distances2[i] = np.take_along_axis(distances[i_slice], argmin, axis=0)
        closest_markers2[i] = np.take_along_axis(
            closest_markers[i_slice], argmin, axis=0
        )

    return distances2, closest_markers2


def validate_markers(
    labels,
    glm_grid,
    glm_distance,
    edge_filter,
    n_glm_in_margin,
    coord=None,
    margin=10,
    time_margin=3,
):
    """
    Get validation results for a set of markers
    """
    marker_distance, closest_marker = get_marker_distance_cylinder(labels, time_margin)
    flash_distance_to_marker = np.repeat(
        marker_distance.ravel(),
        glm_grid.astype(int).ravel(),
    )
    flash_closest_marker = np.repeat(
        closest_marker.ravel(),
        glm_grid.astype(int).ravel(),
    )
    if n_glm_in_margin > 0:
        pod = np.nansum(flash_distance_to_marker <= margin) / n_glm_in_margin
    else:
        pod = np.nan

    margin_flag = apply_func_to_labels(
        labels, edge_filter, func=np.nanmin, index=coord, default=False
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

    return (
        flash_distance_to_marker,
        flash_closest_marker,
        marker_distance_to_flash,
        pod,
        far,
        n_marker_in_margin,
        margin_flag,
    )
