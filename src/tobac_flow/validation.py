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
        labels, distance_array, np.nanmin, index=index, default=np.nan
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
