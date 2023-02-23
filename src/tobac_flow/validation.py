import numpy as np
import xarray as xr
from scipy import ndimage as ndi


def get_min_dist_for_objects(distance_array, labels):
    if isinstance(labels, xr.DataArray):
        bins = np.cumsum(np.bincount(labels.data.ravel()))
        args = np.argsort(labels.data.ravel())
    else:
        bins = np.cumsum(np.bincount(labels.ravel()))
        args = np.argsort(labels.ravel())
    dists = np.full(bins.size - 1, np.nan)
    mask_count = np.full(bins.size - 1, np.nan)
    for i in range(bins.size - 1):
        if bins[i + 1] > bins[i]:
            mask_count[i] = bins[i + 1] - bins[i]
            if isinstance(distance_array, xr.DataArray):
                dists[i] = np.min(
                    distance_array.data.ravel()[args[bins[i] : bins[i + 1]]]
                )
            else:
                dists[i] = np.min(distance_array.ravel()[args[bins[i] : bins[i + 1]]])
    return dists, mask_count


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
