"""
Utilities for calculating various statistical properties, inclduing weighted
    statistics and statistics over labels
"""
import numpy as np
import xarray as xr
from scipy import stats


def find_overlap_mode(x: np.ndarray[float], background: float = 0) -> float:
    """
    Calculate the mode value of an array where the array does not equal the
        background value
    """
    if np.any(x):
        overlap_mode = stats.mode(x[x != background], keepdims=False)[0]
    else:
        overlap_mode = background
    return overlap_mode


def n_unique_along_axis(a: np.ndarray[float], axis: int = 0) -> np.ndarray[int]:
    """
    Calculate the number of unique values along an axis of an array
    """
    b = np.sort(np.moveaxis(a, axis, 0), axis=0)
    return (b[1:] != b[:-1]).sum(axis=0) + (
        np.count_nonzero(a, axis=axis) == a.shape[axis]
    ).astype(int)


def weighted_average_and_std(
    data: np.ndarray[float], weights: np.ndarray[float], unbiased: bool = True
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Calculate both the weighted average and weighted standard distribution of an
        array, applying Bessel's bias correction
    """
    average = np.average(data, weights=weights)
    variance = np.average((data - average) ** 2, weights=weights)
    if unbiased:
        # Bessel's correction for reliability weights
        correction = 1 - (np.sum(weights**2) / np.sum(weights) ** 2)
        if correction >= 0:
            variance /= correction
            std = variance**0.5
        else:
            std = np.nan
    return average, std


def weighted_stats(
    data: np.ndarray[float],
    weights: np.ndarray[float],
    ignore_nan: bool = True,
    default=np.nan,
) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Calculate the weighted average, standard deviation, minimum and maximum of
        an array
    """
    if ignore_nan:
        wh = np.isfinite(data)
        data = data[wh]
        weights = weights[wh]
    if data.size > 0 and np.sum(weights) > 0:
        average, std = weighted_average_and_std(data, weights)
        minimum = np.min(data)
        maximum = np.max(data)
    else:
        average, std, minimum, maximum = default, default, default, default
    return average, std, minimum, maximum


def weighted_average_uncertainty(
    errors: np.ndarray[float], weights: np.ndarray[float]
) -> np.ndarray[float]:
    """
    Calculate the propogated uncertainty on a weighted average
    """
    if errors.size > 0 and np.sum(weights) > 0:
        uncertainty = np.sum(weights**2 * errors**2) ** 0.5 / np.sum(weights)
    else:
        uncertainty = np.nan
    return uncertainty


def weighted_uncertainties(
    data: np.ndarray[float],
    errors: np.ndarray[float],
    weights: np.ndarray[float],
    std: float,
    ignore_nan: bool = True,
) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Calculate the uncertainty, combined mean error, and find the uncertainty in
        the minimum and maximum values of a weighted distribution
    """
    if ignore_nan:
        wh = np.isfinite(data)
        data = data[wh]
        errors = errors[wh]
        weights = weights[wh]
    if data.size > 0 and np.sum(weights) > 0:
        uncertainty = weighted_average_uncertainty(errors, weights)
        combined_error = ((std / data.size**0.5) ** 2 + uncertainty**2) ** 0.5
        min_error = errors[np.argmin(data)]
        max_error = errors[np.argmax(data)]
    else:
        uncertainty, combined_error, min_error, max_error = (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
    return uncertainty, combined_error, min_error, max_error


def weighted_stats_and_uncertainties(
    data: np.ndarray[float],
    errors: np.ndarray[float],
    weights: np.ndarray[float],
    ignore_nan: bool = True,
) -> tuple[
    np.ndarray[float],
    np.ndarray[float],
    np.ndarray[float],
    np.ndarray[float],
    np.ndarray[float],
    np.ndarray[float],
    np.ndarray[float],
    np.ndarray[float],
]:
    """
    Calculate the statistics and their associated uncertainties for a weighted
        distribution
    """
    average, std, minimum, maximum = weighted_stats(
        data, weights, ignore_nan=ignore_nan
    )
    uncertainty, combined_error, min_error, max_error = weighted_uncertainties(
        data, errors, weights, std, ignore_nan=ignore_nan
    )
    return (
        average,
        std,
        minimum,
        maximum,
        uncertainty,
        combined_error,
        min_error,
        max_error,
    )


def get_weighted_proportions(data, weights, flag_values):
    wh_flags = np.expand_dims(data, -1) == flag_values
    weighted_flags = wh_flags.astype(float) * np.expand_dims(weights, -1)
    weights_sum = np.nansum(weights)
    if weights_sum > 0:
        proportions = (
            np.nansum(weighted_flags.reshape([-1, len(list(flag_values))]), 0)
            / weights_sum
        )
    else:
        proportions = np.asarray([np.nan] * len(flag_values))
    return proportions


def calc_combined_mean(step_mean, step_area):
    wh_finite = np.logical_and(np.isfinite(step_mean), np.isfinite(step_area))
    if np.any(wh_finite):
        result = np.sum(step_mean[wh_finite] * step_area[wh_finite]) / np.sum(
            step_area[wh_finite]
        )
    else:
        result = np.nan
    return result


def calc_combined_std(step_std, step_mean, step_area):
    combined_mean = calc_combined_mean(step_mean, step_area)
    wh_finite = np.logical_and.reduce(
        [np.isfinite(step_std), np.isfinite(step_mean), np.isfinite(step_area)]
    )
    if np.any(wh_finite):
        result = (
            (
                np.sum(step_area[wh_finite] * step_std[wh_finite])
                + np.sum(
                    step_area[wh_finite] * (step_mean[wh_finite] - combined_mean) ** 2
                )
            )
            / np.sum(step_area[wh_finite])
        ) ** 0.5
    else:
        result = np.nan
    return result


def combined_mean_groupby(means, area, groups, coord):
    return xr.DataArray(
        [
            calc_combined_mean(means_group[1].data, area_group[1].data)
            for means_group, area_group in zip(
                means.groupby(groups), area.groupby(groups)
            )
        ],
        {coord.name: coord},
    )


def combined_std_groupby(stds, means, area, groups, coord):
    return xr.DataArray(
        [
            calc_combined_std(
                stds_group[1].data, means_group[1].data, area_group[1].data
            )
            for stds_group, means_group, area_group in zip(
                stds.groupby(groups), means.groupby(groups), area.groupby(groups)
            )
        ],
        {coord.name: coord},
    )


def weighted_average_uncertainty_groupby(field, area, groups, coord):
    return xr.DataArray(
        [
            weighted_average_uncertainty(field_group[1], area_group[1])
            for field_group, area_group in zip(
                field.groupby(groups), area.groupby(groups)
            )
        ],
        {coord.name: coord},
    )


def weighted_average_groupby(field, area, groups, coord):
    return xr.DataArray(
        [
            np.average(field_group[1], weights=area_group[1])
            for field_group, area_group in zip(
                field.groupby(groups), area.groupby(groups)
            )
        ],
        {coord.name: coord},
    )


def argmax_groupby(field, find_max, groups, coord):
    return xr.DataArray(
        [
            field_group[1].data[np.argmax(max_group[1].data)]
            for field_group, max_group in zip(
                field.groupby(groups), find_max.groupby(groups)
            )
        ],
        {coord.name: coord},
    )


def argmin_groupby(field, find_min, groups, coord):
    return xr.DataArray(
        [
            field_group[1].data[np.argmin(min_group[1].data)]
            for field_group, min_group in zip(
                field.groupby(groups), find_min.groupby(groups)
            )
        ],
        {coord.name: coord},
    )


def counts_groupby(groups, coord):
    return xr.DataArray(
        xr.ones_like(groups).groupby(groups).sum().data,
        {coord.name: coord},
    )
