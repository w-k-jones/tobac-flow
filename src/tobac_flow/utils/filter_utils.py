"""
Utilities for filtering various detected features
"""
from datetime import timedelta
import numpy as np
import xarray as xr


def remove_orphan_coords(dataset: xr.Dataset) -> xr.Dataset:
    """
    Remove cores/anvils which don't have core/anvil steps and vice versa
    """
    wh_core = np.isin(dataset.core, dataset.core_step_core_index)
    wh_anvil = np.logical_and(
        np.isin(dataset.anvil, dataset.thick_anvil_step_anvil_index),
        np.isin(dataset.anvil, dataset.thin_anvil_step_anvil_index),
    )
    dataset = dataset.sel(
        core=dataset.core.data[wh_core], anvil=dataset.anvil.data[wh_anvil]
    )
    wh_core_step = np.isin(dataset.core_step_core_index, dataset.core)
    wh_thick_anvil_step = np.isin(dataset.thick_anvil_step_anvil_index, dataset.anvil)
    wh_thin_anvil_step = np.isin(dataset.thin_anvil_step_anvil_index, dataset.anvil)
    dataset = dataset.sel(
        core_step=dataset.core_step.data[wh_core_step],
        thick_anvil_step=dataset.thick_anvil_step[wh_thick_anvil_step],
        thin_anvil_step=dataset.thin_anvil_step[wh_thin_anvil_step],
    )
    return dataset


def filter_cores(dataset: xr.Dataset) -> xr.Dataset:
    def start_end_diff(x, *args, **kwargs):
        return x[0] - x[-1]

    if "core_step_BT_mean" in dataset.data_vars:
        core_bt_change = dataset.core_step_BT_mean.groupby(
            dataset.core_step_core_index
        ).reduce(start_end_diff)
        core_invalid_bt = core_bt_change.data < 8
    elif "core_step_ctt_mean" in dataset.data_vars:
        core_bt_change = dataset.core_step_ctt_mean.groupby(
            dataset.core_step_core_index
        ).reduce(start_end_diff)
        core_invalid_bt = core_bt_change.data < 8
    else:
        core_invalid_bt = xr.zeros_like(dataset.core_edge_label_flag)

    def max_t_diff(x, *args, **kwargs):
        if len(x) > 1:
            return np.max(np.diff(x))
        else:
            return np.timedelta64(timedelta(minutes=0))

    core_max_time_diff = dataset.core_step_t.groupby(
        dataset.core_step_core_index
    ).reduce(max_t_diff)

    core_invalid_time_diff = core_max_time_diff > np.timedelta64(
        timedelta(minutes=15, seconds=1)
    )

    def end_start_diff(x, *args, **kwargs):
        return x[-1] - x[0]

    core_lifetime = dataset.core_step_t.groupby(dataset.core_step_core_index).reduce(
        end_start_diff
    )

    core_invalid_lifetime = core_lifetime < np.timedelta64(
        timedelta(minutes=14, seconds=59)
    )

    def any_nan(x, *args, **kwargs):
        return np.any(np.isnan(x))

    core_any_nan_step = dataset.core_step_cot_mean.groupby(
        dataset.core_step_core_index
    ).reduce(any_nan)

    if "core_step_BT_mean" in dataset.data_vars:
        core_any_nan_step = dataset.core_step_BT_mean.groupby(
            dataset.core_step_core_index
        ).reduce(any_nan)
    elif "core_step_ctt_mean" in dataset.data_vars:
        core_any_nan_step = dataset.core_step_ctt_mean.groupby(
            dataset.core_step_core_index
        ).reduce(any_nan)
    else:
        core_any_nan_step = xr.zeros_like(dataset.core_edge_label_flag)

    wh_core_invalid = np.logical_or.reduce(
        [
            dataset.core_start_label_flag.data,
            dataset.core_end_label_flag.data,
            dataset.core_edge_label_flag.data,
            core_invalid_bt,
            core_invalid_time_diff,
            core_invalid_lifetime,
            core_any_nan_step,
        ]
    )

    dataset = dataset.sel(core=dataset.core.data[np.logical_not(wh_core_invalid)])

    wh_core_step = np.isin(dataset.core_step_core_index, dataset.core)
    dataset = dataset.sel(core_step=dataset.core_step.data[wh_core_step])

    return dataset


def filter_anvils(dataset: xr.Dataset) -> xr.Dataset:
    # Filter no cores associated
    anvil_no_core = np.logical_not(np.isin(dataset.anvil, dataset.core_anvil_index))
    dataset = dataset.sel(anvil=dataset.anvil.data[np.logical_not(anvil_no_core)])
    wh_thick_anvil_step = np.isin(dataset.thick_anvil_step_anvil_index, dataset.anvil)
    wh_thin_anvil_step = np.isin(dataset.thin_anvil_step_anvil_index, dataset.anvil)
    dataset = dataset.sel(
        thick_anvil_step=dataset.thick_anvil_step[wh_thick_anvil_step],
        thin_anvil_step=dataset.thin_anvil_step[wh_thin_anvil_step],
    )

    # Filter out any NaN values
    def any_nan(x, *args, **kwargs):
        return np.any(np.isnan(x))

    if "thin_anvil_step_BT_mean" in dataset.data_vars:
        thin_anvil_any_nan_step = dataset.thin_anvil_step_BT_mean.groupby(
            dataset.thin_anvil_step_anvil_index
        ).reduce(any_nan)
    elif "thin_anvil_step_ctt_mean" in dataset.data_vars:
        thin_anvil_any_nan_step = dataset.thin_anvil_step_ctt_mean.groupby(
            dataset.thin_anvil_step_anvil_index
        ).reduce(any_nan)
    else:
        thin_anvil_any_nan_step = xr.zeros_like(dataset.thin_anvil_edge_label_flag)

    # Filter lifetimes less than 15 minutes
    def start_end_diff(x, *args, **kwargs):
        return x[-1] - x[0]

    anvil_lifetime = dataset.thick_anvil_step_t.groupby(
        dataset.thick_anvil_step_anvil_index
    ).reduce(start_end_diff)

    anvil_invalid_lifetime = anvil_lifetime < np.timedelta64(
        timedelta(minutes=14, seconds=59)
    )

    # Filter time gaps greater than 15 minutes
    def max_t_diff(x, *args, **kwargs):
        if len(x) > 1:
            return np.max(np.diff(x))
        else:
            return np.timedelta64(timedelta(minutes=0))

    thick_anvil_max_time_diff = dataset.thick_anvil_step_t.groupby(
        dataset.thick_anvil_step_anvil_index
    ).reduce(max_t_diff)

    thick_anvil_invalid_time_diff = thick_anvil_max_time_diff > np.timedelta64(
        timedelta(minutes=15, seconds=1)
    )

    # Filter max core area greater than max anvil area
    anvil_max_area = xr.DataArray(
        dataset.thick_anvil_step_area.groupby(dataset.thick_anvil_step_anvil_index)
        .max()
        .data,
        {"anvil": dataset.anvil},
    )
    anvil_max_core_area = xr.DataArray(
        dataset.core_max_area.groupby(dataset.core_anvil_index).max().data,
        {"anvil": dataset.anvil},
    )
    wh_anvil_area_invalid = anvil_max_area <= anvil_max_core_area

    # Filter anvil starts before first core starts
    anvil_start_t = xr.DataArray(
        dataset.thick_anvil_step_t.groupby(dataset.thick_anvil_step_anvil_index)
        .min()
        .data,
        {"anvil": dataset.anvil},
    )
    anvil_core_start_t = xr.DataArray(
        dataset.core_start_t.groupby(dataset.core_anvil_index).min().data,
        {"anvil": dataset.anvil},
    )
    wh_anvil_start_t_invalid = anvil_start_t <= anvil_core_start_t

    # Filter anvil ends before last core ends
    anvil_end_t = xr.DataArray(
        dataset.thick_anvil_step_t.groupby(dataset.thick_anvil_step_anvil_index)
        .max()
        .data,
        {"anvil": dataset.anvil},
    )
    anvil_core_end_t = xr.DataArray(
        dataset.core_end_t.groupby(dataset.core_anvil_index).max().data,
        {"anvil": dataset.anvil},
    )
    wh_anvil_end_t_invalid = anvil_end_t <= anvil_core_end_t

    wh_anvil_invalid = np.logical_or.reduce(
        [
            thin_anvil_any_nan_step.data,
            anvil_invalid_lifetime.data,
            thick_anvil_invalid_time_diff.data,
            wh_anvil_area_invalid.data,
            wh_anvil_start_t_invalid.data,
            wh_anvil_end_t_invalid.data,
        ]
    )

    dataset = dataset.sel(anvil=dataset.anvil.data[np.logical_not(wh_anvil_invalid)])

    wh_thick_anvil_step = np.isin(dataset.thick_anvil_step_anvil_index, dataset.anvil)
    wh_thin_anvil_step = np.isin(dataset.thin_anvil_step_anvil_index, dataset.anvil)
    dataset = dataset.sel(
        thick_anvil_step=dataset.thick_anvil_step[wh_thick_anvil_step],
        thin_anvil_step=dataset.thin_anvil_step[wh_thin_anvil_step],
    )

    return dataset
