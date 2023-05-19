"""
Functions for processing data associated with detected regions
"""
from functools import partial
import numpy as np
import xarray as xr
from tobac_flow.utils import (
    get_new_attrs_cell_method,
    apply_func_to_labels,
    weighted_stats,
    weighted_stats_and_uncertainties,
    get_weighted_proportions,
    combined_mean_groupby,
    combined_std_groupby,
    weighted_average_groupby,
    argmax_groupby,
    argmin_groupby,
    weighted_average_uncertainty_groupby,
    counts_groupby,
)


def get_cre(flux, clear_flux):
    cre_flux = flux - clear_flux
    cre_flux.attrs = flux.attrs.copy()
    cre_flux.attrs["long_name"] += " cloud radiative effect"
    cre_flux.attrs["standard_name"] += "_cloud_radiative_effect"
    cre_flux.attrs["valid_min"] = -cre_flux.attrs["valid_max"]
    cre_flux.name = f"{flux.name}_cre"
    return cre_flux


def add_cre_to_dataset(dataset):
    for var in (
        "toa_swup",
        "toa_lwup",
        "boa_swdn",
        "boa_swup",
        "boa_lwdn",
        "boa_lwup",
    ):
        dataset[f"{var}_cre"] = get_cre(dataset[var], dataset[f"{var}_clr"])

    toa_net = dataset.toa_swdn - (dataset.toa_swup + dataset.toa_lwup)
    toa_net.attrs = {
        "long_name": "top of atmosphere net radiation",
        "standard_name": "toa_net_flux",
        "units": "W m-2",
        "valid_min": -1500.0,
        "valid_max": 1500.0,
    }
    toa_net.name = "toa_net"
    dataset[toa_net.name] = toa_net

    toa_net_cre = -(dataset.toa_swup_cre + dataset.toa_lwup_cre)
    toa_net_cre.attrs = {
        "long_name": "top of atmosphere net cloud radiative effect",
        "standard_name": "toa_net_cloud_radiative_effect",
        "units": "W m-2",
        "valid_min": -1500.0,
        "valid_max": 1500.0,
    }
    toa_net_cre.name = "toa_net_cre"
    dataset[toa_net_cre.name] = toa_net_cre

    boa_net = (
        dataset.boa_swdn + dataset.boa_lwdn - (dataset.boa_swup + dataset.boa_lwup)
    )
    boa_net.attrs = {
        "long_name": "bottom of atmosphere net radiation",
        "standard_name": "boa_net_flux",
        "units": "W m-2",
        "valid_min": -1500.0,
        "valid_max": 1500.0,
    }
    boa_net.name = "boa_net"
    dataset[boa_net.name] = boa_net

    boa_net_cre = (
        dataset.boa_swdn_cre
        + dataset.boa_lwdn_cre
        - (dataset.boa_swup_cre + dataset.boa_lwup_cre)
    )
    boa_net_cre.attrs = {
        "long_name": "bottom of atmosphere net cloud radiative effect",
        "standard_name": "boa_net_cloud_radiative_effect",
        "units": "W m-2",
        "valid_min": -1500.0,
        "valid_max": 1500.0,
    }
    boa_net_cre.name = "boa_net_cre"
    dataset[boa_net_cre.name] = boa_net_cre
    return dataset


def weighted_label_stats(
    labels: xr.DataArray,
    weights: xr.DataArray,
    dataset: xr.Dataset,
    var: str,
    coord: xr.DataArray,
    dim: str,
    dim_name: str = None,
    attrs: dict | None = None,
    uncertainty: bool = False,
) -> tuple[xr.DataArray]:
    if dim_name is None:
        dim_name = dim
    if attrs is None:
        attrs = dataset[var].attrs

    if uncertainty:
        stats = apply_func_to_labels(
            labels.to_numpy(),
            dataset[var].to_numpy(),
            dataset[f"{var}_uncertainty"].to_numpy(),
            weights.to_numpy(),
            func=weighted_stats_and_uncertainties,
            index=coord.to_numpy(),
            default=[np.nan] * 8,
        )
    else:
        stats = apply_func_to_labels(
            labels.to_numpy(),
            dataset[var].to_numpy(),
            weights.to_numpy(),
            func=weighted_stats,
            index=coord.to_numpy(),
            default=[np.nan] * 4,
        )
    mean = xr.DataArray(
        stats[..., 0],
        {dim: coord},
        (dim,),
        name=f"{dim_name}_{var}_mean",
        attrs=get_new_attrs_cell_method(attrs, "average", dim_name),
    )
    std = xr.DataArray(
        stats[..., 1],
        {dim: coord},
        (dim,),
        name=f"{dim_name}_{var}_std",
        attrs=get_new_attrs_cell_method(attrs, "standard distribution", dim_name),
    )
    minimum = xr.DataArray(
        stats[..., 2],
        {dim: coord},
        (dim,),
        name=f"{dim_name}_{var}_min",
        attrs=get_new_attrs_cell_method(attrs, "minimum", dim_name),
    )
    maximum = xr.DataArray(
        stats[..., 3],
        {dim: coord},
        (dim,),
        name=f"{dim_name}_{var}_max",
        attrs=get_new_attrs_cell_method(attrs, "maximum", dim_name),
    )
    if uncertainty:
        mean_uncertainty = xr.DataArray(
            stats[..., 4],
            {dim: coord},
            (dim,),
            name=f"{dim_name}_{var}_mean_uncertainty",
            attrs=get_new_attrs_cell_method(attrs, "uncertainty of average", dim_name),
        )
        combined_error = xr.DataArray(
            stats[..., 5],
            {dim: coord},
            (dim,),
            name=f"{dim_name}_{var}_mean_combined_error",
            attrs=get_new_attrs_cell_method(
                attrs, "combined error of average", dim_name
            ),
        )
        min_error = xr.DataArray(
            stats[..., 6],
            {dim: coord},
            (dim,),
            name=f"{dim_name}_{var}_min_error",
            attrs=get_new_attrs_cell_method(attrs, "uncertainy of minimum", dim_name),
        )
        max_error = xr.DataArray(
            stats[..., 7],
            {dim: coord},
            (dim,),
            name=f"{dim_name}_{var}_max_error",
            attrs=get_new_attrs_cell_method(attrs, "uncertainy of maximum", dim_name),
        )
    if uncertainty:
        return (
            mean,
            std,
            minimum,
            maximum,
            mean_uncertainty,
            combined_error,
            min_error,
            max_error,
        )
    else:
        return mean, std, minimum, maximum


def add_weighted_stats_to_dataset(
    dcc_dataset,
    field_dataset,
    weights,
    var,
    dim,
    dim_name=None,
    index=None,
    labels=None,
):
    if dim_name is None:
        dim_name = dim
    if index is None:
        index = dcc_dataset[dim]
    if labels is None:
        labels = dcc_dataset[f"{dim_name}_label"]

    stats_da = weighted_label_stats(
        labels,
        weights,
        field_dataset,
        var,
        index,
        dim,
        dim_name=dim_name,
        uncertainty=(f"{var}_uncertainty" in field_dataset.data_vars),
    )

    for da in stats_da:
        dcc_dataset[da.name] = da

    return dcc_dataset


def get_weighted_proportions_da(
    flag_da, weights, labels, dim, dim_name=None, index=None
):
    if dim_name is None:
        dim_name = dim
    if index is None:
        index = range(1, int(np.nanmax(labels) + 1))
    flag_values = [int(n) for n in flag_da.flag_values.replace("b", "").split(" ")]
    if ":" in flag_da.flag_meanings:
        flag_meanings = {
            int(flag[0]): flag[1]
            for flag in [
                flag.split(":")
                for flag in flag_da.flag_meanings.split(" ")
                if ":" in flag
            ]
            if int(flag[0]) in flag_values
        }
        flag_values = np.asarray(list(flag_meanings.keys()))
    else:
        flag_meanings = {
            value: meaning
            for value, meaning in zip(flag_values, flag_da.flag_meanings.split(" "))
        }
    new_dim = (dim, flag_da.name)
    new_coord = {dim: index, flag_da.name: flag_values}
    proportions = apply_func_to_labels(
        labels.to_numpy(),
        flag_da.to_numpy(),
        weights.to_numpy(),
        func=partial(get_weighted_proportions, flag_values=flag_values),
        index=index,
        default=np.asarray([np.nan] * len(flag_meanings)),
    )
    proportions = xr.DataArray(
        proportions,
        new_coord,
        new_dim,
        f"{dim_name}_{flag_da.name}_proportion",
        attrs=get_new_attrs_cell_method(flag_da.attrs, "proportion of", dim_name),
    )
    return proportions


def add_weighted_proportions_to_dataset(
    dcc_dataset,
    flag_da,
    weights,
    dim,
    dim_name=None,
    index=None,
    labels=None,
):
    if dim_name is None:
        dim_name = dim
    if index is None:
        index = dcc_dataset[dim]
    if labels is None:
        labels = dcc_dataset[f"{dim_name}_label"]

    proportions_da = get_weighted_proportions_da(
        flag_da, weights, labels, dim, dim_name=dim_name, index=index
    )
    dcc_dataset[proportions_da.name] = proportions_da

    return dcc_dataset


def process_core_properties(dataset):
    # Core start/end positions
    core_start_step = argmin_groupby(
        dataset.core_step,
        dataset.core_step_t,
        dataset.core_step_core_index,
        dataset.core,
    )

    dataset["core_start_x"] = dataset.core_step_x.loc[core_start_step]
    dataset["core_start_y"] = dataset.core_step_y.loc[core_start_step]
    dataset["core_start_lat"] = dataset.core_step_lat.loc[core_start_step]
    dataset["core_start_lon"] = dataset.core_step_lon.loc[core_start_step]
    dataset["core_start_t"] = dataset.core_step_t.loc[core_start_step]

    core_end_step = argmax_groupby(
        dataset.core_step,
        dataset.core_step_t,
        dataset.core_step_core_index,
        dataset.core,
    )

    dataset["core_end_x"] = dataset.core_step_x.loc[core_end_step]
    dataset["core_end_y"] = dataset.core_step_y.loc[core_end_step]
    dataset["core_end_lat"] = dataset.core_step_lat.loc[core_end_step]
    dataset["core_end_lon"] = dataset.core_step_lon.loc[core_end_step]
    dataset["core_end_t"] = dataset.core_step_t.loc[core_end_step]
    dataset["core_lifetime"] = dataset.core_end_t - dataset.core_start_t

    dataset["core_average_x"] = weighted_average_groupby(
        dataset.core_step_x,
        dataset.core_step_area,
        dataset.core_step_core_index,
        dataset.core,
    )

    dataset["core_average_y"] = weighted_average_groupby(
        dataset.core_step_y,
        dataset.core_step_area,
        dataset.core_step_core_index,
        dataset.core,
    )

    dataset["core_average_lat"] = weighted_average_groupby(
        dataset.core_step_lat,
        dataset.core_step_area,
        dataset.core_step_core_index,
        dataset.core,
    )

    dataset["core_average_lon"] = weighted_average_groupby(
        dataset.core_step_lon,
        dataset.core_step_area,
        dataset.core_step_core_index,
        dataset.core,
    )

    dataset["core_total_area"] = xr.DataArray(
        dataset.core_step_area.groupby(dataset.core_step_core_index).sum().data,
        {"core": dataset.core},
    )

    dataset["core_max_area"] = xr.DataArray(
        dataset.core_step_area.groupby(dataset.core_step_core_index).max().data,
        {"core": dataset.core},
    )

    dataset["core_max_area_t"] = argmax_groupby(
        dataset.core_step_t,
        dataset.core_step_area,
        dataset.core_step_core_index,
        dataset.core,
    )

    if "core_step_ctt_mean" in dataset.data_vars:
        dataset["core_min_ctt_t"] = argmin_groupby(
            dataset.core_step_t,
            dataset.core_step_ctt_mean,
            dataset.core_step_core_index,
            dataset.core,
        )

        def calc_max_cooling_rate(step_bt, step_t):
            argsort = np.argsort(step_t)
            step_bt = step_bt[argsort]
            step_t = step_t[argsort]
            if len(step_bt) >= 2:
                step_bt_diff = np.max(
                    (step_bt[:-1] - step_bt[1:])
                    / (
                        (step_t[1:] - step_t[:-1])
                        .astype("timedelta64[s]")
                        .astype("int")
                        / 60
                    )
                )
            else:
                step_bt_diff = (step_bt[0] - step_bt[-1]) / (
                    (step_t[0] - step_t[-1]).astype("timedelta64[s]").astype("int") / 60
                )
            return step_bt_diff

        def cooling_rate_groupby(BT, times, groups, coord):
            return xr.DataArray(
                [
                    calc_max_cooling_rate(BT_group[1].data, time_group[1].data)
                    for BT_group, time_group in zip(
                        BT.groupby(groups), times.groupby(groups)
                    )
                ],
                {coord.name: coord},
            )

        dataset["core_cooling_rate"] = cooling_rate_groupby(
            dataset.core_step_ctt_mean,
            dataset.core_step_t,
            dataset.core_step_core_index,
            dataset.core,
        )

    elif "core_step_BT_mean" in dataset.data_vars:
        dataset["core_min_BT_t"] = argmin_groupby(
            dataset.core_step_t,
            dataset.core_step_BT_mean,
            dataset.core_step_core_index,
            dataset.core,
        )

        def calc_max_cooling_rate(step_bt, step_t):
            argsort = np.argsort(step_t)
            step_bt = step_bt[argsort]
            step_t = step_t[argsort]
            if len(step_bt) >= 4:
                step_bt_diff = np.max(
                    (step_bt[:-3] - step_bt[3:])
                    / (
                        (step_t[3:] - step_t[:-3])
                        .astype("timedelta64[s]")
                        .astype("int")
                        / 60
                    )
                )
            else:
                step_bt_diff = (step_bt[0] - step_bt[-1]) / (
                    (step_t[0] - step_t[-1]).astype("timedelta64[s]").astype("int") / 60
                )
            return step_bt_diff

        def cooling_rate_groupby(BT, times, groups, coord):
            return xr.DataArray(
                [
                    calc_max_cooling_rate(BT_group[1].data, time_group[1].data)
                    for BT_group, time_group in zip(
                        BT.groupby(groups), times.groupby(groups)
                    )
                ],
                {coord.name: coord},
            )

        dataset["core_cooling_rate"] = cooling_rate_groupby(
            dataset.core_step_BT_mean,
            dataset.core_step_t,
            dataset.core_step_core_index,
            dataset.core,
        )

    for var in dataset.data_vars:
        if dataset[var].dims == ("core_step",):
            new_var = "core_" + var[10:]
            if var.endswith("_mean"):
                dataset[new_var] = combined_mean_groupby(
                    dataset[var],
                    dataset.core_step_area,
                    dataset.core_step_core_index,
                    dataset.core,
                )
            elif var.endswith("_std"):
                mean_var = var[:-3] + "mean"
                dataset[new_var] = combined_std_groupby(
                    dataset[var],
                    dataset[mean_var],
                    dataset.core_step_area,
                    dataset.core_step_core_index,
                    dataset.core,
                )
            elif var.endswith("_min"):
                dataset[new_var] = xr.DataArray(
                    dataset[var].groupby(dataset.core_step_core_index).min().data,
                    {"core": dataset.core},
                )
            elif var.endswith("_max"):
                dataset[new_var] = xr.DataArray(
                    dataset[var].groupby(dataset.core_step_core_index).max().data,
                    {"core": dataset.core},
                )
            elif var.endswith("_mean_uncertainty"):
                dataset[new_var] = weighted_average_uncertainty_groupby(
                    dataset[var],
                    dataset.core_step_area,
                    dataset.core_step_core_index,
                    dataset.core,
                )
            elif var.endswith("_mean_combined_error"):
                mean_var = var[:-15]
                std_da = combined_std_groupby(
                    dataset[var],
                    dataset[mean_var],
                    dataset.core_step_area,
                    dataset.core_step_core_index,
                    dataset.core,
                )
                uncertainty_da = weighted_average_uncertainty_groupby(
                    dataset[var],
                    dataset.core_step_area,
                    dataset.core_step_core_index,
                    dataset.core,
                )
                counts_da = counts_groupby(
                    dataset.core_step_core_index,
                    dataset.core,
                )
                combined_error = (
                    (std_da.data / counts_da.data**0.5) ** 2
                    + uncertainty_da.data**2
                ) ** 0.5
                dataset[new_var] = xr.DataArray(
                    combined_error,
                    {"core": dataset.core},
                )
            elif var.endswith("_min_error"):
                min_var = var[:-6]
                dataset[new_var] = argmin_groupby(
                    dataset[var],
                    dataset[min_var],
                    dataset.core_step_core_index,
                    dataset.core,
                )
            elif var.endswith("_max_error"):
                max_var = var[:-6]
                dataset[new_var] = argmax_groupby(
                    dataset[var],
                    dataset[max_var],
                    dataset.core_step_core_index,
                    dataset.core,
                )

    return dataset


def process_thick_anvil_properties(dataset):
    thick_anvil_start_step = argmin_groupby(
        dataset.thick_anvil_step,
        dataset.thick_anvil_step_t,
        dataset.thick_anvil_step_anvil_index,
        dataset.anvil,
    )

    dataset["thick_anvil_start_x"] = dataset.thick_anvil_step_x.loc[
        thick_anvil_start_step
    ]
    dataset["thick_anvil_start_y"] = dataset.thick_anvil_step_y.loc[
        thick_anvil_start_step
    ]
    dataset["thick_anvil_start_lat"] = dataset.thick_anvil_step_lat.loc[
        thick_anvil_start_step
    ]
    dataset["thick_anvil_start_lon"] = dataset.thick_anvil_step_lon.loc[
        thick_anvil_start_step
    ]
    dataset["thick_anvil_start_t"] = dataset.thick_anvil_step_t.loc[
        thick_anvil_start_step
    ]

    thick_anvil_end_step = argmax_groupby(
        dataset.thick_anvil_step,
        dataset.thick_anvil_step_t,
        dataset.thick_anvil_step_anvil_index,
        dataset.anvil,
    )

    dataset["thick_anvil_end_x"] = dataset.thick_anvil_step_x.loc[thick_anvil_end_step]
    dataset["thick_anvil_end_y"] = dataset.thick_anvil_step_y.loc[thick_anvil_end_step]
    dataset["thick_anvil_end_lat"] = dataset.thick_anvil_step_lat.loc[
        thick_anvil_end_step
    ]
    dataset["thick_anvil_end_lon"] = dataset.thick_anvil_step_lon.loc[
        thick_anvil_end_step
    ]
    dataset["thick_anvil_end_t"] = dataset.thick_anvil_step_t.loc[thick_anvil_end_step]
    dataset["thick_anvil_lifetime"] = (
        dataset.thick_anvil_end_t - dataset.thick_anvil_start_t
    )

    dataset["thick_anvil_average_x"] = weighted_average_groupby(
        dataset.thick_anvil_step_x,
        dataset.thick_anvil_step_area,
        dataset.thick_anvil_step_anvil_index,
        dataset.anvil,
    )

    dataset["thick_anvil_average_y"] = weighted_average_groupby(
        dataset.thick_anvil_step_y,
        dataset.thick_anvil_step_area,
        dataset.thick_anvil_step_anvil_index,
        dataset.anvil,
    )

    dataset["thick_anvil_average_lat"] = weighted_average_groupby(
        dataset.thick_anvil_step_lat,
        dataset.thick_anvil_step_area,
        dataset.thick_anvil_step_anvil_index,
        dataset.anvil,
    )

    dataset["thick_anvil_average_lon"] = weighted_average_groupby(
        dataset.thick_anvil_step_lon,
        dataset.thick_anvil_step_area,
        dataset.thick_anvil_step_anvil_index,
        dataset.anvil,
    )

    dataset["thick_anvil_total_area"] = xr.DataArray(
        dataset.thick_anvil_step_area.groupby(dataset.thick_anvil_step_anvil_index)
        .sum()
        .data,
        {"anvil": dataset.anvil},
    )

    dataset["thick_anvil_max_area"] = xr.DataArray(
        dataset.thick_anvil_step_area.groupby(dataset.thick_anvil_step_anvil_index)
        .max()
        .data,
        {"anvil": dataset.anvil},
    )

    dataset["thick_anvil_max_area_t"] = argmax_groupby(
        dataset.thick_anvil_step_t,
        dataset.thick_anvil_step_area,
        dataset.thick_anvil_step_anvil_index,
        dataset.anvil,
    )

    for var in dataset.data_vars:
        if dataset[var].dims == ("thick_anvil_step",):
            new_var = "thick_anvil_" + var[17:]
            if var.endswith("_mean"):
                dataset[new_var] = combined_mean_groupby(
                    dataset[var],
                    dataset.thick_anvil_step_area,
                    dataset.thick_anvil_step_anvil_index,
                    dataset.anvil,
                )
            elif var.endswith("_std"):
                mean_var = var[:-3] + "mean"
                dataset[new_var] = combined_std_groupby(
                    dataset[var],
                    dataset[mean_var],
                    dataset.thick_anvil_step_area,
                    dataset.thick_anvil_step_anvil_index,
                    dataset.anvil,
                )
            elif var.endswith("_min"):
                dataset[new_var] = xr.DataArray(
                    dataset[var]
                    .groupby(dataset.thick_anvil_step_anvil_index)
                    .min()
                    .data,
                    {"anvil": dataset.anvil},
                )
            elif var.endswith("_max"):
                dataset[new_var] = xr.DataArray(
                    dataset[var]
                    .groupby(dataset.thick_anvil_step_anvil_index)
                    .max()
                    .data,
                    {"anvil": dataset.anvil},
                )
            elif var.endswith("_mean_uncertainty"):
                dataset[new_var] = weighted_average_uncertainty_groupby(
                    dataset[var],
                    dataset.thick_anvil_step_area,
                    dataset.thick_anvil_step_anvil_index,
                    dataset.anvil,
                )
            elif var.endswith("_mean_combined_error"):
                mean_var = var[:-15]
                std_da = combined_std_groupby(
                    dataset[var],
                    dataset[mean_var],
                    dataset.thick_anvil_step_area,
                    dataset.thick_anvil_step_anvil_index,
                    dataset.anvil,
                )
                uncertainty_da = weighted_average_uncertainty_groupby(
                    dataset[var],
                    dataset.thick_anvil_step_area,
                    dataset.thick_anvil_step_anvil_index,
                    dataset.anvil,
                )
                counts_da = counts_groupby(
                    dataset.thick_anvil_step_anvil_index, dataset.anvil
                )
                combined_error = (
                    (std_da.data / counts_da.data**0.5) ** 2
                    + uncertainty_da.data**2
                ) ** 0.5
                dataset[new_var] = xr.DataArray(
                    combined_error,
                    {"anvil": dataset.anvil},
                )
            elif var.endswith("_min_error"):
                min_var = var[:-6]
                dataset[new_var] = argmin_groupby(
                    dataset[var],
                    dataset[min_var],
                    dataset.thick_anvil_step_anvil_index,
                    dataset.anvil,
                )
            elif var.endswith("_max_error"):
                max_var = var[:-6]
                dataset[new_var] = argmax_groupby(
                    dataset[var],
                    dataset[max_var],
                    dataset.thick_anvil_step_anvil_index,
                    dataset.anvil,
                )

    return dataset


def process_thin_anvil_properties(dataset):
    thin_anvil_start_step = argmin_groupby(
        dataset.thin_anvil_step,
        dataset.thin_anvil_step_t,
        dataset.thin_anvil_step_anvil_index,
        dataset.anvil,
    )

    dataset["thin_anvil_start_x"] = dataset.thin_anvil_step_x.loc[thin_anvil_start_step]
    dataset["thin_anvil_start_y"] = dataset.thin_anvil_step_y.loc[thin_anvil_start_step]
    dataset["thin_anvil_start_lat"] = dataset.thin_anvil_step_lat.loc[
        thin_anvil_start_step
    ]
    dataset["thin_anvil_start_lon"] = dataset.thin_anvil_step_lon.loc[
        thin_anvil_start_step
    ]
    dataset["thin_anvil_start_t"] = dataset.thin_anvil_step_t.loc[thin_anvil_start_step]

    thin_anvil_end_step = argmax_groupby(
        dataset.thin_anvil_step,
        dataset.thin_anvil_step_t,
        dataset.thin_anvil_step_anvil_index,
        dataset.anvil,
    )

    dataset["thin_anvil_end_x"] = dataset.thin_anvil_step_x.loc[thin_anvil_end_step]
    dataset["thin_anvil_end_y"] = dataset.thin_anvil_step_y.loc[thin_anvil_end_step]
    dataset["thin_anvil_end_lat"] = dataset.thin_anvil_step_lat.loc[thin_anvil_end_step]
    dataset["thin_anvil_end_lon"] = dataset.thin_anvil_step_lon.loc[thin_anvil_end_step]
    dataset["thin_anvil_end_t"] = dataset.thin_anvil_step_t.loc[thin_anvil_end_step]
    dataset["thin_anvil_lifetime"] = (
        dataset.thin_anvil_end_t - dataset.thin_anvil_start_t
    )

    dataset["thin_anvil_average_x"] = weighted_average_groupby(
        dataset.thin_anvil_step_x,
        dataset.thin_anvil_step_area,
        dataset.thin_anvil_step_anvil_index,
        dataset.anvil,
    )

    dataset["thin_anvil_average_y"] = weighted_average_groupby(
        dataset.thin_anvil_step_y,
        dataset.thin_anvil_step_area,
        dataset.thin_anvil_step_anvil_index,
        dataset.anvil,
    )

    dataset["thin_anvil_average_lat"] = weighted_average_groupby(
        dataset.thin_anvil_step_lat,
        dataset.thin_anvil_step_area,
        dataset.thin_anvil_step_anvil_index,
        dataset.anvil,
    )

    dataset["thin_anvil_average_lon"] = weighted_average_groupby(
        dataset.thin_anvil_step_lon,
        dataset.thin_anvil_step_area,
        dataset.thin_anvil_step_anvil_index,
        dataset.anvil,
    )

    dataset["thin_anvil_total_area"] = xr.DataArray(
        dataset.thin_anvil_step_area.groupby(dataset.thin_anvil_step_anvil_index)
        .sum()
        .data,
        {"anvil": dataset.anvil},
    )

    dataset["thin_anvil_max_area"] = xr.DataArray(
        dataset.thin_anvil_step_area.groupby(dataset.thin_anvil_step_anvil_index)
        .max()
        .data,
        {"anvil": dataset.anvil},
    )

    dataset["thin_anvil_max_area_t"] = argmax_groupby(
        dataset.thin_anvil_step_t,
        dataset.thin_anvil_step_area,
        dataset.thin_anvil_step_anvil_index,
        dataset.anvil,
    )

    for var in dataset.data_vars:
        if dataset[var].dims == ("thin_anvil_step",):
            new_var = "thin_anvil_" + var[16:]
            if var.endswith("_mean"):
                dataset[new_var] = combined_mean_groupby(
                    dataset[var],
                    dataset.thin_anvil_step_area,
                    dataset.thin_anvil_step_anvil_index,
                    dataset.anvil,
                )
            elif var.endswith("_std"):
                mean_var = var[:-3] + "mean"
                dataset[new_var] = combined_std_groupby(
                    dataset[var],
                    dataset[mean_var],
                    dataset.thin_anvil_step_area,
                    dataset.thin_anvil_step_anvil_index,
                    dataset.anvil,
                )
            elif var.endswith("_min"):
                dataset[new_var] = xr.DataArray(
                    dataset[var]
                    .groupby(dataset.thin_anvil_step_anvil_index)
                    .min()
                    .data,
                    {"anvil": dataset.anvil},
                )
            elif var.endswith("_max"):
                dataset[new_var] = xr.DataArray(
                    dataset[var]
                    .groupby(dataset.thin_anvil_step_anvil_index)
                    .max()
                    .data,
                    {"anvil": dataset.anvil},
                )
            elif var.endswith("_mean_uncertainty"):
                dataset[new_var] = weighted_average_uncertainty_groupby(
                    dataset[var],
                    dataset.thin_anvil_step_area,
                    dataset.thin_anvil_step_anvil_index,
                    dataset.anvil,
                )
            elif var.endswith("_mean_combined_error"):
                mean_var = var[:-15]
                std_var = mean_var[:-4] + "std"
                uncertainty_da = weighted_average_uncertainty_groupby(
                    dataset[var],
                    dataset.thin_anvil_step_area,
                    dataset.thin_anvil_step_anvil_index,
                    dataset.anvil,
                )
                counts_da = counts_groupby(
                    dataset.thin_anvil_step_anvil_index, dataset.anvil
                )
                combined_error = (
                    (dataset[std_var].data / counts_da.data**0.5) ** 2
                    + uncertainty_da.data**2
                ) ** 0.5
                dataset[new_var] = xr.DataArray(
                    combined_error,
                    {"anvil": dataset.anvil},
                )
            elif var.endswith("_min_error"):
                min_var = var[:-6]
                dataset[new_var] = argmin_groupby(
                    dataset[var],
                    dataset[min_var],
                    dataset.thin_anvil_step_anvil_index,
                    dataset.anvil,
                )
            elif var.endswith("_max_error"):
                max_var = var[:-6]
                dataset[new_var] = argmax_groupby(
                    dataset[var],
                    dataset[max_var],
                    dataset.thin_anvil_step_anvil_index,
                    dataset.anvil,
                )

    return dataset
