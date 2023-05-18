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
