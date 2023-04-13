import argparse
import pathlib
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(
    description="Combine multiple files of detected DCCs in GOES-16 ABI data"
)
parser.add_argument("-sd", help="Directory to save output files", default=None)
parser.add_argument("files", help="List of files to combine", nargs="+", type=str)

args = parser.parse_args()

dcc_files = sorted([pathlib.Path(f) for f in args.files])

start_str = dcc_files[0].stem.split("_S")[-1][:15]
end_str = dcc_files[-1].stem.split("_E")[-1][:15]
x_str = dcc_files[0].stem.split("_X")[-1][:9]
y_str = dcc_files[0].stem.split("_Y")[-1][:9]
new_filename = f"dcc_statistics_G16_S{start_str}_E{end_str}_X{x_str}_Y{y_str}.nc"
new_filename

save_dir = pathlib.Path(args.sd)
if not save_dir.exists():
    save_dir.mkdir()

save_path = pathlib.Path(args.sd) / new_filename

# Load files

with xr.open_dataset(dcc_files[0]) as dcc_ds:
    print(dcc_files[0])
    # Add NaN labels back in
    var_list = [
        var
        for var in dcc_ds.data_vars
        if dcc_ds.data_vars[var].dims
        in [("core_step",), ("thick_anvil_step",), ("thin_anvil_step",)]
    ]
    var_list = [
        "core_edge_label_flag",
        "core_start_label_flag",
        "core_end_label_flag",
        "thick_anvil_edge_label_flag",
        "thick_anvil_start_label_flag",
        "thick_anvil_end_label_flag",
        "thin_anvil_edge_label_flag",
        "thin_anvil_start_label_flag",
        "thin_anvil_end_label_flag",
        "core_nan_flag",
        "thick_anvil_nan_flag",
        "thin_anvil_nan_flag",
        "core_anvil_index",
    ] + var_list
    dataset = dcc_ds.get(var_list)
    output_dtypes = {var: dataset[var].dtype for var in var_list}

for f in dcc_files[1:]:
    with xr.open_dataset(f) as dcc_ds:
        print(f)
        dcc_ds = dcc_ds.get(var_list)
        core_overlap = sorted(list(set(dataset.core.data) & set(dcc_ds.core.data)))
        if len(core_overlap) > 0:
            dataset.core_edge_label_flag.loc[core_overlap].data = np.logical_or(
                dataset.core_edge_label_flag.loc[core_overlap].data,
                dcc_ds.core_edge_label_flag.loc[core_overlap].data,
            )
            dataset.core_end_label_flag.loc[
                core_overlap
            ].data = dcc_ds.core_end_label_flag.loc[core_overlap].data
            dataset.core_nan_flag.loc[core_overlap].data = np.logical_or(
                dataset.core_nan_flag.loc[core_overlap].data,
                dcc_ds.core_nan_flag.loc[core_overlap].data,
            )

            wh_zero = dataset.core_anvil_index.loc[core_overlap] == 0
            wh_anvil_is_zero_cores = wh_zero.core.data[wh_zero.data]
            dataset.core_anvil_index.loc[
                wh_anvil_is_zero_cores
            ].data = dcc_ds.core_anvil_index.loc[wh_anvil_is_zero_cores].data

        anvil_overlap = sorted(list(set(dataset.anvil.data) & set(dcc_ds.anvil.data)))
        if len(core_overlap) > 0:
            dataset.thick_anvil_edge_label_flag.loc[anvil_overlap].data = np.logical_or(
                dataset.thick_anvil_edge_label_flag.loc[anvil_overlap].data,
                dcc_ds.thick_anvil_edge_label_flag.loc[anvil_overlap].data,
            )
            dataset.thick_anvil_end_label_flag.loc[
                anvil_overlap
            ].data = dcc_ds.thick_anvil_end_label_flag.loc[anvil_overlap].data
            dataset.thick_anvil_nan_flag.loc[anvil_overlap].data = np.logical_or(
                dataset.thick_anvil_nan_flag.loc[anvil_overlap].data,
                dcc_ds.thick_anvil_nan_flag.loc[anvil_overlap].data,
            )
            dataset.thin_anvil_edge_label_flag.loc[anvil_overlap].data = np.logical_or(
                dataset.thin_anvil_edge_label_flag.loc[anvil_overlap].data,
                dcc_ds.thin_anvil_edge_label_flag.loc[anvil_overlap].data,
            )
            dataset.thin_anvil_end_label_flag.loc[
                anvil_overlap
            ].data = dcc_ds.thin_anvil_end_label_flag.loc[anvil_overlap].data
            dataset.thin_anvil_nan_flag.loc[anvil_overlap].data = np.logical_or(
                dataset.thin_anvil_nan_flag.loc[anvil_overlap].data,
                dcc_ds.thin_anvil_nan_flag.loc[anvil_overlap].data,
            )

        # Now combine the rest, by concatenating along each dimension
        core_different = sorted(list(set(dcc_ds.core.data) - set(dataset.core.data)))
        anvil_different = sorted(list(set(dcc_ds.anvil.data) - set(dataset.anvil.data)))

        dataset = xr.combine_by_coords(
            data_objects=[
                dataset,
                dcc_ds.sel(core=core_different, anvil=anvil_different),
            ],
            data_vars="different",
            coords="different",
            join="outer",
        )

for var, dtype in output_dtypes.items():
    dataset[var] = dataset[var].astype(dtype)

# Filter invalid anvils and cores from dataset

# Filter out cores/anvils or steps which don't appear in the others index

wh_core = np.isin(dataset.core, dataset.core_step_core_index)
wh_core_step = np.isin(dataset.core_step_core_index, dataset.core)
wh_anvil = np.logical_and(
    np.isin(dataset.anvil, dataset.thick_anvil_step_anvil_index),
    np.isin(dataset.anvil, dataset.thin_anvil_step_anvil_index),
)
wh_thick_anvil_step = np.isin(dataset.thick_anvil_step_anvil_index, dataset.anvil)
wh_thin_anvil_step = np.isin(dataset.thin_anvil_step_anvil_index, dataset.anvil)

dataset = dataset.sel(
    core=dataset.core.data[wh_core],
    anvil=dataset.anvil.data[wh_anvil],
    core_step=dataset.core_step.data[wh_core_step],
    thick_anvil_step=dataset.thick_anvil_step[wh_thick_anvil_step],
    thin_anvil_step=dataset.thin_anvil_step[wh_thin_anvil_step],
)

# Anvil NaN filter
def any_nan(x, *args, **kwargs):
    return np.any(np.isnan(x))


thick_anvil_any_nan_step = dataset.thick_anvil_step_BT_mean.groupby(
    dataset.thick_anvil_step_anvil_index
).reduce(any_nan)


def start_end_diff(x, *args, **kwargs):
    return x[-1] - x[0]


anvil_lifetime = dataset.thick_anvil_step_t.groupby(
    dataset.thick_anvil_step_anvil_index
).reduce(start_end_diff)

anvil_invalid_lifetime = anvil_lifetime < np.timedelta64(timedelta(minutes=15))


def max_t_diff(x, *args, **kwargs):
    if len(x) > 1:
        return np.max(np.diff(x))
    else:
        return np.timedelta64(timedelta(minutes=0))


thick_anvil_max_time_diff = dataset.thick_anvil_step_t.groupby(
    dataset.thick_anvil_step_anvil_index
).reduce(max_t_diff)

thick_anvil_invalid_time_diff = thick_anvil_max_time_diff >= np.timedelta64(
    timedelta(minutes=20)
)

anvil_nan_flag = np.logical_or.reduce(
    [
        dataset.thick_anvil_nan_flag.data,
        dataset.thin_anvil_nan_flag.data,
        thick_anvil_any_nan_step.data,
        anvil_invalid_lifetime.data,
        thick_anvil_invalid_time_diff.data,
    ]
)

dataset = dataset.sel(anvil=dataset.anvil.data[np.logical_not(anvil_nan_flag)])

# Core filter

core_invalid_anvil = np.logical_not(
    np.isin(dataset.core_anvil_index.data, dataset.anvil.data)
)


def start_end_diff(x, *args, **kwargs):
    return x[0] - x[-1]


core_bt_change = dataset.core_step_BT_mean.groupby(dataset.core_step_core_index).reduce(
    start_end_diff
)

core_invalid_bt = core_bt_change.data < 8


def max_t_diff(x, *args, **kwargs):
    if len(x) > 1:
        return np.max(np.diff(x))
    else:
        return np.timedelta64(timedelta(minutes=0))


core_max_time_diff = dataset.core_step_t.groupby(dataset.core_step_core_index).reduce(
    max_t_diff
)

core_invalid_time_diff = core_max_time_diff >= np.timedelta64(timedelta(minutes=20))


def end_start_diff(x, *args, **kwargs):
    return x[-1] - x[0]


core_lifetime = dataset.core_step_t.groupby(dataset.core_step_core_index).reduce(
    end_start_diff
)

core_invalid_lifetime = core_lifetime < np.timedelta64(timedelta(minutes=15))

wh_core_invalid = np.logical_or.reduce(
    [
        dataset.core_start_label_flag.data,
        dataset.core_end_label_flag.data,
        dataset.core_edge_label_flag.data,
        dataset.core_nan_flag,
        core_invalid_anvil,
        core_invalid_bt,
        core_invalid_time_diff,
        core_invalid_lifetime,
    ]
)

dataset = dataset.sel(core=dataset.core.data[np.logical_not(wh_core_invalid)])

# Remove anvils with no core

wh_anvil_has_core = np.isin(dataset.anvil, dataset.core_anvil_index)

dataset = dataset.sel(anvil=dataset.anvil.data[wh_anvil_has_core])

# Now remove anvils with no growth

anvil_initial_core = (
    dataset.core.groupby(dataset.core_anvil_index)
    .min()
    .rename(core_anvil_index="anvil")
)

core_end_step = (
    dataset.core_step.groupby(dataset.core_step_core_index)
    .max()
    .rename(core_step_core_index="core")
)

core_end_t = dataset.core_step_t.loc[core_end_step]

thick_anvil_max_area_step = [
    int(group[1].idxmax().item())
    for group in dataset.thick_anvil_step_area.groupby(
        dataset.thick_anvil_step_anvil_index
    )
]
thick_anvil_max_area_t = dataset.thick_anvil_step_t.loc[thick_anvil_max_area_step]

wh_no_growth = (
    core_end_t[np.logical_not(wh_core_invalid)].loc[anvil_initial_core.data].data
    > thick_anvil_max_area_t[np.logical_not(anvil_nan_flag)][wh_anvil_has_core].data
)

dataset = dataset.sel(anvil=dataset.anvil.data[np.logical_not(wh_no_growth)])

# Final pass to remove cores without anvils

core_invalid_anvil = np.logical_not(
    np.isin(dataset.core_anvil_index.data, dataset.anvil.data)
)

dataset = dataset.sel(core=dataset.core.data[np.logical_not(core_invalid_anvil)])

# Now filter step labels

filtered_cores = dataset.core_step.data[
    np.isin(dataset.core_step_core_index, dataset.core)
]

filtered_thick_anvils = dataset.thick_anvil_step.data[
    np.isin(dataset.thick_anvil_step_anvil_index, dataset.anvil)
]

filtered_thin_anvils = dataset.thin_anvil_step.data[
    np.isin(dataset.thin_anvil_step_anvil_index, dataset.anvil)
]

dataset = dataset.sel(
    core_step=filtered_cores,
    thick_anvil_step=filtered_thick_anvils,
    thin_anvil_step=filtered_thin_anvils,
)

# Core start/end positions

core_start_step = (
    dataset.core_step.groupby(dataset.core_step_core_index)
    .min()
    .rename(core_step_core_index="core")
)

dataset["core_start_x"] = dataset.core_step_x.loc[core_start_step]
dataset["core_start_y"] = dataset.core_step_y.loc[core_start_step]

dataset["core_start_lat"] = dataset.core_step_lat.loc[core_start_step]
dataset["core_start_lon"] = dataset.core_step_lon.loc[core_start_step]

dataset["core_start_t"] = dataset.core_step_t.loc[core_start_step]

core_end_step = (
    dataset.core_step.groupby(dataset.core_step_core_index)
    .max()
    .rename(core_step_core_index="core")
)

dataset["core_end_x"] = dataset.core_step_x.loc[core_end_step]
dataset["core_end_y"] = dataset.core_step_y.loc[core_end_step]

dataset["core_end_lat"] = dataset.core_step_lat.loc[core_end_step]
dataset["core_end_lon"] = dataset.core_step_lon.loc[core_end_step]

dataset["core_end_t"] = dataset.core_step_t.loc[core_end_step]

dataset["core_lifetime"] = dataset.core_end_t - dataset.core_start_t


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
            / ((step_t[3:] - step_t[:-3]).astype("timedelta64[s]").astype("int") / 60)
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
            for BT_group, time_group in zip(BT.groupby(groups), times.groupby(groups))
        ],
        {coord.name: coord},
    )


dataset["core_cooling_rate"] = cooling_rate_groupby(
    dataset.core_step_BT_mean,
    dataset.core_step_t,
    dataset.core_step_core_index,
    dataset.core,
)


def calc_combined_mean(step_mean, step_area):
    return np.sum(step_mean * step_area) / np.sum(step_area)


def calc_combined_std(step_std, step_mean, step_area):
    combined_mean = calc_combined_mean(step_mean, step_area)
    return (
        (
            np.sum(step_area * step_std)
            + np.sum(step_area * (step_mean - combined_mean) ** 2)
        )
        / np.sum(step_area)
    ) ** 0.5


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


for var in dataset.data_vars:
    if dataset[var].dims == ("core_step",):
        if var.endswith("_mean"):
            new_var = "core_" + var[10:]
            dataset[new_var] = combined_mean_groupby(
                dataset[var],
                dataset.core_step_area,
                dataset.core_step_core_index,
                dataset.core,
            )
        if var.endswith("_std"):
            new_var = "core_" + var[10:]
            mean_var = var[:-3] + "mean"
            dataset[new_var] = combined_std_groupby(
                dataset[var],
                dataset[mean_var],
                dataset.core_step_area,
                dataset.core_step_core_index,
                dataset.core,
            )
        if var.endswith("_min"):
            new_var = "core_" + var[10:]
            dataset[new_var] = xr.DataArray(
                dataset[var].groupby(dataset.core_step_core_index).min().data,
                {"core": dataset.core},
            )
        if var.endswith("_max"):
            new_var = "core_" + var[10:]
            dataset[new_var] = xr.DataArray(
                dataset[var].groupby(dataset.core_step_core_index).max().data,
                {"core": dataset.core},
            )


def gb_len(x, *args, **kwargs):
    return len(x)


# Thick anvil properties


def gb_len(x, *args, **kwargs):
    return len(x)


dataset["anvil_core_count"] = xr.DataArray(
    dataset.core_anvil_index.groupby(dataset.core_anvil_index).reduce(gb_len).data,
    {"anvil": dataset.anvil},
)

thick_anvil_start_step = (
    dataset.thick_anvil_step.groupby(dataset.thick_anvil_step_anvil_index)
    .min()
    .rename(thick_anvil_step_anvil_index="anvil")
)

dataset["thick_anvil_start_x"] = dataset.thick_anvil_step_x.loc[thick_anvil_start_step]
dataset["thick_anvil_start_y"] = dataset.thick_anvil_step_y.loc[thick_anvil_start_step]

dataset["thick_anvil_start_lat"] = dataset.thick_anvil_step_lat.loc[
    thick_anvil_start_step
]
dataset["thick_anvil_start_lon"] = dataset.thick_anvil_step_lon.loc[
    thick_anvil_start_step
]

dataset["thick_anvil_start_t"] = dataset.thick_anvil_step_t.loc[thick_anvil_start_step]

thick_anvil_end_step = (
    dataset.thick_anvil_step.groupby(dataset.thick_anvil_step_anvil_index)
    .max()
    .rename(thick_anvil_step_anvil_index="anvil")
)

dataset["thick_anvil_end_x"] = dataset.thick_anvil_step_x.loc[thick_anvil_end_step]
dataset["thick_anvil_end_y"] = dataset.thick_anvil_step_y.loc[thick_anvil_end_step]

dataset["thick_anvil_end_lat"] = dataset.thick_anvil_step_lat.loc[thick_anvil_end_step]
dataset["thick_anvil_end_lon"] = dataset.thick_anvil_step_lon.loc[thick_anvil_end_step]

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

dataset["thick_anvil_min_BT_t"] = argmin_groupby(
    dataset.thick_anvil_step_t,
    dataset.thick_anvil_step_BT_mean,
    dataset.thick_anvil_step_anvil_index,
    dataset.anvil,
)

for var in dataset.data_vars:
    if dataset[var].dims == ("thick_anvil_step",):
        if var.endswith("_mean"):
            new_var = "thick_anvil_" + var[17:]
            dataset[new_var] = combined_mean_groupby(
                dataset[var],
                dataset.thick_anvil_step_area,
                dataset.thick_anvil_step_anvil_index,
                dataset.anvil,
            )
        if var.endswith("_std"):
            new_var = "thick_anvil_" + var[17:]
            mean_var = var[:-3] + "mean"
            dataset[new_var] = combined_std_groupby(
                dataset[var],
                dataset[mean_var],
                dataset.thick_anvil_step_area,
                dataset.thick_anvil_step_anvil_index,
                dataset.anvil,
            )
        if var.endswith("_min"):
            new_var = "thick_anvil_" + var[17:]
            dataset[new_var] = xr.DataArray(
                dataset[var].groupby(dataset.thick_anvil_step_anvil_index).min().data,
                {"anvil": dataset.anvil},
            )
        if var.endswith("_max"):
            new_var = "thick_anvil_" + var[17:]
            dataset[new_var] = xr.DataArray(
                dataset[var].groupby(dataset.thick_anvil_step_anvil_index).max().data,
                {"anvil": dataset.anvil},
            )

# Thin anvil properties


def gb_len(x, *args, **kwargs):
    return len(x)


dataset["anvil_core_count"] = xr.DataArray(
    dataset.core_anvil_index.groupby(dataset.core_anvil_index).reduce(gb_len).data,
    {"anvil": dataset.anvil},
)

thin_anvil_start_step = (
    dataset.thin_anvil_step.groupby(dataset.thin_anvil_step_anvil_index)
    .min()
    .rename(thin_anvil_step_anvil_index="anvil")
)

dataset["thin_anvil_start_x"] = dataset.thin_anvil_step_x.loc[thin_anvil_start_step]
dataset["thin_anvil_start_y"] = dataset.thin_anvil_step_y.loc[thin_anvil_start_step]

dataset["thin_anvil_start_lat"] = dataset.thin_anvil_step_lat.loc[thin_anvil_start_step]
dataset["thin_anvil_start_lon"] = dataset.thin_anvil_step_lon.loc[thin_anvil_start_step]

dataset["thin_anvil_start_t"] = dataset.thin_anvil_step_t.loc[thin_anvil_start_step]

thin_anvil_end_step = (
    dataset.thin_anvil_step.groupby(dataset.thin_anvil_step_anvil_index)
    .max()
    .rename(thin_anvil_step_anvil_index="anvil")
)

dataset["thin_anvil_end_x"] = dataset.thin_anvil_step_x.loc[thin_anvil_end_step]
dataset["thin_anvil_end_y"] = dataset.thin_anvil_step_y.loc[thin_anvil_end_step]

dataset["thin_anvil_end_lat"] = dataset.thin_anvil_step_lat.loc[thin_anvil_end_step]
dataset["thin_anvil_end_lon"] = dataset.thin_anvil_step_lon.loc[thin_anvil_end_step]

dataset["thin_anvil_end_t"] = dataset.thin_anvil_step_t.loc[thin_anvil_end_step]

dataset["thin_anvil_lifetime"] = dataset.thin_anvil_end_t - dataset.thin_anvil_start_t

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

dataset["thin_anvil_min_BT_t"] = argmin_groupby(
    dataset.thin_anvil_step_t,
    dataset.thin_anvil_step_BT_mean,
    dataset.thin_anvil_step_anvil_index,
    dataset.anvil,
)

for var in dataset.data_vars:
    if dataset[var].dims == ("thin_anvil_step",):
        if var.endswith("_mean"):
            new_var = "thin_anvil_" + var[16:]
            dataset[new_var] = combined_mean_groupby(
                dataset[var],
                dataset.thin_anvil_step_area,
                dataset.thin_anvil_step_anvil_index,
                dataset.anvil,
            )
        if var.endswith("_std"):
            new_var = "thin_anvil_" + var[16:]
            mean_var = var[:-3] + "mean"
            dataset[new_var] = combined_std_groupby(
                dataset[var],
                dataset[mean_var],
                dataset.thin_anvil_step_area,
                dataset.thin_anvil_step_anvil_index,
                dataset.anvil,
            )
        if var.endswith("_min"):
            new_var = "thin_anvil_" + var[16:]
            dataset[new_var] = xr.DataArray(
                dataset[var].groupby(dataset.thin_anvil_step_anvil_index).min().data,
                {"anvil": dataset.anvil},
            )
        if var.endswith("_max"):
            new_var = "thin_anvil_" + var[16:]
            dataset[new_var] = xr.DataArray(
                dataset[var].groupby(dataset.thin_anvil_step_anvil_index).max().data,
                {"anvil": dataset.anvil},
            )

dataset["anvil_initial_core_index"] = xr.DataArray(
    dataset.core.groupby(dataset.core_anvil_index).min().data, {"anvil": dataset.anvil}
)

# Add valid flags combining the exisiting data flags
dataset["core_is_valid"] = xr.DataArray(
    np.logical_not(
        np.logical_or.reduce(
            [
                dataset.core_edge_label_flag.data,
                dataset.core_start_label_flag.data,
                dataset.core_end_label_flag.data,
                dataset.core_nan_flag.data,
            ]
        )
    ),
    {"core": dataset.core},
)

dataset["thick_anvil_is_valid"] = xr.DataArray(
    np.logical_not(
        np.logical_or.reduce(
            [
                dataset.thick_anvil_edge_label_flag.data,
                dataset.thick_anvil_start_label_flag.data,
                dataset.thick_anvil_end_label_flag.data,
                dataset.thick_anvil_nan_flag.data,
            ]
        )
    ),
    {"anvil": dataset.anvil},
)

dataset["thin_anvil_is_valid"] = xr.DataArray(
    np.logical_not(
        np.logical_or.reduce(
            [
                dataset.thin_anvil_edge_label_flag.data,
                dataset.thin_anvil_start_label_flag.data,
                dataset.thin_anvil_end_label_flag.data,
                dataset.thin_anvil_nan_flag.data,
            ]
        )
    ),
    {"anvil": dataset.anvil},
)

print(datetime.now(), "Saving to %s" % (save_path), flush=True)
# Add compression encoding
comp = dict(zlib=True, complevel=5, shuffle=True)
for var in dataset.data_vars:
    dataset[var].encoding.update(comp)

dataset.to_netcdf(save_path)

dataset.close()
