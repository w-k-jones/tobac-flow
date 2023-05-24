import argparse
import pathlib
from datetime import datetime
import numpy as np
import xarray as xr
from tobac_flow.postprocess import (
    process_core_properties,
    process_thick_anvil_properties,
    process_thin_anvil_properties,
)
from tobac_flow.utils import (
    remove_orphan_coords,
    filter_cores,
    filter_anvils,
    counts_groupby,
    argmin_groupby,
)

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
            dataset.core_edge_label_flag.loc[core_overlap] = np.logical_or(
                dataset.core_edge_label_flag.loc[core_overlap].data,
                dcc_ds.core_edge_label_flag.loc[core_overlap].data,
            )
            dataset.core_end_label_flag.loc[
                core_overlap
            ] = dcc_ds.core_end_label_flag.loc[core_overlap]
            dataset.core_nan_flag.loc[core_overlap] = np.logical_or(
                dataset.core_nan_flag.loc[core_overlap].data,
                dcc_ds.core_nan_flag.loc[core_overlap].data,
            )

            wh_zero = dataset.core_anvil_index.loc[core_overlap] == 0
            wh_anvil_is_zero_cores = wh_zero.core.data[wh_zero.data]
            dataset.core_anvil_index.loc[
                wh_anvil_is_zero_cores
            ] = dcc_ds.core_anvil_index.loc[wh_anvil_is_zero_cores]

        anvil_overlap = sorted(list(set(dataset.anvil.data) & set(dcc_ds.anvil.data)))
        if len(core_overlap) > 0:
            dataset.thick_anvil_edge_label_flag.loc[anvil_overlap] = np.logical_or(
                dataset.thick_anvil_edge_label_flag.loc[anvil_overlap].data,
                dcc_ds.thick_anvil_edge_label_flag.loc[anvil_overlap].data,
            )
            dataset.thick_anvil_end_label_flag.loc[
                anvil_overlap
            ] = dcc_ds.thick_anvil_end_label_flag.loc[anvil_overlap]
            dataset.thick_anvil_nan_flag.loc[anvil_overlap] = np.logical_or(
                dataset.thick_anvil_nan_flag.loc[anvil_overlap].data,
                dcc_ds.thick_anvil_nan_flag.loc[anvil_overlap].data,
            )
            dataset.thin_anvil_edge_label_flag.loc[anvil_overlap] = np.logical_or(
                dataset.thin_anvil_edge_label_flag.loc[anvil_overlap].data,
                dcc_ds.thin_anvil_edge_label_flag.loc[anvil_overlap].data,
            )
            dataset.thin_anvil_end_label_flag.loc[
                anvil_overlap
            ] = dcc_ds.thin_anvil_end_label_flag.loc[anvil_overlap]
            dataset.thin_anvil_nan_flag.loc[anvil_overlap] = np.logical_or(
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

dataset = remove_orphan_coords(dataset)
print(datetime.now(), "Removing orphaned items", flush=True)
# Remove invalid cores and process core properties
print(datetime.now(), "Filtering and processing cores", flush=True)
dataset = filter_cores(dataset, verbose=True)
dataset = process_core_properties(dataset)

print(datetime.now(), "Filtering and processing anvils", flush=True)
dataset = filter_anvils(dataset, verbose=True)
dataset = process_thick_anvil_properties(dataset)
dataset = process_thin_anvil_properties(dataset)

dataset["core_has_anvil_flag"] = xr.DataArray(
    np.isin(dataset.core_anvil_index, dataset.anvil), {"core": dataset.core}
)

dataset["core_anvil_removed"] = xr.DataArray(
    np.logical_and(
        np.logical_not(dataset.core_has_anvil_flag), dataset.core_anvil_index != 0
    ),
    {"core": dataset.core},
)

dataset.core_anvil_index[np.logical_not(dataset.core_has_anvil_flag)] = 0

dataset["anvil_core_count"] = counts_groupby(
    dataset.core_anvil_index[dataset.core_has_anvil_flag], dataset.anvil
)

dataset["anvil_initial_core_index"] = argmin_groupby(
    dataset.core[dataset.core_has_anvil_flag],
    dataset.core_start_t[dataset.core_has_anvil_flag],
    dataset.core_anvil_index[dataset.core_has_anvil_flag],
    dataset.anvil,
)

dataset["anvil_no_growth_flag"] = (
    dataset.thick_anvil_max_area_t
    <= dataset.core_end_t.loc[dataset.anvil_initial_core_index]
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

anvil_has_invalid_cores = np.logical_not(
    dataset.core_is_valid.groupby(dataset.core_anvil_index)
    .reduce(np.all)
    .loc[dataset.anvil.data]
)
dataset["thick_anvil_is_valid"] = xr.DataArray(
    np.logical_not(
        np.logical_or.reduce(
            [
                anvil_has_invalid_cores.data,
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
                anvil_has_invalid_cores.data,
                dataset.thin_anvil_edge_label_flag.data,
                dataset.thin_anvil_start_label_flag.data,
                dataset.thin_anvil_end_label_flag.data,
                dataset.thin_anvil_nan_flag.data,
            ]
        )
    ),
    {"anvil": dataset.anvil},
)

print(f"Final valid core count: {dataset.core_is_valid.data.sum()}")
print(f"Final valid thick anvil count: {dataset.thick_anvil_is_valid.data.sum()}")
print(f"Final valid thin anvil count: {dataset.thin_anvil_is_valid.data.sum()}")

print(datetime.now(), "Saving to %s" % (save_path), flush=True)
# Add compression encoding
comp = dict(zlib=True, complevel=5, shuffle=True)
for var in dataset.data_vars:
    dataset[var].encoding.update(comp)

dataset.to_netcdf(save_path)

print(datetime.now(), "Saving complete, closing datasets", flush=True)

dataset.close()
