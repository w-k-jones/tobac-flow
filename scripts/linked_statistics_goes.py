import argparse
import pathlib
import warnings

# Ignore warnings from datetime conversion
warnings.simplefilter("ignore", category=UserWarning)

from datetime import datetime

import numpy as np
import xarray as xr

from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta

from tobac_flow.linking import increment_step_coords
from tobac_flow.postprocess import (
    add_validity_flags,
    process_core_properties,
    process_thick_anvil_properties,
    process_thin_anvil_properties,
)
from tobac_flow.utils import (
    remove_orphan_coords,
    filter_cores,
    filter_anvils,
)
from tobac_flow.utils.xarray_utils import add_compression_encoding, sel_anvil, sel_core

parser = argparse.ArgumentParser(
    description="Combine multiple files of detected DCCs in GOES-16 ABI data"
)
parser.add_argument("date", help="Date on which to start process", type=str)
parser.add_argument("-id", help="Input directory", type=str, default="")
parser.add_argument("-sd", help="Directory to save output files", type=str, default="")

if __name__=="__main__":
    args = parser.parse_args()

    start_date = parse_date(args.date, fuzzy=True)
    end_date = start_date + relativedelta(months=1)
    print(f'Processing statistics for period {start_date.isoformat()} to {end_date.isoformat()}', flush=True)

    processed_dir = pathlib.Path(args.id)
    assert processed_dir.exists(), f'Invalid path to input directory: {processed_dir} does not exist'
    
    save_dir = pathlib.Path(args.sd)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_processed_files = np.array(sorted(list(processed_dir.rglob("detected_dccs_G16_*.nc"))))

    file_dates = np.array([datetime.strptime(f.name.split("_S")[1][:15], "%Y%m%d_%H%M%S") for f in all_processed_files])

    dcc_files = all_processed_files[(file_dates >= start_date) & (file_dates < end_date)]
    post_files = all_processed_files[file_dates >= end_date]

    start_str = dcc_files[0].stem.split("_S")[-1][:15]
    end_str = dcc_files[-1].stem.split("_E")[-1][:15]
    x_str = dcc_files[0].stem.split("_X")[-1][:9]
    y_str = dcc_files[0].stem.split("_Y")[-1][:9]
    new_filename = f"dcc_statistics_G16_S{start_str}_E{end_str}_X{x_str}_Y{y_str}.nc"

    save_path = pathlib.Path(args.sd) / new_filename

    # Find cores/anvils to drop from output as the exist in previous month
    if np.any(file_dates < start_date):
        prior_file = all_processed_files[file_dates < start_date][-1]
        with xr.open_dataset(prior_file) as prior_ds:
            drop_cores = prior_ds.core.values
            drop_anvils = prior_ds.anvil.values
    
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
        
        output_dtypes = {var:dataset[var].dtype for var in var_list}

    for f in dcc_files[1:]:
        print(f)
        with xr.open_dataset(f) as dcc_ds:
            dcc_ds = dcc_ds.get(var_list)
            
            dcc_ds = increment_step_coords(dcc_ds, dataset)
            
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
                combine_attrs="drop", 
            )

    keep_anvils = np.setdiff1d(dataset.anvil.values, drop_anvils)
    keep_cores = np.setdiff1d(dataset.core.values, drop_cores)

    # Now continue through subsequent files until no more anvils/cores are found:
    for f in post_files:
        print(f)
        with xr.open_dataset(f) as dcc_ds:
            if (
                np.any(np.isin(dcc_ds.core.values, keep_cores)) 
                or np.any(np.isin(dcc_ds.anvil.values, keep_anvils))
            ):
                dcc_ds = dcc_ds.get(var_list)
            
                dcc_ds = increment_step_coords(dcc_ds, dataset)
                
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
                    combine_attrs="drop", 
                )
            else:
                break

    # Update keep cores with cores linked to keep_anvils
    keep_cores = np.union1d(keep_cores, dataset.core.values[np.isin(dataset.core_anvil_index, keep_anvils)])
    keep_cores = np.setdiff1d(keep_cores, dataset.core.values[np.isin(dataset.core_anvil_index, drop_anvils)])

    # Select anvils and cores to keep
    dataset = sel_anvil(sel_core(dataset, keep_cores), keep_anvils)
    
    # Revert all dataarrays to original dtype after merging
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

    print(datetime.now(), "Flagging core and anvil quality", flush=True)
    dataset = remove_orphan_coords(dataset)
    dataset = add_validity_flags(dataset)

    print(f"Final core count: {dataset.core.size}")
    print(f"Final valid core count: {dataset.core_is_valid.data.sum()}")
    print(f"Final anvil count: {dataset.anvil.size}")
    print(f"Final valid thick anvil count: {dataset.thick_anvil_is_valid.data.sum()}")
    print(f"Final valid thin anvil count: {dataset.thin_anvil_is_valid.data.sum()}")

    # Add compression encoding
    print(datetime.now(), "Adding compression encoding", flush=True)
    dataset = add_compression_encoding(dataset, compression="zstd", complevel=5, shuffle=True)

    print(datetime.now(), "Saving to %s" % (save_path), flush=True)
    dataset.to_netcdf(save_path)

    print(datetime.now(), "Saving complete, closing datasets", flush=True)
    dataset.close()
