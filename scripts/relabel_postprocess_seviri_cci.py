import argparse
import pathlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

from tobac_flow.analysis import get_label_stats, weighted_statistics_on_labels
from tobac_flow.dataset import calculate_label_properties
from tobac_flow.linking import process_file
from tobac_flow.utils.datetime_utils import get_dates_from_filename
from tobac_flow.utils.xarray_utils import add_compression_encoding, add_dataarray_to_ds
from tobac_flow.postprocess import (
    add_weighted_stats_to_dataset,
    add_cre_to_dataset,
    add_weighted_proportions_to_dataset,
)

parser = argparse.ArgumentParser(
    description="""Postprocess detected DCCs using GOES-16 data"""
)
parser.add_argument("file", help="File to validate", type=str)
parser.add_argument("links_file", help="Link file containing new labels for each file", type=str)

parser.add_argument("-sd", help="Directory to save preprocess files", default="")
parser.add_argument(
    "-sdf", help="Date formatting string for subdirectories", default=""
)
parser.add_argument(
    "--save_spatial_props",
    help="Save statistics of label spatial properties to output file",
    action="store_true",
)
if __name__ == "__main__":
    args = parser.parse_args()
    
    filename = pathlib.Path(args.file)
    assert filename.exists(), f'File {filename} not found'

    start_date, end_date = get_dates_from_filename(filename)
    x0, x1 = [int(s) for s in filename.name.split("_X")[-1][:9].split("_")]
    y0, y1 = [int(s) for s in filename.name.split("_Y")[-1][:9].split("_")]
    
    save_path = pathlib.Path(args.sd)
    if args.sdf:
        save_path = save_path / start_date.strftime(args.sdf)
    
    save_path.mkdir(parents=True, exist_ok=True)

    save_path = save_path / filename.name
    
    with xr.open_dataset(args.links_file) as links_ds:
        dataset = process_file(args.file, links_ds)
    
    print(datetime.now(), "Calculating label properties", flush=True)
    calculate_label_properties(dataset)

    if args.save_spatial_props:
        print(datetime.now(), "Calculating spatial properties", flush=True)
        get_label_stats(dataset.core_label, dataset)
        get_label_stats(dataset.thick_anvil_label, dataset)
        get_label_stats(dataset.thin_anvil_label, dataset)

    weights = np.repeat(dataset.area.data[np.newaxis, ...], dataset.t.size, 0)

    print(datetime.now(), "Calculating statistics", flush=True)
    if "bt" in dataset.data_vars:
        for field in (dataset.bt,):
            [
                add_dataarray_to_ds(da[dataset.core_step.data - 1], dataset)
                for da in weighted_statistics_on_labels(
                    dataset.core_step_label,
                    field.compute(),
                    weights,
                    name="core_step",
                    dim="core_step",
                    dtype=np.float32,
                )
            ]

            [
                add_dataarray_to_ds(da[dataset.thick_anvil_step.data - 1], dataset)
                for da in weighted_statistics_on_labels(
                    dataset.thick_anvil_step_label,
                    field.compute(),
                    weights,
                    name="thick_anvil_step",
                    dim="thick_anvil_step",
                    dtype=np.float32,
                )
            ]

            [
                add_dataarray_to_ds(da[dataset.thin_anvil_step.data - 1], dataset)
                for da in weighted_statistics_on_labels(
                    dataset.thin_anvil_step_label,
                    field.compute(),
                    weights,
                    name="thin_anvil_step",
                    dim="thin_anvil_step",
                    dtype=np.float32,
                )
            ]

    # Remove BT to reduce file size
    dataset = dataset.drop_vars("bt")

    if "BT" in dataset.data_vars:
        for field in (dataset.BT,):
            [
                add_dataarray_to_ds(da[dataset.core_step.data - 1], dataset)
                for da in weighted_statistics_on_labels(
                    dataset.core_step_label,
                    field.compute(),
                    weights,
                    name="core_step",
                    dim="core_step",
                    dtype=np.float32,
                )
            ]

            [
                add_dataarray_to_ds(da[dataset.thick_anvil_step.data - 1], dataset)
                for da in weighted_statistics_on_labels(
                    dataset.thick_anvil_step_label,
                    field.compute(),
                    weights,
                    name="thick_anvil_step",
                    dim="thick_anvil_step",
                    dtype=np.float32,
                )
            ]

            [
                add_dataarray_to_ds(da[dataset.thin_anvil_step.data - 1], dataset)
                for da in weighted_statistics_on_labels(
                    dataset.thin_anvil_step_label,
                    field.compute(),
                    weights,
                    name="thin_anvil_step",
                    dim="thin_anvil_step",
                    dtype=np.float32,
                )
            ]

        # Remove BT to reduce file size
        dataset = dataset.drop_vars("BT")

    # Load cloud properties file
    print(datetime.now(), "Loading cloud properties", flush=True)
    dates = pd.date_range(
        start_date, end_date, freq="H", inclusive="left"
    ).to_pydatetime()

    seviri_file_path = pathlib.Path(
        "/gws/nopw/j04/eo_shared_data_vol1/satellite/seviri-orac/Data"
    )

    cld_files = sorted(
        sum(
            [
                list(
                    (
                        seviri_file_path
                        / date.strftime("%Y")
                        / date.strftime("%m")
                        / date.strftime("%d")
                        / date.strftime("%H")
                    ).glob(
                        f"{date.strftime('%Y%m%d%H')}[0-9][0-9]00-ESACCI-L2_CLOUD-CLD_PRODUCTS-SEVIRI-MSG[1-4]-fv3.0.nc"
                    )
                )
                for date in dates
            ],
            [],
        )
    )
    print("Files found:", len(cld_files), flush=True)

    cld_ds = (
        xr.open_mfdataset(
            cld_files, combine="nested", concat_dim="t", decode_times=False
        )
        .squeeze()
        .isel(
            along_track=slice(y0, y1 + 1),
            across_track=slice(x0, x1 + 1),
        )
    )

    # Add time coord to cld_ds
    cld_dates = [
        datetime.strptime(f.name[:14], "%Y%m%d%H%M%S")
        + timedelta(minutes=12, seconds=42)
        for f in cld_files
    ]
    cld_ds.coords["t"] = cld_dates
    cld_ds = cld_ds.reindex(
        t=dataset.t,
        method="nearest",
        tolerance=timedelta(minutes=1),
        fill_value={
            var: (np.nan if np.issubdtype(cld_ds[var].dtype, np.floating) else 0)
            for var in cld_ds.data_vars
        },
    )

    print(datetime.now(), "Processing cloud properties", flush=True)
    weights = (
        weights
        * (cld_ds.qcflag == 0).astype(np.float32)
        * (cld_ds.cth < 30).astype(np.float32)
        * np.isfinite(cld_ds.cot).astype(np.float32)
    )
    weights = xr.DataArray(weights, cld_ds.lat.coords, cld_ds.lat.dims)
    weights = weights.compute()

    for var in (
        "cot",
        "cer",
        "cwp",
        "ctp",
        "ctp_corrected",
        "cth",
        "cth_corrected",
        "ctt",
        "ctt_corrected",
        "stemp",
        "dem",
    ):
        print(f"Processing {var}", flush=True)
        for dim in ("core_step", "thick_anvil_step", "thin_anvil_step"):
            dataset = add_weighted_stats_to_dataset(dataset, cld_ds, weights, var, dim)

    for var in ("lsflag", "lusflag", "illum", "cldtype", "phase"):
        print(f"Processing {var}", flush=True)
        for dim in ("core_step", "thick_anvil_step", "thin_anvil_step"):
            dataset = add_weighted_proportions_to_dataset(
                dataset,
                cld_ds[var],
                weights,
                dim,
            )

    cld_ds.close()
    del cld_ds

    # Load flux file
    print(datetime.now(), "Loading flux properties", flush=True)

    flx_files = sorted(
        sum(
            [
                list(
                    (
                        seviri_file_path
                        / date.strftime("%Y")
                        / date.strftime("%m")
                        / date.strftime("%d")
                        / date.strftime("%H")
                    ).glob(
                        f"{date.strftime('%Y%m%d%H')}[0-9][0-9]00-ESACCI-TOA-SEVIRI-MSG[1-4]-fv3.0.nc"
                    )
                )
                for date in dates
            ],
            [],
        )
    )

    flx_ds = (
        xr.open_mfdataset(
            flx_files, combine="nested", concat_dim="t", decode_times=False
        )
        .squeeze()
        .isel(
            along_track=slice(y0, y1 + 1),
            across_track=slice(x0, x1 + 1),
        )
    )

    # Add time coord to flx_ds
    flx_dates = [
        datetime.strptime(f.name[:14], "%Y%m%d%H%M%S")
        + timedelta(minutes=12, seconds=42)
        for f in flx_files
    ]
    flx_ds.coords["t"] = flx_dates
    flx_ds = flx_ds.reindex(
        t=dataset.t,
        method="nearest",
        tolerance=timedelta(minutes=1),
        fill_value={
            var: (np.nan if np.issubdtype(flx_ds[var].dtype, np.floating) else 0)
            for var in flx_ds.data_vars
        },
    )

    print(datetime.now(), "Processing flux properties", flush=True)
    flx_ds = add_cre_to_dataset(flx_ds)

    for var in (
        "toa_swdn",
        "toa_swup",
        "toa_swup_cre",
        "toa_lwup",
        "toa_lwup_cre",
        "toa_net",
        "toa_net_cre",
        "boa_swdn",
        "boa_swdn_cre",
        "boa_swup",
        "boa_swup_cre",
        "boa_lwdn",
        "boa_lwdn_cre",
        "boa_lwup",
        "boa_lwup_cre",
        "boa_net",
        "boa_net_cre",
        "lts",
        "fth",
        "cbh",
    ):
        print(f"Processing {var}", flush=True)
        for dim in ("core_step", "thick_anvil_step", "thin_anvil_step"):
            dataset = add_weighted_stats_to_dataset(dataset, flx_ds, weights, var, dim)


    # Add compression encoding
    print(datetime.now(), "Adding compression encoding", flush=True)
    dataset = add_compression_encoding(dataset, compression="zstd", complevel=5, shuffle=True)

    print(datetime.now(), "Saving to %s" % (save_path), flush=True)
    dataset.to_netcdf(save_path)

    dataset.close()
