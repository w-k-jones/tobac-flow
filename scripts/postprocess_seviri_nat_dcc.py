import numpy as np
import pandas as pd
import xarray as xr
import argparse
import pathlib
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from tobac_flow.analysis import get_label_stats, weighted_statistics_on_labels
from tobac_flow.dataset import calculate_label_properties
from tobac_flow.postprocess import (
    add_weighted_stats_to_dataset,
    add_cre_to_dataset,
    add_weighted_proportions_to_dataset,
)
from tobac_flow.utils.xarray_utils import add_dataarray_to_ds

parser = argparse.ArgumentParser(
    description="""Validate detected DCCs using GOES-16 GLM data"""
)
parser.add_argument("file", help="File to validate", type=str)
parser.add_argument("-sd", help="Directory to save preprocess files", default=None)
parser.add_argument(
    "--save_spatial_props",
    help="Save statistics of label spatial properties to output file",
    action="store_true",
)
args = parser.parse_args()


def main():
    fname = pathlib.Path(args.file)

    if args.sd is None:
        save_dir = pathlib.Path("./")
    else:
        save_dir = pathlib.Path(args.sd)
    if not save_dir.exists():
        save_dir.mkdir()

    save_path = save_dir / fname.name
    print("Saving to:", save_path, flush=True)

    print(datetime.now(), "Loading dataset from:", fname, flush=True)
    dataset = xr.open_dataset(fname)

    dataset = dataset.drop_vars(
        [
            coord
            for coord in dataset.coords
            if coord
            not in [
                "t",
                "y",
                "x",
                "core",
                "anvil",
                "core_step",
                "thick_anvil_step",
                "thin_anvil_step",
            ]
        ],
        errors="ignore",
    )

    start_date = parse_date((str(fname)).split("_S")[-1].split("_E")[0], fuzzy=True)
    end_date = parse_date((str(fname)).split("_E")[-1].split("_X")[0], fuzzy=True)

    print(datetime.now(), "Calculating label properties", flush=True)
    calculate_label_properties(dataset)

    if args.save_spatial_props:
        print(datetime.now(), "Calculating spatial properties", flush=True)
        get_label_stats(dataset.core_label, dataset)
        get_label_stats(dataset.thick_anvil_label, dataset)
        get_label_stats(dataset.thin_anvil_label, dataset)

    print(datetime.now(), "Calculating pixel weights", flush=True)
    weights = np.repeat(dataset.area.data[np.newaxis, ...], dataset.t.size, 0)

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
            along_track=slice(dataset.y[0].item(), dataset.y[-1].item() + 1),
            across_track=slice(dataset.x[0].item(), dataset.x[-1].item() + 1),
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
            along_track=slice(dataset.y[0].item(), dataset.y[-1].item() + 1),
            across_track=slice(dataset.x[0].item(), dataset.x[-1].item() + 1),
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

    print(datetime.now(), "Saving to %s" % (save_path), flush=True)
    # Add compression encoding
    comp = dict(zlib=True, complevel=5, shuffle=True)
    for var in dataset.data_vars:
        dataset[var].encoding.update(comp)

    dataset.to_netcdf(save_path)

    print(datetime.now(), "Saving complete, closing datasets", flush=True)

    dataset.close()
    flx_ds.close()


if __name__ == "__main__":
    main()
