import numpy as np
import xarray as xr
import argparse
import pathlib
from datetime import datetime
from dateutil.parser import parse as parse_date
from tobac_flow.analysis import get_label_stats
from tobac_flow.dataset import calculate_label_properties
from tobac_flow.postprocess import (
    add_weighted_stats_to_dataset,
    add_cre_to_dataset,
    add_weighted_proportions_to_dataset,
)
from tobac_flow.utils import add_area_to_dataset

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

    save_name = fname.stem
    save_name = save_name + "_processed.nc"

    save_path = save_dir / save_name

    print("Saving to:", save_path)

    dataset = xr.open_dataset(fname)

    start_date = parse_date((str(fname)).split("_S")[-1].split("_E")[0], fuzzy=True)
    end_date = parse_date((str(fname)).split("_E")[-1].split("_X")[0], fuzzy=True)

    # Load cloud properties file
    from tobac_flow.dataloader import find_seviri_files

    cld_files = find_seviri_files(
        start_date,
        end_date,
        n_pad_files=0,
        file_type="cloud",
        file_path="/gws/nopw/j04/eo_shared_data_vol2/satellite/seviri-orac/cld",
    )

    cld_ds = xr.open_mfdataset(cld_files, combine="nested", concat_dim="t")

    cld_ds = cld_ds.assign_coords(t=[parse_date(f[-64:-50]) for f in cld_files])

    # Load flux file
    flx_files = find_seviri_files(
        start_date,
        end_date,
        n_pad_files=0,
        file_type="flux",
        file_path="/gws/nopw/j04/eo_shared_data_vol2/satellite/seviri-orac/flx",
    )

    flx_ds = xr.open_mfdataset(flx_files, combine="nested", concat_dim="t")

    flx_ds = flx_ds.assign_coords(t=[parse_date(f[-46:-34]) for f in flx_files])

    dataset["lat"] = cld_ds.isel(t=0).lat.compute()
    dataset["lon"] = cld_ds.isel(t=0).lon.compute()

    # Add area of each pixel
    dataset = add_area_to_dataset(dataset)

    calculate_label_properties(dataset)

    if args.save_spatial_props:
        get_label_stats(dataset.core_label, dataset)
        get_label_stats(dataset.thick_anvil_label, dataset)
        get_label_stats(dataset.thin_anvil_label, dataset)

    weights = np.repeat(dataset.area.data[np.newaxis, ...], dataset.t.size, 0)

    print(datetime.now(), "Processing cloud properties", flush=True)
    cld_weights = np.copy(weights)
    cld_weights[cld_ds.qcflag.to_numpy() != 0] = 0
    cld_weights[cld_ds.cth.to_numpy() > 30] = 0
    cld_weights = xr.DataArray(cld_weights, cld_ds.coords, cld_ds.dims)

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
        for dim in ("core_step", "thick_anvil_step", "thin_anvil_step"):
            dataset = add_weighted_stats_to_dataset(
                dataset, cld_ds, cld_weights, var, dim
            )

    for var in ("lsflag", "lusflag", "illum", "cldtype", "phase"):
        for dim in ("core_step", "thick_anvil_step", "thin_anvil_step"):
            dataset = add_weighted_proportions_to_dataset(
                dataset,
                cld_ds[var],
                weights,
                dim,
            )

    print(datetime.now(), "Processing flux properties", flush=True)
    flx_ds = add_cre_to_dataset(flx_ds)

    for var in (
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
        for dim in ("core_step", "thick_anvil_step", "thin_anvil_step"):
            dataset = add_weighted_stats_to_dataset(
                dataset, flx_ds, cld_weights, var, dim
            )

    print(datetime.now(), "Saving to %s" % (save_path), flush=True)
    # Add compression encoding
    comp = dict(zlib=True, complevel=5, shuffle=True)
    for var in dataset.data_vars:
        dataset[var].encoding.update(comp)

    dataset.to_netcdf(save_path)

    print(datetime.now(), "Saving complete, closing datasets", flush=True)

    dataset.close()
    cld_ds.close()
    flx_ds.close()


if __name__ == "__main__":
    main()
