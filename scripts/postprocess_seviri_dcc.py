import numpy as np
import xarray as xr
import argparse
import pathlib
from datetime import datetime
from dateutil.parser import parse as parse_date
from tobac_flow.utils.legacy_utils import apply_weighted_func_to_labels
from tobac_flow.analysis import get_label_stats, weighted_statistics_on_labels
from tobac_flow.dataset import calculate_label_properties
from tobac_flow.utils.xarray_utils import (
    add_dataarray_to_ds,
    create_dataarray,
)

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
    file_path="/gws/nopw/j04/eo_shared_data_vol1/satellite/seviri-orac/cld",
)

cld_ds = xr.open_mfdataset(cld_files, combine="nested", concat_dim="t")

cld_ds = cld_ds.assign_coords(t=[parse_date(f[-64:-50]) for f in cld_files])

# Load flux file
flx_files = find_seviri_files(
    start_date,
    end_date,
    n_pad_files=0,
    file_type="flux",
    file_path="/gws/nopw/j04/eo_shared_data_vol1/satellite/seviri-orac/flx",
)

flx_ds = xr.open_mfdataset(flx_files, combine="nested", concat_dim="t")

flx_ds = flx_ds.assign_coords(t=[parse_date(f[-46:-34]) for f in flx_files])

dataset["lat"] = cld_ds.isel(t=0).lat.compute()
dataset["lon"] = cld_ds.isel(t=0).lon.compute()

# Add area of each pixel
def get_area_from_lat_lon(lat, lon):
    from pyproj import Geod

    g = Geod(ellps="WGS84")
    dy, dx = np.zeros(lat.shape, dtype=float), np.zeros(lat.shape, dtype=float)
    dy[:-1] = g.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])[-1] / 1e3
    dx[:, :-1] = g.inv(lon[:, :-1], lat[:, :-1], lon[:, 1:], lat[:, 1:])[-1] / 1e3
    dy[1:] += dy[:-1]
    dy[1:-1] /= 2
    dx[:, 1:] += dx[:, :-1]
    dx[:, 1:-1] /= 2
    area = dx * dy

    return area


areas = get_area_from_lat_lon(dataset.lat.data, dataset.lon.data)
add_dataarray_to_ds(
    create_dataarray(
        areas,
        dataset.lat.dims,
        "area",
        long_name="pixel area",
        units="km^2",
        dtype=np.float32,
    ),
    dataset,
)

calculate_label_properties(dataset)

if args.save_spatial_props:
    get_label_stats(dataset.core_label, dataset)
    get_label_stats(dataset.thick_anvil_label, dataset)
    get_label_stats(dataset.thin_anvil_label, dataset)

weights = np.repeat(dataset.area.data[np.newaxis, ...], dataset.t.size, 0)
print(datetime.now(), "Processing cloud properties", flush=True)
cld_weights = np.copy(weights)
cld_weights[cld_ds.qcflag.compute().data != 0] = 0

for field in (
    cld_ds.cot,
    cld_ds.cer,
    cld_ds.ctp,
    cld_ds.stemp,
    cld_ds.cth,
    cld_ds.ctt,
    cld_ds.cwp,
):
    # [
    #     add_dataarray_to_ds(da, dataset)
    #     for da in weighted_statistics_on_labels(
    #         dataset.core_label,
    #         field.compute(),
    #         cld_weights,
    #         name="core",
    #         dim="core",
    #         dtype=np.float32,
    #     )
    # ]

    # [
    #     add_dataarray_to_ds(da, dataset)
    #     for da in weighted_statistics_on_labels(
    #         dataset.thick_anvil_label,
    #         field.compute(),
    #         cld_weights,
    #         name="thick_anvil",
    #         dim="anvil",
    #         dtype=np.float32,
    #     )
    # ]

    # [
    #     add_dataarray_to_ds(da, dataset)
    #     for da in weighted_statistics_on_labels(
    #         dataset.thin_anvil_label,
    #         field.compute(),
    #         cld_weights,
    #         name="thin_anvil",
    #         dim="anvil",
    #         dtype=np.float32,
    #     )
    # ]

    [
        add_dataarray_to_ds(da[dataset.core_step.data - 1], dataset)
        for da in weighted_statistics_on_labels(
            dataset.core_step_label,
            field.compute(),
            cld_weights,
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
            cld_weights,
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
            cld_weights,
            name="thin_anvil_step",
            dim="thin_anvil_step",
            dtype=np.float32,
        )
    ]

ice_proportion = lambda x, w: np.nansum((x == 2) * w) / np.nansum((x > 0) * w)
core_step_ice_proportion = apply_weighted_func_to_labels(
    dataset.core_step_label.data,
    cld_ds.phase.compute().data,
    cld_weights,
    ice_proportion,
)[dataset.core_step.data - 1]

add_dataarray_to_ds(
    create_dataarray(
        core_step_ice_proportion,
        ("core_step",),
        "core_step_ice_proportion",
        long_name="proportion of core in ice phase",
        dtype=np.float32,
    ),
    dataset,
)

print(datetime.now(), "Processing flux properties", flush=True)
# toa_sw, toa_lw, toa_net
toa_net = flx_ds.toa_swdn - flx_ds.toa_swup - flx_ds.toa_lwup
toa_clr = flx_ds.toa_swdn - flx_ds.toa_swup_clr - flx_ds.toa_lwup_clr
toa_cre = toa_net - toa_clr
toa_net = create_dataarray(toa_net.data, flx_ds.dims, "toa_net", units="")
toa_cre = create_dataarray(toa_cre.data, flx_ds.dims, "toa_cre", units="")
toa_swup_cre = create_dataarray(
    flx_ds.toa_swup.data - flx_ds.toa_swup_clr,
    flx_ds.dims,
    "toa_swup_cre",
    units="",
)
toa_lwup_cre = create_dataarray(
    flx_ds.toa_lwup.data - flx_ds.toa_lwup_clr,
    flx_ds.dims,
    "toa_lwup_cre",
    units="",
)

for field in (
    flx_ds.toa_swdn,
    flx_ds.toa_swup,
    flx_ds.toa_lwup,
    toa_net,
    toa_swup_cre,
    toa_lwup_cre,
    toa_cre,
):
    # [
    #     add_dataarray_to_ds(da, dataset)
    #     for da in weighted_statistics_on_labels(
    #         dataset.core_label,
    #         field.compute(),
    #         cld_weights,
    #         name="core",
    #         dim="core",
    #         dtype=np.float32,
    #     )
    # ]

    # [
    #     add_dataarray_to_ds(da, dataset)
    #     for da in weighted_statistics_on_labels(
    #         dataset.thick_anvil_label,
    #         field.compute(),
    #         cld_weights,
    #         name="thick_anvil",
    #         dim="anvil",
    #         dtype=np.float32,
    #     )
    # ]

    # [
    #     add_dataarray_to_ds(da, dataset)
    #     for da in weighted_statistics_on_labels(
    #         dataset.thin_anvil_label,
    #         field.compute(),
    #         cld_weights,
    #         name="thin_anvil",
    #         dim="anvil",
    #         dtype=np.float32,
    #     )
    # ]

    [
        add_dataarray_to_ds(da[dataset.core_step.data - 1], dataset)
        for da in weighted_statistics_on_labels(
            dataset.core_step_label,
            field.compute(),
            cld_weights,
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
            cld_weights,
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
            cld_weights,
            name="thin_anvil_step",
            dim="thin_anvil_step",
            dtype=np.float32,
        )
    ]

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
