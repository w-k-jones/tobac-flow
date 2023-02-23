import os
import numpy as np
from numpy import ma
import scipy as sp
from scipy import ndimage as ndi
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from glob import glob

from tobac_flow.dataset import add_dataarray_to_ds, create_dataarray

import argparse

parser = argparse.ArgumentParser(
    description="""Validate detected DCCs using GOES-16 GLM data"""
)
parser.add_argument("file", help="File to validate", type=str)
parser.add_argument(
    "-sd",
    help="Directory to save preprocess files",
    default="/gws/nopw/j04/eo_shared_data_vol1/satellite/seviri-orac/dcc_detect_cre_time_series/",
    type=str,
)
args = parser.parse_args()

fname = args.file

save_dir = args.sd
if not os.path.isdir(save_dir):
    try:
        os.makedirs(save_dir)
    except (FileExistsError, OSError):
        pass

save_name = fname.split("/")[-1]
save_name = save_name[:-3] + "_cre_time_series.nc"

print("Saving to:", save_path)

dcc_ds = xr.open_dataset(fname)

seviri_coords = {
    "t": dcc_ds.t,
    "along_track": dcc_ds.along_track,
    "across_track": dcc_ds.across_track,
}

dataset = xr.Dataset(coords=seviri_coords)

start_date = parse_date((fname).split("_S")[-1].split("_E")[0], fuzzy=True)
end_date = parse_date((fname).split("_E")[-1].split("_X")[0], fuzzy=True)

from tobac_flow.dataloader import find_seviri_files

# Load flux file
flx_files = find_seviri_files(
    start_date,
    end_date,
    n_pad_files=2,
    file_type="flux",
    file_path="/gws/nopw/j04/eo_shared_data_vol1/satellite/seviri-orac/flx",
)

flx_ds = xr.open_mfdataset(flx_files, combine="nested", concat_dim="t")

flx_ds = flx_ds.assign_coords(t=[parse_date(f[-46:-34]) for f in flx_files])


toa_sw_cre = (flx_ds.toa_swup_clr - flx_ds.toa_swup).compute()
toa_lw_cre = (flx_ds.toa_lwup_clr - flx_ds.toa_lwup).compute()
toa_net_cre = toa_sw_cre + toa_lw_cre

area_weights = (
    dcc_ds.area.data * np.ones(toa_net_cre.shape[0])[:, np.newaxis, np.newaxis]
)

dcc_mask = dcc_ds.thick_anvil_label == 0
non_dcc_mask = dcc_ds.thick_anvil_label != 0

total_area_time_series = np.sum(area_weights, axis=(1, 2))
total_dcc_area_time_series = ma.sum(
    ma.array(area_weights, mask=dcc_mask), axis=(1, 2)
).filled()
total_non_dcc_area_time_series = ma.sum(
    ma.array(area_weights, mask=non_dcc_mask), axis=(1, 2)
).filled()
add_dataarray_to_ds(
    create_dataarray(
        total_area_time_series,
        ("t",),
        "total_area_time_series",
        long_name="",
        dtype=np.float32,
    ),
    dataset,
)
add_dataarray_to_ds(
    create_dataarray(
        total_dcc_area_time_series,
        ("t",),
        "total_dcc_area_time_series",
        long_name="",
        dtype=np.float32,
    ),
    dataset,
)
add_dataarray_to_ds(
    create_dataarray(
        total_non_dcc_area_time_series,
        ("t",),
        "total_non_dcc_area_time_series",
        long_name="",
        dtype=np.float32,
    ),
    dataset,
)

t_cre_all = np.average(toa_net_cre.data, weights=area_weights, axis=(1, 2))
t_cre_dcc = np.average(
    ma.array(toa_net_cre.data, mask=dcc_mask),
    weights=ma.array(area_weights, mask=dcc_mask),
    axis=(1, 2),
)
t_cre_non_dcc = np.average(
    ma.array(toa_net_cre.data, mask=non_dcc_mask),
    weights=ma.array(area_weights, mask=non_dcc_mask),
    axis=(1, 2),
)
add_dataarray_to_ds(
    create_dataarray(t_cre_all, ("t",), "t_cre_all", long_name="", dtype=np.float32),
    dataset,
)
add_dataarray_to_ds(
    create_dataarray(t_cre_dcc, ("t",), "t_cre_dcc", long_name="", dtype=np.float32),
    dataset,
)
add_dataarray_to_ds(
    create_dataarray(
        t_cre_non_dcc, ("t",), "t_cre_non_dcc", long_name="", dtype=np.float32
    ),
    dataset,
)

t_sw_cre_all = np.average(toa_sw_cre.data, weights=area_weights, axis=(1, 2))
t_sw_cre_dcc = np.average(
    ma.array(toa_sw_cre.data, mask=dcc_mask),
    weights=ma.array(area_weights, mask=dcc_mask),
    axis=(1, 2),
)
t_sw_cre_non_dcc = np.average(
    ma.array(toa_sw_cre.data, mask=non_dcc_mask),
    weights=ma.array(area_weights, mask=non_dcc_mask),
    axis=(1, 2),
)
add_dataarray_to_ds(
    create_dataarray(
        t_sw_cre_all, ("t",), "t_sw_cre_all", long_name="", dtype=np.float32
    ),
    dataset,
)
add_dataarray_to_ds(
    create_dataarray(
        t_sw_cre_dcc, ("t",), "t_sw_cre_dcc", long_name="", dtype=np.float32
    ),
    dataset,
)
add_dataarray_to_ds(
    create_dataarray(
        t_sw_cre_non_dcc, ("t",), "t_sw_cre_non_dcc", long_name="", dtype=np.float32
    ),
    dataset,
)

t_lw_cre_all = np.average(toa_lw_cre.data, weights=area_weights, axis=(1, 2))
t_lw_cre_dcc = np.average(
    ma.array(toa_lw_cre.data, mask=dcc_mask),
    weights=ma.array(area_weights, mask=dcc_mask),
    axis=(1, 2),
)
t_lw_cre_non_dcc = np.average(
    ma.array(toa_lw_cre.data, mask=non_dcc_mask),
    weights=ma.array(area_weights, mask=non_dcc_mask),
    axis=(1, 2),
)
add_dataarray_to_ds(
    create_dataarray(
        t_lw_cre_all, ("t",), "t_lw_cre_all", long_name="", dtype=np.float32
    ),
    dataset,
)
add_dataarray_to_ds(
    create_dataarray(
        t_lw_cre_dcc, ("t",), "t_lw_cre_dcc", long_name="", dtype=np.float32
    ),
    dataset,
)
add_dataarray_to_ds(
    create_dataarray(
        t_lw_cre_non_dcc, ("t",), "t_lw_cre_non_dcc", long_name="", dtype=np.float32
    ),
    dataset,
)

print(datetime.now(), "Saving to %s" % (save_path), flush=True)
# Add compression encoding
comp = dict(zlib=True, complevel=5, shuffle=True)
for var in dataset.data_vars:
    dataset[var].encoding.update(comp)

dataset.to_netcdf(save_path)

dataset.close()
dcc_ds.close()
flx_ds.close()
