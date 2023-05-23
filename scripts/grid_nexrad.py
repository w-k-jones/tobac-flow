import os
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
import numpy as np
import pandas as pd
from tobac_flow.dataloader import goes_dataloader
from tobac_flow import nexrad, io
import argparse

parser = argparse.ArgumentParser(
    description="""Validate detected DCCs using GOES-16 GLM data"""
)
parser.add_argument("date", help="Date on which to start process", type=str)
parser.add_argument("hours", help="Number of hours to process", type=float)
parser.add_argument("-sat", help="GOES satellite", default=16, type=int)
parser.add_argument("-x0", help="Initial subset x location", default=0, type=int)
parser.add_argument("-x1", help="End subset x location", default=2500, type=int)
parser.add_argument("-y0", help="Initial subset y location", default=0, type=int)
parser.add_argument("-y1", help="End subset y location", default=1500, type=int)
parser.add_argument("-gd", help="GOES directory", default="../data/GOES16", type=str)
parser.add_argument(
    "-sd", help="Directory to save output files", default="../data/dcc_detect", type=str
)

args = parser.parse_args()
start_date = parse_date(args.date, fuzzy=True)
end_date = start_date + timedelta(hours=args.hours)

satellite = int(args.sat)
x0 = int(args.x0)
x1 = int(args.x1)
y0 = int(args.y0)
y1 = int(args.y1)
t_offset = 3

save_dir = args.sd
if not os.path.isdir(save_dir):
    try:
        os.makedirs(save_dir)
    except (FileExistsError, OSError):
        pass

save_name = "nexrad_regrid_G%02d_S%s_E%s_X%04d_%04d_Y%04d_%04d.nc" % (
    satellite,
    start_date.strftime("%Y%m%d_%H0000"),
    end_date.strftime("%Y%m%d_%H0000"),
    x0,
    x1,
    y0,
    y1,
)

save_path = os.path.join(save_dir, save_name)

print("Saving output to:", save_path)

nexrad_data_path = args.gd
if not os.path.isdir(nexrad_data_path):
    try:
        os.makedirs(nexrad_data_path)
    except (FileExistsError, OSError):
        pass

from tobac_flow import io, glm
from tobac_flow.dataset import (
    create_new_goes_ds,
    add_dataarray_to_ds,
    create_dataarray,
)

bt, _, _, dataset = goes_dataloader(
    start_date,
    end_date,
    n_pad_files=t_offset + 1,
    x0=x0,
    x1=x1,
    y0=y0,
    y1=y1,
    return_new_ds=True,
    satellite=satellite,
    product="MCMIP",
    view="C",
    mode=[3, 4, 6],
    save_dir=nexrad_data_path,
    replicate_path=True,
    check_download=True,
    n_attempts=1,
    download_missing=True,
)

dates = pd.date_range(start_date, end_date, freq="H", closed="left").to_pydatetime()

gridded_nexrad_ds = create_new_goes_ds(dataset)

print(datetime.now(), "Processing NEXRAD data", flush=True)
nexrad_sites = nexrad.filter_nexrad_sites(dataset, extend=0.001)
print("Number of sites in bound: %d" % len(nexrad_sites))
dates = pd.date_range(start_date, end_date, freq="h", inclusive="left")
nexrad_files = sum(
    [
        sum(
            [
                io.find_nexrad_files(
                    date, site, save_dir=nexrad_data_path, download_missing=True
                )
                for site in nexrad_sites
            ],
            [],
        )
        for date in dates
    ],
    [],
)
print(f"Loading {len(nexrad_files)} NEXRAD files")
# Regrid nexrad - note that this is a lengthly operation, expect it to take ~1 hour for the example here
# TODO; make pre-processed regridded nexrad file available.
ref_grid, ref_mask = nexrad.regrid_nexrad(nexrad_files, dataset, min_alt=1500)

add_dataarray_to_ds(
    create_dataarray(
        ref_grid.data,
        ("t", "y", "x"),
        "radar_reflectivity",
        long_name="Column mean of NEXRAD radar reflectivity",
        units="dBz",
        dtype=np.float32,
    ),
    gridded_nexrad_ds,
)

add_dataarray_to_ds(
    create_dataarray(
        np.sum(ref_mask.data),
        tuple(),
        "radar_mask",
        long_name="Mask of NEXRAD radar gates",
        dtype=bool,
    ),
    gridded_nexrad_ds,
)

print(datetime.now(), "Saving to %s" % (save_path), flush=True)
gridded_nexrad_ds.to_netcdf(save_path)
