import os
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
import numpy as np
import pandas as pd
from tobac_flow.dataloader import goes_dataloader

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

save_name = "glm_regrid_G%02d_S%s_E%s_X%04d_%04d_Y%04d_%04d.nc" % (
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

goes_data_path = args.gd
if not os.path.isdir(goes_data_path):
    try:
        os.makedirs(goes_data_path)
    except (FileExistsError, OSError):
        pass

# def validation(file, margin, goes_data_path, save_dir):
if True:
    """
    Validation process for detected DCCs in the given file
    """
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
        save_dir=goes_data_path,
        replicate_path=True,
        check_download=True,
        n_attempts=1,
        download_missing=True,
    )

    dates = pd.date_range(start_date, end_date, freq="H", closed="left").to_pydatetime()

    glm_save_name = "gridded_glm_flashes_%s.nc" % (dates[0].strftime("%Y%m%d_%H0000"))
    glm_save_path = os.path.join(save_dir, glm_save_name)
    validation_save_name = "validation_dccs_%s.nc" % (
        dates[0].strftime("%Y%m%d_%H0000")
    )
    validation_save_path = os.path.join(save_dir, validation_save_name)

    """
    Start validation
    """
    gridded_flash_ds = create_new_goes_ds(dataset)

    print(datetime.now(), "Processing GLM data", flush=True)
    # Get GLM data
    # Process new GLM data
    glm_files = io.find_glm_files(
        dates,
        satellite=16,
        save_dir=goes_data_path,
        replicate_path=True,
        check_download=True,
        n_attempts=1,
        download_missing=True,
        verbose=False,
        min_storage=2**30,
    )
    glm_files = {io.get_goes_date(i): i for i in glm_files}
    print("%d files found" % len(glm_files), flush=True)
    if len(glm_files) == 0:
        raise ValueError("No GLM Files discovered, skipping validation")
    else:
        print(datetime.now(), "Regridding GLM data", flush=True)
        glm_grid = glm.regrid_glm(glm_files, gridded_flash_ds, corrected=True)

    add_dataarray_to_ds(
        create_dataarray(
            glm_grid.data,
            ("t", "y", "x"),
            "glm_flashes",
            long_name="number of flashes detected by GLM",
            units="",
            dtype=np.int32,
        ),
        gridded_flash_ds,
    )

    add_dataarray_to_ds(
        create_dataarray(
            np.sum(glm_grid.data),
            tuple(),
            "glm_flash_count",
            long_name="total number of GLM flashes",
            dtype=np.int32,
        ),
        gridded_flash_ds,
    )

    print(datetime.now(), "Saving to %s" % (glm_save_path), flush=True)
    gridded_flash_ds.to_netcdf(glm_save_path)
