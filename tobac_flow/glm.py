"""
Tools for working with GLM data. Mostly adapted from glmtools/lmatools
"""
from datetime import datetime, timedelta
import warnings
import pathlib
import pandas as pd
import xarray as xr
import numpy as np

from tobac_flow import io
from tobac_flow.abi import get_abi_x_y
from tobac_flow.dataset import create_new_goes_ds
from tobac_flow.utils.xarray_utils import (
    add_dataarray_to_ds,
    create_dataarray,
    get_ds_bin_edges,
    get_ds_core_coords,
)
from tobac_flow.utils.datetime_utils import get_datetime_from_coord
from tobac_flow._lmatools import get_GOESR_coordsys_alt_ellps, get_GOESR_coordsys


def get_glm_parallax_offsets(lon, lat, goes_ds):
    # Get parallax of glm files to goes projection
    x, y = get_abi_x_y(lat, lon, goes_ds)
    z = np.zeros_like(x)

    nadir = goes_ds.goes_imager_projection.longitude_of_projection_origin

    _, grs80lla = get_GOESR_coordsys(nadir)
    geofix_ltg, _ = get_GOESR_coordsys_alt_ellps(nadir)

    lon_ltg, lat_ltg, _ = grs80lla.fromECEF(*geofix_ltg.toECEF(x, y, z))

    return lon_ltg - lon, lat_ltg - lat


def get_corrected_glm_x_y(glm_filename, goes_ds):
    try:
        # print(glm_filename, end="\r")
        with xr.open_dataset(glm_filename) as glm_ds:
            if glm_ds.flash_lat.data.size > 0 and glm_ds.flash_lon.data.size > 0:
                lon_offset, lat_offset = get_glm_parallax_offsets(
                    glm_ds.flash_lon.data, glm_ds.flash_lat.data, goes_ds
                )
                glm_lon = glm_ds.flash_lon.data - lon_offset
                glm_lat = glm_ds.flash_lat.data - lat_offset
                out = get_abi_x_y(glm_lat, glm_lon, goes_ds)
            else:
                out = (np.array([]), np.array([]))
    except (OSError, RuntimeError) as e:
        warnings.warn(e.args[0])
        warnings.warn(f"Unable to process file {glm_filename}")
        out = (np.array([]), np.array([]))
    return out


def get_uncorrected_glm_x_y(glm_filename, goes_ds):
    try:
        # print(glm_filename, end="\r")
        with xr.open_dataset(glm_filename) as glm_ds:
            if glm_ds.flash_lat.data.size > 0 and glm_ds.flash_lon.data.size > 0:
                glm_lon = glm_ds.flash_lon.data
                glm_lat = glm_ds.flash_lat.data
                out = get_abi_x_y(glm_lat, glm_lon, goes_ds)
            else:
                out = (np.array([]), np.array([]))
    except (OSError, RuntimeError) as e:
        warnings.warn(e.args[0])
        warnings.warn(f"Unable to process file {glm_filename}")
        out = (np.array([]), np.array([]))
    return out


def get_corrected_glm_hist(glm_files, goes_ds, start_time, end_time):
    x_bins, y_bins = get_ds_bin_edges(goes_ds, ("x", "y"))
    glm_x, glm_y = (
        np.concatenate(locs)
        for locs in zip(
            *[
                get_corrected_glm_x_y(glm_files[i], goes_ds)
                for i in glm_files
                if i > start_time and i < end_time
            ]
        )
    )
    return np.histogram2d(glm_y, glm_x, bins=(y_bins[::-1], x_bins))[0][::-1]


def get_uncorrected_glm_hist(glm_files, goes_ds, start_time, end_time):
    x_bins, y_bins = get_ds_bin_edges(goes_ds, ("x", "y"))
    glm_x, glm_y = (
        np.concatenate(locs)
        for locs in zip(
            *[
                get_uncorrected_glm_x_y(glm_files[i], goes_ds)
                for i in glm_files
                if i > start_time and i < end_time
            ]
        )
    )
    return np.histogram2d(glm_y, glm_x, bins=(y_bins[::-1], x_bins))[0][::-1]


def regrid_glm(glm_files, goes_ds, corrected=False, max_time_diff=15):
    """
    Regrid GLM flash observations to the ABI grid
    """
    # Max time diff 15 minutes away
    max_diff = max_time_diff * 60
    goes_dates = get_datetime_from_coord(goes_ds.t)
    time_diffs = [
        (goes_dates[i + 1] - goes_dates[i]).total_seconds()
        for i in range(len(goes_dates) - 1)
    ]
    time_diffs = [td / 2 if td < max_diff else max_diff / 2 for td in time_diffs]
    time_diffs = [time_diffs[0]] + time_diffs + [time_diffs[-1]]
    goes_coords = get_ds_core_coords(goes_ds)
    goes_mapping = {k: goes_coords[k].size for k in goes_coords}
    glm_grid_shape = (goes_mapping["t"], goes_mapping["y"], goes_mapping["x"])

    # Fill with -1 for missing value
    glm_grid = np.full(glm_grid_shape, -1)

    for i in range(glm_grid_shape[0]):
        # print(i, end='\r')
        start_time = goes_dates[i] - timedelta(seconds=time_diffs[i])
        end_time = goes_dates[i] + timedelta(seconds=time_diffs[i + 1])
        try:
            if corrected:
                glm_grid[i] = get_corrected_glm_hist(
                    glm_files, goes_ds, start_time, end_time
                )
            else:
                glm_grid[i] = get_uncorrected_glm_hist(
                    glm_files, goes_ds, start_time, end_time
                )
        except (ValueError, IndexError) as e:
            print("Error processing glm data at step %d" % i)
            print(e)

    glm_grid = xr.DataArray(glm_grid, goes_coords, ("t", "y", "x"))
    return glm_grid


def create_gridded_flash_ds(
    detection_ds, goes_data_path, save_ds=False, glm_save_path=None
):
    dates = get_datetime_from_coord(detection_ds.t)
    start_date = datetime(dates[0].year, dates[0].month, dates[0].day, dates[0].hour)
    end_date = datetime(dates[-1].year, dates[-1].month, dates[-1].day, dates[-1].hour)
    if (dates[-1] - end_date).total_seconds() > 0:
        end_date = end_date + timedelta(hours=1)
    dates = pd.date_range(
        start_date, dates[-1], freq="H", inclusive="left"
    ).to_pydatetime()

    if save_ds:
        if not isinstance(glm_save_path, pathlib.Path):
            glm_save_path = pathlib.Path(glm_save_path)

    gridded_flash_ds = create_new_goes_ds(detection_ds)

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
        glm_grid = regrid_glm(glm_files, gridded_flash_ds, corrected=True)

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
            np.nansum(glm_grid.data[glm_grid.data > 0]),
            tuple(),
            "glm_flash_count",
            long_name="total number of GLM flashes",
            dtype=np.int32,
        ),
        gridded_flash_ds,
    )

    if save_ds:
        # Add compression encoding
        comp = dict(zlib=True, complevel=5, shuffle=True)
        for var in gridded_flash_ds.data_vars:
            gridded_flash_ds[var].encoding.update(comp)

        print(datetime.now(), "Saving to %s" % (glm_save_path), flush=True)
        gridded_flash_ds.to_netcdf(glm_save_path)

    return gridded_flash_ds
