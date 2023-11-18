import pathlib
from glob import glob
import os
import warnings
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
import numpy as np
import pandas as pd
import xarray as xr
import satpy

from tobac_flow import io
from tobac_flow.geo import get_pixel_area
from tobac_flow.utils.datetime_utils import get_datetime_from_coord
from tobac_flow.utils.xarray_utils import (
    create_dataarray,
    add_dataarray_to_ds,
)
from tobac_flow.abi import get_abi_lat_lon, get_abi_pixel_area


def goes_dataloader(
    start_date,
    end_date,
    n_pad_files=1,
    x0=None,
    x1=None,
    y0=None,
    y1=None,
    time_gap=timedelta(minutes=15),
    return_new_ds=False,
    **io_kwargs,
):
    """
    Load longwave brightness temperature, water vapour difference and split
        window difference from GOES-ABI data for DCC detection

    Parameters
    ----------
    start_date : datetime.datetime
        Initial date of ABI file to load
    end_data : datetime.datetime
        Final date of ABI files to load
    n_pad_files : int, optional (default : 1)
        Number of files to append on either side of the start_date and end_date
            to pad to resulting dataset
    x0 : int, optional (default : None)
        The initial x index to subset the dataset.
    x1 : int, optional (default : None)
        The final x index to subset the dataset.
    y0 : int, optional (default : None)
        The final y index to subset the dataset.
    y1 : int, optional (default : None)
        The final y index to subset the dataset.
    time_gap : datetime.timedelta, optional (default : timedelta(minutes=15))
        If the time between subsequent files is greater than this, try to fill
            these gaps using data from other scan regions. If this is not
            possible then an all-Nan slice will be inserted in the data at this
            point.
    return_new_ds : bool, optional (default : False)
        If True, returns a dataset of the same dimensions as the bt, wvd and swd
            data that includes dataarrays of latitude, longitude and area of
            each pixel. Note, if any time gaps have been filled with NaNs these
            will NOT be included in the t coord of the new dataset.
    **io_kwargs : optional
        Keywords to be passed to the io.find_goes_files function
    """
    abi_files = find_goes_files(start_date, end_date, n_pad_files, **io_kwargs)

    # Load ABI files
    bt, wvd, swd = load_mcmip(abi_files, x0, x1, y0, y1)

    # Check time is correct on all files, remove if incorrect
    pad_hours = int(np.ceil(n_pad_files / 12))
    padded_start_date = start_date - timedelta(hours=pad_hours)
    padded_end_date = end_date + timedelta(hours=pad_hours)

    datetime_coord = get_datetime_from_coord(bt.t)

    wh_valid_t = np.logical_and(
        [t > padded_start_date for t in datetime_coord],
        [t < padded_end_date for t in datetime_coord],
    )

    if not np.all(wh_valid_t):
        warnings.warn("Invalid time stamps found in ABI data, removing", RuntimeWarning)
        bt = bt[wh_valid_t]
        wvd = wvd[wh_valid_t]
        swd = swd[wh_valid_t]

    # Fill any gaps:
    if io_kwargs["view"] == "M":
        io_kwargs["view"] = "C"
        bt, wvd, swd = fill_time_gap_full_disk(
            bt,
            wvd,
            swd,
            start_date,
            end_date,
            n_pad_files,
            time_gap,
            x0,
            x1,
            y0,
            y1,
            **io_kwargs,
        )

    if io_kwargs["view"] == "C":
        io_kwargs["view"] = "F"
        bt, wvd, swd = fill_time_gap_full_disk(
            bt,
            wvd,
            swd,
            start_date,
            end_date,
            n_pad_files,
            time_gap,
            x0,
            x1,
            y0,
            y1,
            **io_kwargs,
        )

    if np.unique(bt.t).size < bt.t.size:
        raise RuntimeError("Duplicate time steps in input index values")

    if return_new_ds:
        goes_ds = xr.open_dataset(abi_files[0])

        goes_coords = {
            "t": bt.t,
            "y": bt.y,
            "x": bt.x,
            "y_image": goes_ds.y_image,
            "x_image": goes_ds.x_image,
        }

        new_ds = xr.Dataset(coords=goes_coords)
        new_ds["goes_imager_projection"] = goes_ds.goes_imager_projection
        lat, lon = get_abi_lat_lon(new_ds)
        add_dataarray_to_ds(
            create_dataarray(
                lat, ("y", "x"), "lat", long_name="latitude", dtype=np.float32
            ),
            new_ds,
        )
        add_dataarray_to_ds(
            create_dataarray(
                lon, ("y", "x"), "lon", long_name="longitude", dtype=np.float32
            ),
            new_ds,
        )
        add_dataarray_to_ds(
            create_dataarray(
                get_abi_pixel_area(new_ds),
                ("y", "x"),
                "area",
                long_name="pixel area",
                units="km^2",
                dtype=np.float32,
            ),
            new_ds,
        )

    bt = fill_time_gap_nan(bt, time_gap)
    wvd = fill_time_gap_nan(wvd, time_gap)
    swd = fill_time_gap_nan(swd, time_gap)

    print(f"Loaded {bt.t.size} time steps", flush=True)

    wvd.name = "WVD"
    wvd.attrs["standard_name"] = wvd.name
    wvd.attrs["long_name"] = "water vapour difference"
    wvd.attrs["units"] = "K"

    bt.name = "BT"
    bt.attrs["standard_name"] = bt.name
    bt.attrs["long_name"] = "brightness temperature"
    bt.attrs["units"] = "K"

    swd.name = "SWD"
    swd.attrs["standard_name"] = swd.name
    swd.attrs["long_name"] = "split window difference"
    swd.attrs["units"] = "K"

    if return_new_ds:
        return bt, wvd, swd, new_ds

    else:
        return bt, wvd, swd


def find_goes_files(start_date, end_date, n_pad_files=1, **io_kwargs):
    # Find ABI files
    dates = pd.date_range(
        start_date, end_date, freq="H", inclusive="left"
    ).to_pydatetime()

    abi_files = io.find_abi_files(dates, **io_kwargs)

    if n_pad_files > 0:
        pad_hours = int(np.ceil(n_pad_files / 12))

        pre_dates = pd.date_range(
            start_date - timedelta(hours=pad_hours),
            start_date,
            freq="H",
            inclusive="left",
        ).to_pydatetime()
        abi_pre_file = io.find_abi_files(pre_dates, **io_kwargs)
        if len(abi_pre_file):
            abi_pre_file = abi_pre_file[-n_pad_files:]

        post_dates = pd.date_range(
            end_date, end_date + timedelta(hours=pad_hours), freq="H", inclusive="left"
        ).to_pydatetime()
        abi_post_file = io.find_abi_files(post_dates, **io_kwargs)
        if len(abi_post_file):
            abi_post_file = abi_post_file[:n_pad_files]

        abi_files = abi_pre_file + abi_files + abi_post_file

    return abi_files


def get_stripe_deviation(da):
    y_mean = da.mean("y")
    y_std = da.std("y")
    return np.abs(((da - y_mean) / (y_std + 1e-8)).mean("x"))


def load_mcmip(files, x0=None, x1=None, y0=None, y1=None):
    ds_slice = {"x": slice(x0, x1), "y": slice(y0, y1)}
    # Load a stack of goes datasets using xarray
    print(f"Loading {len(files)} files", flush=True)
    goes_ds = xr.open_mfdataset(
        files,
        concat_dim="t",
        combine="nested",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    ).isel(ds_slice)
    # goes_ds = xr.concat(
    #     [xr.open_dataset(f).isel(ds_slice) for f in files],
    #     dim="t",
    #     data_vars="minimal",
    #     coords="minimal",
    #     compat="override",
    # )

    # Extract fields and load into memory
    wvd = (goes_ds.CMI_C08 - goes_ds.CMI_C10).load()

    bt = goes_ds.CMI_C13.load()

    swd = bt - goes_ds.CMI_C15.load()

    # Check for missing data and DQF flags in any channels, propagate to all data
    all_isnan = np.any([~np.isfinite(bt), ~np.isfinite(wvd), ~np.isfinite(swd)], 0)
    all_DQF = np.any(
        [goes_ds.DQF_C08, goes_ds.DQF_C10, goes_ds.DQF_C13, goes_ds.DQF_C15], 0
    )
    all_stripe = np.any(
        [
            get_stripe_deviation(goes_ds.DQF_C08) > 2,
            get_stripe_deviation(goes_ds.DQF_C10) > 2,
            get_stripe_deviation(goes_ds.DQF_C13) > 2,
            get_stripe_deviation(goes_ds.DQF_C15) > 2,
        ],
        0,
    )

    bt.data[all_isnan] = np.nan
    bt.data[all_DQF] = np.nan
    bt.data[all_stripe] = np.nan

    wvd.data[all_isnan] = np.nan
    wvd.data[all_DQF] = np.nan
    wvd.data[all_stripe] = np.nan

    swd.data[all_isnan] = np.nan
    swd.data[all_DQF] = np.nan
    swd.data[all_stripe] = np.nan

    # Sort by time
    if bt.t.size > 1:
        bt, wvd, swd = bt.sortby(bt.t), wvd.sortby(wvd.t), swd.sortby(swd.t)

    goes_ds.close()

    return bt, wvd, swd


def create_nan_slice(da, t_ind):
    slice_t = da.t[t_ind] + (da.t[t_ind + 1] - da.t[t_ind]) / 2
    print(f"Adding NaN slice at {slice_t.item()}", flush=True)
    nan_slice_da = xr.DataArray(
        np.full([1, da.y.size, da.x.size], np.nan),
        {
            "t": [slice_t.data],
            "y": da.y,
            "x": da.x,
            "y_image": 0,
            "x_image": 0,
        },
        ("t", "y", "x"),
    )
    return nan_slice_da


def fill_time_gap_nan(da, time_gap):
    where_time_gap = np.where(np.diff(get_datetime_from_coord(da.t)) > time_gap)[0]

    concat_list = []
    last_t_ind = 0

    if where_time_gap.size > 0:
        for t_ind in where_time_gap:
            concat_list.append(da.isel(t=slice(last_t_ind, t_ind + 1)))
            concat_list.append(create_nan_slice(da, t_ind))
            last_t_ind = t_ind + 1

        concat_list.append(da.isel(t=slice(last_t_ind, None)))

        return xr.concat(concat_list, "t")
    else:
        return da


def find_full_disk_for_time_gap(start_date, end_date, **io_kwargs):
    """
    Given a start date and an end date, find the ABI files that occur between
        the two dates
    """
    start_date = parse_date(start_date.astype("datetime64[s]").astype("str"))
    end_date = parse_date(end_date.astype("datetime64[s]").astype("str"))
    dates = pd.date_range(start_date, end_date, freq="H").to_pydatetime()
    #     io_kwargs["view"] = "F"
    F_files = io.find_abi_files(
        dates, **io_kwargs
    )  # , satellite=16, product='MCMIP', view='F', mode=[3,4,6],
    #                                 save_dir=goes_data_path,
    #                                 replicate_path=True, check_download=True,
    #                                 n_attempts=1, download_missing=True)

    F_dates = [io.get_goes_date(i) for i in F_files]

    return [
        file
        for file, date in zip(F_files, F_dates)
        if date > start_date and date < end_date
    ]


def fill_time_gap_full_disk(
    bt: xr.DataArray,
    wvd: xr.DataArray,
    swd: xr.DataArray,
    start_date: datetime,
    end_date: datetime,
    n_pad_files: int,
    time_gap: timedelta = timedelta(minutes=15),
    x0: int = None,
    x1: int = None,
    y0: int = None,
    y1: int = None,
    **io_kwargs,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    pad_hours = int(np.ceil(n_pad_files / 12))
    padded_start_date = start_date - timedelta(hours=pad_hours)
    padded_end_date = end_date + timedelta(hours=pad_hours)

    # Pad dates list if we are missing pad values
    dates = get_datetime_from_coord(bt.t)
    if np.sum(np.array(dates) < start_date) < n_pad_files:
        dates = [padded_start_date] + dates
    else:
        dates = [dates[0]] + dates

    if np.sum(np.array(dates) > end_date) < n_pad_files:
        dates = dates + [padded_end_date]
    else:
        dates = dates + [dates[-1]]

    where_time_gap = np.where(np.diff(dates) > time_gap)[0]

    bt_concat_list = []
    wvd_concat_list = []
    swd_concat_list = []
    last_t_ind = 0

    if x0:
        x0 += 902
    else:
        x0 = 902
    if x1:
        x1 += 902
    else:
        x1 = 902 + 2500
    if y0:
        y0 += 422
    else:
        y0 = 422
    if y1:
        y1 += 422
    else:
        y1 = 422 + 1500

    if where_time_gap.size > 0:
        for t_ind in where_time_gap:
            print(
                f"Filling time gap between {dates[t_ind].isoformat()} and {dates[t_ind + 1].isoformat()}",
                flush=True,
            )
            full_disk_files = find_full_disk_for_time_gap(
                np.datetime64(dates[t_ind]),
                np.datetime64(dates[t_ind + 1]),
                **io_kwargs,
            )
            if t_ind > 0 and t_ind <= bt.t.size and last_t_ind < bt.t.size:
                bt_concat_list.append(bt.isel(t=slice(last_t_ind, t_ind)))
                wvd_concat_list.append(wvd.isel(t=slice(last_t_ind, t_ind)))
                swd_concat_list.append(swd.isel(t=slice(last_t_ind, t_ind)))

            if len(full_disk_files) > 0:
                full_bt, full_wvd, full_swd = load_mcmip(
                    full_disk_files, x0, x1, y0, y1
                )

                new_coords = {
                    "t": full_bt.t.data,
                    "y": bt.y,
                    "x": bt.x,
                    "y_image": bt.y_image,
                    "x_image": bt.x_image,
                }

                full_bt = xr.DataArray(full_bt.data, new_coords, ("t", "y", "x"))
                full_wvd = xr.DataArray(full_wvd.data, new_coords, ("t", "y", "x"))
                full_swd = xr.DataArray(full_swd.data, new_coords, ("t", "y", "x"))

                bt_concat_list.append(full_bt)
                wvd_concat_list.append(full_wvd)
                swd_concat_list.append(full_swd)

            last_t_ind = t_ind

        if last_t_ind < bt.t.size:
            bt_concat_list.append(bt.isel(t=slice(last_t_ind, None)))
            wvd_concat_list.append(wvd.isel(t=slice(last_t_ind, None)))
            swd_concat_list.append(swd.isel(t=slice(last_t_ind, None)))

        bt = xr.concat(bt_concat_list, "t", join="left")
        wvd = xr.concat(wvd_concat_list, "t", join="left")
        swd = xr.concat(swd_concat_list, "t", join="left")

        datetime_coord = get_datetime_from_coord(bt.t)

        wh_valid_t = np.logical_and(
            [t > padded_start_date for t in datetime_coord],
            [t < padded_end_date for t in datetime_coord],
        )

        if not np.all(wh_valid_t):
            warnings.warn(
                "Invalid time stamps found in ABI data, removing", RuntimeWarning
            )
            bt = bt[wh_valid_t]
            wvd = wvd[wh_valid_t]
            swd = swd[wh_valid_t]

        dates = get_datetime_from_coord(bt.t)
        pre_dates = np.sum(np.array(dates) < start_date)
        extra_pre_steps = pre_dates - n_pad_files if pre_dates > n_pad_files else None

        post_dates = np.sum(np.array(dates) > end_date)
        extra_post_steps = (
            -(post_dates - n_pad_files) if post_dates > n_pad_files else None
        )

        if extra_pre_steps is not None or extra_post_steps is not None:
            print("Trimming excess time steps", flush=True)
            bt = bt.isel(t=slice(extra_pre_steps, extra_post_steps))
            wvd = wvd.isel(t=slice(extra_pre_steps, extra_post_steps))
            swd = swd.isel(t=slice(extra_pre_steps, extra_post_steps))

    return bt, wvd, swd


def glob_seviri_files(
    start_date, end_date, file_type="secondary", file_path="../data/SEVIRI_ORAC/"
):
    if file_type not in ["secondary", "cloud", "flux"]:
        raise ValueError(
            "file_type parameter must be one of 'secondary', 'cloud' or 'flux'"
        )

    dates = pd.date_range(
        start_date, end_date, freq="H", inclusive="left"
    ).to_pydatetime()

    seviri_files = []

    for date in dates:
        datestr = date.strftime("%Y%m%d%H")
        if file_type == "secondary":
            glob_str = f"H-000-MSG3__-MSG3________-_________-EPI______-{datestr}*-__.secondary.nc"
        if file_type == "cloud":
            glob_str = f"{datestr}*00-ESACCI-L2_CLOUD-CLD_PRODUCTS-SEVIRI-MSG3-fv1.0.nc"
        if file_type == "flux":
            glob_str = f"{datestr}*00-ESACCI-TOA-SEVIRI-MSG3-fv1.0.nc"
        seviri_files.extend(glob(os.path.join(file_path, glob_str)))

    return sorted(seviri_files)


def find_seviri_files(
    start_date,
    end_date,
    n_pad_files=1,
    file_type="secondary",
    file_path="../data/SEVIRI_ORAC/",
):
    seviri_files = glob_seviri_files(start_date, end_date, file_type, file_path)

    if n_pad_files > 0:
        pad_hours = int(np.ceil(n_pad_files / 4))

        seviri_pre_file = glob_seviri_files(
            start_date - timedelta(hours=pad_hours), start_date, file_type, file_path
        )
        if len(seviri_pre_file):
            seviri_pre_file = seviri_pre_file[-n_pad_files:]

        seviri_post_file = glob_seviri_files(
            end_date, end_date + timedelta(hours=pad_hours), file_type, file_path
        )
        if len(seviri_post_file):
            seviri_post_file = seviri_post_file[:n_pad_files]

        seviri_files = seviri_pre_file + seviri_files + seviri_post_file

    return seviri_files


def load_seviri_dataset(seviri_files, x0=None, x1=None, y0=None, y1=None):
    seviri_ds = xr.open_mfdataset(seviri_files, combine="nested", concat_dim="t").isel(
        across_track=slice(x0, x1), along_track=slice(y0, y1)
    )

    seviri_ds = seviri_ds.assign_coords(
        t=[parse_date(f[-28:-16]) for f in seviri_files]
    )

    return seviri_ds


def seviri_dataloader(
    start_date,
    end_date,
    n_pad_files=1,
    file_path="../data/SEVIRI_ORAC/",
    x0=None,
    x1=None,
    y0=None,
    y1=None,
    time_gap=timedelta(minutes=30),
    return_new_ds=False,
):
    """
    Load longwave brightness temperature, water vapour difference and split
        window difference from Meteosat seviri_files data for DCC detection

    Parameters
    ----------
    start_date : datetime.datetime
        Initial date of SEVIRI file to load
    end_data : datetime.datetime
        Final date of SEVIRI files to load
    n_pad_files : int, optional (default : 1)
        Number of files to append on either side of the start_date and end_date
            to pad to resulting dataset
    file_path : string, optional (default : "../data/SEVIRI_ORAC/")
        The path to the directory in which the SEVIRI secondary files are stored
    x0 : int, optional (default : None)
        The initial x index to subset the dataset.
    x1 : int, optional (default : None)
        The final x index to subset the dataset.
    y0 : int, optional (default : None)
        The final y index to subset the dataset.
    y1 : int, optional (default : None)
        The final y index to subset the dataset.
    time_gap : datetime.timedelta, optional (default : timedelta(minutes=15))
        If the time between subsequent files is greater than this, try to fill
            these gaps using data from other scan regions. If this is not
            possible then an all-Nan slice will be inserted in the data at this
            point.
    return_new_ds : bool, optional (default : False)
        If True, returns a dataset of the same dimensions as the bt, wvd and swd
            data that includes dataarrays of latitude, longitude and area of
            each pixel. Note, if any time gaps have been filled with NaNs these
            will NOT be included in the t coord of the new dataset.
    """
    seviri_files = find_seviri_files(
        start_date, end_date, n_pad_files=n_pad_files, file_path=file_path
    )

    seviri_ds = load_seviri_dataset(seviri_files, x0, x1, y0, y1)

    bt = seviri_ds.brightness_temperature_in_channel_no_9.load()

    wvd = (
        seviri_ds.brightness_temperature_in_channel_no_5
        - seviri_ds.brightness_temperature_in_channel_no_6
    ).load()

    swd = bt - seviri_ds.brightness_temperature_in_channel_no_10.load()

    all_isnan = np.any([~np.isfinite(bt), ~np.isfinite(wvd), ~np.isfinite(swd)], 0)

    bt.data[all_isnan] = np.nan
    wvd.data[all_isnan] = np.nan
    swd.data[all_isnan] = np.nan

    if return_new_ds:
        seviri_coords = {
            "t": bt.t,
            "along_track": bt.along_track,
            "across_track": bt.across_track,
        }

        new_ds = xr.Dataset(coords=seviri_coords)

    bt = fill_time_gap_nan(bt, time_gap)
    wvd = fill_time_gap_nan(wvd, time_gap)
    swd = fill_time_gap_nan(swd, time_gap)

    wvd.name = "WVD"
    wvd.attrs["standard_name"] = wvd.name
    wvd.attrs["long_name"] = "water vapour difference"
    wvd.attrs["units"] = "K"

    bt.name = "BT"
    bt.attrs["standard_name"] = bt.name
    bt.attrs["long_name"] = "brightness temperature"
    bt.attrs["units"] = "K"

    swd.name = "SWD"
    swd.attrs["standard_name"] = swd.name
    swd.attrs["long_name"] = "split window difference"
    swd.attrs["units"] = "K"

    seviri_ds.close()

    if return_new_ds:
        return bt, wvd, swd, new_ds
    else:
        return bt, wvd, swd

def glob_seviri_nat_files(
    start_date, end_date, satellite=None, file_path=pathlib.Path("../data/seviri/")
):
    if satellite is None:
        satellite = "[1234]"
    elif satellite not in [1,2,3,4, "1", "2", "3", "4"]:
        raise ValueError("satellite keywrod must be one of '1', '2', '3', '4'")

    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)
    elif not isinstance(file_path, pathlib.Path):
        raise ValueError("file_path must be either a string or a Path object")
    
    dates = pd.date_range(
        start_date, end_date, freq="H", inclusive="left"
    ).to_pydatetime()

    seviri_files = []

    for date in dates:
        datestr = date.strftime("%Y%m%d%H")
        glob_str = f"MSG{satellite}-SEVI-MSG*-NA-{datestr}*-NA.nat"
        seviri_files.extend(list((file_path / date.strftime("%Y/%m/%d")).glob(glob_str)))

    return sorted(seviri_files)


def find_seviri_nat_files(
    start_date,
    end_date,
    n_pad_files=1,
    satellite=None, 
    file_path=pathlib.Path("../data/seviri/"),
):
    seviri_files = glob_seviri_nat_files(start_date, end_date, satellite, file_path)

    if n_pad_files > 0:
        pad_hours = int(np.ceil(n_pad_files / 4))

        seviri_pre_file = glob_seviri_nat_files(
            start_date - timedelta(hours=pad_hours), start_date, satellite, file_path
        )
        if len(seviri_pre_file):
            seviri_pre_file = seviri_pre_file[-n_pad_files:]

        seviri_post_file = glob_seviri_nat_files(
            end_date, end_date + timedelta(hours=pad_hours), satellite, file_path
        )
        if len(seviri_post_file):
            seviri_post_file = seviri_post_file[:n_pad_files]

        seviri_files = seviri_pre_file + seviri_files + seviri_post_file

    return seviri_files

def get_seviri_nat_date_from_filename(filename):
    if isinstance(filename, pathlib.Path):
        filename = filename.name
    elif isinstance(filename, str):
        filename = filename.split("/")[-1]

    date = datetime.strptime(filename[24:38], '%Y%m%d%H%M%S')
    return date

def seviri_nat_dataloader(
    start_date,
    end_date,
    n_pad_files=1,
    satellite=None,
    file_path=pathlib.Path("../data/seviri/"),
    x0=None,
    x1=None,
    y0=None,
    y1=None,
    time_gap=timedelta(minutes=30),
    return_new_ds=False,
):
    files = find_seviri_nat_files(
        start_date,
        end_date,
        n_pad_files=n_pad_files,
        satellite=satellite, 
        file_path=file_path,
    )

    scn = satpy.Scene(reader="seviri_l1b_native", filenames=files)

    scn.load(["WV_062", "WV_073", "IR_087", "IR_108", "IR_120"])

    ds = scn.to_xarray()

    ds = ds.coarsen(y=ds.x.size).construct(y=("t", "y"))

    dates = [get_seviri_nat_date_from_filename(f) for f in files]

    ds.coords["t"] = ("t", dates)

    bt = ds.IR_108.load()
    wvd = (ds.WV_062 - ds.WV_073).load()
    twd = (ds.IR_087 - ds.IR_120).load()
    twd = np.maximum(twd, 0)

    all_isnan = np.any([~np.isfinite(bt), ~np.isfinite(wvd), ~np.isfinite(twd)], 0)

    bt.data[all_isnan] = np.nan
    wvd.data[all_isnan] = np.nan
    twd.data[all_isnan] = np.nan

    if return_new_ds:
        seviri_coords = {
            "t": bt.t,
            "y": bt.y,
            "x": bt.x,
        }

        new_ds = xr.Dataset(coords=seviri_coords)

        add_dataarray_to_ds(
            create_dataarray(
                ds.latitude.values[0], ("y", "x"), "lat", long_name="latitude", dtype=np.float32
            ),
            new_ds,
        )
        add_dataarray_to_ds(
            create_dataarray(
                ds.longitude.values[0], ("y", "x"), "lon", long_name="longitude", dtype=np.float32
            ),
            new_ds,
        )

        area = get_pixel_area(ds.latitude.values, ds.longitude.values)

        add_dataarray_to_ds(
            create_dataarray(
                area,
                ("y", "x"),
                "area",
                long_name="pixel area",
                units="km^2",
                dtype=np.float32,
            ),
            new_ds,
        )
    
    bt = fill_time_gap_nan(bt, time_gap)
    wvd = fill_time_gap_nan(wvd, time_gap)
    twd = fill_time_gap_nan(twd, time_gap)

    print(f"Loaded {bt.t.size} time steps", flush=True)

    wvd.name = "WVD"
    wvd.attrs["standard_name"] = wvd.name
    wvd.attrs["long_name"] = "water vapour difference"
    wvd.attrs["units"] = "K"

    bt.name = "BT"
    bt.attrs["standard_name"] = bt.name
    bt.attrs["long_name"] = "brightness temperature"
    bt.attrs["units"] = "K"

    twd.name = "TWD"
    twd.attrs["standard_name"] = twd.name
    twd.attrs["long_name"] = "two window difference"
    twd.attrs["units"] = "K"

    if return_new_ds:
        return bt, wvd, twd, new_ds

    else:
        return bt, wvd, twd