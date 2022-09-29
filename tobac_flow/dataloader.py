import numpy as np
import pandas as pd
import xarray as xr
import os
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from glob import glob
from tobac_flow import io
from tobac_flow.dataset import get_datetime_from_coord, create_dataarray, add_dataarray_to_ds
from tobac_flow.abi import get_abi_lat_lon, get_abi_pixel_area

def goes_dataloader(start_date, end_date, n_pad_files=1,
                    x0=None, x1=None, y0=None, y1=None,
                    time_gap=timedelta(minutes=15),
                    return_new_ds=False,
                    **io_kwargs):

    abi_files = find_goes_files(start_date, end_date, n_pad_files, **io_kwargs)

    # Load ABI files
    bt, wvd, swd = load_mcmip(abi_files, x0, x1, y0, y1)

    # Fill any gaps:
    if io_kwargs["view"] == "M":
        io_kwargs["view"] = "C"
        bt, wvd, swd = fill_time_gap_full_disk(bt, wvd, swd, time_gap, x0, x1, y0, y1, **io_kwargs)

    if io_kwargs["view"] == "C":
        io_kwargs["view"] = "F"
        bt, wvd, swd = fill_time_gap_full_disk(bt, wvd, swd, time_gap, x0, x1, y0, y1, **io_kwargs)

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

    if return_new_ds:
        goes_ds = xr.open_dataset(abi_files[0])

        goes_coords = {'t':bt.t, 'y':bt.y, 'x':bt.x,
                       'y_image':goes_ds.y_image, 'x_image':goes_ds.x_image}

        new_ds = xr.Dataset(coords=goes_coords)
        new_ds["goes_imager_projection"] = goes_ds.goes_imager_projection
        lat, lon = get_abi_lat_lon(new_ds)
        add_dataarray_to_ds(create_dataarray(lat, ('y', 'x'), 'lat', long_name="latitude", dtype=np.float32), new_ds)
        add_dataarray_to_ds(create_dataarray(lon, ('y', 'x'), 'lon', long_name="longitude", dtype=np.float32), new_ds)
        add_dataarray_to_ds(create_dataarray(get_abi_pixel_area(new_ds), ('y', 'x'), 'area',
                                             long_name="pixel area", units='km^2', dtype=np.float32), new_ds)

        return bt, wvd, swd, new_ds

    else:
        return bt, wvd, swd

def find_goes_files(start_date, end_date, n_pad_files=1, **io_kwargs):
    # Find ABI files
    dates = pd.date_range(start_date, end_date, freq='H', closed='left').to_pydatetime()

    abi_files = io.find_abi_files(dates, **io_kwargs)

    if n_pad_files > 0:
        pad_hours = int(np.ceil(n_pad_files/12))

        pre_dates = pd.date_range(start_date-timedelta(hours=pad_hours), start_date,
                                  freq='H', closed='left').to_pydatetime()
        abi_pre_file = io.find_abi_files(pre_dates, **io_kwargs)
        if len(abi_pre_file):
            abi_pre_file = abi_pre_file[-n_pad_files:]

        post_dates = pd.date_range(end_date, end_date+timedelta(hours=pad_hours),
                                  freq='H', closed='left').to_pydatetime()
        abi_post_file = io.find_abi_files(post_dates, **io_kwargs)
        if len(abi_post_file):
            abi_post_file = abi_post_file[:n_pad_files]

        abi_files = abi_pre_file + abi_files + abi_post_file

    return abi_files

def get_stripe_deviation(da):
    y_mean = da.mean('y')
    y_std = da.std('y')
    return np.abs(((da-y_mean)/(y_std+1e-8)).mean('x'))

def load_mcmip(files, x0=None, x1=None, y0=None, y1=None):
    ds_slice = {'x':slice(x0,x1), 'y':slice(y0,y1)}
    # Load a stack of goes datasets using xarray
    if len(files)>1:
        goes_ds = xr.open_mfdataset(files, concat_dim='t', combine='nested').isel(ds_slice)
    else:
        goes_ds = xr.open_dataset(files[0]).isel(ds_slice)

    # Extract fields and load into memory
    wvd = (goes_ds.CMI_C08 - goes_ds.CMI_C10).load()

    bt = goes_ds.CMI_C13.load()

    swd = (bt - goes_ds.CMI_C15.load())

    # Check for missing data and DQF flags in any channels, propagate to all data
    all_isnan = np.any([~np.isfinite(bt), ~np.isfinite(wvd), ~np.isfinite(swd)], 0)
    all_DQF = np.any([goes_ds.DQF_C08, goes_ds.DQF_C10, goes_ds.DQF_C13, goes_ds.DQF_C15], 0)
    all_stripe = np.any([get_stripe_deviation(goes_ds.DQF_C08)>2,
                         get_stripe_deviation(goes_ds.DQF_C10)>2,
                         get_stripe_deviation(goes_ds.DQF_C13)>2,
                         get_stripe_deviation(goes_ds.DQF_C15)>2], 0)

    bt.data[all_isnan] = np.nan
    bt.data[all_DQF] = np.nan
    bt.data[all_stripe] = np.nan

    wvd.data[all_isnan] = np.nan
    wvd.data[all_DQF] = np.nan
    wvd.data[all_stripe] = np.nan

    swd.data[all_isnan] = np.nan
    swd.data[all_DQF] = np.nan
    swd.data[all_stripe] = np.nan

    goes_ds.close()

    return bt, wvd, swd

def create_nan_slice(da, t_ind):
    nan_slice_da = xr.full_like(da.isel(t=slice(0,1)), np.nan)
    nan_slice_da.t.data[0] = (da.t[t_ind]+(da.t[t_ind+1]-da.t[t_ind])/2).item()
    return nan_slice_da

def fill_time_gap_nan(da, time_gap=timedelta(minutes=15)):
    where_time_gap = np.where(np.diff(get_datetime_from_coord(da.t))>time_gap)[0]

    concat_list = []
    last_t_ind = 0

    if where_time_gap.size > 0:
        for t_ind in where_time_gap:
            concat_list.append(da.isel(t=slice(last_t_ind, t_ind+1)))
            concat_list.append(create_nan_slice(da, t_ind))
            last_t_ind = t_ind+1

        concat_list.append(da.isel(t=slice(last_t_ind, None)))

        return xr.concat(concat_list, 't')
    else:
        return da

def get_full_disk_for_time_gap(start_date, end_date, **io_kwargs):
    start_date = parse_date(start_date.astype('datetime64[s]').astype('str'))
    end_date = parse_date(end_date.astype('datetime64[s]').astype('str'))
    dates = pd.date_range(start_date, end_date, freq='H').to_pydatetime()
#     io_kwargs["view"] = "F"
    F_files = io.find_abi_files(dates, **io_kwargs)#, satellite=16, product='MCMIP', view='F', mode=[3,4,6],
#                                 save_dir=goes_data_path,
#                                 replicate_path=True, check_download=True,
#                                 n_attempts=1, download_missing=True)

    F_dates = [io.get_goes_date(i) for i in F_files]

    return [file for file, date in zip(F_files, F_dates) if date>start_date and date<end_date]

def fill_time_gap_full_disk(bt, wvd, swd, time_gap=timedelta(minutes=15),
                            x0=None, x1=None, y0=None, y1=None,
                            **io_kwargs):
    where_time_gap = np.where(np.diff(get_datetime_from_coord(bt.t))>time_gap)[0]

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
        x1 = 902+2500
    if y0:
        y0 += 422
    else:
        y0 = 422
    if y1:
        y1 += 422
    else:
        y1 = 422+1500

    if where_time_gap.size > 0:
        for t_ind in where_time_gap:
            full_disk_files = get_full_disk_for_time_gap(bt.t.data[t_ind], bt.t.data[t_ind+1], **io_kwargs)
            bt_concat_list.append(bt.isel(t=slice(last_t_ind, t_ind+1)))
            wvd_concat_list.append(wvd.isel(t=slice(last_t_ind, t_ind+1)))
            swd_concat_list.append(swd.isel(t=slice(last_t_ind, t_ind+1)))

            if len(full_disk_files) > 0:
                full_bt, full_wvd, full_swd = load_mcmip(full_disk_files, x0, x1, y0, y1)

                bt_concat_list.append(full_bt)
                wvd_concat_list.append(full_wvd)
                swd_concat_list.append(full_swd)

            last_t_ind = t_ind+1

#         raise ValueError

        bt_concat_list.append(bt.isel(t=slice(last_t_ind, None)))
        wvd_concat_list.append(wvd.isel(t=slice(last_t_ind, None)))
        swd_concat_list.append(swd.isel(t=slice(last_t_ind, None)))

        return (xr.concat(bt_concat_list, 't', join="left"),
                xr.concat(wvd_concat_list, 't', join="left"),
                xr.concat(swd_concat_list, 't', join="left"))
    else:
        return bt, wvd, swd

def glob_seviri_files(start_date, end_date,
                      file_type="secondary",
                      file_path="../data/SEVIRI_ORAC/"):
    if file_type not in ["secondary", "cloud", "flux"]:
        raise ValueError("file_type parameter must be one of 'secondary', 'cloud' or 'flux'")

    dates = pd.date_range(start_date, end_date, freq='H', closed='left').to_pydatetime()

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

def find_seviri_files(start_date, end_date, n_pad_files=1,
                      file_type="secondary",
                      file_path="../data/SEVIRI_ORAC/"):

    seviri_files = glob_seviri_files(start_date, end_date,
                                     file_type, file_path)

    if n_pad_files > 0:
        pad_hours = int(np.ceil(n_pad_files/4))

        seviri_pre_file = glob_seviri_files(start_date-timedelta(hours=pad_hours),
                                            start_date, file_type, file_path)
        if len(seviri_pre_file):
            seviri_pre_file = seviri_pre_file[-n_pad_files:]

        seviri_post_file = glob_seviri_files(end_date,
                                             end_date+timedelta(hours=pad_hours),
                                             file_type, file_path)
        if len(seviri_post_file):
            seviri_post_file = seviri_post_file[:n_pad_files]

        seviri_files = seviri_pre_file + seviri_files + seviri_post_file

    return seviri_files

def load_seviri_dataset(seviri_files, x0=None, x1=None, y0=None, y1=None):
    seviri_ds = xr.open_mfdataset(seviri_files,
                                  combine="nested",
                                  concat_dim="t").isel(across_track=slice(x0,x1), along_track=slice(y0,y1))

    seviri_ds = seviri_ds.assign_coords(t=[parse_date(f[-28:-16]) for f in seviri_files])

    return seviri_ds


def seviri_dataloader(start_date, end_date, n_pad_files=1,
                      file_path="../data/SEVIRI_ORAC/",
                      x0=None, x1=None, y0=None, y1=None,
                      time_gap=timedelta(minutes=30),
                      return_new_ds=False):

    seviri_files = find_seviri_files(start_date, end_date,
                                     n_pad_files=n_pad_files,
                                     file_path=file_path)

    seviri_ds = load_seviri_dataset(seviri_files, x0, x1, y0, y1)

    bt = seviri_ds.brightness_temperature_in_channel_no_9.load()

    wvd = (seviri_ds.brightness_temperature_in_channel_no_5 - seviri_ds.brightness_temperature_in_channel_no_6).load()

    swd = (bt - seviri_ds.brightness_temperature_in_channel_no_10.load())

    all_isnan = np.any([~np.isfinite(bt), ~np.isfinite(wvd), ~np.isfinite(swd)], 0)

    bt.data[all_isnan] = np.nan
    wvd.data[all_isnan] = np.nan
    swd.data[all_isnan] = np.nan

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
        seviri_coords = {'t':bt.t,
                         'along_track':bt.along_track,
                         'across_track':bt.across_track}

        new_ds = xr.Dataset(coords=bt.coords)

        return bt, wvd, swd, new_ds
    else:
        return bt, wvd, swd
