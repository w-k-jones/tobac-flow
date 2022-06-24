import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from tobac_flow import io
from tobac_flow.dataset import get_datetime_from_coord

def goes_dataloader(start_date, end_date, n_pad_files=1,
                    x0=None, x1=None, y0=None, y1=None,
                    time_gap=timedelta(minutes=15), **io_kwargs):
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

        all_abi_files = abi_pre_file + abi_files + abi_post_file

    else:
        all_abi_files = abi_files

    # Load ABI files
    bt, wvd, swd = load_mcmip(all_abi_files, x0, x1, y0, y1)

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

    return bt, wvd, swd

def load_mcmip(files, x0=None, x1=None, y0=None, y1=None):
    ds_slice = {'x':slice(x0,x1), 'y':slice(y0,y1)}
    # Load a stack of goes datasets using xarray
    if len(files)>1:
        goes_ds = xr.open_mfdataset(files, concat_dim='t', combine='nested').isel(ds_slice)
    else:
        goes_ds = xr.open_dataset(files[0]).isel(ds_slice)

    # Extract fields and load into memory
    wvd = goes_ds.CMI_C08 - goes_ds.CMI_C10
    try:
        wvd = wvd.compute()
    except AttributeError:
        pass

    bt = goes_ds.CMI_C13
    try:
        bt = bt.compute()
    except AttributeError:
        pass

    swd = goes_ds.CMI_C13 - goes_ds.CMI_C15
    try:
        swd = swd.compute()
    except AttributeError:
        pass

    # Check for missing data and DQF flags in any channels, propagate to all data
    all_isnan = np.any([~np.isfinite(bt), ~np.isfinite(wvd), ~np.isfinite(swd)], 0)
    all_DQF = np.any([goes_ds.DQF_C08, goes_ds.DQF_C10, goes_ds.DQF_C13, goes_ds.DQF_C15], 0)

    bt.data[all_isnan] = np.nan
    bt.data[all_DQF] = np.nan

    wvd.data[all_isnan] = np.nan
    wvd.data[all_DQF] = np.nan

    swd.data[all_isnan] = np.nan
    swd.data[all_DQF] = np.nan

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
    dates = pd.date_range(start_date, start_date+hours, freq='H').to_pydatetime()
#     io_kwargs["view"] = "F"
    F_files = io.find_abi_files(dates, **io_kwargs)#, satellite=16, product='MCMIP', view='F', mode=[3,4,6],
#                                 save_dir=goes_data_path,
#                                 replicate_path=True, check_download=True,
#                                 n_attempts=1, download_missing=True)

    F_dates = [io.get_goes_date(i) for i in F_files]

    return [file for file,date in zip(F_files, F_dates) if date>start_date and date<end_date]

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
