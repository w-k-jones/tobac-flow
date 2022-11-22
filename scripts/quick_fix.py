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
from tobac_flow.analysis import apply_weighted_func_to_labels

import argparse
parser = argparse.ArgumentParser(description="""Validate detected DCCs using GOES-16 GLM data""")
parser.add_argument('file', help='File to validate', type=str)
args = parser.parse_args()

fname = args.file
dataset = xr.open_dataset(fname)

start_date = parse_date((fname).split("_S")[-1].split("_E")[0], fuzzy=True)
end_date = parse_date((fname).split("_E")[-1].split("_X")[0], fuzzy=True)

flx_files = find_seviri_files(start_date, end_date, n_pad_files=2, file_type="flux",
                                 file_path="/gws/nopw/j04/eo_shared_data_vol1/satellite/seviri-orac/flx")

flx_ds = xr.open_mfdataset(flx_files, combine="nested", concat_dim="t")

flx_ds = flx_ds.assign_coords(t=[parse_date(f[-46:-34]) for f in flx_files])

def weighted_statistics_on_labels(labels, da, cld_weights, name=None, dim=None, dtype=None):
    if not dim:
        dim = labels.name.split("_label")[0]
    if dtype == None:
        dtype = da.dtype

    try:
        long_name = da.long_name
    except AttributeError:
        long_name = da.name

    try:
        units = da.units
    except AttributeError:
        units = ""

    def weighted_average(values, weights, ignore_nan=True):
        if ignore_nan:
            wh_nan = np.isnan(values)
            values = values[~wh_nan]
            weights = weights[~wh_nan]

        return np.average(values, weights=weights)

    weighted_std = lambda x, w : weighted_average((x - weighted_average(x, w))**2, w)**0.5
    weighted_stats = lambda x, w : [weighted_average(x, w),
                                    weighted_std(x, w),
                                    np.nanmax(x[w>0]),
                                    np.nanmin(x[w>0])] if np.nansum(w>0) else [np.nan, np.nan, np.nan, np.nan]

    stats_array = apply_weighted_func_to_labels(labels.data,
                                                da.data,
                                                cld_weights,
                                                weighted_stats)

    mean_da = create_dataarray(stats_array[...,0],
                               (dim,),
                               f"{name}_{da.name}_mean",
                               long_name=f"Mean of {long_name} for each {dim}",
                               units=units,
                               dtype=dtype)

    std_da = create_dataarray(stats_array[...,1],
                              (dim,),
                              f"{name}_{da.name}_std",
                              long_name=f"Standard deviation of {long_name} for each {dim}",
                              units=units,
                              dtype=dtype)
    max_da = create_dataarray(stats_array[...,2],
                              (dim,),
                              f"{name}_{da.name}_max",
                              long_name=f"Maximum of {long_name} for each {dim}",
                              units=units,
                              dtype=dtype)
    min_da = create_dataarray(stats_array[...,3],
                              (dim,),
                              f"{name}_{da.name}_min",
                              long_name=f"Minimum of {long_name} for each {dim}",
                              units=units,
                              dtype=dtype)

    return mean_da, std_da, max_da, min_da

# toa_sw, toa_lw, toa_net
toa_net = flx_ds.toa_swdn-flx_ds.toa_swup-flx_ds.toa_lwup
# toa_net.attrs["name"] = toa_net
toa_clr = flx_ds.toa_swdn-flx_ds.toa_swup_clr-flx_ds.toa_lwup_clr
# toa_clr.attrs["name"] = toa_cld
toa_cld = toa_net-toa_clr
# toa_cld.attrs["name"] = toa_cld
toa_net = create_dataarray(toa_net.data, flx_ds.dims, "toa_net", units="")
toa_cld = create_dataarray(toa_cld.data, flx_ds.dims, "toa_cld", units="")

toa_swup_cld = create_dataarray(flx_ds.toa_swup.data-flx_ds.toa_swup_clr, flx_ds.dims, "toa_swup_cld", units="")
toa_lwup_cld = create_dataarray(flx_ds.toa_lwup.data-flx_ds.toa_lwup_clr, flx_ds.dims, "toa_lwup_cld", units="")

for field in (flx_ds.toa_swdn, flx_ds.toa_swup, flx_ds.toa_lwup, toa_net,
              toa_swup_cld, toa_lwup_cld, toa_cld):
    [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.core_label,
                                                                              field.compute(),
                                                                              area_stack,
                                                                              name='core',
                                                                              dim='core',
                                                                              dtype=np.float32)]

    [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.thick_anvil_label,
                                                                              field.compute(),
                                                                              area_stack,
                                                                              name='thick_anvil',
                                                                              dim='anvil',
                                                                              dtype=np.float32)]

    [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.thin_anvil_label,
                                                                              field.compute(),
                                                                              area_stack,
                                                                              name='thin_anvil',
                                                                              dim='anvil',
                                                                              dtype=np.float32)]

    [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.core_step_label,
                                                                              field.compute(),
                                                                              area_stack,
                                                                              name='core_step',
                                                                              dim='core_step',
                                                                              dtype=np.float32)]

    [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.thick_anvil_step_label,
                                                                              field.compute(),
                                                                              area_stack,
                                                                              name='thick_anvil_step',
                                                                              dim='thick_anvil_step',
                                                                              dtype=np.float32)]

    [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.thin_anvil_step_label,
                                                                              field.compute(),
                                                                              area_stack,
                                                                              name='thin_anvil_step',
                                                                              dim='thin_anvil_step',
                                                                              dtype=np.float32)]

print(datetime.now(), 'Saving to %s' % (save_path), flush=True)
# Add compression encoding
comp = dict(zlib=True, complevel=5, shuffle=True)
for var in dataset.data_vars:
    dataset[var].encoding.update(comp)

dataset.to_netcdf(save_path)

dataset.close()
dcc_ds.close()
cld_ds.close()
flx_ds.close()
