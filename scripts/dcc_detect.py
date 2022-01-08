import os
import sys
import inspect
import itertools
import warnings

import numpy as np
from numpy import ma
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from scipy import ndimage as ndi

import argparse
parser = argparse.ArgumentParser(description="""Regrid GLM and NEXRAD data to the GOES-16 projection""")
parser.add_argument('date', help='Date on which to start process', type=str)
parser.add_argument('days', help='Number of days to process', type=float)
parser.add_argument('-x0', help='Initial subset x location', default=0, type=int)
parser.add_argument('-x1', help='End subset x location', default=2500, type=int)
parser.add_argument('-y0', help='Initial subset y location', default=0, type=int)
parser.add_argument('-y1', help='End subset y location', default=1500, type=int)
parser.add_argument('-sd', help='Directory to save preprocess files',
                    default='../data/dcc_detect', type=str)
parser.add_argument('-gd', help='GOES directory',
                    default='../data/GOES16', type=str)
parser.add_argument('--extend_path', help='Extend save directory using year/month/day subdirectories',
                    default=True, type=bool)

start_time = datetime.now()
args = parser.parse_args()
start_date = parse_date(args.date, fuzzy=True)
end_date = start_date + timedelta(days=args.days)

x0 = int(args.x0)
x1 = int(args.x1)
y0 = int(args.y0)
y1 = int(args.y1)

save_dir = args.sd
# if args.extend_path:
    # save_dir = os.path.join(save_dir, start_date.strftime('%Y/%m/%d'))
if not os.path.isdir(save_dir):
    try:
        os.makedirs(save_dir)
    except (FileExistsError, OSError):
        pass

save_name = 'detected_dccs_%s.nc' % (start_date.strftime('%Y%m%d_%H0000'))

save_path = os.path.join(save_dir, save_name)

# code from https://stackoverflow.com/questions/279237/import-a-module-from-a-relative-path?lq=1#comment15918105_6098238 to load a realitive folde from a notebook
# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.dirname(os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from tobac_flow import io, abi, glm
from tobac_flow.flow import Flow
from tobac_flow.dataset import get_datetime_from_coord, get_time_diff_from_coord, create_new_goes_ds
from tobac_flow.detection import detect_growth_markers, edge_watershed
from tobac_flow.analysis import filter_labels_by_length, filter_labels_by_length_and_mask, apply_func_to_labels
from tobac_flow.validation import get_min_dist_for_objects, get_marker_distance

goes_data_path = args.gd
if not os.path.isdir(goes_data_path):
    try:
        os.makedirs(goes_data_path)
    except (FileExistsError, OSError):
        pass

print(datetime.now(),'Loading ABI data', flush=True)
print('Saving data to:',goes_data_path, flush=True)
dates = pd.date_range(start_date, end_date, freq='H', closed='left').to_pydatetime()
abi_files = io.find_abi_files(dates, satellite=16, product='MCMIP',
                              view='C', mode=[3,4,6], save_dir=goes_data_path,
                              replicate_path=True, check_download=True,
                              n_attempts=1, download_missing=False, verbose=True,
                              min_storage=2**30)

# Test with some multichannel data
ds_slice = {'x':slice(x0,x1), 'y':slice(y0,y1)}
# Load a stack of goes datasets using xarray. Select a region over Northern Florida. (full file size in 1500x2500 pixels)
with xr.open_mfdataset(abi_files, concat_dim='t', combine='nested').isel(ds_slice) as goes_ds:
    goes_dates = get_datetime_from_coord(goes_ds.t)
    # Check for invalid dates (which are given a date in 2000)
    wh_valid_dates = [gd > datetime(2001,1,1) for gd in goes_dates]
    if np.any(np.logical_not(wh_valid_dates)):
        warnings.warn("Missing timestep found, removing")
        goes_ds = goes_ds.isel({'t':wh_valid_dates})

    print('%d files found'%len(abi_files), flush=True)

    if len(abi_files)==0:
        raise ValueError("No ABI files discovered, aborting")

    # Extract fields and load into memory
    print(datetime.now(),'Loading WVD', flush=True)
    wvd = goes_ds.CMI_C08 - goes_ds.CMI_C10
    if hasattr(wvd, "compute"):
        wvd = wvd.compute()
    print(datetime.now(),'Loading BT', flush=True)
    bt = goes_ds.CMI_C13
    if hasattr(bt, "compute"):
        bt = bt.compute()
    print(datetime.now(),'Loading SWD', flush=True)
    swd = goes_ds.CMI_C13 - goes_ds.CMI_C15
    if hasattr(swd, "compute"):
        swd = swd.compute()

    wh_all_missing = np.any([np.all(np.isnan(wvd), (1,2)),
                             np.all(np.isnan(bt), (1,2)),
                             np.all(np.isnan(swd), (1,2))],
                             0)
    if np.any(wh_all_missing):
        warnings.warn("Missing data found at timesteps")
        goes_ds = goes_ds.isel({'t':np.logical_not(wh_all_missing)})

        print(datetime.now(),'Loading WVD', flush=True)
        wvd = goes_ds.CMI_C08 - goes_ds.CMI_C10
        if hasattr(wvd, "compute"):
            wvd = wvd.compute()
        print(datetime.now(),'Loading BT', flush=True)
        bt = goes_ds.CMI_C13
        if hasattr(bt, "compute"):
            bt = bt.compute()
        print(datetime.now(),'Loading SWD', flush=True)
        swd = goes_ds.CMI_C13 - goes_ds.CMI_C15
        if hasattr(swd, "compute"):
            swd = swd.compute()

    # Now we have all the valid timesteps, check for gaps in the time series
    goes_timedelta = get_time_diff_from_coord(goes_ds.t)

    if np.any([td>15.5 for td in goes_timedelta]):
        raise ValueError("Time gaps in abi data greater than 15 minutes, aborting")

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

    dataset = create_new_goes_ds(goes_ds)

print(datetime.now(),'Calculating flow field', flush=True)
flow_kwargs = {'pyr_scale':0.5, 'levels':5, 'winsize':16, 'iterations':3,
               'poly_n':5, 'poly_sigma':1.1, 'flags':256}

flow = Flow(bt, flow_kwargs=flow_kwargs, smoothing_passes=3)

print(datetime.now(),'Detecting growth markers', flush=True)
wvd_growth, growth_markers = detect_growth_markers(flow, wvd)

print('Growth above threshold: area =', np.sum(wvd_growth>=0.5), flush=True)
print('Detected markers: area =', np.sum(growth_markers.data!=0), flush=True)
print('Detected markers: n =', growth_markers.data.max(), flush=True)

print(datetime.now(), 'Detecting thick anvil region', flush=True)
inner_watershed = edge_watershed(flow, wvd-swd+np.maximum(wvd_growth,0)*5, growth_markers!=0, -5, -15, verbose=True)
inner_labels = filter_labels_by_length_and_mask(flow.label(inner_watershed),
                                                growth_markers.data!=0, 3)
print('Detected thick anvils: area =', np.sum(inner_labels!=0), flush=True)
print('Detected thick anvils: n =', inner_labels.max(), flush=True)

print(datetime.now(), 'Detecting thin anvil region', flush=True)
outer_watershed = edge_watershed(flow, wvd+swd+np.maximum(wvd_growth,0)*5, inner_labels, 0, -10, verbose=True)
print('Detected thin anvils: area =', np.sum(outer_watershed!=0), flush=True)

print(datetime.now(),'Processing GLM data', flush=True)
# Get GLM data
# Process new GLM data
glm_files = io.find_glm_files(dates, satellite=16, save_dir=goes_data_path,
                              replicate_path=True, check_download=True,
                              n_attempts=1, download_missing=True, verbose=True,
                              min_storage=2**30)
glm_files = {io.get_goes_date(i):i for i in glm_files}
print('%d files found'%len(glm_files), flush=True)
if len(glm_files)==0:
    warnings.warn("No GLM Files discovered, skipping validation")
    glm_grid = xr.zeros_like(wvd)
else:
    print(datetime.now(),'Regridding GLM data', flush=True)
    glm_grid = glm.regrid_glm(glm_files, goes_ds, corrected=False)

print(datetime.now(),'Calculating marker distances', flush=True)
marker_distance = get_marker_distance(growth_markers, time_range=3)
anvil_distance = get_marker_distance(inner_labels, time_range=3)
glm_distance = get_marker_distance(glm_grid, time_range=3)

s_struct = ndi.generate_binary_structure(2,1)[np.newaxis]
wvd_labels = flow.label(ndi.binary_opening(wvd>=-5, structure=s_struct))
wvd_labels = filter_labels_by_length_and_mask(wvd_labels, wvd.data>=-5, 3)
print("warm WVD regions: n =",wvd_labels.max(), flush=True)
wvd_distance = get_marker_distance(wvd_labels, time_range=3)

print(datetime.now(), 'Validating detection accuracy', flush=True)
marker_pod_hist = np.histogram(marker_distance[glm_grid.data>0],
                               weights=glm_grid.data[glm_grid.data>0], bins=40,
                               range=[0,40])[0] / np.sum(glm_grid.data[glm_grid.data>0])
wvd_pod_hist = np.histogram(wvd_distance[glm_grid.data>0],
                            weights=glm_grid.data[glm_grid.data>0], bins=40,
                            range=[0,40])[0] / np.sum(glm_grid.data[glm_grid.data>0])
anvil_pod_hist = np.histogram(anvil_distance[glm_grid.data>0],
                              weights=glm_grid.data[glm_grid.data>0], bins=40,
                              range=[0,40])[0] / np.sum(glm_grid.data[glm_grid.data>0])

growth_min_distance = get_min_dist_for_objects(glm_distance, growth_markers)[0]
growth_validation_flag = growth_min_distance<10
growth_far_hist = np.histogram(growth_min_distance, bins=40,
                               range=[0,40])[0] / growth_markers.data.max()
wvd_min_distance = get_min_dist_for_objects(glm_distance, wvd_labels)[0]
wvd_validation_flag = wvd_min_distance<10
wvd_far_hist = np.histogram(wvd_min_distance, bins=40,
                            range=[0,40])[0] / wvd_labels.max()
anvil_min_distance = get_min_dist_for_objects(glm_distance, inner_labels)[0]
anvil_validation_flag = anvil_min_distance<10
anvil_far_hist = np.histogram(anvil_min_distance, bins=40,
                              range=[0,40])[0] / inner_labels.max()

print('markers:', flush=True)
print('n =', growth_markers.data.max(), flush=True)
print(np.sum(marker_pod_hist[:10]), flush=True)
print(1-np.sum(growth_far_hist[:10]), flush=True)

print('WVD:', flush=True)
print('n =', wvd_labels.max(), flush=True)
print(np.sum(wvd_pod_hist[:10]), flush=True)
print(1-np.sum(wvd_far_hist[:10]), flush=True)

print('anvil:', flush=True)
print('n =', inner_labels.max(), flush=True)
print(np.sum(anvil_pod_hist[:10]), flush=True)
print(1-np.sum(anvil_far_hist[:10]), flush=True)

print('total GLM flashes: ', np.sum(glm_grid.data), flush=True)

# Get statistics about various properties of each label

# Growth value
from tobac_flow.dataset import add_dataarray_to_ds, create_dataarray
add_dataarray_to_ds(create_dataarray(wvd_growth, ('t','y','x'), "growth_rate",
                                     long_name="detected cloud top cooling rate", units="K/min",
                                     dtype=np.float32), dataset)

# GLM Flash counts
add_dataarray_to_ds(create_dataarray(glm_grid.data, ('t','y','x'), "glm_flashes",
                                     long_name="number of flashes detected by GLM", units="",
                                     dtype=np.int32), dataset)

# Detected growth_regions
add_dataarray_to_ds(create_dataarray(growth_markers, ('t','y','x'), "core_label",
                                     long_name="labels for detected cores", units="",
                                     dtype=np.int32), dataset)

dataset.coords["core"] = np.arange(1, growth_markers.max()+1, dtype=np.int32)

from tobac_flow.analysis import get_label_stats, get_stats_for_labels
get_label_stats(dataset.core_label, dataset)

for field in (bt, wvd, swd, dataset.growth_rate):
    [add_dataarray_to_ds(da, dataset) for da in get_stats_for_labels(dataset.core_label, field, dim='core')]

from tobac_flow.analysis import slice_label_da
core_step_label = slice_label_da(dataset.core_label)
add_dataarray_to_ds(core_step_label, dataset)
dataset.coords["core_step"] = np.arange(1, core_step_label.max()+1, dtype=np.int32)

# Now get individual step
for field in (bt, wvd, swd, dataset.growth_rate):
    [add_dataarray_to_ds(da, dataset) for da in get_stats_for_labels(dataset.core_step_label, field, dim='core_step')]

# Now we have stats of each field for each core and each core time step
# Next get other data: weighted x,y,lat,lon locations for each step, t per step
from tobac_flow.analysis import apply_weighted_func_to_labels

tt, yy, xx = np.meshgrid(dataset.t, dataset.y, dataset.x, indexing='ij')

core_step_x = apply_weighted_func_to_labels(core_step_label.data, xx,
                                            np.maximum(dataset.growth_rate.data-0.25, 0.01),
                                            lambda a,b: np.average(a, weights=b))
add_dataarray_to_ds(create_dataarray(core_step_x, ('core_step',), "core_step_x",
                                     long_name="x location of core at time step",
                                     dtype=np.float64), dataset)

core_step_y = apply_weighted_func_to_labels(core_step_label.data, yy,
                                            np.maximum(dataset.growth_rate.data-0.25, 0.01),
                                            lambda a,b: np.average(a, weights=b))
add_dataarray_to_ds(create_dataarray(core_step_y, ('core_step',), "core_step_y",
                                     long_name="y location of core at time step",
                                     dtype=np.float64), dataset)

core_step_t = apply_func_to_labels(core_step_label.data, tt,
                                   np.nanmin)
add_dataarray_to_ds(create_dataarray(core_step_t, ('core_step',), "core_step_t",
                                     long_name="time of core at step",
                                     dtype="datetime64[ns]"), dataset)

core_start_t = apply_func_to_labels(dataset.core_label.data, tt,
                                   np.nanmin)
add_dataarray_to_ds(create_dataarray(core_start_t, ('core',), "core_start_t",
                                     long_name="initial detection time of core",
                                     dtype="datetime64[ns]"), dataset)

core_end_t = apply_func_to_labels(dataset.core_label.data, tt,
                                   np.nanmax)
add_dataarray_to_ds(create_dataarray(core_end_t, ('core',), "core_end_t",
                                     long_name="final detection time of core",
                                     dtype="datetime64[ns]"), dataset)

add_dataarray_to_ds(create_dataarray(core_end_t-core_start_t, ('core',), "core_lifetime",
                                     long_name="total lifetime of core",
                                     dtype="timedelta64[ns]"), dataset)

from tobac_flow.abi import get_abi_proj
p = get_abi_proj(dataset)
core_step_lons, core_step_lats = p(core_step_x*dataset.goes_imager_projection.perspective_point_height,
                                   core_step_y*dataset.goes_imager_projection.perspective_point_height,
                                   inverse=True)
add_dataarray_to_ds(create_dataarray(core_step_lons, ('core_step',), "core_step_lon",
                                     long_name="longitude of core at time step",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(core_step_lats, ('core_step',), "core_step_lat",
                                     long_name="latitude of core at time step",
                                     dtype=np.float32), dataset)

core_step_glm_count = apply_func_to_labels(core_step_label.data,
                                           dataset.glm_flashes.data, np.nansum)
add_dataarray_to_ds(create_dataarray(core_step_glm_count, ('core_step',), "core_step_glm_count",
                                     long_name="number of GLM flashes for core at time step",
                                     dtype=np.int32), dataset)

core_glm_count = apply_func_to_labels(dataset.core_label.data,
                                      dataset.glm_flashes.data, np.nansum)
add_dataarray_to_ds(create_dataarray(core_glm_count, ('core',), "core_total_glm_count",
                                     long_name="total number of GLM flashes for core",
                                     dtype=np.int32), dataset)
# Get area in 3d
aa = np.meshgrid(dataset.t, dataset.area, indexing='ij')[1].reshape(tt.shape)

core_step_pixels = np.bincount(dataset.core_step_label.data.ravel())[1:]
add_dataarray_to_ds(create_dataarray(core_step_pixels, ('core_step',), "core_step_pixels",
                                     long_name="number of pixels for core at time step",
                                     dtype=np.int32), dataset)

core_total_pixels = np.bincount(dataset.core_label.data.ravel())[1:]
add_dataarray_to_ds(create_dataarray(core_total_pixels, ('core',), "core_total_pixels",
                                     long_name="total number of pixels for core",
                                     dtype=np.int32), dataset)

core_step_area = apply_func_to_labels(dataset.core_step_label.data, aa, np.nansum)
add_dataarray_to_ds(create_dataarray(core_step_area, ('core_step',), "core_step_area",
                                     long_name="area of core at time step",
                                     dtype=np.float32), dataset)

core_total_area = apply_func_to_labels(dataset.core_label.data, aa, np.nansum)
add_dataarray_to_ds(create_dataarray(core_total_area, ('core',), "core_total_area",
                                     long_name="total area of core",
                                     dtype=np.float32), dataset)

core_index_for_step = apply_func_to_labels(dataset.core_step_label.data,
                                           dataset.core_label.data, np.nanmax)
add_dataarray_to_ds(create_dataarray(core_index_for_step, ('core_step',), "core_index_for_step",
                                     long_name="core index for each core time step",
                                     dtype=np.int32), dataset)

# Now for thick anvils
add_dataarray_to_ds(create_dataarray(inner_labels, ('t','y','x'), "thick_anvil_label",
                                     long_name="labels for detected thick anvil regions", units="",
                                     dtype=np.int32), dataset)

dataset.coords["anvil"] = np.arange(1, inner_labels.max()+1, dtype=np.int32)

from tobac_flow.analysis import get_label_stats, get_stats_for_labels
get_label_stats(dataset.thick_anvil_label, dataset)

for field in (bt, wvd, swd, dataset.growth_rate):
    [add_dataarray_to_ds(da, dataset) for da in get_stats_for_labels(dataset.thick_anvil_label, field, dim='anvil')]

from tobac_flow.analysis import slice_label_da
thick_anvil_step_label = slice_label_da(dataset.thick_anvil_label)
add_dataarray_to_ds(thick_anvil_step_label, dataset)
dataset.coords["anvil_step"] = np.arange(1, thick_anvil_step_label.max()+1, dtype=np.int32)

# Now get individual step
for field in (bt, wvd, swd, dataset.growth_rate):
    [add_dataarray_to_ds(da, dataset) for da in get_stats_for_labels(dataset.thick_anvil_step_label, field, dim='anvil_step')]

# Now we have stats of each field for each core and each core time step
# Next get other data: weighted x,y,lat,lon locations for each step, t per step
from tobac_flow.analysis import apply_weighted_func_to_labels

tt, yy, xx = np.meshgrid(dataset.t, dataset.y, dataset.x, indexing='ij')

thick_anvil_step_x = apply_weighted_func_to_labels(thick_anvil_step_label.data, xx,
                                            np.maximum(wvd.data+15, 0.01),
                                            lambda a,b: np.average(a, weights=b))
add_dataarray_to_ds(create_dataarray(thick_anvil_step_x, ('anvil_step',), "thick_anvil_step_x",
                                     long_name="x location of thick_anvil at time step",
                                     dtype=np.float64), dataset)

thick_anvil_step_y = apply_weighted_func_to_labels(thick_anvil_step_label.data, yy,
                                            np.maximum(dataset.growth_rate.data-0.25, 0.01),
                                            lambda a,b: np.average(a, weights=b))
add_dataarray_to_ds(create_dataarray(thick_anvil_step_y, ('anvil_step',), "thick_anvil_step_y",
                                     long_name="y location of thick_anvil at time step",
                                     dtype=np.float64), dataset)

thick_anvil_step_t = apply_func_to_labels(thick_anvil_step_label.data, tt,
                                   np.nanmin)
add_dataarray_to_ds(create_dataarray(thick_anvil_step_t, ('anvil_step',), "thick_anvil_step_t",
                                     long_name="time of thick_anvil at step",
                                     dtype="datetime64[ns]"), dataset)

thick_anvil_start_t = apply_func_to_labels(dataset.thick_anvil_label.data, tt,
                                   np.nanmin)
add_dataarray_to_ds(create_dataarray(thick_anvil_start_t, ('anvil',), "thick_anvil_start_t",
                                     long_name="initial detection time of thick_anvil",
                                     dtype="datetime64[ns]"), dataset)

thick_anvil_end_t = apply_func_to_labels(dataset.thick_anvil_label.data, tt,
                                   np.nanmax)
add_dataarray_to_ds(create_dataarray(thick_anvil_end_t, ('anvil',), "thick_anvil_end_t",
                                     long_name="final detection time of thick_anvil",
                                     dtype="datetime64[ns]"), dataset)

add_dataarray_to_ds(create_dataarray(thick_anvil_end_t-thick_anvil_start_t, ('anvil',), "thick_anvil_lifetime",
                                     long_name="total lifetime of thick_anvil",
                                     dtype="timedelta64[ns]"), dataset)

from tobac_flow.abi import get_abi_proj
p = get_abi_proj(dataset)
thick_anvil_step_lons, thick_anvil_step_lats = p(thick_anvil_step_x*dataset.goes_imager_projection.perspective_point_height,
                                   thick_anvil_step_y*dataset.goes_imager_projection.perspective_point_height,
                                   inverse=True)
add_dataarray_to_ds(create_dataarray(thick_anvil_step_lons, ('anvil_step',), "thick_anvil_step_lon",
                                     long_name="longitude of thick_anvil at time step",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(thick_anvil_step_lats, ('anvil_step',), "thick_anvil_step_lat",
                                     long_name="latitude of thick_anvil at time step",
                                     dtype=np.float32), dataset)

thick_anvil_step_glm_count = apply_func_to_labels(thick_anvil_step_label.data,
                                           dataset.glm_flashes.data, np.nansum)
add_dataarray_to_ds(create_dataarray(thick_anvil_step_glm_count, ('anvil_step',), "thick_anvil_step_glm_count",
                                     long_name="number of GLM flashes for thick_anvil at time step",
                                     dtype=np.int32), dataset)

thick_anvil_glm_count = apply_func_to_labels(dataset.thick_anvil_label.data,
                                      dataset.glm_flashes.data, np.nansum)
add_dataarray_to_ds(create_dataarray(thick_anvil_glm_count, ('anvil',), "thick_anvil_total_glm_count",
                                     long_name="total number of GLM flashes for thick_anvil",
                                     dtype=np.int32), dataset)
# Get area in 3d
aa = np.meshgrid(dataset.t, dataset.area, indexing='ij')[1].reshape(tt.shape)

thick_anvil_step_pixels = np.bincount(dataset.thick_anvil_step_label.data.ravel())[1:]
add_dataarray_to_ds(create_dataarray(thick_anvil_step_pixels, ('anvil_step',), "thick_anvil_step_pixels",
                                     long_name="number of pixels for thick_anvil at time step",
                                     dtype=np.int32), dataset)

thick_anvil_total_pixels = np.bincount(dataset.thick_anvil_label.data.ravel())[1:]
add_dataarray_to_ds(create_dataarray(thick_anvil_total_pixels, ('anvil',), "thick_anvil_total_pixels",
                                     long_name="total number of pixels for thick_anvil",
                                     dtype=np.int32), dataset)

thick_anvil_step_area = apply_func_to_labels(dataset.thick_anvil_step_label.data, aa, np.nansum)
add_dataarray_to_ds(create_dataarray(thick_anvil_step_area, ('anvil_step',), "thick_anvil_step_area",
                                     long_name="area of thick_anvil at time step",
                                     dtype=np.float32), dataset)

thick_anvil_total_area = apply_func_to_labels(dataset.thick_anvil_label.data, aa, np.nansum)
add_dataarray_to_ds(create_dataarray(thick_anvil_total_area, ('anvil',), "thick_anvil_total_area",
                                     long_name="total area of thick_anvil",
                                     dtype=np.float32), dataset)

thick_anvil_index_for_step = apply_func_to_labels(dataset.thick_anvil_step_label.data,
                                           dataset.thick_anvil_label.data, np.nanmax)
add_dataarray_to_ds(create_dataarray(thick_anvil_index_for_step, ('anvil_step',), "thick_anvil_index_for_step",
                                     long_name="thick_anvil index for each thick_anvil time step",
                                     dtype=np.int32), dataset)

anvil_index_for_core = apply_func_to_labels(dataset.core_label.data,
                                           dataset.thick_anvil_label.data, np.nanmax)
add_dataarray_to_ds(create_dataarray(anvil_index_for_core, ('core',), "anvil_index_for_core",
                                     long_name="anvil index for each core",
                                     dtype=np.int32), dataset)
if np.any(anvil_index_for_core):
    cores_per_anvil = np.bincount(anvil_index_for_core)[1:]
else:
    cores_per_anvil = np.array([], dtype=int)

add_dataarray_to_ds(create_dataarray(cores_per_anvil, ('anvil',), "cores_per_anvil",
                                     long_name="number of cores per detected anvil region",
                                     dtype=np.int32), dataset)

# Now for thin anvils
add_dataarray_to_ds(create_dataarray(outer_watershed, ('t','y','x'), "thin_anvil_label",
                                     long_name="labels for detected thick anvil regions", units="",
                                     dtype=np.int32), dataset)

# dataset.coords["anvil"] = np.arange(1, inner_labels.max()+1, dtype=np.int32)

from tobac_flow.analysis import get_label_stats, get_stats_for_labels
get_label_stats(dataset.thin_anvil_label, dataset)

for field in (bt, wvd, swd, dataset.growth_rate):
    [add_dataarray_to_ds(da, dataset) for da in get_stats_for_labels(dataset.thin_anvil_label, field, dim='anvil')]

from tobac_flow.analysis import slice_label_da
thin_anvil_step_label = slice_label_da(dataset.thin_anvil_label)
add_dataarray_to_ds(thin_anvil_step_label, dataset)
dataset.coords["thin_anvil_step"] = np.arange(1, thin_anvil_step_label.max()+1, dtype=np.int32)

# Now get individual step
for field in (bt, wvd, swd, dataset.growth_rate):
    [add_dataarray_to_ds(da, dataset) for da in get_stats_for_labels(dataset.thin_anvil_step_label, field, dim='thin_anvil_step')]

# Now we have stats of each field for each core and each core time step
# Next get other data: weighted x,y,lat,lon locations for each step, t per step
from tobac_flow.analysis import apply_weighted_func_to_labels

tt, yy, xx = np.meshgrid(dataset.t, dataset.y, dataset.x, indexing='ij')

thin_anvil_step_x = apply_weighted_func_to_labels(thin_anvil_step_label.data, xx,
                                            np.maximum(wvd.data+15, 0.01),
                                            lambda a,b: np.average(a, weights=b))
add_dataarray_to_ds(create_dataarray(thin_anvil_step_x, ('thin_anvil_step',), "thin_anvil_step_x",
                                     long_name="x location of thin_anvil at time step",
                                     dtype=np.float64), dataset)

thin_anvil_step_y = apply_weighted_func_to_labels(thin_anvil_step_label.data, yy,
                                            np.maximum(dataset.growth_rate.data-0.25, 0.01),
                                            lambda a,b: np.average(a, weights=b))
add_dataarray_to_ds(create_dataarray(thin_anvil_step_y, ('thin_anvil_step',), "thin_anvil_step_y",
                                     long_name="y location of thin_anvil at time step",
                                     dtype=np.float64), dataset)

thin_anvil_step_t = apply_func_to_labels(thin_anvil_step_label.data, tt,
                                   np.nanmin)
add_dataarray_to_ds(create_dataarray(thin_anvil_step_t, ('thin_anvil_step',), "thin_anvil_step_t",
                                     long_name="time of thin_anvil at step",
                                     dtype="datetime64[ns]"), dataset)

thin_anvil_start_t = apply_func_to_labels(dataset.thin_anvil_label.data, tt,
                                   np.nanmin)
add_dataarray_to_ds(create_dataarray(thin_anvil_start_t, ('anvil',), "thin_anvil_start_t",
                                     long_name="initial detection time of thin_anvil",
                                     dtype="datetime64[ns]"), dataset)

thin_anvil_end_t = apply_func_to_labels(dataset.thin_anvil_label.data, tt,
                                   np.nanmax)
add_dataarray_to_ds(create_dataarray(thin_anvil_end_t, ('anvil',), "thin_anvil_end_t",
                                     long_name="final detection time of thin_anvil",
                                     dtype="datetime64[ns]"), dataset)

add_dataarray_to_ds(create_dataarray(thin_anvil_end_t-thin_anvil_start_t, ('anvil',), "thin_anvil_lifetime",
                                     long_name="total lifetime of thin_anvil",
                                     dtype="timedelta64[ns]"), dataset)

from tobac_flow.abi import get_abi_proj
p = get_abi_proj(dataset)
thin_anvil_step_lons, thin_anvil_step_lats = p(thin_anvil_step_x*dataset.goes_imager_projection.perspective_point_height,
                                   thin_anvil_step_y*dataset.goes_imager_projection.perspective_point_height,
                                   inverse=True)
add_dataarray_to_ds(create_dataarray(thin_anvil_step_lons, ('thin_anvil_step',), "thin_anvil_step_lon",
                                     long_name="longitude of thin_anvil at time step",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(thin_anvil_step_lats, ('thin_anvil_step',), "thin_anvil_step_lat",
                                     long_name="latitude of thin_anvil at time step",
                                     dtype=np.float32), dataset)

thin_anvil_step_glm_count = apply_func_to_labels(thin_anvil_step_label.data,
                                           dataset.glm_flashes.data, np.nansum)
add_dataarray_to_ds(create_dataarray(thin_anvil_step_glm_count, ('thin_anvil_step',), "thin_anvil_step_glm_count",
                                     long_name="number of GLM flashes for thin_anvil at time step",
                                     dtype=np.int32), dataset)

thin_anvil_glm_count = apply_func_to_labels(dataset.thin_anvil_label.data,
                                      dataset.glm_flashes.data, np.nansum)
add_dataarray_to_ds(create_dataarray(thin_anvil_glm_count, ('anvil',), "thin_anvil_total_glm_count",
                                     long_name="total number of GLM flashes for thin_anvil",
                                     dtype=np.int32), dataset)
# Get area in 3d
aa = np.meshgrid(dataset.t, dataset.area, indexing='ij')[1].reshape(tt.shape)

thin_anvil_step_pixels = np.bincount(dataset.thin_anvil_step_label.data.ravel())[1:]
add_dataarray_to_ds(create_dataarray(thin_anvil_step_pixels, ('thin_anvil_step',), "thin_anvil_step_pixels",
                                     long_name="number of pixels for thin_anvil at time step",
                                     dtype=np.int32), dataset)

thin_anvil_total_pixels = np.bincount(dataset.thin_anvil_label.data.ravel())[1:]
add_dataarray_to_ds(create_dataarray(thin_anvil_total_pixels, ('anvil',), "thin_anvil_total_pixels",
                                     long_name="total number of pixels for thin_anvil",
                                     dtype=np.int32), dataset)

thin_anvil_step_area = apply_func_to_labels(dataset.thin_anvil_step_label.data, aa, np.nansum)
add_dataarray_to_ds(create_dataarray(thin_anvil_step_area, ('thin_anvil_step',), "thin_anvil_step_area",
                                     long_name="area of thin_anvil at time step",
                                     dtype=np.float32), dataset)

thin_anvil_total_area = apply_func_to_labels(dataset.thin_anvil_label.data, aa, np.nansum)
add_dataarray_to_ds(create_dataarray(thin_anvil_total_area, ('anvil',), "thin_anvil_total_area",
                                     long_name="total area of thin_anvil",
                                     dtype=np.float32), dataset)

thin_anvil_index_for_step = apply_func_to_labels(dataset.thin_anvil_step_label.data,
                                           dataset.thin_anvil_label.data, np.nanmax)
add_dataarray_to_ds(create_dataarray(thin_anvil_index_for_step, ('thin_anvil_step',), "thin_anvil_index_for_step",
                                     long_name="thin_anvil index for each thin_anvil time step",
                                     dtype=np.int32), dataset)

add_dataarray_to_ds(create_dataarray(np.sum(glm_grid.data), tuple(), "glm_flash_count",
                                     long_name="total number of GLM flashes",
                                     dtype=np.int32), dataset)

# anvil validation
add_dataarray_to_ds(create_dataarray(anvil_min_distance, ('anvil',), "anvil_glm_distance",
                                     long_name="closest distance from anvil to GLM flash",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(anvil_validation_flag, ('anvil',), "anvil_validation_flag",
                                     long_name="validation flag for anvil",
                                     dtype=bool), dataset)
add_dataarray_to_ds(create_dataarray(anvil_pod_hist, ('bins',), "anvil_far_histogram",
                                     long_name="FAR histogram for anvils",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(anvil_pod_hist, ('bins',), "anvil_pod_histogram",
                                     long_name="POD histogram for anvils",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(np.sum(anvil_pod_hist[:10]), tuple(), "anvil_pod",
                                     long_name="POD for anvils",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(1-np.sum(anvil_far_hist[:10]), tuple(), "anvil_far",
                                     long_name="FAR for anvils",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(inner_labels.max(), tuple(), "anvil_count",
                                     long_name="total number of anvils",
                                     dtype=np.int32), dataset)

# wvd validation
add_dataarray_to_ds(create_dataarray(wvd_far_hist, ('bins',), "wvd_far_histogram",
                                     long_name="FAR histogram for wvd regions",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(wvd_pod_hist, ('bins',), "wvd_pod_histogram",
                                     long_name="POD histogram for cores",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(np.sum(wvd_pod_hist[:10]), tuple(), "wvd_pod",
                                     long_name="POD for wvd regions",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(1-np.sum(wvd_far_hist[:10]), tuple(), "wvd_far",
                                     long_name="FAR for wvd regions",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(wvd_labels.max(), tuple(), "wvd_count",
                                     long_name="total number of wvd regions",
                                     dtype=np.int32), dataset)

# core validation
add_dataarray_to_ds(create_dataarray(growth_min_distance, ('core',), "core_glm_distance",
                                     long_name="closest distance from core to GLM flash",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(growth_validation_flag, ('core',), "core_validation_flag",
                                     long_name="validation flag for core",
                                     dtype=bool), dataset)
add_dataarray_to_ds(create_dataarray(growth_far_hist, ('bins',), "core_far_histogram",
                                     long_name="FAR histogram for cores",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(marker_pod_hist, ('bins',), "core_pod_histogram",
                                     long_name="POD histogram for cores",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(np.sum(marker_pod_hist[:10]), tuple(), "core_pod",
                                     long_name="POD for cores",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(1-np.sum(growth_far_hist[:10]), tuple(), "core_far",
                                     long_name="FAR for cores",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(growth_markers.data.max(), tuple(), "core_count",
                                     long_name="total number of cores",
                                     dtype=np.int32), dataset)

print(datetime.now(), 'Preparing output', flush=True)

print(datetime.now(), 'Saving to %s' % (save_path), flush=True)
dataset.to_netcdf(save_path)
print(datetime.now(), 'Finished successfully, time elapsed:', datetime.now()-start_time, flush=True)
