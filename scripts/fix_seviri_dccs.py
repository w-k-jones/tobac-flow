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
parser = argparse.ArgumentParser(description="""Validate detected DCCs using GOES-16 GLM data""")
parser.add_argument('file', help='File to validate', type=str)
args = parser.parse_args()

fname = args.file
dcc_ds = xr.open_dataset(fname)

seviri_coords = {'t':dcc_ds.t,
                 'along_track':dcc_ds.along_track,
                 'across_track':dcc_ds.across_track}

dataset = xr.Dataset(coords=seviri_coords)

start_date = parse_date((fname).split("_S")[-1].split("_E")[0], fuzzy=True)
end_date = parse_date((fname).split("_E")[-1].split("_X")[0], fuzzy=True)

# Load cloud properties file
from tobac_flow.dataloader import find_seviri_files

cld_files = find_seviri_files(start_date, end_date, n_pad_files=2, file_type="cloud",
                              file_path="/gws/nopw/j04/eo_shared_data_vol1/satellite/seviri-orac/cld")

cld_ds = xr.open_mfdataset(cld_files, combine="nested", concat_dim="t")

cld_ds = cld_ds.assign_coords(t=[parse_date(f[-64:-50]) for f in cld_files])

# Load flux file
flx_files = find_seviri_files(datetime(2016,7,1), datetime(2016,7,2), n_pad_files=2, file_type="flux",
                                 file_path="/gws/nopw/j04/eo_shared_data_vol1/satellite/seviri-orac/flx")

flx_ds = xr.open_mfdataset(flx_files, combine="nested", concat_dim="t")

flx_ds = flx_ds.assign_coords(t=[parse_date(f[-46:-34]) for f in flx_files])

# Add lat and lon fields from cloud data to output dataset
dataset["lat"] = cld_ds.isel(t=0).lat.compute()
dataset["lon"] = cld_ds.isel(t=0).lon.compute()

# Add area of each pixel
def get_area_from_lat_lon(lat, lon):
    from pyproj import Geod
    g = Geod(ellps='WGS84')
    dy, dx = np.zeros(lat.shape, dtype=float), np.zeros(lat.shape, dtype=float)
    dy[:-1] = g.inv(lon[:-1],lat[:-1],lon[1:],lat[1:])[-1]/1e3
    dx[:,:-1] = g.inv(lon[:,:-1],lat[:,:-1],lon[:,1:],lat[:,1:])[-1]/1e3
    dy[1:]+=dy[:-1]
    dy[1:-1]/=2
    dx[:,1:]+=dx[:,:-1]
    dx[:,1:-1]/=2
    area = dx*dy

    return area

areas = get_area_from_lat_lon(dataset.lat.data, dataset.lon.data)
add_dataarray_to_ds(create_dataarray(areas,
                                     dataset.lat.dims,
                                     'area',
                                     long_name="pixel area",
                                     units='km^2',
                                     dtype=np.float32),
                    dataset)

# Calculate BT change rate for each core
bt_diff_max = lambda x: np.nanmax(x[:-1] - x[1:])/15

core_bt_diff_mean = np.asarray([bt_diff_max(dcc_ds.core_step_BT_mean.data[dcc_ds.core_index_for_step.data==i])
                                for i in dcc_ds.core.data])

wh_valid_core = core_bt_diff_mean>0.5

# remap core and anvil labels based on valid core flag
def remap_labels(labels, locations):
    remapper = np.zeros(np.nanmax(labels)+1, labels.dtype)
    remapper[1:][locations] = np.arange(1, np.sum(locations)+1)

    remapped_labels = remapper[labels]

    return remapped_labels

add_dataarray_to_ds(create_dataarray(remap_labels(dcc_ds.core_label, wh_valid_core),
                                     ('t','y','x'),
                                     "core_label",
                                     long_name="labels for detected cores",
                                     units="",
                                     dtype=np.int32), dataset)

dataset.coords["core"] = np.arange(1, dataset.core_label.data.max()+1, dtype=np.int32)

def labeled_comprehension(field, labels, func, index=False, dtype=None, default=None):
    if not dtype:
        dtype = field.dtype

    if not index:
        index = range(1, int(np.nanmax(labels))+1)

    comp = ndi.labeled_comprehension(field, labels, index, func, dtype, default)

    return comp

wh_valid_core_step = labeled_comprehension(dataset.core_label.data, dcc_ds.core_step_label.data, np.any, dtype="bool", default=False)

add_dataarray_to_ds(create_dataarray(remap_labels(dcc_ds.core_step_label, wh_valid_core_step),
                                     ('t','y','x'),
                                     "core_step_label",
                                     long_name="labels for detected cores at each time step",
                                     units="",
                                     dtype=np.int32), dataset)

dataset.coords["core_step"] = np.arange(1, dataset.core_step_label.data.max()+1, dtype=np.int32)

wh_valid_anvil = labeled_comprehension(dataset.core_label,
                                       dcc_ds.thick_anvil_label.data,
                                       np.any,
                                       dtype="bool",
                                       default=False)

add_dataarray_to_ds(create_dataarray(remap_labels(dcc_ds.thick_anvil_label, wh_valid_anvil),
                                     ('t','y','x'),
                                     "thick_anvil_label",
                                     long_name="labels for detected thick anvil regions",
                                     units="",
                                     dtype=np.int32), dataset)

dataset.coords["anvil"] = np.arange(1, dataset.thick_anvil_label.data.max()+1, dtype=np.int32)

wh_valid_thick_anvil_step = labeled_comprehension(dataset.thick_anvil_label,
                                                  dcc_ds.thick_anvil_step_label.data,
                                                  np.any,
                                                  dtype="bool",
                                                  default=False)

add_dataarray_to_ds(create_dataarray(remap_labels(dcc_ds.thick_anvil_step_label, wh_valid_thick_anvil_step),
                                     ('t','y','x'),
                                     "thick_anvil_step_label",
                                     long_name="labels for detected thick anvil regions at each time step",
                                     units="",
                                     dtype=np.int32), dataset)

dataset.coords["thick_anvil_step"] = np.arange(1, dataset.thick_anvil_step_label.data.max()+1, dtype=np.int32)

add_dataarray_to_ds(create_dataarray(remap_labels(dcc_ds.thin_anvil_label, wh_valid_anvil),
                                     ('t','y','x'),
                                     "thin_anvil_label",
                                     long_name="labels for detected thin anvil regions",
                                     units="",
                                     dtype=np.int32), dataset)

wh_valid_thin_anvil_step = labeled_comprehension(dataset.thin_anvil_label,
                                                  dcc_ds.thin_anvil_step_label.data,
                                                  np.any,
                                                  dtype="bool",
                                                  default=False)

add_dataarray_to_ds(create_dataarray(remap_labels(dcc_ds.thin_anvil_step_label, wh_valid_thin_anvil_step),
                                     ('t','y','x'),
                                     "thin_anvil_step_label",
                                     long_name="labels for detected thin anvil regions at each time step",
                                     units="",
                                     dtype=np.int32), dataset)

dataset.coords["thin_anvil_step"] = np.arange(1, dataset.thin_anvil_step_label.data.max()+1, dtype=np.int32)

# Add linking indices between each label
core_step_core_index = labeled_comprehension(dataset.core_label.data,
                                             dataset.core_step_label.data,
                                             np.nanmax,
                                             dtype=np.int32,
                                             default=0)
add_dataarray_to_ds(create_dataarray(core_step_core_index,
                                     ('core_step',),
                                     "core_step_core_index",
                                     long_name="core index for each core time step",
                                     dtype=np.int32), dataset)

core_anvil_index = labeled_comprehension(dataset.thick_anvil_label.data,
                                         dataset.core_label.data,
                                         np.nanmax,
                                         dtype=np.int32,
                                         default=0)
add_dataarray_to_ds(create_dataarray(core_anvil_index,
                                     ('core',),
                                     "core_anvil_index",
                                     long_name="anvil index for each core",
                                     dtype=np.int32), dataset)

thick_anvil_step_anvil_index = labeled_comprehension(dataset.thick_anvil_label.data,
                                                     dataset.thick_anvil_step_label.data,
                                                     np.nanmax,
                                                     dtype=np.int32,
                                                     default=0)
add_dataarray_to_ds(create_dataarray(thick_anvil_step_anvil_index,
                                     ('thick_anvil_step',),
                                     "thick_anvil_step_anvil_index",
                                     long_name="anvil index for each thick anvil time step",
                                     dtype=np.int32), dataset)

thin_anvil_step_anvil_index = labeled_comprehension(dataset.thin_anvil_label.data,
                                                    dataset.thin_anvil_step_label.data,
                                                    np.nanmax,
                                                    dtype=np.int32,
                                                    default=0)
add_dataarray_to_ds(create_dataarray(thin_anvil_step_anvil_index,
                                     ('thin_anvil_step',),
                                     "thin_anvil_step_anvil_index",
                                     long_name="anvil index for each thin anvil time step",
                                     dtype=np.int32), dataset)

anvil_core_count = np.asarray([np.sum(core_anvil_index==i) for i in dataset.anvil.data])
add_dataarray_to_ds(create_dataarray(anvil_core_count,
                                     ('anvil',),
                                     "anvil_core_count",
                                     long_name="number of cores associated with anvil",
                                     dtype=np.int32), dataset)

# Add the BT diff max value for each core
core_max_bt_diff = core_bt_diff_mean[wh_valid_core]

add_dataarray_to_ds(create_dataarray(core_max_bt_diff,
                                     ('core'),
                                     "core_max_BT_diff",
                                     long_name="Maximum change in core brightness temperature per minute",
                                     units="K/minute",
                                     dtype=np.float32), dataset)

# Add edge flags for cores
core_edge_labels = np.unique(
                    np.concatenate(
                        [np.unique(dataset.core_label[:,0]),
                         np.unique(dataset.core_label[:,-1]),
                         np.unique(dataset.core_label[:,:,0]),
                         np.unique(dataset.core_label[:,:,-1])
                         ]))

core_edge_label_flag = np.zeros_like(dataset.core, bool)

if core_edge_labels[0] == 0:
    core_edge_label_flag[core_edge_labels[1:]-1] = True
else:
    core_edge_label_flag[core_edge_labels-1] = True

core_start_labels = np.unique(np.unique(dataset.core_label[0]))

core_start_label_flag = np.zeros_like(dataset.core, bool)

if core_start_labels[0] == 0:
    core_start_label_flag[core_start_labels[1:]-1] = True
else:
    core_start_label_flag[core_start_labels-1] = True

core_end_labels = np.unique(np.unique(dataset.core_label[-1]))

core_end_label_flag = np.zeros_like(dataset.core, bool)

if core_end_labels[0] == 0:
    core_end_label_flag[core_end_labels[1:]-1] = True
else:
    core_end_label_flag[core_end_labels-1] = True

add_dataarray_to_ds(create_dataarray(core_edge_label_flag, ('core',), "core_edge_label_flag",
                                     long_name="flag for cores intersecting the domain edge",
                                     dtype=bool), dataset)

add_dataarray_to_ds(create_dataarray(core_start_label_flag, ('core',), "core_start_label_flag",
                                     long_name="flag for cores intersecting the domain start time",
                                     dtype=bool), dataset)

add_dataarray_to_ds(create_dataarray(core_end_label_flag, ('core',), "core_end_label_flag",
                                     long_name="flag for cores intersecting the domain end time",
                                     dtype=bool), dataset)

# Add edge flags for thick_anvils
thick_anvil_edge_labels = np.unique(
                    np.concatenate(
                        [np.unique(dataset.thick_anvil_label[:,0]),
                         np.unique(dataset.thick_anvil_label[:,-1]),
                         np.unique(dataset.thick_anvil_label[:,:,0]),
                         np.unique(dataset.thick_anvil_label[:,:,-1])
                         ]))

thick_anvil_edge_label_flag = np.zeros_like(dataset.anvil, bool)

if thick_anvil_edge_labels[0] == 0:
    thick_anvil_edge_label_flag[thick_anvil_edge_labels[1:]-1] = True
else:
    thick_anvil_edge_label_flag[thick_anvil_edge_labels-1] = True

thick_anvil_start_labels = np.unique(np.unique(dataset.thick_anvil_label[0]))

thick_anvil_start_label_flag = np.zeros_like(dataset.anvil, bool)

if thick_anvil_start_labels[0] == 0:
    thick_anvil_start_label_flag[thick_anvil_start_labels[1:]-1] = True
else:
    thick_anvil_start_label_flag[thick_anvil_start_labels-1] = True

thick_anvil_end_labels = np.unique(np.unique(dataset.thick_anvil_label[-1]))

thick_anvil_end_label_flag = np.zeros_like(dataset.anvil, bool)

if thick_anvil_end_labels[0] == 0:
    thick_anvil_end_label_flag[thick_anvil_end_labels[1:]-1] = True
else:
    thick_anvil_end_label_flag[thick_anvil_end_labels-1] = True

add_dataarray_to_ds(create_dataarray(thick_anvil_edge_label_flag, ('anvil',), "thick_anvil_edge_label_flag",
                                     long_name="flag for thick anvils intersecting the domain edge",
                                     dtype=bool), dataset)

add_dataarray_to_ds(create_dataarray(thick_anvil_start_label_flag, ('anvil',), "thick_anvil_start_label_flag",
                                     long_name="flag for thick anvils intersecting the domain start time",
                                     dtype=bool), dataset)

add_dataarray_to_ds(create_dataarray(thick_anvil_end_label_flag, ('anvil',), "thick_anvil_end_label_flag",
                                     long_name="flag for thick anvils intersecting the domain end time",
                                     dtype=bool), dataset)

# Add edge flags for thin_anvils
thin_anvil_edge_labels = np.unique(
                    np.concatenate(
                        [np.unique(dataset.thin_anvil_label[:,0]),
                         np.unique(dataset.thin_anvil_label[:,-1]),
                         np.unique(dataset.thin_anvil_label[:,:,0]),
                         np.unique(dataset.thin_anvil_label[:,:,-1])
                         ]))

thin_anvil_edge_label_flag = np.zeros_like(dataset.anvil, bool)

if thin_anvil_edge_labels[0] == 0:
    thin_anvil_edge_label_flag[thin_anvil_edge_labels[1:]-1] = True
else:
    thin_anvil_edge_label_flag[thin_anvil_edge_labels-1] = True

thin_anvil_start_labels = np.unique(np.unique(dataset.thin_anvil_label[0]))

thin_anvil_start_label_flag = np.zeros_like(dataset.anvil, bool)

if thin_anvil_start_labels[0] == 0:
    thin_anvil_start_label_flag[thin_anvil_start_labels[1:]-1] = True
else:
    thin_anvil_start_label_flag[thin_anvil_start_labels-1] = True

thin_anvil_end_labels = np.unique(np.unique(dataset.thin_anvil_label[-1]))

thin_anvil_end_label_flag = np.zeros_like(dataset.anvil, bool)

if thin_anvil_end_labels[0] == 0:
    thin_anvil_end_label_flag[thin_anvil_end_labels[1:]-1] = True
else:
    thin_anvil_end_label_flag[thin_anvil_end_labels-1] = True

add_dataarray_to_ds(create_dataarray(thin_anvil_edge_label_flag, ('anvil',), "thin_anvil_edge_label_flag",
                                     long_name="flag for thin anvils intersecting the domain edge",
                                     dtype=bool), dataset)

add_dataarray_to_ds(create_dataarray(thin_anvil_start_label_flag, ('anvil',), "thin_anvil_start_label_flag",
                                     long_name="flag for thin anvils intersecting the domain start time",
                                     dtype=bool), dataset)

add_dataarray_to_ds(create_dataarray(thin_anvil_end_label_flag, ('anvil',), "thin_anvil_end_label_flag",
                                     long_name="flag for thin anvils intersecting the domain end time",
                                     dtype=bool), dataset)

# Pixel count and area
core_total_pixels = np.bincount(dataset.core_label.data.ravel())[1:]
add_dataarray_to_ds(create_dataarray(core_total_pixels, ('core',), "core_pixel_count",
                                     long_name="total number of pixels for core",
                                     dtype=np.int32), dataset)

core_step_pixels = np.bincount(dataset.core_step_label.data.ravel())[1:]
add_dataarray_to_ds(create_dataarray(core_step_pixels, ('core_step',), "core_step_pixel_count",
                                     long_name="total number of pixels for core at time step",
                                     dtype=np.int32), dataset)

core_total_area = labeled_comprehension(dataset.area.data[np.newaxis, ...],
                                        dataset.core_label.data,
                                        np.nansum,
                                        dtype=np.float32,
                                        default=np.nan)
add_dataarray_to_ds(create_dataarray(core_total_area, ('core',), "core_total_area",
                                     long_name="total area of core",
                                     dtype=np.float32), dataset)

core_step_area = labeled_comprehension(dataset.area.data[np.newaxis, ...],
                                        dataset.core_step_label.data,
                                        np.nansum,
                                        dtype=np.float32,
                                        default=np.nan)
add_dataarray_to_ds(create_dataarray(core_step_area, ('core_step',), "core_step_area",
                                     long_name="area of core at time step",
                                     dtype=np.float32), dataset)

core_step_max_area_index = np.asarray([dataset.core_step[dataset.core_step_core_index.data==i][np.argmax(core_step_area[dataset.core_step_core_index.data==i])]
                                       for i in dataset.core.data])

core_max_area = core_step_area[core_step_max_area_index-1]

add_dataarray_to_ds(create_dataarray(core_max_area, ('core',), "core_max_area",
                                     long_name="maximum area of core",
                                     dtype=np.float32), dataset)

# Time stats for core
core_start_t = labeled_comprehension(dataset.t.data[:, np.newaxis, np.newaxis],
                                     dataset.core_label.data,
                                     np.nanmin,
                                     dtype="datetime64[ns]",
                                     default=None)
add_dataarray_to_ds(create_dataarray(core_start_t, ('core',), "core_start_t",
                                     long_name="initial detection time of core",
                                     dtype="datetime64[ns]"), dataset)

core_end_t = labeled_comprehension(dataset.t.data[:, np.newaxis, np.newaxis],
                                     dataset.core_label.data,
                                     np.nanmax,
                                     dtype="datetime64[ns]",
                                     default=None)
add_dataarray_to_ds(create_dataarray(core_end_t, ('core',), "core_end_t",
                                         long_name="final detection time of core",
                                         dtype="datetime64[ns]"), dataset)

add_dataarray_to_ds(create_dataarray(core_end_t-core_start_t, ('core',), "core_lifetime",
                                         long_name="total lifetime of core",
                                         dtype="timedelta64[ns]"), dataset)

core_step_t = labeled_comprehension(dataset.t.data[:, np.newaxis, np.newaxis],
                                     dataset.core_step_label.data,
                                     np.nanmin,
                                     dtype="datetime64[ns]",
                                     default=None)
add_dataarray_to_ds(create_dataarray(core_step_t, ('core_step',), "core_step_t",
                                         long_name="time of core at time step",
                                         dtype="datetime64[ns]"), dataset)

core_max_area_t = core_step_t[core_step_max_area_index-1]
add_dataarray_to_ds(create_dataarray(core_max_area_t, ('core',), "core_max_area_t",
                                     long_name="time of core maximum area",
                                     dtype="datetime64[ns]"), dataset)

# Pixel count and area for thick anvil
thick_anvil_total_pixels = np.bincount(dataset.thick_anvil_label.data.ravel())[1:]
add_dataarray_to_ds(create_dataarray(thick_anvil_total_pixels, ('anvil',), "thick_anvil_pixel_count",
                                     long_name="total number of pixels for thick anvil",
                                     dtype=np.int32), dataset)

thick_anvil_step_pixels = np.bincount(dataset.thick_anvil_step_label.data.ravel())[1:]
add_dataarray_to_ds(create_dataarray(thick_anvil_step_pixels, ('thick_anvil_step',), "thick_anvil_step_pixel_count",
                                     long_name="total number of pixels for thick anvil at time step",
                                     dtype=np.int32), dataset)

thick_anvil_total_area = labeled_comprehension(dataset.area.data[np.newaxis, ...],
                                        dataset.thick_anvil_label.data,
                                        np.nansum,
                                        dtype=np.float32,
                                        default=np.nan)
add_dataarray_to_ds(create_dataarray(thick_anvil_total_area, ('anvil',), "thick_anvil_total_area",
                                     long_name="total area of thick anvil",
                                     dtype=np.float32), dataset)

thick_anvil_step_area = labeled_comprehension(dataset.area.data[np.newaxis, ...],
                                        dataset.thick_anvil_step_label.data,
                                        np.nansum,
                                        dtype=np.float32,
                                        default=np.nan)
add_dataarray_to_ds(create_dataarray(thick_anvil_step_area, ('thick_anvil_step',), "thick_anvil_step_area",
                                     long_name="area of thick anvil at time step",
                                     dtype=np.float32), dataset)

thick_anvil_step_max_area_index = np.asarray([dataset.thick_anvil_step[dataset.thick_anvil_step_anvil_index.data==i][np.argmax(thick_anvil_step_area[dataset.thick_anvil_step_anvil_index.data==i])]
                                       for i in dataset.anvil.data])

thick_anvil_max_area = thick_anvil_step_area[thick_anvil_step_max_area_index-1]

add_dataarray_to_ds(create_dataarray(thick_anvil_max_area, ('anvil',), "thick_anvil_max_area",
                                     long_name="maximum area of thick anvil",
                                     dtype=np.float32), dataset)

# Time stats for thick_anvil
thick_anvil_start_t = labeled_comprehension(dataset.t.data[:, np.newaxis, np.newaxis],
                                     dataset.thick_anvil_label.data,
                                     np.nanmin,
                                     dtype="datetime64[ns]",
                                     default=None)
add_dataarray_to_ds(create_dataarray(thick_anvil_start_t, ('anvil',), "thick_anvil_start_t",
                                     long_name="initial detection time of thick anvil",
                                     dtype="datetime64[ns]"), dataset)

thick_anvil_end_t = labeled_comprehension(dataset.t.data[:, np.newaxis, np.newaxis],
                                     dataset.thick_anvil_label.data,
                                     np.nanmax,
                                     dtype="datetime64[ns]",
                                     default=None)
add_dataarray_to_ds(create_dataarray(thick_anvil_end_t, ('anvil',), "thick_anvil_end_t",
                                         long_name="final detection time of thick anvil",
                                         dtype="datetime64[ns]"), dataset)

add_dataarray_to_ds(create_dataarray(thick_anvil_end_t-thick_anvil_start_t, ('anvil',), "thick_anvil_lifetime",
                                         long_name="total lifetime of thick anvil",
                                         dtype="timedelta64[ns]"), dataset)

thick_anvil_step_t = labeled_comprehension(dataset.t.data[:, np.newaxis, np.newaxis],
                                     dataset.thick_anvil_step_label.data,
                                     np.nanmin,
                                     dtype="datetime64[ns]",
                                     default=None)
add_dataarray_to_ds(create_dataarray(thick_anvil_step_t, ('thick_anvil_step',), "thick_anvil_step_t",
                                         long_name="time of thick anvil at time step",
                                         dtype="datetime64[ns]"), dataset)

thick_anvil_max_area_t = thick_anvil_step_t[thick_anvil_step_max_area_index-1]
add_dataarray_to_ds(create_dataarray(thick_anvil_max_area_t, ('anvil',), "thick_anvil_max_area_t",
                                     long_name="time of thick anvil maximum area",
                                     dtype="datetime64[ns]"), dataset)

# Pixel count and area for thin anvil
thin_anvil_total_pixels = np.bincount(dataset.thin_anvil_label.data.ravel())[1:]
add_dataarray_to_ds(create_dataarray(thin_anvil_total_pixels, ('anvil',), "thin_anvil_pixel_count",
                                     long_name="total number of pixels for thin anvil",
                                     dtype=np.int32), dataset)

thin_anvil_step_pixels = np.bincount(dataset.thin_anvil_step_label.data.ravel())[1:]
add_dataarray_to_ds(create_dataarray(thin_anvil_step_pixels, ('thin_anvil_step',), "thin_anvil_step_pixel_count",
                                     long_name="total number of pixels for thin anvil at time step",
                                     dtype=np.int32), dataset)

thin_anvil_total_area = labeled_comprehension(dataset.area.data[np.newaxis, ...],
                                        dataset.thin_anvil_label.data,
                                        np.nansum,
                                        dtype=np.float32,
                                        default=np.nan)
add_dataarray_to_ds(create_dataarray(thin_anvil_total_area, ('anvil',), "thin_anvil_total_area",
                                     long_name="total area of thin anvil",
                                     dtype=np.float32), dataset)

thin_anvil_step_area = labeled_comprehension(dataset.area.data[np.newaxis, ...],
                                        dataset.thin_anvil_step_label.data,
                                        np.nansum,
                                        dtype=np.float32,
                                        default=np.nan)
add_dataarray_to_ds(create_dataarray(thin_anvil_step_area, ('thin_anvil_step',), "thin_anvil_step_area",
                                     long_name="area of thin anvil at time step",
                                     dtype=np.float32), dataset)

thin_anvil_step_max_area_index = np.asarray([dataset.thin_anvil_step[dataset.thin_anvil_step_anvil_index.data==i][np.argmax(thin_anvil_step_area[dataset.thin_anvil_step_anvil_index.data==i])]
                                       for i in dataset.anvil.data])

thin_anvil_max_area = thin_anvil_step_area[thin_anvil_step_max_area_index-1]

add_dataarray_to_ds(create_dataarray(thin_anvil_max_area, ('anvil',), "thin_anvil_max_area",
                                     long_name="maximum area of thin anvil",
                                     dtype=np.float32), dataset)

# Time stats for thin_anvil
thin_anvil_start_t = labeled_comprehension(dataset.t.data[:, np.newaxis, np.newaxis],
                                     dataset.thin_anvil_label.data,
                                     np.nanmin,
                                     dtype="datetime64[ns]",
                                     default=None)
add_dataarray_to_ds(create_dataarray(thin_anvil_start_t, ('anvil',), "thin_anvil_start_t",
                                     long_name="initial detection time of thin anvil",
                                     dtype="datetime64[ns]"), dataset)

thin_anvil_end_t = labeled_comprehension(dataset.t.data[:, np.newaxis, np.newaxis],
                                     dataset.thin_anvil_label.data,
                                     np.nanmax,
                                     dtype="datetime64[ns]",
                                     default=None)
add_dataarray_to_ds(create_dataarray(thin_anvil_end_t, ('anvil',), "thin_anvil_end_t",
                                         long_name="final detection time of thin anvil",
                                         dtype="datetime64[ns]"), dataset)

add_dataarray_to_ds(create_dataarray(thin_anvil_end_t-thin_anvil_start_t, ('anvil',), "thin_anvil_lifetime",
                                         long_name="total lifetime of thin anvil",
                                         dtype="timedelta64[ns]"), dataset)

thin_anvil_step_t = labeled_comprehension(dataset.t.data[:, np.newaxis, np.newaxis],
                                     dataset.thin_anvil_step_label.data,
                                     np.nanmin,
                                     dtype="datetime64[ns]",
                                     default=None)
add_dataarray_to_ds(create_dataarray(thin_anvil_step_t, ('thin_anvil_step',), "thin_anvil_step_t",
                                         long_name="time of thin anvil at time step",
                                         dtype="datetime64[ns]"), dataset)

thin_anvil_max_area_t = thin_anvil_step_t[thin_anvil_step_max_area_index-1]
add_dataarray_to_ds(create_dataarray(thin_anvil_max_area_t, ('anvil',), "thin_anvil_max_area_t",
                                     long_name="time of thin anvil maximum area",
                                     dtype="datetime64[ns]"), dataset)

# Flag no growth anvils
anvil_no_growth_flag = np.asarray([True if dataset.anvil_core_count.data[i-1]==1
                                           and dataset.thick_anvil_max_area_t.data[i-1]<=dataset.core_end_t[core_anvil_index==i]
                                   else False
                                   for i in dataset.anvil.data])

add_dataarray_to_ds(create_dataarray(anvil_no_growth_flag, ('anvil',), "anvil_no_growth_flag",
                                     long_name="flag for anvils that do not grow after core activity ends",
                                     dtype=bool), dataset)

# Location and lat/lon for cores
from tobac_flow.analysis import apply_weighted_func_to_labels

area_stack = np.repeat(dataset.area.data[np.newaxis,...], dataset.t.size, 0)
lat_stack = np.repeat(dataset.lat.data[np.newaxis,...], dataset.t.size, 0)
lon_stack = np.repeat(dataset.lon.data[np.newaxis,...], dataset.t.size, 0)

xx, yy = np.meshgrid(dataset.across_track, dataset.along_track)
x_stack = np.repeat(xx[np.newaxis,...], dataset.t.size, 0)
y_stack = np.repeat(yy[np.newaxis,...], dataset.t.size, 0)

core_step_x = apply_weighted_func_to_labels(dataset.core_step_label.data,
                                            x_stack,
                                            area_stack,
                                            lambda x, w : np.average(x, weights=w))
add_dataarray_to_ds(create_dataarray(core_step_x, ('core_step',), "core_step_x",
                                     long_name="x location of core at time step",
                                     dtype=np.float32), dataset)

core_step_y = apply_weighted_func_to_labels(dataset.core_step_label.data,
                                            y_stack,
                                            area_stack,
                                            lambda x, w : np.average(x, weights=w))
add_dataarray_to_ds(create_dataarray(core_step_y, ('core_step',), "core_step_y",
                                     long_name="y location of core at time step",
                                     dtype=np.float32), dataset)

core_step_lat = apply_weighted_func_to_labels(dataset.core_step_label.data,
                                              lat_stack,
                                              area_stack,
                                              lambda x, w : np.average(x, weights=w))
add_dataarray_to_ds(create_dataarray(core_step_lat, ('core_step',), "core_step_lat",
                                     long_name="latitude of core at time step",
                                     dtype=np.float32), dataset)

core_step_lon = apply_weighted_func_to_labels(dataset.core_step_label.data,
                                              lon_stack,
                                              area_stack,
                                              lambda x, w : np.average(x, weights=w))
add_dataarray_to_ds(create_dataarray(core_step_lon, ('core_step',), "core_step_lon",
                                     long_name="longitude of core at time step",
                                     dtype=np.float32), dataset)

core_start_index = np.asarray([np.nanmin(dataset.core_step[dataset.core_step_core_index.data==i]) for i in dataset.core.data])
core_end_index = np.asarray([np.nanmax(dataset.core_step[dataset.core_step_core_index.data==i]) for i in dataset.core.data])

core_start_x = core_step_x[core_start_index-1]
core_start_y = core_step_y[core_start_index-1]
core_start_lat = core_step_lat[core_start_index-1]
core_start_lon = core_step_lon[core_start_index-1]

add_dataarray_to_ds(create_dataarray(core_start_x, ('core',), "core_start_x",
                                     long_name="initial x location of core",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(core_start_y, ('core',), "core_start_y",
                                     long_name="initial y location of core",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(core_start_lat, ('core',), "core_start_lat",
                                     long_name="initial latitude of core",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(core_start_lon, ('core',), "core_start_lon",
                                     long_name="initial longitude of core",
                                     dtype=np.float32), dataset)

# Location and lat/lon for anvils
area_stack = np.repeat(dataset.area.data[np.newaxis,...], dataset.t.size, 0)
lat_stack = np.repeat(dataset.lat.data[np.newaxis,...], dataset.t.size, 0)
lon_stack = np.repeat(dataset.lon.data[np.newaxis,...], dataset.t.size, 0)

xx, yy = np.meshgrid(dataset.across_track, dataset.along_track)
x_stack = np.repeat(xx[np.newaxis,...], dataset.t.size, 0)
y_stack = np.repeat(yy[np.newaxis,...], dataset.t.size, 0)

thick_anvil_step_x = apply_weighted_func_to_labels(dataset.thick_anvil_step_label.data,
                                            x_stack,
                                            area_stack,
                                            lambda x, w : np.average(x, weights=w))
add_dataarray_to_ds(create_dataarray(thick_anvil_step_x, ('thick_anvil_step',), "thick_anvil_step_x",
                                     long_name="x location of thick anvil at time step",
                                     dtype=np.float32), dataset)

thick_anvil_step_y = apply_weighted_func_to_labels(dataset.thick_anvil_step_label.data,
                                            y_stack,
                                            area_stack,
                                            lambda x, w : np.average(x, weights=w))
add_dataarray_to_ds(create_dataarray(thick_anvil_step_y, ('thick_anvil_step',), "thick_anvil_step_y",
                                     long_name="y location of thick anvil at time step",
                                     dtype=np.float32), dataset)

thick_anvil_step_lat = apply_weighted_func_to_labels(dataset.thick_anvil_step_label.data,
                                              lat_stack,
                                              area_stack,
                                              lambda x, w : np.average(x, weights=w))
add_dataarray_to_ds(create_dataarray(thick_anvil_step_lat, ('thick_anvil_step',), "thick_anvil_step_lat",
                                     long_name="latitude of thick anvil at time step",
                                     dtype=np.float32), dataset)

thick_anvil_step_lon = apply_weighted_func_to_labels(dataset.thick_anvil_step_label.data,
                                              lon_stack,
                                              area_stack,
                                              lambda x, w : np.average(x, weights=w))
add_dataarray_to_ds(create_dataarray(thick_anvil_step_lon, ('thick_anvil_step',), "thick_anvil_step_lon",
                                     long_name="longitude of thick anvil at time step",
                                     dtype=np.float32), dataset)

thick_anvil_start_index = np.asarray([np.nanmin(dataset.thick_anvil_step[dataset.thick_anvil_step_anvil_index.data==i]) for i in dataset.anvil.data])
thick_anvil_end_index = np.asarray([np.nanmax(dataset.thick_anvil_step[dataset.thick_anvil_step_anvil_index.data==i]) for i in dataset.anvil.data])

thick_anvil_start_x = thick_anvil_step_x[thick_anvil_start_index-1]
thick_anvil_start_y = thick_anvil_step_y[thick_anvil_start_index-1]
thick_anvil_start_lat = thick_anvil_step_lat[thick_anvil_start_index-1]
thick_anvil_start_lon = thick_anvil_step_lon[thick_anvil_start_index-1]

add_dataarray_to_ds(create_dataarray(thick_anvil_start_x, ('anvil',), "anvil_start_x",
                                     long_name="initial x location of anvil",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(thick_anvil_start_y, ('anvil',), "anvil_start_y",
                                     long_name="initial y location of anvil",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(thick_anvil_start_lat, ('anvil',), "anvil_start_lat",
                                     long_name="initial latitude of anvil",
                                     dtype=np.float32), dataset)
add_dataarray_to_ds(create_dataarray(thick_anvil_start_lon, ('anvil',), "anvil_start_lon",
                                     long_name="initial longitude of anvil",
                                     dtype=np.float32), dataset)

thin_anvil_step_x = apply_weighted_func_to_labels(dataset.thin_anvil_step_label.data,
                                            x_stack,
                                            area_stack,
                                            lambda x, w : np.average(x, weights=w))
add_dataarray_to_ds(create_dataarray(thin_anvil_step_x, ('thin_anvil_step',), "thin_anvil_step_x",
                                     long_name="x location of thin anvil at time step",
                                     dtype=np.float32), dataset)

thin_anvil_step_y = apply_weighted_func_to_labels(dataset.thin_anvil_step_label.data,
                                            y_stack,
                                            area_stack,
                                            lambda x, w : np.average(x, weights=w))
add_dataarray_to_ds(create_dataarray(thin_anvil_step_y, ('thin_anvil_step',), "thin_anvil_step_y",
                                     long_name="y location of thin anvil at time step",
                                     dtype=np.float32), dataset)

thin_anvil_step_lat = apply_weighted_func_to_labels(dataset.thin_anvil_step_label.data,
                                              lat_stack,
                                              area_stack,
                                              lambda x, w : np.average(x, weights=w))
add_dataarray_to_ds(create_dataarray(thin_anvil_step_lat, ('thin_anvil_step',), "thin_anvil_step_lat",
                                     long_name="latitude of thin anvil at time step",
                                     dtype=np.float32), dataset)

thin_anvil_step_lon = apply_weighted_func_to_labels(dataset.thin_anvil_step_label.data,
                                              lon_stack,
                                              area_stack,
                                              lambda x, w : np.average(x, weights=w))
add_dataarray_to_ds(create_dataarray(thin_anvil_step_lon, ('thin_anvil_step',), "thin_anvil_step_lon",
                                     long_name="longitude of thin anvil at time step",
                                     dtype=np.float32), dataset)


# Get cloud and flux statistics
from tobac_flow.analysis import apply_func_to_labels, apply_weighted_func_to_labels

cld_weights = np.copy(area_stack)
cld_weights[cld_ds.qcflag.compute().data!=0] = 0

def weighted_statistics_on_labels(labels, da, cld_weights, name=None, dim=None, dtype=None):
    if not dim:
        dim = labels.name.split("_label")[0]
    if not name:
        name = labels.name.split("_label")[0]
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

    weighted_average = lambda x, w : np.average(x, weights=w)
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


# cot, cer, ctp, stemp, cth, ctt, cwp
# for field in (cld_ds.cot, cld_ds.cer, cld_ds.ctp, cld_ds.stemp, cld_ds.cth, cld_ds.ctt, cld_ds.cwp):
for field in (cld_ds.cot, cld_ds.cer, cld_ds.cth, cld_ds.ctt, cld_ds.cwp):
    [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.core_label,
                                                                              field.compute(),
                                                                              cld_weights,
                                                                              name='core',
                                                                              dim='core',
                                                                              dtype=np.float32)]

# for field in (cld_ds.cot, cld_ds.cer, cld_ds.ctp, cld_ds.stemp, cld_ds.cth, cld_ds.ctt, cld_ds.cwp):
for field in (cld_ds.cot, cld_ds.cer, cld_ds.cth, cld_ds.ctt, cld_ds.cwp):
    [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.thick_anvil_label,
                                                                              field.compute(),
                                                                              cld_weights,
                                                                              name='thick_anvil',
                                                                              dim='anvil',
                                                                              dtype=np.float32)]

# for field in (cld_ds.cot, cld_ds.cer, cld_ds.ctp, cld_ds.stemp, cld_ds.cth, cld_ds.ctt, cld_ds.cwp):
for field in (cld_ds.cot, cld_ds.cer, cld_ds.cth, cld_ds.ctt, cld_ds.cwp):
    [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.thin_anvil_label,
                                                                              field.compute(),
                                                                              cld_weights,
                                                                              name='thin_anvil',
                                                                              dim='anvil',
                                                                              dtype=np.float32)]

for field in (cld_ds.cot, cld_ds.cer, cld_ds.cth, cld_ds.ctt, cld_ds.cwp):
    [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.core_step_label,
                                                                              field.compute(),
                                                                              cld_weights,
                                                                              name='core_step',
                                                                              dim='core_step',
                                                                              dtype=np.float32)]

# for field in (cld_ds.cot, cld_ds.cer, cld_ds.ctp, cld_ds.stemp, cld_ds.cth, cld_ds.ctt, cld_ds.cwp):
for field in (cld_ds.cot, cld_ds.cer, cld_ds.cth, cld_ds.ctt, cld_ds.cwp):
    [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.thick_anvil_step_label,
                                                                              field.compute(),
                                                                              cld_weights,
                                                                              name='thick_anvil_step',
                                                                              dim='thick_anvil_step',
                                                                              dtype=np.float32)]

# for field in (cld_ds.cot, cld_ds.cer, cld_ds.ctp, cld_ds.stemp, cld_ds.cth, cld_ds.ctt, cld_ds.cwp):
for field in (cld_ds.cot, cld_ds.cer, cld_ds.cth, cld_ds.ctt, cld_ds.cwp):
    [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.thin_anvil_step_label,
                                                                              field.compute(),
                                                                              cld_weights,
                                                                              name='thin_anvil_step',
                                                                              dim='thin_anvil_step',
                                                                              dtype=np.float32)]

# toa_sw, toa_lw, toa_net
toa_net = flx_ds.toa_swdn-flx_ds.toa_swup-flx_ds.toa_lwup
# toa_net.attrs["name"] = toa_net
toa_clr = flx_ds.toa_swdn-flx_ds.toa_swup_clr-flx_ds.toa_lwup_clr
# toa_clr.attrs["name"] = toa_cld
toa_cld = toa_net-toa_clr
# toa_cld.attrs["name"] = toa_cld
toa_net = create_dataarray(toa_net.data, flx_ds.dims, "toa_net", units="")
toa_cld = create_dataarray(toa_cld.data, flx_ds.dims, "toa_cld", units="")

for field in (toa_net, toa_cld):
    [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.core_label,
                                                                              field.compute(),
                                                                              area_stack,
                                                                              name='core',
                                                                              dim='core',
                                                                              dtype=np.float32)]

for field in (toa_net, toa_cld):
    [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.thick_anvil_label,
                                                                              field.compute(),
                                                                              area_stack,
                                                                              name='thick_anvil',
                                                                              dim='anvil',
                                                                              dtype=np.float32)]
for field in (toa_net, toa_cld):
    [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.thin_anvil_label,
                                                                              field.compute(),
                                                                              area_stack,
                                                                              name='thin_anvil',
                                                                              dim='anvil',
                                                                              dtype=np.float32)]

for field in (toa_net, toa_cld):
    [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.core_step_label,
                                                                              field.compute(),
                                                                              area_stack,
                                                                              name='core_step',
                                                                              dim='core_step',
                                                                              dtype=np.float32)]

for field in (toa_net, toa_cld):
    [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.thick_anvil_step_label,
                                                                              field.compute(),
                                                                              area_stack,
                                                                              name='thick_anvil_step',
                                                                              dim='thick_anvil_step',
                                                                              dtype=np.float32)]
for field in (toa_net, toa_cld):
    [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.thin_anvil_step_label,
                                                                              field.compute(),
                                                                              area_stack,
                                                                              name='thin_anvil_step',
                                                                              dim='thin_anvil_step',
                                                                              dtype=np.float32)]
