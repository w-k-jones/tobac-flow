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
parser = argparse.ArgumentParser(description="""Detect and track DCCs in SEVIRI data""")
parser.add_argument('date', help='Date on which to start process', type=str)
parser.add_argument('hours', help='Number of hours to process', type=float)
parser.add_argument('-x0', help='Initial subset x location', default=0, type=int)
parser.add_argument('-x1', help='End subset x location', default=2081, type=int)
parser.add_argument('-y0', help='Initial subset y location', default=0, type=int)
parser.add_argument('-y1', help='End subset y location', default=1601, type=int)
parser.add_argument('-sd', help='Directory to save preprocess files',
                    default='../data/dcc_detect_seviri', type=str)
parser.add_argument('-fd', help='SEVIRI file directory',
                    default='../data/SEVIRI_ORAC/', type=str)

start_time = datetime.now()
args = parser.parse_args()
start_date = parse_date(args.date, fuzzy=True)
end_date = start_date + timedelta(hours=args.hours)

x0 = int(args.x0)
x1 = int(args.x1)
y0 = int(args.y0)
y1 = int(args.y1)

save_dir = args.sd
if not os.path.isdir(save_dir):
    try:
        os.makedirs(save_dir)
    except (FileExistsError, OSError):
        pass

save_name = 'detected_dccs_SEVIRI_S%s_E%s_X%04d_%04d_Y%04d_%04d.nc' % \
    (start_date.strftime('%Y%m%d_%H0000'), end_date.strftime('%Y%m%d_%H0000'),
     x0, x1, y0, y1)

save_path = os.path.join(save_dir, save_name)

print("Saving output to:", save_path)

seviri_data_path = args.fd
if not os.path.isdir(seviri_data_path):
    try:
        os.makedirs(seviri_data_path)
    except (FileExistsError, OSError):
        pass

def main(start_date, end_date, x0, x1, y0, y1, save_path, seviri_data_path):
    from tobac_flow.flow import Flow
    from tobac_flow.dataset import get_datetime_from_coord, get_time_diff_from_coord, create_new_goes_ds, add_dataarray_to_ds, create_dataarray
    from tobac_flow.detection import detect_growth_markers, detect_growth_markers_multichannel, edge_watershed
    from tobac_flow.analysis import filter_labels_by_length, filter_labels_by_length_and_mask, apply_func_to_labels, apply_weighted_func_to_labels, get_label_stats, get_stats_for_labels, slice_label_da
    from tobac_flow.validation import get_min_dist_for_objects, get_marker_distance
    # from tobac_flow.abi import get_abi_proj
    from tobac_flow.dataloader import seviri_dataloader

    print(datetime.now(),'Loading SEVIRI data', flush=True)
    print('Loading SEVIRI data from:',seviri_data_path , flush=True)
    bt, wvd, swd, dataset = seviri_dataloader(start_date, end_date, n_pad_files=2,
                                              x0=x0, x1=x1, y0=y0, y1=y1,
                                              file_path=seviri_data_path,
                                              return_new_ds=True)

    print("Dataloader dtype:", bt.dtype, wvd.dtype, swd.dtype)

    print(datetime.now(),'Calculating flow field', flush=True)
    flow_kwargs = {'pyr_scale':0.5, 'levels':5, 'winsize':16, 'iterations':3,
                   'poly_n':5, 'poly_sigma':1.1, 'flags':256}

    flow = Flow(bt, flow_kwargs=flow_kwargs, smoothing_passes=3)

    print(datetime.now(),'Detecting growth markers', flush=True)
    wvd_growth, bt_growth, growth_markers = detect_growth_markers_multichannel(flow, wvd, bt,
                                                                               overlap=0.5,
                                                                               subsegment_shrink=0,
                                                                               min_length=3,
                                                                               lower_threshold=0.125,
                                                                               upper_threshold=0.25)

    print('WVD growth above threshold: area =', np.sum(wvd_growth.data>=0.5))
    print('BT growth above threshold: area =', np.sum(bt_growth.data<=-0.5))
    print('Detected markers: area =', np.sum(growth_markers.data!=0))
    print('Detected markers: n =', growth_markers.data.max())

    print("Growth dtype:", wvd_growth.dtype, bt_growth.dtype)
    print("Marker dtype:", growth_markers.dtype)

    print(datetime.now(), 'Detecting thick anvil region', flush=True)
    inner_watershed = edge_watershed(flow, wvd-swd, growth_markers!=0, -5, -15,
                                     verbose=True)
    print("Watershed dtype:", inner_watershed.dtype)
    print(datetime.now(), 'Labelling thick anvil region', flush=True)
    inner_watershed = flow.label(inner_watershed, overlap=0.75,
                                 subsegment_shrink=0.25,
                                 dtype=np.int32)
    inner_watershed = filter_labels_by_length_and_mask(inner_watershed,
                                                       growth_markers.data!=0,
                                                       2)

    print('Detected thick anvils: area =', np.sum(inner_watershed!=0), flush=True)
    print('Detected thick anvils: n =', inner_watershed.max(), flush=True)

    print("Label dtype:", inner_watershed.dtype)

    print(datetime.now(), 'Detecting thin anvil region', flush=True)
    outer_watershed = edge_watershed(flow, wvd+swd, inner_watershed, 0, -10,
                                     verbose=True)
    print('Detected thin anvils: area =', np.sum(outer_watershed!=0), flush=True)

    print("Outer label dtype:", outer_watershed.dtype)

    print(datetime.now(), 'Preparing output', flush=True)

    # Growth value
    add_dataarray_to_ds(create_dataarray(wvd_growth, ('t','y','x'), "wvd_growth_rate",
                                         long_name="detected wvd cooling rate", units="K/min",
                                         dtype=np.float32), dataset)
    add_dataarray_to_ds(create_dataarray(bt_growth, ('t','y','x'), "bt_growth_rate",
                                         long_name="detected bt cooling rate", units="K/min",
                                         dtype=np.float32), dataset)

    # Detected growth_regions
    add_dataarray_to_ds(create_dataarray(growth_markers, ('t','y','x'), "core_label",
                                         long_name="labels for detected cores", units="",
                                         dtype=np.int32), dataset)

    dataset.coords["core"] = np.arange(1, growth_markers.max()+1, dtype=np.int32)

    for field in (bt, dataset.wvd_growth_rate, dataset.bt_growth_rate):
        [add_dataarray_to_ds(da, dataset) for da in get_stats_for_labels(dataset.core_label, field, dim='core', dtype=np.float32)]

    # Now get individual step
    core_step_label = slice_label_da(dataset.core_label)
    add_dataarray_to_ds(core_step_label, dataset)
    dataset.coords["core_step"] = np.arange(1, core_step_label.max()+1, dtype=np.int32)

    for field in (bt, dataset.wvd_growth_rate, dataset.bt_growth_rate):
        [add_dataarray_to_ds(da, dataset) for da in get_stats_for_labels(dataset.core_step_label, field, dim='core_step', dtype=np.float32)]

    # Now we have stats of each field for each core and each core time step
    # Next get other data: weighted x,y,lat,lon locations for each step, t per step
    tt, yy, xx = np.meshgrid(dataset.t, dataset.along_track, dataset.across_track, indexing='ij')

    core_step_x = apply_func_to_labels(core_step_label.data, xx, np.nanmean)
    add_dataarray_to_ds(create_dataarray(core_step_x, ('core_step',), "core_step_x",
                                         long_name="x location of core at time step",
                                         dtype=np.float64), dataset)

    core_step_y = apply_func_to_labels(core_step_label.data, yy, np.nanmean)
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

    # p = get_abi_proj(dataset)
    # core_step_lons, core_step_lats = p(core_step_x*dataset.goes_imager_projection.perspective_point_height,
    #                                    core_step_y*dataset.goes_imager_projection.perspective_point_height,
    #                                    inverse=True)
    # add_dataarray_to_ds(create_dataarray(core_step_lons, ('core_step',), "core_step_lon",
    #                                      long_name="longitude of core at time step",
    #                                      dtype=np.float32), dataset)
    # add_dataarray_to_ds(create_dataarray(core_step_lats, ('core_step',), "core_step_lat",
    #                                      long_name="latitude of core at time step",
    #                                      dtype=np.float32), dataset)

    # Get area in 3d
    # aa = np.meshgrid(dataset.t, dataset.area, indexing='ij')[1].reshape(tt.shape)

    core_step_pixels = np.bincount(dataset.core_step_label.data.ravel())[1:]
    add_dataarray_to_ds(create_dataarray(core_step_pixels, ('core_step',), "core_step_pixels",
                                         long_name="number of pixels for core at time step",
                                         dtype=np.int32), dataset)

    core_total_pixels = np.bincount(dataset.core_label.data.ravel())[1:]
    add_dataarray_to_ds(create_dataarray(core_total_pixels, ('core',), "core_total_pixels",
                                         long_name="total number of pixels for core",
                                         dtype=np.int32), dataset)

    # core_step_area = apply_func_to_labels(dataset.core_step_label.data, aa, np.nansum)
    # add_dataarray_to_ds(create_dataarray(core_step_area, ('core_step',), "core_step_area",
    #                                      long_name="area of core at time step",
    #                                      dtype=np.float32), dataset)
    #
    # core_total_area = apply_func_to_labels(dataset.core_label.data, aa, np.nansum)
    # add_dataarray_to_ds(create_dataarray(core_total_area, ('core',), "core_total_area",
    #                                      long_name="total area of core",
    #                                      dtype=np.float32), dataset)

    core_index_for_step = apply_func_to_labels(dataset.core_step_label.data,
                                               dataset.core_label.data, np.nanmax)
    add_dataarray_to_ds(create_dataarray(core_index_for_step, ('core_step',), "core_index_for_step",
                                         long_name="core index for each core time step",
                                         dtype=np.int32), dataset)

    # Now for thick anvils
    add_dataarray_to_ds(create_dataarray(inner_watershed, ('t','y','x'), "thick_anvil_label",
                                         long_name="labels for detected thick anvil regions", units="",
                                         dtype=np.int32), dataset)

    dataset.coords["anvil"] = np.arange(1, inner_watershed.max()+1, dtype=np.int32)

    get_label_stats(dataset.thick_anvil_label, dataset)

    for field in (bt, dataset.wvd_growth_rate, dataset.bt_growth_rate):
        [add_dataarray_to_ds(da, dataset) for da in get_stats_for_labels(dataset.thick_anvil_label, field, dim='anvil', dtype=np.float32)]

    thick_anvil_step_label = slice_label_da(dataset.thick_anvil_label)
    add_dataarray_to_ds(thick_anvil_step_label, dataset)
    dataset.coords["anvil_step"] = np.arange(1, thick_anvil_step_label.max()+1, dtype=np.int32)

    # Now get individual step
    for field in (bt, dataset.wvd_growth_rate, dataset.bt_growth_rate):
        [add_dataarray_to_ds(da, dataset) for da in get_stats_for_labels(dataset.thick_anvil_step_label, field, dim='anvil_step', dtype=np.float32)]

    # Now we have stats of each field for each core and each core time step
    # Next get other data: weighted x,y,lat,lon locations for each step, t per step
    tt, yy, xx = np.meshgrid(dataset.t, dataset.along_track, dataset.across_track, indexing='ij')

    thick_anvil_step_x = apply_func_to_labels(thick_anvil_step_label.data, xx, np.nanmean)
    add_dataarray_to_ds(create_dataarray(thick_anvil_step_x, ('anvil_step',), "thick_anvil_step_x",
                                         long_name="x location of thick_anvil at time step",
                                         dtype=np.float64), dataset)

    thick_anvil_step_y = apply_func_to_labels(thick_anvil_step_label.data, yy, np.nanmean)
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

    # p = get_abi_proj(dataset)
    # thick_anvil_step_lons, thick_anvil_step_lats = p(thick_anvil_step_x*dataset.goes_imager_projection.perspective_point_height,
    #                                    thick_anvil_step_y*dataset.goes_imager_projection.perspective_point_height,
    #                                    inverse=True)
    # add_dataarray_to_ds(create_dataarray(thick_anvil_step_lons, ('anvil_step',), "thick_anvil_step_lon",
    #                                      long_name="longitude of thick_anvil at time step",
    #                                      dtype=np.float32), dataset)
    # add_dataarray_to_ds(create_dataarray(thick_anvil_step_lats, ('anvil_step',), "thick_anvil_step_lat",
    #                                      long_name="latitude of thick_anvil at time step",
    #                                      dtype=np.float32), dataset)

    # thick_anvil_step_glm_count = apply_func_to_labels(thick_anvil_step_label.data,
    #                                            dataset.glm_flashes.data, np.nansum)
    # add_dataarray_to_ds(create_dataarray(thick_anvil_step_glm_count, ('anvil_step',), "thick_anvil_step_glm_count",
    #                                      long_name="number of GLM flashes for thick_anvil at time step",
    #                                      dtype=np.int32), dataset)
    #
    # thick_anvil_glm_count = apply_func_to_labels(dataset.thick_anvil_label.data,
    #                                       dataset.glm_flashes.data, np.nansum)
    # add_dataarray_to_ds(create_dataarray(thick_anvil_glm_count, ('anvil',), "thick_anvil_total_glm_count",
    #                                      long_name="total number of GLM flashes for thick_anvil",
    #                                      dtype=np.int32), dataset)
    # Get area in 3d
    # aa = np.meshgrid(dataset.t, dataset.area, indexing='ij')[1].reshape(tt.shape)

    thick_anvil_step_pixels = np.bincount(dataset.thick_anvil_step_label.data.ravel())[1:]
    add_dataarray_to_ds(create_dataarray(thick_anvil_step_pixels, ('anvil_step',), "thick_anvil_step_pixels",
                                         long_name="number of pixels for thick_anvil at time step",
                                         dtype=np.int32), dataset)

    thick_anvil_total_pixels = np.bincount(dataset.thick_anvil_label.data.ravel())[1:]
    add_dataarray_to_ds(create_dataarray(thick_anvil_total_pixels, ('anvil',), "thick_anvil_total_pixels",
                                         long_name="total number of pixels for thick_anvil",
                                         dtype=np.int32), dataset)

    # thick_anvil_step_area = apply_func_to_labels(dataset.thick_anvil_step_label.data, aa, np.nansum)
    # add_dataarray_to_ds(create_dataarray(thick_anvil_step_area, ('anvil_step',), "thick_anvil_step_area",
    #                                      long_name="area of thick_anvil at time step",
    #                                      dtype=np.float32), dataset)
    #
    # thick_anvil_total_area = apply_func_to_labels(dataset.thick_anvil_label.data, aa, np.nansum)
    # add_dataarray_to_ds(create_dataarray(thick_anvil_total_area, ('anvil',), "thick_anvil_total_area",
    #                                      long_name="total area of thick_anvil",
    #                                      dtype=np.float32), dataset)

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

    # dataset.coords["anvil"] = np.arange(1, inner_watershed.max()+1, dtype=np.int32)

    get_label_stats(dataset.thin_anvil_label, dataset)

    for field in (bt, wvd, swd, dataset.wvd_growth_rate, dataset.bt_growth_rate):
        [add_dataarray_to_ds(da, dataset) for da in get_stats_for_labels(dataset.thin_anvil_label, field, dim='anvil', dtype=np.float32)]

    thin_anvil_step_label = slice_label_da(dataset.thin_anvil_label)
    add_dataarray_to_ds(thin_anvil_step_label, dataset)
    dataset.coords["thin_anvil_step"] = np.arange(1, thin_anvil_step_label.max()+1, dtype=np.int32)

    # Now get individual step
    for field in (bt, wvd, swd, dataset.wvd_growth_rate, dataset.bt_growth_rate):
        [add_dataarray_to_ds(da, dataset) for da in get_stats_for_labels(dataset.thin_anvil_step_label, field, dim='thin_anvil_step', dtype=np.float32)]

    # Now we have stats of each field for each core and each core time step
    # Next get other data: weighted x,y,lat,lon locations for each step, t per step
    tt, yy, xx = np.meshgrid(dataset.t, dataset.along_track, dataset.across_track, indexing='ij')

    thin_anvil_step_x = apply_func_to_labels(thin_anvil_step_label.data, xx, np.nanmean)
    add_dataarray_to_ds(create_dataarray(thin_anvil_step_x, ('thin_anvil_step',), "thin_anvil_step_x",
                                         long_name="x location of thin_anvil at time step",
                                         dtype=np.float64), dataset)

    thin_anvil_step_y = apply_func_to_labels(thin_anvil_step_label.data, yy, np.nanmean)
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

    # p = get_abi_proj(dataset)
    # thin_anvil_step_lons, thin_anvil_step_lats = p(thin_anvil_step_x*dataset.goes_imager_projection.perspective_point_height,
    #                                    thin_anvil_step_y*dataset.goes_imager_projection.perspective_point_height,
    #                                    inverse=True)
    # add_dataarray_to_ds(create_dataarray(thin_anvil_step_lons, ('thin_anvil_step',), "thin_anvil_step_lon",
    #                                      long_name="longitude of thin_anvil at time step",
    #                                      dtype=np.float32), dataset)
    # add_dataarray_to_ds(create_dataarray(thin_anvil_step_lats, ('thin_anvil_step',), "thin_anvil_step_lat",
    #                                      long_name="latitude of thin_anvil at time step",
    #                                      dtype=np.float32), dataset)

    # thin_anvil_step_glm_count = apply_func_to_labels(thin_anvil_step_label.data,
    #                                            dataset.glm_flashes.data, np.nansum)
    # add_dataarray_to_ds(create_dataarray(thin_anvil_step_glm_count, ('thin_anvil_step',), "thin_anvil_step_glm_count",
    #                                      long_name="number of GLM flashes for thin_anvil at time step",
    #                                      dtype=np.int32), dataset)
    #
    # thin_anvil_glm_count = apply_func_to_labels(dataset.thin_anvil_label.data,
    #                                       dataset.glm_flashes.data, np.nansum)
    # add_dataarray_to_ds(create_dataarray(thin_anvil_glm_count, ('anvil',), "thin_anvil_total_glm_count",
    #                                      long_name="total number of GLM flashes for thin_anvil",
    #                                      dtype=np.int32), dataset)
    # Get area in 3d
    # aa = np.meshgrid(dataset.t, dataset.area, indexing='ij')[1].reshape(tt.shape)

    thin_anvil_step_pixels = np.bincount(dataset.thin_anvil_step_label.data.ravel())[1:]
    add_dataarray_to_ds(create_dataarray(thin_anvil_step_pixels, ('thin_anvil_step',), "thin_anvil_step_pixels",
                                         long_name="number of pixels for thin_anvil at time step",
                                         dtype=np.int32), dataset)

    thin_anvil_total_pixels = np.bincount(dataset.thin_anvil_label.data.ravel())[1:]
    add_dataarray_to_ds(create_dataarray(thin_anvil_total_pixels, ('anvil',), "thin_anvil_total_pixels",
                                         long_name="total number of pixels for thin_anvil",
                                         dtype=np.int32), dataset)

    # thin_anvil_step_area = apply_func_to_labels(dataset.thin_anvil_step_label.data, aa, np.nansum)
    # add_dataarray_to_ds(create_dataarray(thin_anvil_step_area, ('thin_anvil_step',), "thin_anvil_step_area",
    #                                      long_name="area of thin_anvil at time step",
    #                                      dtype=np.float32), dataset)

    # thin_anvil_total_area = apply_func_to_labels(dataset.thin_anvil_label.data, aa, np.nansum)
    # add_dataarray_to_ds(create_dataarray(thin_anvil_total_area, ('anvil',), "thin_anvil_total_area",
    #                                      long_name="total area of thin_anvil",
    #                                      dtype=np.float32), dataset)

    thin_anvil_index_for_step = apply_func_to_labels(dataset.thin_anvil_step_label.data,
                                               dataset.thin_anvil_label.data, np.nanmax)
    add_dataarray_to_ds(create_dataarray(thin_anvil_index_for_step, ('thin_anvil_step',), "thin_anvil_index_for_step",
                                         long_name="thin_anvil index for each thin_anvil time step",
                                         dtype=np.int32), dataset)

    print(datetime.now(), 'Saving to %s' % (save_path), flush=True)
    # Add compression encoding
    comp = dict(zlib=True, complevel=5, shuffle=True)
    for var in dataset.data_vars:
        dataset[var].encoding.update(comp)

    dataset.to_netcdf(save_path)

    dataset.close()

if __name__=="__main__":
    start_time = datetime.now()
    print(start_time, 'Commencing DCC detection', flush=True)

    print('Start date:', start_date)
    print('End date:', end_date)
    print('x0,x1,y0,y1:', x0, x1, y0 ,y1)
    print('Output save path:', save_path)
    print('SEVIRI data path:', seviri_data_path)

    main(start_date, end_date, x0, x1, y0, y1, save_path, seviri_data_path)

    print(datetime.now(), 'Finished successfully, time elapsed:', datetime.now()-start_time, flush=True)
