import os
import numpy as np
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from scipy import ndimage as ndi
from scipy import stats

from tobac_flow import io, abi
from tobac_flow.flow import Flow
from tobac_flow.dataloader import goes_dataloader
from tobac_flow.dataset import get_time_diff_from_coord, add_dataarray_to_ds, create_dataarray
from tobac_flow.analysis import (get_label_stats, apply_weighted_func_to_labels,
                                 weighted_statistics_on_labels, find_object_lengths, remap_labels,
                                 labeled_comprehension, mask_labels)
from tobac_flow.detection import get_curvature_filter, get_growth_rate
from tobac_flow.label import slice_labels

import argparse
parser = argparse.ArgumentParser(description="""Detect and track DCCs in GOES-16 ABI data""")
parser.add_argument('date', help='Date on which to start process', type=str)
parser.add_argument('hours', help='Number of hours to process', type=float)
parser.add_argument('-sat', help='GOES satellite', default=16, type=int)
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

save_name = 'detected_dccs_G%02d_S%s_E%s_X%04d_%04d_Y%04d_%04d.nc' % \
    (satellite, start_date.strftime('%Y%m%d_%H0000'), end_date.strftime('%Y%m%d_%H0000'),
     x0, x1, y0, y1)

save_path = os.path.join(save_dir, save_name)

print("Saving output to:", save_path)

goes_data_path = args.gd
if not os.path.isdir(goes_data_path):
    try:
        os.makedirs(goes_data_path)
    except (FileExistsError, OSError):
        pass

def main(start_date, end_date, satellite, x0, x1, y0, y1, save_path, goes_data_path):

    print(datetime.now(),'Loading ABI data', flush=True)
    print('Loading goes data from:',goes_data_path, flush=True)

    bt, wvd, swd, dataset = goes_dataloader(start_date, end_date,
                                            n_pad_files=t_offset+1,
                                            x0=x0, x1=x1, y0=y0, y1=y1,
                                            return_new_ds=True,
                                            satellite=16,
                                            product='MCMIP',
                                            view='C',
                                            mode=[3,4,6],
                                            save_dir=goes_data_path,
                                            replicate_path=True,
                                            check_download=True,
                                            n_attempts=1,
                                            download_missing=True)

    print(datetime.now(),'Calculating flow field', flush=True)

    flow = Flow(bt, model="DIS", vr_steps=1, smoothing_passes=1)

    print(datetime.now(),'Detecting growth markers', flush=True)

    wvd_growth = get_growth_rate(flow, wvd)
    bt_growth = get_growth_rate(flow, bt)

    wvd_curvature_filter = get_curvature_filter(wvd, direction="negative")
    bt_curvature_filter = get_curvature_filter(bt, direction="positive")

    wvd_threshold = 0.125
    bt_threshold = 0.25

    wvd_markers = np.logical_and(wvd_growth > wvd_threshold,
                                 wvd_curvature_filter)
    bt_markers =  np.logical_and(bt_growth  < -bt_threshold,
                                 bt_curvature_filter)

    s_struct = ndi.generate_binary_structure(3,1)
    s_struct *= np.array([0,1,0])[:,np.newaxis, np.newaxis].astype(bool)

    combined_markers = ndi.binary_opening(np.logical_or(wvd_markers, bt_markers), structure=s_struct)

    print('WVD growth above threshold: area =', np.sum(wvd_markers))
    print('BT growth above threshold: area =', np.sum(bt_markers))
    print('Detected markers: area =', np.sum(combined_markers))

    overlap = 0.5
    subsegment_shrink = 0.

    print(datetime.now(),'Labeling cores', flush=True)
    core_labels = flow.label(combined_markers,
                             overlap=overlap,
                             subsegment_shrink=subsegment_shrink)

    # Filter labels by length and wvd growth threshold
    core_label_lengths = find_object_lengths(core_labels)

    print("Core labels meeting length threshold:", np.sum(core_label_lengths>t_offset))

    core_label_wvd_mask = mask_labels(core_labels, wvd_growth > wvd_threshold*2)

    print("Core labels meeting WVD growth threshold:", np.sum(core_label_wvd_mask))

    combined_mask = np.logical_and(core_label_lengths>t_offset, core_label_wvd_mask)

    core_labels = remap_labels(core_labels, combined_mask)

    print("Filtered core count:", core_labels.max())

    # Split into step labels
    print(datetime.now(),'Filtering cores by BT cooling rate', flush=True)
    core_step_labels = slice_labels(core_labels)

    mode = lambda x : stats.mode(x, keepdims=False)[0]
    core_step_core_index = labeled_comprehension(core_labels, core_step_labels,
                                                 mode, default=0)

    core_step_bt_mean = labeled_comprehension(bt, core_step_labels, np.nanmean,
                                              default=np.nan)

    core_step_t = labeled_comprehension(bt.t.data[:, np.newaxis, np.newaxis],
                                        core_step_labels, np.nanmin, default=0)

    def bt_diff_func(step_bt, pos):
        step_t = core_step_t[pos]
        args = np.argsort(step_t)

        step_bt = step_bt[args]
        step_t = step_t[args]

    #     step_bt_diff = [((step_bt[i] - step_bt[i+3])
    #                      / ((step_t[i+3] - step_t[i]).astype("timedelta64[s]").astype("int")/60) )
    #                     for i in range(len(step_bt)-3)]

        step_bt_diff = ((step_bt[:-t_offset] - step_bt[t_offset:])
                        / ((step_t[t_offset:] - step_t[:-t_offset]).astype("timedelta64[s]").astype("int")/60) )

        return np.nanmax(step_bt_diff)

    core_bt_diff_mean = labeled_comprehension(core_step_bt_mean,
                                              core_step_core_index,
                                              bt_diff_func,
                                              default=0,
                                              pass_positions=True)

    wh_valid_core = core_bt_diff_mean>=0.5

    core_labels = remap_labels(core_labels, wh_valid_core)

    print('Final detected core count: n =', core_labels.max())

    print(datetime.now(), 'Detecting thick anvil region', flush=True)
    upper_threshold = -5
    lower_threshold = -15
    erode_distance = 2

    field = (wvd-swd).data
    field = np.maximum(np.minimum(field, upper_threshold), lower_threshold)

    structure = ndi.generate_binary_structure(3,1)
    s_struct = structure * np.array([0,1,0])[:,np.newaxis, np.newaxis].astype(bool)

    markers = ndi.binary_erosion(field>=upper_threshold, structure=s_struct)
    mask = ndi.binary_erosion(
        field<=lower_threshold, structure=np.ones([3,3,3]),
        iterations=erode_distance, border_value=1
    )

    edges = flow.sobel(field, direction='uphill', method='linear')

    watershed = flow.watershed(edges, markers, mask=mask, structure=structure)

    print(datetime.now(), 'Labelling thick anvil region', flush=True)
    thick_anvil_labels = flow.label(ndi.binary_opening(watershed, structure=s_struct),
                                    overlap=overlap,
                                    subsegment_shrink=subsegment_shrink)

    print('Initial detected thick anvils: area =', np.sum(thick_anvil_labels!=0), flush=True)
    print('Initial detected thick anvils: n =', thick_anvil_labels.max(), flush=True)

    thick_anvil_label_lengths = find_object_lengths(thick_anvil_labels)
    thick_anvil_label_threshold = mask_labels(thick_anvil_labels, markers)

    thick_anvil_labels = remap_labels(thick_anvil_labels,
                                      np.logical_and(thick_anvil_label_lengths>t_offset,
                                                     thick_anvil_label_threshold))

    print('Final detected thick anvils: area =', np.sum(thick_anvil_labels!=0), flush=True)
    print('Final detected thick anvils: n =', thick_anvil_labels.max(), flush=True)

    print(datetime.now(), 'Detecting thin anvil region', flush=True)
    upper_threshold = 0
    lower_threshold = -7.5

    markers = thick_anvil_labels
    markers *= ndi.binary_erosion(markers, structure=s_struct).astype(int)

    field = (wvd+swd).data
    field = np.maximum(np.minimum(field, upper_threshold), lower_threshold)
    field[markers!=0] = upper_threshold

    mask = ndi.binary_erosion(
        field<=lower_threshold, structure=np.ones([3,3,3]),
        iterations=erode_distance, border_value=1
    )

    edges = flow.sobel(field, direction='uphill', method='linear')

    thin_anvil_labels = flow.watershed(edges, markers, mask=mask,
                                       structure=structure)

    thin_anvil_labels *= ndi.binary_opening(thin_anvil_labels, structure=s_struct).astype(int)

    print('Detected thin anvils: area =', np.sum(thin_anvil_labels!=0), flush=True)

    print(datetime.now(), 'Preparing output', flush=True)

    # Create output dataset
    # Core labels
    add_dataarray_to_ds(create_dataarray(core_labels,
                                         ('t','y','x'),
                                         "core_label",
                                         coords={"t":bt.t},
                                         long_name="labels for detected cores",
                                         units="",
                                         dtype=np.int32).sel(t=dataset.t), dataset)

    dataset.coords["core"] = np.arange(1, dataset.core_label.data.max()+1, dtype=np.int32)

    core_step_labels = slice_labels(core_labels)

    # Core step labels
    add_dataarray_to_ds(create_dataarray(core_step_labels,
                                         ('t','y','x'),
                                         "core_step_label",
                                         coords={"t":bt.t},
                                         long_name="labels for detected cores at each time step",
                                         units="",
                                         dtype=np.int32).sel(t=dataset.t), dataset)

    dataset.coords["core_step"] = np.arange(1, dataset.core_step_label.data.max()+1, dtype=np.int32)

    # Thick anvil
    add_dataarray_to_ds(create_dataarray(thick_anvil_labels,
                                         ('t','y','x'),
                                         "thick_anvil_label",
                                         coords={"t":bt.t},
                                         long_name="labels for detected thick anvil regions",
                                         units="",
                                         dtype=np.int32).sel(t=dataset.t), dataset)

    dataset.coords["anvil"] = np.arange(1, dataset.thick_anvil_label.data.max()+1, dtype=np.int32)

    thick_anvil_step_labels = slice_labels(thick_anvil_labels)

    add_dataarray_to_ds(create_dataarray(thick_anvil_step_labels,
                                         ('t','y','x'),
                                         "thick_anvil_step_label",
                                         coords={"t":bt.t},
                                         long_name="labels for detected thick anvil regions at each time step",
                                         units="",
                                         dtype=np.int32).sel(t=dataset.t), dataset)

    dataset.coords["thick_anvil_step"] = np.arange(1, dataset.thick_anvil_step_label.data.max()+1, dtype=np.int32)

    # Thin anvil
    add_dataarray_to_ds(create_dataarray(thin_anvil_labels,
                                         ('t','y','x'),
                                         "thin_anvil_label",
                                         coords={"t":bt.t},
                                         long_name="labels for detected thin anvil regions",
                                         units="",
                                         dtype=np.int32).sel(t=dataset.t), dataset)

    thin_anvil_step_labels = slice_labels(thin_anvil_labels)

    add_dataarray_to_ds(create_dataarray(thin_anvil_step_labels,
                                         ('t','y','x'),
                                         "thin_anvil_step_label",
                                         coords={"t":bt.t},
                                         long_name="labels for detected thin anvil regions at each time step",
                                         units="",
                                         dtype=np.int32).sel(t=dataset.t), dataset)

    dataset.coords["thin_anvil_step"] = np.arange(1, dataset.thin_anvil_step_label.data.max()+1, dtype=np.int32)

    # Add linking indices between each label
    def find_overlap_mode(x):
        if np.any(x):
            return stats.mode(x[x!=0], keepdims=False)[0]
        else:
            return 0

    core_step_core_index = labeled_comprehension(dataset.core_label.data,
                                                 dataset.core_step_label.data,
                                                 find_overlap_mode,
                                                 dtype=np.int32,
                                                 default=0)

    add_dataarray_to_ds(create_dataarray(core_step_core_index,
                                         ('core_step',),
                                         "core_step_core_index",
                                         long_name="core index for each core time step",
                                         dtype=np.int32), dataset)

    core_anvil_index = labeled_comprehension(dataset.thick_anvil_label.data,
                                             dataset.core_label.data,
                                             find_overlap_mode,
                                             dtype=np.int32,
                                             default=0)
    add_dataarray_to_ds(create_dataarray(core_anvil_index,
                                         ('core',),
                                         "core_anvil_index",
                                         long_name="anvil index for each core",
                                         dtype=np.int32), dataset)

    thick_anvil_step_anvil_index = labeled_comprehension(dataset.thick_anvil_label.data,
                                                         dataset.thick_anvil_step_label.data,
                                                         find_overlap_mode,
                                                         dtype=np.int32,
                                                         default=0)
    add_dataarray_to_ds(create_dataarray(thick_anvil_step_anvil_index,
                                         ('thick_anvil_step',),
                                         "thick_anvil_step_anvil_index",
                                         long_name="anvil index for each thick anvil time step",
                                         dtype=np.int32), dataset)

    thin_anvil_step_anvil_index = labeled_comprehension(dataset.thin_anvil_label.data,
                                                        dataset.thin_anvil_step_label.data,
                                                        find_overlap_mode,
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
    area_stack = np.repeat(dataset.area.data[np.newaxis,...], dataset.t.size, 0)
    lat_stack = np.repeat(dataset.lat.data[np.newaxis,...], dataset.t.size, 0)
    lon_stack = np.repeat(dataset.lon.data[np.newaxis,...], dataset.t.size, 0)

    xx, yy = np.meshgrid(dataset.x, dataset.y)
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

    del x_stack, y_stack, lat_stack, lon_stack

    get_label_stats(dataset.core_label, dataset)
    get_label_stats(dataset.thick_anvil_label, dataset)
    get_label_stats(dataset.thin_anvil_label, dataset)

    # Add BT, WVD, SWD stats
    weights = area_stack

    for field in (bt.sel(t=dataset.t), wvd.sel(t=dataset.t), swd.sel(t=dataset.t)):
        [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.core_label,
                                                                                  field.compute(),
                                                                                  weights,
                                                                                  name='core',
                                                                                  dim='core',
                                                                                  dtype=np.float32)]

        [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.thick_anvil_label,
                                                                                  field.compute(),
                                                                                  weights,
                                                                                  name='thick_anvil',
                                                                                  dim='anvil',
                                                                                  dtype=np.float32)]

        [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.thin_anvil_label,
                                                                                  field.compute(),
                                                                                  weights,
                                                                                  name='thin_anvil',
                                                                                  dim='anvil',
                                                                                  dtype=np.float32)]

        [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.core_step_label,
                                                                                  field.compute(),
                                                                                  weights,
                                                                                  name='core_step',
                                                                                  dim='core_step',
                                                                                  dtype=np.float32)]

        [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.thick_anvil_step_label,
                                                                                  field.compute(),
                                                                                  weights,
                                                                                  name='thick_anvil_step',
                                                                                  dim='thick_anvil_step',
                                                                                  dtype=np.float32)]

        [add_dataarray_to_ds(da, dataset) for da in weighted_statistics_on_labels(dataset.thin_anvil_step_label,
                                                                                  field.compute(),
                                                                                  weights,
                                                                                  name='thin_anvil_step',
                                                                                  dim='thin_anvil_step',
                                                                                  dtype=np.float32)]

    core_nan_flag = np.zeros_like(dataset.core, bool)
    thick_anvil_nan_flag = np.zeros_like(dataset.anvil, bool)
    thin_anvil_nan_flag = np.zeros_like(dataset.anvil, bool)

    if np.any(np.isnan(bt.data)):
        wh_nan = ndi.binary_dilation(np.isnan(bt.data), structure=np.ones([3,3,3]))
        core_nan_labels = np.unique(core_labels[wh_nan])

        if core_nan_labels[0] == 0:
            core_nan_flag[core_nan_labels[1:]-1] = True
        else:
            core_nan_flag[core_nan_labels-1] = True

        thick_anvil_nan_labels = np.unique(thick_anvil_labels[wh_nan])

        if thick_anvil_nan_labels[0] == 0:
            thick_anvil_nan_flag[thick_anvil_nan_labels[1:]-1] = True
        else:
            thick_anvil_nan_flag[thick_anvil_nan_labels-1] = True

        thin_anvil_nan_labels = np.unique(thin_anvil_labels[wh_nan])

        if thin_anvil_nan_labels[0] == 0:
            thin_anvil_nan_flag[thin_anvil_nan_labels[1:]-1] = True
        else:
            thin_anvil_nan_flag[thin_anvil_nan_labels-1] = True

    add_dataarray_to_ds(create_dataarray(core_nan_flag, ('core',), "core_nan_flag",
                                         long_name="flag for cores intersecting missing values",
                                         dtype=bool), dataset)

    add_dataarray_to_ds(create_dataarray(thick_anvil_nan_flag, ('anvil',), "thick_anvil_nan_flag",
                                         long_name="flag for thick anvils intersecting missing values",
                                         dtype=bool), dataset)

    add_dataarray_to_ds(create_dataarray(thin_anvil_nan_flag, ('anvil',), "thin_anvil_nan_flag",
                                         long_name="flag for thin anvils intersecting missing values",
                                         dtype=bool), dataset)

    print(datetime.now(), 'Saving to %s' % (save_path), flush=True)
    # Add compression encoding
    comp = dict(zlib=True, complevel=5, shuffle=True)
    for var in dataset.data_vars:
        dataset[var].encoding.update(comp)

    dataset.to_netcdf(save_path)

    dataset.close()
    bt.close()
    wvd.close()
    swd.close()

if __name__=="__main__":
    start_time = datetime.now()
    print(start_time, 'Commencing DCC detection', flush=True)

    print('Start date:', start_date)
    print('End date:', end_date)
    print('x0,x1,y0,y1:', x0, x1, y0 ,y1)
    print('Output save path:', save_path)
    print('GOES data path:', goes_data_path)

    main(start_date, end_date, satellite, x0, x1, y0, y1, save_path, goes_data_path)

    print(datetime.now(), 'Finished successfully, time elapsed:', datetime.now()-start_time, flush=True)
