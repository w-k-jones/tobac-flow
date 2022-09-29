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
                                                                           min_length=2)

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
