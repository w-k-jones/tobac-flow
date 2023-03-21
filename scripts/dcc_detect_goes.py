import os
import numpy as np
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from scipy import ndimage as ndi
from scipy import stats

from tobac_flow.flow import create_flow
from tobac_flow.dataloader import goes_dataloader
from tobac_flow.dataset import (
    add_dataarray_to_ds,
    add_label_coords,
    create_dataarray, 
    add_step_labels, 
    add_label_coords, 
    flag_edge_labels,
    flag_nan_adjacent_labels, 
    link_step_labels, 
    calculate_label_properties, 
)
from tobac_flow.analysis import (
    get_label_stats,
    weighted_statistics_on_labels,
    find_object_lengths,
    remap_labels, 
    mask_labels,
)
from tobac_flow.utils import labeled_comprehension

from tobac_flow.detection import get_curvature_filter, get_growth_rate
from tobac_flow.label import slice_labels

import argparse

parser = argparse.ArgumentParser(
    description="""Detect and track DCCs in GOES-16 ABI data"""
)
parser.add_argument("date", help="Date on which to start process", type=str)
parser.add_argument("hours", help="Number of hours to process", type=float)
parser.add_argument("-sat", help="GOES satellite", default=16, type=int)
parser.add_argument("-x0", help="Initial subset x location", default=0, type=int)
parser.add_argument("-x1", help="End subset x location", default=2500, type=int)
parser.add_argument("-y0", help="Initial subset y location", default=0, type=int)
parser.add_argument("-y1", help="End subset y location", default=1500, type=int)
parser.add_argument(
    "-sd",
    help="Directory to save preprocess files",
    default="../data/dcc_detect",
    type=str,
)
parser.add_argument("-gd", help="GOES directory", default="../data/GOES16", type=str)
parser.add_argument(
    "--extend_path",
    help="Extend save directory using year/month/day subdirectories",
    default=True,
    type=bool,
)

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

save_name = "detected_dccs_G%02d_S%s_E%s_X%04d_%04d_Y%04d_%04d.nc" % (
    satellite,
    start_date.strftime("%Y%m%d_%H0000"),
    end_date.strftime("%Y%m%d_%H0000"),
    x0,
    x1,
    y0,
    y1,
)

save_path = os.path.join(save_dir, save_name)

print("Saving output to:", save_path)

goes_data_path = args.gd
if not os.path.isdir(goes_data_path):
    try:
        os.makedirs(goes_data_path)
    except (FileExistsError, OSError):
        pass


def main() -> None:

    print(datetime.now(), "Loading ABI data", flush=True)
    print("Loading goes data from:", goes_data_path, flush=True)

    bt, wvd, swd, dataset = goes_dataloader(
        start_date,
        end_date,
        n_pad_files=t_offset + 1,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        return_new_ds=True,
        satellite=satellite,
        product="MCMIP",
        view="C",
        mode=[3, 4, 6],
        save_dir=goes_data_path,
        replicate_path=True,
        check_download=True,
        n_attempts=1,
        download_missing=True,
    )

    print(datetime.now(), "Calculating flow field", flush=True)

    flow = create_flow(bt, model="DIS", vr_steps=1, smoothing_passes=1)

    print(datetime.now(), "Detecting growth markers", flush=True)

    wvd_growth = get_growth_rate(flow, wvd)
    bt_growth = get_growth_rate(flow, bt)

    wvd_curvature_filter = get_curvature_filter(wvd, direction="negative")
    bt_curvature_filter = get_curvature_filter(bt, direction="positive")

    wvd_threshold = 0.125
    bt_threshold = 0.25

    wvd_markers = np.logical_and(wvd_growth > wvd_threshold, wvd_curvature_filter)
    bt_markers = np.logical_and(bt_growth < -bt_threshold, bt_curvature_filter)

    s_struct = ndi.generate_binary_structure(3, 1)
    s_struct *= np.array([0, 1, 0])[:, np.newaxis, np.newaxis].astype(bool)

    combined_markers = ndi.binary_opening(
        np.logical_or(wvd_markers, bt_markers), structure=s_struct
    )

    print("WVD growth above threshold: area =", np.sum(wvd_markers))
    print("BT growth above threshold: area =", np.sum(bt_markers))
    print("Detected markers: area =", np.sum(combined_markers))

    overlap = 0.5
    subsegment_shrink = 0.0

    print(datetime.now(), "Labeling cores", flush=True)
    core_labels = flow.label(
        combined_markers, overlap=overlap, subsegment_shrink=subsegment_shrink
    )

    # Filter labels by length and wvd growth threshold
    core_label_lengths = find_object_lengths(core_labels)

    print(
        "Core labels meeting length threshold:", np.sum(core_label_lengths > t_offset)
    )

    core_label_wvd_mask = mask_labels(core_labels, wvd_growth > wvd_threshold * 2)

    print("Core labels meeting WVD growth threshold:", np.sum(core_label_wvd_mask))

    combined_mask = np.logical_and(core_label_lengths > t_offset, core_label_wvd_mask)

    core_labels = remap_labels(core_labels, combined_mask)

    print("Filtered core count:", core_labels.max())

    # Split into step labels
    print(datetime.now(), "Filtering cores by BT cooling rate", flush=True)
    core_step_labels = slice_labels(core_labels)

    mode = lambda x: stats.mode(x, keepdims=False)[0]
    core_step_core_index = labeled_comprehension(
        core_labels, core_step_labels, mode, default=0
    )

    core_step_bt_mean = labeled_comprehension(
        bt, core_step_labels, np.nanmean, default=np.nan
    )

    core_step_t = labeled_comprehension(
        bt.t.data[:, np.newaxis, np.newaxis], core_step_labels, np.nanmin, default=0
    )

    def bt_diff_func(step_bt, pos):
        step_t = core_step_t[pos]
        args = np.argsort(step_t)

        step_bt = step_bt[args]
        step_t = step_t[args]

        step_bt_diff = (step_bt[:-t_offset] - step_bt[t_offset:]) / (
            (step_t[t_offset:] - step_t[:-t_offset])
            .astype("timedelta64[s]")
            .astype("int")
            / 60
        )

        return np.nanmax(step_bt_diff)

    core_bt_diff_mean = labeled_comprehension(
        core_step_bt_mean,
        core_step_core_index,
        bt_diff_func,
        default=0,
        pass_positions=True,
    )

    wh_valid_core = core_bt_diff_mean >= 0.5

    core_labels = remap_labels(core_labels, wh_valid_core)

    print("Final detected core count: n =", core_labels.max())

    print(datetime.now(), "Detecting thick anvil region", flush=True)
    upper_threshold = -5
    lower_threshold = -15
    erode_distance = 2

    field = (wvd - swd).data
    field = np.maximum(np.minimum(field, upper_threshold), lower_threshold)

    structure = ndi.generate_binary_structure(3, 1)
    s_struct = structure * np.array([0, 1, 0])[:, np.newaxis, np.newaxis].astype(bool)

    markers = ndi.binary_erosion(field >= upper_threshold, structure=s_struct)
    mask = ndi.binary_erosion(
        field <= lower_threshold,
        structure=np.ones([3, 3, 3]),
        iterations=erode_distance,
        border_value=1,
    )

    edges = flow.sobel(field, direction="uphill", method="linear")

    watershed = flow.watershed(edges, markers, mask=mask, structure=structure)

    print(datetime.now(), "Labelling thick anvil region", flush=True)
    thick_anvil_labels = flow.label(
        ndi.binary_opening(watershed, structure=s_struct),
        overlap=overlap,
        subsegment_shrink=subsegment_shrink,
    )

    print(
        "Initial detected thick anvils: area =",
        np.sum(thick_anvil_labels != 0),
        flush=True,
    )
    print("Initial detected thick anvils: n =", thick_anvil_labels.max(), flush=True)

    thick_anvil_label_lengths = find_object_lengths(thick_anvil_labels)
    thick_anvil_label_threshold = mask_labels(thick_anvil_labels, markers)

    thick_anvil_labels = remap_labels(
        thick_anvil_labels,
        np.logical_and(
            thick_anvil_label_lengths > t_offset, thick_anvil_label_threshold
        ),
    )

    print(
        "Final detected thick anvils: area =",
        np.sum(thick_anvil_labels != 0),
        flush=True,
    )
    print("Final detected thick anvils: n =", thick_anvil_labels.max(), flush=True)

    print(datetime.now(), "Detecting thin anvil region", flush=True)
    upper_threshold = 0
    lower_threshold = -7.5

    markers = thick_anvil_labels
    markers *= ndi.binary_erosion(markers, structure=s_struct).astype(int)

    field = (wvd + swd).data
    field = np.maximum(np.minimum(field, upper_threshold), lower_threshold)
    field[markers != 0] = upper_threshold

    mask = ndi.binary_erosion(
        field <= lower_threshold,
        structure=np.ones([3, 3, 3]),
        iterations=erode_distance,
        border_value=1,
    )

    edges = flow.sobel(field, direction="uphill", method="linear")

    thin_anvil_labels = flow.watershed(edges, markers, mask=mask, structure=structure)

    thin_anvil_labels *= ndi.binary_opening(
        thin_anvil_labels, structure=s_struct
    ).astype(int)

    print("Detected thin anvils: area =", np.sum(thin_anvil_labels != 0), flush=True)

    print(datetime.now(), "Preparing output", flush=True)

    # Create output dataset
    # Core labels
    add_dataarray_to_ds(
        create_dataarray(
            core_labels,
            ("t", "y", "x"),
            "core_label",
            coords={"t": bt.t},
            long_name="labels for detected cores",
            units="",
            dtype=np.int32,
        ).sel(t=dataset.t),
        dataset,
    )   

    # Thick anvil
    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_labels,
            ("t", "y", "x"),
            "thick_anvil_label",
            coords={"t": bt.t},
            long_name="labels for detected thick anvil regions",
            units="",
            dtype=np.int32,
        ).sel(t=dataset.t),
        dataset,
    )

    # Thin anvil
    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_labels,
            ("t", "y", "x"),
            "thin_anvil_label",
            coords={"t": bt.t},
            long_name="labels for detected thin anvil regions",
            units="",
            dtype=np.int32,
        ).sel(t=dataset.t),
        dataset,
    )

    add_step_labels(dataset)

    dataset = add_label_coords(dataset)

    link_step_labels(dataset)

    # Add data quality flags
    flag_edge_labels(dataset, start_date, end_date)
    flag_nan_adjacent_labels(dataset, bt)

    core_max_bt_diff = core_bt_diff_mean[wh_valid_core]

    add_dataarray_to_ds(
        create_dataarray(
            core_max_bt_diff,
            ("core"),
            "core_max_BT_diff",
            long_name="Maximum change in core brightness temperature per minute",
            units="K/minute",
            dtype=np.float32,
        ),
        dataset,
    )
    
    calculate_label_properties(dataset)

    get_label_stats(dataset.core_label, dataset)
    get_label_stats(dataset.thick_anvil_label, dataset)
    get_label_stats(dataset.thin_anvil_label, dataset)

    # Add BT, WVD, SWD stats
    weights = np.repeat(dataset.area.data[np.newaxis, ...], dataset.t.size, 0)

    for field in (bt.sel(t=dataset.t), wvd.sel(t=dataset.t), swd.sel(t=dataset.t)):
        [
            add_dataarray_to_ds(da, dataset)
            for da in weighted_statistics_on_labels(
                dataset.core_label,
                field.compute(),
                weights,
                name="core",
                dim="core",
                dtype=np.float32,
            )
        ]

        [
            add_dataarray_to_ds(da, dataset)
            for da in weighted_statistics_on_labels(
                dataset.thick_anvil_label,
                field.compute(),
                weights,
                name="thick_anvil",
                dim="anvil",
                dtype=np.float32,
            )
        ]

        [
            add_dataarray_to_ds(da, dataset)
            for da in weighted_statistics_on_labels(
                dataset.thin_anvil_label,
                field.compute(),
                weights,
                name="thin_anvil",
                dim="anvil",
                dtype=np.float32,
            )
        ]

        [
            add_dataarray_to_ds(da, dataset)
            for da in weighted_statistics_on_labels(
                dataset.core_step_label,
                field.compute(),
                weights,
                name="core_step",
                dim="core_step",
                dtype=np.float32,
            )
        ]

        [
            add_dataarray_to_ds(da, dataset)
            for da in weighted_statistics_on_labels(
                dataset.thick_anvil_step_label,
                field.compute(),
                weights,
                name="thick_anvil_step",
                dim="thick_anvil_step",
                dtype=np.float32,
            )
        ]

        [
            add_dataarray_to_ds(da, dataset)
            for da in weighted_statistics_on_labels(
                dataset.thin_anvil_step_label,
                field.compute(),
                weights,
                name="thin_anvil_step",
                dim="thin_anvil_step",
                dtype=np.float32,
            )
        ]

    

    print(datetime.now(), "Saving to %s" % (save_path), flush=True)
    # Add compression encoding
    comp = dict(zlib=True, complevel=5, shuffle=True)
    for var in dataset.data_vars:
        dataset[var].encoding.update(comp)

    dataset.to_netcdf(save_path)

    dataset.close()
    bt.close()
    wvd.close()
    swd.close()


if __name__ == "__main__":
    start_time = datetime.now()
    print(start_time, "Commencing DCC detection", flush=True)

    print("Start date:", start_date)
    print("End date:", end_date)
    print("x0,x1,y0,y1:", x0, x1, y0, y1)
    print("Output save path:", save_path)
    print("GOES data path:", goes_data_path)

    main()

    print(
        datetime.now(),
        "Finished successfully, time elapsed:",
        datetime.now() - start_time,
        flush=True,
    )
