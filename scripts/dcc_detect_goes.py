import pathlib
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
import numpy as np

from tobac_flow.flow import create_flow
from tobac_flow.dataloader import goes_dataloader
from tobac_flow.dataset import (
    add_label_coords,
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
)
from tobac_flow.detection import (
    detect_cores,
    get_anvil_markers,
    detect_anvils,
    relabel_anvils,
)
from tobac_flow.utils import (
    create_dataarray,
    add_dataarray_to_ds,
)

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
parser.add_argument(
    "--save_bt",
    help="Save brightness temperature field to output file",
    action="store_true",
)
parser.add_argument(
    "--save_wvd",
    help="Save water vapour difference field to output file",
    action="store_true",
)
parser.add_argument(
    "--save_swd",
    help="Save split window difference field to output file",
    action="store_true",
)
parser.add_argument(
    "--save_label_props",
    help="Save statistics of label properties to output file",
    action="store_true",
)
parser.add_argument(
    "--save_field_props",
    help="Save statistics of field properties to output file",
    action="store_true",
)
parser.add_argument(
    "--save_spatial_props",
    help="Save statistics of label spatial properties to output file",
    action="store_true",
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

save_dir = pathlib.Path(args.sd)
if not save_dir.exists():
    try:
        save_dir.mkdir()
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

save_path = save_dir / save_name

print("Saving output to:", save_path)

goes_data_path = pathlib.Path(args.gd)
if not goes_data_path.exists():
    try:
        goes_data_path.mkdir()
    except (FileExistsError, OSError):
        pass


def main() -> None:
    print(datetime.now(), "Loading ABI data", flush=True)
    print("Loading goes data from:", goes_data_path, flush=True)

    io_kwargs = dict(
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

    bt, wvd, swd, dataset = goes_dataloader(
        start_date,
        end_date,
        n_pad_files=t_offset * 2,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        return_new_ds=True,
        **io_kwargs,
    )

    print(datetime.now(), "Calculating flow field", flush=True)

    flow = create_flow(
        bt, model="Farneback", vr_steps=1, smoothing_passes=1, interp_method="cubic"
    )

    print(datetime.now(), "Detecting growth markers", flush=True)
    wvd_threshold = 0.25
    bt_threshold = 0.5
    overlap = 0.5
    absolute_overlap = 4
    subsegment_shrink = 0.0

    core_labels = detect_cores(
        flow,
        bt,
        wvd,
        swd,
        wvd_threshold=wvd_threshold,
        bt_threshold=bt_threshold,
        overlap=overlap,
        absolute_overlap=absolute_overlap,
        subsegment_shrink=subsegment_shrink,
        min_length=t_offset,
        use_wvd=False,
    )

    print("Final detected core count: n =", core_labels.max())

    print(datetime.now(), "Detecting thick anvil region", flush=True)
    # Detect anvil regions
    upper_threshold = -5
    lower_threshold = -12.5
    erode_distance = 2

    anvil_markers = get_anvil_markers(
        flow,
        wvd - swd,
        threshold=upper_threshold,
        overlap=overlap,
        absolute_overlap=absolute_overlap,
        subsegment_shrink=subsegment_shrink,
        min_length=t_offset,
    )

    print("Final thick anvil markers: area =", np.sum(anvil_markers != 0), flush=True)
    print("Final thick anvil markers: n =", anvil_markers.max(), flush=True)

    thick_anvil_labels = detect_anvils(
        flow,
        wvd - swd,
        markers=anvil_markers,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        erode_distance=erode_distance,
        min_length=t_offset,
    )

    print(
        "Initial detected thick anvils: area =",
        np.sum(thick_anvil_labels != 0),
        flush=True,
    )
    print("Initial detected thick anvils: n =", thick_anvil_labels.max(), flush=True)

    thick_anvil_labels = relabel_anvils(
        flow,
        thick_anvil_labels,
        markers=anvil_markers,
        overlap=overlap,
        absolute_overlap=absolute_overlap,
        min_length=t_offset,
    )

    print(
        "Final detected thick anvils: area =",
        np.sum(thick_anvil_labels != 0),
        flush=True,
    )
    print("Final detected thick anvils: n =", thick_anvil_labels.max(), flush=True)

    print(datetime.now(), "Detecting thin anvil region", flush=True)
    # Detect thin anvil regions
    upper_threshold = 0
    lower_threshold = -7.5

    thin_anvil_labels = detect_anvils(
        flow,
        wvd + swd,
        markers=thick_anvil_labels,
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        erode_distance=erode_distance,
        min_length=t_offset,
    )

    print("Detected thin anvils: area =", np.sum(thin_anvil_labels != 0), flush=True)
    print("Detected thin anvils: n =", np.max(thin_anvil_labels), flush=True)

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

    # bt, wvd, swd = bt.sel(t=dataset.t), wvd.sel(t=dataset.t), swd.sel(t=dataset.t)

    add_step_labels(dataset)

    dataset = add_label_coords(dataset)

    link_step_labels(dataset)

    # Add data quality flags
    flag_edge_labels(dataset, start_date, end_date)
    flag_nan_adjacent_labels(dataset, bt.sel(t=dataset.t))

    # core_max_bt_diff = core_bt_diff_mean[wh_valid_core]

    # add_dataarray_to_ds(
    #     create_dataarray(
    #         core_max_bt_diff,
    #         ("core"),
    #         "core_max_BT_diff",
    #         long_name="Maximum change in core brightness temperature per minute",
    #         units="K/minute",
    #         dtype=np.float32,
    #     ),
    #     dataset,
    # )

    if args.save_label_props:
        calculate_label_properties(dataset)

    if args.save_spatial_props:
        get_label_stats(dataset.core_label, dataset)
        get_label_stats(dataset.thick_anvil_label, dataset)
        get_label_stats(dataset.thin_anvil_label, dataset)

    if args.save_bt:
        add_dataarray_to_ds(
            bt.sel(t=dataset.t),
            dataset,
        )
    if args.save_wvd:
        add_dataarray_to_ds(
            wvd.sel(t=dataset.t),
            dataset,
        )
    if args.save_swd:
        add_dataarray_to_ds(
            swd.sel(t=dataset.t),
            dataset,
        )

    if args.save_field_props:
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
