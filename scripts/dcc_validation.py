import argparse
import pathlib
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage as ndi

from tobac_flow import io, glm
from tobac_flow.dataset import (
    create_new_goes_ds,
    add_dataarray_to_ds,
    create_dataarray,
)
from tobac_flow.utils import get_dates_from_filename
from tobac_flow.validation import (
    get_marker_distance,
    validate_markers,
)
from tobac_flow.postprocess import add_validity_flags

parser = argparse.ArgumentParser(
    description="""Validate detected DCCs using GOES-16 GLM data"""
)
parser.add_argument("file", help="File to validate", type=str)
parser.add_argument(
    "-margin", help="Tolerance margin for validation (in pixels)", default=10, type=int
)
parser.add_argument(
    "-time_margin",
    help="Tolerance margin for validation (in time steps)",
    default=3,
    type=int,
)
parser.add_argument("-gd", help="GOES directory", default="../data/GOES16", type=str)
parser.add_argument(
    "-glmsd",
    help="Directory to save gridded glm files",
    default="../data/glm_grid",
    type=str,
)

parser.add_argument(
    "-sd",
    help="Directory to save output files",
    default="../data/dcc_validation",
    type=str,
)
parser.add_argument("-cglm", help="clobber existing glm files", action="store_true")
parser.add_argument(
    "--filter",
    help="Filter cores/anvils using stats file",
    action="store_true",
)
parser.add_argument(
    "--is_valid",
    help="Filter valid cores/anvils",
    action="store_true",
)
parser.add_argument(
    "-stats_path",
    help="Directory of monthly statistics files",
    default="/work/scratch-nopw2/wkjones/statistics/",
    type=str,
)
args = parser.parse_args()

file = pathlib.Path(args.file)
margin = args.margin
time_margin = args.time_margin
clobber_glm = args.cglm

goes_data_path = pathlib.Path(args.gd)
if not goes_data_path.exists():
    try:
        goes_data_path.mkdir()
    except (FileExistsError, OSError):
        pass

glm_save_dir = pathlib.Path(args.glmsd)
if not glm_save_dir.exists():
    try:
        glm_save_dir.mkdir()
    except (FileExistsError, OSError):
        pass

save_dir = pathlib.Path(args.sd)
if not save_dir.exists():
    try:
        save_dir.mkdir()
    except (FileExistsError, OSError):
        pass

if args.filter:
    stats_path = pathlib.Path(args.stats_path)
    if not stats_path.exists():
        raise ValueError(f"{str(stats_path)} does not exist")


# def validation(file, margin, goes_data_path, save_dir):
def main():
    """
    Validation process for detected DCCs in the given file
    """
    print(datetime.now(), "Loading detected DCCs", flush=True)
    print(file, flush=True)
    detection_ds = xr.open_dataset(file)

    validation_ds = xr.Dataset()

    start_date, end_date = get_dates_from_filename(file)
    start_str = file.stem.split("_S")[-1][:15]
    end_str = file.stem.split("_E")[-1][:15]
    x_str = file.stem.split("_X")[-1][:9]
    y_str = file.stem.split("_Y")[-1][:9]
    new_file_str = f"S{start_str}_E{end_str}_X{x_str}_Y{y_str}"
    dates = pd.date_range(
        start_date, end_date, freq="H", inclusive="left"
    ).to_pydatetime()

    glm_save_name = f"gridded_glm_flashes_{new_file_str}.nc"
    glm_save_path = glm_save_dir / glm_save_name
    validation_save_name = f"validation_dccs_{new_file_str}.nc"
    validation_save_path = save_dir / validation_save_name

    """
    Start validation
    """
    if clobber_glm or not glm_save_path.exists():
        gridded_flash_ds = create_new_goes_ds(detection_ds)

        print(datetime.now(), "Processing GLM data", flush=True)
        # Get GLM data
        # Process new GLM data
        glm_files = io.find_glm_files(
            dates,
            satellite=16,
            save_dir=goes_data_path,
            replicate_path=True,
            check_download=True,
            n_attempts=1,
            download_missing=True,
            verbose=False,
            min_storage=2**30,
        )
        glm_files = {io.get_goes_date(i): i for i in glm_files}
        print("%d files found" % len(glm_files), flush=True)
        if len(glm_files) == 0:
            raise ValueError("No GLM Files discovered, skipping validation")
        else:
            print(datetime.now(), "Regridding GLM data", flush=True)
            glm_grid = glm.regrid_glm(glm_files, gridded_flash_ds, corrected=True)

        add_dataarray_to_ds(
            create_dataarray(
                glm_grid.data,
                ("t", "y", "x"),
                "glm_flashes",
                long_name="number of flashes detected by GLM",
                units="",
                dtype=np.int32,
            ),
            gridded_flash_ds,
        )

        add_dataarray_to_ds(
            create_dataarray(
                np.nansum(glm_grid.data[glm_grid.data > 0]),
                tuple(),
                "glm_flash_count",
                long_name="total number of GLM flashes",
                dtype=np.int32,
            ),
            gridded_flash_ds,
        )

        # Add compression encoding
        comp = dict(zlib=True, complevel=5, shuffle=True)
        for var in gridded_flash_ds.data_vars:
            gridded_flash_ds[var].encoding.update(comp)

        print(datetime.now(), "Saving to %s" % (glm_save_path), flush=True)
        gridded_flash_ds.to_netcdf(glm_save_path)
        glm_grid = gridded_flash_ds.glm_flashes.to_numpy()

    else:
        print(datetime.now(), "Loading from %s" % (glm_save_path), flush=True)
        gridded_flash_ds = xr.open_dataset(glm_save_path)
        glm_grid = gridded_flash_ds.glm_flashes.to_numpy()

    if args.filter:
        print(
            datetime.now(),
            "Filtering cores/anvils using statistics dataset",
            flush=True,
        )
        stats_file = list(
            stats_path.glob(
                f"dcc_statistics_G16_S{start_str[:6]}*_X{x_str}_Y{y_str}.nc"
            )
        )[0]
        print(datetime.now(), "Loading from %s" % (stats_ds), flush=True)
        stats_ds = xr.open_dataset(stats_file)

        if args.is_valid:
            core_label = detection_ds.core_label.data
            core_label *= np.isin(
                core_label, stats_ds.core.data[stats_ds.core_is_valid.data]
            ).astype(int)
            core_coord = detection_ds.core.data
            core_coord = core_coord[
                np.isin(core_coord, stats_ds.core.data[stats_ds.core_is_valid.data])
            ]

            thick_anvil_label = detection_ds.thick_anvil_label.data
            thick_anvil_label *= np.isin(
                thick_anvil_label,
                stats_ds.anvil.data[stats_ds.thin_anvil_is_valid.data],
            ).astype(int)
            anvil_coord = detection_ds.anvil.data
            anvil_coord = anvil_coord[
                np.isin(
                    anvil_coord, stats_ds.anvil.data[stats_ds.thin_anvil_is_valid.data]
                )
            ]

        else:
            core_label = detection_ds.core_label.data
            core_label *= np.isin(core_label, stats_ds.core.data).astype(int)
            core_coord = detection_ds.core.data
            core_coord = core_coord[np.isin(core_coord, stats_ds.core.data)]

            thick_anvil_label = detection_ds.thick_anvil_label.data
            thick_anvil_label *= np.isin(
                thick_anvil_label,
                stats_ds.anvil.data,
            ).astype(int)
            anvil_coord = detection_ds.anvil.data
            anvil_coord = anvil_coord[np.isin(anvil_coord, stats_ds.anvil.data)]

    else:
        core_label = detection_ds.core_label.data
        thick_anvil_label = detection_ds.thick_anvil_label.data
        core_coord = detection_ds.core.data
        anvil_coord = detection_ds.anvil.data

    validation_ds = validation_ds.assign_coords(core=core_coord, anvil=anvil_coord)
    print(datetime.now(), "Calculating flash distance", flush=True)
    # marker_distance = get_marker_distance(core_label, time_range=3)
    # anvil_distance = get_marker_distance(thick_anvil_label, time_range=3)
    glm_distance = get_marker_distance(glm_grid, time_range=time_margin)
    # wvd_distance = get_marker_distance(detection_ds.wvd_label, time_range=3)

    # Create an array to filter objects near to boundaries
    edge_filter_array = np.full(glm_distance.shape, 1).astype("bool")
    edge_filter_array[:time_margin] = 0
    edge_filter_array[-time_margin:] = 0
    edge_filter_array[:, :margin] = 0
    edge_filter_array[:, -margin:] = 0
    edge_filter_array[:, :, :margin] = 0
    edge_filter_array[:, :, -margin:] = 0

    # Filter objects near to missing glm data
    wh_missing_glm = ndi.binary_dilation(glm_grid == -1, iterations=time_margin)
    edge_filter_array[wh_missing_glm] = 0
    glm_grid[edge_filter_array == 0] = 0
    n_glm_in_margin = np.nansum(glm_grid[edge_filter_array])

    print(datetime.now(), "Validating cores", flush=True)
    (
        flash_distance_to_marker,
        core_min_distance,
        core_pod,
        core_far,
        n_growth_in_margin,
        core_margin_flag,
    ) = validate_markers(
        core_label,
        glm_grid,
        glm_distance,
        edge_filter_array,
        n_glm_in_margin,
        coord=core_coord,
        margin=margin,
        time_margin=time_margin,
    )

    print(datetime.now(), "Validating anvils", flush=True)
    (
        flash_distance_to_anvil,
        anvil_min_distance,
        anvil_pod,
        anvil_far,
        n_anvil_in_margin,
        anvil_margin_flag,
    ) = validate_markers(
        thick_anvil_label,
        glm_grid,
        glm_distance,
        edge_filter_array,
        n_glm_in_margin,
        coord=anvil_coord,
        margin=margin,
        time_margin=time_margin,
    )

    print("markers:", flush=True)
    print("n =", n_growth_in_margin, flush=True)
    print("POD =", core_pod, flush=True)
    print("FAR = ", core_far, flush=True)

    print("anvil:", flush=True)
    print("n =", n_anvil_in_margin, flush=True)
    print("POD =", anvil_pod, flush=True)
    print("FAR = ", anvil_far, flush=True)

    print("total GLM flashes: ", np.nansum(glm_grid[glm_grid > 0]), flush=True)
    print("total in margin: ", n_glm_in_margin, flush=True)

    """
    Finish validation
    """
    # GLM validation
    add_dataarray_to_ds(
        create_dataarray(
            flash_distance_to_marker,
            ("flash",),
            "flash_core_distance",
            long_name="closest distance from flash to detected core",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            flash_distance_to_anvil,
            ("flash",),
            "flash_anvil_distance",
            long_name="closest distance from flash to detected anvil",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            n_glm_in_margin,
            tuple(),
            "flash_count",
            long_name="total number of flashes inside margin",
            dtype=np.int32,
        ),
        validation_ds,
    )
    # anvil validation
    add_dataarray_to_ds(
        create_dataarray(
            anvil_min_distance,
            ("anvil",),
            "anvil_glm_distance",
            long_name="closest distance from anvil to GLM flash",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_margin_flag,
            ("anvil",),
            "anvil_margin_flag",
            long_name="margin flag for anvil",
            dtype=bool,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_pod,
            tuple(),
            "anvil_pod",
            long_name="POD for anvils",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_far,
            tuple(),
            "anvil_far",
            long_name="FAR for anvils",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            n_anvil_in_margin,
            tuple(),
            "anvil_count",
            long_name="total number of anvils inside margin",
            dtype=np.int32,
        ),
        validation_ds,
    )
    # growth validation
    add_dataarray_to_ds(
        create_dataarray(
            core_min_distance,
            ("core",),
            "core_glm_distance",
            long_name="closest distance from core to GLM flash",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_margin_flag,
            ("core",),
            "core_margin_flag",
            long_name="margin flag for core",
            dtype=bool,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_pod, tuple(), "core_pod", long_name="POD for cores", dtype=np.float32
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_far, tuple(), "core_far", long_name="FAR for cores", dtype=np.float32
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            n_growth_in_margin,
            tuple(),
            "core_count",
            long_name="total number of cores inside margin",
            dtype=np.int32,
        ),
        validation_ds,
    )

    # Add compression encoding
    comp = dict(zlib=True, complevel=5, shuffle=True)
    for var in validation_ds.data_vars:
        validation_ds[var].encoding.update(comp)
    print(datetime.now(), "Saving to %s" % (validation_save_path), flush=True)
    validation_ds.to_netcdf(validation_save_path)


if __name__ == "__main__":
    main()
