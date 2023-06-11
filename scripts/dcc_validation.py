#!/home/users/wkjones/miniconda3/envs/tobac_flow/bin/python
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
from tobac_flow.utils import get_dates_from_filename, trim_file_start, trim_file_end
from tobac_flow.validation import (
    get_edge_filter,
    get_marker_distance_cylinder,
    validate_anvil_markers,
    validate_anvils,
    validate_anvils_with_cores,
    validate_cores_with_anvils,
    validate_markers,
    validate_cores,
)

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


# def validation(file, margin, goes_data_path, save_dir):
def main():
    """
    Validation process for detected DCCs in the given file
    """
    print(datetime.now(), "Loading detected DCCs", flush=True)
    print(file, flush=True)
    detection_ds = xr.open_dataset(file)

    # Trim any padding regions
    detection_ds = trim_file_start(trim_file_end(detection_ds, file), file)
    # now trim core, anvil coordinates
    detection_ds = detection_ds.sel(
        core=detection_ds.core[np.isin(detection_ds.core, detection_ds.core_label)],
        anvil=detection_ds.anvil[
            np.logical_or(
                np.isin(detection_ds.anvil, detection_ds.thick_anvil_label),
                np.isin(detection_ds.anvil, detection_ds.thin_anvil_label),
            )
        ],
    )

    if "anvil_marker" in detection_ds.coords:
        detection_ds = detection_ds.sel(
            anvil_marker=detection_ds.anvil_marker[
                np.isin(detection_ds.anvil_marker, detection_ds.anvil_marker_label)
            ]
        )

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
    Grid GLM Flashes
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

    """
    Calculate flash distances
    """
    print(datetime.now(), "Calculating flash distance", flush=True)
    glm_distance, _ = get_marker_distance_cylinder(glm_grid, time_margin)

    n_glm_total = np.nansum(glm_grid)

    edge_filter_array = get_edge_filter(gridded_flash_ds, margin, time_margin)
    glm_grid[np.logical_not(edge_filter_array)] = 0
    n_glm_in_margin = np.nansum(glm_grid)

    print("total GLM flashes: ", n_glm_total, flush=True)
    print("total in margin: ", n_glm_in_margin, flush=True)

    add_dataarray_to_ds(
        create_dataarray(
            n_glm_total,
            tuple(),
            "flash_count_total",
            long_name="total number of flashes",
            dtype=np.int32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            n_glm_in_margin,
            tuple(),
            "flash_count_in_margin",
            long_name="total number of flashes inside margin",
            dtype=np.int32,
        ),
        validation_ds,
    )

    """
    Start validation
    """
    print(datetime.now(), "Validating cores", flush=True)
    validate_cores(
        detection_ds,
        validation_ds,
        glm_grid,
        glm_distance,
        edge_filter_array,
        n_glm_in_margin,
        margin,
        time_margin,
    )

    print(datetime.now(), "Validating cores with anvils", flush=True)
    validate_cores_with_anvils(
        detection_ds,
        validation_ds,
        glm_grid,
        glm_distance,
        edge_filter_array,
        n_glm_in_margin,
        margin,
        time_margin,
    )

    print(datetime.now(), "Validating anvils", flush=True)
    validate_anvils(
        detection_ds,
        validation_ds,
        glm_grid,
        glm_distance,
        edge_filter_array,
        n_glm_in_margin,
        margin,
        time_margin,
    )

    print(datetime.now(), "Validating anvils with cores", flush=True)
    validate_anvils_with_cores(
        detection_ds,
        validation_ds,
        glm_grid,
        glm_distance,
        edge_filter_array,
        n_glm_in_margin,
        margin,
        time_margin,
    )

    if "anvil_marker" in detection_ds.coords:
        print(datetime.now(), "Validating anvil markerss", flush=True)
        validate_anvil_markers(
            detection_ds,
            validation_ds,
            glm_grid,
            glm_distance,
            edge_filter_array,
            n_glm_in_margin,
            margin,
            time_margin,
        )

    """
    Finish validation
    """
    # Add compression encoding
    comp = dict(zlib=True, complevel=5, shuffle=True)
    for var in validation_ds.data_vars:
        validation_ds[var].encoding.update(comp)
    print(datetime.now(), "Saving to %s" % (validation_save_path), flush=True)
    validation_ds.to_netcdf(validation_save_path)


if __name__ == "__main__":
    main()
