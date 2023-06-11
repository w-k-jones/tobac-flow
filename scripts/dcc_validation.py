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
    get_marker_distance_cylinder,
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
        anvil=detection_ds.anvil[np.logical_or(
            np.isin(detection_ds.anvil, detection_ds.thick_anvil_label),
            np.isin(detection_ds.anvil, detection_ds.thin_anvil_label)
        )]
    )

    if "anvil_marker" in detection_ds.coords:
        detection_ds = detection_ds.sel(
            anvil_marker=detection_ds.anvil_marker[np.isin(detection_ds.anvil_marker, detection_ds.anvil_marker_label)]
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

    core_label = detection_ds.core_label.to_numpy()
    core_coord = detection_ds.core.to_numpy()

    core_with_anvil_coord = core_coord[detection_ds.core_anvil_index.to_numpy() != 0]
    core_with_anvil_label = core_label * np.isin(core_label, core_with_anvil_coord).astype(int)

    thick_anvil_label = detection_ds.thick_anvil_label.to_numpy()
    anvil_coord = detection_ds.anvil.to_numpy()

    anvil_with_core_coord = anvil_coord[np.isin(anvil_coord, detection_ds.core_anvil_index.to_numpy())]
    anvil_with_core_label = thick_anvil_label * np.isin(thick_anvil_label, anvil_with_core_coord).astype(int)

    if "anvil_marker" in detection_ds.coords:
        anvil_marker_label = detection_ds.anvil_marker_label.to_numpy()
        anvil_marker_coord = detection_ds.anvil_marker.to_numpy()

        validation_ds = validation_ds.assign_coords(
            core=core_coord, 
            core_with_anvil=core_with_anvil_coord, 
            anvil=anvil_coord,
            anvil_with_core=anvil_with_core_coord, 
            anvil_marker=anvil_marker_coord
        )
    
    else:
        validation_ds = validation_ds.assign_coords(
            core=core_coord, 
            core_with_anvil=core_with_anvil_coord, 
            anvil=anvil_coord,
            anvil_with_core=anvil_with_core_coord,
        )
    
    print(datetime.now(), "Calculating flash distance", flush=True)
    glm_distance, _ = get_marker_distance_cylinder(glm_grid, time_margin)

    # Create an array to filter objects near to boundaries
    edge_filter_array = np.full(glm_distance.shape, 1).astype("bool")

    # Filter edges
    edge_filter_array[:time_margin] = False
    edge_filter_array[-time_margin:] = False
    edge_filter_array[:, :margin] = False
    edge_filter_array[:, -margin:] = False
    edge_filter_array[:, :, :margin] = False
    edge_filter_array[:, :, -margin:] = False

    time_gap = np.where((np.diff(detection_ds.t)/1e9).astype(int) > 900)[0]
    if time_gap.size > 0:
        print("Time gaps detected, filtering")
        for i in time_gap:
            i_slice = slice(np.maximum(i-time_margin+1,0),np.minimum(i+time_margin+2, detection_ds.t.size))
            edge_filter_array[i_slice] = False

    if np.any(glm_grid==-1):
        print("Missing glm data detected, filtering")
        margin_structure = np.stack([
            np.sum([(arr-10)**2 for arr in np.meshgrid(np.arange(margin*2+1), np.arange(margin*2+1))], 0)**0.5 < margin
        ]*(time_margin*2+1), 0)
        wh_missing_glm = ndi.binary_dilation(glm_grid==-1, structure=margin_structure)
        edge_filter_array[wh_missing_glm] = False

    glm_grid_filtered = np.zeros_like(glm_grid)
    glm_grid_filtered[edge_filter_array] = glm_grid[edge_filter_array]
    n_glm_total = np.nansum(glm_grid)
    n_glm_in_margin = np.nansum(glm_grid[edge_filter_array])

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

    print(datetime.now(), "Validating cores", flush=True)
    (
        flash_distance_to_core,
        flash_nearest_core,
        core_min_distance,
        core_pod,
        core_far,
        n_core_in_margin,
        core_margin_flag,
    ) = validate_markers(
        core_label,
        glm_grid_filtered,
        glm_distance,
        edge_filter_array,
        n_glm_in_margin,
        coord=core_coord,
        margin=margin,
        time_margin=time_margin,
    )

    print("cores:", flush=True)
    print("n =", n_core_in_margin, flush=True)
    print("POD =", core_pod, flush=True)
    print("FAR = ", core_far, flush=True)

    # Write to dataset
    add_dataarray_to_ds(
        create_dataarray(
            flash_distance_to_core,
            ("flash",),
            "flash_core_distance",
            long_name="closest distance from flash to detected core",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            flash_nearest_core,
            ("flash",),
            "flash_core_index",
            long_name="index of nearest detected core to each flash",
            dtype=np.int32,
        ),
        validation_ds,
    )
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
            n_core_in_margin,
            tuple(),
            "core_count_in_margin",
            long_name="total number of cores inside margin",
            dtype=np.int32,
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

    print(datetime.now(), "Validating cores with anvils", flush=True)
    (
        flash_distance_to_core_with_anvil,
        flash_nearest_core_with_anvil,
        core_with_anvil_min_distance,
        core_with_anvil_pod,
        core_with_anvil_far,
        n_core_with_anvil_in_margin,
        core_with_anvil_margin_flag,
    ) = validate_markers(
        core_with_anvil_label,
        glm_grid_filtered,
        glm_distance,
        edge_filter_array,
        n_glm_in_margin,
        coord=core_with_anvil_coord,
        margin=margin,
        time_margin=time_margin,
    )

    print("cores with anvils:", flush=True)
    print("n =", n_core_with_anvil_in_margin, flush=True)
    print("POD =", core_with_anvil_pod, flush=True)
    print("FAR = ", core_with_anvil_far, flush=True)

    # Write to dataset
    add_dataarray_to_ds(
        create_dataarray(
            flash_distance_to_core_with_anvil,
            ("flash",),
            "flash_core_with_anvil_distance",
            long_name="closest distance from flash to detected core_with_anvil",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            flash_nearest_core_with_anvil,
            ("flash",),
            "flash_core_with_anvil_index",
            long_name="index of nearest detected core_with_anvil to each flash",
            dtype=np.int32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_with_anvil_min_distance,
            ("core_with_anvil",),
            "core_with_anvil_glm_distance",
            long_name="closest distance from core_with_anvil to GLM flash",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_with_anvil_pod, tuple(), "core_with_anvil_pod", long_name="POD for core_with_anvils", dtype=np.float32
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_with_anvil_far, tuple(), "core_with_anvil_far", long_name="FAR for core_with_anvils", dtype=np.float32
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            n_core_with_anvil_in_margin,
            tuple(),
            "core_with_anvil_count_in_margin",
            long_name="total number of core_with_anvils inside margin",
            dtype=np.int32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_with_anvil_margin_flag,
            ("core_with_anvil",),
            "core_with_anvil_margin_flag",
            long_name="margin flag for core_with_anvil",
            dtype=bool,
        ),
        validation_ds,
    )

    print(datetime.now(), "Validating anvils", flush=True)
    (
        flash_distance_to_anvil,
        flash_nearest_anvil,
        anvil_min_distance,
        anvil_pod,
        anvil_far,
        n_anvil_in_margin,
        anvil_margin_flag,
    ) = validate_markers(
        thick_anvil_label,
        glm_grid_filtered,
        glm_distance,
        edge_filter_array,
        n_glm_in_margin,
        coord=anvil_coord,
        margin=margin,
        time_margin=time_margin,
    )

    print("anvil:", flush=True)
    print("n =", n_anvil_in_margin, flush=True)
    print("POD =", anvil_pod, flush=True)
    print("FAR = ", anvil_far, flush=True)

    # Write to dataset
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
            flash_nearest_anvil,
            ("flash",),
            "flash_anvil_index",
            long_name="index of nearest detected anvil to each flash",
            dtype=np.int32,
        ),
        validation_ds,
    )
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
            anvil_pod, tuple(), "anvil_pod", long_name="POD for anvils", dtype=np.float32
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_far, tuple(), "anvil_far", long_name="FAR for anvils", dtype=np.float32
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            n_anvil_in_margin,
            tuple(),
            "anvil_count_in_margin",
            long_name="total number of anvils inside margin",
            dtype=np.int32,
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

    print(datetime.now(), "Validating anvils with cores", flush=True)
    (
        flash_distance_to_anvil_with_core,
        flash_nearest_anvil_with_core,
        anvil_with_core_min_distance,
        anvil_with_core_pod,
        anvil_with_core_far,
        n_anvil_with_core_in_margin,
        anvil_with_core_margin_flag,
    ) = validate_markers(
        anvil_with_core_label,
        glm_grid_filtered,
        glm_distance,
        edge_filter_array,
        n_glm_in_margin,
        coord=anvil_with_core_coord,
        margin=margin,
        time_margin=time_margin,
    )

    print("anvil with cores:", flush=True)
    print("n =", n_anvil_with_core_in_margin, flush=True)
    print("POD =", anvil_with_core_pod, flush=True)
    print("FAR = ", anvil_with_core_far, flush=True)

    # Write to dataset
    add_dataarray_to_ds(
        create_dataarray(
            flash_distance_to_anvil_with_core,
            ("flash",),
            "flash_anvil_with_core_distance",
            long_name="closest distance from flash to detected anvil_with_core",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            flash_nearest_anvil_with_core,
            ("flash",),
            "flash_anvil_with_core_index",
            long_name="index of nearest detected anvil_with_core to each flash",
            dtype=np.int32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_with_core_min_distance,
            ("anvil_with_core",),
            "anvil_with_core_glm_distance",
            long_name="closest distance from anvil_with_core to GLM flash",
            dtype=np.float32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_with_core_pod, tuple(), "anvil_with_core_pod", long_name="POD for anvil_with_cores", dtype=np.float32
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_with_core_far, tuple(), "anvil_with_core_far", long_name="FAR for anvil_with_cores", dtype=np.float32
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            n_anvil_with_core_in_margin,
            tuple(),
            "anvil_with_core_count_in_margin",
            long_name="total number of anvil_with_cores inside margin",
            dtype=np.int32,
        ),
        validation_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_with_core_margin_flag,
            ("anvil_with_core",),
            "anvil_with_core_margin_flag",
            long_name="margin flag for anvil_with_core",
            dtype=bool,
        ),
        validation_ds,
    )

    if "anvil_marker" in detection_ds.coords:
        print(datetime.now(), "Validating anvil markerss", flush=True)
        (
            flash_distance_to_anvil_marker,
            flash_nearest_anvil_marker,
            anvil_marker_min_distance,
            anvil_marker_pod,
            anvil_marker_far,
            n_anvil_marker_in_margin,
            anvil_marker_margin_flag,
        ) = validate_markers(
            anvil_marker_label,
            glm_grid_filtered,
            glm_distance,
            edge_filter_array,
            n_glm_in_margin,
            coord=anvil_marker_coord,
            margin=margin,
            time_margin=time_margin,
        )

        print("anvil marker:", flush=True)
        print("n =", n_anvil_marker_in_margin, flush=True)
        print("POD =", anvil_marker_pod, flush=True)
        print("FAR = ", anvil_marker_far, flush=True)

        # Write to dataset
        add_dataarray_to_ds(
            create_dataarray(
                flash_distance_to_anvil_marker,
                ("flash",),
                "flash_anvil_marker_distance",
                long_name="closest distance from flash to detected anvil_marker",
                dtype=np.float32,
            ),
            validation_ds,
        )
        add_dataarray_to_ds(
            create_dataarray(
                flash_nearest_anvil_marker,
                ("flash",),
                "flash_anvil_marker_index",
                long_name="index of nearest detected anvil_marker to each flash",
                dtype=np.int32,
            ),
            validation_ds,
        )
        add_dataarray_to_ds(
            create_dataarray(
                anvil_marker_min_distance,
                ("anvil_marker",),
                "anvil_marker_glm_distance",
                long_name="closest distance from anvil_marker to GLM flash",
                dtype=np.float32,
            ),
            validation_ds,
        )
        add_dataarray_to_ds(
            create_dataarray(
                anvil_marker_pod, tuple(), "anvil_marker_pod", long_name="POD for anvil_markers", dtype=np.float32
            ),
            validation_ds,
        )
        add_dataarray_to_ds(
            create_dataarray(
                anvil_marker_far, tuple(), "anvil_marker_far", long_name="FAR for anvil_markers", dtype=np.float32
            ),
            validation_ds,
        )
        add_dataarray_to_ds(
            create_dataarray(
                n_anvil_marker_in_margin,
                tuple(),
                "anvil_marker_count_in_margin",
                long_name="total number of anvil_markers inside margin",
                dtype=np.int32,
            ),
            validation_ds,
        )
        add_dataarray_to_ds(
            create_dataarray(
                anvil_marker_margin_flag,
                ("anvil_marker",),
                "anvil_marker_margin_flag",
                long_name="margin flag for anvil_marker",
                dtype=bool,
            ),
            validation_ds,
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
