from functools import partial
import numpy as np
from scipy import stats
from scipy import ndimage as ndi
import xarray as xr
from tobac_flow.abi import get_abi_lat_lon, get_abi_pixel_area
from tobac_flow.utils.xarray_utils import create_dataarray, add_dataarray_to_ds
from tobac_flow.utils.label_utils import (
    apply_func_to_labels,
    labeled_comprehension,
    slice_labels,
    remap_labels,
)
from tobac_flow.utils.datetime_utils import get_datetime_from_coord
from tobac_flow.utils.legacy_utils import apply_weighted_func_to_labels
from tobac_flow.utils.stats_utils import find_overlap_mode, n_unique_along_axis


def get_bulk_stats(da):
    mean_da = create_dataarray(
        np.nanmean(da.data),
        tuple(),
        f"{da.name}_mean",
        long_name=f"Mean of {da.long_name}",
        units=da.units,
        dtype=da.dtype,
    )
    std_da = create_dataarray(
        np.nanstd(da.data),
        tuple(),
        f"{da.name}_std",
        long_name=f"Standard deviation of {da.long_name}",
        units=da.units,
        dtype=da.dtype,
    )
    median_da = create_dataarray(
        np.median(da.data),
        tuple(),
        f"{da.name}_median",
        long_name=f"Median of {da.long_name}",
        units=da.units,
        dtype=da.dtype,
    )
    max_da = create_dataarray(
        np.nanmax(da.data),
        tuple(),
        f"{da.name}_max",
        long_name=f"Maximum of {da.long_name}",
        units=da.units,
        dtype=da.dtype,
    )
    min_da = create_dataarray(
        np.nanmin(da.data),
        tuple(),
        f"{da.name}_min",
        long_name=f"Minimum of {da.long_name}",
        units=da.units,
        dtype=da.dtype,
    )
    return mean_da, std_da, median_da, max_da, min_da


def get_spatial_stats(da):
    mean_da = create_dataarray(
        da.mean(("x", "y")).data,
        ("t",),
        f"{da.name}_spatial_mean",
        long_name=f"Spatial mean of {da.long_name}",
        units=da.units,
        dtype=da.dtype,
    )
    std_da = create_dataarray(
        da.std(("x", "y")).data,
        ("t",),
        f"{da.name}_spatial_std",
        long_name=f"Spatial standard deviation of {da.long_name}",
        units=da.units,
        dtype=da.dtype,
    )
    median_da = create_dataarray(
        da.median(("x", "y")).data,
        ("t",),
        f"{da.name}_spatial_median",
        long_name=f"Spatial median of {da.long_name}",
        units=da.units,
        dtype=da.dtype,
    )
    max_da = create_dataarray(
        da.max(("x", "y")).data,
        ("t",),
        f"{da.name}_spatial_max",
        long_name=f"Spatial maximum of {da.long_name}",
        units=da.units,
        dtype=da.dtype,
    )
    min_da = create_dataarray(
        da.min(("x", "y")).data,
        ("t",),
        f"{da.name}_spatial_min",
        long_name=f"Spatial minimum of {da.long_name}",
        units=da.units,
        dtype=da.dtype,
    )
    return mean_da, std_da, median_da, max_da, min_da


def get_temporal_stats(da):
    mean_da = create_dataarray(
        da.mean("t").data,
        ("y", "x"),
        f"{da.name}_temporal_mean",
        long_name=f"Temporal mean of {da.long_name}",
        units=da.units,
        dtype=da.dtype,
    )
    std_da = create_dataarray(
        da.std("t").data,
        ("y", "x"),
        f"{da.name}_temporal_std",
        long_name=f"Temporal standard deviation of {da.long_name}",
        units=da.units,
        dtype=da.dtype,
    )
    median_da = create_dataarray(
        da.median("t").data,
        ("y", "x"),
        f"{da.name}_temporal_median",
        long_name=f"Temporal median of {da.long_name}",
        units=da.units,
        dtype=da.dtype,
    )
    max_da = create_dataarray(
        da.max("t").data,
        ("y", "x"),
        f"{da.name}_temporal_max",
        long_name=f"Temporal maximum of {da.long_name}",
        units=da.units,
        dtype=da.dtype,
    )
    min_da = create_dataarray(
        da.min("t").data,
        ("y", "x"),
        f"{da.name}_temporal_min",
        long_name=f"Temporal minimum of {da.long_name}",
        units=da.units,
        dtype=da.dtype,
    )
    return mean_da, std_da, median_da, max_da, min_da


def create_new_goes_ds(goes_ds):
    goes_coords = {
        "t": goes_ds.t,
        "y": goes_ds.y,
        "x": goes_ds.x,
        "y_image": goes_ds.y_image,
        "x_image": goes_ds.x_image,
    }

    new_ds = xr.Dataset(coords=goes_coords)
    new_ds["goes_imager_projection"] = goes_ds.goes_imager_projection
    lat, lon = get_abi_lat_lon(new_ds)
    add_dataarray_to_ds(
        create_dataarray(
            lat, ("y", "x"), "lat", long_name="latitude", dtype=np.float32
        ),
        new_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            lon, ("y", "x"), "lon", long_name="longitude", dtype=np.float32
        ),
        new_ds,
    )
    add_dataarray_to_ds(
        create_dataarray(
            get_abi_pixel_area(new_ds),
            ("y", "x"),
            "area",
            long_name="pixel area",
            units="km^2",
            dtype=np.float32,
        ),
        new_ds,
    )
    return new_ds


def add_step_labels(dataset: xr.Dataset) -> None:
    add_dataarray_to_ds(
        create_dataarray(
            slice_labels(dataset.core_label.data),
            ("t", "y", "x"),
            "core_step_label",
            coords={"t": dataset.t},
            long_name="labels for detected cores at each time step",
            units="",
            dtype=np.int32,
        ).sel(t=dataset.t),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            slice_labels(dataset.thick_anvil_label.data),
            ("t", "y", "x"),
            "thick_anvil_step_label",
            coords={"t": dataset.t},
            long_name="labels for detected thick anvil regions at each time step",
            units="",
            dtype=np.int32,
        ).sel(t=dataset.t),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            slice_labels(dataset.thin_anvil_label.data),
            ("t", "y", "x"),
            "thin_anvil_step_label",
            coords={"t": dataset.t},
            long_name="labels for detected thin anvil regions at each time step",
            units="",
            dtype=np.int32,
        ).sel(t=dataset.t),
        dataset,
    )


def add_label_coords(dataset: xr.Dataset) -> xr.Dataset:
    new_coords = {}

    cores = np.asarray(
        sorted(
            list(set(np.unique(dataset.core_label.data).astype(np.int32)) - set([0]))
        ),
        dtype=np.int32,
    )
    new_coords["core"] = cores

    anvils = np.asarray(
        sorted(
            list(
                (
                    set(np.unique(dataset.thick_anvil_label.data))
                    | set(np.unique(dataset.thin_anvil_label.data))
                )
                - set([0])
            )
        ),
        dtype=np.int32,
    )
    new_coords["anvil"] = anvils

    if "core_step_label" in dataset.data_vars:
        core_steps = np.asarray(
            sorted(
                list(
                    set(np.unique(dataset.core_step_label.data).astype(np.int32))
                    - set([0])
                )
            ),
            dtype=np.int32,
        )
        new_coords["core_step"] = core_steps

    if "thick_anvil_step_label" in dataset.data_vars:
        thick_anvil_steps = np.asarray(
            sorted(
                list(
                    set(np.unique(dataset.thick_anvil_step_label.data).astype(np.int32))
                    - set([0])
                )
            ),
            dtype=np.int32,
        )
        new_coords["thick_anvil_step"] = thick_anvil_steps

    if "thin_anvil_step_label" in dataset.data_vars:
        thin_anvil_steps = np.asarray(
            sorted(
                list(
                    set(np.unique(dataset.thin_anvil_step_label.data).astype(np.int32))
                    - set([0])
                )
            ),
            dtype=np.int32,
        )
        new_coords["thin_anvil_step"] = thin_anvil_steps

    # Need to check if any coords already exist in the dataset, if so select those values:
    if any([coord in dataset.coords for coord in new_coords]):
        dataset = dataset.sel(**{
            coord:new_coords[coord] for coord in new_coords if coord in dataset.coords 
        })

    return dataset.assign_coords(new_coords)

def find_max_overlap(x, atol, max_label):
    overlap_counts = np.bincount(x, minlength=max_label + 1)

    wh_overlap = np.argmax(overlap_counts)

    return wh_overlap if overlap_counts[wh_overlap] >= atol else 0

def link_cores_and_anvils(
    dataset: xr.Dataset, atol: int = 5, add_cores_to_anvils: bool = True
) -> None:
    # core_anvil_index = apply_func_to_labels(
    #     dataset.core_label.to_numpy(),
    #     dataset.thick_anvil_label.to_numpy(),
    #     func=find_overlap_mode,
    #     index=dataset.core.to_numpy(),
    #     default=0,
    # )
    comp_func = partial(
        find_max_overlap,
        atol=atol,
        max_label=dataset.core.max().item(),
    )

    from scipy.ndimage import labeled_comprehension

    core_anvil_index = labeled_comprehension(
        dataset.thick_anvil_label.values.flatten(),
        dataset.core_label.values.flatten(),
        dataset.core.values,
        comp_func,
        int,
        0,
    )

    add_dataarray_to_ds(
        create_dataarray(
            core_anvil_index,
            ("core",),
            "core_anvil_index",
            long_name="anvil index for each core",
            dtype=np.int32,
        ),
        dataset,
    )

    if add_cores_to_anvils:
        remapped_cores = remap_labels(
            dataset.core_label.values, 
            locations=dataset.core.values, 
            new_labels=core_anvil_index
        )
        wh_core_labels = remapped_cores != 0
        dataset.thick_anvil_label.data[wh_core_labels] = remapped_cores[wh_core_labels]
        dataset.thin_anvil_label.data[wh_core_labels] = remapped_cores[wh_core_labels]

    anvil_core_count = np.asarray(
        [np.sum(core_anvil_index == i) for i in dataset.anvil.data]
    )
    add_dataarray_to_ds(
        create_dataarray(
            anvil_core_count,
            ("anvil",),
            "anvil_core_count",
            long_name="number of cores associated with anvil",
            dtype=np.int32,
        ),
        dataset,
    )


def link_step_labels(dataset: xr.Dataset) -> None:
    # Add linking indices between each label
    # core_step_core_index = labeled_comprehension(
    #     dataset.core_label.data,
    #     dataset.core_step_label.data,
    #     find_overlap_mode,
    #     dtype=np.int32,
    #     default=0,
    # )

    core_step_core_index = apply_func_to_labels(
        dataset.core_step_label.to_numpy(),
        dataset.core_label.to_numpy(),
        func=find_overlap_mode,
        index=dataset.core_step.to_numpy(),
        default=0,
    )

    add_dataarray_to_ds(
        create_dataarray(
            core_step_core_index,
            ("core_step",),
            "core_step_core_index",
            long_name="core index for each core time step",
            dtype=np.int32,
        ),
        dataset,
    )

    # core_anvil_index = labeled_comprehension(
    #     dataset.thick_anvil_label.data,
    #     dataset.core_label.data,
    #     find_overlap_mode,
    #     dtype=np.int32,
    #     default=0,
    # )

    # thick_anvil_step_anvil_index = labeled_comprehension(
    #     dataset.thick_anvil_label.data,
    #     dataset.thick_anvil_step_label.data,
    #     find_overlap_mode,
    #     dtype=np.int32,
    #     default=0,
    # )
    thick_anvil_step_anvil_index = apply_func_to_labels(
        dataset.thick_anvil_step_label.to_numpy(),
        dataset.thick_anvil_label.to_numpy(),
        func=find_overlap_mode,
        index=dataset.thick_anvil_step.to_numpy(),
        default=0,
    )

    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_step_anvil_index,
            ("thick_anvil_step",),
            "thick_anvil_step_anvil_index",
            long_name="anvil index for each thick anvil time step",
            dtype=np.int32,
        ),
        dataset,
    )

    # thin_anvil_step_anvil_index = labeled_comprehension(
    #     dataset.thin_anvil_label.data,
    #     dataset.thin_anvil_step_label.data,
    #     find_overlap_mode,
    #     dtype=np.int32,
    #     default=0,
    # )
    thin_anvil_step_anvil_index = apply_func_to_labels(
        dataset.thin_anvil_step_label.to_numpy(),
        dataset.thin_anvil_label.to_numpy(),
        func=find_overlap_mode,
        index=dataset.thin_anvil_step.to_numpy(),
        default=0,
    )

    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_step_anvil_index,
            ("thin_anvil_step",),
            "thin_anvil_step_anvil_index",
            long_name="anvil index for each thin anvil time step",
            dtype=np.int32,
        ),
        dataset,
    )


def find_edge_labels(
    labels, label_dim, start_date=None, end_date=None, max_time_gap=900
):
    # Find labels that touch the sides of the domain:
    edge_labels = np.unique(
        np.concatenate(
            [
                np.unique(labels[:, 0]),
                np.unique(labels[:, -1]),
                np.unique(labels[:, :, 0]),
                np.unique(labels[:, :, -1]),
            ]
        )
    )

    if edge_labels[0] == 0:
        edge_labels = edge_labels[1:]

    edge_label_flag = xr.zeros_like(label_dim, dtype=bool)
    edge_label_flag.loc[edge_labels] = True

    # Find labels at the start and end of the period:
    if (start_date is not None) and (get_datetime_from_coord(labels.t)[0] < start_date):
        start_labels = np.unique(labels.sel(t=slice(None, start_date)))
    else:
        start_labels = np.unique(labels[0])

    if (end_date is not None) and (get_datetime_from_coord(labels.t)[-1] > end_date):
        end_labels = np.unique(labels.sel(t=slice(end_date, None)))
    else:
        end_labels = np.unique(labels[-1])

    # Now find time gaps and append to start and end labels
    time_gap_locs = np.where((labels.t.diff("t").astype(int) / 1e9 > max_time_gap))[0]

    if time_gap_locs.size:
        start_labels = np.unique(
            np.concatenate([start_labels, np.unique(labels.isel(t=time_gap_locs))])
        )

        end_labels = np.unique(
            np.concatenate([end_labels, np.unique(labels.isel(t=time_gap_locs + 1))])
        )

    if start_labels[0] == 0:
        start_labels = start_labels[1:]

    start_label_flag = xr.zeros_like(label_dim, dtype=bool)
    start_label_flag.loc[start_labels] = True

    if end_labels[0] == 0:
        end_labels = end_labels[1:]

    end_label_flag = xr.zeros_like(label_dim, dtype=bool)
    end_label_flag.loc[end_labels] = True

    return edge_label_flag, start_label_flag, end_label_flag


def flag_edge_labels(dataset, start_date=None, end_date=None, max_time_gap=900):
    # Add edge flags for cores
    core_edge_label_flag, core_start_label_flag, core_end_label_flag = find_edge_labels(
        dataset.core_label, dataset.core, start_date, end_date, max_time_gap
    )

    add_dataarray_to_ds(
        create_dataarray(
            core_edge_label_flag.data,
            ("core",),
            "core_edge_label_flag",
            long_name="flag for cores intersecting the domain edge",
            dtype=bool,
        ),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            core_start_label_flag,
            ("core",),
            "core_start_label_flag",
            long_name="flag for cores intersecting the domain start time",
            dtype=bool,
        ),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            core_end_label_flag,
            ("core",),
            "core_end_label_flag",
            long_name="flag for cores intersecting the domain end time",
            dtype=bool,
        ),
        dataset,
    )

    # Add edge flags for thick_anvils
    (
        thick_anvil_edge_label_flag,
        thick_anvil_start_label_flag,
        thick_anvil_end_label_flag,
    ) = find_edge_labels(
        dataset.thick_anvil_label, dataset.anvil, start_date, end_date, max_time_gap
    )

    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_edge_label_flag.data,
            ("anvil",),
            "thick_anvil_edge_label_flag",
            long_name="flag for thick anvils intersecting the domain edge",
            dtype=bool,
        ),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_start_label_flag.data,
            ("anvil",),
            "thick_anvil_start_label_flag",
            long_name="flag for thick anvils intersecting the domain start time",
            dtype=bool,
        ),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_end_label_flag.data,
            ("anvil",),
            "thick_anvil_end_label_flag",
            long_name="flag for thick anvils intersecting the domain end time",
            dtype=bool,
        ),
        dataset,
    )

    # Add edge flags for thin_anvils
    (
        thin_anvil_edge_label_flag,
        thin_anvil_start_label_flag,
        thin_anvil_end_label_flag,
    ) = find_edge_labels(
        dataset.thin_anvil_label, dataset.anvil, start_date, end_date, max_time_gap
    )

    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_edge_label_flag.data,
            ("anvil",),
            "thin_anvil_edge_label_flag",
            long_name="flag for thin anvils intersecting the domain edge",
            dtype=bool,
        ),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_start_label_flag.data,
            ("anvil",),
            "thin_anvil_start_label_flag",
            long_name="flag for thin anvils intersecting the domain start time",
            dtype=bool,
        ),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_end_label_flag.data,
            ("anvil",),
            "thin_anvil_end_label_flag",
            long_name="flag for thin anvils intersecting the domain end time",
            dtype=bool,
        ),
        dataset,
    )


def flag_nan_adjacent_labels(dataset: xr.Dataset, da: xr.DataArray) -> None:
    core_nan_flag = xr.zeros_like(dataset.core, dtype=bool)
    thick_anvil_nan_flag = xr.zeros_like(dataset.anvil, dtype=bool)
    thin_anvil_nan_flag = xr.zeros_like(dataset.anvil, dtype=bool)

    if np.any(np.isnan(da.data)):
        wh_nan = ndi.binary_dilation(np.isnan(da.data), structure=np.ones([3, 3, 3]))
        core_nan_labels = np.unique(dataset.core_label.data[wh_nan])

        if core_nan_labels[0] == 0:
            core_nan_flag.loc[core_nan_labels[1:]] = True
        else:
            core_nan_flag.loc[core_nan_labels] = True

        thick_anvil_nan_labels = np.unique(dataset.thick_anvil_label.data[wh_nan])

        if thick_anvil_nan_labels[0] == 0:
            thick_anvil_nan_flag.loc[thick_anvil_nan_labels[1:]] = True
        else:
            thick_anvil_nan_flag.loc[thick_anvil_nan_labels] = True

        thin_anvil_nan_labels = np.unique(dataset.thin_anvil_label.data[wh_nan])

        if thin_anvil_nan_labels[0] == 0:
            thin_anvil_nan_flag.loc[thin_anvil_nan_labels[1:]] = True
        else:
            thin_anvil_nan_flag.loc[thin_anvil_nan_labels] = True

    add_dataarray_to_ds(
        create_dataarray(
            core_nan_flag.data,
            ("core",),
            "core_nan_flag",
            long_name="flag for cores intersecting missing values",
            dtype=bool,
        ),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_nan_flag.data,
            ("anvil",),
            "thick_anvil_nan_flag",
            long_name="flag for thick anvils intersecting missing values",
            dtype=bool,
        ),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_nan_flag.data,
            ("anvil",),
            "thin_anvil_nan_flag",
            long_name="flag for thin anvils intersecting missing values",
            dtype=bool,
        ),
        dataset,
    )


def calculate_label_properties(dataset: xr.Dataset) -> None:
    # Pixel count and area
    core_total_pixels = np.bincount(dataset.core_label.data.ravel())[dataset.core.data]
    add_dataarray_to_ds(
        create_dataarray(
            core_total_pixels,
            ("core",),
            "core_pixel_count",
            long_name="total number of pixels for core",
            dtype=np.int32,
        ),
        dataset,
    )

    core_step_pixels = np.bincount(dataset.core_step_label.data.ravel())[
        dataset.core_step.data
    ]
    add_dataarray_to_ds(
        create_dataarray(
            core_step_pixels,
            ("core_step",),
            "core_step_pixel_count",
            long_name="total number of pixels for core at time step",
            dtype=np.int32,
        ),
        dataset,
    )

    core_total_area = labeled_comprehension(
        dataset.area.data[np.newaxis, ...],
        dataset.core_label.data,
        np.nansum,
        index=dataset.core.data,
        dtype=np.float32,
        default=np.nan,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_total_area,
            ("core",),
            "core_total_area",
            long_name="total area of core",
            dtype=np.float32,
        ),
        dataset,
    )

    core_step_area = labeled_comprehension(
        dataset.area.data[np.newaxis, ...],
        dataset.core_step_label.data,
        np.nansum,
        index=dataset.core_step.data,
        dtype=np.float32,
        default=np.nan,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_step_area,
            ("core_step",),
            "core_step_area",
            long_name="area of core at time step",
            dtype=np.float32,
        ),
        dataset,
    )

    core_step_max_area_index = np.asarray(
        [
            dataset.core_step[dataset.core_step_core_index.data == i][
                np.argmax(core_step_area[dataset.core_step_core_index.data == i])
            ]
            for i in dataset.core.data
        ]
    )

    core_max_area = dataset.core_step_area.loc[core_step_max_area_index].data

    add_dataarray_to_ds(
        create_dataarray(
            core_max_area,
            ("core",),
            "core_max_area",
            long_name="maximum area of core",
            dtype=np.float32,
        ),
        dataset,
    )

    # Time stats for core
    core_start_t = labeled_comprehension(
        dataset.t.data[:, np.newaxis, np.newaxis],
        dataset.core_label.data,
        np.nanmin,
        index=dataset.core.data,
        dtype="datetime64[ns]",
        default=None,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_start_t,
            ("core",),
            "core_start_t",
            long_name="initial detection time of core",
            dtype="datetime64[ns]",
        ),
        dataset,
    )

    core_end_t = labeled_comprehension(
        dataset.t.data[:, np.newaxis, np.newaxis],
        dataset.core_label.data,
        np.nanmax,
        index=dataset.core.data,
        dtype="datetime64[ns]",
        default=None,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_end_t,
            ("core",),
            "core_end_t",
            long_name="final detection time of core",
            dtype="datetime64[ns]",
        ),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            core_end_t - core_start_t,
            ("core",),
            "core_lifetime",
            long_name="total lifetime of core",
            dtype="timedelta64[ns]",
        ),
        dataset,
    )

    core_step_t = labeled_comprehension(
        dataset.t.data[:, np.newaxis, np.newaxis],
        dataset.core_step_label.data,
        np.nanmin,
        index=dataset.core_step.data,
        dtype="datetime64[ns]",
        default=None,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_step_t,
            ("core_step",),
            "core_step_t",
            long_name="time of core at time step",
            dtype="datetime64[ns]",
        ),
        dataset,
    )

    core_max_area_t = dataset.core_step_t.loc[core_step_max_area_index].data
    add_dataarray_to_ds(
        create_dataarray(
            core_max_area_t,
            ("core",),
            "core_max_area_t",
            long_name="time of core maximum area",
            dtype="datetime64[ns]",
        ),
        dataset,
    )

    # Pixel count and area for thick anvil
    # thick_anvil_total_pixels = np.bincount(dataset.thick_anvil_label.data.ravel())[
    #     dataset.anvil.data
    # ]
    # add_dataarray_to_ds(
    #     create_dataarray(
    #         thick_anvil_total_pixels,
    #         ("anvil",),
    #         "thick_anvil_pixel_count",
    #         long_name="total number of pixels for thick anvil",
    #         dtype=np.int32,
    #     ),
    #     dataset,
    # )

    thick_anvil_step_pixels = np.bincount(dataset.thick_anvil_step_label.data.ravel())[
        dataset.thick_anvil_step.data
    ]
    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_step_pixels,
            ("thick_anvil_step",),
            "thick_anvil_step_pixel_count",
            long_name="total number of pixels for thick anvil at time step",
            dtype=np.int32,
        ),
        dataset,
    )

    thick_anvil_total_area = labeled_comprehension(
        dataset.area.data[np.newaxis, ...],
        dataset.thick_anvil_label.data,
        np.nansum,
        index=dataset.anvil.data,
        dtype=np.float32,
        default=np.nan,
    )
    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_total_area,
            ("anvil",),
            "thick_anvil_total_area",
            long_name="total area of thick anvil",
            dtype=np.float32,
        ),
        dataset,
    )

    thick_anvil_step_area = labeled_comprehension(
        dataset.area.data[np.newaxis, ...],
        dataset.thick_anvil_step_label.data,
        np.nansum,
        index=dataset.thick_anvil_step.data,
        dtype=np.float32,
        default=np.nan,
    )
    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_step_area,
            ("thick_anvil_step",),
            "thick_anvil_step_area",
            long_name="area of thick anvil at time step",
            dtype=np.float32,
        ),
        dataset,
    )

    # thick_anvil_step_max_area_index = np.asarray(
    #     [
    #         dataset.thick_anvil_step[dataset.thick_anvil_step_anvil_index.data == i][
    #             np.argmax(
    #                 thick_anvil_step_area[
    #                     dataset.thick_anvil_step_anvil_index.data == i
    #                 ]
    #             )
    #         ]
    #         for i in dataset.anvil.data
    #     ]
    # )

    # thick_anvil_max_area = dataset.thick_anvil_step_area.loc[
    #     thick_anvil_step_max_area_index
    # ].data

    # add_dataarray_to_ds(
    #     create_dataarray(
    #         thick_anvil_max_area,
    #         ("anvil",),
    #         "thick_anvil_max_area",
    #         long_name="maximum area of thick anvil",
    #         dtype=np.float32,
    #     ),
    #     dataset,
    # )

    # Time stats for thick_anvil
    thick_anvil_start_t = labeled_comprehension(
        dataset.t.data[:, np.newaxis, np.newaxis],
        dataset.thick_anvil_label.data,
        np.nanmin,
        index=dataset.anvil.data,
        dtype="datetime64[ns]",
        default=None,
    )
    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_start_t,
            ("anvil",),
            "thick_anvil_start_t",
            long_name="initial detection time of thick anvil",
            dtype="datetime64[ns]",
        ),
        dataset,
    )

    thick_anvil_end_t = labeled_comprehension(
        dataset.t.data[:, np.newaxis, np.newaxis],
        dataset.thick_anvil_label.data,
        np.nanmax,
        index=dataset.anvil.data,
        dtype="datetime64[ns]",
        default=None,
    )
    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_end_t,
            ("anvil",),
            "thick_anvil_end_t",
            long_name="final detection time of thick anvil",
            dtype="datetime64[ns]",
        ),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_end_t - thick_anvil_start_t,
            ("anvil",),
            "thick_anvil_lifetime",
            long_name="total lifetime of thick anvil",
            dtype="timedelta64[ns]",
        ),
        dataset,
    )

    thick_anvil_step_t = labeled_comprehension(
        dataset.t.data[:, np.newaxis, np.newaxis],
        dataset.thick_anvil_step_label.data,
        np.nanmin,
        index=dataset.thick_anvil_step.data,
        dtype="datetime64[ns]",
        default=None,
    )
    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_step_t,
            ("thick_anvil_step",),
            "thick_anvil_step_t",
            long_name="time of thick anvil at time step",
            dtype="datetime64[ns]",
        ),
        dataset,
    )

    # thick_anvil_max_area_t = dataset.thick_anvil_step_t.loc[
    #     thick_anvil_step_max_area_index
    # ].data
    # add_dataarray_to_ds(
    #     create_dataarray(
    #         thick_anvil_max_area_t,
    #         ("anvil",),
    #         "thick_anvil_max_area_t",
    #         long_name="time of thick anvil maximum area",
    #         dtype="datetime64[ns]",
    #     ),
    #     dataset,
    # )

    # Pixel count and area for thin anvil
    # thin_anvil_total_pixels = np.bincount(dataset.thin_anvil_label.data.ravel())[
    #     dataset.anvil.data
    # ]
    # add_dataarray_to_ds(
    #     create_dataarray(
    #         thin_anvil_total_pixels,
    #         ("anvil",),
    #         "thin_anvil_pixel_count",
    #         long_name="total number of pixels for thin anvil",
    #         dtype=np.int32,
    #     ),
    #     dataset,
    # )

    thin_anvil_step_pixels = np.bincount(dataset.thin_anvil_step_label.data.ravel())[
        dataset.thin_anvil_step.data
    ]
    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_step_pixels,
            ("thin_anvil_step",),
            "thin_anvil_step_pixel_count",
            long_name="total number of pixels for thin anvil at time step",
            dtype=np.int32,
        ),
        dataset,
    )

    # thin_anvil_total_area = labeled_comprehension(
    #     dataset.area.data[np.newaxis, ...],
    #     dataset.thin_anvil_label.data,
    #     np.nansum,
    #     index=dataset.anvil.data,
    #     dtype=np.float32,
    #     default=np.nan,
    # )
    # add_dataarray_to_ds(
    #     create_dataarray(
    #         thin_anvil_total_area,
    #         ("anvil",),
    #         "thin_anvil_total_area",
    #         long_name="total area of thin anvil",
    #         dtype=np.float32,
    #     ),
    #     dataset,
    # )

    thin_anvil_step_area = labeled_comprehension(
        dataset.area.data[np.newaxis, ...],
        dataset.thin_anvil_step_label.data,
        np.nansum,
        index=dataset.thin_anvil_step.data,
        dtype=np.float32,
        default=np.nan,
    )
    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_step_area,
            ("thin_anvil_step",),
            "thin_anvil_step_area",
            long_name="area of thin anvil at time step",
            dtype=np.float32,
        ),
        dataset,
    )

    # thin_anvil_step_max_area_index = np.asarray(
    #     [
    #         dataset.thin_anvil_step[dataset.thin_anvil_step_anvil_index.data == i][
    #             np.argmax(
    #                 thin_anvil_step_area[dataset.thin_anvil_step_anvil_index.data == i]
    #             )
    #         ]
    #         for i in dataset.anvil.data
    #     ]
    # )

    # thin_anvil_max_area = dataset.thin_anvil_step_area.loc[
    #     thin_anvil_step_max_area_index
    # ].data

    # add_dataarray_to_ds(
    #     create_dataarray(
    #         thin_anvil_max_area,
    #         ("anvil",),
    #         "thin_anvil_max_area",
    #         long_name="maximum area of thin anvil",
    #         dtype=np.float32,
    #     ),
    #     dataset,
    # )

    # Time stats for thin_anvil
    thin_anvil_start_t = labeled_comprehension(
        dataset.t.data[:, np.newaxis, np.newaxis],
        dataset.thin_anvil_label.data,
        np.nanmin,
        index=dataset.anvil.data,
        dtype="datetime64[ns]",
        default=None,
    )
    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_start_t,
            ("anvil",),
            "thin_anvil_start_t",
            long_name="initial detection time of thin anvil",
            dtype="datetime64[ns]",
        ),
        dataset,
    )

    thin_anvil_end_t = labeled_comprehension(
        dataset.t.data[:, np.newaxis, np.newaxis],
        dataset.thin_anvil_label.data,
        np.nanmax,
        index=dataset.anvil.data,
        dtype="datetime64[ns]",
        default=None,
    )
    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_end_t,
            ("anvil",),
            "thin_anvil_end_t",
            long_name="final detection time of thin anvil",
            dtype="datetime64[ns]",
        ),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_end_t - thin_anvil_start_t,
            ("anvil",),
            "thin_anvil_lifetime",
            long_name="total lifetime of thin anvil",
            dtype="timedelta64[ns]",
        ),
        dataset,
    )

    thin_anvil_step_t = labeled_comprehension(
        dataset.t.data[:, np.newaxis, np.newaxis],
        dataset.thin_anvil_step_label.data,
        np.nanmin,
        index=dataset.thin_anvil_step.data,
        dtype="datetime64[ns]",
        default=None,
    )
    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_step_t,
            ("thin_anvil_step",),
            "thin_anvil_step_t",
            long_name="time of thin anvil at time step",
            dtype="datetime64[ns]",
        ),
        dataset,
    )

    # thin_anvil_max_area_t = dataset.thin_anvil_step_t.loc[
    #     thin_anvil_step_max_area_index
    # ].data
    # add_dataarray_to_ds(
    #     create_dataarray(
    #         thin_anvil_max_area_t,
    #         ("anvil",),
    #         "thin_anvil_max_area_t",
    #         long_name="time of thin anvil maximum area",
    #         dtype="datetime64[ns]",
    #     ),
    #     dataset,
    # )

    # Flag no growth anvils
    # anvil_no_growth_flag = np.asarray(
    #     [
    #         True
    #         if dataset.anvil_core_count.loc[i] == 1
    #         and dataset.thick_anvil_max_area_t.loc[i]
    #         <= dataset.core_end_t[dataset.core_anvil_index.data == i]
    #         else False
    #         for i in dataset.anvil.data
    #     ]
    # )

    # add_dataarray_to_ds(
    #     create_dataarray(
    #         anvil_no_growth_flag,
    #         ("anvil",),
    #         "anvil_no_growth_flag",
    #         long_name="flag for anvils that do not grow after core activity ends",
    #         dtype=bool,
    #     ),
    #     dataset,
    # )

    # Location and lat/lon for cores
    area_stack = np.repeat(dataset.area.data[np.newaxis, ...], dataset.t.size, 0)
    if (len(dataset.lat.shape)==2) and (len(dataset.lon.shape)==2):
        lat_stack = np.repeat(dataset.lat.data[np.newaxis, ...], dataset.t.size, 0)
        lon_stack = np.repeat(dataset.lon.data[np.newaxis, ...], dataset.t.size, 0)
    elif (len(dataset.lat.shape)==1) and (len(dataset.lon.shape)==1):
        lons, lats = np.meshgrid(dataset.lon, dataset.lat)
        lat_stack = np.repeat(lats[np.newaxis, ...], dataset.t.size, 0)
        lon_stack = np.repeat(lons[np.newaxis, ...], dataset.t.size, 0)
    

    xx, yy = np.meshgrid(dataset.x, dataset.y)
    x_stack = np.repeat(xx[np.newaxis, ...], dataset.t.size, 0)
    y_stack = np.repeat(yy[np.newaxis, ...], dataset.t.size, 0)

    core_step_x = apply_weighted_func_to_labels(
        dataset.core_step_label.data,
        x_stack,
        area_stack,
        lambda x, w: np.average(x, weights=w),
    )[dataset.core_step.data - 1]
    add_dataarray_to_ds(
        create_dataarray(
            core_step_x,
            ("core_step",),
            "core_step_x",
            long_name="x location of core at time step",
            dtype=np.float32,
        ),
        dataset,
    )

    core_step_y = apply_weighted_func_to_labels(
        dataset.core_step_label.data,
        y_stack,
        area_stack,
        lambda x, w: np.average(x, weights=w),
    )[dataset.core_step.data - 1]
    add_dataarray_to_ds(
        create_dataarray(
            core_step_y,
            ("core_step",),
            "core_step_y",
            long_name="y location of core at time step",
            dtype=np.float32,
        ),
        dataset,
    )

    core_step_lat = apply_weighted_func_to_labels(
        dataset.core_step_label.data,
        lat_stack,
        area_stack,
        lambda x, w: np.average(x, weights=w),
    )[dataset.core_step.data - 1]
    add_dataarray_to_ds(
        create_dataarray(
            core_step_lat,
            ("core_step",),
            "core_step_lat",
            long_name="latitude of core at time step",
            dtype=np.float32,
        ),
        dataset,
    )

    core_step_lon = apply_weighted_func_to_labels(
        dataset.core_step_label.data,
        lon_stack,
        area_stack,
        lambda x, w: np.average(x, weights=w),
    )[dataset.core_step.data - 1]
    add_dataarray_to_ds(
        create_dataarray(
            core_step_lon,
            ("core_step",),
            "core_step_lon",
            long_name="longitude of core at time step",
            dtype=np.float32,
        ),
        dataset,
    )

    core_start_index = np.asarray(
        [
            np.nanmin(dataset.core_step[dataset.core_step_core_index.data == i])
            for i in dataset.core.data
        ]
    )
    # core_end_index = np.asarray(
    #     [
    #         np.nanmax(dataset.core_step[dataset.core_step_core_index.data == i])
    #         for i in dataset.core.data
    #     ]
    # )

    core_start_x = dataset.core_step_x.loc[core_start_index].data
    core_start_y = dataset.core_step_y.loc[core_start_index].data
    core_start_lat = dataset.core_step_lat.loc[core_start_index].data
    core_start_lon = dataset.core_step_lon.loc[core_start_index].data

    add_dataarray_to_ds(
        create_dataarray(
            core_start_x,
            ("core",),
            "core_start_x",
            long_name="initial x location of core",
            dtype=np.float32,
        ),
        dataset,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_start_y,
            ("core",),
            "core_start_y",
            long_name="initial y location of core",
            dtype=np.float32,
        ),
        dataset,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_start_lat,
            ("core",),
            "core_start_lat",
            long_name="initial latitude of core",
            dtype=np.float32,
        ),
        dataset,
    )
    add_dataarray_to_ds(
        create_dataarray(
            core_start_lon,
            ("core",),
            "core_start_lon",
            long_name="initial longitude of core",
            dtype=np.float32,
        ),
        dataset,
    )

    # Location and lat/lon for anvils
    thick_anvil_step_x = apply_weighted_func_to_labels(
        dataset.thick_anvil_step_label.data,
        x_stack,
        area_stack,
        lambda x, w: np.average(x, weights=w),
    )[dataset.thick_anvil_step.data - 1]
    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_step_x,
            ("thick_anvil_step",),
            "thick_anvil_step_x",
            long_name="x location of thick anvil at time step",
            dtype=np.float32,
        ),
        dataset,
    )

    thick_anvil_step_y = apply_weighted_func_to_labels(
        dataset.thick_anvil_step_label.data,
        y_stack,
        area_stack,
        lambda x, w: np.average(x, weights=w),
    )[dataset.thick_anvil_step.data - 1]
    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_step_y,
            ("thick_anvil_step",),
            "thick_anvil_step_y",
            long_name="y location of thick anvil at time step",
            dtype=np.float32,
        ),
        dataset,
    )

    thick_anvil_step_lat = apply_weighted_func_to_labels(
        dataset.thick_anvil_step_label.data,
        lat_stack,
        area_stack,
        lambda x, w: np.average(x, weights=w),
    )[dataset.thick_anvil_step.data - 1]
    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_step_lat,
            ("thick_anvil_step",),
            "thick_anvil_step_lat",
            long_name="latitude of thick anvil at time step",
            dtype=np.float32,
        ),
        dataset,
    )

    thick_anvil_step_lon = apply_weighted_func_to_labels(
        dataset.thick_anvil_step_label.data,
        lon_stack,
        area_stack,
        lambda x, w: np.average(x, weights=w),
    )[dataset.thick_anvil_step.data - 1]
    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_step_lon,
            ("thick_anvil_step",),
            "thick_anvil_step_lon",
            long_name="longitude of thick anvil at time step",
            dtype=np.float32,
        ),
        dataset,
    )

    # thick_anvil_start_index = np.asarray(
    #     [
    #         np.nanmin(
    #             dataset.thick_anvil_step[dataset.thick_anvil_step_anvil_index.data == i]
    #         )
    #         for i in dataset.anvil.data
    #     ]
    # )
    # thick_anvil_end_index = np.asarray(
    #     [
    #         np.nanmax(
    #             dataset.thick_anvil_step[dataset.thick_anvil_step_anvil_index.data == i]
    #         )
    #         for i in dataset.anvil.data
    #     ]
    # )

    # thick_anvil_start_x = dataset.thick_anvil_step_x.loc[thick_anvil_start_index].data
    # thick_anvil_start_y = dataset.thick_anvil_step_y.loc[thick_anvil_start_index].data
    # thick_anvil_start_lat = dataset.thick_anvil_step_lat.loc[
    #     thick_anvil_start_index
    # ].data
    # thick_anvil_start_lon = dataset.thick_anvil_step_lon.loc[
    #     thick_anvil_start_index
    # ].data

    # add_dataarray_to_ds(
    #     create_dataarray(
    #         thick_anvil_start_x,
    #         ("anvil",),
    #         "anvil_start_x",
    #         long_name="initial x location of anvil",
    #         dtype=np.float32,
    #     ),
    #     dataset,
    # )
    # add_dataarray_to_ds(
    #     create_dataarray(
    #         thick_anvil_start_y,
    #         ("anvil",),
    #         "anvil_start_y",
    #         long_name="initial y location of anvil",
    #         dtype=np.float32,
    #     ),
    #     dataset,
    # )
    # add_dataarray_to_ds(
    #     create_dataarray(
    #         thick_anvil_start_lat,
    #         ("anvil",),
    #         "anvil_start_lat",
    #         long_name="initial latitude of anvil",
    #         dtype=np.float32,
    #     ),
    #     dataset,
    # )
    # add_dataarray_to_ds(
    #     create_dataarray(
    #         thick_anvil_start_lon,
    #         ("anvil",),
    #         "anvil_start_lon",
    #         long_name="initial longitude of anvil",
    #         dtype=np.float32,
    #     ),
    #     dataset,
    # )

    thin_anvil_step_x = apply_weighted_func_to_labels(
        dataset.thin_anvil_step_label.data,
        x_stack,
        area_stack,
        lambda x, w: np.average(x, weights=w),
    )[dataset.thin_anvil_step.data - 1]
    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_step_x,
            ("thin_anvil_step",),
            "thin_anvil_step_x",
            long_name="x location of thin anvil at time step",
            dtype=np.float32,
        ),
        dataset,
    )

    thin_anvil_step_y = apply_weighted_func_to_labels(
        dataset.thin_anvil_step_label.data,
        y_stack,
        area_stack,
        lambda x, w: np.average(x, weights=w),
    )[dataset.thin_anvil_step.data - 1]
    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_step_y,
            ("thin_anvil_step",),
            "thin_anvil_step_y",
            long_name="y location of thin anvil at time step",
            dtype=np.float32,
        ),
        dataset,
    )

    thin_anvil_step_lat = apply_weighted_func_to_labels(
        dataset.thin_anvil_step_label.data,
        lat_stack,
        area_stack,
        lambda x, w: np.average(x, weights=w),
    )[dataset.thin_anvil_step.data - 1]
    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_step_lat,
            ("thin_anvil_step",),
            "thin_anvil_step_lat",
            long_name="latitude of thin anvil at time step",
            dtype=np.float32,
        ),
        dataset,
    )

    thin_anvil_step_lon = apply_weighted_func_to_labels(
        dataset.thin_anvil_step_label.data,
        lon_stack,
        area_stack,
        lambda x, w: np.average(x, weights=w),
    )[dataset.thin_anvil_step.data - 1]
    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_step_lon,
            ("thin_anvil_step",),
            "thin_anvil_step_lon",
            long_name="longitude of thin anvil at time step",
            dtype=np.float32,
        ),
        dataset,
    )