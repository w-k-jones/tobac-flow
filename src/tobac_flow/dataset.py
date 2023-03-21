import numpy as np
from scipy import stats
from scipy import ndimage as ndi
import xarray as xr
from dateutil.parser import parse as parse_date
from tobac_flow.abi import get_abi_lat_lon, get_abi_pixel_area
from tobac_flow.label import slice_labels
from tobac_flow.utils import labeled_comprehension, apply_weighted_func_to_labels


def get_coord_bin_edges(coord):
    # Now set up the bin edges for the goes dataset coordinates. Note we multiply by height to convert into the Proj coords
    bins = np.zeros(coord.size + 1)
    bins[:-1] += coord.data
    bins[1:] += coord.data
    bins[1:-1] /= 2
    return bins


def get_ds_bin_edges(ds, dims=None):
    if dims is None:
        dims = [coord for coord in ds.coords]
    elif isinstance(dims, str):
        dims = [dims]

    return [get_coord_bin_edges(ds.coords[dim]) for dim in dims]


def get_ds_shape(ds):
    shape = tuple(
        [
            ds.coords[k].size
            for k in ds.coords
            if k in set(ds.coords.keys()).intersection(set(ds.dims))
        ]
    )
    return shape


def get_ds_core_coords(ds):
    coords = {
        k: ds.coords[k]
        for k in ds.coords
        if k in set(ds.coords.keys()).intersection(set(ds.dims))
    }
    return coords


def get_datetime_from_coord(coord):
    return [parse_date(t.item()) for t in coord.astype("datetime64[s]").astype(str)]


def time_diff(datetime_list):
    return (
        [(datetime_list[1] - datetime_list[0]).total_seconds() / 60]
        + [
            (datetime_list[i + 2] - datetime_list[i]).total_seconds() / 120
            for i in range(len(datetime_list) - 2)
        ]
        + [(datetime_list[-1] - datetime_list[-2]).total_seconds() / 60]
    )


def get_time_diff_from_coord(coord):
    return np.array(time_diff(get_datetime_from_coord(coord)))


def create_dataarray(
    array, dims, name, coords=None, long_name=None, units=None, dtype=None
):
    da = xr.DataArray(array.astype(dtype), coords=coords, dims=dims)
    da.name = name
    da.attrs["standard_name"] = name
    if long_name:
        da.attrs["long_name"] = long_name
    else:
        da.attrs["long_name"] = name
    if units:
        da.attrs["units"] = units

    return da


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


def n_unique_along_axis(a, axis=0):
    b = np.sort(np.moveaxis(a, axis, 0), axis=0)
    return (b[1:] != b[:-1]).sum(axis=0) + (
        np.count_nonzero(a, axis=axis) == a.shape[axis]
    ).astype(int)


def add_dataarray_to_ds(da, ds):
    ds[da.name] = da


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
    core_step_labels = slice_labels(dataset.core_label.data)

    add_dataarray_to_ds(
        create_dataarray(
            core_step_labels,
            ("t", "y", "x"),
            "core_step_label",
            coords={"t": dataset.t},
            long_name="labels for detected cores at each time step",
            units="",
            dtype=np.int32,
        ).sel(t=dataset.t),
        dataset,
    )

    thick_anvil_step_labels = slice_labels(dataset.thick_anvil_label.data)

    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_step_labels,
            ("t", "y", "x"),
            "thick_anvil_step_label",
            coords={"t": dataset.t},
            long_name="labels for detected thick anvil regions at each time step",
            units="",
            dtype=np.int32,
        ).sel(t=dataset.t),
        dataset,
    )

    thin_anvil_step_labels = slice_labels(dataset.thin_anvil_label.data)

    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_step_labels,
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
    cores = np.unique(dataset.core_label.data).astype(np.int32)
    if cores[0] == 0 and cores.size > 1:
        cores = cores[1:]
    anvils = np.unique(dataset.thick_anvil_label.data).astype(np.int32)
    if anvils[0] == 0 and anvils.size > 1:
        anvils = anvils[1:]
    core_steps = np.unique(dataset.core_step_label.data).astype(np.int32)
    if core_steps[0] == 0 and core_steps.size > 1:
        core_steps = core_steps[1:]
    thick_anvil_steps = np.unique(dataset.thick_anvil_step_label.data).astype(np.int32)
    if thick_anvil_steps[0] == 0 and thick_anvil_steps.size > 1:
        thick_anvil_steps = thick_anvil_steps[1:]
    thin_anvil_steps = np.unique(dataset.thin_anvil_step_label.data).astype(np.int32)
    if thin_anvil_steps[0] == 0 and thin_anvil_steps.size > 1:
        thin_anvil_steps = thin_anvil_steps[1:]
    return dataset.assign_coords(
        {
            "core": cores,
            "core_step": core_steps,
            "anvil": anvils,
            "thick_anvil_step": thick_anvil_steps,
            "thin_anvil_step": thin_anvil_steps,
        }
    )


def link_step_labels(dataset: xr.Dataset) -> None:
    # Add linking indices between each label
    def find_overlap_mode(x):
        if np.any(x):
            return stats.mode(x[x != 0], keepdims=False)[0]
        else:
            return 0

    core_step_core_index = labeled_comprehension(
        dataset.core_label.data,
        dataset.core_step_label.data,
        find_overlap_mode,
        dtype=np.int32,
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

    core_anvil_index = labeled_comprehension(
        dataset.thick_anvil_label.data,
        dataset.core_label.data,
        find_overlap_mode,
        dtype=np.int32,
        default=0,
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

    thick_anvil_step_anvil_index = labeled_comprehension(
        dataset.thick_anvil_label.data,
        dataset.thick_anvil_step_label.data,
        find_overlap_mode,
        dtype=np.int32,
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

    thin_anvil_step_anvil_index = labeled_comprehension(
        dataset.thin_anvil_label.data,
        dataset.thin_anvil_step_label.data,
        find_overlap_mode,
        dtype=np.int32,
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


def flag_edge_labels(dataset, start_date, end_date):
    # Add edge flags for cores
    core_edge_labels = np.unique(
        np.concatenate(
            [
                np.unique(dataset.core_label[:, 0]),
                np.unique(dataset.core_label[:, -1]),
                np.unique(dataset.core_label[:, :, 0]),
                np.unique(dataset.core_label[:, :, -1]),
            ]
        )
    )

    core_edge_label_flag = np.zeros_like(dataset.core, bool)

    if core_edge_labels[0] == 0:
        core_edge_label_flag[core_edge_labels[1:] - 1] = True
    else:
        core_edge_label_flag[core_edge_labels - 1] = True

    core_start_labels = np.unique(dataset.core_label.sel(t=slice(None, start_date)))

    core_start_label_flag = np.zeros_like(dataset.core, bool)

    if core_start_labels[0] == 0:
        core_start_label_flag[core_start_labels[1:] - 1] = True
    else:
        core_start_label_flag[core_start_labels - 1] = True

    core_end_labels = np.unique(dataset.core_label.sel(t=slice(end_date, None)))

    core_end_label_flag = np.zeros_like(dataset.core, bool)

    if core_end_labels[0] == 0:
        core_end_label_flag[core_end_labels[1:] - 1] = True
    else:
        core_end_label_flag[core_end_labels - 1] = True

    add_dataarray_to_ds(
        create_dataarray(
            core_edge_label_flag,
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
    thick_anvil_edge_labels = np.unique(
        np.concatenate(
            [
                np.unique(dataset.thick_anvil_label[:, 0]),
                np.unique(dataset.thick_anvil_label[:, -1]),
                np.unique(dataset.thick_anvil_label[:, :, 0]),
                np.unique(dataset.thick_anvil_label[:, :, -1]),
            ]
        )
    )

    thick_anvil_edge_label_flag = np.zeros_like(dataset.anvil, bool)

    if thick_anvil_edge_labels[0] == 0:
        thick_anvil_edge_label_flag[thick_anvil_edge_labels[1:] - 1] = True
    else:
        thick_anvil_edge_label_flag[thick_anvil_edge_labels - 1] = True

    thick_anvil_start_labels = np.unique(
        dataset.thick_anvil_label.sel(t=slice(None, start_date))
    )

    thick_anvil_start_label_flag = np.zeros_like(dataset.anvil, bool)

    if thick_anvil_start_labels[0] == 0:
        thick_anvil_start_label_flag[thick_anvil_start_labels[1:] - 1] = True
    else:
        thick_anvil_start_label_flag[thick_anvil_start_labels - 1] = True

    thick_anvil_end_labels = np.unique(
        dataset.thick_anvil_label.sel(t=slice(end_date, None))
    )

    thick_anvil_end_label_flag = np.zeros_like(dataset.anvil, bool)

    if thick_anvil_end_labels[0] == 0:
        thick_anvil_end_label_flag[thick_anvil_end_labels[1:] - 1] = True
    else:
        thick_anvil_end_label_flag[thick_anvil_end_labels - 1] = True

    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_edge_label_flag,
            ("anvil",),
            "thick_anvil_edge_label_flag",
            long_name="flag for thick anvils intersecting the domain edge",
            dtype=bool,
        ),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_start_label_flag,
            ("anvil",),
            "thick_anvil_start_label_flag",
            long_name="flag for thick anvils intersecting the domain start time",
            dtype=bool,
        ),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_end_label_flag,
            ("anvil",),
            "thick_anvil_end_label_flag",
            long_name="flag for thick anvils intersecting the domain end time",
            dtype=bool,
        ),
        dataset,
    )

    # Add edge flags for thin_anvils
    thin_anvil_edge_labels = np.unique(
        np.concatenate(
            [
                np.unique(dataset.thin_anvil_label[:, 0]),
                np.unique(dataset.thin_anvil_label[:, -1]),
                np.unique(dataset.thin_anvil_label[:, :, 0]),
                np.unique(dataset.thin_anvil_label[:, :, -1]),
            ]
        )
    )

    thin_anvil_edge_label_flag = np.zeros_like(dataset.anvil, bool)

    if thin_anvil_edge_labels[0] == 0:
        thin_anvil_edge_label_flag[thin_anvil_edge_labels[1:] - 1] = True
    else:
        thin_anvil_edge_label_flag[thin_anvil_edge_labels - 1] = True

    thin_anvil_start_labels = np.unique(
        dataset.thin_anvil_label.sel(t=slice(None, start_date))
    )

    thin_anvil_start_label_flag = np.zeros_like(dataset.anvil, bool)

    if thin_anvil_start_labels[0] == 0:
        thin_anvil_start_label_flag[thin_anvil_start_labels[1:] - 1] = True
    else:
        thin_anvil_start_label_flag[thin_anvil_start_labels - 1] = True

    thin_anvil_end_labels = np.unique(
        dataset.thin_anvil_label.sel(t=slice(end_date, None))
    )

    thin_anvil_end_label_flag = np.zeros_like(dataset.anvil, bool)

    if thin_anvil_end_labels[0] == 0:
        thin_anvil_end_label_flag[thin_anvil_end_labels[1:] - 1] = True
    else:
        thin_anvil_end_label_flag[thin_anvil_end_labels - 1] = True

    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_edge_label_flag,
            ("anvil",),
            "thin_anvil_edge_label_flag",
            long_name="flag for thin anvils intersecting the domain edge",
            dtype=bool,
        ),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_start_label_flag,
            ("anvil",),
            "thin_anvil_start_label_flag",
            long_name="flag for thin anvils intersecting the domain start time",
            dtype=bool,
        ),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_end_label_flag,
            ("anvil",),
            "thin_anvil_end_label_flag",
            long_name="flag for thin anvils intersecting the domain end time",
            dtype=bool,
        ),
        dataset,
    )


def flag_nan_adjacent_labels(dataset: xr.Dataset, da: xr.DataArray) -> None:
    core_nan_flag = np.zeros_like(dataset.core, bool)
    thick_anvil_nan_flag = np.zeros_like(dataset.anvil, bool)
    thin_anvil_nan_flag = np.zeros_like(dataset.anvil, bool)

    if np.any(np.isnan(da.data)):
        wh_nan = ndi.binary_dilation(np.isnan(da.data), structure=np.ones([3, 3, 3]))
        core_nan_labels = np.unique(dataset.core_labels.data[wh_nan])

        if core_nan_labels[0] == 0:
            core_nan_flag[core_nan_labels[1:] - 1] = True
        else:
            core_nan_flag[core_nan_labels - 1] = True

        thick_anvil_nan_labels = np.unique(dataset.thick_anvil_labels.data[wh_nan])

        if thick_anvil_nan_labels[0] == 0:
            thick_anvil_nan_flag[thick_anvil_nan_labels[1:] - 1] = True
        else:
            thick_anvil_nan_flag[thick_anvil_nan_labels - 1] = True

        thin_anvil_nan_labels = np.unique(dataset.thin_anvil_labels.data[wh_nan])

        if thin_anvil_nan_labels[0] == 0:
            thin_anvil_nan_flag[thin_anvil_nan_labels[1:] - 1] = True
        else:
            thin_anvil_nan_flag[thin_anvil_nan_labels - 1] = True

    add_dataarray_to_ds(
        create_dataarray(
            core_nan_flag,
            ("core",),
            "core_nan_flag",
            long_name="flag for cores intersecting missing values",
            dtype=bool,
        ),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_nan_flag,
            ("anvil",),
            "thick_anvil_nan_flag",
            long_name="flag for thick anvils intersecting missing values",
            dtype=bool,
        ),
        dataset,
    )

    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_nan_flag,
            ("anvil",),
            "thin_anvil_nan_flag",
            long_name="flag for thin anvils intersecting missing values",
            dtype=bool,
        ),
        dataset,
    )


def calculate_label_properties(dataset: xr.Dataset) -> None:
    # Pixel count and area
    core_total_pixels = np.bincount(dataset.core_label.data.ravel())[1:]
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

    core_step_pixels = np.bincount(dataset.core_step_label.data.ravel())[1:]
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

    core_max_area = core_step_area[core_step_max_area_index - 1]

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

    core_max_area_t = core_step_t[core_step_max_area_index - 1]
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
    thick_anvil_total_pixels = np.bincount(dataset.thick_anvil_label.data.ravel())[1:]
    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_total_pixels,
            ("anvil",),
            "thick_anvil_pixel_count",
            long_name="total number of pixels for thick anvil",
            dtype=np.int32,
        ),
        dataset,
    )

    thick_anvil_step_pixels = np.bincount(dataset.thick_anvil_step_label.data.ravel())[
        1:
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

    thick_anvil_step_max_area_index = np.asarray(
        [
            dataset.thick_anvil_step[dataset.thick_anvil_step_anvil_index.data == i][
                np.argmax(
                    thick_anvil_step_area[
                        dataset.thick_anvil_step_anvil_index.data == i
                    ]
                )
            ]
            for i in dataset.anvil.data
        ]
    )

    thick_anvil_max_area = thick_anvil_step_area[thick_anvil_step_max_area_index - 1]

    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_max_area,
            ("anvil",),
            "thick_anvil_max_area",
            long_name="maximum area of thick anvil",
            dtype=np.float32,
        ),
        dataset,
    )

    # Time stats for thick_anvil
    thick_anvil_start_t = labeled_comprehension(
        dataset.t.data[:, np.newaxis, np.newaxis],
        dataset.thick_anvil_label.data,
        np.nanmin,
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

    thick_anvil_max_area_t = thick_anvil_step_t[thick_anvil_step_max_area_index - 1]
    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_max_area_t,
            ("anvil",),
            "thick_anvil_max_area_t",
            long_name="time of thick anvil maximum area",
            dtype="datetime64[ns]",
        ),
        dataset,
    )

    # Pixel count and area for thin anvil
    thin_anvil_total_pixels = np.bincount(dataset.thin_anvil_label.data.ravel())[1:]
    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_total_pixels,
            ("anvil",),
            "thin_anvil_pixel_count",
            long_name="total number of pixels for thin anvil",
            dtype=np.int32,
        ),
        dataset,
    )

    thin_anvil_step_pixels = np.bincount(dataset.thin_anvil_step_label.data.ravel())[1:]
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

    thin_anvil_total_area = labeled_comprehension(
        dataset.area.data[np.newaxis, ...],
        dataset.thin_anvil_label.data,
        np.nansum,
        dtype=np.float32,
        default=np.nan,
    )
    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_total_area,
            ("anvil",),
            "thin_anvil_total_area",
            long_name="total area of thin anvil",
            dtype=np.float32,
        ),
        dataset,
    )

    thin_anvil_step_area = labeled_comprehension(
        dataset.area.data[np.newaxis, ...],
        dataset.thin_anvil_step_label.data,
        np.nansum,
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

    thin_anvil_step_max_area_index = np.asarray(
        [
            dataset.thin_anvil_step[dataset.thin_anvil_step_anvil_index.data == i][
                np.argmax(
                    thin_anvil_step_area[dataset.thin_anvil_step_anvil_index.data == i]
                )
            ]
            for i in dataset.anvil.data
        ]
    )

    thin_anvil_max_area = thin_anvil_step_area[thin_anvil_step_max_area_index - 1]

    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_max_area,
            ("anvil",),
            "thin_anvil_max_area",
            long_name="maximum area of thin anvil",
            dtype=np.float32,
        ),
        dataset,
    )

    # Time stats for thin_anvil
    thin_anvil_start_t = labeled_comprehension(
        dataset.t.data[:, np.newaxis, np.newaxis],
        dataset.thin_anvil_label.data,
        np.nanmin,
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

    thin_anvil_max_area_t = thin_anvil_step_t[thin_anvil_step_max_area_index - 1]
    add_dataarray_to_ds(
        create_dataarray(
            thin_anvil_max_area_t,
            ("anvil",),
            "thin_anvil_max_area_t",
            long_name="time of thin anvil maximum area",
            dtype="datetime64[ns]",
        ),
        dataset,
    )

    # Flag no growth anvils
    anvil_no_growth_flag = np.asarray(
        [
            True
            if dataset.anvil_core_count.data[i - 1] == 1
            and dataset.thick_anvil_max_area_t.data[i - 1]
            <= dataset.core_end_t[dataset.core_anvil_index.data == i]
            else False
            for i in dataset.anvil.data
        ]
    )

    add_dataarray_to_ds(
        create_dataarray(
            anvil_no_growth_flag,
            ("anvil",),
            "anvil_no_growth_flag",
            long_name="flag for anvils that do not grow after core activity ends",
            dtype=bool,
        ),
        dataset,
    )

    # Location and lat/lon for cores
    area_stack = np.repeat(dataset.area.data[np.newaxis, ...], dataset.t.size, 0)
    lat_stack = np.repeat(dataset.lat.data[np.newaxis, ...], dataset.t.size, 0)
    lon_stack = np.repeat(dataset.lon.data[np.newaxis, ...], dataset.t.size, 0)

    xx, yy = np.meshgrid(dataset.x, dataset.y)
    x_stack = np.repeat(xx[np.newaxis, ...], dataset.t.size, 0)
    y_stack = np.repeat(yy[np.newaxis, ...], dataset.t.size, 0)

    core_step_x = apply_weighted_func_to_labels(
        dataset.core_step_label.data,
        x_stack,
        area_stack,
        lambda x, w: np.average(x, weights=w),
    )
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
    )
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
    )
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
    )
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
    core_end_index = np.asarray(
        [
            np.nanmax(dataset.core_step[dataset.core_step_core_index.data == i])
            for i in dataset.core.data
        ]
    )

    core_start_x = core_step_x[core_start_index - 1]
    core_start_y = core_step_y[core_start_index - 1]
    core_start_lat = core_step_lat[core_start_index - 1]
    core_start_lon = core_step_lon[core_start_index - 1]

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
    )
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
    )
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
    )
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
    )
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

    thick_anvil_start_index = np.asarray(
        [
            np.nanmin(
                dataset.thick_anvil_step[dataset.thick_anvil_step_anvil_index.data == i]
            )
            for i in dataset.anvil.data
        ]
    )
    thick_anvil_end_index = np.asarray(
        [
            np.nanmax(
                dataset.thick_anvil_step[dataset.thick_anvil_step_anvil_index.data == i]
            )
            for i in dataset.anvil.data
        ]
    )

    thick_anvil_start_x = thick_anvil_step_x[thick_anvil_start_index - 1]
    thick_anvil_start_y = thick_anvil_step_y[thick_anvil_start_index - 1]
    thick_anvil_start_lat = thick_anvil_step_lat[thick_anvil_start_index - 1]
    thick_anvil_start_lon = thick_anvil_step_lon[thick_anvil_start_index - 1]

    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_start_x,
            ("anvil",),
            "anvil_start_x",
            long_name="initial x location of anvil",
            dtype=np.float32,
        ),
        dataset,
    )
    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_start_y,
            ("anvil",),
            "anvil_start_y",
            long_name="initial y location of anvil",
            dtype=np.float32,
        ),
        dataset,
    )
    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_start_lat,
            ("anvil",),
            "anvil_start_lat",
            long_name="initial latitude of anvil",
            dtype=np.float32,
        ),
        dataset,
    )
    add_dataarray_to_ds(
        create_dataarray(
            thick_anvil_start_lon,
            ("anvil",),
            "anvil_start_lon",
            long_name="initial longitude of anvil",
            dtype=np.float32,
        ),
        dataset,
    )

    thin_anvil_step_x = apply_weighted_func_to_labels(
        dataset.thin_anvil_step_label.data,
        x_stack,
        area_stack,
        lambda x, w: np.average(x, weights=w),
    )
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
    )
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
    )
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
    )
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
