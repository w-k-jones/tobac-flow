import numpy as np
import xarray as xr
from dateutil.parser import parse as parse_date
from .abi import get_abi_lat_lon, get_abi_pixel_area

def get_coord_bin_edges(coord):
    # Now set up the bin edges for the goes dataset coordinates. Note we multiply by height to convert into the Proj coords
    bins = np.zeros(coord.size+1)
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
    shape = tuple([ds.coords[k].size for k in ds.coords if k in set(ds.coords.keys()).intersection(set(ds.dims))])
    return shape

def get_ds_core_coords(ds):
    coords = {k:ds.coords[k] for k in ds.coords if k in set(ds.coords.keys()).intersection(set(ds.dims))}
    return coords

def get_datetime_from_coord(coord):
    return [parse_date(t.item()) for t in coord.astype('datetime64[s]').astype(str)]

def time_diff(datetime_list):
    return [(datetime_list[1]-datetime_list[0]).total_seconds()/60] \
             + [(datetime_list[i+2]-datetime_list[i]).total_seconds()/120 \
                for i in range(len(datetime_list)-2)] \
             + [(datetime_list[-1]-datetime_list[-2]).total_seconds()/60]

def get_time_diff_from_coord(coord):
    return np.array(time_diff(get_datetime_from_coord(coord)))

def create_dataarray(array, dims, name, long_name=None, units=None, dtype=None):
    da = xr.DataArray(array.astype(dtype), dims=dims)
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
    mean_da = create_dataarray(np.nanmean(da.data), tuple(), f"{da.name}_mean", long_name=f"Mean of {da.long_name}", units=da.units, dtype=da.dtype)
    std_da = create_dataarray(np.nanstd(da.data), tuple(), f"{da.name}_std", long_name=f"Standard deviation of {da.long_name}", units=da.units, dtype=da.dtype)
    median_da = create_dataarray(np.median(da.data), tuple(), f"{da.name}_median", long_name=f"Median of {da.long_name}", units=da.units, dtype=da.dtype)
    max_da = create_dataarray(np.nanmax(da.data), tuple(), f"{da.name}_max", long_name=f"Maximum of {da.long_name}", units=da.units, dtype=da.dtype)
    min_da = create_dataarray(np.nanmin(da.data), tuple(), f"{da.name}_min", long_name=f"Minimum of {da.long_name}", units=da.units, dtype=da.dtype)
    return mean_da, std_da, median_da, max_da, min_da

def get_spatial_stats(da):
    mean_da = create_dataarray(da.mean(('x','y')).data, ('t',), f"{da.name}_spatial_mean", long_name=f"Spatial mean of {da.long_name}", units=da.units, dtype=da.dtype)
    std_da = create_dataarray(da.std(('x','y')).data, ('t',), f"{da.name}_spatial_std", long_name=f"Spatial standard deviation of {da.long_name}", units=da.units, dtype=da.dtype)
    median_da = create_dataarray(da.median(('x','y')).data, ('t',), f"{da.name}_spatial_median", long_name=f"Spatial median of {da.long_name}", units=da.units, dtype=da.dtype)
    max_da = create_dataarray(da.max(('x','y')).data, ('t',), f"{da.name}_spatial_max", long_name=f"Spatial maximum of {da.long_name}", units=da.units, dtype=da.dtype)
    min_da = create_dataarray(da.min(('x','y')).data, ('t',), f"{da.name}_spatial_min", long_name=f"Spatial minimum of {da.long_name}", units=da.units, dtype=da.dtype)
    return mean_da, std_da, median_da, max_da, min_da

def get_temporal_stats(da):
    mean_da = create_dataarray(da.mean('t').data, ('y', 'x'), f"{da.name}_temporal_mean", long_name=f"Temporal mean of {da.long_name}", units=da.units, dtype=da.dtype)
    std_da = create_dataarray(da.std('t').data, ('y', 'x'), f"{da.name}_temporal_std", long_name=f"Temporal standard deviation of {da.long_name}", units=da.units, dtype=da.dtype)
    median_da = create_dataarray(da.median('t').data, ('y', 'x'), f"{da.name}_temporal_median", long_name=f"Temporal median of {da.long_name}", units=da.units, dtype=da.dtype)
    max_da = create_dataarray(da.max('t').data, ('y', 'x'), f"{da.name}_temporal_max", long_name=f"Temporal maximum of {da.long_name}", units=da.units, dtype=da.dtype)
    min_da = create_dataarray(da.min('t').data, ('y', 'x'), f"{da.name}_temporal_min", long_name=f"Temporal minimum of {da.long_name}", units=da.units, dtype=da.dtype)
    return mean_da, std_da, median_da, max_da, min_da

def n_unique_along_axis(a, axis=0):
    b = np.sort(np.moveaxis(a, axis, 0), axis=0)
    return (b[1:] != b[:-1]).sum(axis=0) + (np.count_nonzero(a, axis=axis)==a.shape[axis]).astype(int)

def add_dataarray_to_ds(da, ds):
    ds[da.name] = da

def create_new_goes_ds(goes_ds):
    goes_coords = {'t':goes_ds.t, 'y':goes_ds.y, 'x':goes_ds.x,
                   'y_image':goes_ds.y_image, 'x_image':goes_ds.x_image}

    new_ds = xr.Dataset(coords=goes_coords)
    new_ds["goes_imager_projection"] = goes_ds.goes_imager_projection
    lat, lon = get_abi_lat_lon(new_ds)
    add_dataarray_to_ds(create_dataarray(lat, ('y', 'x'), 'lat', long_name="latitude", dtype=np.float32), new_ds)
    add_dataarray_to_ds(create_dataarray(lon, ('y', 'x'), 'lon', long_name="longitude", dtype=np.float32), new_ds)
    add_dataarray_to_ds(create_dataarray(get_abi_pixel_area(new_ds), ('y', 'x'), 'area',
                                         long_name="pixel area", units='km^2', dtype=np.float32), new_ds)
    return new_ds
