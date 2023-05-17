import numpy as np
import xarray as xr
from pyproj import Geod

def get_area_from_lat_lon(
    lat:np.ndarray[float], lon:np.ndarray[float]
) -> np.ndarray[float]:
    g = Geod(ellps="WGS84")
    dy, dx = np.zeros(lat.shape, dtype=float), np.zeros(lat.shape, dtype=float)
    dy[:-1] = g.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])[-1] / 1e3
    dx[:, :-1] = g.inv(lon[:, :-1], lat[:, :-1], lon[:, 1:], lat[:, 1:])[-1] / 1e3
    dy[1:] += dy[:-1]
    dy[1:-1] /= 2
    dx[:, 1:] += dx[:, :-1]
    dx[:, 1:-1] /= 2
    area = dx * dy

    return area

def add_area_to_dataset(dataset: xr.Dataset, squeeze:bool = False) -> xr.Dataset:
    area_attrs = {"long_name":"pixel area", "standard_name":"area", "units":"km2"}
    if "t" in dataset.lat.dims:
        lat = dataset.lat.isel(t=0)
        lon = dataset.lon.isel(t=0)
        area = get_area_from_lat_lon(lat.data, lon.data)
        wh_t = dataset.lat.dims.index("t")
        if not squeeze:
            area = np.repeat(
                np.expand_dims(area, wh_t), dataset.t.size, wh_t
            )
            area_da = xr.DataArray(
                area, 
                dataset.lat.coords,
                dataset.lat.dims, 
                attrs=area_attrs
            )
        else:
            xr.DataArray(
                area, 
                lat.coords,
                lat.dims, 
                attrs=area_attrs
            )
    else:
        lat = dataset.lat
        lon = dataset.lon
        area = get_area_from_lat_lon(lat.data, lon.data)
        area_da = xr.DataArray(
            area, 
            dataset.lat.coords,
            dataset.lat.dims, 
            attrs=area_attrs
        )
    dataset["area"] = area_da
    return dataset
