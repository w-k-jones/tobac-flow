import numpy as np
import xarray as xr
from scipy.stats import circmean
from pyproj import Geod

GEO = Geod(ellps="WGS84")


def get_grid_spacing_from_lat_lon(
    lat: np.ndarray[float], lon: np.ndarray[float]
) -> np.ndarray[float]:
    """
    Get the grid spacing of each pixel from 2d latitude and longitude arrays
    """
    dy, dx = np.zeros(lat.shape, dtype=float), np.zeros(lat.shape, dtype=float)
    dy[:-1] = GEO.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])[-1] / 1e3
    dx[:, :-1] = GEO.inv(lon[:, :-1], lat[:, :-1], lon[:, 1:], lat[:, 1:])[-1] / 1e3
    dy[1:] += dy[:-1]
    dy[1:-1] /= 2
    dx[:, 1:] += dx[:, :-1]
    dx[:, 1:-1] /= 2

    return dx, dy


def get_area_from_lat_lon(
    lat: np.ndarray[float], lon: np.ndarray[float]
) -> np.ndarray[float]:
    """
    Get the area of each pixel from 2d latitude and longitude arrays
    """
    dx, dy = get_grid_spacing_from_lat_lon(lat, lon)
    area = dx * dy
    return area


def add_area_to_dataset(dataset: xr.Dataset, squeeze: bool = False) -> xr.Dataset:
    area_attrs = {"long_name": "pixel area", "standard_name": "area", "units": "km2"}
    if "t" in dataset.lat.dims:
        lat = dataset.lat.isel(t=0)
        lon = dataset.lon.isel(t=0)
        area = get_area_from_lat_lon(lat.data, lon.data)
        wh_t = dataset.lat.dims.index("t")
        if not squeeze:
            area = np.repeat(np.expand_dims(area, wh_t), dataset.t.size, wh_t)
            area_da = xr.DataArray(
                area, dataset.lat.coords, dataset.lat.dims, attrs=area_attrs
            )
        else:
            xr.DataArray(area, lat.coords, lat.dims, attrs=area_attrs)
    else:
        lat = dataset.lat
        lon = dataset.lon
        area = get_area_from_lat_lon(lat.data, lon.data).astype(np.float32)
        area_da = xr.DataArray(
            area, dataset.lat.coords, dataset.lat.dims, attrs=area_attrs
        )
    dataset["area"] = area_da
    return dataset


def get_mean_object_azimuth_and_speed(
    lons: np.ndarray[float], lats: np.ndarray[float], t: np.ndarray[np.datetime64]
) -> tuple[float, float]:
    """
    Given array of longitude, latitude and time, find the average direction of
    propagation and speed
    """
    sort_args = np.argsort(t)
    lifetime_seconds = np.diff(t[sort_args]).astype(int) / 1e9
    azimuths, _, distances = GEO.inv(
        lons[sort_args][:-1],
        lats[sort_args][:-1],
        lons[sort_args][1:],
        lats[sort_args][1:],
    )
    speeds = distances / lifetime_seconds
    wh_finite = np.logical_and(np.isfinite(azimuths), np.isfinite(speeds))
    if np.any(wh_finite):
        return circmean(azimuths[wh_finite], high=180, low=-180), np.mean(
            speeds[wh_finite]
        )
    else:
        return np.nan, np.nan


__all__ = (
    "get_grid_spacing_from_lat_lon",
    "get_area_from_lat_lon",
    "add_area_to_dataset",
    "get_mean_object_azimuth_and_speed",
)
