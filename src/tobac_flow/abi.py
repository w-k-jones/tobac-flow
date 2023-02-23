import numpy as np
import xarray as xr
from pyproj import Proj, Geod
from dateutil.parser import parse as parse_date
from .geo import get_sza


def get_abi_proj(dataset: xr.Dataset) -> Proj:
    """
    Return a pyproj projection from the information contained within an ABI file
    """
    return Proj(
        proj="geos",
        h=dataset.goes_imager_projection.perspective_point_height,
        lon_0=dataset.goes_imager_projection.longitude_of_projection_origin,
        lat_0=dataset.goes_imager_projection.latitude_of_projection_origin,
        sweep=dataset.goes_imager_projection.sweep_angle_axis,
    )


def get_abi_lat_lon(
    dataset: xr.Dataset, dtype: type = float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns latitude and longitude for each location in an ABI dataset
    """
    p = get_abi_proj(dataset)
    xx, yy = np.meshgrid(
        (
            dataset.x.data * dataset.goes_imager_projection.perspective_point_height
        ).astype(dtype),
        (
            dataset.y.data * dataset.goes_imager_projection.perspective_point_height
        ).astype(dtype),
    )
    lons, lats = p(xx, yy, inverse=True)
    lons[lons >= 1e30] = np.nan
    lats[lats >= 1e30] = np.nan
    return lats, lons


def get_abi_pixel_lengths(dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the length scales in x and y of each pixel in the input dataset in
        km.
    """
    g = Geod(ellps="WGS84")
    lat, lon = get_abi_lat_lon(dataset)
    dy, dx = np.zeros(lat.shape, dtype=float), np.zeros(lat.shape, dtype=float)
    dy[:-1] = g.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])[-1] / 1e3
    dx[:, :-1] = g.inv(lon[:, :-1], lat[:, :-1], lon[:, 1:], lat[:, 1:])[-1] / 1e3
    dy[1:] += dy[:-1]
    dy[1:-1] /= 2
    dx[:, 1:] += dx[:, :-1]
    dx[:, 1:-1] /= 2
    return dx, dy


def get_abi_pixel_area(dataset: xr.Dataset) -> np.ndarray:
    """
    Returns the area of each pixel in the input dataset in square km
    """
    dx, dy = get_abi_pixel_lengths(dataset)
    area = dx * dy
    return area


def get_abi_x_y(
    lat: np.ndarray, lon: np.ndarray, dataset: xr.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the x, y coordinates in the ABI projection for given latitudes and
        longitudes
    """
    p = get_abi_proj(dataset)
    x, y = p(lon, lat)
    return (
        x / dataset.goes_imager_projection.perspective_point_height,
        y / dataset.goes_imager_projection.perspective_point_height,
    )


def get_abi_ref(dataset, check=False, dtype=None):
    """
    Get reflectance values from level 1 ABI datasets (for channels 1-6)
    """
    ref = dataset.Rad * dataset.kappa0
    if check:
        DQF = dataset.DQF
        ref[DQF < 0] = np.nan
        ref[DQF > 1] = np.nan
    if dtype == None:
        return ref
    else:
        return ref.astype(dtype)


def get_abi_bt(dataset, check=False, dtype=None):
    """
    Get brightness temeprature values for level 1 ABI datasets (for channels 7-16)
    """
    bt = (
        dataset.planck_fk2 / (np.log((dataset.planck_fk1 / dataset.Rad) + 1))
        - dataset.planck_bc1
    ) / dataset.planck_bc2
    if check:
        DQF = dataset.DQF
        bt[DQF < 0] = np.nan
        bt[DQF > 1] = np.nan
    if dtype == None:
        return bt
    else:
        return bt.astype(dtype)


def get_abi_da(dataset, check=False, dtype=None):
    """
    Calibrate raw (level 1) ABI data to brightness temperature or reflectances depending on the channel
    """
    channel = dataset.band_id.data[0]
    if channel < 7:
        dataarray = get_abi_ref(dataset, check, dtype)
    else:
        dataarray = get_abi_bt(dataset, check, dtype)
    #   Add in attributes
    dataarray.attrs["goes_imager_projection"] = dataset.goes_imager_projection
    dataarray.attrs["band_id"] = dataset.band_id
    dataarray.attrs["band_wavelength"] = dataset.band_wavelength
    return dataarray


def _contrast_correction(color, contrast):
    """
    Modify the contrast of an R, G, or B color channel
    See: #www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
    Input:
        C - contrast level
    """
    F = (259 * (contrast + 255)) / (255.0 * 259 - contrast)
    COLOR = F * (color - 0.5) + 0.5
    COLOR = np.minimum(COLOR, 1)
    COLOR = np.maximum(COLOR, 0)
    return COLOR


def _get_channel_range(data, min=0, max=1, gamma=1):
    out = np.maximum(data, min)
    out = np.minimum(data, max)
    out = (out - min) / (max - min)
    out = np.power(out, gamma)
    return out


def get_abi_rgb(
    mcmip_ds, gamma=0.4, contrast=100, correct_sza=False, min_sza=0.05, night_IR=False
):
    if correct_sza:
        cossza = np.cos(get_goes_sza(mcmip_ds))
        cossza = np.maximum(cossza, min_sza)

        RGB = _get_rgb(
            mcmip_ds.CMI_C01 / cossza,
            mcmip_ds.CMI_C02 / cossza,
            mcmip_ds.CMI_C03 / cossza,
            gamma=gamma,
            contrast=contrast,
        )

    else:
        RGB = _get_rgb(
            mcmip_ds.CMI_C01,
            mcmip_ds.CMI_C02,
            mcmip_ds.CMI_C03,
            gamma=gamma,
            contrast=contrast,
        )
    if night_IR:
        IR = _contrast_correction(
            (
                1
                - (
                    (np.minimum(np.maximum(mcmip_ds.CMI_C13.data, 90), 313) - 90)
                    / (313 - 90)
                )
            ),
            contrast=contrast,
        )
        RGB = np.stack([np.maximum(RGB[..., i], IR) for i in range(3)], -1)
    return RGB


def _get_rgb(C01, C02, C03, gamma=0.4, contrast=0.05):
    R = _get_channel_range(C02, gamma=gamma)
    G = _get_channel_range(C03, gamma=gamma)
    B = _get_channel_range(C01, gamma=gamma)
    G_true = 0.48358168 * R + 0.45706946 * B + 0.06038137 * G
    G_true = np.maximum(G_true, 0)
    G_true = np.minimum(G_true, 1)
    RGB = np.maximum(
        np.minimum(
            _contrast_correction(np.stack([R, G_true, B], -1), contrast=contrast), 1
        ),
        0,
    )
    return RGB


def get_abi_deep_cloud_rgb(mcmip_ds, min_sza=0.05):
    cossza = np.cos(get_goes_sza(mcmip_ds))
    cossza = np.maximum(cossza, min_sza)

    R = _get_channel_range(mcmip_ds.CMI_C08 - mcmip_ds.CMI_C13, -35, 5)

    G = _get_channel_range(mcmip_ds.CMI_C02 / cossza, 0.7, 1.0)

    B = _get_channel_range(mcmip_ds.CMI_C13, 243.6, 292.6)

    RGB = np.maximum(
        np.minimum(
            np.stack([R, G, B], -1)
            * (np.minimum(cossza, min_sza) / min_sza)[..., np.newaxis],
            1,
        ),
        0,
    )
    return RGB


def get_goes_sza(goes_ds):
    date = parse_date(str(goes_ds.t.data))
    lats, lons = get_abi_lat_lon(goes_ds)
    return get_sza(date, lats, lons)
