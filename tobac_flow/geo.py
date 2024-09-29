"""
geo
===

This module contains functions relating to goemetric operations with 
    observational data
"""

from datetime import datetime
import numpy as np
from pyproj import Geod


def get_sza(dt: datetime, lat: float, lon: float) -> tuple[float, float]:
    """
    Get the solar zenith angle at a specific time/lat/lon
    """
    srd = (dt - datetime(dt.year, 1, 1)).days + 1
    hour = dt.hour
    minute = dt.minute
    utc = srd + hour / 24.0 + float(minute / (24.0 * 60.0))
    # print('UTC = ',utc)

    # dim = 1

    # calculate the number of days since the 1.1. of the specific year
    daynum = np.floor(utc) + 1
    # print(daynum)

    # calculate the relative SUN-earth distance for the given day
    # resulting from the elliptic orbit of the earth
    eta = 2.0 * np.pi * daynum / 365.0
    # fluxfac = (
    #     1.000110
    #     + 0.034221 * np.cos(eta)
    #     + 0.000719 * np.cos(2.0 * eta)
    #     + 0.001280 * np.sin(eta)
    #     + 0.000077 * np.sin(2.0 * eta)
    # )
    # dist = 1.0 / np.sqrt(fluxfac)

    # calculate the solar declination for the given day
    # the declination varies due to the fact, that the earth rotation axis
    # is not perpendicular to the ecliptic plane
    delta = (
        0.006918
        - 0.399912 * np.cos(eta)
        - 0.006758 * np.cos(2.0 * eta)
        - 0.002697 * np.cos(3.0 * eta)
        + 0.070257 * np.sin(eta)
        + 0.000907 * np.sin(2.0 * eta)
        + 0.001480 * np.sin(3.0 * eta)
    )

    # equation of time, used to compensate for the earth's elliptical orbit
    # around the sun and its axial tilt when calculating solar time
    # eqt is the correction in hours
    et = 2.0 * np.pi * daynum / 366.0
    eqt = (
        0.0072 * np.cos(et)
        - 0.0528 * np.cos(2.0 * et)
        - 0.0012 * np.cos(3.0 * et)
        - 0.1229 * np.sin(et)
        - 0.1565 * np.sin(2.0 * et)
        - 0.0041 * np.sin(3.0 * et)
    )

    # calculate the solar zenith angle
    # dtr = !pi/180. ; degrees to radian conversion factor
    time = (utc + 1.0 - daynum) * 24  # time in hours
    omega = np.radians((360.0 / 24.0) * (time + lon / 15.0 + eqt - 12.0))
    sunh = np.sin(delta) * np.sin(np.radians(lat)) + np.cos(delta) * np.cos(
        np.radians(lat)
    ) * np.cos(omega)

    # IF (sunh lt -1.0) THEN sunh=-1.0
    # IF (sunh gt 1.0) THEN sunh=1.0
    solel = np.arcsin(sunh)
    sza = np.pi / 2.0 - solel

    # Solar azimuth added by yaswant
    # azi = (
    #     np.sin(delta) * np.cos(np.radians(lat))
    #     - np.cos(delta) * np.sin(np.radians(lat)) * np.cos(omega)
    # ) / np.cos(np.pi / 2.0 - sza)

    # IF (azi lt -1.0) THEN azi=-1.0
    # IF (azi gt 1.0) THEN azi=1.0
    # azim = np.arccos(azi)
    # print('Solar Azimuth (in degs) :        ',r2d(azim) )
    # print('Solar Zenith Angle (in degs) :',r2d(sza) )
    # print('Solar Elevation Angle (in degs) :',r2d(solel) )
    # print('Hour Angle (in degrees) :           ',r2d(omega) )
    # Function return is solar zenith in radians
    return sza


def get_sza_and_azi(date: datetime, lat: float, lon: float) -> tuple[float, float]:
    """
    Get the solar zenith angle at a specific time/lat/lon
    """
    day_of_year = int(date.strftime("%j"))
    hour_of_day = (
        date - datetime(date.year, date.month, date.day, 0, 0, 0)
    ).total_seconds() / 3600

    # calculate the relative Sun-earth distance for the given day
    # resulting from the elliptic orbit of the earth
    equation_of_time_approx = 2.0 * np.pi * day_of_year / 365.0
    # fluxfac = (
    #     1.000110
    #     + 0.034221 * np.cos(eta)
    #     + 0.000719 * np.cos(2.0 * eta)
    #     + 0.001280 * np.sin(eta)
    #     + 0.000077 * np.sin(2.0 * eta)
    # )
    # dist = 1.0 / np.sqrt(fluxfac)

    # calculate the solar declination for the given day
    # the declination varies due to the fact, that the earth rotation axis
    # is not perpendicular to the ecliptic plane
    solar_declination = (
        0.006918
        - 0.399912 * np.cos(equation_of_time_approx)
        - 0.006758 * np.cos(2.0 * equation_of_time_approx)
        - 0.002697 * np.cos(3.0 * equation_of_time_approx)
        + 0.070257 * np.sin(equation_of_time_approx)
        + 0.000907 * np.sin(2.0 * equation_of_time_approx)
        + 0.001480 * np.sin(3.0 * equation_of_time_approx)
    )

    # equation of time, used to compensate for the earth's elliptical orbit
    # around the sun and its axial tilt when calculating solar time
    # eqt is the correction in hours
    equation_of_time = 2.0 * np.pi * day_of_year / 366.0
    equation_of_time = (
        0.0072 * np.cos(equation_of_time)
        - 0.0528 * np.cos(2.0 * equation_of_time)
        - 0.0012 * np.cos(3.0 * equation_of_time)
        - 0.1229 * np.sin(equation_of_time)
        - 0.1565 * np.sin(2.0 * equation_of_time)
        - 0.0041 * np.sin(3.0 * equation_of_time)
    )

    # calculate the solar zenith angle
    omega = np.radians(
        (360.0 / 24.0) * (hour_of_day + lon / 15.0 + equation_of_time - 12.0)
    )
    sunh = np.sin(solar_declination) * np.sin(np.radians(lat)) + np.cos(
        solar_declination
    ) * np.cos(np.radians(lat)) * np.cos(omega)

    solar_elevation = np.arcsin(np.clip(sunh, -1, 1))
    solar_zenith_angle = np.pi / 2.0 - solar_elevation

    # Solar azimuth added by yaswant
    azimuth = (
        np.sin(solar_declination) * np.cos(np.radians(lat))
        - np.cos(solar_declination) * np.sin(np.radians(lat)) * np.cos(omega)
    ) / np.cos(np.pi / 2.0 - solar_zenith_angle)

    solar_azimuth_angle = np.arccos(np.clip(azimuth, -1, 1))

    return np.degrees(solar_zenith_angle), np.degrees(solar_azimuth_angle)


def get_satellite_viewing_angles(
    lat: float,
    lon: float,
    sat_lat: float = 0,
    sat_lon: float = 0,
    sat_alt: float = 35_793,
) -> tuple[float, float]:
    """Calculate satellite zenith and azimuth angles

    Parameters
    ----------
    lat : float
        latitude of surface point in degrees
    lon : float
        longitude of surface point in degrees
    sat_lat : float, optional
        latitude of sub-satellite point in degrees, by default 0
    sat_lon : float, optional
        longitude of sub-satellite point in degrees, by default 0
    sat_alt : float, optional
        altitude of satellite in km, by default 35_793 (geostationary orbit
        height over average earth radius)

    Returns
    -------
    tuple[float, float]
        satellite zenith and azimuth angles in degrees
    """

    Re = 6_371
    Rgeo = sat_alt + Re

    # Caclulate the beta angle
    cos_beta = np.cos(np.radians(lat - sat_lat)) * np.cos(np.radians(lon - sat_lon))
    sin_beta = np.sin(np.arccos(cos_beta))

    # Calculate satellite zenith angle
    geo_dist = (
        Rgeo**2 + Re**2 - 2 * Rgeo * Re * cos_beta
    ) ** 0.5  # distance from surface to satellite
    sin_theta = (Rgeo * sin_beta) / geo_dist
    zenith_angle = np.degrees(np.arcsin(sin_theta))
    # Find where satellite-surface path intersects the earth and make these > 90
    zenith_angle = np.where(
        geo_dist**2 < (Rgeo**2 - Re**2), zenith_angle, 180 - zenith_angle
    )

    # Calculate satellite azimuthal angle 
    x_sat = np.cos(np.radians(lat - sat_lat)) * np.sin(np.radians(lon - sat_lon))
    y_sat = np.sin(np.radians(lat - sat_lat))
    azimuth_angle = np.where(
        np.isfinite(x_sat), np.degrees(np.arctan2(x_sat, y_sat)) % 360, np.nan
    )

    return zenith_angle, azimuth_angle


def get_pixel_lengths(lat, lon) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the length scales in x and y of each pixel in the input dataset in
        km.
    """
    g = Geod(ellps="WGS84")
    dy, dx = np.zeros(lat.shape, dtype=float), np.zeros(lat.shape, dtype=float)
    dy[:-1] = g.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])[-1] / 1e3
    dx[:, :-1] = g.inv(lon[:, :-1], lat[:, :-1], lon[:, 1:], lat[:, 1:])[-1] / 1e3
    dy[1:] += dy[:-1]
    dy[1:-1] /= 2
    dx[:, 1:] += dx[:, :-1]
    dx[:, 1:-1] /= 2
    return dx, dy


def get_pixel_area(lat, lon) -> np.ndarray:
    """
    Returns the area of each pixel in the input dataset in square km
    """
    dx, dy = get_pixel_lengths(lat, lon)
    area = dx * dy
    return area
