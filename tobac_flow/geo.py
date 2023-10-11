"""
geo
===

This module contains functions relating to goemetric operations with 
    observational data
"""
from datetime import datetime
import numpy as np


def get_sza(dt, lat, lon):
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


def get_sza_and_azi(dt, lat, lon):
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
    azi = (
        np.sin(delta) * np.cos(np.radians(lat))
        - np.cos(delta) * np.sin(np.radians(lat)) * np.cos(omega)
    ) / np.cos(np.pi / 2.0 - sza)

    # IF (azi lt -1.0) THEN azi=-1.0
    # IF (azi gt 1.0) THEN azi=1.0
    azim = np.arccos(azi)
    # print('Solar Azimuth (in degs) :        ',r2d(azim) )
    # print('Solar Zenith Angle (in degs) :',r2d(sza) )
    # print('Solar Elevation Angle (in degs) :',r2d(solel) )
    # print('Hour Angle (in degrees) :           ',r2d(omega) )
    # Function return is solar zenith in radians
    return sza, azim
