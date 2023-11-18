import numpy as np
import pyproj
from pyproj import Transformer

# from glmtools.io.lightning_ellipse import lightning_ellipse_rev
# from lmatools.coordinateSystems import CoordinateSystem
# from lmatools.grid.fixed import get_GOESR_coordsys
# from lmatools.coordinateSystems import GeostationaryFixedGridSystem, GeographicSystem


# equatorial, polar radii
lightning_ellipse_rev = {
    # Values at launch
    0: (6.394140e6, 6.362755e6),
    # DO.07, late 2018. First Virts revision.
    # The GRS80 altitude + 6 km differs by about 3 m from the value above
    # which is the exact that was provided at the time of launch. Use the
    # original value instead of doing the math.
    # 6.35675231414e6+6.0e3
    1: (6.378137e6 + 14.0e3, 6.362755e6),
}
this_ellps = 0
ltg_ellps_re, ltg_ellps_rp = lightning_ellipse_rev[this_ellps]


# Functions from GLM notebook for parallax correction
def semiaxes_to_invflattening(semimajor, semiminor):
    """Calculate the inverse flattening from the semi-major
    and semi-minor axes of an ellipse"""
    rf = semimajor / (semimajor - semiminor)
    return rf


def get_GOESR_coordsys(sat_lon_nadir=-75.0):
    """
    Values from the GOES-R PUG Volume 3, L1b data

    Returns geofixcs, grs80lla: the fixed grid coordinate system and the
    latitude, longitude, altitude coordinate system referenced to the GRS80
    ellipsoid used by GOES-R as its earth reference.
    """
    goes_sweep = "x"  # Meteosat is 'y'
    ellipse = "GRS80"
    datum = "WGS84"
    sat_ecef_height = 35786023.0
    geofixcs = GeostationaryFixedGridSystem(
        subsat_lon=sat_lon_nadir,
        ellipse=ellipse,
        datum=datum,
        sweep_axis=goes_sweep,
        sat_ecef_height=sat_ecef_height,
    )
    grs80lla = GeographicSystem(ellipse="GRS80", datum="WGS84")
    return geofixcs, grs80lla


class CoordinateSystem(object):
    """The abstract coordinate system handling provided here works as follows.

    Each coordinate system must be able to convert data to a common coordinate system, which is chosen to be ECEF cartesian.
    data -> common system
    common system -> dislpay coordinates
    This is implemented by the fromECEF and toECEF methods in each coordinate system object.
    User code is responsible for taking data in its native coord system,
        transforming it using to/fromECEF using the a coord system appropriate to the data, and then
        transforming that data to the final coordinate system using another coord system.

    Subclasses should maintain an attribute ERSxyz that can be used in
        transformations to/from an ECEF cartesian system, e.g.
        >>> self.ERSxyz = pyproj.CRS(proj='geocent', ellps='WGS84', datum='WGS84')
        >>> self.ERSlla = pyproj.CRS(proj='latlong', ellps='WGS84', datum='WGS84')
        >>> projectedData = pyproj.transform(self.ERSlla, self.ERSxyz, lat, lon, alt )
    The ECEF system has its origin at the center of the earth, with the +Z toward the north pole,
        +X toward (lat=0, lon=0), and +Y right-handed orthogonal to +X, +Z

    Depends on pyproj, http://code.google.com/p/pyproj/ to handle the ugly details of
    various map projections, geodetic transforms, etc.

    "You can think of a coordinate system as being something like character encodings,
    but messier, and without an obvious winner like UTF-8." - Django OSCON tutorial, 2007
    http://toys.jacobian.org/presentations/2007/oscon/tutorial/
    """

    # WGS84xyz = pyproj.CRS(proj='geocent',  ellps='WGS84', datum='WGS84')

    def coordinates():
        """Return a tuple of standarized coordinate names"""
        raise NotImplemented

    def fromECEF(self, x, y, z):
        """Take ECEF x, y, z values and return x, y, z in the coordinate system defined by the object subclass"""
        raise NotImplemented

    def toECEF(self, x, y, z):
        """Take x, y, z in the coordinate system defined by the object subclass and return ECEF x, y, z"""
        raise NotImplemented


class GeostationaryFixedGridSystem(CoordinateSystem):
    def __init__(
        self,
        subsat_lon=0.0,
        subsat_lat=0.0,
        sweep_axis="y",
        sat_ecef_height=35785831.0,
        ellipse="WGS84",
        datum="WGS84",
    ):
        """
        Satellite height is with respect to the ellipsoid. Fixed grid
        coordinates are in radians.
        """
        self.ECEFxyz = pyproj.CRS(proj="geocent", ellps=ellipse)  # , datum=datum)
        self.fixedgrid = pyproj.CRS(
            proj="geos",
            lon_0=subsat_lon,
            lat_0=subsat_lat,
            h=sat_ecef_height,
            x_0=0.0,
            y_0=0.0,
            units="m",
            sweep=sweep_axis,
            ellps=ellipse,
        )
        self.h = sat_ecef_height
        self.transformer_to = Transformer.from_crs(self.fixedgrid, self.ECEFxyz)
        self.transformer_from = Transformer.from_crs(self.ECEFxyz, self.fixedgrid)

    def toECEF(self, x, y, z):
        X, Y, Z = x * self.h, y * self.h, z * self.h
        return self.transformer_to.transform(X, Y, Z)

    def fromECEF(self, x, y, z):
        X, Y, Z = self.transformer_from.transform(x, y, z)
        return X / self.h, Y / self.h, Z / self.h


class GeographicSystem(CoordinateSystem):
    """
    Coordinate system defined on the surface of the earth using latitude,
    longitude, and altitude, referenced by default to the WGS84 ellipse.

    Alternately, specify the ellipse shape using an ellipse known
    to pyproj, or [NOT IMPLEMENTED] specify r_equator and r_pole directly.
    """

    def __init__(self, ellipse="WGS84", datum="WGS84", r_equator=None, r_pole=None):
        if (r_equator is not None) | (r_pole is not None):
            if r_pole is None:
                r_pole = r_equator
            self.ERSlla = pyproj.CRS(proj="latlong", a=r_equator, b=r_pole)
        else:
            # lat lon alt in some earth reference system
            self.ERSlla = pyproj.CRS(proj="latlong", ellps=ellipse, datum=datum)
        self.ERSxyz = pyproj.CRS(proj="geocent", ellps=ellipse, datum=datum)
        self.transformer_to = Transformer.from_crs(self.ERSlla, self.ERSxyz)
        self.transformer_from = Transformer.from_crs(self.ERSxyz, self.ERSlla)

    def toECEF(self, lon, lat, alt):
        lat = np.atleast_1d(lat)  # proj doesn't like scalars
        lon = np.atleast_1d(lon)
        alt = np.atleast_1d(alt)
        if lat.shape[0] == 0:
            return lon, lat, alt  # proj doesn't like empties
        projectedData = np.array(self.transformer_to.transform(lon, lat, alt))
        if len(projectedData.shape) == 1:
            return projectedData[0], projectedData[1], projectedData[2]
        else:
            return projectedData[0, :], projectedData[1, :], projectedData[2, :]

    def fromECEF(self, x, y, z):
        x = np.atleast_1d(x)  # proj doesn't like scalars
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)
        if x.shape[0] == 0:
            return x, y, z  # proj doesn't like empties
        projectedData = np.array(self.transformer_from.transform(x, y, z))
        if len(projectedData.shape) == 1:
            return projectedData[0], projectedData[1], projectedData[2]
        else:
            return projectedData[0, :], projectedData[1, :], projectedData[2, :]


class GeostationaryFixedGridSystemAltEllipse(CoordinateSystem):
    def __init__(
        self,
        subsat_lon=0.0,
        subsat_lat=0.0,
        sweep_axis="y",
        sat_ecef_height=35785831.0,
        semimajor_axis=None,
        semiminor_axis=None,
        datum="WGS84",
    ):
        """
        Satellite height is with respect to an arbitray ellipsoid whose
        shape is given by semimajor_axis (equatorial) and semiminor_axis(polar)

        Fixed grid coordinates are in radians.
        """
        rf = semiaxes_to_invflattening(semimajor_axis, semiminor_axis)
        # print("Defining alt ellipse for Geostationary with rf=", rf)
        self.ECEFxyz = pyproj.CRS(proj="geocent", a=semimajor_axis, rf=rf)
        self.fixedgrid = pyproj.CRS(
            proj="geos",
            lon_0=subsat_lon,
            lat_0=subsat_lat,
            h=sat_ecef_height,
            x_0=0.0,
            y_0=0.0,
            units="m",
            sweep=sweep_axis,
            a=semimajor_axis,
            rf=rf,
        )
        self.h = sat_ecef_height
        self.transformer_to = Transformer.from_crs(self.fixedgrid, self.ECEFxyz)
        self.transformer_from = Transformer.from_crs(self.ECEFxyz, self.fixedgrid)

    def toECEF(self, x, y, z):
        X, Y, Z = x * self.h, y * self.h, z * self.h
        return self.transformer_to.transform(X, Y, Z)

    def fromECEF(self, x, y, z):
        X, Y, Z = self.transformer_from.transform(x, y, z)
        return X / self.h, Y / self.h, Z / self.h


class GeographicSystemAltEllps(CoordinateSystem):
    """
    Coordinate system defined on the surface of the earth using latitude,
    longitude, and altitude, referenced by default to the WGS84 ellipse.

    Alternately, specify the ellipse shape using an ellipse known
    to pyproj, or [NOT IMPLEMENTED] specify r_equator and r_pole directly.
    """

    def __init__(self, ellipse="WGS84", datum="WGS84", r_equator=None, r_pole=None):
        if (r_equator is not None) | (r_pole is not None):
            rf = semiaxes_to_invflattening(r_equator, r_pole)
            # print("Defining alt ellipse for Geographic with rf", rf)
            self.ERSlla = pyproj.CRS(proj="latlong", a=r_equator, rf=rf)  # datum=datum,
            self.ERSxyz = pyproj.CRS(proj="geocent", a=r_equator, rf=rf)  # datum=datum,
        else:
            # lat lon alt in some earth reference system
            self.ERSlla = pyproj.CRS(proj="latlong", ellps=ellipse, datum=datum)
            self.ERSxyz = pyproj.CRS(proj="geocent", ellps=ellipse, datum=datum)
        self.transformer_to = Transformer.from_crs(self.ERSlla, self.ERSxyz)
        self.transformer_from = Transformer.from_crs(self.ERSxyz, self.ERSlla)

    def toECEF(self, lon, lat, alt):
        projectedData = np.array(self.transformer_to.transform(lon, lat, alt))
        if len(projectedData.shape) == 1:
            return projectedData[0], projectedData[1], projectedData[2]
        else:
            return projectedData[0, :], projectedData[1, :], projectedData[2, :]

    def fromECEF(self, x, y, z):
        projectedData = np.array(self.transformer_from.transform(x, y, z))
        if len(projectedData.shape) == 1:
            return projectedData[0], projectedData[1], projectedData[2]
        else:
            return projectedData[0, :], projectedData[1, :], projectedData[2, :]


def get_GOESR_coordsys_alt_ellps(sat_lon_nadir=-75.0):
    goes_sweep = "x"  # Meteosat is 'y'
    datum = "WGS84"
    sat_ecef_height = 35786023.0
    geofixcs = GeostationaryFixedGridSystemAltEllipse(
        subsat_lon=sat_lon_nadir,
        semimajor_axis=ltg_ellps_re,
        semiminor_axis=ltg_ellps_rp,
        datum=datum,
        sweep_axis=goes_sweep,
        sat_ecef_height=sat_ecef_height,
    )
    grs80lla = GeographicSystemAltEllps(
        r_equator=ltg_ellps_re, r_pole=ltg_ellps_rp, datum="WGS84"
    )
    return geofixcs, grs80lla
