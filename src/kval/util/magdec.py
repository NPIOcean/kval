'''
KVAL.UTIL.MAGDEC

Wrapper for the Python API for British Geological Survey magnetic field
calculator (https://github.com/filips123/MagneticFieldCalculator/).
'''

from typing import Iterable
from datetime import datetime
from functools import lru_cache
from pygeomag import GeoMag
from kval.util import time
import numpy as np


@lru_cache(maxsize=None)
def _geomag(coefficients_file: str) -> GeoMag:
    """GeoMag object for a coefficients file. Reading the file is the slow
    part of a declination calculation, so each file is only read once."""
    return GeoMag(coefficients_file=coefficients_file)


def _as_datetime64(date) -> np.datetime64:
    """One time input (see get_declination) as datetime64[us]. Accepts the
    same inputs as time.time_to_decimal_year."""
    if isinstance(date, str):
        try:
            date = datetime.fromisoformat(date)
        except ValueError:
            raise ValueError(f"Invalid time format: {date}")
    if isinstance(date, (np.datetime64, datetime)):
        return np.datetime64(date, "us")
    if isinstance(date, (int, float, np.integer, np.floating)) and not (
            isinstance(date, (bool, np.bool_))):
        # days since 1970-01-01
        return (np.datetime64("1970-01-01T00:00:00", "us")
                + np.timedelta64(int(round(float(date) * 86400e6)), "us"))
    raise TypeError("Expected time to be datetime, string, or numeric,"
                    f" got {type(date)} instead.")


def _decimal_years(times: np.ndarray) -> np.ndarray:
    """datetime64 array -> decimal years (same definition as
    time.time_to_decimal_year: year + elapsed / length of that year)."""
    times = times.astype("datetime64[us]")
    year = times.astype("datetime64[Y]")
    start = year.astype("datetime64[us]")
    length = (year + np.timedelta64(1, "Y")).astype("datetime64[us]") - start
    return (year.astype(int) + 1970) + (times - start) / length


def _declination_at(lat, lon, decimal_year, altitude, model):
    """Declination (degrees) at one decimal year; see
    get_declination_point. `altitude` in metres."""
    if model == 'auto':
        coefficients_file = choose_wmm_model(decimal_year)
        allow_date_outside_lifespan = False
    else:
        allow_date_outside_lifespan = True
        coefficients_file = _MODEL_FILES.get(str(model))
        if coefficients_file is None:
            raise ValueError(
                f'Invalid option model={model} for choice of World Magnetic '
                'Model version. Options are ["2010", "2015", "2020", "2025"].'
            )
    return float(_geomag(coefficients_file).calculate(
        glat=lat, glon=lon, alt=altitude / 1000.0,  # pygeomag wants km
        time=decimal_year,
        allow_date_outside_lifespan=allow_date_outside_lifespan).d)


_MODEL_FILES = {
    '2010': 'wmm/WMM_2010.COF',
    '2015': 'wmm/WMM_2015.COF',
    '2020': 'wmm/WMM_2020.COF',
    '2025': 'wmm/WMM_2025.COF',
}


def get_declination(
    lat: float,
    lon: float,
    dates: Iterable[datetime | np.datetime64 | str | int | float],
    altitude: float = 0.0,
    model: str = 'auto',  # Which WMM model to use
    daily: bool = True,
) -> np.ndarray:

    """
    Get the magnetic declination for multiple date points at a specific
    location using the World Magnetic Model (WMM).

    Parameters:
    ----------
    lat : float
        Latitude in decimal degrees.
    lon : float
        Longitude in decimal degrees.
    dates : Iterable[datetime | np.datetime64 | str | int | float]
        An iterable of time inputs, where each element can be:
        - A `datetime.datetime` object.
        - A `numpy.datetime64` object.
        - A string in ISO format (e.g., '2021-02-01' or '2021-02-01 23:30:00').
        - A numeric value representing days since '1970-01-01'.
        Times are taken to be UTC.
    altitude : float, optional
        Altitude in meters above sea level (default is 0.0).
    model: str
        Which WMM version to use.
        Options: ['2010', '2015', '2020', '2025'], otherwise determined based on the
        data.
    daily : bool, default True
        If True, the declination is calculated once per day (at 00:00 UTC
        of each day with data, and at the last time) and linearly
        interpolated to the given times. The declination changes by much
        less than 0.01 degrees per day, so this gives the same values to
        well within the model's accuracy, and is much faster for long,
        densely sampled records. If False, it is calculated for every
        time.

    Returns:
    -------
    np.ndarray
        An array of magnetic declinations in degrees corresponding to each
        date.

    Notes:
    -----
    - The World Magnetic Model (WMM) data covers the years 2010-2030.
    - Automatic determination of WMM model can cause discontinuities
      crossing over from 2014 to 2015 or 2019 to 2020. These are typically
      small (<1 deg). With daily=True, the jump is spread over the last
      day before the new model starts.
    """
    times = np.array([_as_datetime64(date) for date in dates],
                     dtype="datetime64[us]")
    if times.size == 0:
        return np.array([], dtype=float)
    decimal_years = _decimal_years(times)

    if not daily:
        return np.array([_declination_at(lat, lon, dy, altitude, model)
                         for dy in decimal_years])

    # Nodes: 00:00 of each day with data, plus the last time (so no node is
    # outside the range of the data, e.g. past the end of a model)
    nodes = np.unique(np.concatenate([
        times.astype("datetime64[D]").astype("datetime64[us]"),
        [times.max()]]))
    node_years = _decimal_years(nodes)
    node_decl = np.array([_declination_at(lat, lon, dy, altitude, model)
                          for dy in node_years])
    # Interpolate on the circle (declination can be near +-180 deg close to
    # the magnetic poles), then wrap back to [-180, 180)
    node_decl = np.unwrap(node_decl, period=360)
    decl = np.interp(decimal_years, node_years, node_decl)
    return (decl + 180.0) % 360.0 - 180.0


def get_declination_point(
    lat: float,
    lon: float,
    date: datetime | np.datetime64 | str | int | float,
    altitude: float = 0.0,
    model: str = 'auto',  # Which WMM model to use
) -> float:
    """
    Get the magnetic declination at a specific point in space and time using
    the World Magnetic Model (WMM).

    Parameters:
    ----------
    lat : float
        Latitude in decimal degrees.
    lon : float
        Longitude in decimal degrees.
    date: Iterable[datetime | np.datetime64 | str | int | float]
            The time input which can be:
            - A `datetime.datetime` object.
            - A `numpy.datetime64` object.
            - A string in ISO format (e.g., '2021-02-01' or
              '2021-02-01 23:30:00').
            - A numeric value representing days since '1970-01-01'.
    altitude : float, optional
        Altitude in meters above sea level (default is 0.0).
    model: str
        Which WMM version to use.
        Options: ['2010', '2015', '2020', '2025'], otherwise determined
        based on the data.

    Returns:
    -------
    float
        The magnetic declination in degrees.

    Notes:
    -----
    - The World Magnetic Model (WMM) data covers the years 2010-2030.
    - Specifying the *model* will allow computing the declination outside
      the model range (e.g 2010-2015 for WMM2010). Be careful!
    """


    # Convert time to decimal year if necessary
    decimal_year = time.time_to_decimal_year(date)

    # Calculate the magnetic declination
    return _declination_at(lat, lon, decimal_year, altitude, model)


def choose_wmm_model(decimal_year: float) -> str:
    """
    Choose the appropriate WMM model file based on the decimal year.

    Parameters:
    ----------
    decimal_year : float
        The decimal year for which the WMM model file is needed.

    Returns:
    -------
    str
        The file path to the WMM coefficients file.

    Raises:
    ------
    ValueError
        If the decimal year is outside the range 2010-2025.
    """
    if 2010 <= decimal_year < 2015:
        wmm_coefficients_file = 'wmm/WMM_2010.COF'
    elif 2015 <= decimal_year < 2020:
        wmm_coefficients_file = 'wmm/WMM_2015.COF'
    elif 2020 <= decimal_year < 2025:
        wmm_coefficients_file = 'wmm/WMM_2020.COF'
    elif 2025 <= decimal_year < 2030:
        wmm_coefficients_file = 'wmm/WMM_2025.COF'
    else:
        raise ValueError(
            f'Year {decimal_year:.0f} is out of range. The World Magnetic '
            'Model only covers 2010-2030. Please obtain declination values '
            'elsewhere.'
        )

    return wmm_coefficients_file
