'''
KVAL.UTIL.MAGDEC

Wrapper for the Python API for British Geological Survey magnetic field
calculator (https://github.com/filips123/MagneticFieldCalculator/).
'''

from typing import Union, Iterable
from datetime import datetime
from pygeomag import GeoMag
from kval.util import time
import numpy as np


def get_declination(
    lat: float,
    lon: float,
    dates: Iterable[Union[datetime, np.datetime64, str, Union[int, float]]],
    altitude: float = 0.0,
    model: str = 'auto',  # Which WMM model to use
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
    dates : Iterable[Union[datetime, np.datetime64, str, Union[int, float]]]
        An iterable of time inputs, where each element can be:
        - A `datetime.datetime` object.
        - A `numpy.datetime64` object.
        - A string in ISO format (e.g., '2021-02-01' or '2021-02-01 23:30:00').
        - A numeric value representing days since '1970-01-01'.
    altitude : float, optional
        Altitude in meters above sea level (default is 0.0).
    model: str
        Which WMM version to use.
        Options: ['2010', '2015', '2020'], otherwise determined based on the
        data.

    Returns:
    -------
    np.ndarray
        An array of magnetic declinations in degrees corresponding to each
        date.

    Notes:
    -----
    - The World Magnetic Model (WMM) data covers the years 2010-2025.
    - Automatic determination of WMM model can cause discontinuities
      crossing over from 2014 to 2015 or 2019 to 2020. These are typically
      small (<1 deg).
    """
    # Initialize an empty list to store declinations
    declinations = []

    # Calculate declination for each date and append to the list
    for date in dates:
        declination = get_declination_point(lat, lon, date, altitude, model)
        declinations.append(declination)

    # Convert the list of declinations to a NumPy array
    return np.array(declinations)


def get_declination_point(
    lat: float,
    lon: float,
    date: Union[datetime, np.datetime64, str, Union[int, float]],
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
    date : (Union[datetime.datetime, np.datetime64, str, int, float])
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
        Options: ['2010', '2015', '2020'], otherwise determined based on the
        data.

    Returns:
    -------
    float
        The magnetic declination in degrees.

    Notes:
    -----
    - The World Magnetic Model (WMM) data covers the years 2010-2025.
    - Specifying the *model* will allow computing the declination outside
      the model range (e.g 2010-2015 for WMM2010). Be careful!
    """


    # Convert time to decimal year if necessary
    decimal_year = time.time_to_decimal_year(date)

    # Choose the appropriate WMM model file for the given decimal year
    if model == 'auto':
        wmm_coefficients_file = choose_wmm_model(decimal_year)
        allow_date_outside_lifespan = False
    else:
        allow_date_outside_lifespan = True
        if model == '2010':
            wmm_coefficients_file = 'wmm/WMM_2010.COF'
        elif model == '2015':
            wmm_coefficients_file = 'wmm/WMM_2015.COF'
        elif model == '2020':
            wmm_coefficients_file = 'wmm/WMM_2020.COF'
        else:
            raise ValueError(
                f'Invalid option model={model} for choice of World Magnetic '
                'Model version. Options are ["2010", "2015", "2020"].'
            )

    # Initialize the GeoMag object with the coefficients file
    geo_mag = GeoMag(coefficients_file=wmm_coefficients_file)

    # Calculate the magnetic declination
    declination = geo_mag.calculate(
        glat=lat, glon=lon, alt=altitude, time=decimal_year,
        allow_date_outside_lifespan=allow_date_outside_lifespan
    ).d

    return declination


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
