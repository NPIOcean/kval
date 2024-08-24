'''
KVAL.UTIL.MAGDEC

Wrapper for the Python API for British Geological Survey magnetic field
calculator (https://github.com/filips123/MagneticFieldCalculator/).
'''

from typing import Union, Iterable
from datetime import datetime
from pygeomag import GeoMag  #
from kval.util import time
import numpy as np


from typing import Iterable, Union
from datetime import datetime
import numpy as np

def get_declination(
    lat: float,
    lon: float,
    dates: Iterable[Union[datetime, np.datetime64, str, Union[int, float]]],
    altitude: float = 0.0
) -> np.ndarray:
    """
    Get the magnetic declination for multiple date points at a specific location using
    the World Magnetic Model (WMM).

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

    Returns:
    -------
    np.ndarray
        An array of magnetic declinations in degrees corresponding to each date.

    Notes:
    -----
    - The World Magnetic Model (WMM) data covers the years 2010-2025.
    """
    # Initialize an empty list to store declinations
    declinations = []

    # Calculate declination for each date and append to the list
    for date in dates:
        declination = get_declination_point(lat, lon, date, altitude)
        declinations.append(declination)

    # Convert the list of declinations to a NumPy array
    return np.array(declinations)


def get_declination_point(
    lat: float,
    lon: float,
    date: Union[datetime, np.datetime64, str, Union[int, float]],
    altitude: float = 0.0
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

    Returns:
    -------
    float
        The magnetic declination in degrees.

    Notes:
    -----
    - The World Magnetic Model (WMM) data covers the years 2010-2025.
    """

    # Convert time to decimal year if necessary
    decimal_year = time.time_to_decimal_year(date)

    # Choose the appropriate WMM model file for the given decimal year
    wmm_coefficients_file = choose_wmm_model(decimal_year)

    # Initialize the GeoMag object with the coefficients file
    geo_mag = GeoMag(coefficients_file=wmm_coefficients_file)

    # Calculate the magnetic declination
    declination = geo_mag.calculate(
        glat=lat, glon=lon, alt=altitude, time=decimal_year
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
        wmm_coefficients_file = 'wmm/WMM.COF'
    else:
        raise ValueError(
            f'Year {decimal_year:.0f} is out of range. The World Magnetic Model '
            'only covers 2010-2025. Please obtain declination values elsewhere.'
        )

    return wmm_coefficients_file
