"""
Various short functions for geographical calculations.
"""

import numpy as np
from typing import Union, Tuple, Sequence


def great_circle_distance(
    lon0: Union[float, Sequence[float]],
    lat0: Union[float, Sequence[float]],
    lon1: Union[float, Sequence[float]],
    lat1: Union[float, Sequence[float]],
    earth_radius: float = 6378e3,
) -> Union[float, Sequence[float]]:
    """
    Calculate the great circle distance between two points on the
    Earth's surface.

    Parameters:
        lon0 (float or array-like): Longitude of the first point in degrees.
        lat0 (float or array-like): Latitude of the first point in degrees.
        lon1 (float or array-like): Longitude of the second point in degrees.
        lat1 (float or array-like): Latitude of the second point in degrees.
        earth_radius (float, optional): Earth's radius in meters.
                                        Default is 6378e3 (mean radius).

    Returns:
        float or array-like: The great circle distance between the two points
                             in meters.

    Notes:
        - lon0, lat0, lon1, lat1 must be broadcastable sequences of longitudes
          and latitudes in degrees.
        - Distance is returned in meters, or in whatever units are used for the
          earth radius.
        - This function largely borrows from pycurrents:
          https://currents.soest.hawaii.edu/hgstage/pycurrents/file/tip/pycurrents/data/navcalc.py
    """
    # Convert degrees to radians
    lon0 = np.deg2rad(lon0)
    lat0 = np.deg2rad(lat0)
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)

    # Calculate the great circle distance from the Haversine formula
    dist = earth_radius * np.arccos(
        np.cos(lat0) * np.cos(lat1) * np.cos(lon0 - lon1)
        + np.sin(lat0) * np.sin(lat1)
    )

    return dist


def closest_coord(
    lon: Union[float, Sequence[float]],
    lat: Union[float, Sequence[float]],
    lon0: float,
    lat0: float,
) -> Tuple[int, int]:
    """
    Use the great circle distance to compute the distance between points and
    returns the index of the point in (lon, lat) with the shortest distance
    from (lon0, lat0).

    Useful for e.g. finding the cell index of a gridded product (with grid lat,
    lon) closest to a particular grid point (lat0, lon0).

    Parameters:
        lon (array-like): Array of longitudes.
        lat (array-like): Array of latitudes.
        lon0 (float): Longitude of the reference point in degrees.
        lat0 (float): Latitude of the reference point in degrees.

    Returns:
        tuple: Index of the closest point in the (lon, lat) grid.

    Notes:
        - lon and lat are expected to be arrays representing grid coordinates.
        - lon and lat arrays can be independent axes defining the grid, or part
          of a spatial grid defined by other coordinates.
        - The function uses the great circle distance with Earth radius 6378 km
          to find the closest point.
        - If lon and lat are independent axes defining the grid (e.g., lon(y),
          lat(x)), closest_index returns (ind_x, ind_y) such that the closest
          point is:    lon[ind_x], lat[ind_y].
        - If lat and lon are on a spatial grid defined by other coordinates
          (e.g., lon(x, y), lat(x, y)), closest_index returns (ind_x, ind_y)
          such that the closest point is:    lon[ind_x, ind_y], lat[ind_x,
          ind_y].
    """

    # For 1D case: lon[x], lat[y]
    if lon.ndim == 1 and lat.ndim == 1:
        closest_index = (
            np.abs(lon - lon0).argmin(),
            np.abs(lat - lat0).argmin(),
        )

    # For 2D case: lon[x, y], lat[x, y]
    elif lon.ndim == 2 and lat.ndim == 2:
        # Calculate great circle distance
        gc_dist = great_circle_distance(lon0, lat0, lon, lat)

        # Find the index of the point with the shortest distance
        closest_index = np.unravel_index(gc_dist.argmin(), gc_dist.shape)

    else:
        raise ValueError(
            "Unexpected dimensions of lon and lat arrays. "
            "They should both have the same shape (1D or 2D)."
        )

    return closest_index
