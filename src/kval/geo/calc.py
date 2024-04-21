'''
Various short functions for geographical calculations.
'''

import numpy as np



def great_circle_distance(lon0, lat0, lon1, lat1, radius=6378e3):
    """
    Calculate the great circle distance between two points on the Earth's surface.

    Parameters:
        lon0 (float or array-like): Longitude of the first point in degrees.
        lat0 (float or array-like): Latitude of the first point in degrees.
        lon1 (float or array-like): Longitude of the second point in degrees.
        lat1 (float or array-like): Latitude of the second point in degrees.
        rad (float, optional): Earth's radius in meters. Default is 6378e3 (mean radius).

    Returns:
        float or array-like: The great circle distance between the two points in meters.

    Notes:
        - lon0, lat0, lon1, lat1 must be broadcastable sequences of longitudes and latitudes in degrees.
        - Distance is returned in meters, or in whatever units are used for the radius.
        - This function largely borrows from pycurrents: 
          https://currents.soest.hawaii.edu/hgstage/pycurrents/file/tip/pycurrents/data/navcalc.py
    """
    # Convert degrees to radians
    lon0 = np.deg2rad(lon0)
    lat0 = np.deg2rad(lat0)
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)

    # Calculate the great circle distance from the Havrsine formula
    dist = radius * np.arccos(
        np.cos(lat0) * np.cos(lat1) * np.cos(lon0 - lon1) 
        + np.sin(lat0) * np.sin(lat1))

    return dist

def closest_coord(lon, lat, lon0, lat0):
    '''
    Use the great circle distance to compute the distance between points and returns 
    the index of the point in (lon, lat) with the shortest distance from (lon0, lat0).

    Useful for e.g. finding the cell index of a gridded product (with grid lat, lon) 
    closest to a particular grid point (lat0, lon0). 

    Parameters:
        lon (array-like): Array of longitudes.
        lat (array-like): Array of latitudes.
        lon0 (float): Longitude of the reference point in degrees.
        lat0 (float): Latitude of the reference point in degrees.

    Returns:
        tuple: Index of the closest point in the (lon, lat) grid.

    Notes:
        - lon and lat are expected to be arrays representing grid coordinates.
        - lon and lat arrays can be independent axes defining the grid, 
          or part of a spatial grid defined by other coordinates.
        - The function uses the great circle distance with Earth radius 6378 km 
          to find the closest point.
        - If lon and lat are independent axes defining the grid 
          (e.g., lon(y), lat(x)), closest_index returns (ind_x, ind_y) such that 
          the closest point is:    lon[ind_x], lat[ind_y].
        - If lat and lon are on a spatial grid defined by other coordinates 
          (e.g., lon(x, y), lat(x, y)), closest_index returns (ind_x, ind_y) 
          such that the closest point is:    lon[ind_x, ind_y], lat[ind_x, ind_y].
    '''
    
    # Calculate great circle distance
    gc_dist = great_circle_distance(lon0, lat0, lon, lat)
    
    # Find the index of the point with the shortest distance
    closest_index = np.unravel_index(gc_dist.argmin(), gc_dist.shape)

    return closest_index