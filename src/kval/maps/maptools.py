'''
KVAL.MAPS.MAPTOOLS

Utility functions for geospatial plotting and data visualization on maps. 
'''

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pyproj import Transformer
from matplotlib.quiver import Quiver

def quiver(
    ax: object,
    lon: np.ndarray | list[float],
    lat: np.ndarray | list[float],
    u_east: np.ndarray | list[float],
    v_north: np.ndarray | list[float],
    scale: float = 100,
    pole_buffer: float = 0.01,
    stride: int = 1,
    **quiver_kwargs,
) -> Quiver | None:
    """
    Reprojects and plots a vector field (u, v in east/north directions) on a Cartopy map,
    irrespective of projection.

    Args:
        ax: A Cartopy GeoAxes instance where vectors will be plotted.
        lon: Longitudes in degrees (1D or 2D array-like).
        lat: Latitudes in degrees (1D or 2D array-like).
        u_east: Eastward vector components (same shape as lon/lat).
        v_north: Northward vector components (same shape as lon/lat).
        scale: Scaling factor for arrow length passed to quiver (default 100).
        pole_buffer: Degrees from the poles within which vectors are masked (default 0.01).
        stride: Subsampling stride to reduce vector density (default 1, no subsampling).
        **quiver_kwargs: Additional keyword arguments passed to matplotlib's quiver.

    Returns:
        A matplotlib.quiver.Quiver instance if vectors plotted, else None.
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    u = np.asarray(u_east)
    v = np.asarray(v_east)

    # Normalize longitudes to [-180, 180]
    lon = (lon + 180) % 360 - 180

    # Apply stride subsampling if stride > 1
    if stride > 1:
        if lon.ndim == 2 and lat.ndim == 2 and u.ndim == 2 and v.ndim == 2:
            ny, nx = lon.shape
            lon = lon[::stride, ::stride].flatten()
            lat = lat[::stride, ::stride].flatten()
            u = u[::stride, ::stride].flatten()
            v = v[::stride, ::stride].flatten()
        else:
            # fallback: 1D stride
            lon = lon.flatten()[::stride]
            lat = lat.flatten()[::stride]
            u = u.flatten()[::stride]
            v = v.flatten()[::stride]
    else:
        lon = lon.flatten()
        lat = lat.flatten()
        u = u.flatten()
        v = v.flatten()
    
    valid = (
        np.isfinite(lon) &
        np.isfinite(lat) &
        np.isfinite(u) &
        np.isfinite(v) &
        (np.abs(lat) < 90 - pole_buffer)
    )
    lon = lon[valid]
    lat = lat[valid]
    u = u[valid]
    v = v[valid]

    if len(lon) == 0:
        print("No valid vectors to plot.")
        return None

    proj = ax.projection
    transformer = Transformer.from_crs("epsg:4326", proj, always_xy=True)

    x, y = transformer.transform(lon, lat)

    d = 1e-5  # numerical derivative step in degrees

    x_east, y_east = transformer.transform(lon + d, lat)
    x_north, y_north = transformer.transform(lon, lat + d)

    dx_dlon = (x_east - x) / d
    dy_dlon = (y_east - y) / d
    dx_dlat = (x_north - x) / d
    dy_dlat = (y_north - y) / d

    u_proj = u * dx_dlon + v * dx_dlat
    v_proj = u * dy_dlon + v * dy_dlat

    q = ax.quiver(x, y, u_proj, v_proj, transform=proj, scale=scale, **quiver_kwargs)
    return q
