'''
KVAL.MAPS.MAPTOOLS

Utility functions for geospatial plotting and data visualization on maps. 
'''
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pyproj import Transformer, Geod
from matplotlib.quiver import Quiver

def quiver_proj(
    ax: object,
    lon: np.ndarray | list[float],
    lat: np.ndarray | list[float],
    u_east: np.ndarray | list[float],
    v_north: np.ndarray | list[float],
    scale: str | float = 'auto',
    pole_buffer: float = 0.01,
    stride: int = 1,
    step_km: float = 0.1,
    scale_length_by_basis: bool = True,  
    **quiver_kwargs,
) -> Quiver | None:
    """
    Reprojects and plots a vector field (u, v in east/north directions) on a Cartopy map,
    handling vector rotation and scaling due to map projection distortions.

    Args:
        ax: A Cartopy GeoAxes instance where vectors will be plotted.
        lon: Longitudes in degrees (1D or 2D array-like).
        lat: Latitudes in degrees (1D or 2D array-like).
        u_east: Eastward vector components (same shape as lon/lat).
        v_north: Northward vector components (same shape as lon/lat).
        scale: Scaling factor for arrow length passed to matplotlib.quiver (default 100).
        pole_buffer: Degrees from the poles within which vectors are masked (default 0.01).
        stride: Subsampling stride to reduce vector density (default 1, no subsampling).
        step_km: Distance (km) used to estimate projection-space basis vectors (default 1 km).
        scale_length_by_basis: If True (default), vector lengths are scaled by local projected basis vectors,
        causing length distortion near map edges due to projection. If False, lengths are kept consistent
        with original east/north magnitudes, preserving uniform arrow lengths.
        **quiver_kwargs: Additional keyword arguments passed to matplotlib's quiver.

    Note:
        Autoscaling behaviour doesnt work well not revisit at some point.

    Returns:
        A matplotlib.quiver.Quiver instance if vectors were plotted, else None.
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    u = np.asarray(u_east)
    v = np.asarray(v_north)

    # Normalize longitudes to [-180, 180]
    lon = (lon + 180) % 360 - 180

    # Subsample if needed
    if stride > 1:
        if lon.ndim == 2:
            lon = lon[::stride, ::stride].flatten()
            lat = lat[::stride, ::stride].flatten()
            u = u[::stride, ::stride].flatten()
            v = v[::stride, ::stride].flatten()
        else:
            lon = lon.flatten()[::stride]
            lat = lat.flatten()[::stride]
            u = u.flatten()[::stride]
            v = v.flatten()[::stride]
    else:
        lon = lon.flatten()
        lat = lat.flatten()
        u = u.flatten()
        v = v.flatten()

    # Mask invalid or polar-edge points
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

    # Set up projection transformer
    proj = ax.projection
    transformer = Transformer.from_crs("EPSG:4326", proj, always_xy=True)

    # Project point locations
    x, y = transformer.transform(lon, lat)

    # Compute local basis using small steps east and north
    geod = Geod(ellps="WGS84")
    step_m = step_km * 1000

    # Create azimuth and distance arrays matching lon/lat shape
    az_east = np.full_like(lon, 90.0)
    az_north = np.full_like(lon, 0.0)
    dist = np.full_like(lon, step_m)

    # Compute coordinates `ellps` km to the east and north using forward azimuths
    lon_east, lat_east, _ = geod.fwd(lon, lat, az=az_east, dist=dist)
    lon_north, lat_north, _ = geod.fwd(lon, lat, az=az_north, dist=dist)
    # Project those points
    x_east, y_east = transformer.transform(lon_east, lat_east)
    x_north, y_north = transformer.transform(lon_north, lat_north)

    # Projected center points already computed: x, y

    # Check for valid projected points and basis vectors (exclude nan or inf)
    valid_proj = (
        np.isfinite(x_east) & np.isfinite(y_east) &
        np.isfinite(x_north) & np.isfinite(y_north) &
        np.isfinite(x) & np.isfinite(y)
    )

    # Filter all arrays by valid_proj
    x = x[valid_proj]
    y = y[valid_proj]
    u = u[valid_proj]
    v = v[valid_proj]

    x_east = x_east[valid_proj]
    y_east = y_east[valid_proj]
    x_north = x_north[valid_proj]
    y_north = y_north[valid_proj]

    # Local projected basis vectors
    dx_east = x_east - x
    dy_east = y_east - y
    dx_north = x_north - x
    dy_north = y_north - y

    # Convert u, v to projected space (direction + length scaling)
    u_proj_raw = u * dx_east + v * dx_north
    v_proj_raw = u * dy_east + v * dy_north

    if scale_length_by_basis:
        # Use current behavior (length distorted by projection basis)
        u_proj, v_proj = u_proj_raw, v_proj_raw
    else:
        # Normalize to unit vectors in projected space to preserve direction
        norm_proj = np.sqrt(u_proj_raw**2 + v_proj_raw**2)
        norm_proj[norm_proj == 0] = 1  # prevent division by zero

        # True vector magnitude in physical space (u, v are east/north)
        mag_true = np.sqrt(u**2 + v**2)

        # Rescale projected unit vectors by true magnitude (length independent of projection distortion)
        u_proj = (u_proj_raw / norm_proj) * mag_true
        v_proj = (v_proj_raw / norm_proj) * mag_true

    if scale =='auto':
        dx = x.max() - x.min()
        dy = y.max() - y.min()

        diag = np.hypot(dx, dy)

        if diag < 1e-6:
            # Avoid divide-by-zero or meaningless scale in degenerate case
            scale = None
        else:
            desired_arrow_length = diag / 30  # Ish desired visual density
            avg_vec_length = np.nanmean(np.hypot(u_proj, v_proj))

            if not np.isfinite(avg_vec_length) or avg_vec_length == 0:
                scale = None
            else:
                scale = desired_arrow_length / avg_vec_length
                scale = min(max(scale, 1e-10), 1e10)  # Clamp to avoid crazy extremes

    q = ax.quiver(x, y, u_proj, v_proj, transform=proj, scale=scale, **quiver_kwargs)

    return q


