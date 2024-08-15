import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, Tuple, Union

def quick_map_stere(
    lon_ctr: float,
    lat_ctr: float,
    height: int = 1000,
    width: int = 1000,
    fig: Optional[Figure] = None,
    coast_resolution: str = '50m'
) -> Union[Tuple[Figure, Axes], Axes]:
    """
    Generate a quick map using a stereographic projection centered 
    around lat_ctr and lon_ctr.

    Parameters:
        lon_ctr (float): Central longitude for the map.
        lat_ctr (float): Central latitude for the map.
        height (int): Height of the map extent in km. Default is 1000 km.
        width (int): Width of the map extent in km. Default is 1000 km.
        fig (Optional[Figure]): Matplotlib figure object. If None, a new 
                                figure is created.
        coast_resolution (str): Resolution of coastlines to be used 
                                ('10m', '50m', '110m'). Default is '50m'.

    Returns:
        Union[Tuple[Figure, Axes], Axes]: The map figure and axes objects.
    """
    projection = ccrs.Stereographic(central_latitude=lat_ctr,
                                    central_longitude=lon_ctr)

    if fig is None:
        fig = plt.figure(figsize=(4, 3))
        return_fig = True
    else:
        return_fig = False

    ax = plt.axes(projection=projection)

    # Set map extent
    ax.set_extent([-width / 2 * 1e3, width / 2 * 1e3,
                   -height / 2 * 1e3, height / 2 * 1e3],
                  crs=projection)

    # Add land feature
    land_50m = cfeature.NaturalEarthFeature(
        'physical', 'land', coast_resolution,
        edgecolor='k', linewidth=1,
        facecolor=cfeature.COLORS['land'])
    
    ax.add_feature(land_50m)

    if return_fig:
        return fig, ax
    else:
        return ax
