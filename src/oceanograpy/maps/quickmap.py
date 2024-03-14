import cartopy.crs as ccrs
import cartopy.feature as cfeature 
import matplotlib.pyplot as plt

def quick_map_stere(lon_ctr, lat_ctr, height = 1000, width = 1000, 
                    fig = None, coast_resolution = '50m', ):
    '''
    Quick map osing stereographic projection around lat_ctr, lon_ctr.

    height, width: 
      Dimensions of map extent in km
    '''
    projection = ccrs.Stereographic(
        central_latitude=lat_ctr,
        central_longitude=lon_ctr,
        scale_factor=None, globe=None)

    if fig==None:
        fig = plt.figure(figsize = (4, 3))
        return_fig = True

    ax = plt.axes(projection=projection)

    # Map extent
    ax.set_extent((-width/2*1e3, width/2*1e3, -height/2*1e3, height/2*1e3),
        crs=projection)

    # Land
    land_50m = cfeature.NaturalEarthFeature(
        'physical', 'land', coast_resolution,
        edgecolor='k', linewidth = 1,
        facecolor=cfeature.COLORS['land'])

    ax.add_feature(land_50m)

    if return_fig:
        return fig, ax 
    else:
        return ax
