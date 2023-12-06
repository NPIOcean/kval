'''
Functions for visualizing ctd data.

These are meant to be run in a Jupyter Lab notebook using the 
%matplotlib widget backend
'''

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from oceanograpy.util import time
from oceanograpy.maps import quickmap

def inspect_profiles(d):
    """
    Function to interactively inspect profiles in an xarray dataset.

    Parameters:
    - d (xr.Dataset): The xarray dataset containing variables 'PRES', 'STATION', and other profile variables.

    """

    # Function to create a profile plot
    def plot_profile(station_index, variable):
        fig, ax = plt.subplots()

        # Plot all profiles in black in the background
        for nn, station in enumerate(d['STATION']):
            if nn != station_index:  # Skip the selected profile
                profile =  d[variable].where(d['STATION'] == station, drop=True).squeeze()
                ax.plot(profile, d['PRES'], color='tab:blue', lw=0.5, alpha=0.4)

        # Plot the selected profile in color
        profile = d[variable].isel(TIME=station_index)

        ax.plot(profile, d['PRES'], alpha=0.8, lw=0.7, color='k')
        ax.plot(profile, d['PRES'], '.', ms=2, alpha=1, color='tab:orange')

        station_time_string = time.convert_timenum_to_datetime(profile.TIME, d.TIME.units)
        ax.set_title(f'Station: {d["STATION"].values[station_index]}, {station_time_string}')
        ax.set_xlabel(f'{variable} [{d[variable].units}]')
        ax.set_ylabel('PRES')
        ax.invert_yaxis()
        ax.grid()
        fig.canvas.header_visible = False  # Hide the figure header
        plt.tight_layout()

        plt.show()

    # Get the descriptions for the slider
    station_descriptions = [str(station) for station in d['STATION'].values]

    # Create the slider for selecting a station
    station_index_slider = widgets.IntSlider(
        min=0, max=len(d['STATION']) - 1, step=1, value=0, description='Profile #:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='600px')  # Set the width of the slider
    )

    # Get the profile variables
    def get_profile_variables(d):
        profile_variables = [varnm for varnm in d.data_vars if 'PRES' in d[varnm].dims and 'TIME' in d[varnm].dims]
        return profile_variables

    profile_vars = get_profile_variables(d)

    # Create the dropdown for selecting a variable
    variable_dropdown = widgets.Dropdown(
        options=profile_vars,
        value=profile_vars[0],
        description='Variable:'
    )

    # Create the dropdown for selecting a station
    station_dropdown = widgets.Dropdown(
        options=station_descriptions,
        value=station_descriptions[0],
        description='Station:'
    )

    # Update slider value when dropdown changes
    def update_slider_from_dropdown(change):
        station_index = station_descriptions.index(change.new)
        station_index_slider.value = station_index

    # Update dropdown value when slider changes
    def update_dropdown_from_slider(change):
        station_description = str(d['STATION'].values[change.new])
        station_dropdown.value = station_description

    # Link slider and dropdown
    station_dropdown.observe(update_slider_from_dropdown, names='value')
    station_index_slider.observe(update_dropdown_from_slider, names='value')

    # Use interactive_output to create interactive controls
    output = widgets.interactive_output(
        plot_profile, {'station_index': station_index_slider, 'variable': variable_dropdown}
    )

    # Display the widgets in a vertically stacked layout
    display(widgets.VBox([
        widgets.HBox([station_index_slider]),
        widgets.HBox([variable_dropdown, station_dropdown]),
        output
    ]))






def map(D, height=1000, width=1000, return_fig_ax=False, coast_resolution='50m', figsize=None):
    '''
    Quick map of cruise
    '''
    # These would maybe be useful for auto-scaling of the map..
    lat_span = float(D.LATITUDE.max() - D.LATITUDE.min())
    lon_span = float(D.LONGITUDE.max() - D.LONGITUDE.min())
    lat_ctr = float(0.5 * (D.LATITUDE.max() + D.LATITUDE.min()))
    lon_ctr = float(0.5 * (D.LONGITUDE.max() + D.LONGITUDE.min()))

    fig, ax = quickmap.quick_map_stere(lon_ctr, lat_ctr, height=height,
                                       width=width,
                                       coast_resolution=coast_resolution,)

    fig.canvas.header_visible = False  # Hide the figure header
    
    ax.plot(D.LONGITUDE, D.LATITUDE, '-k', transform=ccrs.PlateCarree(), alpha=0.5)
    ax.plot(D.LONGITUDE, D.LATITUDE, 'or', transform=ccrs.PlateCarree())

    plt.tight_layout()
    
    if figsize:
        fig.set_size_inches(figsize)
    else:
        figsize = fig.get_size_inches()

    # Create a button to minimize the plot
    minimize_button = widgets.Button(description="Minimize")

    def minimize_plot(_):
        # Resize the figure to 2x
        fig.set_size_inches(0.1, 0.1)
        fig.canvas.draw()

    minimize_button.on_click(minimize_plot)

    # Create a button to restore full size
    org_size_button = widgets.Button(description="Original Size")

    def org_size_plot(_):
        # Resize the figure to its original size
        fig.set_size_inches(figsize)
        fig.canvas.draw()

    # Create a button to restore full size
    full_size_button = widgets.Button(description="Larger")

    def full_size_plot(_):
        # Resize the figure to its original size
        fig.set_size_inches(fig.get_size_inches()*2)
        fig.canvas.draw()

    minimize_button.on_click(minimize_plot)
    org_size_button.on_click(org_size_plot)
    full_size_button.on_click(full_size_plot)

    # Create a static text widget
    static_text = widgets.HTML(value='<p>Use the menu on the left of the figure to zoom/move around/save</p>')

    # Display both buttons and text with decreased vertical spacing
    display(
        widgets.HBox([minimize_button, org_size_button, full_size_button, static_text], layout=widgets.Layout(margin='0 0 5px 0', align_items='center')))
    
    if return_fig_ax:
        return fig, ax
