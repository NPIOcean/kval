'''
Functions for visualizing ctd data.

These are meant to be run in a Jupyter Lab notebook using the 
%matplotlib widget backend.

These functions are rather complex and clunky because they are tuned for the visual 
input and menu functionality we want for the specific application. I am not too worried
about that.
'''

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from oceanograpy.util import time
from oceanograpy.data.ship_ctd_tools import _ctd_tools 
from oceanograpy.maps import quickmap
from matplotlib.ticker import MaxNLocator
import cmocean
import numpy as np

def inspect_profiles(d):

    """
    Interactively inspect profiles in an xarray dataset.

    Parameters:
    - d (xr.Dataset): The xarray dataset containing variables 'PRES', 'STATION', and other profile variables.

    This function creates an interactive plot allowing the user to explore profiles within the given xarray dataset.
    It displays a slider to choose a profile by its index, a dropdown menu to select a variable for visualization, and
    another dropdown to pick a specific station. The selected profile is highlighted in color, while others are shown
    in the background.

    Parameters:
    - d (xr.Dataset): The xarray dataset containing variables 'PRES', 'STATION', and other profile variables.

    Examples:
    ```python
    inspect_profiles(my_dataset)
    ```

    Note: This function utilizes Matplotlib for plotting and ipywidgets for interactive controls.
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
        layout=widgets.Layout(width='500px')  # Set the width of the slider
    )

    profile_vars = _ctd_tools._get_profile_variables(d)

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


    # Create a button to close the plot
    close_button = widgets.Button(description="Close")

    def close_plot(_):
        # Resize the figure to 0 and close it
        fig = plt.gcf()
        fig.set_size_inches(0, 0)
        widgets_collected.close()
        plt.close(fig)

    close_button.on_click(close_plot)

    # Display the widgets in a vertically stacked layout
    widgets_collected = widgets.VBox([
        widgets.HBox([station_index_slider, close_button]),
        widgets.HBox([variable_dropdown, station_dropdown]),
        output])
    display(widgets_collected)



def inspect_dual_sensors(D):
    """
    Function to interactively inspect profiles of sensor pairs (e.g., PSAL1 and PSAL2).

    Parameters:
    - D: xarray.Dataset, the dataset containing the variables.

    Usage:
    - Call inspect_dual_sensors(D) to interactively inspect profiles.
    """

    def plot_dual_sensor(station_index, variable_pair):
        """
        Plot profiles of the selected variable pair for a given station.

        Parameters:
        - station_index: int, index of the selected station.
        - variable_pair: tuple, pair of variables to be plotted.
        """
        variable1, variable2 = variable_pair

        fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

        profile_1 = D[variable1].isel(TIME=station_index)
        profile_2 = D[variable2].isel(TIME=station_index)

        ax[0].plot(profile_1, D['PRES'], alpha=0.8, lw=1, color='tab:orange', label=variable1)
        ax[0].plot(profile_2, D['PRES'], alpha=0.8, lw=1, color='tab:blue', label=variable2)
        ax[0].fill_betweenx(D['PRES'], profile_1, profile_2, alpha=0.2, color='k')

        ax[1].plot(profile_2 - profile_1, D['PRES'], alpha=0.8, lw=1, color='tab:red',
                   label=f'{variable2} - {variable1}')
        ax[1].fill_betweenx(D['PRES'], profile_2 - profile_1, alpha=0.2, color='tab:red',)

        difference_mean = np.round((profile_2 - profile_1).mean(), 6)
        ax[1].axvline(difference_mean, color='k', ls=':', alpha=0.5, label=f'Mean: {difference_mean.values}')

        station_time_string = time.convert_timenum_to_datetime(profile_1.TIME, D.TIME.units)
        fig.suptitle(f'Station: {D["STATION"].values[station_index]}, {station_time_string}')
        ax[0].set_xlabel(f'{D[variable1].units}')
        ax[1].set_xlabel(f'{D[variable1].units}')
        ax[0].set_title('Dual sensor profiles')
        ax[1].set_title('Dual sensor difference')
        
        ax[0].set_ylabel('PRES')
        ax[0].invert_yaxis()
        ax[0].grid()
        ax[1].grid()

        fig.canvas.header_visible = False  # Hide the figure header
        plt.tight_layout()
        ax[0].legend(fontsize=10)
        ax[1].legend(fontsize=8)
        plt.show()

    def _find_variable_pairs(D):
        """
        Find pairs of variables with the same name but different numbers.

        Parameters:
        - D: xarray.Dataset, the dataset containing the variables.

        Returns:
        - variable_pairs: list of tuples, pairs of variables.
        """
        variable_pairs = []
        data_vars = list(D.data_vars.keys())

        for i in range(len(data_vars)):
            current_var = data_vars[i]

            for j in range(i + 1, len(data_vars)):
                next_var = data_vars[j]

                # Extract the variable name and number from the variables
                current_name, current_num = ''.join(filter(str.isalpha, current_var)), ''.join(
                    filter(str.isdigit, current_var))
                next_name, next_num = ''.join(filter(str.isalpha, next_var)), ''.join(filter(str.isdigit, next_var))

                # Check if the variable names match and have different numbers
                if current_name == next_name and current_num != next_num:
                    variable_pairs.append((current_var, next_var))
        return variable_pairs

    # Get the descriptions for the slider
    station_descriptions = [str(station) for station in D['STATION'].values]

    # Create the slider for selecting a station
    station_index_slider = widgets.IntSlider(
        min=0, max=len(D['STATION']) - 1, step=1, value=0, description='Profile #:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')  # Set the width of the slider
    )

    variable_pairs = _find_variable_pairs(D)
    variable_pairs_labels = [f'{var_pair[0]} and {var_pair[1]}' for var_pair in variable_pairs]
    variable_pairs_dict = {vlab: vpair for vlab, vpair in zip(variable_pairs_labels, variable_pairs)}

    # Create the dropdown for selecting a variable
    variable_dropdown = widgets.Dropdown(
        options=variable_pairs_dict,
        value=variable_pairs[0],
        description='Variable'
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
        station_description = str(D['STATION'].values[change.new])
        station_dropdown.value = station_description

    # Link slider and dropdown
    station_dropdown.observe(update_slider_from_dropdown, names='value')
    station_index_slider.observe(update_dropdown_from_slider, names='value')

    # Use interactive_output to create interactive controls
    output = widgets.interactive_output(
        plot_dual_sensor, {'station_index': station_index_slider,
                           'variable_pair': variable_dropdown, }
    )

    # Create a button to close the plot
    close_button = widgets.Button(description="Close")

    def close_plot(_):
        # Resize the figure to 0 and close it
        fig = plt.gcf()
        fig.set_size_inches(0, 0)
        widgets_collected.close()
        plt.close(fig)

    close_button.on_click(close_plot)

    # Display the widgets in a vertically stacked layout
    widgets_collected = widgets.VBox([
        widgets.HBox([station_index_slider, close_button]),
        widgets.HBox([variable_dropdown, station_dropdown]),
        output])
    display(widgets_collected)




def map(D, height=1000, width=1000, return_fig_ax=False, coast_resolution='50m', figsize=None):
    '''
    Generate a quick map of a cruise using xarray Dataset coordinates.

    Parameters:
    - D (xarray.Dataset): The dataset containing latitude and longitude coordinates.
    - height (int, optional): Height of the map figure. Default is 1000.
    - width (int, optional): Width of the map figure. Default is 1000.
    - return_fig_ax (bool, optional): If True, return the Matplotlib figure and axis objects.
      Default is False.
    - coast_resolution (str, optional): Resolution of the coastline data ('50m', '110m', '10m').
      Default is '50m'.
    - figsize (tuple, optional): Size of the figure. If None, the default size is used.

    Displays a quick map using the provided xarray Dataset with latitude and longitude information.
    The map includes a plot of the cruise track and red dots at data points.

    Additionally, the function provides buttons for interaction:
    - "Close" minimizes and closes the plot.
    - "Original Size" restores the plot to its original size.
    - "Larger" increases the plot size.

    Examples:
    ```python
    map(my_dataset)
    ```
    or
    ```python
    fig, ax = map(my_dataset, return_fig_ax=True)
    ```

    Note: This function utilizes the `quickmap` module for generating a stereographic map.
    '''
    # These two are currently not used, but would maybe be useful for
    # auto-scaling of the map..
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
    minimize_button = widgets.Button(description="Close")

    def minimize_plot(_):
        # Resize the figure to 0 and close it
        fig.set_size_inches(0, 0)
        button_widgets.close()
        fig.close()

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

    button_widgets =  widgets.HBox([
        minimize_button, org_size_button, full_size_button, static_text], 
        layout=widgets.Layout(margin='0 0 5px 0', align_items='center'))
    
    display(button_widgets)
       
    if return_fig_ax:
        return fig, ax




def ctd_contours(D):
    """
    Create interactive contour plots based on an xarray dataset.

    Parameters:
    - D (xr.Dataset): The xarray dataset containing profile variables and coordinates.

    This function generates interactive contour plots for two selected profile variables
    from the given xarray dataset. It provides dropdown menus to choose the variables,
    select the x-axis variable (e.g., 'TIME', 'LONGITUDE', 'LATITUDE', 'Profile #'), and
    set the maximum depth for the y-axis.

    Additionally, the function includes a button to close the plot.

    Parameters:
    - D (xr.Dataset): The xarray dataset containing profile variables and coordinates.

    Examples:
    ```python
    ctd_contours(my_dataset)
    ```

    Note: This function uses the Matplotlib library for creating contour plots and the
    ipywidgets library for interactive elements.
    """

    # Function to update plots based on variable, xvar, and max depth selection
    def update_plots(variable1, variable2, xvar, max_depth):
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
        fig.canvas.header_visible = False  # Hide the figure header

        for axn, varnm in zip(ax, [variable1, variable2]):
            colormap = _cmap_picker(varnm)
            plt.xticks(rotation=0)
            
            if xvar == 'TIME':
                x_data = result_timestamp = time.datenum_to_timestamp(
                    D.TIME, D.TIME.units)
                plt.xticks(rotation=90)
                x_label = 'Time'
            elif xvar == 'LONGITUDE':
                x_data = D[xvar]
                x_label = 'Longitude'
            elif xvar == 'LATITUDE':
                x_data = D[xvar]
                x_label = 'Latitude'
            elif xvar == 'Profile #':
                x_data = np.arange(D.dims['TIME'])
                x_label = 'Profile #'
            else:
                raise ValueError(f"Invalid value for xvar: {xvar}")

            if xvar in ['TIME', 'Profile #']:
                D_sorted = D
            else:
                D_sorted = D.sortby(xvar)
                x_data = D_sorted[xvar] 

            C = axn.contourf(x_data, D_sorted.PRES, D_sorted[varnm].T, cmap=colormap, levels = 30)
            cb = plt.colorbar(C, ax=axn, label=D_sorted[varnm].units)
            
            # Set colorbar ticks using MaxNLocator
            cb.locator = MaxNLocator(nbins=6)  # Adjust the number of ticks as needed
            cb.update_ticks()

            axn.plot(x_data, np.zeros(D_sorted.dims['TIME']), '|k', clip_on = False, zorder = 0)

            axn.set_title(varnm)

            conts = axn.contour(x_data, D_sorted.PRES, D_sorted[varnm].T, colors = 'k', 
                                linewidths = 0.8, alpha = 0.2, levels = cb.get_ticks()[::2])

            axn.set_facecolor('lightgray')
            axn.set_ylabel('PRES [dbar]')

        ax[1].set_xlabel(x_label)
        ax[0].set_ylim(max_depth, 0)
        plt.tight_layout()

        plt.show()

    # Get the list of available variables
    available_variables = _ctd_tools._get_profile_variables(D)

    # Create dropdowns for variable selection
    variable_dropdown1 = widgets.Dropdown(options=available_variables, 
                                          value=available_variables[0], description='Variable 1:')
    variable_dropdown2 = widgets.Dropdown(options=available_variables, 
                                          value=available_variables[1], description='Variable 2:')

    # Create dropdown for x-variable selection
    xvar_dropdown = widgets.Dropdown(options=['TIME', 'LONGITUDE', 'LATITUDE', 'Profile #'], 
                                     value='Profile #', description='x axis:')

    # Create a button to minimize the plot
    close_button = widgets.Button(description="Close")

    def close_plot(_):
        # Resize the figure to 0 and close it
        fig = plt.gcf()
        fig.set_size_inches(0, 0)
        widgets_collected.close()
        plt.close(fig)

    close_button.on_click(close_plot)

    # Create slider for max depth selection
    max_depth_slider = widgets.IntSlider(min=1, max=D.PRES[-1].values, step=1, 
                                         value=D.PRES[-1].values, description='Max depth [m]:')

    # Use interactive to update plots based on variable, xvar, and max depth selection
    out = widgets.interactive_output(update_plots, 
                                     {'variable1': variable_dropdown1, 
                                      'variable2': variable_dropdown2, 
                                      'xvar': xvar_dropdown, 
                                      'max_depth': max_depth_slider})
    
    widgets_collected = widgets.VBox([widgets.HBox([variable_dropdown1, variable_dropdown2]), 
                          widgets.HBox([xvar_dropdown, close_button,]),  max_depth_slider, out])
    display(widgets_collected)



def _cmap_picker(varnm):
    '''
    Choose the appropriate colormap fir different variables.
    '''
    cmap_name = 'amp'
    if 'TEMP' in varnm:
        cmap_name = 'thermal'
    elif 'PSAL' in varnm:
        cmap_name = 'haline'
    elif 'CHLA' in varnm:
        cmap_name = 'algae'
    elif 'SIGTH' in varnm:
        cmap_name = 'deep'
    cmap = getattr(cmocean.cm, cmap_name)

    return cmap
