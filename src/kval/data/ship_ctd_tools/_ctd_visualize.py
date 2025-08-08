"""
Functions for visualizing CTD data.

These functions are designed to be run in a Jupyter Lab notebook using
the %matplotlib widget backend. They provide interactive controls for
exploring CTD data profiles and comparing sensor pairs.
"""

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import cartopy.crs as ccrs
from kval.util import time
from kval.data.ship_ctd_tools import _ctd_tools
from kval.maps import quickmap
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Colormap
import cmocean
import numpy as np
from kval.util import internals
from typing import Union
import xarray as xr

# thinf

def inspect_profiles(ds: 'xr.Dataset') -> None:
    """
    Interactively inspect profiles in an xarray dataset.

    Parameters:
    - ds: xr.Dataset
        The xarray dataset containing variables 'PRES', 'STATION', and
        other profile variables.

    Examples:
    ```python
    inspect_profiles(my_dataset)
    ```

    Note: This function utilizes Matplotlib for plotting and ipywidgets for
    interactive controls.
    """
    # Making sure we are in an interactive notebook environment
    internals.check_interactive()

    # Determine the vertical axis variable
    y_varnm = 'PRES' if 'PRES' in ds.dims else 'NISKIN_NUMBER'
    y_label = f'{y_varnm} [{ds[y_varnm].units}]' if 'units' in ds[y_varnm].attrs \
              else f'{y_varnm}'

    # Determine if this is a single profile
    is_single_profile = ds.sizes['TIME'] == 1

    def plot_profile(TIME_index: int, variable: str, y_varnm: str, y_label: str,
                     is_single_profile: bool = False) -> None:
        """
        Plot a single profile with the option to view others in the background.

        Parameters:
        - TIME_index: int
            Index of the selected profile.
        - variable: str
            Name of the variable to plot.
        - y_varnm: str
            Vertical axis variable name.
        - y_label: str
            Label for the vertical axis.
        """
        try:
            previous_fig = plt.gcf()
            plt.close(previous_fig)
        except Exception:
            pass

        fig, ax = plt.subplots()

        # Plot all profiles in black in the background
        if not is_single_profile:
            for nn in np.arange(ds.sizes['TIME']):
                if nn != TIME_index:  # Skip the selected profile
                    profile = ds[variable].isel(TIME=nn, drop=True).squeeze()
                    ax.plot(profile, ds[y_varnm], color='tab:blue', lw=0.5, alpha=0.4)

        # Choose marker size based on the number of points
        Nz = len(ds[y_varnm])
        ms = 2 if Nz > 100 else 2 + (100 - Nz) * 0.05

        # Plot the selected profile
        if not is_single_profile:
            profile = ds[variable].isel(TIME=TIME_index)
        else:
            profile = ds[variable]
        ax.plot(profile, ds[y_varnm], alpha=0.8, lw=0.7, color='k')
        ax.plot(profile, ds[y_varnm], '.', ms=ms, alpha=1, color='tab:orange')

        if not is_single_profile:
            time_string = time.convert_timenum_to_datestring(
                profile.TIME, ds.TIME.units)
            station = (ds["STATION"].values[TIME_index]
                       if 'STATION' in ds else 'N/A')
        else:
            time_string = time.convert_timenum_to_datestring(ds.TIME.item(),
                                                             ds.TIME.units)
            station = ds["STATION"].values.item() if 'STATION' in ds else 'N/A'

        ax.set_title(f'Station: {station}, {time_string}')
        var_unit = (ds[variable].units if 'units' in ds[variable].attrs
                    else 'no unit specified')
        ax.set_xlabel(f'{variable} [{var_unit}]')
        ax.set_ylabel(y_label)
        ax.invert_yaxis()
        ax.grid()
        fig.canvas.header_visible = False  # Hide the figure header
        plt.tight_layout()
        plt.show()

    # Create interactive widgets
    time_values = list(range(ds.sizes['TIME']))
    time_descriptions = [f'{nn} ({ds["STATION"].values[nn]})' for nn in time_values] \
                        if 'STATION' in ds else time_values

    time_index_slider = widgets.IntSlider(
        min=0, max=len(ds['TIME']) - 1, step=1, value=0,
        description='Profile #:', continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )


    profile_vars = _ctd_tools._get_profile_variables(
        ds, profile_var=y_varnm, require_TIME=not is_single_profile)

    variable_dropdown = widgets.Dropdown(
        options=profile_vars,
        value=profile_vars[0],
        description='Variable:'
    )

    time_option_tuples = [(desc, val) for desc, val in zip(time_descriptions, time_values)]
    time_ind_dropdown = widgets.Dropdown(
        options=time_option_tuples,
        value=time_values[0],
        description='TIME index:'
    )

    def update_slider_from_dropdown(change: widgets.Dropdown) -> None:
        time_index_slider.value = time_values.index(change.new)

    def update_dropdown_from_slider(change: widgets.IntSlider) -> None:
        time_ind_dropdown.value = time_values[change.new]

    time_ind_dropdown.observe(update_slider_from_dropdown, names='value')
    time_index_slider.observe(update_dropdown_from_slider, names='value')

    output = widgets.interactive_output(
        plot_profile, {
            'TIME_index': time_index_slider,
            'variable': variable_dropdown,
            'y_varnm': widgets.fixed(y_varnm),
            'y_label': widgets.fixed(y_label),
            'is_single_profile': widgets.fixed(is_single_profile)
        }
    )

    close_button = widgets.Button(description="Close")

    def close_plot(_) -> None:
        fig = plt.gcf()
        fig.set_size_inches(0, 0)
        widgets_collected.close()
        plt.close(fig)

    close_button.on_click(close_plot)

    widgets_collected = widgets.VBox([
        widgets.HBox([time_index_slider, close_button]),
        widgets.HBox([variable_dropdown, time_ind_dropdown]),
        output
    ])
    display(widgets_collected)

def inspect_phase_space(ds: 'xr.Dataset') -> None:
    """
    Interactively plot phase space (any variable vs any variable) from a profile dataset.

    Parameters:
    - ds: xr.Dataset
        The xarray dataset containing dimensions 'TIME' and other variables.

    Example:
        inspect_phase_space(my_dataset)
    """
    internals.check_interactive()

    # Determine if this is a single profile
    is_single_profile = ds.sizes['TIME'] == 1

    profile_vars = _ctd_tools._get_profile_variables(ds)
    
    def plot_profile(TIME_index: int, x_var: str, y_var: str) -> None:
        try:
            plt.close(plt.gcf())
        except Exception:
            pass

        fig, ax = plt.subplots()

        # Plot all other profiles in background
        if not is_single_profile:
            for nn in np.arange(ds.sizes['TIME']):
                if nn != TIME_index:
                    x = ds[x_var].isel(TIME=nn, drop=True).squeeze()
                    y = ds[y_var].isel(TIME=nn, drop=True).squeeze()
                    ax.plot(x, y, color='tab:blue', lw=0.5, alpha=0.4)

        # Plot selected profile
        if not is_single_profile:
            x = ds[x_var].isel(TIME=TIME_index)
            y = ds[y_var].isel(TIME=TIME_index)
        else:
            x = ds[x_var]
            y = ds[y_var]

        ms = 2 if len(y) > 100 else 2 + (100 - len(y)) * 0.05

        ax.plot(x, y, color='k', lw=0.7)
        ax.plot(x, y, '.', color='tab:orange', ms=ms)

        # Title & axis labels
        time_string = time.convert_timenum_to_datestring(
            ds.TIME[TIME_index] if not is_single_profile else ds.TIME.item(),
            ds.TIME.units
        )
        station = ds["STATION"].values[TIME_index] if 'STATION' in ds else 'N/A'

        ax.set_title(f'Station: {station}, {time_string}')

        xlabel = f"{x_var} [{ds[x_var].attrs.get('units', '')}]"
        ylabel = f"{y_var} [{ds[y_var].attrs.get('units', '')}]"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Auto reverse y-axis if pressure is on y
        if 'pres' in y_var.lower():
            ax.invert_yaxis()

        ax.grid()
        fig.canvas.header_visible = False
        plt.tight_layout()
        plt.show()

    # Widgets
    time_values = list(range(ds.sizes['TIME']))
    time_descriptions = [f'{nn} ({ds["STATION"].values[nn]})' for nn in time_values] \
                        if 'STATION' in ds else time_values

    time_slider = widgets.IntSlider(
        min=0, max=len(time_values)-1, value=0, description='Profile #:',
        style={'description_width': 'initial'}, layout=widgets.Layout(width='500px')
    )

    time_dropdown = widgets.Dropdown(
        options=[(desc, val) for desc, val in zip(time_descriptions, time_values)],
        value=0, description='TIME index:'
    )

    def sync_slider(change): time_slider.value = change.new
    def sync_dropdown(change): time_dropdown.value = time_values[change.new]

    time_dropdown.observe(sync_slider, names='value')
    time_slider.observe(sync_dropdown, names='value')

    x_dropdown = widgets.Dropdown(
        options=profile_vars,
        value=profile_vars[0],
        description='X-axis:'
    )

    y_dropdown = widgets.Dropdown(
        options=profile_vars,
        value='PRES' if 'PRES' in profile_vars else profile_vars[1],
        description='Y-axis:'
    )

    output = widgets.interactive_output(
        plot_profile, {
            'TIME_index': time_slider,
            'x_var': x_dropdown,
            'y_var': y_dropdown
        }
    )

    close_button = widgets.Button(description="Close")

    def close_plot(_):
        plt.close()
        widgets_collected.close()

    close_button.on_click(close_plot)

    widgets_collected = widgets.VBox([
        widgets.HBox([time_slider, close_button]),
        widgets.HBox([x_dropdown, y_dropdown, time_dropdown]),
        output
    ])
    display(widgets_collected)

def inspect_dual_sensors(ds: 'xr.Dataset') -> None:
    """
    Interactively inspect profiles of sensor pairs (e.g., PSAL1 and PSAL2).

    Parameters:
    - ds: xr.Dataset
        The dataset containing the variables.
    """
    # Making sure we are in an interactive notebook environment
    internals.check_interactive()

    def plot_dual_sensor(station_index: int, variable_pair: tuple) -> None:
        """
        Plot profiles of the selected variable pair for a given station.

        Parameters:
        - station_index: int
            Index of the selected station.
        - variable_pair: tuple
            Pair of variables to be plotted.
        """
        variable1, variable2 = variable_pair

        fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

        profile_1 = ds[variable1].isel(TIME=station_index)
        profile_2 = ds[variable2].isel(TIME=station_index)

        ax[0].plot(profile_1, ds['PRES'], alpha=0.8, lw=1, color='tab:orange',
                   label=variable1)
        ax[0].plot(profile_2, ds['PRES'], alpha=0.8, lw=1, color='tab:blue',
                   label=variable2)
        ax[0].fill_betweenx(ds['PRES'], profile_1, profile_2, alpha=0.2, color='k')

        diff_profile = profile_2 - profile_1
        ax[1].plot(diff_profile, ds['PRES'], alpha=0.8, lw=1, color='tab:red',
                   label=f'{variable2} - {variable1}')
        ax[1].fill_betweenx(ds['PRES'], diff_profile, alpha=0.2, color='tab:red')

        difference_mean = np.round(diff_profile.mean(), 6)
        ax[1].axvline(difference_mean, color='k', ls=':', alpha=0.5,
                      label=f'Mean: {difference_mean.values}')

        station_time_string = time.convert_timenum_to_datetime(profile_1.TIME, ds.TIME.units)
        fig.suptitle(f'Station: {ds["STATION"].values[station_index]}, {station_time_string}')

        var_unit = ds[variable1].units if 'units' in ds[variable1].attrs else 'no unit specified'
        ax[0].set_xlabel(f'{var_unit}')
        ax[1].set_xlabel(f'{var_unit}')
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

    def _find_variable_pairs(ds: 'xr.Dataset') -> list:
        """
        Find pairs of variables with the same name but different numbers.

        Parameters:
        - ds: xr.Dataset
            The dataset containing the variables.

        Returns:
        - list of tuples
            Pairs of variables.
        """
        variable_pairs = []
        data_vars = list(ds.data_vars.keys())

        for i in range(len(data_vars)):
            current_var = data_vars[i]
            for j in range(i + 1, len(data_vars)):
                next_var = data_vars[j]
                current_name, current_num = ''.join(filter(str.isalpha, current_var)), \
                                            ''.join(filter(str.isdigit, current_var))
                next_name, next_num = ''.join(filter(str.isalpha, next_var)), \
                                       ''.join(filter(str.isdigit, next_var))
                if current_name == next_name and current_num != next_num:
                    variable_pairs.append((current_var, next_var))
        return variable_pairs

    # Create interactive widgets
    station_descriptions = [str(station) for station in ds['STATION'].values]
    station_index_slider = widgets.IntSlider(
        min=0, max=len(ds['STATION']) - 1, step=1, value=0,
        description='Profile #:', continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )

    variable_pairs = _find_variable_pairs(ds)
    if not variable_pairs:
        print('No dual sensors found in the dataset.')
        return

    variable_pairs_labels = [f'{var_pair[0]} and {var_pair[1]}' for var_pair in variable_pairs]
    variable_pairs_dict = {label: pair for label, pair in zip(variable_pairs_labels, variable_pairs)}

    variable_dropdown = widgets.Dropdown(
        options=variable_pairs_dict,
        value=variable_pairs[0],
        description='Variable'
    )

    station_dropdown = widgets.Dropdown(
        options=station_descriptions,
        value=station_descriptions[0],
        description='Station:'
    )

    def update_slider_from_dropdown(change: widgets.Dropdown) -> None:
        station_index_slider.value = station_descriptions.index(change.new)

    def update_dropdown_from_slider(change: widgets.IntSlider) -> None:
        station_dropdown.value = station_descriptions[change.new]

    station_dropdown.observe(update_slider_from_dropdown, names='value')
    station_index_slider.observe(update_dropdown_from_slider, names='value')

    output = widgets.interactive_output(
        plot_dual_sensor, {
            'station_index': station_index_slider,
            'variable_pair': variable_dropdown
        }
    )

    close_button = widgets.Button(description="Close")

    def close_plot(_) -> None:
        fig = plt.gcf()
        fig.set_size_inches(0, 0)
        widgets_collected.close()
        plt.close(fig)

    close_button.on_click(close_plot)

    widgets_collected = widgets.VBox([
        widgets.HBox([station_index_slider, close_button]),
        widgets.HBox([variable_dropdown, station_dropdown]),
        output
    ])
    display(widgets_collected)


def map(
    ds: 'xr.Dataset',
    height: int = 1000,
    width: int = 1000,
    return_fig_ax: bool = False,
    coast_resolution: str = '50m',
    figsize: tuple[int, int] = None,
    station_labels: Union[bool, str] = False,
    station_label_alpha: float = 0.5
) -> Union[tuple[plt.Figure, plt.Axes], None]:

    # Making sure we are in an interactive notebook environment
    internals.check_interactive()

    lat_span = float(ds.LATITUDE.max() - ds.LATITUDE.min())
    lon_span = float(ds.LONGITUDE.max() - ds.LONGITUDE.min())
    lat_ctr = float(0.5 * (ds.LATITUDE.max() + ds.LATITUDE.min()))
    lon_ctr = float(0.5 * (ds.LONGITUDE.max() + ds.LONGITUDE.min()))

    fig, ax = quickmap.quick_map_stere(
        lon_ctr, lat_ctr, height=height, width=width, coast_resolution=coast_resolution
    )

    fig.canvas.header_visible = False  # Hide the figure header

    ax.plot(ds.LONGITUDE, ds.LATITUDE, '-k', transform=ccrs.PlateCarree(), alpha=0.5)
    ax.plot(ds.LONGITUDE, ds.LATITUDE, 'or', transform=ccrs.PlateCarree())

    # Add labels next to each point if required
    if station_labels:
        xytext, ha, va = (0, 0), 'center', 'center'
        if station_labels == 'left' or station_labels is True:
            xytext, ha, va = (-5, 0), 'right', 'center'
        elif station_labels == 'right':
            xytext, ha, va = (5, 0), 'left', 'center'
        elif station_labels == 'above':
            xytext, ha, va = (0, 5), 'center', 'bottom'
        elif station_labels == 'below':
            xytext, ha, va = (0, -5), 'center', 'top'

        for label, x, y in zip(ds.STATION.values, ds.LONGITUDE, ds.LATITUDE):
            plt.annotate(
                label, (x, y), textcoords="offset points", xytext=xytext, ha=ha, va=va,
                transform=ccrs.PlateCarree(), alpha=station_label_alpha, fontsize=8
            )

    plt.tight_layout()

    if figsize:
        fig.set_size_inches(figsize)
    else:
        figsize = fig.get_size_inches()

    # Create interactive buttons
    minimize_button = widgets.Button(description="Close")

    def minimize_plot(_) -> None:
        fig.set_size_inches(0, 0)
        button_widgets.close()
        plt.close(fig)

    minimize_button.on_click(minimize_plot)

    org_size_button = widgets.Button(description="Original Size")

    def org_size_plot(_) -> None:
        fig.set_size_inches(figsize)
        fig.canvas.draw()

    org_size_button.on_click(org_size_plot)

    full_size_button = widgets.Button(description="Larger")

    def full_size_plot(_) -> None:
        fig.set_size_inches(fig.get_size_inches() * 2)
        fig.canvas.draw()

    full_size_button.on_click(full_size_plot)

    static_text = widgets.HTML(
        value='<p>Use the menu on the left of the figure to zoom/move around/save</p>'
    )

    button_widgets = widgets.HBox(
        [minimize_button, org_size_button, full_size_button, static_text],
        layout=widgets.Layout(margin='0 0 5px 0', align_items='center')
    )

    display(button_widgets)

    if return_fig_ax:
        return fig, ax




def ctd_contours(ds):
    """

    This function generates interactive contour plots for two selected profile variables
    from the given xarray dataset. It provides dropdown menus to choose the variables,
    select the x-axis variable (e.g., 'TIME', 'LONGITUDE', 'LATITUDE', 'Profile #'), and
    set the maximum depth for the y-axis.

    Additionally, the function includes a button to close the plot.

    Parameters:
    - ds (xr.Dataset): The xarray dataset containing profile variables and coordinates.

    Examples:
    ```python
    ctd_contours(my_dataset)
    ```

    Note: This function uses the Matplotlib library for creating contour plots and the
    ipywidgets library for interactive elements.
    """

    # Assign the profile variable to PRES (profile files) or
    # NISKIN_NUMBER (btl or water sample files)
    if 'PRES' in ds.dims:
        y_varnm = 'PRES'
    elif 'NISKIN_NUMBER' in ds.dims:
        y_varnm = 'NISKIN_NUMBER'

    if 'units' in ds[y_varnm].attrs:
        y_label = f'{y_varnm} [{ds[y_varnm].units}]'
    else:
        y_label = f'{y_varnm}'

    # Function to update plots based on variable, xvar, and max depth selection
    def update_plots(variable1, variable2, xvar, max_depth):

        try:
            previous_fig = plt.gcf()
            plt.close(previous_fig)
        except:
            pass

        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
        fig.canvas.header_visible = False  # Hide the figure header

        for axn, varnm in zip(ax, [variable1, variable2]):
            colormap = _cmap_picker(varnm)
            plt.xticks(rotation=0)

            if xvar == 'TIME':
                x_data = result_timestamp = time.datenum_to_timestamp(
                    ds.TIME, ds.TIME.units)
                plt.xticks(rotation=90)
                x_label = 'Time'
            elif xvar == 'LONGITUDE':
                x_data = ds[xvar]
                x_label = 'Longitude'
            elif xvar == 'LATITUDE':
                x_data = ds[xvar]
                x_label = 'Latitude'
            elif xvar == 'Profile #':
                x_data = np.arange(ds.sizes['TIME'])
                x_label = 'Profile #'
            else:
                raise ValueError(f"Invalid value for xvar: {xvar}")

            if xvar in ['TIME', 'Profile #']:
                ds_sorted = ds
            else:
                ds_sorted = ds.sortby(xvar)
                x_data = ds_sorted[xvar]

            C = axn.contourf(x_data, ds_sorted[y_varnm], ds_sorted[varnm].T,
                             cmap=colormap, levels = 30)

            if 'units' in ds_sorted[varnm].attrs:
                var_unit_ = ds_sorted[varnm].units
            else:
                var_unit_ = 'no unit specified'

            cb = plt.colorbar(C, ax=axn, label=var_unit_)

            # Set colorbar ticks using MaxNLocator
            cb.locator = MaxNLocator(nbins=6)  # Adjust the number of ticks as needed
            cb.update_ticks()

            axn.plot(x_data, np.zeros(ds_sorted.sizes['TIME']), '|k',
                     clip_on = False, zorder = 0)

            axn.set_title(varnm)

            conts = axn.contour(x_data, ds_sorted[y_varnm], ds_sorted[varnm].T,
                        colors = 'k', linewidths = 0.8, alpha = 0.2,
                        levels = cb.get_ticks()[::2])

            axn.set_facecolor('lightgray')
            axn.set_ylabel(y_label)

        ax[1].set_xlabel(x_label)
        ax[0].set_ylim(max_depth, 0)
        plt.tight_layout()

        plt.show()

    # Get the list of available variables
    available_variables = _ctd_tools._get_profile_variables(
        ds, profile_var = y_varnm)

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
    max_depth_slider = widgets.IntSlider(min=1, max=ds[y_varnm][-1].values, step=1,
                                         value=ds[y_varnm][-1].values, description='Max depth [m]:')

    # Use interactive to update plots based on variable, xvar, and max depth selection
    out = widgets.interactive_output(update_plots,
                                     {'variable1': variable_dropdown1,
                                      'variable2': variable_dropdown2,
                                      'xvar': xvar_dropdown,
                                      'max_depth': max_depth_slider})

    widgets_collected = widgets.VBox([widgets.HBox([variable_dropdown1, variable_dropdown2]),
                          widgets.HBox([xvar_dropdown, close_button,]),  max_depth_slider, out])
    display(widgets_collected)


def _cmap_picker(varnm: str) -> Colormap:
    '''
    Choose the appropriate colormap for different variables.
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
    elif 'CDOM' in varnm:
        cmap_name = 'turbid'
    elif 'DOXY' in varnm:
        cmap_name = 'tempo'
    elif 'ATTN' in varnm:
        cmap_name = 'matter'
    elif 'SVEL' in varnm:
        cmap_name = 'speed'
    cmap = getattr(cmocean.cm, cmap_name)

    return cmap