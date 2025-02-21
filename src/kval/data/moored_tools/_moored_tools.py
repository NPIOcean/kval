'''
## kval.data.moored_tools._moored_tools

Various functions for making modifications to moored sensor data in xarray format
'''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib.widgets import RectangleSelector
import ipywidgets as widgets
from IPython.display import display, clear_output
from kval.util import time, internals, index
from kval.data import edit, moored
from kval.signal.filt import rolling
import xarray as xr
import pandas as pd
#from kval.data.ship_ctd_tools import _ctd_tools

class hand_remove_points:
    """
    A class for interactive removal of data points from CTD profiles

    Parameters:
    - ds (xarray.Dataset): The dataset containing the CTD data.
    - varnm (str): The name of the variable to visualize and edit (e.g. 'TEMP1', 'CHLA').
    - station (str): The name of the station (E.g. '003', '012_01', 'AT290', 'StationA').

    Example usage that will let you hand edit the profile of "TEMP1" from station "StationA":
    ```python
    HandRemovePoints(my_dataset, 'TEMP1', 'StationA')
    ```
    Note: Use the interactive plot to select points for removal, then click the corresponding buttons for actions.
    """

    def __init__(self, ds, varnm, varnm_edit=None):
        """
        Initialize the HandRemovePoints instance.

        Parameters:
        - ds (xarray.Dataset): The dataset containing the data.
        - varnm (str): The name of the variable to visualize .
        - varnm_edit (str), Optional:
            The name of the variable to edit.
            Defaults to `varnm`
        """


        # Check that we in a notebook and with the ipympl backend..
        # (raise a warning otherwise)

       # internals.check_interactive()

        if not varnm_edit:
            varnm_edit = varnm

        if varnm not in ds.data_vars:
            raise Exception(f'Invalid variable ("{varnm}")')

        if varnm_edit not in ds.data_vars:
            raise Exception(f'Invalid variable ("{varnm_edit}")')

        self.varnm = varnm
        self.varnm_edit = varnm_edit

        self.ds = ds

        self.var_data = ds[varnm]

        # Read time (as float - convert if necessary)
        if isinstance(ds.TIME[0].values, np.datetime64):
            self.TIME = date2num(ds.TIME.copy())
            #self.TIME.attrs['units'] = 'Days since 1970-01-01'
            time_units = 'Days since 1970-01-01'
        else:
            self.TIME = ds.TIME.copy()

        self.Npres = len(ds.TIME)

        self.fig, self.ax = plt.subplots(figsize = (11, 6))


        line, = self.ax.plot(self.TIME, self.var_data, )
        point = self.ax.plot(self.TIME, self.var_data, '.k', zorder=3)
        picks = np.array([])
        xlims = self.ax.get_xlim()
        self.ax.set_xlim(xlims)
        ylims = self.ax.get_ylim()
        self.ax.set_ylim(ylims)

        if varnm=='PRES':
            self.ax.invert_yaxis()

        if 'units' in self.ds[varnm].attrs:
            self.ax.set_ylabel(f'{varnm} [{self.ds[varnm].units}]')
        else:
            self.ax.set_ylabel(f'{varnm} [no units]')
        if hasattr(self.TIME, 'units'):
            if 'units' in self.TIME.attrs:
                self.ax.set_xlabel(f'TIME [{self.TIME.units}]')

            else:
                try:
                    self.ax.set_xlabel(f'TIME [{time_units}]')
                except:
                    self.ax.set_xlabel(f'TIME [no units]')

        self.ax.grid()
        #station_time_string = time.convert_timenum_to_datetime(self.ds.TIME.values[TIME_index], ds.TIME.units)
        self.fig.canvas.header_visible = False  # Hide the figure header
      #  self.ax.set_title(f'Station: {self.station}: {station_time_string}')
        plt.tight_layout()


        # Use interactive_output to create interactive controls
        self.es = RectangleSelector(self.ax, self.onselect, interactive=True)

        # Can maybe cut these 4 (use remove_bool instead)?
        self.var_points_remove = np.array([])
        self.var_points_selected = np.array([])
        self.TIME_points_selected = np.array([])
        self.TIME_points_remove = np.array([])
        self.remove_bool = np.bool_(np.zeros(self.Npres))

        self.TF_indices_selected = np.bool_(np.zeros(self.Npres))

        self.temp_label = 'Points to remove'
        self.remove_label = 'Selected points to remove'

        # Add widget buttons
        self.button_apply_var = widgets.Button(description=f"Exit and apply to {self.varnm_edit}")
        self.button_apply_var.on_click(self.exit_and_apply_var)
        self.button_apply_var.layout.width = '200px'

        self.button_exit_nochange = widgets.Button(description=f"Discard and exit")
        self.button_exit_nochange.on_click(self.exit_and_discard)
        self.button_exit_nochange.layout.width = '200px'

        self.text_widget1 = widgets.HTML(value=(
            'Drag a rectangle to select points. '))
        self.text_widget2 = widgets.HTML(value=(    '<span style="color: gray;">Click icons on left side of the plot'
            ' to zoom/move/save (click icon again to regain cursor).</span>'))

        self.button_remove = widgets.Button(description="Remove selected")
        self.button_remove.on_click(self.remove_selected)
        self.button_remove.style.button_color = '#FFB6C1'  # You can use any valid CSS color

        self.button_forget = widgets.Button(description="Forget selection")
        self.button_forget.on_click(self.forget_selection)
        self.button_forget.style.button_color = 'lightblue'  # You can use any valid CSS color

        self.button_restart = widgets.Button(description="Start over")
        self.button_restart.on_click(self.start_over_selection)
        self.button_restart.style.button_color = 'yellow'  # You can use any valid CSS color


        self.buttons_container_1 = widgets.HBox([self.button_apply_var,
                                                self.button_exit_nochange])
        self.buttons_container_2 = widgets.HBox([
            self.button_remove, self.button_forget, self.button_restart])
        self.buttons_container_3 = widgets.HBox([
            widgets.VBox([self.text_widget1, self.text_widget2])])

        # Add an Output widget to capture the print statement
        self.output_widget = widgets.Output()

        self.widgets_all = widgets.VBox([self.buttons_container_1,
                                         widgets.Output(), self.buttons_container_2,
                                         self.buttons_container_3,])

        # Display the widgets
        display(self.widgets_all)
        display(self.output_widget)


    def onselect(self, eclick, erelease):
        """
        Handle the selection of points in the plot.

        Parameters:
        - eclick (tuple): The coordinates of the press event.
        - erelease (tuple): The coordinates of the release event.
        """
        ext = self.es.extents
        rectangle = plt.Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2])

        self.contains_TF_ = rectangle.contains_points(np.vstack([self.TIME,
                                                                 self.var_data]).T)
        print
        self.var_points_selected = np.concatenate([
            self.var_points_selected,
            self.var_data[self.contains_TF_]])

        self.TIME_points_selected = np.concatenate([
            self.TIME_points_selected,
            self.TIME[self.contains_TF_]])

        self.TF_indices_selected = np.bool_(
            self.TF_indices_selected + self.contains_TF_)

        try:
            self.temp_scatter.remove()
            plt.draw()
        except:
            pass

        self.temp_scatter = self.ax.scatter(
            self.TIME_points_selected,
            self.var_points_selected,
            color='b', label=self.temp_label)

        self.ax.legend()
        self.TF_indices_selected[self.contains_TF_] = True
        plt.draw()

    def remove_selected(self, button):
        """
        Handle the removal of selected points.

        Parameters:
        - button: The button click event.
        """
        # Check that we in a notebook and with the ipympl backends..
        # (raise a warning otherwise)

        internals.check_interactive()

        self.var_points_remove = np.concatenate(
            [self.var_points_remove, self.var_points_selected])
        self.TIME_points_remove = np.concatenate(
            [self.TIME_points_remove, self.TIME_points_selected])

        try:
            self.remove_scatter.remove()
            plt.draw()
        except:
            pass

        self.remove_scatter = self.ax.scatter(
            self.TIME_points_remove,
            self.var_points_remove,
            color='r', label=self.remove_label, zorder=2)
        plt.draw()
        self.remove_label = None
        self.var_points_selected = np.array([])
        self.TIME_points_selected = np.array([])
        self.remove_bool[self.TF_indices_selected] = True

        self.ax.legend()


    def forget_selection(self, button):
        """
        Handle the forgetting of the current selection.

        Parameters:
        - button: The button click event.
        """
        try:
            self.temp_scatter.remove()
            plt.draw()
        except:
            pass
        self.TIME_points_selected = np.array([])
        self.var_points_selected = np.array([])
        self.TF_indices_selected = np.bool_(np.zeros(self.Npres))

    def start_over_selection(self, button):
        """
        Handle starting over the selection process.

        Parameters:
        - button: The button click event.
        """
        try:
            self.temp_scatter.remove()
            plt.draw()
        except:
            pass
        self.var_points_remove = np.array([])
        self.var_points_selected = np.array([])
        self.TIME_points_selected = np.array([])
        self.TIME_points_remove = np.array([])
        self.remove_bool = np.bool_(np.zeros(self.Npres))
        try:
            self.remove_scatter.remove()
            plt.draw()
        except:
            pass

    def exit_and_apply_var(self, button):
        """
        Handle the exit and apply action.

        Parameters:
        - button: The button click event.
        """

        self.remove_inds = index.indices_to_slices(np.where(self.remove_bool)[0])

        self.ds = moored.remove_points(
            self.ds, self.varnm_edit, self.remove_inds)
        # If we have a PROCESSING field:
        # Count how many points we removed
        self.points_removed = np.sum(self.remove_bool)

        # Add info as a variable attribute
        if 'manual_editing' in self.ds[self.varnm_edit].attrs.keys():
            previous_edits = int(self.ds[self.varnm_edit].attrs['manual_editing'].split()[0])
            total_edits = previous_edits + self.points_removed
            self.ds[self.varnm_edit].attrs['manual_editing'] = (
               f'{total_edits} data points have been removed '
                'from this variable based on visual inspection.')
        else:
            self.ds[self.varnm_edit].attrs['manual_editing'] = (
               f'{self.points_removed} data points have been '
                'removed from this variable based on visual inspection.')
            if self.points_removed == 1:
                self.ds[self.varnm_edit].attrs['manual_editing']  = (
                    self.ds[self.varnm_edit].attrs['manual_editing'].replace(
                        'points have', 'point has')
                    )

        with self.output_widget:
            clear_output(wait=True)
            print(f'APPLIED TO DATASET - Removed {self.points_removed} point(s)')

        self.close_everything()


    def exit_and_discard(self, button):
        """
        Handle the exit and discard action.

        Parameters:
        - button: The button click event.
        """
        self.close_everything()

        with self.output_widget:
            clear_output(wait=True)
            print(f'EXITED WITHOUT CHANGING ANYTHING')

    def close_everything(self):
        """
        Close the figure and widgets.
        """
        fig = plt.gcf()
        fig.set_size_inches(0, 0)
        self.widgets_all.close()
        plt.close(fig)

    # Set this so we don't return an object (and thereby printing a messy string)
    def __repr__(self):
        """
        Return an empty string.
        """
        # Return an empty string or any custom string
        return ''

################################################################################


def inspect_time_series(ds: xr.Dataset) -> None:
    """
    Visualizes time series data interactively with options for applying hourly and daily mean filters.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the time series data. The dataset must have a 'TIME' dimension.
    """
    # Get the list of variables with a TIME dimension
    time_vars = [var for var in ds.data_vars if 'TIME' in ds[var].dims]

    if not time_vars:
        print("No variables with a TIME dimension found.")
        return

    # Create widgets
    var_selector = widgets.Dropdown(
        options=time_vars,
        description='Variable: ',
        value=time_vars[0],
        layout=widgets.Layout(width='200px')
    )

    # Checkbox for displaying grid
    grid_checkbox = widgets.Checkbox(
        value=False, description='Grid ',
        layout=widgets.Layout(width='200px')
    )

    # Widgets for hourly mean options
    hourly_mean_checkbox = widgets.Checkbox(
        value=False, description='Hourly mean'
    )

    hours_input = widgets.IntSlider(
        value=1, min=1, max=24, step=1, description='# hours: ',
        layout=widgets.Layout(width='200px'),
        readout=False
    )

    # Update the hourly mean checkbox label based on the slider value
    def update_hourly_mean_label(change):
        hourly_mean_checkbox.description = f"{change['new']}-hour mean"

    # Attach the update function to the slider
    hours_input.observe(update_hourly_mean_label, names='value')

    # Widgets for daily mean options
    daily_mean_checkbox = widgets.Checkbox(
        value=False, description='Daily mean'
    )

    days_input = widgets.IntSlider(
        value=1, min=1, max=30, step=1, description='# days:',
        readout=False,
        layout=widgets.Layout(width='200px'),
    )

    # Update the daily mean checkbox label based on the slider value
    def update_daily_mean_label(change):
        daily_mean_checkbox.description = f"{change['new']}-day mean"

    # Attach the update function to the slider
    days_input.observe(update_daily_mean_label, names='value')


    # Clse button
    close_button = widgets.Button(description="Close")

    def close_plot(_) -> None:
        fig = plt.gcf()
        fig.set_size_inches(0, 0)
        controls.close()
        plt.close(fig)

    close_button.on_click(close_plot)

    # Layout: Stack the controls horizontally
    controls = widgets.HBox([
        widgets.VBox([var_selector, grid_checkbox]),
        widgets.VBox([hourly_mean_checkbox, hours_input],
                     layout=widgets.Layout(width='200px')),
        widgets.VBox([daily_mean_checkbox, days_input]),
        close_button
    ])  # Align items to the start)

    # Define the plot function
    def plot_time_series(variable, hourly_apply, hours, daily_apply, days, grid_apply):
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot original data
        ds[variable].plot(ax=ax, label=f"{variable} (original)", color='k', alpha=0.7)

        # Apply hourly mean if checkbox is checked
        if hourly_apply:
            try:
                ds_resampled = ds[variable].resample(TIME=f'{hours}h', label='right').mean()
                half_window_offset = pd.Timedelta(hours=hours / 2)
                ds_resampled['TIME'] = ds_resampled.TIME - half_window_offset
                ax.plot(ds_resampled.TIME, ds_resampled, label=(
                    f"{variable} (hourly mean, window={hours} hours)"),
                    color='tab:cyan')
            except Exception as e:
                print(f"Error applying hourly mean: {e}")

        # Apply daily mean if checkbox is checked
        if daily_apply:
            try:
                ds_resampled = ds[variable].resample(TIME=f'{days}D', label='right').mean()
                half_window_offset = pd.Timedelta(days=days / 2)
                ds_resampled['TIME'] = ds_resampled.TIME - half_window_offset
                ax.plot(ds_resampled.TIME, ds_resampled, label=(
                    f"{variable} (daily mean, window={days} days)"),
                    color='tab:orange')
            except Exception as e:
                print(f"Error applying daily mean: {e}")

        # Show grid if checkbox is checked
        if grid_checkbox.value:
            ax.grid()

        ax.legend()
        ax.set_title(f"Time Series: {variable}")
        fig.canvas.header_visible = False  # Hide the figure header
        plt.tight_layout()

        plt.show(block=False)
    # Define interaction without calling display explicitly
    interact_plot = widgets.interactive(
        plot_time_series,
        variable=var_selector,
        hourly_apply=hourly_mean_checkbox,
        hours=hours_input,
        daily_apply=daily_mean_checkbox,
        days=days_input,
        grid_apply=grid_checkbox,
    )


    # Display controls and plot
    display(widgets.VBox([controls, interact_plot.children[-1]],
                         layout=widgets.Layout(padding='0px', margin='0px')))
