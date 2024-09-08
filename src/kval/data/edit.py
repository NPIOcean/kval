"""
EDIT.PY

Functions for editing (generalized) datasets.

Interactive functions toward the end
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import ipywidgets as widgets
from typing import Optional
from IPython.display import display, clear_output
from kval.util import internals
from kval.data import ctd, moored
from kval.calc.number import order_of_magnitude

def remove_points_profile(ds: xr.Dataset, varnm: str, TIME_index: int,
                          remove_inds) -> xr.Dataset:
    """
    Remove specified points from a profile in the dataset by setting them to NaN.

    Parameters:
    - ds: xarray.Dataset
      The dataset containing the variable to modify.
    - varnm: str
      The name of the variable to modify.
    - TIME_index: int
      The index along the TIME dimension to modify.
    - remove_inds: list or array-like
      Indices of points to remove (set to NaN) within the specified TIME profile.

    Returns:
    - ds: xarray.Dataset
      The dataset with specified points removed (set to NaN).
    """

    # Convert remove_inds to a list if it's not already
    remove_inds = np.asarray(remove_inds)

    # Create a boolean array for removal
    remove_bool = np.zeros(len(ds[varnm].isel(TIME=TIME_index)), dtype=bool)
    remove_bool[remove_inds] = True

    # Use the `where` method to set the selected points to NaN
    ds[varnm].isel(TIME=TIME_index).values[:] = np.where(remove_bool,
                                                         np.nan,
                                                         ds[varnm].isel(TIME=TIME_index).values)

    return ds

def offset(ds: xr.Dataset, variable: str, offset: float) -> xr.Dataset:
    """
    Apply a fixed offset to a specified variable in an xarray Dataset.

    This function modifies the values of the specified variable by adding a fixed
    offset to them. The `valid_min` and `valid_max` attributes are updated to reflect
    the new range of values after applying the offset.

    Parameters
    ----------
    ds : xr.Dataset
        The input xarray Dataset.
    variable : str
        The name of the variable within the Dataset to which the offset will be applied.
    offset : float
        The fixed offset value to add to the variable.

    Returns
    -------
    xr.Dataset
        A new xarray Dataset with the offset applied to the specified variable. The
        `valid_min` and `valid_max` attributes are updated accordingly.

    Examples
    --------
    # Apply an offset of 5 to the 'TEMP' variable
    ds_offset = offset(ds, 'TEMP', offset=5)
    """

    ds_new = ds.copy()

    if variable not in ds_new:
        raise ValueError(f"Variable '{variable}' not found in the Dataset.")

    # Apply the offset
    ds_new[variable] = ds_new[variable] + offset
    ds_new[variable].attrs = ds[variable].attrs

    # Update the valid_min and valid_max attributes if they exist
    if 'valid_min' in ds_new[variable].attrs:
        ds_new[variable].attrs['valid_min'] += offset

    if 'valid_max' in ds_new[variable].attrs:
        ds_new[variable].attrs['valid_max'] += offset

    return ds_new

def threshold(ds: xr.Dataset, variable: str,
              max_val: Optional[float] = None,
              min_val: Optional[float] = None) -> xr.Dataset:
    """
    Apply a threshold to a specified variable in an xarray Dataset, setting
    values outside the specified range (min_val, max_val) to NaN.

    Also modifies the valid_min and valid_max variable attributes.

    Parameters
    ----------
    ds : xr.Dataset
        The input xarray Dataset.
    variable : str
        The name of the variable within the Dataset to be thresholded.
    max_val : Optional[float], default=None
        The maximum allowed value for the variable. Values greater than
        this will be set to NaN.
        If None, no upper threshold is applied.
    min_val : Optional[float], default=None
        The minimum allowed value for the variable. Values less than
        this will be set to NaN.
        If None, no lower threshold is applied.

    Returns
    -------
    xr.Dataset
        A new xarray Dataset with the thresholded variable. The `valid_min`
        and `valid_max` attributes are updated accordingly.

    Examples
    --------
    # Reject temperatures below -1 and above 3
    ds_thresholded = threshold(ds, 'TEMP', min_val=-1, max_val=3)
    """

    ds_new = ds.copy()

    if max_val is not None:
        ds_new[variable] = ds_new[variable].where(ds_new[variable] <= max_val)
        ds_new[variable].attrs['valid_max'] = max_val

    if min_val is not None:
        ds_new[variable] = ds_new[variable].where(ds_new[variable] >= min_val)
        ds_new[variable].attrs['valid_min'] = min_val

        if max_val is not None and max_val <= min_val:
            raise ValueError(f'Threshold editing: max_val ({max_val}) must be greater than min_val ({min_val}).')

    return ds_new






class drop_vars_pick:
    '''
    Interactive class for dropping selected variables from an xarray Dataset.

    Parameters:
    - D (xarray.Dataset): The dataset from which variables will be dropped.

    Displays an interactive widget with checkboxes for each variable, allowing users
    to select variables to remove. The removal is performed by clicking the "Drop variables"
    button. The removed variables are also printed to the output.

    Examples:
    ```python
    drop_vars_pick(my_dataset)
    ```

    Note: This class utilizes IPython widgets for interactive use within a Jupyter environment.
    '''


    def __init__(self, D, moored=None):

        # Check that we in a notebook and with the ipympl backend..
        # (raise a warning otherwise)
        internals.check_interactive()

        self.D = D
        self.moored = moored
        self.selected_options = []

        # List of checkbox labels
        self.checkbox_labels = list(D.data_vars)

        # Create Checkbox widgets
        self.checkbox_widgets = [widgets.Checkbox(description=label) for label in self.checkbox_labels]

        # Calculate the number of checkboxes per column
        checkboxes_per_column = len(self.checkbox_labels) // 2 + 1

        # Create VBox widgets for each column with reduced spacing
        column1 = widgets.VBox(self.checkbox_widgets[:checkboxes_per_column], layout=widgets.Layout(margin='0px'))
        column2 = widgets.VBox(self.checkbox_widgets[checkboxes_per_column:], layout=widgets.Layout(margin='0px'))

        # Arrange columns in an HBox
        self.checkbox_columns = widgets.HBox([column1, column2])

        # Text widget
        self.text_widget = widgets.HTML(value="Select variables to remove:")

        # Create an Output widget to capture the print statement when done
        self.output_widget = widgets.Output()

        # Create buttons
        self.remove_button = widgets.ToggleButton(value=False, description='Drop variables', button_style='success')
        self.exit_button = widgets.ToggleButton(value=False, description='Exit', button_style='danger')

        # Attach button event handlers
        self.remove_button.observe(lambda change: self.on_remove_button_click(change, D), names='value')
        self.exit_button.observe(self.on_exit_button_click, names='value')

        # Layout for buttons
        hbox_layout = widgets.Layout(align_items='flex-end')
        self.hbox_buttons = widgets.HBox([self.remove_button, self.exit_button], layout=hbox_layout)

         # Function to handle checkbox changes
        def handle_checkbox_change(change):
            self.selected_options = [label for checkbox, label in zip(self.checkbox_widgets, self.checkbox_labels) if checkbox.value]

        # Attach the handle_checkbox_change function to the observe method of each checkbox
        for checkbox in self.checkbox_widgets:
            checkbox.observe(handle_checkbox_change, names='value')

        # Display widgets
        display(self.text_widget)
        display(self.checkbox_columns)
        display(self.hbox_buttons)
        display(self.output_widget)

    def on_remove_button_click(self, change, D):
        if change['new']:

            # Clunky: Want to preserve the metadata from drop_variables,
            # but the straight forward approach doesnt actually reduce the
            # variables.

            # For a mooring dataset:
            if self.moored:
            # Otherwise assume CTD dataset
                D_dropped = moored.drop_variables(
                  self.D, drop_vars = self.selected_options)
            else:
                D_dropped = ctd.drop_variables(
                  self.D, drop_vars = self.selected_options)

            if 'PROCESSING' in D_dropped:
                self.D['PROCESSING'] = D_dropped.PROCESSING

            for key in self.selected_options:
                del self.D[key]


            self.close_widgets()
            with self.output_widget:
                clear_output(wait=True)
                print(f'-> Removed these variables: {self.selected_options}')

    def on_exit_button_click(self, change):
        if change['new']:
            self.close_widgets()
            with self.output_widget:
                clear_output(wait=True)
                print(' -> Exited without changing anything.')

    def close_widgets(self):
        self.checkbox_columns.close()
        self.text_widget.close()
        self.hbox_buttons.close()


#########################################################################


def threshold_edit(d, variables):
    """

    Docstring needs updating
    Interactive tool for threshold editing of a variable in a dataset.

    Parameters:
    - d (xarray.Dataset): The input dataset containing the variable to be
                          threshold-edited.


    Usage:
    Call this function with the dataset as an argument to interactively
    set threshold values and visualize the impact on the data.

    Returns: None
    """


    # Check that we are in a notebook and with the ipympl backend..
    # (raise a warning otherwise)
    internals.check_interactive()

    def get_slider_params(d, variable):
        """
        Helper function to calculate slider parameters based on the variable's
        data range.

        A little clunky as we have to deal with floating point errors to get
        nice output.

        Also used to get order of magnitude, so we output the order of
        magnitude of the step.

        Parameters:
        - d (xarray.Dataset): The input dataset.
        - variable (str): The variable for which slider parameters are
                          calculated.

        Returns:
        Tuple (float, float, float, int): Lower floor, upper ceil, step,
                                          order of magnitude step.
        """
        upper = float(d[variable].max())
        lower = float(d[variable].min())

        range = upper-lower
        oom_range = order_of_magnitude(range)
        oom_step = oom_range-2
        step = 10**oom_step

        lower_floor = np.round(np.floor(lower*10**(-oom_step))
                               * 10**(oom_step), -oom_step)
        upper_ceil = np.round(np.ceil(upper*10**(-oom_step))
                              * 10**(oom_step), -oom_step)

        return lower_floor, upper_ceil, step, oom_step



    # Dropdown menu for variable selection
    variable_dropdown = widgets.Dropdown(
        options=variables,
        value=variables[0],
        description='Variable:'
    )

    # Function to update plots based on variable, xvar, and max depth selection
    def update_plots(min_value, max_value, variable):
        """
        Helper function to update plots based on variable, min, and max values.

        Parameters:
        - min_value (float): Minimum threshold value.
        - max_value (float): Maximum threshold value.
        - variable (str): The variable being threshold-edited.

        Returns:
        None
        """
        try:
            previous_fig = plt.gcf()
            plt.close(previous_fig)
        except:
            pass

        fig = plt.figure(figsize=(6, 3))
        ax0 = plt.subplot2grid((1, 1), (0, 0))
        fig.canvas.header_visible = False  # Hide the figure header

        var_range = (np.nanmin(d[variable].values),
                     np.nanmax(d[variable].values))
        var_span = var_range[1] - var_range[0]

        hist_all = ax0.hist(d[variable].values.flatten(), bins=100,
                            range= var_range, color='tab:orange',
                            alpha=0.7, label='Distribution outside range')

        condition = ((d[variable] >= min_value)
                    & (d[variable] <= max_value))
        d_reduced = d.copy()
        d_reduced[variable] = d_reduced[variable].where(condition)

        # Count non-nan values in each dataset
        count_valid_d = int(d[variable].count())
        count_valid_d_reduced = int(d_reduced[variable].count())

        # Calculate the number of points that would be dropped by the
        # threshold cut
        points_cut = count_valid_d - count_valid_d_reduced
        points_pct = points_cut / count_valid_d * 100

        ax0.set_title(
            f'Histogram of {variable}', fontsize = 10)

        ax0.hist(d_reduced[variable].values.flatten(), bins=100,
                 range=var_range, color='tab:blue', alpha=1,
                 label='Distribution inside range')

        ax0.set_xlabel(f'[{d[variable].units}]')
        ax0.set_ylabel('Frequency')
        var_span = (np.nanmax(d[variable].values)
                    - np.nanmin(d[variable].values))
        ax0.set_xlim(np.nanmin(d[variable].values) - var_span * 0.05,
                      np.nanmax(d[variable].values) + var_span * 0.05)

        # Vertical lines for range values
        ax0.axvline(x=min_value, color='k', linestyle='--', label='Min Range')
        ax0.axvline(x=max_value, color='k', linestyle=':', label='Max Range')
        ax0.legend()

        # Display the updated plot
        clear_output(wait=True)
        plt.tight_layout()
        plt.show()

        # Update the cut information text
        update_cut_info_text(variable, points_cut, points_pct)



    def apply_cut(_):
        """
        Apply the threshold cut to the selected variable. Makes suitable
        changes to the metadata (variable attributes)

        Parameters:
        - _: Unused parameter.

        Returns:
        None
        """

        # Check that we in a notebook and with the ipympl backend..
        # (raise a warning otherwise)
        internals.check_interactive()

        variable = variable_dropdown.value

        lower_floor, upper_ceil, step, oom_step = get_slider_params(d, variable)

        min_value = np.round(np.floor(min_slider.value*10**(-oom_step))
                                     *10**(oom_step), -oom_step)
        max_value = np.round(np.floor(max_slider.value*10**(-oom_step))
                                     *10**(oom_step), -oom_step)

        var_max = float(d[variable].max())
        var_min = float(d[variable].min())
        unit = d[variable].units

        if unit == '1':
            unit=''

        if min_value<=var_min and max_value>=var_max:
            thr = None
        elif min_value<=var_min and max_value<var_max:
            thr = True
            thr_str = f'Rejected all data above {max_value} {unit}.'
            d[variable].attrs['valid_max'] = max_value
        elif min_value>var_min and max_value>=var_max:
            thr = True
            thr_str = f'Rejected all data below {min_value} {unit}.'
            d[variable].attrs['valid_min'] = min_value
        elif min_value>var_min and max_value<var_max:
            thr = True
            thr_str = ('Rejected all data outside ('
                       f'{min_value}, {max_value}) {unit}.')
            d[variable].attrs['valid_max'] = max_value
            d[variable].attrs['valid_min'] = min_value

        if thr:
            # Count non-nan values in the dataset
            count_valid_before = int(d[variable].count())

            d_thr = threshold(ds=d, variable = variable,
                        max_val = max_value, min_val = min_value)
            d[variable] = d_thr[variable]
            if 'PROCESSING' in d:
                d['PROCESSING'] = d_thr['PROCESSING']
                print(d_thr['PROCESSING'])
            # Count non-nan values in the dataset
            count_valid_after = int(d[variable].count())

            # Calculate the number of points that would be dropped by the
            # threshold cut
            points_cut = count_valid_before - count_valid_after
            points_pct = points_cut / count_valid_before * 100


            # Update plots
            update_plots(min_value, max_value, variable)


    # Create a button to minimize the plot
    close_button = widgets.Button(description="Exit",
                                  style={'button_color': '#FF9999'})

    def close_plot(_):
        # Resize the figure to 0 and close it
        fig = plt.gcf()
        fig.set_size_inches(0, 0)
        widgets_collected.close()
        plt.close(fig)

    # Function to reset sliders
    def reset_sliders(_):
        min_slider.value = min_slider.min
        max_slider.value = max_slider.max
        update_plot_sliders(None)


    # Determine the range of values in the dataset
    value_range = (d[variable_dropdown.value].max()
                   - d[variable_dropdown.value].min())

    # Calculate a suitable step size that includes whole numbers
    step_size = max(1, np.round(value_range / 100, 2))  # Adjust 100 as needed

    # Range sliders and input boxes
    slider_min, slider_max, slider_step, oom_step = get_slider_params(
        d, variable_dropdown.value)


    min_slider = widgets.FloatSlider(
        value=slider_min,
        min=slider_min,
        max=slider_max,
        step=slider_step,
        description=f'Lower cutoff value:',
        style={'description_width': 'initial'},
        layout={'width': '680px'}
    )

    max_slider = widgets.FloatSlider(
        value=slider_max,
        min=slider_min,
        max=slider_max,
        step=slider_step,
        description=f'Upper cutoff value:',
        style={'description_width': 'initial'},
        layout={'width': '680px'}
    )

    # Numeric input boxes for min and max values
    min_value_text = widgets.FloatText(
        value=d[variable_dropdown.value].min(),
        description='Min Value:',
        style={'description_width': 'initial'},
        layout={'width': '200px'}
    )

    max_value_text = widgets.FloatText(
        value=d[variable_dropdown.value].max(),
        description='Max Value:',
        style={'description_width': 'initial'},
        layout={'width': '200px'}
)

    reset_button = widgets.Button(description="Reset Sliders")
    reset_button.on_click(reset_sliders)

    apply_button = widgets.Button(
        description=f"Apply cut to {variable_dropdown.value}",
        style={'button_color': '#99FF99'})
    apply_button.on_click(apply_cut)

    # Text widget to display the cut information
    cut_info_text = widgets.Text(value='', description='', disabled=True,
                        layout={'width': '600px'}, style={'color': 'black'})

    # Function to update the cut information text
    def update_cut_info_text(variable, points_cut, points_pct):
        text = (f'This cut would reduce {variable} by {points_cut} data points'
                f' ({points_pct:.1f}%).')
        cut_info_text.value = text

    def update_apply_button_label(change):
        apply_button.description = f"Apply cut to {change.new}"

    # Observer for min value text box
    def update_min_slider(change):
        min_slider.value = min_value_text.value

    # Observer for max value text box
    def update_max_slider(change):
        max_slider.value = max_value_text.value

    # Observer for min slider
    def update_min_value(change):
        min_value_text.value = min_slider.value

    # Observer for max slider
    def update_max_value(change):
        max_value_text.value = max_slider.value


    # Observer for variable dropdown
    def update_sliders(change):
        variable = change.new
        slider_min, slider_max, slider_step, oom = get_slider_params(
            d, variable)

        max_slider.min, max_slider.max = -1e9, 1e9
        min_slider.min, min_slider.max = -1e9, 1e9

        try:
            max_slider.min = slider_min
            max_slider.max = slider_max
        except:
            max_slider.max = slider_max
            max_slider.min = slider_min
        try:
            min_slider.min = slider_min
            min_slider.max = slider_max
        except:
            min_slider.max = slider_max
            min_slider.min = slider_min

        min_slider.description = (
            f'Lower cutoff value (units: {d[variable_dropdown.value].units}):')
        max_slider.description = (
            f'Upper cutoff value (units: {d[variable_dropdown.value].units}):')

        min_slider.value = min_slider.min
        max_slider.value = max_slider.max
        min_slider.step = slider_step
        max_slider.step = slider_step

        # Rebuild the sliders every time the variable changes
        display(widgets_collected)


    variable_dropdown.observe(update_sliders, names='value')
    variable_dropdown.observe(update_apply_button_label, names='value')

    # Observer for sliders
    def update_plot_sliders(change):
        if change is not None:  # Check if change is None to prevent closing the figure on reset
            update_plots(min_slider.value, max_slider.value, variable_dropdown.value)

    min_slider.observe(update_plot_sliders, names='value')
    max_slider.observe(update_plot_sliders, names='value')

    # Set up observers
    min_slider.observe(update_min_value, names='value')
    max_slider.observe(update_max_value, names='value')
    min_value_text.observe(update_min_slider, names='value')
    max_value_text.observe(update_max_slider, names='value')

    close_button.on_click(close_plot)

    reset_button = widgets.Button(description="Reset Sliders")
    reset_button.on_click(reset_sliders)

    # Use interactive to update plots based on variable,
    # xvar, and max depth selection
    out = widgets.interactive_output(update_plots,
                {'min_value': min_slider, 'max_value': max_slider,
                 'variable': variable_dropdown})


    # Include the new widgets in the final layout
    widgets_collected = widgets.VBox([
        widgets.HBox([variable_dropdown, apply_button, reset_button, close_button]),
        widgets.HBox([min_slider, min_value_text]),
        widgets.HBox([max_slider, max_value_text]),
        cut_info_text, widgets.HBox([out, ])
    ])


    # Display the initial state of the title text
    display(widgets_collected)

