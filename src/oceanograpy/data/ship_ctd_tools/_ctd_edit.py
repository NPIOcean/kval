import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import ipywidgets as widgets
from IPython.display import display, clear_output
from oceanograpy.util import time
from oceanograpy.data.ship_ctd_tools import _ctd_tools
from oceanograpy.calc.numbers import order_of_magnitude

###########################

class hand_remove_points:
    """
    A class for interactive removal of data points from CTD profiles 

    Parameters:
    - d (xarray.Dataset): The dataset containing the CTD data.
    - varnm (str): The name of the variable to visualize and edit (e.g. 'TEMP1', 'CHLA').
    - station (str): The name of the station (E.g. '003', '012_01', 'AT290', 'StationA').

    Example usage that will let you hand edit the profile of "TEMP1" from station "StationA":
    ```python
    HandRemovePoints(my_dataset, 'TEMP1', 'StationA')
    ```
    Note: Use the interactive plot to select points for removal, then click the corresponding buttons for actions.
    """
    def __init__(self, d, varnm, station):
        """
        Initialize the HandRemovePoints instance.

        Parameters:
        - d (xarray.Dataset): The dataset containing the data.
        - varnm (str): The name of the variable to visualize and edit.
        - station (str): The name of the station.
        """
        if station not in d.STATION:
            raise Exception(f'Invalid station ("{station}")')
        if varnm not in d.data_vars:
            raise Exception(f'Invalid variable ("{varnm}")')
        
        # Find the index where d.STATION == station
        self.station_index = (d['STATION'] == station).argmax(dim='TIME').item()
        
        self.varnm = varnm
        self.d = d
        self.var_data = d.isel(TIME=self.station_index)[varnm]
        self.PRES = d.PRES
        self.Npres = len(d.PRES) 
        
        self.fig, self.ax = plt.subplots()


        line, = self.ax.plot(self.var_data, self.PRES)
        point = self.ax.plot(self.var_data, self.PRES, '.k', zorder=3)
        picks = np.array([])
        xlims = self.ax.get_xlim()
        self.ax.set_xlim(xlims)
        self.ax.invert_yaxis()
        self.ax.set_xlabel(f'{varnm} [{self.d[varnm].units}]')
        self.ax.set_ylabel(f'PRES [{self.PRES.units}]')
        self.ax.grid()
        station_time_string = time.convert_timenum_to_datetime(self.d.TIME.values[self.station_index], d.TIME.units)
        self.fig.canvas.header_visible = False  # Hide the figure header
        self.ax.set_title(f'Station: {station}: {station_time_string}')
        plt.tight_layout()


        # Use interactive_output to create interactive controls
        self.es = RectangleSelector(self.ax, self.onselect, interactive=True)

        # Can maybe cut these 4 (use remove_bool instead)?
        self.var_points_remove = np.array([])
        self.var_points_selected = np.array([])
        self.PRES_points_selected = np.array([])
        self.PRES_points_remove = np.array([])
        self.remove_bool = np.bool_(np.zeros(self.Npres))

        
        self.temp_label = 'Points to remove'
        self.remove_label = 'Selected points to remove'

        # Add widget buttons
        self.button_apply_var = widgets.Button(description=f"Exit and apply to {varnm}")
        self.button_apply_var.on_click(self.exit_and_apply_var)
        self.button_apply_var.layout.width = '200px' 

        self.button_exit_nochange = widgets.Button(description=f"Discard and exit")
        self.button_exit_nochange.on_click(self.exit_and_discard)
        self.button_exit_nochange.layout.width = '200px' 

        self.text_widget = widgets.HTML(value="Drag a rectangle to select points  ")
        
        self.button_remove = widgets.Button(description="Remove selected")
        self.button_remove.on_click(self.remove_selected)
        self.button_remove.style.button_color = '#FFB6C1'  # You can use any valid CSS color

        self.button_forget = widgets.Button(description="Forget selection")
        self.button_forget.on_click(self.forget_selection)
        self.button_forget.style.button_color = 'lightblue'  # You can use any valid CSS color

        self.button_restart = widgets.Button(description="Start over")
        self.button_restart.on_click(self.start_over_selection)
        self.button_restart.style.button_color = 'yellow'  # You can use any valid CSS color


        self.buttons_container_1 = widgets.HBox([self.text_widget, self.button_apply_var, 
                                                self.button_exit_nochange])
        self.buttons_container_2 = widgets.HBox([
            self.button_remove, self.button_forget, self.button_restart])

        
        # Add an Output widget to capture the print statement
        self.output_widget = widgets.Output()

        self.widgets_all = widgets.VBox([self.buttons_container_1,  
                                         widgets.Output(), self.buttons_container_2,])

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
        self.contains_TF_ = rectangle.contains_points(np.vstack([self.var_data, self.PRES]).T)

        self.var_points_selected = np.concatenate([self.var_points_selected, 
                                                self.var_data[self.contains_TF_]])
        self.PRES_points_selected = np.concatenate([self.PRES_points_selected, 
                                                self.PRES[self.contains_TF_]])
        try:
            self.temp_scatter.remove()
            plt.draw()
        except:
            pass
            
        self.temp_scatter = self.ax.scatter(self.var_points_selected, 
                        self.PRES_points_selected, color='b', label=self.temp_label)
        self.ax.legend()
        plt.draw()

    def remove_selected(self, button):
        """
        Handle the removal of selected points.

        Parameters:
        - button: The button click event.
        """
        self.var_points_remove = np.concatenate(
            [self.var_points_remove, self.var_points_selected])
        self.PRES_points_remove = np.concatenate(
            [self.PRES_points_remove, self.PRES_points_selected])

        try:
            self.remove_scatter.remove()
            plt.draw()
        except:
            pass
        
        self.remove_scatter = self.ax.scatter(self.var_points_remove, 
                        self.PRES_points_remove, color='r', 
                        label=self.remove_label, zorder=2)
        plt.draw()
        self.remove_label = None
        self.var_points_selected = np.array([])
        self.PRES_points_selected = np.array([])
       # self.var_data_return[self.contains_TF_] = np.nan
        self.remove_bool[self.contains_TF_] = True
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
        self.PRES_points_selected = np.array([])
        self.var_points_selected = np.array([])


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
        self.PRES_points_selected = np.array([])
        self.PRES_points_remove = np.array([])
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

        # Indexer used to access this specific profile
        time_loc = dict(TIME=self.d['TIME'].isel(TIME=self.station_index))
        # Set remove-flagged indices to NaN
        self.d[self.varnm].loc[time_loc] = np.where(self.remove_bool, 
                                    np.nan, self.d[self.varnm].loc[time_loc])
        
        # Count how many points we removed
        self.points_removed = np.sum(self.remove_bool)

        # Add info as a variable attribute
        if 'manual_editing' in self.d[self.varnm].attrs.keys(): 
            previous_edits = int(self.d[self.varnm].attrs['manual_editing'].split()[0])
            total_edits = previous_edits + self.points_removed
            self.d[self.varnm].attrs['manual_editing'] = (
               f'{total_edits} data points have been removed '
                'from this variable based on visual inspection.') 
        else:
            self.d[self.varnm].attrs['manual_editing'] = (
               f'{self.points_removed} data points have been ' 
                'removed from this variable based on visual inspection.') 
            if self.points_removed==1:
                self.d[self.varnm].attrs['manual_editing']  = (
                    self.d[self.varnm].attrs['manual_editing'].replace(
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
           # print(f'APPLIED TO DATASET - Removed {self.points_removed} POINTS')
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

def apply_offset(D):
    """
    Apply an offset to selected variables in a given xarray Dataset.

    Parameters:
    - D (xarray.Dataset): The dataset to which the offset will be applied.

    Displays interactive widgets for selecting the variable, choosing the application scope
    (all stations or a single station), entering the offset value, and applying or exiting.

    The offset information is stored as metadata in the dataset's variable attributes.

    Examples:
    ```python
    apply_offset(my_dataset)
    ```

    Note: This function utilizes IPython widgets for interactive use within a Jupyter environment.
    """
    vars = _ctd_tools._get_profile_variables(D)

    var_buttons = widgets.RadioButtons(
        options=vars,
        description='Apply offset to:',
        disabled=False
    )
    
    station_buttons = widgets.ToggleButtons(
    options=['All stations', 'Single station →', ],
    description='Apply offset to:',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltips=['Apply an offset only to the profile from this particular station', 
        'Apply an offset to every single profile in the dataset'],
    )

    value_textbox = widgets.Text(
    #value='Enter a value',
    placeholder='Enter a numerical value',
    disabled=False   
    )
    value_label = widgets.Label(value=f'Value of offset:')
    value_widget = widgets.VBox([value_label, value_textbox])


    # Function to update the description of value_textbox
    def update_description(change):
        selected_variable = change['new']
        # Assuming you have a function to get units based on the selected variable
        units = D[var_buttons.value].units
        value_label.value = f'Value of offset ({units}):'

    # Attach the update_description function to the observe method of var_buttons
    var_buttons.observe(update_description, names='value')

    # Increase the width of the description
    value_textbox.style.description_width = '150px'  # Adjust the width as needed
    


    # Dropdown for selecting a single profile
    station_dropdown = widgets.Dropdown(
        options=list(D.STATION.values),  # Assuming keys are profile names
        value=D.STATION.values[0],  # Set default value
        disabled=False,
        style={'description_width': 'initial'},  # Adjust width
    )
    
  # Align items to the flex-end to move the dropdown to the bottom of the HBox
    hbox_layout = widgets.Layout(align_items='flex-end')
    
    # Create an HBox with station_buttons and station_dropdown
    hbox_station = widgets.HBox([station_buttons, station_dropdown], layout=hbox_layout)

    apply_button = widgets.ToggleButton(
        value=False,
        description='Apply offset',
        disabled=False,
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
       # tooltip='Description',
       # icon='check' # (FontAwesome names without the `fa-` prefix)
        )

    exit_button = widgets.ToggleButton(
        value=False,
        description='Exit',
        disabled=False,
        button_style='danger', # 'success', 'info', 'warning', 'danger' or ''
       # tooltip='Description',
       # icon='check' # (FontAwesome names without the `fa-` prefix)
        )
    
    def on_apply_button_click(change):
        if change['new']:
            if value_textbox.value != '':
    
                varnm_sel = var_buttons.value
                offset_value = float(value_textbox.value)
                units = D[varnm_sel].units
                var_attrs = D[varnm_sel].attrs
                
                if station_buttons.value == 'Single station →':
                    station_sel = station_dropdown.value
                    station_string = f'station {station_sel}'
                else:
                    station_string = 'all stations'
                    D[varnm_sel] = D[varnm_sel] + offset_value
    
                D[varnm_sel].attrs = var_attrs

                offset_metadata = f"Applied offset of {offset_value} [{units}] ({station_string})"
                
                if 'applied_offset' in D[varnm_sel].attrs:
                    D[varnm_sel].attrs['applied_offset'] += f'\n{offset_metadata}'
                else:
                    D[varnm_sel].attrs['applied_offset'] = offset_metadata
                    
                var_buttons.close()
                hbox_station.close()
                hbox_enter_value.close()
    
                with output_widget:
                    clear_output(wait=True)
                    print(f'-> {varnm_sel}: {offset_metadata}')

            else:
                var_buttons.close()
                hbox_station.close()
                hbox_enter_value.close()
    
                with output_widget:
                    clear_output(wait=True)
                    print(f' No offset value was entered -> Exited without changing anything. ')
                    
    def on_exit_button_click(change):
        if change['new']:
            
            var_buttons.close()
            hbox_station.close()
            hbox_enter_value.close()

            with output_widget:
                clear_output(wait=True)
                print(f' -> Exited without changing anything. ')

    apply_button.observe(on_apply_button_click, names='value')
    exit_button.observe(on_exit_button_click, names='value')

    # Create an HBox with station_buttons and station_dropdown
    hbox_enter_value = widgets.HBox([value_widget, apply_button, exit_button], layout=hbox_layout)


    display(var_buttons)
    display(hbox_station)
    display(hbox_enter_value)

    # Add an Output widget to capture the print statement when we are done
    output_widget = widgets.Output()
    display(output_widget)

#########################################################################


class drop_stations_pick:
    '''

    NOT FINISHED!!
    Interactive class for dropping selected time points from an xarray Dataset based on the value of STATION(TIME).

    Parameters:
    - D (xarray.Dataset): The dataset from which time points will be dropped.

    Displays an interactive widget with checkboxes for each time point, showing the associated STATION.
    Users can select time points to remove. The removal is performed by clicking the "Drop time points"
    button. The removed time points are also printed to the output.

    Examples:
    ```python
    drop_time_points_pick(my_dataset)
    ```

    Note: This class utilizes IPython widgets for interactive use within a Jupyter environment.
    '''
    def __init__(self, D):
        self.D = D
        self.selected_time_points = []
        self.max_stations_per_row = 3
        self.checkbox_spacing = '0px'  # Adjust this value to control spacing

        # Create Checkbox widgets with STATION labels
        self.checkbox_widgets = [widgets.Checkbox(description=str(D['STATION'].sel(TIME=time_point).item()), indent=False) for time_point in D['TIME'].values]

        # Calculate the number of checkboxes per row
        checkboxes_per_row = min(self.max_stations_per_row, len(self.checkbox_widgets))

        # Calculate the number of rows
        num_rows = (len(self.checkbox_widgets) // checkboxes_per_row) + (1 if len(self.checkbox_widgets) % checkboxes_per_row != 0 else 0)

        # Create VBox widgets for each column with adjusted spacing
        columns = [widgets.VBox(self.checkbox_widgets[i * num_rows:(i + 1) * num_rows], layout=widgets.Layout(margin=f'0 0 0 {self.checkbox_spacing}')) for i in np.arange(checkboxes_per_row)]

        # Arrange columns in an HBox
        checkboxes_hbox = widgets.HBox(columns, layout=widgets.Layout(margin='0px'))

        # Text widget
        self.text_widget = widgets.HTML(value="Select stations to remove:")

        # Create an Output widget to capture the print statement when done
        self.output_widget = widgets.Output()

        # Create buttons
        self.remove_button = widgets.ToggleButton(value=False, description='Drop stations from dataset', button_style='success')


        # Set the width of the buttons
        button_width = '200px'  # Adjust the width as needed
        self.remove_button.layout.width = button_width
        self.exit_button = widgets.ToggleButton(value=False, description='Exit', button_style='danger')

        # Attach button event handlers
        self.remove_button.observe(lambda change: self.on_remove_button_click(change, D), names='value')
        self.exit_button.observe(self.on_exit_button_click, names='value')

        # Layout for buttons
        hbox_layout = widgets.Layout(align_items='flex-end')
        self.hbox_buttons = widgets.HBox([self.remove_button, self.exit_button], layout=hbox_layout)

        # Function to handle checkbox changes
        def handle_checkbox_change(change):
            self.selected_time_points = [time_point for checkbox, time_point in zip(self.checkbox_widgets, D['TIME'].values) if checkbox.value]
            print(f"Selected time points: {', '.join(map(str, self.selected_time_points))}")

        # Attach the handle_checkbox_change function to the observe method of each checkbox
        for checkbox in self.checkbox_widgets:
            checkbox.observe(handle_checkbox_change, names='value')

        # Display widgets
        display(self.text_widget)
        display(checkboxes_hbox)
        display(self.hbox_buttons)
        display(self.output_widget)

    def on_remove_button_click(self, change, D):
        if change['new']:
            stations_removed = []
            
            for time_point in self.selected_time_points:
                for variable in D.variables:
                    if 'TIME' in D[variable].dims:
                        stations_removed.append(f"{variable}_"
                                f"{str(self.D.STATION.sel(TIME=self.D['TIME'] == time_point).values[0])}")
            
            # Perform the removal after collecting all items to remove
            for time_point in self.selected_time_points:
                for variable in D.variables:
                    if 'TIME' in D[variable].dims:
                        index_to_remove = np.where(D['TIME'].values == time_point)[0]
                        
                        # Remove the corresponding values from the variable
                        del D[variable][index_to_remove]

            self.close_widgets()
            with self.output_widget:
                clear_output(wait=True)
               # print(f'-> Removed stations: {", ".join(stations_removed)}')

                self.close_widgets()
                with self.output_widget:
                    clear_output(wait=True)
                   # print(f'-> Removed stations: {", ".join(stations_removed)}')

    def on_exit_button_click(self, change):
        if change['new']:
            self.close_widgets()
            with self.output_widget:
                clear_output(wait=True)
                print(' -> Exited without changing anything.')

    def close_widgets(self):
        self.text_widget.close()
        self.hbox_buttons.close()
        #self.output_widget.close()
        for checkbox in self.checkbox_widgets:
            checkbox.close()


#########################################################################


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
    def __init__(self, D):
        self.D = D
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


def threshold_edit(d):
    """
    Interactive tool for threshold editing of a variable in a dataset.

    Parameters:
    - d (xarray.Dataset): The input dataset containing the variable to be threshold-edited.

    Usage:
    Call this function with the dataset as an argument to interactively set threshold values and visualize the impact on the data.

    Returns:
    None
    """
    
    def get_slider_params(d, variable):
        """
        Helper function to calculate slider parameters based on the variable's data range.
        A little clunky as we have to deal with floating point errors to get nice output.

        Also used to get order of magnitude, so we output the order of magnitude of the step. 

        Parameters:
        - d (xarray.Dataset): The input dataset.
        - variable (str): The variable for which slider parameters are calculated.

        Returns:
        Tuple (float, float, float, int): Lower floor, upper ceil, step, order of magnitude step.
        """
        upper = float(d[variable].max())
        lower = float(d[variable].min())

        range = upper-lower
        oom_range = order_of_magnitude(range)
        oom_step = oom_range-2
        step = 10**oom_step
        
        lower_floor = np.round(np.floor(lower*10**(-oom_step))*10**(oom_step), -oom_step)
        upper_ceil = np.round(np.ceil(upper*10**(-oom_step))*10**(oom_step), -oom_step)
        
        return lower_floor, upper_ceil, step, oom_step


    # Get the list of available variables
    variable_names = _ctd_tools._get_profile_variables(d)

    # Dropdown menu for variable selection
    variable_dropdown = widgets.Dropdown(
        options=variable_names,
        value=variable_names[0],
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

        var_range = (np.nanmin(d[variable].values), np.nanmax(d[variable].values))
        var_span = var_range[1] - var_range[0]

        hist_all = ax0.hist(d[variable].values.flatten(), bins=100, range=var_range, color='tab:orange',
                            alpha=0.7, label='Distribution outside range')

        d_reduced = d.where((d[variable] >= min_value) & (d[variable] <= max_value))

        # Count non-nan values in each dataset
        count_valid_d = int(d[variable].count())
        count_valid_d_reduced = int(d_reduced[variable].count())

        # Calculate the number of points that would be dropped by the threshold cut
        points_cut = count_valid_d - count_valid_d_reduced
        points_pct = points_cut / count_valid_d * 100

        ax0.set_title(
            f'Histogram of {variable}', fontsize = 10)

        ax0.hist(d_reduced[variable].values.flatten(), bins=100,
                 range=var_range, color='tab:blue', alpha=1,
                 label='Distribution inside range')

        ax0.set_xlabel(f'[{d[variable].units}]')
        ax0.set_ylabel('Frequency')
        var_span = np.nanmax(d[variable].values) - np.nanmin(d[variable].values)
        ax0.set_xlim(np.nanmin(d[variable].values) - var_span * 0.05, np.nanmax(d[variable].values) + var_span * 0.05)

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
        Apply the threshold cut to the selected variable. Makes suitable changes to the metadata (variable attributes)

        Parameters:
        - _: Unused parameter.

        Returns:
        None
        """
        variable = variable_dropdown.value

        lower_floor, upper_ceil, step, oom_step = get_slider_params(d, variable)
        
        min_value = np.round(np.floor(min_slider.value*10**(-oom_step))*10**(oom_step), -oom_step)  
        max_value = np.round(np.floor(max_slider.value*10**(-oom_step))*10**(oom_step), -oom_step)  
        
        var_max = float(d[variable].max())
        var_min = float(d[variable].min())
        unit = d[variable].units
        
        if unit == '1':
            unit=''
    
        if min_value<=var_min and max_value>=var_max:
            threshold = None
        elif min_value<=var_min and max_value<var_max:
            threshold = True
            thr_str = f'Rejected all data above {max_value} {unit}.'
            d[variable].attrs['valid_max'] = max_value
        elif min_value>var_min and max_value>=var_max:
            threshold = True
            thr_str = f'Rejected all data below {min_value} {unit}.'
            d[variable].attrs['valid_min'] = min_value
        elif min_value>var_min and max_value<var_max:
            threshold = True
            thr_str = f'Rejected all data outside ({min_value}, {max_value}) {unit}.'
            d[variable].attrs['valid_max'] = max_value
            d[variable].attrs['valid_min'] = min_value

        if threshold:
            # Count non-nan values in the dataset
            count_valid_before = int(d[variable].count())
    
            # Apply the cut
            d[variable] = d[variable].where((d[variable] >= min_value) & (d[variable] <= max_value), np.nan)
            
            # Count non-nan values in the dataset
            count_valid_after = int(d[variable].count())
    
            # Calculate the number of points that would be dropped by the threshold cut
            points_cut = count_valid_before - count_valid_after
            points_pct = points_cut / count_valid_before * 100

            # Add info to metadata variable
            count_str = f'This removed {points_cut} data points ({points_pct:.1f}%).'

            if 'threshold_editing' in d[variable].attrs.keys(): 
                d[variable].attrs['threshold_editing']+= f'\n{thr_str} {count_str}'
            else:
                d[variable].attrs['threshold_editing']= f'{thr_str} {count_str}'

            
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
    value_range = d[variable_dropdown.value].max() - d[variable_dropdown.value].min()
    
    # Calculate a suitable step size that includes whole numbers
    step_size = max(1, np.round(value_range / 100, 2))  # Adjust 100 as needed

    # Range sliders and input boxes
    slider_min, slider_max, slider_step, oom_step = get_slider_params(d, variable_dropdown.value)


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
    cut_info_text = widgets.Text(value='', description='', disabled=True, layout={'width': '600px'}, style={'color': 'black'})

    # Function to update the cut information text
    def update_cut_info_text(variable, points_cut, points_pct):
        text = f'This cut would reduce {variable} by {points_cut} data points ({points_pct:.1f}%).'
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
        slider_min, slider_max, slider_step, oom = get_slider_params(d, variable)

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
            
        min_slider.description = f'Lower cutoff value (units: {d[variable_dropdown.value].units}):'
        max_slider.description = f'Upper cutoff value (units: {d[variable_dropdown.value].units}):'

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

    # Use interactive to update plots based on variable, xvar, and max depth selection
    out = widgets.interactive_output(update_plots,
                                     {'min_value': min_slider, 'max_value': max_slider, 'variable': variable_dropdown})

   # widgets_collected = widgets.VBox([widgets.HBox([variable_dropdown, apply_button, reset_button]),
   #                                   widgets.HBox([min_slider]), max_slider, out, close_button])
    
    # Include the new widgets in the final layout
    widgets_collected = widgets.VBox([
        widgets.HBox([variable_dropdown, apply_button, reset_button, close_button]),
        widgets.HBox([min_slider, min_value_text]),
        widgets.HBox([max_slider, max_value_text]),
        cut_info_text, widgets.HBox([out, ])
    ])
    

    # Display the initial state of the title text
    display(widgets_collected)
