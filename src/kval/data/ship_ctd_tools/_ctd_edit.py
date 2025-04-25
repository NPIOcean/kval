import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import ipywidgets as widgets
from IPython.display import display, clear_output
from kval.util import time, internals
from kval.data import edit, ctd
from kval.data.ship_ctd_tools import _ctd_tools


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

    # Check that we in a notebook and with the ipympl backend..
    # (raise a warning otherwise)

    def __init__(self, d, varnm, TIME_index):
        """
        Initialize the HandRemovePoints instance.

        Parameters:
        - d (xarray.Dataset): The dataset containing the data.
        - varnm (str): The name of the variable to visualize and edit.
        - station (str): The name of the station.
        """
        internals.check_interactive()

        if varnm not in d.data_vars:
            raise Exception(f'Invalid variable ("{varnm}")')

        self.TIME_index = TIME_index

        self.varnm = varnm
        self.d = d.copy(deep=True)

        self.var_data = d.isel(TIME=TIME_index)[varnm]
        if 'STATION' in d.keys():
            self.station = d.isel(TIME=TIME_index).STATION.item()
        else:
            self.station = 'N/A'
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
        station_time_string = time.convert_timenum_to_datetime(self.d.TIME.values[TIME_index], d.TIME.units)
        self.fig.canvas.header_visible = False  # Hide the figure header
        self.ax.set_title(f'Station: {self.station}: {station_time_string}')
        plt.tight_layout()


        # Use interactive_output to create interactive controls
        self.es = RectangleSelector(self.ax, self.onselect, interactive=True)

        # Can maybe cut these 4 (use remove_bool instead)?
        self.var_points_remove = np.array([])
        self.var_points_selected = np.array([])
        self.PRES_points_selected = np.array([])
        self.PRES_points_remove = np.array([])
        self.remove_bool = np.bool_(np.zeros(self.Npres))
        self.TF_indixes_selected = np.bool_(np.zeros(self.Npres))

        self.temp_label = 'Points to remove'
        self.remove_label = 'Selected points to remove'

        # Add widget buttons
        self.button_apply_var = widgets.Button(description=f"Exit and apply to {varnm}")
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
        self.contains_TF_ = rectangle.contains_points(np.vstack([self.var_data, self.PRES]).T)

        self.var_points_selected = np.concatenate([self.var_points_selected,
                                                self.var_data[self.contains_TF_]])
        self.PRES_points_selected = np.concatenate([self.PRES_points_selected,
                                                self.PRES[self.contains_TF_]])
        self.TF_indixes_selected = np.bool_(self.TF_indixes_selected + self.contains_TF_)

        try:
            self.temp_scatter.remove()
            plt.draw()
        except:
            pass

        self.temp_scatter = self.ax.scatter(self.var_points_selected,
                        self.PRES_points_selected, color='b', label=self.temp_label)
        self.ax.legend()
        self.TF_indixes_selected[self.contains_TF_] = True
        plt.draw()

    def remove_selected(self, button):
        """
        Handle the removal of selected points.

        Parameters:
        - button: The button click event.
        """
        # Check that we in a notebook and with the ipympl backend..
        # (raise a warning otherwise)

        internals.check_interactive()

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
        self.remove_bool[self.TF_indixes_selected] = True
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
        self.TF_indixes_selected = np.bool_(len())

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
        time_loc = dict(TIME=self.d['TIME'].isel(TIME=self.TIME_index))
        # Set remove-flagged indices to NaN

        ## HERE: CALL XR FUNCTION AND PRODUCE EXACT RECORD!
        #self.d[self.varnm].loc[time_loc] = np.where(self.remove_bool,
         #                           np.nan, self.d[self.varnm].loc[time_loc])

        self.remove_inds = np.where(self.remove_bool)[0]
        self.d = edit.remove_points_profile(self.d, self.varnm, self.TIME_index, self.remove_inds)

        # If we have a PROCESSING field:
        if hasattr(self.d, 'PROCESSING'):
            prof_num = self.TIME_index + 1
            N_profs = self.d.sizes['TIME']

            self.d.PROCESSING.attrs['post_processing'] += (
                f'Manually edited out (->NaN) the following points from profile {prof_num}/{N_profs} (STATION= {self.station}): '
                f'\n{self.remove_inds}\n')

            self.d.PROCESSING.attrs['python_script'] += (
                f'\n\n# Manually removing points from the {self.varnm} variable (from the TIME/STATION index {self.TIME_index}):'
                f'\nds = data.edit.remove_points_profile(ds, "{self.varnm}", {self.TIME_index}, [{", ".join(map(str, self.remove_inds))}])')


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
    """

    # Check that we in a notebook and with the ipympl backend..
    # (raise a warning otherwise)
    internals.check_interactive()

    vars = list(D.data_vars.keys())

    var_buttons = widgets.RadioButtons(
        options=vars,
        description='Apply offset to:',
        disabled=False
    )

    station_buttons = widgets.ToggleButtons(
        options=['All stations', 'Single station →'],
        description='Apply offset to:',
        disabled=False,
        button_style='',
        tooltips=['Apply an offset to every single profile in the dataset',
                  'Apply an offset only to the profile from this particular station']
    )

    value_textbox = widgets.Text(
        placeholder='Enter a numerical value',
        disabled=False
    )
    value_label = widgets.Label(value='Value of offset:')
    value_widget = widgets.VBox([value_label, value_textbox])

    # Dropdown for selecting a single profile
    station_dropdown = widgets.Dropdown(
        options=list(D.STATION.values) if 'STATION' in D else [],
        value=D.STATION.values[0] if 'STATION' in D else None,
        disabled=False,
        style={'description_width': 'initial'}
    )

    # Align items to the flex-end to move the dropdown to the bottom of the HBox
    hbox_layout = widgets.Layout(align_items='flex-end')

    # Create an HBox with station_buttons and station_dropdown
    hbox_station = widgets.HBox([station_buttons, station_dropdown], layout=hbox_layout)

    apply_button = widgets.ToggleButton(
        value=False,
        description='Apply offset',
        disabled=False,
        button_style='success'
    )

    exit_button = widgets.ToggleButton(
        value=False,
        description='Exit',
        disabled=False,
        button_style='danger'
    )

    output_widget = widgets.Output()

    # Function to update the description of value_textbox
    def update_description(change):
        selected_variable = change['new']
        units = D[selected_variable].attrs.get('units', 'unknown')
        value_label.value = f'Value of offset ({units}):'

    def on_apply_button_click(change):
        if change['new']:
            if value_textbox.value:
                varnm_sel = var_buttons.value
                offset_value = float(value_textbox.value)
                units = D[varnm_sel].attrs.get('units', 'unknown')

                if station_buttons.value == 'Single station →':
                    station_sel = station_dropdown.value
                    station_string = f'station {station_sel}'
                    time_at_station = D['TIME'].where(D['STATION'] == station_sel, drop=True).values[0]
                    D[varnm_sel].loc[{'TIME': time_at_station}] += offset_value

                else:
                    station_string = 'all stations'
                    D_offset = ctd.offset(D.copy(), varnm_sel, offset_value)
                    D[varnm_sel] = D_offset[varnm_sel]
                    D['PROCESSING'] = D_offset['PROCESSING']


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
                    print('No offset value was entered -> Exited without changing anything.')

    def on_exit_button_click(change):
        if change['new']:
            var_buttons.close()
            hbox_station.close()
            hbox_enter_value.close()

            with output_widget:
                clear_output(wait=True)
                print('-> Exited without changing anything.')

    # Attach the update_description function to the observe method of var_buttons
    var_buttons.observe(update_description, names='value')

    apply_button.observe(on_apply_button_click, names='value')
    exit_button.observe(on_exit_button_click, names='value')

    # Create an HBox with value_widget, apply_button, and exit_button
    hbox_enter_value = widgets.HBox([value_widget, apply_button, exit_button], layout=hbox_layout)

    display(var_buttons)
    display(hbox_station)
    display(hbox_enter_value)
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


        # Check that we in a notebook and with the ipympl backend..
        # (raise a warning otherwise)
        internals.check_interactive()

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

                self.close_widgets()
                with self.output_widget:
                    clear_output(wait=True)

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

