import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import ipywidgets as widgets
from IPython.display import display, clear_output
from oceanograpy.util import time
from oceanograpy.data.ship_ctd_tools import _ctd_tools

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
        self.v
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
            print(f"Selected options: {', '.join(self.selected_options)}")
        
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
