import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import ipywidgets as widgets
from IPython.display import display, clear_output
from oceanograpy.util import time

class hand_remove_points:
    def __init__(self, d, varnm, station):

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
        try:
            self.temp_scatter.remove()
            plt.draw()
        except:
            pass
        self.PRES_points_selected = np.array([])
        self.var_points_selected = np.array([])


    def start_over_selection(self, button):
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
        # Apply to this variable

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
        self.close_everything()
        
        with self.output_widget:
            clear_output(wait=True)
           # print(f'APPLIED TO DATASET - Removed {self.points_removed} POINTS')
            print(f'EXITED WITHOUT CHANGING ANYTHING')

    def close_everything(self):
        fig = plt.gcf()
        fig.set_size_inches(0, 0)
        self.widgets_all.close()
        plt.close(fig)
            