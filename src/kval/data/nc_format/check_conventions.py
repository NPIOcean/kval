from compliance_checker.runner import ComplianceChecker, CheckSuite
import os
import xarray as xr
from IPython.display import display, clear_output
import ipywidgets as widgets

def check_file(file):
    '''
    Use the IOOS compliance checker 
    (https://github.com/ioos/compliance-checker-web)
    to chek an nc file (CF and ACDD conventions).

    Can take a file path or an xr.Dataset as input
    '''
    # Load all available checker classes
    temp = False
    if isinstance(file, xr.Dataset):
        # Store a temp copy for checking
        temp_file = './temp.nc'
        file.to_netcdf(temp_file)
        D = xr.open_dataset(temp_file)
        file = temp_file
        temp = True

    check_suite = CheckSuite()
    check_suite.load_all_available_checkers()
    
    # Run cf and adcc checks
    path = file
    checker_names = ['cf', 'acdd']
    verbose = 0
    criteria = 'normal'

    return_value, errors = ComplianceChecker.run_checker(path,
                                                         checker_names,
                                                         verbose,
                                                         criteria,)
    
    if temp:
        os.remove(temp_file)



def check_file_with_button(file):
    '''
    (Wrapper for check_file() with a "close" button)

    Use the IOOS compliance checker 
    (https://github.com/ioos/compliance-checker-web)
    to chek an nc file (CF and ACDD conventions).

    Can take a file path or an xr.Dataset as input
    '''

    output_widget = widgets.Output()

    def on_button_click(b):
        with output_widget:
            # Clear the output widget
            clear_output(wait=True)

        # Remove the button and the output widget
        close_button.close()
        output_widget.close()

    close_button = widgets.Button(description="Close")
    close_button.on_click(on_button_click)

    display(widgets.VBox([close_button, output_widget, ]))

    # Your existing code
    with output_widget:
        check_file(file)
