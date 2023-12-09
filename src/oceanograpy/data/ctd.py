'''
OCEANOGRAPY.CTD.PY

This should be what a base user interacts with.

NOTE: Need to look over and see what functions can be moved to other modules.
This should mainly contain functions *specific* to CTD profile data. Could
instead add some wrappers that use functions from nc_attrs.conventionalize.py,
ship_ctd.tools.py, for example.
'''

import xarray as xr
from oceanograpy.data.ship_ctd_tools import _ctd_tools as tools
from oceanograpy.data.ship_ctd_tools import _ctd_visualize as viz
from oceanograpy.data.ship_ctd_tools import _ctd_edit as edit

from oceanograpy.io import matfile

import re
from collections import Counter
from oceanograpy.util import time
import pandas as pd
from oceanograpy.data.nc_format import conventionalize, _standard_attrs, check_conventions
import os


## LOADING AND SAVING DATA

def ctds_from_cnv_dir(
    path: str,
    station_from_filename: bool = False,
    time_warnings: bool = True,
    verbose: bool = True
) -> xr.Dataset:
    """
    Create CTD datasets from CNV files in the specified path.

    Parameters:
    - path (str): Path to the CNV files.
    - station_from_filename (bool): Whether to extract station information from filenames.
    - time_warnings (bool): Enable/disable time-related warnings.
    - verbose: If False, suppress some prints output.
    Returns:
    - D (xarray.Dataset): Joined CTD dataset.
    """
    cnv_files = tools._cnv_files_from_path(path)
    number_of_cnv_files = len(cnv_files)
    if number_of_cnv_files==0:
        raise Exception(f'Did not find any .cnv files in the specified directory ("{path}").'
                        ' Is there an error in the path?')
    else:
        print(f'Found {number_of_cnv_files} .cnv files in  "{path}".')

    profile_datasets = tools._datasets_from_cnvlist(
        cnv_files, verbose = verbose)
    D = tools.join_cruise(profile_datasets,
        verbose=verbose)

    return D


def ctds_from_cnv_list(
    cnv_list: list,
    station_from_filename: bool = False,
    time_warnings: bool = True,
    verbose: bool = True
) -> xr.Dataset:
    """
    Create CTD datasets from CNV files in the specified path.

    Parameters:
    - path (str): List of CNV file paths.
    - station_from_filename (bool): Whether to extract station information from filenames.
    - time_warnings (bool): Enable/disable time-related warnings.
    - verbose: If False, suppress some prints output.
    Returns:
    - D (xarray.Dataset): Joined CTD dataset.
    """
    profile_datasets = tools._datasets_from_cnvlist(
        cnv_list, verbose = verbose)
    D = tools.join_cruise(profile_datasets,
        verbose=verbose)
    return D


def dataset_from_btl_dir(
    path: str,
    station_from_filename: bool = False,
    time_warnings: bool = True,
    verbose: bool = True
) -> xr.Dataset:
    """
    Create CTD datasets from CNV files in the specified path.

    Parameters:
    - path (str): Path to the CNV files.
    - station_from_filename (bool): Whether to extract station information from filenames.
    - time_warnings (bool): Enable/disable time-related warnings.
    - verbose: If False, suppress some prints output.
    Returns:
    - D (xarray.Dataset): Joined CTD dataset.
    """
    btl_files = tools._btl_files_from_path(path)
    profile_datasets = tools._datasets_from_btllist(
        btl_files, verbose = verbose)
    D = tools.join_cruise_btl(profile_datasets,
        verbose=verbose)

    return D



def make_publishing_ready(D, NPI = True, 
                retain_vars = ['TEMP1', 'CNDC1', 'PSAL1', 'CHLA1', 'PRES'],
                drop_vars = None, retain_all = False):
    '''
    Wrapper function:
    Various modifications to the metadata in order to make the dataset
    nicely formatted for publication.
    '''
    if retain_all:
        drop_vars = []

    D = drop_variables(D, retain_vars = retain_vars,
                       drop_vars = drop_vars, retain_all=retain_all)
    D = remove_numbers_in_names(D)
    D = conventionalize.add_standard_var_attrs_ctd(D)
    D = conventionalize.add_standard_glob_attrs_ctd(D, NPI = NPI, 
                                                    override = False)
    D = conventionalize.add_gmdc_keywords_ctd(D)
    D = conventionalize.add_range_attrs_ctd(D)

    D = reorder_attrs(D)

    return D


def quick_metadata_check(D,):
    # Consider moving to conventionalize.py?
    # (Or maybe keep this one CTD specific)

    print('--  QUICK METADATA CHECK --')
    print('NOTE: Not comprehensive! A true check is done on export to netcdf.')

    print('\n# GLOBAL #')

    ### GLOBAL
    attrs_dict_ref = _standard_attrs.global_attrs_ordered.copy()

    attrs_dict_ref.remove('date_created')
    attrs_dict_ref.remove('processing_level')

    for attr in attrs_dict_ref:
        if attr not in D.attrs:
            print(f'- Possibly missing {attr}')

    print('\n# VARIABLE #')

    ### VARIABLE
    attrs_dict_ref_var = _standard_attrs.variable_attrs_necessary

    for varnm in D.variables:
        if 'PRES' in D[varnm].dims:
            _attrs_dict_ref_var = attrs_dict_ref_var.copy()

            if varnm == 'CHLA':
                _attrs_dict_ref_var += [
                    'calibration_formula',
                    'coefficient_A',
                    'coefficient_B',]
            if varnm == 'PRES':
                _attrs_dict_ref_var += [
                    'axis',
                    'positive',]
                _attrs_dict_ref_var.remove('processing_level')
                _attrs_dict_ref_var.remove('QC_indicator')

            any_missing = False
            for var_attr in _attrs_dict_ref_var:
                if var_attr not in D[varnm].attrs:
                    print(f'- {varnm}: Possibly missing {var_attr}')
                    any_missing = True
            if not any_missing:
                    print(f'- {varnm}: OK')

    other_dict = {
        'LATITUDE':['units', 'standard_name', 'long_name', 'axis'],
        'LONGITUDE':['units', 'standard_name', 'long_name', 'axis'],
        'STATION':['cf_role', 'long_name'],
        'CRUISE':['long_name'],
        }
    for varnm, _attrs_dict_ref_var in other_dict.items():
        if varnm in D:
            any_missing = False
            for var_attr in _attrs_dict_ref_var:
                if var_attr not in D[varnm].attrs:
                    print(f'- {varnm}: Possibly missing {var_attr}')
                    any_missing = True
            if not any_missing:
                    print(f'- {varnm}: OK')


###########

def to_netcdf(D, path, file_name = None, convention_check = False, add_to_history = True):
    '''
    Export xarray Dataset to netCDF.
    Using the 'id' attribute as file name if file_name not specified
    (if that doesn't exist, use 'CTD_DATASET_NO_NAME').
    '''
    # Consider moving to a more general module?

    D = add_now_as_date_created(D)
    D = reorder_attrs(D)

    if file_name == None:
        try:
            file_name = D.id
        except:
            file_name = 'CTD_DATASET_NO_NAME'

    file_path = f'{path}{file_name}.nc'

    if add_to_history:
        now_time = pd.Timestamp.now().strftime('%Y-%m-%d')
        D.attrs['history'] = D.history + f'\n{now_time}: Creation of this netcdf file.'
        print(f'Updated history attribute. Current content:\n---')
        print(D.history)
        print('---')

    # Save the file          
    try:
        D.to_netcdf(file_path)
    
    # If the file already exists, we may get a permission error. If so, ask  
    # the user whether to delete the old file and override. 
    except PermissionError:
        user_input = input(f"The file {file_path} already exists. "
                           "Do you want to override it? (y/n): ")
    
        if user_input.lower() in ['yes', 'y']:
            # User wants to override the file
            os.remove(file_path)
            D.to_netcdf(file_path)
            print(f"File {file_path} overridden.")
        else:
            # User does not want to override
            print("Operation canceled. File not overridden.")

    print(f'Exported netCDF file as: {path}{file_name}.nc')

    if convention_check:
        print('Running convention checker:')
        check_conventions.check_file(file_path)


#############

def to_mat(D, outfile, simplify = False):
    """
    Convert the CTD data (xarray.Dataset) to a MATLAB .mat file.
      
    A field 'TIME_mat' with Matlab datenums is added along with the data. 

    Parameters:
    - D (xarray.Dataset): Input dataset to be converted.
    - outfile (str): Output file path for the MATLAB .mat file. If the path
      doesn't end with '.mat', it will be appended.
    - simplify (bool, optional): If True, simplify the dataset by extracting
      only coordinate and data variables (no metadata attributes). If False, 
      the matfile will be a struct containing [attrs, data_vars, coords, dims]. 
      Defaults to False.
      
    Returns:
    None: The function saves the dataset as a MATLAB .mat file.

    Example:
    >>> ctd.xr_to_mat(D, 'output_matfile', simplify=True)
    """

    matfile.xr_to_mat(D, outfile, simplify = simplify)

############

def to_csv(D, outfile):
    """
    Convert the CTD data (xarray.Dataset) to a human-readable .csv file.
    
    The file shows columnar data for all data paramaters for all stations.
    Statins are separated by a header with the station name/time/lat/lon.

    Parameters:
    - D (xarray.Dataset): Input dataset to be converted.
    - outfile (str): Output file path for the .csv file. If the path
      doesn't end with '.csv', it will be appended.

    Returns:
    None: The function saves the dataset as a .cnv file.

    Example:
    >>> ctd.to_csv(D, 'output_cnvfile', )
    """    
    prof_vars = ['PRES']
    
    for key in D.data_vars.keys():
        if  'TIME' in D[key].dims:
            if 'PRES' in D[key].dims:
                prof_vars += [key]

    if not outfile.endswith('.csv'):
        outfile += '.csv'
                
    with open(outfile, 'w') as f:
        for time_ in D.TIME.values:
            D_prof = D.sel(TIME=time_)
            time_str= time.datenum_to_timestamp(time_).strftime('%Y-%m-%d %H:%M:%S')
            print('#'*88, file=f)
            print(f'#####  {D_prof.STATION.values:<8} ###  {time_str}  ###  LAT: {D_prof.LATITUDE.values:<10}'
                  f' ### LON: {D_prof.LONGITUDE.values:<10} #####', file=f)
            print('#'*88 + '\n', file=f)
        
            D_pd = D_prof[prof_vars].to_pandas()
            D_pd = D_pd.drop('TIME', axis = 1)
            
            D_pd = D_pd.dropna(subset=D_pd.columns.difference(['PRES']), how='all')
            print(D_pd.to_csv(), file=f)


############

def metadata_to_txt(D, outfile):
    """
    Write metadata information from an xarray.Dataset to a text file.

    Parameters:
    - D (xarray.Dataset): The dataset containing metadata.
    - outfile (str): Output file path for the text file. If the path doesn't
      end with '.txt', it will be appended.

    Returns:
    None: The function writes metadata to the specified text file.

    Example:
    >>> metadata_to_txt(D, 'metadata_output')

    NOTE: This function is pretty general. Consider putting it somethere else!
    """

    # Ensure the output file has a '.txt' extension
    if not outfile.endswith('.txt'):
        outfile += '.txt'

    # Open the text file for writing
    with open(outfile, 'w') as f:
        # Create the file header based on the presence of 'id' attribute
        if hasattr(D, 'id'):
            file_header = f'FILE METADATA FROM: {D.id}'
        else:
            file_header = f'FILE METADATA'

        # Print the file header with formatting
        print('#'*80, file=f)
        print(f'####  {file_header:<68}  ####', file=f)
        print('#'*80, file=f)
        print('\n' + '#'*27, file=f)
        print('### GLOBAL ATTRIBUTES   ###', file=f)
        print('#'*27, file=f)
        print('', file=f)

        # Print global attributes
        for key, item in D.attrs.items():
            print(f'# {key}:', file=f)
            print(item, file=f)

        print('', file=f)
        print('#'*27, file=f)
        print('### VARIABLE ATTRIBUTES ###', file=f)
        print('#'*27, file=f)

        # Get all variable names (coordinates and data variables)
        all_vars = list(D.coords.keys()) + list(D.data_vars.keys())

        # Iterate through variables
        for varnm in all_vars:
            print('\n' + '-'*50, file=f)

            # Print variable name with indication of coordinate status
            if varnm in D.coords:
                print(f'{varnm} (coordinate)', file=f)
            else:
                print(f'{varnm}', file=f)

            print('-'*50, file=f)

            # Print variable attributes
            for key, item in D[varnm].attrs.items():
                print(f'# {key}:', file=f)
                print(item, file=f)


## APPLYING CORRECTIONS ETC

import xarray as xr
from typing import Optional

def calibrate_chl(
    D: xr.Dataset,
    A: float,
    B: float,
    chl_name_in: Optional[str] = 'CHLA1_fluorescence',
    chl_name_out: Optional[str] = None,
    verbose: Optional[bool] = True,
    remove_uncal: Optional[bool] = False
) -> xr.Dataset:
    """
    Apply a calibration to chlorophyll based on a fit based on water samples.

    CHLA -> A * CHLA_fluorescence + B

    Where CHLA_fluorescence is the name of the variable containing uncalibrated 
    chlorophyll from the instrument.

    Append suitable metadata.

    Parameters
    ----------
    D : xr.Dataset
        Dataset containing A, B: Linear coefficients based on fitting to chl samples.
    A : float
        Linear coefficient for calibration.
    B : float
        Linear coefficient for calibration.
    chl_name_in : str, optional
        Name of the variable containing uncalibrated chlorophyll from the instrument.
        Default is 'CHLA1_fluorescence'.
    chl_name_out : str, optional
        Name of the calibrated chlorophyll variable. If not provided, it is derived from
        chl_name_in. Default is None.
    verbose : bool, optional
        If True, print messages about the calibration process. Default is True.
    remove_uncal : bool, optional
        If True, remove the uncalibrated chlorophyll from the dataset. Default is False.

    Returns
    -------
    xr.Dataset
        Updated dataset with calibrated chlorophyll variable.
    """
    # Determine the output variable name for calibrated chlorophyll:
    # If we don't specify a variable name for the calibrated chlorophyll, the
    # default behaviour is to 
    # a) Use the uncalibrated chl name without the '_instr' suffix (if it 
    #    has one), or 
    # b) Use the uncalibrated chl name with the suffix '_cal' 
    if not chl_name_out:
        
        if '_instr' in chl_name_in or '_fluorescence' in chl_name_in:
            chl_name_out = chl_name_in.replace('_instr', '').replace('_fluorescence', '')
        else: 
            chl_name_out = f'{chl_name_in}_cal'


    # Create a new variable with the coefficients applied
    D[chl_name_out] = A * D[chl_name_in] + B
    D[chl_name_out].attrs = {key: item for key, item in D[chl_name_in].attrs.items()}

    # Add suitable attributes
    new_attrs = {
        'long_name': ('Chlorophyll-A concentration calibrated against'
                      ' water sample measurements'),
        'calibration_formula': 'chla_calibrated = A * chla_from_ctd + B',
        'coefficient_A': A,
        'coefficient_B': B,
        'comment':'No correction for near-surface fluorescence quenching '
                   '(see e.g. https://doi.org/10.4319/lom.2012.10.483) has been applied.'
    }

    for key, item in new_attrs.items():
        D[chl_name_out].attrs[key] = item

    # Remove the uncalibrated chla 
    if remove_uncal:
        remove_str = f' Removed uncalibrated Chl-A ("{chl_name_in}") from the dataset.'
        D = D.drop(chl_name_in)
    else:
        remove_str = ''

    # Print
    if verbose:
        print(f'Added calibrated Chl-A ("{chl_name_out}").{remove_str}')

    return D




def drop_variables(D, retain_vars = ['TEMP1', 'CNDC1', 'PSAL1', 
                                     'CHLA1', 'PRES'], 
                   drop_vars = None, 
                   retain_all = False
                    ):
        # Consider moving to a more general module?
        '''
        
        Drop measurement variables from the dataset.

        Will retain variables that don't have a PRES dimension, such as
        LATITUDE, TIME, STATION.

        Provide *either* strip_vars or retain_vars (not both). retain_vars
        is ignored if drop_vars is specified.
        
        Parameters:

        D: xarray dataset 
        retain_vars: (list or bool) Variables to retain (others will be retained) 
        drop_vars: (list or bool) Variables to drop (others will be retained) 
        retain_vars: (bool) Retain all (no change to the dataset)
        '''
        if retain_all:
            return D
        
        if drop_vars:
            D = D.drop(drop_vars)
            dropped = drop_vars
        else:
            all_vars = [varnm for varnm in D.data_vars]
            dropped = []
            for varnm in all_vars:
                if (varnm not in retain_vars and 
                    ('PRES' in D[varnm].dims 
                     or 'NISKIN_NUMBER' in D[varnm].dims)):
                    D = D.drop(varnm)
                    dropped += [varnm]
    
        if len(dropped)>1:
            print(f'Dropped these variables from the Dataset: {dropped}.')

        return D



## SMALL FUNCTIONS FOR MODIFYING METADATA ETC

## Look over and consider moving some (all?) of these to 
## nc_attrs.conventionalize?

def remove_numbers_in_names(D):
    '''
    Remove numbers from variable names like "TEMP1", "PSAL2".

    If more than one exists (e.g. "TEMP1", "TEMP2") -> don't change anything.
    '''
    # Get variable names
    all_varnms = [varnm for varnm in D.data_vars]

    # Get number-stripped names 
    varnms_stripped = [re.sub(r'\d', '', varnm) for varnm in all_varnms]

    # Identify duplicates 
    counter = Counter(varnms_stripped)
    duplicates = [item for item, count in counter.items() if count > 1]

    # Strip names
    for varnm in all_varnms:
        if re.sub(r'\d', '', varnm) not in duplicates:
            varnm_stripped = re.sub(r'\d', '', varnm)
            D = D.rename_vars({varnm:varnm_stripped})

    return D

def add_now_as_date_created(D):
    '''
    Add a global attribute "date_created" with todays date.
    '''
    now_time = pd.Timestamp.now()
    now_str = time.datetime_to_ISO8601(now_time)

    D.attrs['date_created'] = now_str

    return D


def reorder_attrs(D):
    """
    Reorder global and variable attributes of a dataset based on the 
    specified order in _standard_attrs.

    Parameters:
        ds (xarray.Dataset): The dataset containing global attributes.
        ordered_list (list): The desired order of global attributes.

    Returns:
        xarray.Dataset: The dataset with reordered global attributes.
    """
    ### GLOBAL
    reordered_list = _reorder_list(D.attrs, 
                                  _standard_attrs.global_attrs_ordered)
    attrs_dict = D.attrs
    D.attrs = {}
    for attr_name in reordered_list:
        D.attrs[attr_name] = attrs_dict[attr_name]

    ### VARIABLE
    for varnm in D.data_vars:
        reordered_list_var = _reorder_list(D[varnm].attrs, 
                      _standard_attrs.variable_attrs_ordered)
        var_attrs_dict = D[varnm].attrs
        D[varnm].attrs = {}
        for attr_name in reordered_list_var:
            D[varnm].attrs[attr_name] = var_attrs_dict[attr_name]
    return D



def _reorder_list(input_list, ordered_list):
    '''
    reorder a list input_list according to the order specified in ordered_list
    '''
    # Create a set of existing attributes for quick lookup
    existing_attributes = set(input_list)

    # Extract ordered attributes that exist in the dataset
    ordered_attributes = [attr for attr in ordered_list if attr in existing_attributes]

    # Add any remaining attributes that are not in the ordered list
    remaining_attributes = [attr for attr in input_list if attr not in ordered_attributes]

    # Concatenate the ordered and remaining attributes
    reordered_attributes = ordered_attributes + remaining_attributes

    return reordered_attributes

#### EDITING (WRAPPER FOR FUNCTIONS IN THE data.ship_ctd_tools._ctd_edit.py module)

def hand_remove_points(D, variable, station):
    """
    Interactive removal of data points from CTD profiles 

    Parameters:
    - d (xarray.Dataset): The dataset containing the CTD data.
    - varnm (str): The name of the variable to visualize and edit 
                  (e.g. 'TEMP1', 'CHLA').
    - station (str): The name of the station 
                    (E.g. '003', '012_01', 'AT290', 'StationA').

    Example usage that will let you hand edit the profile 
    of "TEMP1" from station "StationA":
    
    ```python
    HandRemovePoints(my_dataset, 'TEMP1', 'StationA')
    ```
    Note: Use the interactive plot to select points for removal, then 
    click the corresponding buttons for actions.
    """
    edit.hand_remove_points(D, variable, station)

def apply_offset(D):
    """
    Apply an offset to a selected variable in a given xarray CTD Dataset.

    Parameters:
    - D (xarray.Dataset): The CTD dataset to which the offset will be applied.

    Displays interactive widgets for selecting the variable, choosing the application scope
    (all stations or a single station), entering the offset value, and applying or exiting.

    The offset information is stored as metadata in the dataset's variable attributes.

    Examples:
    ```python
    apply_offset(my_dataset)
    ```

    Note: This function utilizes IPython widgets for interactive use within a Jupyter environment.
    """

    edit.apply_offset(D)

def drop_vars_pick(D):
    '''
    Interactively drop (remove) selected variables from an xarray Dataset.

    Parameters:
    - D (xarray.Dataset): The dataset from which variables will be dropped.

    Displays an interactive widget with checkboxes for each variable, allowing users
    to select variables to remove. The removal is performed by clicking the "Drop variables"
    button. The removed variables are also printed to the output.

    After running this function, D will be updated.

    Examples:
    ```python
    drop_vars(my_dataset)
    ```
    Note: This class utilizes IPython widgets for interactive use within a Jupyter environment.
    '''

    edit.drop_vars_pick(D)


#### VISUALIZATION (WRAPPER FOR FUNCTIONS IN THE data.ship_ctd_tools._ctd_visualize.py module)

def map(D):
    '''
    Generate a quick map of the cruise CTD stations.

    Parameters:
    - D (xarray.Dataset): The dataset containing LATITUDE and LONGITUDE.
    - height (int, optional): Height of the map figure. Default is 1000.
    - width (int, optional): Width of the map figure. Default is 1000.
    - return_fig_ax (bool, optional): If True, return the Matplotlib figure and axis objects.
      Default is False.
    - coast_resolution (str, optional): Resolution of the coastline data ('50m', '110m', '10m').
      Default is '50m'.
    - figsize (tuple, optional): Size of the figure. If None, the original size is used.

    Displays a quick map using the provided xarray Dataset with latitude and longitude information.
    The map includes a plot of the cruise track and red dots at data points.

    Additionally, the function provides buttons for interaction:
    - "Close" minimizes and closes the plot.
    - "Original Size" restores the plot to its original size.
    - "Larger" increases the plot size.

    Examples:
    ```python
    ctd.map(D)
    ```
    or
    ```python
    fig, ax = map(my_dataset, return_fig_ax=True)
    ```

    Note: This function utilizes the `quickmap` module for generating a stereographic map.

    TBD: 
    - Should come up with an reasonable autoscaling.
    - Should produce some grid lines.
    '''
    
    viz.map(D)


def inspect_profiles(D):
    """
    Interactively inspect individual CTD profiles in an xarray dataset.

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
    viz.inspect_profiles(D)


def inspect_dual_sensors(D):
    """
    Interactively inspect profiles of sensor pairs (e.g., PSAL1 and PSAL2).

    Parameters:
    - D: xarray.Dataset, the dataset containing the variables.

    Usage:
    - Call inspect_dual_sensors(D) to interactively inspect profiles.
    """
    viz.inspect_dual_sensors(D)


def contour(D):
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

    viz.ctd_contours(D)

