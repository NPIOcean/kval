'''
kval.ctd

--------------------------------------------------------------
A note about maintaining a metadata record of processing steps
--------------------------------------------------------------

We want to maintain a record in the file metadata of all operations
that modify the file in significant ways. 

This is done by populating the variable attributes of the
PROCESSING variable of the dataset. Specifically:

- *ds.PROCESSING.post_processing* should contain an algorithmic 
  description of steps that were applied. Should be human readable
  but contain all necessary details to reproduce the processing step.
- *ds.PROCESSING.python_script* should contain a python script 
  reproducing the processing procedure. In cases where data are changed
  based on interactive user input (e.g. hand selecting points), the
  corresponding line of code in ds.PROCESSING.python_script should be
  a call to a corresponding non-interactive function performing the exact
  equivalent modifications to the data.

The preferred method of updating the these metadata attributes is using
the decorator function defined at the start of the script. The decorator
is defined below in record_processing(). An example of how it is used can 
be found above the function metadata_auto().

In cases with interactive input, it is not always feasible to use the 
decorator approach. In such cases, it may be necessary to update 
ds.PROCESSING.post_processing and ds.PROCESSING.python_script
more directly.

'''

import xarray as xr
from kval.data.ship_ctd_tools import _ctd_tools as tools
from kval.data.ship_ctd_tools import _ctd_visualize as viz
from kval.data.ship_ctd_tools import _ctd_edit as ctd_edit
from kval.file import matfile
from kval.data import dataset, edit
from kval.util import time
from kval.metadata import conventionalize, _standard_attrs
from kval.metadata.check_conventions import check_file_with_button
from typing import List, Optional, Union
import numpy as np
import functools
import inspect


# Want to be able to use these functions directly..
from kval.data.dataset import metadata_to_txt, to_netcdf

##### DECORATOR TO PRESERVE PROCESSING STEPS IN METADATA

def record_processing(description_template, py_comment = None):
    """
    A decorator to record processing steps and their input arguments in the 
    dataset's metadata.
    
    Parameters:
    - description_template (str): A template for the description that includes 
                                  placeholders for input arguments.
    
    Returns:
    - decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(ds, *args, **kwargs):

            # Apply the function
            ds = func(ds, *args, **kwargs)

            # Check if the 'PROCESSING' field exists in the dataset
            if 'PROCESSING' not in ds:
                # If 'PROCESSING' is not present, return the dataset without any changes
                return ds


            # Prepare the description with input arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(ds, *args, **kwargs)
            bound_args.apply_defaults()

            # Format the description template with the actual arguments
            description = description_template.format(**bound_args.arguments)
            
            # Record the processing step
            ds['PROCESSING'].attrs['post_processing'] += description + '\n'
            
            # Prepare the function call code with arguments
            args_list = []
            for name, value in bound_args.arguments.items():
                if name != 'ds':  # Skip the 'ds' argument as it's always present
                    default_value = sig.parameters[name].default
                    if value != default_value:
                        if isinstance(value, str):
                            args_list.append(f"{name}='{value}'")
                        else:
                            args_list.append(f"{name}={value}")

            function_call = (f"ds = data.ctd.{func.__name__}(ds, "
                             f"{', '.join(args_list)})")

            if py_comment:
                ds['PROCESSING'].attrs['python_script'] += (
                    f"\n\n# {py_comment.format(**bound_args.arguments)}"
                    f"\n{function_call}")
            else:
                ds['PROCESSING'].attrs['python_script'] += (
                    f"\n\n{function_call}")
            return ds
        return wrapper
    return decorator




## LOADING AND SAVING DATA

def ctds_from_cnv_dir(
    path: str,
    station_from_filename: bool = False,
    verbose: bool = False,
    start_time_NMEA = False,
    processing_variable = True,
) -> xr.Dataset:
    
    """
    Create CTD datasets from CNV files in the specified path.

    Parameters:
    - path (str): Path to the CNV files.
    - station_from_filename (bool): Whether to extract station information 
                                    from filenames.
    - time_warnings (bool): Enable/disable time-related warnings.
    - verbose: If False, suppress some prints output.
    - start_time_NMEA (bool, optional)
      If True: get start_time attribute from the "NMEA UTC (Time)" 
      header line. Default (False) is to grab it from the "start_time" line. 
      (That seems to occasionally cause problems).

    Returns:
    - ds (xarray.Dataset): Joined CTD dataset.

    """
    cnv_files = tools._cnv_files_from_path(path)
    number_of_cnv_files = len(cnv_files)
    if number_of_cnv_files==0:
        raise Exception('Did not find any .cnv files in the specified '
                f'directory ("{path}"). Is there an error in the path?')
    else:
        print(f'Found {number_of_cnv_files} .cnv files in  "{path}".')

    profile_datasets = tools._datasets_from_cnvlist(cnv_files,
        station_from_filename = station_from_filename,
        verbose = verbose, start_time_NMEA = start_time_NMEA)
    
    ds = tools.join_cruise(profile_datasets,
        verbose=verbose)
    
    # Add PROCESSING variable
    if processing_variable:
        ds = dataset.add_processing_history_var(ds, 
                                    source_files = np.sort(cnv_files) )
        ds.attrs['history'] = ds.history.replace('"SBE_processing"',
                                            '"PROCESSING.SBE_processing"')
        
        # Add python scipt snipped to reproduce this operation
        ds.PROCESSING.attrs['python_script'] += (
    f"""from kval import data

# Path to directory containing *source_files* (MUST BE SET BY THE USER!)
cnv_dir = "./"

# Load all .cnv files and join together into a single xarray Dataset:
ds = data.ctd.ctds_from_cnv_dir(
    cnv_dir,
    station_from_filename={station_from_filename},
    start_time_NMEA={start_time_NMEA},
    processing_variable={processing_variable}
    )"""
            )   


    return ds


def ctds_from_cnv_list(
    cnv_list: list,
    station_from_filename: bool = False,
    time_warnings: bool = True,
    verbose: bool = True,
    start_time_NMEA = False,
    processing_variable = True,
) -> xr.Dataset:
    """
    Create CTD datasets from CNV files in the specified path.

    Parameters:
    - path (str): List of CNV file paths.
    - station_from_filename (bool): Whether to extract station 
                                    information from filenames.
    - time_warnings (bool): Enable/disable time-related warnings.
    - verbose: If False, suppress some prints output.
    - start_time_NMEA (bool, optional)
      If True: get start_time attribute from the "NMEA UTC (Time)" 
      header line. Default (False) is to grab it from the "start_time" line. 
      (That seems to occasionally cause problems).

    Returns:
    - ds (xarray.Dataset): Joined CTD dataset.
    """
    profile_datasets = tools._datasets_from_cnvlist(
        cnv_list, verbose = verbose, start_time_NMEA = start_time_NMEA,
        station_from_filename = station_from_filename)
    ds = tools.join_cruise(profile_datasets,
        verbose=verbose)
    
    # Add PROCESSING variable
    if processing_variable:
        ds = dataset.add_processing_history_var(ds, 
                                                source_files = np.sort(cnv_list) )
        ds.attrs['history'] = ds.history.replace(
            '"SBE_processing"', '"PROCESSING.SBE_processing"')
        
        # Add python scipt snipped to reproduce this operation
        ds.PROCESSING.attrs['python_script'] += (
            'from kval import data\n'
            +'cnv_list = [{files}] # A list of strings specifying paths to all'
            +' files in *source_files*.\n\n# Load all .cnv files and join '
            +'together into a single xarray Dataset:\n'
            +'ds = data.ctd.ctds_from_cnv_list(cnv_list,\n'
            +f'    station_from_filename={station_from_filename},\n'
            +f'    start_time_NMEA={start_time_NMEA},\n'
            +f'    processing_variable={processing_variable})'
            )   

    return ds


def dataset_from_btl_dir(
    path: str,
    station_from_filename: bool = False,
    start_time_NMEA: bool = False,
    time_adjust_NMEA: bool = False,
    verbose: bool = True
) -> xr.Dataset:
    """
    Create CTD datasets from CNV files in the specified path.

    Parameters:
    - path (str): Path to the CNV files.
    - station_from_filename (bool): Whether to extract station information 
                                    from filenames.
    - time_warnings (bool): Enable/disable time-related warnings.
    - verbose: If False, suppress some prints output.
    Returns:
    - ds (xarray.Dataset): Joined CTD dataset.
    """
    btl_files = tools._btl_files_from_path(path)
    number_of_btl_files = len(btl_files)
    if number_of_btl_files==0:
        raise Exception('Did not find any .btl files in the specified '
                f'directory ("{path}"). Is there an error in the path?')
    else:
        print(f'Found {number_of_btl_files} .btl  files in  "{path}".')
    profile_datasets = tools._datasets_from_btllist(
        btl_files, verbose = verbose, start_time_NMEA = start_time_NMEA,
        time_adjust_NMEA = time_adjust_NMEA,
        station_from_filename = station_from_filename)
    ds = tools.join_cruise_btl(profile_datasets,
        verbose=verbose)
    ds = ds.transpose()
    
    return ds


#############

def from_netcdf(path_to_file):
    '''
    Import a netCDF file - e.g. one previously generated
    with these tools.
    '''

    d = xr.open_dataset(path_to_file, decode_cf = False)

    return d

#############

def to_mat(ds, outfile, simplify = False):
    """
    Convert the CTD data (xarray.Dataset) to a MATLAB .mat file.
      
    A field 'TIME_mat' with Matlab datenums is added along with the data. 

    Parameters:
    - ds (xarray.Dataset): Input dataset to be converted.
    - outfile (str): Output file path for the MATLAB .mat file. If the path
      doesn't end with '.mat', it will be appended.
    - simplify (bool, optional): If True, simplify the dataset by extracting
      only coordinate and data variables (no metadata attributes). If False, 
      the matfile will be a struct containing 
      [attrs, data_vars, coords, dims]. 
      Defaults to False.
      
    Returns:
    None: The function saves the dataset as a MATLAB .mat file.

    Example:
    >>> ctd.xr_to_mat(ds, 'output_matfile', simplify=True)
    """
    # Drop the empty PROCESSING variable (doesnt work well with 
    # MATLAB)
    ds_wo_proc = drop_variables(ds, drop_vars = 'PROCESSING')

    # Also transposing dimensions to PRES, TIME for ease of plotting etc in 
    # matlab.
    matfile.xr_to_mat(ds_wo_proc.transpose(), outfile, simplify = simplify)

############


def to_csv(ds, outfile):
    """
    Convert the CTD data (xarray.Dataset) to a human-readable .csv file.
    
    The file shows columnar data for all data paramaters for all stations.
    Statins are separated by a header with the station name/time/lat/lon.

    Parameters:
    - ds (xarray.Dataset): Input dataset to be converted.
    - outfile (str): Output file path for the .csv file. If the path
      doesn't end with '.csv', it will be appended.

    Returns:
    None: The function saves the dataset as a .cnv file.

    Example:
    >>> ctd.to_csv(ds, 'output_cnvfile', )
    """    
    prof_vars = ['PRES']
    
    for key in ds.data_vars.keys():
        if  'TIME' in ds[key].dims:
            if 'PRES' in ds[key].dims:
                prof_vars += [key]

    if not outfile.endswith('.csv'):
        outfile += '.csv'
                
    with open(outfile, 'w') as f:
        for time_ in ds.TIME.values:
            ds_prof = ds.sel(TIME=time_)
            time_str= time.datenum_to_timestamp(time_).strftime(
                '%Y-%m-%d %H:%M:%S')
            print('#'*88, file=f)
            print(f'#####  {ds_prof.STATION.values:<8} ###  {time_str}  '
                  f'###  LAT: {ds_prof.LATITUDE.values:<10}'
                  f' ### LON: {ds_prof.LONGITUDE.values:<10} #####', file=f)
            print('#'*88 + '\n', file=f)
        
            ds_pd = ds_prof[prof_vars].to_pandas()
            ds_pd = ds_pd.drop('TIME', axis = 1)
            
            ds_pd = ds_pd.dropna(
                subset=ds_pd.columns.difference(['PRES']), how='all')
            print(ds_pd.to_csv(), file=f)


### MODIFYING DATA


@record_processing(
    'Rejected values of {variable} outside the range ({min_val}, {max_val})',
    py_comment = ('Rejecting values of {variable} outside'
                  ' the range ({min_val}, {max_val}):'))
def threshold(ds: xr.Dataset, variable: str, 
                min_val: Optional[float] = None, 
                max_val: Optional[float] = None
                ) -> xr.Dataset:
        
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
    ds_thresholded = threshold_edit(ds, 'TEMP', max_val=3, min_val=-1)
    """
    ds = edit.threshold(ds=ds, variable = variable, 
                        max_val = max_val, min_val = min_val)
    return ds



@record_processing(
    'Applied offset ={offset} to the variable {variable}.',
    py_comment = ('Applied offset {offset} to variable {variable}:'))
def offset(ds: xr.Dataset, variable: str, offset: float) -> xr.Dataset:
    """
    Apply a fixed offset to a specified variable in an xarray Dataset.

    This function modifies the values of the specified variable by adding a 
    fixed offset to them. The `valid_min` and `valid_max` attributes are 
    updated to reflect the new range of values after applying the offset.

    Parameters
    ----------
    ds : xr.Dataset
        The input xarray Dataset.
    variable : str
        The name of the variable within the Dataset to which the offset 
        will be applied.
    offset : float
        The fixed offset value to add to the variable.

    Returns
    -------
    xr.Dataset
        A new xarray Dataset with the offset applied to the specified 
        variable. The `valid_min` and `valid_max` attributes are updated 
        accordingly.

    Examples
    --------
    # Apply an offset of 5 to the 'TEMP' variable
    ds_offset = apply_offset(ds, 'TEMP', offset=5)
    """

    ds = edit.offset(ds=ds, variable=variable, offset=offset)

    return ds


## APPLYING CORRECTIONS ETC
@record_processing(
    ('Applied a calibration to chlorophyll: {chl_name_out} = {A} '
    '* {chl_name_in} + {B}. '),
    py_comment = ('Applying chlorophyll calibration based on fit to lab'
    ' values:'))
def calibrate_chl(
    ds: xr.Dataset,
    A: float,
    B: float,
    chl_name_in: Optional[str] = 'CHLA_fluorescence',
    chl_name_out: Optional[str] = 'CHLA',
    verbose: Optional[bool] = True,
    remove_uncal: Optional[bool] = False
) -> xr.Dataset:
    """
    Apply a calibration to chlorophyll based on a fit based on water samples.
    
    CHLA -> A * CHLA_fluorescence + B

    Where CHLA_fluorescence is the name of the variable containing 
    uncalibrated chlorophyll from the instrument.

    Append suitable metadata.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing A, B: Linear coefficients based on fitting to 
        chl samples.
    A : float
        Linear coefficient for calibration.
    B : float
        Linear coefficient for calibration.
    chl_name_in : str, optional
        Name of the variable containing uncalibrated chlorophyll from the 
        instrument. Default is 'CHLA_fluorescence', will look for 
        'CHLA1_fluoresence' if that doesn't exist.
    chl_name_out : str, optional
        Name of the calibrated chlorophyll variable. If not provided, it is 
        derived from chl_name_in. Default is None.
    verbose : bool, optional
        If True, print messages about the calibration process. 
        Default is True.
    remove_uncal : bool, optional
        If True, remove the uncalibrated chlorophyll from the dataset. 
        Default is False.

    Returns
    -------
    xr.Dataset
        Updated dataset with calibrated chlorophyll variable.
    """

    # Determine the input variable name
    if chl_name_in not in ds.keys():
        if 'CHLA1_fluorescence' in ds.keys():
            chl_name_in = 'CHLA1_fluorescence'
        else:
            raise Exception(
                f'Did not find {chl_name_in} or "CHLA1_fluorescence" '
                'in the dataset. Please specify the variable name of '
                'uncalibrated chlorophyll using the *chl_name_in* flag.')


    # Determine the output variable name for calibrated chlorophyll
    if not chl_name_out:
        if '_instr' in chl_name_in or '_fluorescence' in chl_name_in:
            chl_name_out = (chl_name_in.replace('_instr', '')
                            .replace('_fluorescence', ''))
        else:
            chl_name_out = f'{chl_name_in}_cal'

    # Create a new variable with the coefficients applied
    ds[chl_name_out] = A * ds[chl_name_in] + B
    ds[chl_name_out].attrs = {key: item for key, item in 
                              ds[chl_name_in].attrs.items()}

    # Add suitable attributes
    new_attrs = {
        'long_name': ('Chlorophyll-A concentration calibrated against '
                      'water sample measurements'),
        'calibration_formula': f'{chl_name_out} = {A} * {chl_name_in} + {B}',
        'coefficient_A': A,
        'coefficient_B': B,
        'comment': ('No correction for near-surface fluorescence quenching '
                   '(see e.g. https://doi.org/10.4319/lom.2012.10.483) has '
                   'been applied.'),
        'processing_level': 'Post-recovery calibrations have been applied',
        'QC_indicator': 'good data',
    }

    for key, item in new_attrs.items():
        ds[chl_name_out].attrs[key] = item

    # Remove the uncalibrated chl
    if remove_uncal:
        remove_str = (f' Removed uncalibrated Chl-A ("{chl_name_in}") from'
                        ' the dataset.')
        ds = ds.drop(chl_name_in)
    else:
        remove_str = ''

    # Print
    if verbose:
        print(f'Added calibrated Chl-A ("{chl_name_out}") calculated '
              f'from variable "{chl_name_in}".{remove_str}')

    return ds





### MODIFYING METADATA

@record_processing('Applied automatic standardization of metadata,',
    py_comment = 'Applying standard metadata (global+variable attributes):')
def metadata_auto(ds, NPI = True,):
    '''
    Various modifications to the metadata in order to make the dataset
    more nicely formatted for publication.
    '''

    ds = conventionalize.remove_numbers_in_var_names(ds)
    ds = conventionalize.add_standard_var_attrs(ds)
    ds = conventionalize.add_standard_glob_attrs_ctd(ds, override = False)
    ds = conventionalize.add_standard_glob_attrs_org(ds)
    ds = conventionalize.add_gmdc_keywords_ctd(ds)
    ds = conventionalize.add_range_attrs(ds)
    ds = conventionalize.reorder_attrs(ds)

    return ds

# Note: Doing PROCESSING.post_processing record keeping within the 
# drop_variables() function because we want to access the *dropped* list.
@record_processing('',
    py_comment = 'Dropping some variables')
def drop_variables(
    ds: xr.Dataset,
    retain_vars: Optional[Union[List[str], bool]] = None,
    drop_vars: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Drop measurement variables from the dataset based on specified criteria.

    This function retains or drops variables from an xarray.Dataset based on
    provided lists of variables to retain or drop. If `retain_all` is True, 
    no variables will be dropped.

    Parameters:
    - ds (xr.Dataset): The dataset from which variables will be dropped.
    - retain_vars (Optional[Union[List[str], bool]]): List of variables to retain.
      If a boolean `True` is provided, all variables are retained (no changes made). 
      This parameter is ignored if `drop_vars` is specified.
    - drop_vars (Optional[List[str]]): List of variables to drop from the dataset.
      If specified, this will override `retain_vars`.

    Returns:
    - xr.Dataset: The modified dataset with specified variables dropped or retained.
    
    Notes:
    - Provide *either* `retain_vars` or `drop_vars`, but not both.
    - Variables that do not have a 'PRES' or 'NISKIN_NUMBER' dimension will always be retained.
    """

    if retain_vars== None and drop_vars == None:
        return ds
    
    if drop_vars is not None:
        ds = ds.drop(drop_vars)
        dropped = drop_vars
    else:
        if retain_vars is None:
            raise ValueError("Either `drop_vars` or `retain_vars` must be specified, not both.")
        
        if isinstance(retain_vars, bool):
            if retain_vars:
                return ds
            retain_vars = []

        all_vars = list(ds.data_vars)
        dropped = []
        for varnm in all_vars:
            if (varnm not in retain_vars and 
                ('PRES' in ds[varnm].dims or 'NISKIN_NUMBER' in ds[varnm].dims)):
                ds = ds.drop(varnm)
                dropped.append(varnm)
    
    if dropped:
        drop_str = f'Dropped these variables from the Dataset: {dropped}.'
        print(drop_str)
        if 'PROCESSING' in ds:
            ds['PROCESSING'].attrs['post_processing'] += f'{drop_str}\n'

            

    return ds





#### VISUALIZATION (WRAPPER FOR FUNCTIONS IN THE data.ship_ctd_tools._ctd_visualize.py module)

def map(ds, station_labels = False, 
        station_label_alpha = 0.5):
    '''
    Generate a quick map of the cruise CTD stations.

    Parameters:
    - ds (xarray.Dataset): The dataset containing LATITUDE and LONGITUDE.
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
    ctd.map(ds)
    ```
    or
    ```python
    fig, ax = map(my_dataset, return_fig_ax=True)
    ```

    Note: This function utilizes the `quickmap` module for generating a stereographic map.

    TBds: 
    - Should come up with an reasonable autoscaling.
    - Should produce some grid lines.
    '''
    
    viz.map(ds, station_labels = station_labels, station_label_alpha = station_label_alpha)


def inspect_profiles(ds):
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
    viz.inspect_profiles(ds)


def inspect_dual_sensors(ds):
    """
    Interactively inspect profiles of sensor pairs (e.g., PSAL1 and PSAL2).

    Parameters:
    - ds: xarray.Dataset, the dataset containing the variables.

    Usage:
    - Call inspect_dual_sensors(ds) to interactively inspect profiles.
    """
    viz.inspect_dual_sensors(ds)


def contour(ds):
    """
    Create interactive contour plots based on an xarray dataset.

    Parameters:
    - ds (xr.Dataset): The xarray dataset containing profile variables and coordinates.

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

    viz.ctd_contours(ds)




### INSPECTING METADATA


def quick_metadata_check(ds,):
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
        if attr not in ds.attrs:
            print(f'- Possibly missing {attr}')

    print('\n# VARIABLE #')

    ### VARIABLE
    attrs_dict_ref_var = _standard_attrs.variable_attrs_necessary

    for varnm in ds.variables:
        if 'PRES' in ds[varnm].dims:
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
                if var_attr not in ds[varnm].attrs:
                    print(f'- {varnm}: Possibly missing {var_attr}')
                    any_missing = True
            if not any_missing:
                    print(f'- {varnm}: Possibly missing {var_attr}')
                    any_missing = True
            if not any_missing:
                    print(f'- {varnm}: OK')


############

def check_metadata(ds):
    '''
    Use the IOOS compliance checker 
    (https://github.com/ioos/compliance-checker-web)
    to chek an nc file (CF and ACDD conventions).

    Can take a file path or an xr.Dataset as input
    
    Output is closed with a "Close" button.
    '''

    check_file_with_button(ds)




############



## SMALL FUNCTIONS FOR MODIFYING METADATA ETC

## Look over and consider moving some (all?) of these to 
## nc_attrs.conventionalize?

def set_attr_glob(ds, attr):
    """
    Set a global attribute (metadata) for the dataset.

    Parameters:
        ds (xarray dataset): The dictionary representing the dataset or data frame.
        attr (str): The global attribute name (e.g. "title").

    Returns:
        xr.Dataset: The updated xarray Dataset with the global attribute set.

    Example:
        To set an attribute 'title' in the dataset ds:
        >>> ds = set_attr_var(ds, 'title')
    """
    ds = conventionalize.set_glob_attr(ds, attr)
    return ds


def set_attr_var(ds, variable, attr):
    """
    Set a variable attribute (metadata) for a specific variable in the dataset.

    Parameters:
        ds (xarray dataset): The dictionary representing the dataset or data frame.
        variable (str): The variable name for which the attribute will be set (e.g. "PRES").
        attr (str): The attribute name (e.g. "long_name").

    Returns:
        xr.Dataset: The updated xarray Dataset with the variable attribute set.

    Example:
        To set an attribute 'units' for the variable 'TEMP1' in the dataset ds:
        >>> ds = set_attr_var(ds, 'TEMP1', 'units')
    """
    ds = conventionalize.set_var_attr(ds, variable, attr)
    return ds



#### EDITING (WRAPPER FOR FUNCTIONS IN THE data.ship_ctd_tools._ctd_edit.py module)

def hand_remove_points(ds, variable, TIME_index):
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
    ds0 = ds.copy()
    hand_remove = ctd_edit.hand_remove_points(ds, variable, TIME_index)
    ds = hand_remove.d

    return ds

def apply_threshold(ds):
    '''
    Interactively select a valid range for data variables,
    and apply thresholds to the data.
    '''
    ctd_edit.threshold_edit(ds)
    return ds


def apply_offset(ds):
    """
    Apply an offset to a selected variable in a given xarray CTD Dataset.

    Parameters:
    - ds (xarray.Dataset): The CTD dataset to which the offset will be applied.

    Displays interactive widgets for selecting the variable, choosing the application scope
    (all stations or a single station), entering the offset value, and applying or exiting.

    The offset information is stored as metadata in the dataset's variable attributes.

    Examples:
    ```python
    apply_offset(my_dataset)
    ```

    Note: This function utilizes IPython widgets for interactive use within a Jupyter environment.
    """

    ctd_edit.apply_offset(ds)
    
    return ds

def drop_vars_pick(ds):
    '''
    Interactively drop (remove) selected variables from an xarray Dataset.

    Parameters:
    - ds (xarray.Dataset): The dataset from which variables will be dropped.

    Displays an interactive widget with checkboxes for each variable, allowing users
    to select variables to remove. The removal is performed by clicking the "Drop variables"
    button. The removed variables are also printed to the output.

    After running this function, ds will be updated.from IPython.display import clear_output


    Note: This class utilizes IPython widgets for interactive use within a Jupyter environment.
    '''

    edit_obj = ctd_edit.drop_vars_pick(ds)
    return edit_obj.D

### TABLED/UNFINISHED/COULD PERHAPS BECOME USEFUL..

if False:

    def _drop_stations_pick(ds):
        '''
        UNFINISHED! Tabled for fixing..

        Interactive class for dropping selected time points from an xarray Dataset based on the value of STATION(TIME).

        Parameters:
        - ds (xarray.Dataset): The dataset from which time points will be dropped.

        Displays an interactive widget with checkboxes for each time point, showing the associated STATION.
        Users can select time points to remove. The removal is performed by clicking the "Drop time points"
        button. The removed time points are also printed to the output.

        Note: This class utilizes IPython widgets for interactive use within a Jupyter environment.
        '''

        edit_obj = ctd_edit.drop_stations_pick(ds)
        return edit_obj.D

