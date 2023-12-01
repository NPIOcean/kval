'''
OCEANOGRAPY.IO.MATFILE
Writing to and from .mat files

NOTE:
This is a copy of a function used on a specific project
-> Rather hardcoded, want to look into a general .mat reader.

This is slightly clunky since the mat files as ranslated by scipy.io 
occasionally gives rather clunky output (nested ndarrays..).

mat_to_dict below is a good starting point, though.

'''


import oceanograpy as opy
import xarray as xr
from scipy.io import matlab
import numpy as np
import datetime
from matplotlib.dates import date2num
from oceanograpy.util import time


def mat_to_xr_1D(matfile, time_name = 'time', epoch = '1970-01-01'):
    '''
    Read a matfile to an xr Dataset. 
    
    0-D variables are interpreteted as metadata and stored as global attributes.

    Assuming just one dimension: time.

    Will only work when the *time_name* variable is on the Matlab datenum format.
    I.e. time must be on the form, e.g., [737510.3754, ..], and not, e.g.:
        - ['2021-02-03-03:22:21', ..] or
        - [[2021, 2, 3, 3, 22, 21], ..] 

    '''
    # Read data/metadata from matfile
    data_dict, attr_dict = _parse_matfile_to_dict(matfile)

    # (Attempt to) parse time
    try:
        time_stamp = _parse_time(data_dict, time_name = time_name)
        # Remove time variable if we successfully parsed time
        data_dict.pop(time_name)
    except:
        print(f'NOTE: Unable to parse time from the {time_name} field.')

    # Collect in an xr Dataset
    ds = xr.Dataset(coords = {'TIME':time_stamp})
    # Add data variables
    for varnm, item in data_dict.items():
        ds[varnm] = (('TIME'), data_dict[varnm])
    # Add metadata
    for attrnm, item in attr_dict.items():
        ds.attrs[attrnm] = attr_dict[attrnm]

    
    # Sort chronologically
    ds = ds.sortby('TIME')
    
    return ds


def _parse_matfile_to_dict(matfile):
    '''
    Use scipy.io.matlab to parse a matfile. 

    Will skip variable names containing data of internal Matlab types, 
    e.g. Datetime, which are not accessible outside Matlab.

    Probably only works for <v7.3 (=>7.3 needs its own parser, I think)

    Parameters: - matfile: math to a file with suffix .mat

    Returns: 
    - data_dict: Dictionary with variables interpreted as data 
    - attr_dict: Dictionary with variables interpreted as global attributes
      (metadata)
    '''


    ### 1. LOAD DATA TO PYTHON
    matfile_dict = matlab.loadmat(matfile, squeeze_me = True)


    ### 2. IDENTIFY THE DICTIONARY CONTAINING THE DATA
    # (By default also loads a bunch of non-useful metadata)

    # Identify the name of the structure containing the data
    # (assuming this is the only field not on the format '__[]__')
    data_key_list = []
    for dict_key in matfile_dict.keys():
        if not dict_key.startswith('__'):
             data_key_list += [dict_key]

    # If there is one such name: Use it to know where to extract data
    if len(data_key_list)==1:
        data_key = data_key_list[0]

    # If there is more than one: Return an exception fo now (not sure whether this
    # is an issue)   
    elif len(data_key_list)>1:
        raise Exception(f'Found multiple data keys in {matfile_dict}:\n'
            f'{data_key_list}. Not supported right now (eventually: give the)'
            'user a choice or import from both..')
    
    # If there is no apparent data field: Raise an exception 
    elif len(data_key_list) == 0:
        raise Exception(f'No valid data key found in {matfile_dict}:\n'
            f'-> Inspect your matfile!')
    
    # Load the dictionary containing the data
    ddict = matfile_dict[data_key]

    ### 3. SORT VARIABLES INTO DAtA AND ATTRIBUTE ARRAYS

    variables = ddict.dtype.names
    data_dict = {}
    attr_dict = {}

    for varnm in variables:
        parsed_data = ddict[varnm].flatten()[0]

        # If the variable is of type MatlabOpaque, it cannot be read outside MATLAB
        # -> Print a message and skip the variable
        if isinstance(parsed_data, matlab._mio5_params.MatlabOpaque):
            print(f'NOTE: "{varnm}" is of an internal Matlab class type (e.g. Datetime) '
                  'which cannot be read outside Matlab -> Skip')

        # If the object is iterable (list, array, etc): Interpret is as a data variable
        elif isinstance(parsed_data, (list, tuple, np.ndarray)):
            data_dict[varnm] = parsed_data

        # Otherwise (float, int, str, etc): Interpret is as a global attribute
        elif isinstance(parsed_data, (int, float, str)):
            attr_dict[varnm] = parsed_data

    ### 4. RETURN OUTPUT

    return data_dict, attr_dict


def _parse_time(data_dict, time_name = 'time'):
    '''
    Parse Matlab time read to a dictionary data_dict using _parse_matfile_to_dict.

    Currently only works for time on the matlab datenum format. May want to expand to 
    other formats like [yr, mo .. min, sec]. On the other hand, it might be cleanest
    to just require the fields  
    '''
    try:
        time_stamps = time.matlab_time_to_timestamp(data_dict[time_name])
        return time_stamps
    except: # May have to build other cases here, eventually. 
        print(f'Unable to parse time from the {time} variable. '
              '(Expecting Matlab datenum format)')









def mat_to_xr(matfile, struct_name = None, pressure_name = 'PR', time_name = 'TIME'):
    """
    Convert MATLAB .mat file to an xarray Dataset.

    (Perhaps slightly hardcoded for ATWAIN 2015 dataset. Should test more
    extensively!)

    Looks for (TIME) or (TIME, PRES) variables. Should generalize 
    and maybe make wrappers for specific applications. 
    
    Parameters:
    - matfile (str): Path to the MATLAB .mat file.
    - struct_name (str): Name of the structure in the .mat file containing the data.
    - pressure_name (str): Name of the variable representing pressure.

    Returns:
    - xr.Dataset: An xarray Dataset containing the converted data.

    This function reads data from a MATLAB .mat file, specifically from a
    structure named 'struct_name'. It flattens the 1D variables and creates a
    Dataset with 1D and 2D variables, replacing the 'TIME' field with parsed
    datetime values. Pressure values are checked for consistency across
    profiles, and the resulting data is organized into an xarray Dataset.

    Note: This function assumes that the time information in the 'TIME' field is
    in the format [year, month, day, hour, minute, second].
    """

    matfile_dict = matlab.loadmat(matfile, squeeze_me = True)

    # Identify the name of the structure containing the data
    # (assuming this is the only field not on the format '__[]__')
    data_key_list = []
    for dict_key in matfile_dict.keys():
        if not dict.key.startswith('__'):
             data_key_list += [dict_key]

    # If there is one such name: Use it to know where to extract data
    if len(data_key_list)==1:
        data_key = data_key_list[0]

    # If there is more than one: Return an exception fo now (not sure whether this
    # is an issue)   
    elif len(data_key_list)>1:
        raise Exception(f'Found multiple data keys in {matfile_dict}:\n'
            f'{data_key_list}. Not supported right now (eventually: give the)'
            'user a choice or import from both..')
    
    # If there is no apparent data field: Raise an exception 
    elif len(data_key_list) == 0:
        raise Exception(f'No valid data key found in {matfile_dict}:\n'
            f'-> Inspect your matfile!')
    
    # Load the dictionary containing the data
    ddict = matfile_dict[data_key]


    variables = ddict.dtype.names
    data_dict = {}
    varnms_1d, varnms_2d = [], []

    for varnm in variables:
        data_dict[varnm] = ddict[varnm].flatten()
        data_dict[varnm] = data_dict[varnm][0]

        if isinstance(data_dict[varnm][0], np.ndarray): # 1D variables 
            varnms_2d += [varnm]
        else:
            varnms_1d += [varnm]

    varnms_2d.remove('TIME')
    
    if False:
        varnms_2d.remove(pressure_name)


        # Get the pressure grid (and check that is the same for all variables..)
        PRES = []
        for level_pres in data_dict[pressure_name]:
            level_pres_valid = level_pres[~np.isnan(level_pres)]

            pres_unique = np.unique(level_pres_valid)
            
            if len(pres_unique)>1:
                raise Exception(f'{pressure_name} is'
                    ' different across profiles!'
                    '\n-> cannot load this as a gridded dataset.')
            elif len(pres_unique)==0:
                PRES += [np.nan]
            else:
                PRES += [pres_unique[0]]

    # Parse time from  [2015,    9,   17,   22,   19,   33] format.
    # -> Replace the TIME field with parsed datetime times
    TIME_lists = data_dict['TIME'].copy()
    TIME = []
    for time_list in TIME_lists:
        TIME += [date2num(datetime.datetime(*time_list))]




    # Collect in an xr Dataset
    ds = xr.Dataset(coords = {'TIME':TIME, 'PRES': PRES, })
    for varnm in varnms_1d:
        ds[varnm] = (('TIME'), data_dict[varnm].T)
    for varnm in varnms_2d:
        ds[varnm] = (('TIME', 'PRES'), data_dict[varnm].T)

    # Remove empty pressure bins (could be present at the start of end of
    # profiles..
    ds = ds.dropna('PRES', subset = ['PRES'])
    
    # Sort chronologically
    ds = ds.sortby('TIME')
    
    return ds


def mat_to_xr_old(matfile, struct_name = 'CTD', pressure_name = 'PR'):
    """
    Convert MATLAB .mat file to an xarray Dataset.

    (Perhaps slightly hardcoded for ATWAIN 2015 dataset. Should test more
    extensively!)

    Looks for (TIME) or (TIME, PRES) variables. Should generalize 
    and maybe make wrappers for specific applications. 
    
    Parameters:
    - matfile (str): Path to the MATLAB .mat file.
    - struct_name (str): Name of the structure in the .mat file containing the data.
    - pressure_name (str): Name of the variable representing pressure.

    Returns:
    - xr.Dataset: An xarray Dataset containing the converted data.

    This function reads data from a MATLAB .mat file, specifically from a
    structure named 'struct_name'. It flattens the 1D variables and creates a
    Dataset with 1D and 2D variables, replacing the 'TIME' field with parsed
    datetime values. Pressure values are checked for consistency across
    profiles, and the resulting data is organized into an xarray Dataset.

    Note: This function assumes that the time information in the 'TIME' field is
    in the format [year, month, day, hour, minute, second].
    """

    ddict = matlab.loadmat(matfile, squeeze_me = True)[struct_name]

    variables = ddict.dtype.names
    data_dict = {}
    varnms_1d, varnms_2d = [], []

    for varnm in variables:
        data_dict[varnm] = ddict[varnm].flatten()
        data_dict[varnm] = data_dict[varnm][0]
        
        if isinstance(data_dict[varnm][0], np.ndarray): #1 D variables 
            varnms_2d += [varnm]
        else:
            varnms_1d += [varnm]

    varnms_2d.remove('TIME')
    varnms_2d.remove(pressure_name)


    # Get the pressure grid (and check that is the same for all variables..)
    PRES = []
    for level_pres in data_dict[pressure_name]:
        level_pres_valid = level_pres[~np.isnan(level_pres)]

        pres_unique = np.unique(level_pres_valid)
        
        if len(pres_unique)>1:
            raise Exception(f'{pressure_name} is'
                ' different across profiles!'
                '\n-> cannot load this as a gridded dataset.')
        elif len(pres_unique)==0:
            PRES += [np.nan]
        else:
            PRES += [pres_unique[0]]

    # Parse time from  [2015,    9,   17,   22,   19,   33] format.
    # -> Replace the TIME field with parsed datetime times
    TIME_lists = data_dict['TIME'].copy()
    TIME = []
    for time_list in TIME_lists:
        TIME += [date2num(datetime.datetime(*time_list))]

    # Collect in an xr Dataset
    ds = xr.Dataset(coords = {'TIME':TIME, 'PRES': PRES, })
    for varnm in varnms_1d:
        ds[varnm] = (('TIME'), data_dict[varnm].T)
    for varnm in varnms_2d:
        ds[varnm] = (('TIME', 'PRES'), data_dict[varnm].T)

    # Remove empty pressure bins (could be present at the start of end of
    # profiles..
    ds = ds.dropna('PRES', subset = ['PRES'])
    
    # Sort chronologically
    ds = ds.sortby('TIME')
    
    return ds