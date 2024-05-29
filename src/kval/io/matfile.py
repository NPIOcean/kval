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

import xarray as xr
from scipy.io import matlab
import numpy as np
import datetime
from matplotlib.dates import date2num
from kval.util import time


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
        time_num = date2num(time_stamp)

        # Remove time variable if we successfully parsed time
        data_dict.pop(time_name)
    except:
        print(f'NOTE: Unable to parse time from the {time_name} field.')


    # Collect in an xr Dataset
    ds = xr.Dataset(coords = {'TIME':time_num})
    
    # Add data variables
    for varnm in data_dict:
        try:
            ds[varnm] = (('TIME'), data_dict[varnm])
        except:
            print(f'NOTE: Could not parse the variable "{varnm}" '
                  f' with shape: {data_dict[varnm].shape} as a TIME variable - expected shape ({ds.dims["TIME"]}). '
                  ' -> Skipping this variable.')
    # Add metadata
    for attrnm in attr_dict:
        ds.attrs[attrnm] = attr_dict[attrnm]

    # Sort chronologically
    ds = ds.sortby('TIME')
    
    return ds



def mat_to_xr_2D(matfile, time_name = 'time', depth_name_in = 'PRES', depth_name_out = 'PRES', 
                 epoch = '1970-01-01'):
    '''
    Read a .mat file with 2D variables and convert its data to an xarray (xr) Dataset.

    Assumes two dimensions: time and an depth-or pressure-type variable.

    Parameters:
    - matfile (str): Path to the MATLAB file.
    - time_name (str): Name of the time variable in the MATLAB file (default is 'time').
    - depth_name_in (str): Name of the depth variable in the MATLAB file (default is 'PRES').
    - depth_name_out (str): Name to be used for the depth-type variable in the xarray Dataset (default is 'PRES').
    - epoch (str): Reference epoch for time conversion, in the format 'YYYY-MM-DD' (default is '1970-01-01').

    Returns:
    - xr.Dataset: xarray Dataset containing the converted data.

    Notes:
    - 0-D variables are interpreted as metadata and stored as global attributes.
    - 
    - The time variable in the MATLAB file must be in the datenum format (e.g., [737510.3754, ..]).

    The function attempts to parse time and adds data variables to the xarray Dataset. It handles various cases 
    of variable dimensions.

    If parsing time is unsuccessful, a message is printed, and the time variable remains in the Dataset.

    Variables that do not conform to expected dimensions are skipped, and a message is printed.

    Metadata from the MATLAB file is added as global attributes to the xarray Dataset. The resulting Dataset is 
    sorted chronologically by time.

    Example:
    ds = mat_to_xr_2D('example.mat', time_name='time', depth_name_in='pres', depth_name_out='PRES')
    '''

    # Read data/metadata from matfile
    data_dict, attr_dict = _parse_matfile_to_dict(matfile)

    # (Attempt to) parse time
    try:
        time_stamp = _parse_time(data_dict, time_name = time_name)
        time_num = date2num(time_stamp)

        # Remove time variable if we successfully parsed time
        data_dict.pop(time_name)
    except:
        print(f'NOTE: Unable to parse time from the {time_name} field.')

    # Collect in an xr Dataset
    ds = xr.Dataset(coords = {'TIME':time_num, 
                              depth_name_out:data_dict[depth_name_in]})

    # Add data variables
    # (Assigning the coordinates by looking at the dimensionality of the fields)
    for varnm, item in data_dict.items():
        dshape = data_dict[varnm].shape  
        if dshape == (ds.dims['TIME'],):
            ds[varnm] = (('TIME'), data_dict[varnm])
        elif dshape == (ds.dims[depth_name_out],):
            ds[varnm] = ((depth_name_out), data_dict[varnm])
        elif dshape == (ds.dims['TIME'], ds.dims[depth_name_out]):
            ds[varnm] = (('TIME', depth_name_out), data_dict[varnm])
        elif dshape == (ds.dims[depth_name_out], ds.dims['TIME']):
            ds[varnm] = (('TIME', depth_name_out), data_dict[varnm].T)
        else:
            print(f'NOTE: Trouble with variable {varnm} (shape: {data_dict[varnm].shape})- does not seem '
                f'to fit into either TIME = ({ds.dims["TIME"]}) '
                f' or {depth_name_out} ({ds.dims[depth_name_out]}).\n-> Skipping this variable')

    # Add metadata
    for attrnm, item in attr_dict.items():
        ds.attrs[attrnm] = attr_dict[attrnm]

    # Sort chronologically
    ds = ds.sortby('TIME')
    
    return ds





def _parse_matfile_to_dict(matfile, unwrap_dict = True):
    '''
    Use scipy.io.matlab to parse a matfile. 

    Will skip variable names containing data of internal Matlab types, 
    e.g. Datetime, which are not accessible outside Matlab.

    Attempts to handle up to three levels of nesting - may not work
    for terribly complex matfiles. 

    Probably only works for <v7.3 (=>7.3 needs its own parser, I think)

    Parameters: 
    - matfile: math to a file with suffix .mat
    - unwrap_struct: If the file contents are distributed across multiple 
      fields:

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
        parsed_variable = ddict[varnm].flatten()[0]


        # If the object is (float, int, str, etc): Interpret is as a global attribute
        if isinstance(parsed_variable, (int, float, str)):
            attr_dict[varnm] = parsed_variable

        # This is none if ddict[varnm] is an array with data or similar
        # -> assign a variable
        elif parsed_variable.dtype.fields==None:

            parsed_data = parsed_variable
            
            # If the variable is of type MatlabOpaque, it cannot be read outside MATLAB
            # -> Print a message and skip the variable
            if isinstance(parsed_data, matlab._mio5_params.MatlabOpaque):
                print(f'NOTE: "{varnm}" is of an internal Matlab class type (e.g. Datetime) '
                    'which cannot be read outside Matlab -> Skip')

            # If the object is iterable (list, array, etc): Interpret is as a data variable
            elif isinstance(parsed_data, (list, tuple, np.ndarray)):
                data_dict[varnm] = parsed_data



        # If ddict[varnm] is an array containing other names variables: parse them
        # -> Loop through variables and add them individualy as variables
        else:
            for varnm_internal in parsed_variable.dtype.names:
                parsed_data = parsed_variable[varnm_internal].flatten()[0]

                # If the variable is of type MatlabOpaque, it cannot be read outside MATLAB
                # -> Print a message and skip the variable
                if isinstance(parsed_data, matlab._mio5_params.MatlabOpaque):
                    print(f'NOTE: "{varnm_internal}" is of an internal Matlab class type (e.g. Datetime) '
                        'which cannot be read outside Matlab -> Skip')

                # If the object is iterable (list, array, etc): Interpret is as a data variable
                elif isinstance(parsed_data, (list, tuple, np.ndarray)):
                    data_dict[varnm_internal] = parsed_data

                # Otherwise (float, int, str, etc): Interpret is as a global attribute
                elif isinstance(parsed_data, (int, float, str)):
                    attr_dict[varnm_internal] = parsed_data



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
        print(f'Unable to parse time from the "{time_name}" variable. '
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



def xr_to_mat(D, outfile, simplify = False):
    """
    Convert an xarray.Dataset to a MATLAB .mat file.
      
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
    >>> xr_to_mat(D, 'output_matfile', simplify=True)
    """
    
    time_epoch = D.TIME.units.upper().replace('DAYS SINCE ', '')[:11]
    time_stamp = time.datenum_to_timestamp(D.TIME, D.TIME.units)
    time_mat = time.timestamp_to_matlab_time(time_stamp)
    
    data_dict = D.to_dict()
    
    if simplify:
        ds = {}
        for sub_dict_name in ['coords', 'data_vars']:
            sub_dict = data_dict[sub_dict_name]
            for varnm, item in sub_dict.items():
                ds[varnm] = sub_dict[varnm]['data']
        
        ds['TIME_mat'] = time_mat
        data_dict = ds
        simple_str = ' (simplified)'
    else:
        data_dict['coords']['TIME_mat'] = time_mat
        simple_str = ''

    if not outfile.endswith('.mat'):
        outfile += '.mat'

        
    matlab.savemat(outfile, data_dict)
    print(f'Saved the{simple_str} Dataset to: {outfile}')