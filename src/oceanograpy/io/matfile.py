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

def mat_to_xr(matfile, struct_name = 'CTD', pressure_name = 'PR'):
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

    matlab.loadmat(matfile, squeeze_me = True) 
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
        #    data_dict[varnm] = data_dict[varnm][0]
       # data_dict[varnm] = list(data_dict[varnm])
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