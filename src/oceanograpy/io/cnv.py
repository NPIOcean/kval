### OceanograPy.io.cnv.py ###

'''
CNV parsing.

Including pressure binning -> does that belong here or elsewhere? 
'''

import pandas as pd
import xarray as xr
import numpy as np
from oceanograpy.io import _variable_defs as vardef
from oceanograpy.util import time
import glob2
import matplotlib.pyplot as plt
import re
from typing import Optional
import warnings
from tqdm.notebook import tqdm 
import cftime

def read_cnv(cnvfile: str,
             apply_flags: Optional[bool] = True,
             profile: Optional[str] = 'downcast',
             time_dim: Optional[bool] = False,
             inspect_plot: Optional[bool] = False,
             start_scan: Optional[int] = None,
             end_scan: Optional[int] = None,
             suppress_time_warning: Optional[int] = None, 
             start_time_NMEA: Optional[bool] = True) -> xr.Dataset:
    '''
    Reads CTD data and metadata from a .cnv file into a more handy format.
  
    Parameters
    ----------
    cnvfile: str
        Path to a .cnv file containing the data 

    apply_flags: bool, optional 
        If True, flags (from the SBE *flag* column) are applied as NaNs across
        all variables (recommended). Default is True.
     
    profile: str, optional
        Specify the profile type. Options are ['upcast', 'downcast', 'none'].

    time_dim: bool, optional
        Choose whether to include a 0-D TIME coordinate. Useful if combining
        several profiles. Deffault is False.

    inspect_plot : bool, optional
        Return a plot of the whole pressure time series, showing the part of the profile 
        we extracted (useful for inspecting up/downcast extraction and SBE flags).
        Default is False.

    start_scan : int, optional
        Manually specify the scan at which to start the profile (in *addition* to profile 
        detection and flags). Default is None.

    end_scan : int, optional
        Manually specify the scan at which to end the profile (in *addition* to profile 
        detection and flags). Default is None.
    
    suppress_time_warning: bool, optional
        Don't show a warning if there are no timeJ or timeS fields
        Detault is False.

    start_time_NMEA: bool, optional
        Choose whether to get start_time attribute from the "NMEA UTC (Time)" 
        header line. Default is to grab it from the "start_time" line - this
        is technically correct but typically identical results, and the 
        "start_time" line can occasionally look funny. If unsure, check your 
        header! Default is False (= read from "start_time" header line).        

    Returns
    -------
    xarray.Dataset
        A dataset containing CTD data and associated attributes.

    TO DO
    ----- 
    - Checking and testing
    - Better docs (esp a good docstring for the read_cnv function!)
    - Look into *axis* (T, Z, X, Y) - is this necessary?
    - Apply to some other datasets for testing
        - Maybe also moored sensors/TSG? (or should those be separate?)
    - Figure out whether to make a split between this and a separate processing
      module (e.g. binning, chopping, concatenation)
    - Tests
        - Make a test_ctd_data.cnv file with mock values and use pytest

    '''

    header_info = read_header(cnvfile)
    ds = _read_column_data_xr(cnvfile, header_info)
    ds = _update_variables(ds, cnvfile)
    ds = _convert_time(ds, header_info, 
                       suppress_time_warning = suppress_time_warning)
    ds.attrs['history'] = header_info['start_history']
    ds = _add_start_time(ds, header_info, 
                             start_time_NMEA = start_time_NMEA)
    ds = _read_SBE_proc_steps(ds, header_info)
  #  ds0 = ds.copy()


    if time_dim:
        ds = _add_time_dim_profile(ds)

    if apply_flags:
        ds = _apply_flag(ds)
        ds.attrs['SBE flags applied'] = 'yes'
    else:
        ds.attrs['SBE flags applied'] = 'no'


    if profile in ['upcast', 'downcast', 'dncast']:
        ds = _remove_up_dncast(ds, keep = profile)
    else:
        ds.attrs['profile_direction'] = 'All good data'

    if start_scan:
        ds = _remove_start_scan(ds, start_scan)
    if end_scan:
        ds = _remove_end_scan(ds, end_scan)

    if inspect_plot:
        _inspect_extracted(ds, ds0, start_scan, end_scan)

    ds = _add_attrs(ds, header_info)

    now_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    ds.attrs['history'] += f'\n{now_str}: Post-processing.'

    return ds



def _add_time_dim_profile(ds, epoch = '1970-01-01', 
                          time_source = 'sample_time'):
    '''
    Add a 0-dimensional TIME coordinate to a profile.

    time_source:
    
    sample_time:  Average of TIME_SAMPLE varibale (which was calculated 
                  from timeS or timeJ fields).

    start_time:   Use the start_time field (star of scans). This is extracted
                  from either the header; either the 'start_time' or 
                  'NMEA UTC' lines (which of the twois specified in the 
                  read_cnv() function usng the start_time_NMEA flag).
    '''


    if 'TIME_SAMPLE' in ds:
        ds = ds.assign_coords({'TIME':[ds.TIME_SAMPLE.mean()]})
        ds.TIME.attrs = {'units' : ds.TIME_SAMPLE.units,
            'standard_name' : 'time',
            'long_name':'Average time of measurement',
            'SBE_source_variable':ds.TIME_SAMPLE.SBE_source_variable}
    else:
        start_time_num = time.ISO8601_to_datenum(ds.attrs['start_time'])
        ds = ds.assign_coords({'TIME':[start_time_num]})
        ds.TIME.attrs = {'units' : f'Days since {epoch} 00:00',
                        'standard_name' : 'time',
                        'long_name':'Start time of profile',
                        'comment':f'Source: {ds.start_time_source}'}
    return ds






def join_cruise(nc_files, bins_dbar = 1, verbose = True,
                epoch = '1970-01-01'):
    '''
    Takes a number of cnv profiles, pressure bins them if necessary,
    and joins the profiles into one single file.  

    Inputs:

    nc_files: string, list
        A dictionary containing either:
        1. A path to the location of individual profiles, 
           e. g. 'path/to/files/'. 
        2. A list containing xr.Datasets with the individual profiles. 

    epoch: Only necessary to specify here if there is no TIME_SAMPLE
           field in the data files. (In that case, we assign time based on 
           the profile start time, and need to know the epoch) 
           
        
    '''

    ### PREPARE INPUTS
    # Grab the input data on the form of a list of xr.Datasets
    # (1) Load all from path if we are given a path.
    if isinstance(nc_files, str):
        if nc_files.endswith('/'):
            nc_files = nc_files[:-1]
        file_list = glob2.glob(f'{nc_files}/*.nc')

        # Note: We don't decode CF since we want time as a numerical variable.
        prog_bar_msg_1 = 'Loading profiles from netcdf files'
        ns_input = []
        for file in tqdm(file_list, desc=prog_bar_msg_1):
            ns_input += [xr.open_dataset(file, decode_cf=False)]
        n_profs = len(ns_input)
        
        valid_nc_files = True
        if verbose:
            print(f'Loaded {n_profs} profiles from netcdf files: {file_list}')

    # (2) Load from list of xr.Datasets if that is what we are given.
    elif isinstance(nc_files, list):
        if all(isinstance(nc_file, xr.Dataset) for nc_file in nc_files):
            ns_input = nc_files
            n_profs = len(ns_input)
            valid_nc_files = True
            if verbose:
                print(f'Loaded {n_profs} profiles from list of Datasets.')
        else:
            valid_nc_files = False
    else:
        valid_nc_files = False

    
    # Raise an exception if we don't have either (1) or (2)
    if valid_nc_files == False:
        raise Exception('''            
            Input *nc_files* invalid. Must be either:
            1. A path to the location of individual profiles,
               e.g. "path/to/files/" --or--
            2. A list containing xr.Datasets with the individual profiles,
               e.g. [d1, d2] with d1, d2 being xr.Datasets.
                        ''')

    ### BINNING
    # If unbinned: bin data
    if any(n_input.binned == 'no' for n_input in ns_input):
  
        prog_bar_msg_2 = f'Binning all profiles to {bins_dbar} dbar'
        
        ns_binned = []
        for n_input in tqdm(ns_input, desc = prog_bar_msg_2):
            ns_binned += [bin_to_pressure(n_input, bins_dbar)] 
    else: 
        # replace this with using the warnings module?
        print('NOTE: Input data already binned -> using preexisting binning.')
        ns_binned = ns_input
    

    ### JOINING PROFILES TO ONE DATASET
    #  Concatenation loop.
    # - Assigning a TIME dimension based on average of measurement time (TIME_SAMPLE)
    # - Also adding a STATION variable if we have a station attribute
    # - Add a CRUISE variable 

    first = True

    # Keeping track of these as we want to replace them with a date *range*
    post_proc_times = []
    SBE_proc_times = []

    prog_bar_msg_3 = f'Joining profiles together'

    for n in tqdm(ns_binned, prog_bar_msg_3):

        sbe_timestamp, proc_timestamp = _dates_from_history(n)
        post_proc_times += [proc_timestamp]
        SBE_proc_times += [sbe_timestamp]

        if first:
            if 'TIME_SAMPLE' in n:
                N = n.assign_coords({'TIME':[n.TIME_SAMPLE.mean()]})
                N.TIME.attrs = {'units' : n.TIME_SAMPLE.units,
                                'standard_name' : 'time',
                                'long_name':'Average time of measurement'}
            else:
                start_time_num = time.ISO8601_to_datenum(n.attrs['start_time'])
                N = n.assign_coords({'TIME':[start_time_num]})
                N.TIME.attrs = {'units' : f'Days since {epoch} 00:00',
                                'standard_name' : 'time',
                                'long_name':'Start time of profile'}
                
            # Want to keep track of whether we use different 
            # start_time sources
            start_time_source = n.start_time_source
            different_start_time_sources = False
            first = False

            if 'station' in N.attrs:
                station = N.station
                del N.attrs['station']
                N['STATION'] = xr.DataArray([station], dims='TIME', coords={'TIME': N['TIME']})

        else:
            if 'TIME_SAMPLE' in n:
                n = n.assign_coords({'TIME':[N.TIME_SAMPLE.mean()]})
            else:
                start_time_num = time.ISO8601_to_datenum(n.attrs['start_time'])
                n = n.assign_coords({'TIME':[start_time_num]})

            if 'station' in n.attrs:
                n['STATION'] = xr.DataArray([n.station], dims='TIME', coords={'TIME': n['TIME']})

            if n.start_time_source != start_time_source:
                different_start_time_sources = True

            N = xr.concat([N, n], dim = 'TIME')



    ### FINAL TOUCHES AND EXPORTS
    # Transpose so that variables are structured like (TIME, PRES) rather than
    # (PRES, TIME). Convenient when plotting etc.
    N = N.transpose() 

    # Add some metadata to the STATION variable
    if 'STATION' in N.data_vars:
        N['STATION'].attrs = {'long_name' : 'CTD station ID',
                              'cf_role':'profile_id'}
    
    # Warn the user if we used different sources for profile start time
    if different_start_time_sources:
        print('\nThe start_time variable was read from different '
            'header lines for different profiles. This likely means that '
            'your .cnv files were formatted differently through the cruise. '
            'This is probably fine, but it is a good idea to sanity check '
            ' the TIME field.')

    # Add a cruise variable
    if 'cruise' in N.attrs:
        cruise = N.cruise
    else:
        cruise = '!! CRUISE !!'
        
        print('\nNo cruise ID found in the dataset. Remember to assign!'
                      '\n-> ds = cnv.set_cruise(ds, "cruisename").')
    N['CRUISE'] =  xr.DataArray(cruise, dims=())
    N['CRUISE'].attrs = {'long_name':'Cruise ID', 'cf_role':'trajectory_id'}

    # Generalize (insert "e.g.") in attributes with specific file names 
    N.attrs['source_files'] = f"E.g. {N.attrs['source_files']}"
    SBEproc_file_ind = N.SBE_processing.rfind('Raw data read from ')+19
    N.attrs['SBE_processing'] = (N.SBE_processing[:SBEproc_file_ind]
                + 'e.g. ' + N.SBE_processing[SBEproc_file_ind:])

    # Add date *ranges* to the history entries if we have different dates
    N = _replace_history_dates_with_ranges(N, post_proc_times, SBE_proc_times)

    # Delete some non-useful attributes
    del N.attrs['SBE_processing_date']
    del N.attrs['start_time_source']

    # Set the featureType attribute
    N.attrs['featureType'] = 'TrajectoryProfile'

    # Sort chronologically
    N = N.sortby('TIME')
    
    return N
    



def set_cruise(D, cruise_string):
    '''
    Assign a value to the CRUISE variable in a joined 
    CTD dataset D.

    E.g.:  D = cnv.set_cruise(D, 'cruisename') 
    '''
    cr_attrs = D.CRUISE.attrs
    D['CRUISE'] = cruise_string
    D['CRUISE'].attrs = cr_attrs 
    return D



def bin_to_pressure(ds, dp = 1):
    '''
    Apply pressure binning into bins of *dp* dbar.
    Reproducing the SBE algorithm as documented in:
    https://www.seabird.com/cms-portals/seabird_com/
    cms/documents/training/Module13_AdvancedDataProcessing.pdf
    # Provides not a bin *average* but a*linear estimate of variable at bin
    pressure* (in practice a small but noteiceable difference)
    (See page 13 for the formula used)
    No surface bin included.
    Equivalent to this in SBE terms (I think)
    # binavg_bintype = decibars
    # binavg_binsize = *dp*
    # binavg_excl_bad_scans = yes
    # binavg_skipover = 0
    # binavg_omit = 0
    # binavg_min_scans_bin = 1
    # binavg_max_scans_bin = 2147483647
    # binavg_surface_bin = no, min = 0.000, max = 0.000, value = 0.000
    '''

    # Tell xarray to conserve attributes across operations
    # (we will set this value back to whatever it was after the calculation)
    _keep_attr_value = xr.get_options()['keep_attrs']
    xr.set_options(keep_attrs=True)

    # Define the bins over which to average
    pmax = float(ds.PRES.max())
    pmax_bound = np.floor(pmax-dp/2)+dp/2

    pmin = float(ds.PRES.min())
    pmin_bound = np.floor(pmin+dp/2)-dp/2

    p_bounds = np.arange(pmin_bound, pmax_bound+1e-9, dp) 
    p_centre = np.arange(pmin_bound, pmax_bound, dp)+dp/2

    # Pressure averaged 
    ds_pavg = ds.groupby_bins('PRES', bins = p_bounds).mean()

    # Get pressure *binned* according to formula on page 13 in SBEs module 13 document

    ds_curr = ds_pavg.isel({'PRES_bins':slice(1, None)})
    ds_prev = ds_pavg.isel({'PRES_bins':slice(None, -1)})
    # Must assign the same coordinates in order to be able to matrix multiply
    ds_prev.coords['PRES_bins'] =  ds_curr.PRES_bins

    p_target = p_centre[slice(1, None)]
    _numerator = ds_pavg.diff('PRES_bins')*(p_target - ds_prev.PRES)
    _denominator = ds_pavg.PRES.diff('PRES_bins')

    ds_binned = _numerator/_denominator + ds_prev


    # Replace the PRES_bins coordinate and dimension
    # with PRES
    ds_binned = (ds_binned
        .swap_dims({'PRES_bins':'PRES'})
        .drop_vars('PRES_bins'))
    
    ds_binned.attrs['binned'] = f'{dp} dbar'

    # Set xarray option "keep_attrs" back to whatever it was
    xr.set_options(keep_attrs=_keep_attr_value)

    return ds_binned




def read_header(cnvfile):
    '''
    Reads a SBE .cnv (or .hdr, .btl) file and returns a dictionary with various
    metadata parameters extracted from the header. 

    TBD: Grab instrument serial numbers.
    '''
    
    with open(cnvfile, 'r', encoding = 'latin-1') as f:

        # Empty dictionary: Will fill these parameters up as we go
        hkeys = ['col_nums', 'col_names', 'col_longnames', 'SN_info', 
                 'moon_pool', 'SBEproc_hist', 'hdr_end_line', 
                 'latitude', 'longitude', 'NMEA_time', 'start_time', ]
        hdict = {hkey:[] for hkey in hkeys}

        # Flag that will be turned on when we read the SBE history section 
        start_read_history = False 

        # Go through the header line by line and extract specific information 
        # when we encounter specific terms dictated by the format  
        
        for n_line, line in enumerate(f.readlines()):

            # Read the column header info (which variable is in which data column)
            if '# name' in line:
                # Read column number
                hdict['col_nums'] += [int(line.split()[2])]
                # Read column header
                col_name = line.split()[4].replace(':', '')
                col_name = col_name.replace('/', '_') #  "/" in varnames not allowed in netcdf 
                hdict['col_names'] += [col_name]
                # Read column longname
                hdict['col_longnames'] += [' '.join(line.split()[5:])]            

            # Read NMEA lat/lon/time
            if 'NMEA Latitude' in line:
                hdict['latitude'] = _nmea_lat_to_decdeg(*line.split()[-3:])
            if 'NMEA Longitude' in line:
                hdict['longitude'] = _nmea_lon_to_decdeg(*line.split()[-3:])
            if 'NMEA UTC' in line:
                nmea_time_split = line.split()[-4:]
                hdict['NMEA_time'] = _nmea_time_to_datetime(*nmea_time_split)
            
            # Read start time
            if 'start_time' in line:
                hdict['start_time'] = (
                    _nmea_time_to_datetime(*line.split()[3:7]))

                hdict['start_history'] = (
                    hdict['start_time'].strftime('%Y-%m-%d')
                    + ': Data collection.')

            # Read cruise/ship/station/bottom depth/operator if available
            if '** CRUISE' in line.upper():
                hdict['cruise'] = line[(line.rfind(': ')+2):].replace('\n','')
            if '** STATION' in line.upper():
                hdict['station'] = line[(line.rfind(': ')+2):].replace('\n','')
            if '** SHIP' in line.upper():
                hdict['ship'] = line[(line.rfind(': ')+2):].replace('\n','')
            if '** BOTTOM DEPTH' in line.upper():
                hdict['bdep'] = line[(line.rfind(': ')+2):].replace('\n','')

            # Read moon pool info
            if 'Skuteside' in line:
                mp_str = line.split()[-1]
                if mp_str == 'M':
                    hdict['moon_pool'] = True
                elif mp_str == 'S':
                    hdict['moon_pool'] = False

            # At the end of the SENSORS section: read the history lines
            if '</Sensors>' in line:
                start_read_history = True

            if start_read_history:
                hdict['SBEproc_hist'] += [line] 

            # Read the line containing the END string
            # (and stop reading the file after that)
            if '*END*' in line:
                hdict['hdr_end_line'] = n_line
                break

        # Remove the first ('</Sensors>') and last ('*END*') lines from the SBE history string.
        hdict['SBEproc_hist'] = hdict['SBEproc_hist'] [1:-1]

        # Assign the file name without the directory path

        hdict['cnvfile'] = cnvfile[cnvfile.rfind('/')+1:]

        return hdict


def _read_column_data_xr(cnvfile, header_info):
    '''
    Reads columnar data from a single .cnv to an xarray Dataset.

    (By way of a pandas DataFrame)

    TBD: 
     - Standardize variable names and attributes. 
     - Add relevant attributes from header

    '''
    df = pd.read_csv(cnvfile, header = header_info['hdr_end_line']+1,
                 delim_whitespace=True, encoding = 'latin-1',
                 names = header_info['col_names'])
    
    # Convert to xarray DataFrame
    ds = xr.Dataset(df).rename({'dim_0':'scan_count'})
    ds.attrs['binned'] = 'no'


    return ds


def _add_start_time(ds, header_info, start_time_NMEA=False):
    '''
    Add a start_time attribute.

    Default behavior: 
        - Use start_time header line
    If start_time_NMEA = True:
        - Use "NMEA UTC" line if present
        - If not, use start_time
    
    Compicated way of doing this because there are some occasional 
    oddities where e.g. 
    - The "start_time" line is some times incorrect
    - The "NMEA UTC" is not always present.

    Important to get right since this is used for assigning
    a time stamp to profiles. 
    '''

    if start_time_NMEA:
        try:
            ds.attrs['start_time'] = time.datetime_to_ISO8601(
                            header_info['NMEA_time'])
            ds.attrs['start_time_source'] = '"NMEA UTC" header line'

        except:
            try:
                ds.attrs['start_time'] = time.datetime_to_ISO8601(
                    header_info['start_time'])
                ds.attrs['start_time_source'] = '"start_time" header line'

            except:
                raise Warning('Did not find a start time!'
                    ' (no "start_time" or NMEA UTC" header lines).')
    else:
        ds.attrs['start_time'] = time.datetime_to_ISO8601(
                            header_info['start_time'])
        ds.attrs['start_time_source'] = '"start_time" header line'

    return ds


def _add_attrs(ds, header_info):
    '''
    Add the following as attributes if they are available from the header:

        ship, cruise, station.

    If we don't have a station, we use the cnv file name base.
    '''

    for key in ['ship', 'cruise', 'station']:
        if key in header_info:
            ds.attrs[key] = header_info[key]

    if 'station' not in ds.attrs:
        station_from_filename = (
            header_info['cnvfile'].replace(
            '.cnv', '').replace('.CNV', ''))
        ds.attrs['station'] = station_from_filename

    return ds



def _apply_flag(ds):
    '''
    Applies the *flag* value assigned by the SBE processing.

    -> Remove scans  where flag != 0 
    
    '''
    ds = ds.where(ds.SBE_FLAG==0, drop = True)      

    return ds


def _remove_up_dncast(ds, keep = 'downcast'):
    '''
    Takes a ctd Dataframe and returns a subset containing only either the
    upcast or downcast.

    Note:
    Very basic algorithm right now - just removing everything before/after the
    pressure max and relying on SBE flaggig for the rest.
    -> will likely have to replace with something more sophisticated. 
    '''
    # Index of max pressure, taken as "end of downcast"
    max_pres_index = int(ds.PRES.argmax().data)

    # If the max index is a single value following invalid values,
    # we interpret it as the start of the upcast and use the preceding 
    # point as the "end of upcast index" 
    if (ds.scan_count[max_pres_index] 
        - ds.scan_count[max_pres_index-1]) > 1:
        
        max_pres_index -= 1

    if keep == 'upcast':
        # Remove everything *after* pressure max
        ds = ds.isel({'scan_count':slice(max_pres_index+1, None)})
        ds.attrs['profile_direction'] = 'upcast'

    elif keep in ['dncast', 'downcast']:
        # Remove everything *before* pressure max
        ds = ds.isel({'scan_count':slice(None, max_pres_index+1)})
        ds.attrs['profile_direction'] = 'downcast'

    else:
        raise Exception('"keep" must be either "upcast" or "dncast"') 

    return ds


def _inspect_extracted(ds, ds0, start_scan=None, end_scan=None):
    '''
    Plot the pressure tracks showing the portion extracted after 
    and/or removing upcast.
    '''

    fig, ax = plt.subplots(figsize = (10, 6))

    ax.plot(ds0.scan_count, ds0.PRES, '.k', ms = 3, label = 'All scans')
    ax.plot(ds.scan_count, ds.PRES, '.r', ms = 4, label ='Extracted for use')

    if start_scan:
        ax.axvline(start_scan, ls = '--', zorder = 0, label = 'start_scan')
    if end_scan:
        ax.axvline(end_scan, ls = '--', zorder = 0, label = 'end_scan')
    ax.set_ylabel('PRES [dbar]')
    ax.set_xlabel('SCAN COUNTS')
    ax.legend()
    ax.invert_yaxis()
    ax.grid(alpha = 0.5)
    plt.show()

def _update_variables(ds, cnvfile):
    '''
    Take a Dataset and 
    - Update the header names from SBE names (e.g. 't090C')
      to more standardized name (e.g., 'TEMP1'). 
    - Assign the appropriate units and standard_name as attributes.
    - Assign sensor serial number(s) and/or calibration date(s)
      where available.

    What to look for is specified in _variable_defs.py
    -> Will update dictionaries in there as we encounter differently
       formatted files.
    '''
    
    sensor_info = _read_sensor_info(cnvfile)

    for old_name in ds.keys():
        old_name_cap = old_name.upper()
        if old_name_cap in vardef.SBE_name_map:

            var_dict = vardef.SBE_name_map[old_name_cap]

            new_name = var_dict['name']
            unit = var_dict['units']
            ds = ds.rename({old_name:new_name})
            ds[new_name].attrs['units'] = unit

            if 'sensors' in var_dict:
                sensor_SNs = []
                sensor_caldates = []

                for sensor in var_dict['sensors']:
                    sensor_SNs += [sensor_info[sensor]['SN']]
                    sensor_caldates += [sensor_info[sensor]['cal_date']]

                ds[new_name].attrs['sensor_serial_number'] = (
                    ', '.join(sensor_SNs))

                ds[new_name].attrs['sensor_calibration_date'] = (
                    ', '.join(sensor_caldates)) 
    return ds

def _read_sensor_info(cnvfile, verbose = False):
    '''
    Look through the header for information about sensors:
        - Serial numbers
        - Calibration dates  
    '''

    sensor_dict = {}

    # Define stuff we want to remove from strings
    drop_str_patterns = ['<.*?>', '\n', ' NPI']
    drop_str_pattern_comb = '|'.join(drop_str_patterns)

    with open(cnvfile, 'r', encoding = 'latin-1') as f:
        look_sensor = False

        # Loop through header lines 
        for n_line, line in enumerate(f.readlines()):

            # When encountering a <sensor> flag: 
            # Begin looking for instrument info
            if '<sensor' in line:
                # Set initial flags
                look_sensor = True
                sensor_header_line = n_line+1
                store_sensor_info = False

            if look_sensor:
                # Look for an entry corresponding to the sensor in the 
                # _sensor_info_dict (prescribed) dictionary
                # If found: read info. If not: Ignore.
                if n_line == sensor_header_line:
                    for sensor_str, var_key in vardef.sensor_info_dict.items():
                        if sensor_str in line:
                            store_sensor_info = True
                            var_key_sensor = var_key
                    
                    # Print if verbose
                    shline = line.replace('#     <!-- ', '').replace('\n', '')
                    (print(f'\nRead from: {var_key_sensor} ({shline})') 
                    if verbose else None)

                
                if store_sensor_info:
                    # Grab instrument serial number
                    if '<SerialNumber>' in line:
                        rind_sn = line.rindex('<SerialNumber>')+14

                        SN_instr = re.sub(drop_str_pattern_comb, '', 
                                               line[rind_sn:])
                       # SN_instr = (line[rind_sn:]
                       #             .replace('</SerialNumber>', '')
                       #             .replace('\n', '')
                       #             .replace(' NPI', ''))
                    # Grab calibration date
                    if '<CalibrationDate>' in line:
                        rind_cd = line.rindex('<CalibrationDate>')+17
                        cal_date_instr = (line[rind_cd:]
                                    .replace('</CalibrationDate>', '')
                                    .replace('\n', ''))
            
            # When encountering a <sensor> flag: 
            # Stop looking for instrument info and store
            # in dictionary
            if '</sensor>' in line:

                # Store to dictionary
                if look_sensor and store_sensor_info:
                    sensor_dict[var_key_sensor] = {
                        'SN':SN_instr,
                        'cal_date':cal_date_instr
                    }

                # Print if verbose
                (print(f'SN: {SN_instr}  // cal date: {cal_date_instr}+n',
                       f'Stop reading from {var_key_sensor} (save: {store_sensor_info})') 
                if verbose else None)

                # Reset flags
                look_sensor, var_key_sensor, = False, None
                SN_instr, cal_date_instr = None, None


            # Stop reading after the END string
            if '*END*' in line:
                return sensor_dict

    

def _read_SBE_proc_steps(ds, header_info):
    '''
    Parse the information about SBE processing steps from the cnv header into 
    a more easily readable format and storing the information as the global
    variable *SBE_processing*.

    Also:
    - Adds a *SBE_processing_date* global variable (is this useful?)
    - Adds a *source_files* variable with the names of the .hex, .xmlcon,
      and .cnv files
    - Appends SBE processing history line to the *history* attribute.
    '''
    SBElines = header_info['SBEproc_hist']

    dmy_fmt = '%Y-%m-%d'

    sbe_proc_str = ['SBE SOFTWARE PROCESSING STEPS (extracted'
                    ' from .cnv file header):',]

    for line in SBElines:
        # Get processing date
        if 'datcnv_date' in line:
            proc_date_str = re.search(r'(\w{3} \d{2} \d{4} \d{2}:\d{2}:\d{2})',
                                line).group(1)
            proc_date = pd.to_datetime(proc_date_str) 
            proc_date_ISO8601 = time.datetime_to_ISO8601(proc_date) 
            proc_date_dmy = proc_date.strftime(dmy_fmt)
            history_str = (f'{proc_date_dmy}: Processed using'
                            ' SBE software (details in "SBE_processing").')

        # Get input file names (without paths)
        if 'datcnv_in' in line:
            hex_fn = re.search(r'\\([^\\]+\.HEX)', line.upper()).group(1)
            xmlcon_fn = re.search(r'\\([^\\]+\.XMLCON)', line.upper()).group(1)
            src_files_raw = f'{hex_fn}, {xmlcon_fn}'
            sbe_proc_str += [f'- Raw data read from {hex_fn}, {xmlcon_fn}.']

        # SBE processing details
        # Get skipover scans
        if 'datcnv_skipover' in line:
            skipover_scans = int(re.search(r'= (\d+)', line).group(1))
            if skipover_scans != 0:
                sbe_proc_str += [f'- Skipped over {skipover_scans} initial scans.']

        # Get ox hysteresis correction 
        if 'datcnv_ox_hysteresis_correction' in line:
            ox_hyst_yn = re.search(r'= (\w+)', line).group(1)
            if ox_hyst_yn == 'yes':
                sbe_proc_str += [f'- Oxygen hysteresis correction applied.']

        # Get ox tau correction 
        if 'datcnv_ox_tau_correction' in line:
            ox_hyst_yn = re.search(r'= (\w+)', line).group(1)
            if ox_hyst_yn == 'yes':
                sbe_proc_str += [f'- Oxygen tau correction applied.']

        # Get low pass filter details
        if 'filter_low_pass_tc_A' in line:
            lp_A = float(re.search(r' = (\d+\.\d+)', line).group(1))
        if 'filter_low_pass_tc_B' in line:
            lp_B = float(re.search(r' = (\d+\.\d+)', line).group(1))
        if 'filter_low_pass_A_vars' in line:
            try:
                lp_vars_A = re.search(r' = (.+)$', line).group(1).split()
                sbe_proc_str += [f'- Low-pass filter with time constant {lp_A}'
                    + f' seconds applied to: {" ".join(lp_vars_A)}.']
            except: 
                print('FYI: Looks like filter A was not applied to any variables.')
        if 'filter_low_pass_B_vars' in line:
            try:
                lp_vars_B = re.search(r' = (.+)$', line).group(1).split()
                sbe_proc_str += [f'- Low-pass filter with time constant {lp_B}'
                    + f' seconds applied to: {" ".join(lp_vars_B)}.']
            except:
                print('FYI: Looks like filter A was not applied to any variables.')

        # Get cell thermal mass correction details 
        if 'celltm_alpha' in line:
            celltm_alpha = re.search(r'= (.+)$', line).group(1)
        if 'celltm_tau' in line:
            celltm_tau= re.search(r'= (.+)$', line).group(1)
        if 'celltm_temp_sensor_use_for_cond' in line:
            celltm_sensors = re.search(r'= (.+)$', line).group(1)
            sbe_proc_str += ['- Cell thermal mass correction applied to conductivity' 
                 f' from sensors: [{celltm_sensors}]. ',
                 f'   > Parameters ALPHA: [{celltm_alpha}], TAU: [{celltm_tau}].']

        # Get loop edit details
        if 'loopedit_minVelocity' in line:
            loop_minvel = re.search(r'= (\d+(\.\d+)?)', line).group(1)                
        if 'loopedit_surfaceSoak' in line and float(loop_minvel)>0:
            loop_ss_mindep = re.search(r'minDepth = (\d+(\.\d+)?)', line).group(1)
            loop_ss_maxdep = re.search(r'maxDepth = (\d+(\.\d+)?)', line).group(1)
            loop_ss_deckpress = re.search(r'useDeckPress = (\d+(\.\d+)?)', line).group(1)
            if loop_ss_deckpress=='0':
                loop_ss_deckpress_str = 'No'
            else:
                loop_ss_deckpress_str = 'Yes'
        if 'loopedit_excl_bad_scans' in line and float(loop_minvel)>0:
            loop_excl_bad_scans = re.search(r'= (.+)', line).group(1)
            if loop_excl_bad_scans == 'yes':
                loop_excl_str = 'Bad scans excluded'
            else:
                loop_excl_str = 'Bad scans not excluded'
            sbe_proc_str += ['- Loop editing applied.',
                 (f'   > Parameters: Minimum velocity (ms-1): {loop_minvel}, '
                  f'Soak depth range (m): {loop_ss_mindep} to {loop_ss_maxdep}, '
                  + f'\n   > {loop_excl_str}. '
                  + f'Deck pressure offset: {loop_ss_deckpress_str}.')]

        # Get wild edit details
        if 'wildedit_date' in line:
            sbe_proc_str += ['- Wild editing applied.']
        if 'wildedit_vars' in line:
            we_vars = re.search(r' = (.+)$', line).group(1).split()
            sbe_proc_str += [f'   > Applied to variables: {" ".join(we_vars)}.']
        if 'wildedit_pass1_nstd' in line:
            we_pass1 = float(re.search(r' = (\d+\.\d+)', line).group(1))
        if 'wildedit_pass2_nstd' in line:
            we_pass2 = float(re.search(r' = (\d+\.\d+)', line).group(1))
        if 'wildedit_pass2_mindelta' in line:
            we_mindelta = float(re.search(r' = (\d+\.\d+)', line).group(1))
        if 'wildedit_npoint' in line:
            we_npoint = float(re.search(r' = (\d+)', line).group(1))
            sbe_proc_str += [(f'   > Parameters: n_std (first pass): {we_pass1}, '
                f'n_std (second pass): {we_pass2}, min_delta: {we_mindelta},\n'
                f'   > # points per test: {we_npoint}.')]

        # Get align CTD details
        if 'alignctd_adv' in line:
            # Find all matches in the string
            matches = re.findall(r'(\w+)\s+([0-9.]+)', line)
            # Rerutn a list of tuples with (variable, advance time in seconds)
            align_tuples = [(key, float(value)) for key, value in matches]
            sbe_proc_str += ['- Misalignment correction applied.']
            sbe_proc_str += [f'   > Parameters [variable (advance time, sec)]:']
            align_str = []
            for align_tuple in align_tuples:
                align_str += [f'{align_tuple[0]} ({align_tuple[1]})'] 
            sbe_proc_str += [f'   > {", ".join(align_str)}']

        # Get bin averaging details
        if 'binavg_bintype' in line:
            bin_unit = re.search(r' = (.+)$', line).group(1)
        if 'binavg_binsize' in line:
            bin_size = re.search(r' = (.+)$', line).group(1)
        if 'binavg_excl_bad_scans' in line:
            binavg_excl_bad_scans = re.search(r'= (.+)', line)
            if binavg_excl_bad_scans == 'yes':
                binavg_excl_str = 'Bad scans excluded'
            else:
                binavg_excl_str = 'Bad scans not excluded'
        if 'binavg_skipover' in line:
            bin_skipover = re.search(r' = (.+)$', line).group(1)
            if bin_skipover != 0:
                bin_skipover_str = f', skipped over {bin_skipover} initial scans'
            else:
                bin_skipover = ''
        if 'binavg_surface_bin' in line:
            surfbin_yn = re.search(r'surface_bin = (.+)$', line).group(1).split()
            if surfbin_yn != 'yes':
                surfbin_str = '(No surface bin)'
            else:
                surfbin_params = re.search(
                    r'yes, (.+)$', line).group(1).split().upper()
                surfbin_str = f'Surface bin parameters: {surfbin_params}'
            sbe_proc_str += [f'- Bin averaged ({bin_size} {bin_unit}).']
            sbe_proc_str += [f'   > {binavg_excl_str}{bin_skipover_str}.']
            sbe_proc_str += [f'   > {surfbin_str}.']
            SBE_binned = f'{bin_size} {bin_unit} (SBE software)'
        else:
            SBE_binned = 'no'

    ds.attrs['binned'] = SBE_binned
    ds.attrs['SBE_processing'] = '\n'.join(sbe_proc_str)
    ds.attrs['SBE_processing_date'] = proc_date_ISO8601
    ds.attrs['history'] += f'\n{history_str}'
    ds.attrs['source_files'] = f'{src_files_raw} -> {header_info["cnvfile"].upper()}'

    return ds

### UTILITY FUNCTIONS

def _nmea_lon_to_decdeg(deg_str, min_str, EW_str):
    '''
    Convert NMEA longitude to decimal degrees longitude.
    
    E.g.:
    
    ['006', '02.87', 'E'] (string) --> 6.04783333 (float)
    '''

    if EW_str=='E':
        dec_sign = 1
    elif EW_str=='W':
        dec_sign = -1

    decdeg = int(deg_str) + float(min_str)/60 

    return decdeg*dec_sign


def _nmea_lat_to_decdeg(deg_str, min_str, NS_str):
    '''
    Convert NMEA latitude to decimal degrees latitude.
    
    E.g.:
    
    ['69', '03.65', 'S'] (string) --> -69.060833 (float)
    '''
    
    if NS_str=='N':
        dec_sign = 1
    elif NS_str=='S':
        dec_sign = -1

    decdeg = int(deg_str) + float(min_str)/60 

    return decdeg*dec_sign


def _nmea_time_to_datetime(mon, da, yr, hms):
    '''
    Convert NMEA time to datetime timestamp.
    
    E.g.:
    
    ['Jan', '05', '2021', '15:58:23']  --> Timestamp('2021-01-05 15:58:23')
    '''
    nmea_time_dt = pd.to_datetime(f'{mon} {da} {yr} {hms}')
    
    return nmea_time_dt




def _replace_history_dates_with_ranges(D, post_proc_times, SBE_proc_times):
    '''
    When joining files: Change the history strng to show time *ranges*,
    e.g. 
       2017-09-24: Data collection
    -> 2017-09-24 to 2017-09-24: Data collection
    '''
    ppr_min, ppr_max = np.min(post_proc_times), np.max(post_proc_times)
    sbe_min, sbe_max = np.min(SBE_proc_times), np.max(SBE_proc_times)
    ctd_min, ctd_max = D.TIME.min(), D.TIME.max()

    date_fmt = '%Y-%m-%d'

    if ctd_max>ctd_min:
        ctd_range = (
            f'{cftime.num2date(ctd_min, D.TIME.units).strftime(date_fmt)}'
            f' to {cftime.num2date(ctd_max, D.TIME.units).strftime(date_fmt)}')
        D.attrs['history'] = ctd_range + D.history[10:] 

    if sbe_max>sbe_min:
        sbe_range = f'{sbe_min.strftime(date_fmt)} to {sbe_max.strftime(date_fmt)}'
        rind = D.history.find(': Processed using SBE')
        D.attrs['history'] = D.history[:rind-10] + sbe_range + D.attrs['history'][rind:]

    if ppr_max>ppr_min:
        ppr_range = f'{ppr_min.strftime(date_fmt)} to {ppr_max.strftime(date_fmt)}'
        rind = D.history.find(': Post-processing.')
        D.attrs['history'] = D.history[:rind-10] + ppr_range + D.attrs['history'][rind:]
        
    return D


def _dates_from_history(ds):
    '''
    Grab the dates of 1) SBE processing and 2) Post-processing from the
    "history" attribute.
    '''
    sbe_pattern = r"(\d{4}-\d{2}-\d{2}): Processed using SBE software"
    sbe_time_match_str = re.search(sbe_pattern, ds.history).group(1)
    sbe_timestamp = pd.Timestamp(sbe_time_match_str)
    
    proc_pattern = r"(\d{4}-\d{2}-\d{2}): Post-processing"
    proc_time_match_str = re.search(proc_pattern, ds.history).group(1)
    proc_timestamp = pd.Timestamp(proc_time_match_str)

    return sbe_timestamp, proc_timestamp



def _convert_time(ds, header_info, epoch = '1970-01-01', 
                  suppress_time_warning = False):
    '''
    Convert time either from julian days (TIME_JULD extracted from 'timeJ') 
    or from time elapsed in seconds (TIME_ELAPSED extracted from timeS). 

    suppress_time_warning: Don't show a warning if there are no 
    timeJ or timeS fields (useful for loops etc).

    '''

    if 'TIME_ELAPSED' in ds.keys():
        ds = _convert_time_from_elapsed(ds, header_info, epoch = epoch)
        ds.TIME_SAMPLE.attrs['SBE_source_variable'] = 'timeS' 
    elif 'TIME_JULD' in ds.keys():
        ds = _convert_time_from_juld(ds, header_info, epoch = epoch)
        ds.TIME_SAMPLE.attrs['SBE_source_variable'] = 'timeJ' 

    else:
        if not suppress_time_warning:
            print('\nNOTE: Failed to extract sample time info '
                      '(no timeS or timeJ in .cnv file).'
                      '\n(Not a big problem, the start_time '
                      'can be used to assign a profile time).')
    return ds


def _convert_time_from_elapsed(ds, header_info, epoch = '1970-01-01'):
    '''
    Convert TIME_ELAPSED (sec)
    to TIME_SAMPLE (days since 1970-01-01)
    
    Only sensible reference I could fnd is here;
    https://search.r-project.org/CRAN/refmans/oce/html/read.ctd.sbe.html

    (_DSE: time since epoch)
    '''

    start_time_DSE = ((header_info['start_time'] 
                                    - pd.Timestamp(epoch))
                                    / pd.to_timedelta(1, unit='D'))
    
    elapsed_time_days = ds.TIME_ELAPSED/86400

    time_stamp_DSE = start_time_DSE + elapsed_time_days

    ds['TIME_SAMPLE'] = time_stamp_DSE
    ds.TIME_SAMPLE.attrs['units'] = f'Days since {epoch} 00:00:00'
    ds = ds.drop('TIME_ELAPSED')

    return ds


def _convert_time_from_juld(ds, header_info, epoch = '1970-01-01'):
    '''
    Convert TIME_ELAPSED (sec)
    to TIME (days since 1970-01-01)
    
    Only sensible reference I could fnd is here;
    https://search.r-project.org/CRAN/refmans/oce/html/read.ctd.sbe.html

    (_DSE: time since epoch)
    '''

    year_start = header_info['start_time'].replace(month=1, day=1, 
                        hour=0, minute=0, second=0)
    time_stamp = pd.to_datetime(ds.TIME_JULD-1, origin=year_start, 
                            unit='D', yearfirst=True, ).round('1s')    
    time_stamp_DSE = ((time_stamp- pd.Timestamp(epoch))
                                    / pd.to_timedelta(1, unit='D'))

    ds['TIME_SAMPLE'] = (('scan_count'), time_stamp_DSE)
    ds.TIME_SAMPLE.attrs['units'] = f'Days since {epoch} 00:00:00'
    ds = ds.drop('TIME_JULD')

    return ds


def _remove_start_scan(ds, start_scan):
    '''
    Remove all scans before *start_scan*
    '''
    ds = ds.sel({'scan_count':slice(start_scan, None)})
    return ds


def _remove_end_scan(ds, end_scan):
    '''
    Remove all scans after *end_scan*
    '''
    ds = ds.sel({'scan_count':slice(None, end_scan+1)})
    return ds