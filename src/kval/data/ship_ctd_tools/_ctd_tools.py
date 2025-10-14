'''
## kval.data.ship_ctd.tools

Various functions for making modifications to CTD dataframes in the
format produced by kval.file.cnv:

- Joining insividual profiles into one ship file.
- Pressure binning.
- Chopping.
- Applying offsets.

'''

from kval.file import sbe
from kval.util import xr_funcs

import numpy as np
import xarray as xr
from tqdm.notebook import tqdm
import glob2
import re
import pandas as pd
import cftime
from collections import OrderedDict
from pathlib import Path

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

    NOTE. Should rename *nc_files*! ("datsets"?)
    '''
    ### PREPARE INPUTS
    # Grab the input data on the form of a list of xr.Datasets
    # (1) Load all from path if we are given a path.
    if isinstance(nc_files, str):
        for nn in np.arange(len(nc_files)):
            if nc_files[nn].endswith('/'):
                nc_files[nn] = nc_files[nn][:-1]
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
        print('NOTE: It seems the input data already binned ->'
              ' using preexisting binning.')

        ns_binned = [
            xr_funcs.swap_var_coord(n, 'scan_count', 'PRES', drop_original=True)
                    for n in ns_input]


    ### CHECK FOR DUPLICATE PRESSURE LABELS
    drop_inds = [] # Profiles we will drop
    for ii, n in enumerate(ns_binned):

        # Check whether we have dupliactes
        pres_arr = n.PRES.to_numpy()
        has_duplicate = len(np.unique(pres_arr)) < len(pres_arr)

        # If duplicates exist: Prompt user for further action
        if has_duplicate:
            print(f'The file from station {n.station} appears to contain'
                ' duplicate pressure values!\n(This is unexcpected since we '
                ' expected pressure binned data - good idea to inspect the .cnv file!)'
                'Please select one of the following (number + ENTER):'
                '\n1: Remove duplicates (keep first entry).',
                '\n2: Remove duplicates (keep last entry).',
                "\n3: Don't include this profile.")

            user_input = input('\nInput choice: ')
            if user_input=='1':
                ns_binned[ii] = n.drop_duplicates(dim="PRES", keep='first')
            elif user_input=='2':
                ns_binned[ii] = n.drop_duplicates(dim="PRES", keep='last')
            elif user_input=='3':
                drop_inds += [ii]
            else:
                raise Exception(f'Invalid input "{user_input}. '
                                'Must be "1", "2", or "3"')

    if len(drop_inds)>0:
        for drop_ind in sorted(drop_inds, reverse=True):
            ns_binned.pop(drop_ind)


    ### JOINING PROFILES TO ONE DATASET
    #  Concatenation loop.
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
            N = n
            # Want to keep track of whether we use different
            # start_time sources
            start_time_source = n.start_time_source
            different_start_time_sources = False
            first = False
        else:
            if n.start_time_source != start_time_source:
                different_start_time_sources = True
            N = xr.concat([N, n], dim = 'TIME')



    ### CHECK IF ANY SENSORS CHANGED
    # Modify metadata if they did
    for varnm in list(N.data_vars) + ['PRES']:
        caldates, sns, stations = [], [], []

        for n in nc_files:
            stations += [n.STATION.values[0]]
            for _list, _attr  in zip([caldates, sns, ],
                ['sensor_calibration_date', 'sensor_serial_number',]):
                if varnm in n.keys():
                    if _attr in n[varnm].attrs:
                        _list += [n[varnm].attrs[_attr]]
                    else:
                        _list += [None]
                else:
                    _list += [None]


        sns_filtered = np.array([value for value in sns if value is not None])
        sns_unique = np.array(list(OrderedDict.fromkeys(sns_filtered)))
        caldates_filtered = np.array([value for value in caldates if value is not None])
        caldates_unique = np.array(list(OrderedDict.fromkeys(caldates_filtered)))

        if len(sns_unique)==2:

            # Handle case where two different sensors have the same calibration
            # dates (make a duplicate entry in caldates_unique)
            if len(caldates_unique)==1:
                caldates_unique = [caldates_unique] + [caldates_unique]

            if 'comment' in N[varnm].attrs:
                comment_0 = N[varnm].comment
            else:
                comment_0 = ''
            N[varnm].attrs['comment'] = comment_0 + ' Sensors changed underway.'
            where_sensor_A = np.array(stations)[np.array(sns)==sns_unique[0]]
            where_sensor_B = np.array(stations)[np.array(sns)==sns_unique[1]]
            N[varnm].attrs['stations_A'] = ', '.join(where_sensor_A)
            N[varnm].attrs['stations_B'] = ', '.join(where_sensor_B)
            N[varnm].attrs['sensor_calibration_date'] = (
                f'A: {caldates_unique[0]}, B: {caldates_unique[1]}')
            N[varnm].attrs['sensor_serial_number'] = (
                f'A: {sns_unique[0]}, B: {sns_unique[1]}')
        elif len(sns_unique)>2:
            print(f'NOTE: More than 2 sensors were used for the variable {varnm}. '
                  '\nMetadata attributes will have to be adjusted manually!')


    ### FINAL TOUCHES AND EXPORTS

    # Transpose so that variables are structured like (TIME, PRES) rather than
    # (PRES, TIME). Convenient when plotting etc.
    # **Note**: Dropping doing this because CF conventions seem to require T first
    # N = N.transpose()

    # Sort chronologically
    N = N.sortby('TIME')

    # Add some standard metadata to the measurement variables
    N = _add_standard_variable_attributes(N)

    # Set LATITUTDE, LONGITUDE and STATION as auxiliary coordinates (if available)
    N = N.set_coords([v for v in ['LATITUDE', 'LONGITUDE', 'STATION'] if v in N])

    # Add some metadata to the STATION variable
    if 'STATION' in N.data_vars:
        N['STATION'].attrs = {'long_name' : 'CTD station ID',
                              'cf_role':'profile_id'}

    # Warn the user if we used different sources for profile start time
    if different_start_time_sources:
        print('\nNOTE: The start_time variable was read from different '
            'header lines for different profiles. \nThis likely means that '
            'your .cnv files were formatted differently through the cruise. '
            '\nThis is probably fine, but it is a good idea to sanity check '
            'the TIME field.')


    # Generalize (insert "e.g.") in attributes with specific file names
    N.attrs['source_file'] = f"E.g. {N.attrs['source_file']}"
    SBEproc_file_ind = N.SBE_processing.rfind('Raw data read from ')+19
    N.attrs['SBE_processing'] = (N.SBE_processing[:SBEproc_file_ind]
                + 'e.g. ' + N.SBE_processing[SBEproc_file_ind:])

    # Add date *ranges* to the history entries if we have different dates
    N = _replace_history_dates_with_ranges(N, post_proc_times, SBE_proc_times)

    # Delete some non-useful attributes (that are only relevant to individual
    # profiles)
    del_attrs = ['SBE_processing_date', 'start_time_source', 'station',
                 'start_time', 'SBE_flags_applied', 'latitude', 'longitude']
    for attr in del_attrs:
        del N.attrs[attr]

    # Set the featureType attribute
    N.attrs['featureType'] = 'profile'
    N.PRES.attrs['coverage_content_type'] = 'coordinate'

    return N




def join_cruise_btl(datasets, verbose = True,
                epoch = '1970-01-01'):
    '''
    Takes a number of btl profiles and joins the profiles into one single file.

    Inputs:

    datasets: string, list
        A dictionary containing either:
        1. A path to the location of individual profiles,
           e. g. 'path/to/files/'.
        2. A list containing xr.Datasets with the individual profiles.

    epoch: Only necessary to specify here if there is no TIME_SAMPLE
           field in the data files. (In that case, we assign time based on
           the profile start time, and need to know the epoch)

    NOTE: This is basically a slightly modified version of join_cruise()
          which does the same for .cnvs. Should optimize and/or see if we
          can refactor so we don't have to repeat the same code.


    '''
    ### PREPARE INPUTS
    # Grab the input data on the form of a list of xr.Datasets
    # (1) Load all from path if we are given a path.
    if isinstance(datasets, str):
        for nn in np.arange(len(datasets)):
            if datasets[nn].endswith('/'):
                datasets[nn] = datasets[nn][:-1]
        file_list = glob2.glob(f'{datasets}/*.nc')

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
    elif isinstance(datasets, list):
        if all(isinstance(nc_file, xr.Dataset) for nc_file in datasets):
            ns_input = datasets
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
            Input *datasets* invalid. Must be either:
            1. A path to the location of individual profiles,
               e.g. "path/to/files/" --or--
            2. A list containing xr.Datasets with the individual profiles,
               e.g. [d1, d2] with d1, d2 being xr.Datasets.
                        ''')


    ### JOINING PROFILES TO ONE DATASET
    #  Concatenation loop.
    # - Add a CRUISE variable

    first = True

    # Keeping track of these as we want to replace them with a date *range*
    SBE_proc_times = []

    prog_bar_msg_3 = f'Joining profiles together'

    for n in tqdm(datasets, prog_bar_msg_3):

        sbe_timestamp, proc_timestamp = _dates_from_history(n)
        SBE_proc_times += [sbe_timestamp]

        if first:
            N = n
            # Want to keep track of whether we use different
            # start_time sources
            start_time_source = n.start_time_source
            different_start_time_sources = False
            first = False
        else:
            if n.start_time_source != start_time_source:
                different_start_time_sources = True
            N = xr.concat([N, n], dim = 'TIME')

    ### CHECK IF ANY SENSORS CHANGED
    # Modify metadata if they did
    for varnm in list(N.data_vars) + ['NISKIN_NUMBER']:
        caldates, sns, units, stations = [], [], [], []

        for n in datasets:
            stations += [n.STATION.values[0]]
            for _list, _attr  in zip([caldates, sns, ],
                ['sensor_calibration_date', 'sensor_serial_number',]):
                if varnm in n.keys():
                    if _attr in n[varnm].attrs:
                        _list += [n[varnm].attrs[_attr]]
                    else:
                        _list += [None]
                else:
                    _list += [None]

        sns_filtered = np.array([value for value in sns if value is not None])
        sns_unique = unique_entries = list(set(sns_filtered))
        caldates_filtered = np.array([value for value in caldates if value is not None])
        caldates_unique = unique_entries = list(set(caldates_filtered))

        if len(sns_unique)==2:
            if 'comment' in N[varnm].attrs:
                comment_0 = N[varnm].comment
            else:
                comment_0 = ''
            N[varnm].attrs['comment'] = comment_0 + ' Sensors changed underway.'
            where_sensor_A = np.array(stations)[np.array(sns)==sns_unique[0]]
            where_sensor_B = np.array(stations)[np.array(sns)==sns_unique[1]]
            N[varnm].attrs['stations_A'] = ', '.join(where_sensor_A)
            N[varnm].attrs['stations_B'] = ', '.join(where_sensor_B)
            N[varnm].attrs['sensor_calibration_date'] = (
                f'A: {caldates_unique[0]}, B: {caldates_unique[1]}')
            N[varnm].attrs['sensor_serial_number'] = (
                f'A: {sns_unique[0]}, B: {sns_unique[1]}')

    ### FINAL TOUCHES AND EXPORTS

    # Transpose so that variables are structured like (TIME, PRES) rather than
    # (PRES, TIME). Convenient when plotting etc.
    N = N.transpose()

    # Sort chronologically
    N = N.sortby('TIME')

    # Add some standard metadata to the measurement variables
    N = _add_standard_variable_attributes(N)

    # Add some metadata to the STATION variable
    if 'STATION' in N.data_vars:
        N['STATION'].attrs = {'long_name' : 'CTD station ID',
                              'cf_role':'profile_id'}

    # Warn the user if we used different sources for profile start time
    if different_start_time_sources:
        print('\nNOTE: The start_time variable was read from different '
            'header lines for different profiles. \nThis likely means that '
            'your .cnv files were formatted differently through the cruise. '
            '\nThis is probably fine, but it is a good idea to sanity check '
            'the TIME field.')

    # Add a cruise variable
    if 'cruise' in N.attrs:
        cruise = N.cruise_name
    else:
        cruise = '!! CRUISE !!'

    N['CRUISE'] =  xr.DataArray(cruise, dims=())
    N['CRUISE'].attrs = {'long_name':'Cruise ID',}

    # Generalize (insert "e.g.") in attributes with specific file names
    N.attrs['source_file'] = f"E.g. {N.attrs['source_file']}"
    SBEproc_file_ind = N.SBE_processing.rfind('Raw data read from ')+19
    N.attrs['SBE_processing'] = (N.SBE_processing[:SBEproc_file_ind]
                + 'e.g. ' + N.SBE_processing[SBEproc_file_ind:])

    # Add date *ranges* to the history entries if we have different dates
    # post_proc_fimes = None for btl files..
    N = _replace_history_dates_with_ranges(N, None, SBE_proc_times)

    # Delete some non-useful attributes (that are only relevant to individual
    # profiles)
    del_attrs = ['SBE_processing_date', 'start_time_source', 'station',
                 'start_time', 'SBE_flags_applied', 'latitude', 'longitude']

    for attr in del_attrs:
        if attr in N.attrs:
            del N.attrs[attr]

    # Set the featureType attribute
    N.attrs['featureType'] = 'profile'

    return N



def bin_to_pressure(ds, dp = 1):
    '''
    Apply pressure binning into bins of *dp* dbar.
    Reproducing the SBE algorithm as documented in:
    https://www.seabird.com/cms-portals/seabird_com/
    cms/documents/training/Module13_AdvancedDataProcessing.pdf

    # Provides not a bin *average* but a *linear estimate of variable at bin
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

    # We have to reassign the values of the variables which are not on the
    # pressure dimension.

    # Save a copy of the pre-binned data
    ds0 = ds.copy()
    # Make a list of variables with only a TIME dimension
    time_vars = [var_name for var_name, var in ds.variables.items()
                 if 'TIME' in var.dims and len(var.dims) == 1]
    # Make a copy with only the pressure-dimensional variables
    ds_pres = ds.drop_vars(time_vars)

    # Define the bins over which to average
    pmax = float(ds_pres.PRES.max())
    pmax_bound = np.floor(pmax-dp/2)+dp/2

    pmin = float(ds_pres.PRES.min())
    pmin_bound = np.floor(pmin+dp/2)-dp/2

    p_bounds = np.arange(pmin_bound, pmax_bound+1e-9, dp)
    p_centre = np.arange(pmin_bound, pmax_bound, dp)+dp/2

    # Pressure averaged
    ds_pavg = ds_pres.groupby_bins('PRES', bins = p_bounds).mean(dim = 'scan_count')

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

    # Remove any bins with no data (PRES will be NaN)
    ds_binned = ds_binned.dropna(dim='PRES', subset=['PRES'])

    # Set xarray option "keep_attrs" back to whatever it was
    xr.set_options(keep_attrs=_keep_attr_value)

    # Add back the non-pressure dimensional variables
    for time_var in time_vars:
        ds_binned[time_var] = ds0[time_var]

    return ds_binned

### UTILITY FUNCTIONS

def _add_standard_variable_attributes(ds):
    '''
    Add standard attributes to measurement variables.
    '''
    var_attrs = {
        'coverage_content_type':'physicalMeasurement',
        'sensor_mount':'mounted_on_shipborne_profiler'}

    for varnm in ds.keys():
        if ('PRES' in ds[varnm].dims
            and varnm not in ['SBE_FLAG', 'TIME_SAMPLE', 'TIME']):
            for key, item in var_attrs.items():
                ds[varnm].attrs[key] = item
    return ds

def _cnv_files_from_path(path):
    '''
    Get a list of .cnv files from a path.
    '''
    if not path.endswith('/'):
        path += '/'
    cnv_list = glob2.glob(f'{path}*.cnv')
    return cnv_list


def _btl_files_from_path(path):
    """
    Get a list of .btl files from a directory path.

    Args:
        path (str or Path): Directory containing .btl files.

    Returns:
        list[str]: Sorted list of full paths to .btl files.
    """
    path = Path(path)  # ensure Path object
    btl_list = sorted(glob2.glob(str(path / '*.btl')))
    return btl_list


def _datasets_from_cnvlist(cnv_list,
                           station_from_filename = False,
                           verbose = True, start_time_NMEA = False,
                           profile = "downcast",
                           remove_duplicates=True):
    '''
    Get a list of profile xr.Datasets from a list of .cnv files.
    '''
    dataset_list = []
    for fn in cnv_list:
        try:
            dataset_list += [sbe.read_cnv(fn, time_dim=True,
                            station_from_filename = station_from_filename,
                            profile=profile,
                            suppress_time_warning=not verbose,
                            suppress_latlon_warning=not verbose,
                            start_time_NMEA = start_time_NMEA,
                            remove_duplicates=remove_duplicates)]
        except:
            print(f'\n*NOTE*: Could not read file {fn}.')
            print('(This usually indicates some sort of problem with the file.'
                  ' For example, sensor setup may not match variables.) '
                  '\n-> LOOK AT THE .CNV FILE! (skipping this file for now)\n')
    return dataset_list


def _datasets_from_btllist(btl_list,
                           station_from_filename = False,
                           verbose = True, start_time_NMEA = False,
                           time_adjust_NMEA = False):
    '''
    Get a list of profile xr.Datasets from a list of .btl files.
    '''
    dataset_list = [sbe.read_btl(fn, time_dim=True,
                        station_from_filename = station_from_filename,
                        start_time_NMEA = start_time_NMEA,
                        time_adjust_NMEA = time_adjust_NMEA)
                       for fn in btl_list]


    return dataset_list



def _replace_history_dates_with_ranges(D, post_proc_times, SBE_proc_times):
    '''
    When joining files: Change the history string to show time *ranges*,
    e.g.
       2017-09-24: Data collection
    -> 2017-09-24 to 2017-09-24: Data collection

    (Note: post_proc_times can be None)
    '''

    if post_proc_times:
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
        rind = D.history.find(': Processed to .cnv using SBE')
        D.attrs['history'] = D.history[:rind-10] + sbe_range + D.attrs['history'][rind:]

    if post_proc_times:

        if ppr_max>ppr_min and post_proc_times:
            ppr_range = (f'{ppr_min.strftime(date_fmt)} to'
                f'{ppr_max.strftime(date_fmt)}')
            rind = D.history.find(': Post-processing.')
            D.attrs['history'] = (D.history[:rind-10] + ppr_range +
                                  D.attrs['history'][rind:])

    return D


def _dates_from_history(ds):
    '''
    Grab the dates of 1) SBE processing and 2) Post-processing from the
    "history" attribute.
    '''
    try:
        sbe_pattern = r"(\d{4}-\d{2}-\d{2}): Processed to .cnv using SBE software"
        sbe_time_match_str = re.search(sbe_pattern, ds.history).group(1)
        sbe_timestamp = pd.Timestamp(sbe_time_match_str)
    except:
        try:
            sbe_pattern = r"(\d{4}-\d{2}-\d{2}): Processed to .btl using SBE software"
            sbe_time_match_str = re.search(sbe_pattern, ds.history).group(1)
            sbe_timestamp = pd.Timestamp(sbe_time_match_str)
        except:
            sbe_timestamp = None


    try:
        proc_pattern = r"(\d{4}-\d{2}-\d{2}): Post-processing"
        proc_time_match_str = re.search(proc_pattern, ds.history).group(1)
        proc_timestamp = pd.Timestamp(proc_time_match_str)
    except:
        proc_timestamp = None
    return sbe_timestamp, proc_timestamp


# Get the profile variables
def _get_profile_variables(ds, profile_var = 'PRES', require_TIME = True):
    '''
    Return a list of profile variables (i.e. variables with TIME, PRES
    dimensions).

    Alternatively, specify another profile dimension instead of PRES, e.g.
    NISKIN_NUMBER.
    '''

    profile_variables = []
    for varnm in ds.data_vars:
        if require_TIME:
            crit = profile_var in ds[varnm].dims and 'TIME' in ds[varnm].dims
        else:
            crit = profile_var in ds[varnm].dims
        if crit:
            profile_variables += [varnm]

    return profile_variables

