'''
### OCEANOGRAPY.IO.SBE.py ###

Parsing data from seabird format (.cnv, .hdr, .btl) to xarray Datasets.



Some of these functions are rather long and clunky. This is mainly because the
input files are clunky and the format changes quite a lot. 

Key functions
-------------

read_cnv:
  Loading single profiles. Reads CTD data and metadata from a .cnv file into a
  Dataset (-> netCDF file) with any potentially useful metadata we can extract
  from the .cnv header.

read_btl:
  Similar: Load single .btl files.

join_cruise:
  Joing several profiles (e.g. from an entire cruise) into a single Dataset (->
  netCDF file).

read_header:
  Parse useful information from a .cnv header into a dictionary. 
  Mainly used within read_cnv, but may have its additional uses.  

May eventually want to look into the organization: -> Do all these things belong
here or elsewhere? -> This is a massive script, should it be split up?

TBD:
- Go through documentation and PEP8 formatting
    - Pretty decent already
- Consider usign the *logging module*


'''

## IMPORTS

import pandas as pd
import xarray as xr
import numpy as np
from oceanograpy.io import _variable_defs as vardef
from oceanograpy.util import time
import matplotlib.pyplot as plt
import re
from typing import Optional
from tqdm.notebook import tqdm 
from typing import Optional
from itertools import zip_longest
from matplotlib.dates import date2num

## KEY FUNCTIONS

def read_cnv(
    source_file: str,
    apply_flags: Optional[bool] = True,
    profile: Optional[str] = 'downcast',
    time_dim: Optional[bool] = False,
    inspect_plot: Optional[bool] = False,
    start_scan: Optional[int] = None,
    end_scan: Optional[int] = None,
    suppress_time_warning: Optional[bool] = False,
    suppress_latlon_warning: Optional[bool] = False,
    start_time_NMEA: Optional[bool] = False,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    station: Optional[str] = None,
    station_from_filename: Optional[bool] = False,
) -> xr.Dataset:
    """
    Reads CTD data and metadata from a .cnv file into a more handy format.
    (I.e. an xarray Dataframe with any potentially useful metadata we can 
    extract from the .cnv header.)
    
    -> This does not mean that the profile is CF/ACDD compliant (we will 
       need more metadata not found in the file for that). This should be
       a very good start, however.

    -> Profiles are assigned the coordinates ['scan_count'], or, if 
       time_dim is set to True, ['scan_count', 'TIME'], where TIME
       is one-dimensional.
       
    -> Profiles created using this function can be joined together using
       the join_cruise() function. For this, we need to set time_dim=True
       when using read_cnv().

    Parameters
    ----------
    source_file : str
        Path to a .cnv file containing the data 

    apply_flags : bool, optional 
        If True, flags (from the SBE *flag* column) are applied as NaNs across
        all variables (recommended). Default is True.
     
    profile : str, optional
        Specify the profile type. Options are ['upcast', 'downcast', 'none'].

    time_dim : bool, optional
        Choose whether to include a 0-D TIME coordinate. Useful if combining
        several profiles. Default is False.

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
    
    suppress_time_warning : bool, optional
        Don't show a warning if there are no timeJ or timeS fields. Default is False.

    suppress_latlon_warning : bool, optional
        Don't show a warning if there is no lat/lon information. Default is False.

    start_time_NMEA : bool, optional
        Choose whether to get start_time attribute from the "NMEA UTC (Time)" 
        header line. Default is to grab it from the "start_time" line - this
        is technically correct but typically identical results, and the 
        "start_time" line can occasionally look funny. If unsure, check your 
        header! Default is False (= read from "start_time" header line).        

    lat, lon : [float, bool], optional
        Option to specify latitude/longitude manually. (E.g. if there is no lat/lon 
        info in the header or variables).

    station : [str, bool], optional
        Option to specify station manually. Otherwise: extracting from 
        'station line' if available and from the cnv file name if not.
    
    station_from_filename : bool, optional
        Option to read the station name from the file name, e.g. "STA001"
        from a file "STA001.cnv". Otherwise, we try to grab it from the header.
        Default is False.

    Returns
    -------
    xr.Dataset
        A dataset containing CTD data and associated attributes.

    TO DO
    ----- 
    - Checking and testing
    - Apply to some other datasets for testing
        - Maybe also moored sensors/TSG? (or should those be separate?)
    - Tests
        - Make a test_ctd_data.cnv file with mock values and use pytest
    """

    header_info = read_header(source_file)
    ds = _read_column_data_xr(source_file, header_info)
    ds = _update_variables(ds, source_file)
    ds = _assign_specified_lat_lon_station(ds, lat, lon, station)
    ds = _convert_time(ds, header_info, 
                       suppress_time_warning = suppress_time_warning,
                       start_time_NMEA = start_time_NMEA)
    ds.attrs['history'] = header_info['start_history']
    ds = _add_header_attrs(ds, header_info, station_from_filename)
    ds = _add_start_time(ds, header_info, 
                             start_time_NMEA = start_time_NMEA)
    ds = _read_SBE_proc_steps(ds, header_info)

    ds0 = ds.copy()
    
    if apply_flags:
        ds = _apply_flag(ds)
        ds.attrs['SBE_flags_applied'] = 'yes'
    else:
        ds.attrs['SBE_flags_applied'] = 'no'

    if time_dim:
        ds = _add_time_dim_profile(ds, 
            suppress_latlon_warning= suppress_latlon_warning)

    if profile in ['upcast', 'downcast', 'dncast']:
        if ds.binned == 'no':
            ds = _remove_up_dncast(ds, keep = profile)
    else:
        ds.attrs['profile_direction'] = 'All available good data'

    if start_scan:
        ds = _remove_start_scan(ds, start_scan)
    if end_scan:
        ds = _remove_end_scan(ds, end_scan)

    if inspect_plot:
        _inspect_extracted(ds, ds0, start_scan, end_scan)

    now_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    ds.attrs['history'] += f'\n{now_str}: Post-processing.'


    return ds


def read_btl(source_file, verbose = False, 
             time_dim = True,
             station_from_filename= False,
             start_time_NMEA = True, 
             time_adjust_NMEA = False):
    '''
    time_adjust_NMEA: Use this if the "start time" in the file
    header is incorrect (occasionally seems to be reset to 2000-1-1)  
    '''

    # Parse the header
    header_info = read_header(source_file)

    # Parse the columnar data
    ds = _read_btl_column_data_xr(source_file, 
                                  header_info, verbose = verbose)
    # Add nice variable names and attributes 
    ds = _update_variables(ds, source_file)
    # Add a history string
    ds.attrs['history'] = header_info['start_history']
    # Add ship, cruise, station, latitude, longitude attributes from the header
    ds = _add_header_attrs(ds, header_info, 
                           station_from_filename = station_from_filename)
    # Add a start time attribute from the header 
    ds = _add_start_time(ds, header_info, 
                                start_time_NMEA = start_time_NMEA)
    # Add the SBE processing string from the header
    ds = _read_SBE_proc_steps(ds, header_info)
    # Add a (0-D) TIME dimension
    time_dim = True
    if time_dim:

        ds = _add_time_dim_profile(ds)
    

        if time_adjust_NMEA:


            # Calculate the difference bewteen nmae and start time in days
            diff_days = ((header_info['NMEA_time'] - header_info['start_time'])
                             .total_seconds() / (24 * 3600))
            
            # Hold on to variable attributes
            time_attrs = ds.TIME.attrs

            # Apply the offset to TIME
            ds = ds.assign_coords(TIME=ds.TIME.values + diff_days)

            # Reapply variable attributes
            ds.TIME.attrs = time_attrs
    return ds 


def read_header(filename: str) -> dict:
    """
    Reads a SBE .cnv (or .hdr, .btl) file and returns a dictionary with various
    metadata parameters extracted from the header.

    NOTE: Only tested for .cnv and .btl.

    Parameters:
    ----------
    source_file : str
        The path to the SBE .cnv, .hdr, or .btl file.

    Returns:
    -------
    dict
        A dictionary containing various metadata parameters extracted from the header.
    """

    with open(filename, 'r', encoding = 'latin-1') as f:

        # Empty dictionary: Will fill these parameters up as we go
        hkeys = ['col_nums', 'col_names', 'col_longnames', 'SN_info', 
                 'moon_pool', 'SBEproc_hist', 'hdr_end_line', 
                 'latitude', 'longitude', 'NMEA_time', 'start_time', ]
        hdict = {hkey:[] for hkey in hkeys}

        # Flag that will be turned on when we read the SBE history section 
        start_read_history = False 


        # Go through the header line by line and extract specific information 
        # when we encounter specific terms dictated by the format  
        
        lines = f.readlines()
        #return lines
        for n_line, line in enumerate(lines):

            # Read the column header info (which variable is in which data column)
            if '# name' in line and not filename.endswith('.btl'):
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
            
            # If no NMEA lat/lon: Look for "** Latitude/Longitude"
            # (for some poorly structured .cnv files)
            if '** LATITUDE' in line.upper() and isinstance(
                hdict['latitude'], list):
                if len(hdict['latitude']) == 0:
                    lat_value = _decdeg_from_line(line)
                    if lat_value:
                        hdict['latitude'] = lat_value

            if '** LONGITUDE' in line.upper()and isinstance(
                hdict['longitude'], list):
                if len(hdict['longitude']) == 0:
                    lon_value= _decdeg_from_line(line)
                    if lon_value:
                        hdict['longitude'] = lon_value
                    
            # Read start time
            if 'start_time' in line:
                hdict['start_time'] = (
                    _nmea_time_to_datetime(*line.split()[3:7]))

                hdict['start_history'] = (
                    hdict['start_time'].strftime('%Y-%m-%d')
                    + ': Data collection.')

            # Read cruise/ship/station/bottom depth/operator if available
            if '** CRUISE' in line.upper():
                hdict['cruise_name'] = line[(line.rfind(': ')+2):].replace('\n','').strip()
            if '** STATION' in line.upper():
                hdict['station'] = line[(line.rfind(': ')+2):].replace('\n','').strip()
            if '** SHIP' in line.upper():
                hdict['ship'] = line[(line.rfind(': ')+2):].replace('\n','').strip()
            if '** BOTTOM DEPTH' in line.upper():
                hdict['bdep'] = line[(line.rfind(': ')+2):].replace('\n','').strip()

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

            # For .cnvs:
            # Read the line containing the END string
            # (and stop reading the file after that)
            if '*END*' in line and filename.endswith('.cnv'):
                hdict['hdr_end_line'] = n_line

                # Break the loop through all lines
                break

            # For .btls:
            # Read the line containing the header information string
            # (and stop reading the file after that)
 
            if line.split()[0] =='Bottle' and filename.endswith('.btl'):
                # Grab the header names
                hdict['col_names'] = line.split()

                # The header is typically wrapped across two lines. 
                # -> adding info from the next line
                # NOTE: Assuming that 2-row column names cannot follow 1-row ones.
                # There *could* be cases where this is otherwise, but I haven't
                # encountered this.
                  
                # Loop through following lines (usually just one)
                nline_next = n_line + 1
                while True: # Arbitrary clause, we will break this loop when getting to the data

                    # Look at the following line
                    line_next = lines[nline_next] 

                    # If line_next starts with "1", this is the beginning of the data and 
                    # we stop looking for additional column name information, but store
                    # the start line.
                    if line_next.split()[0]=='1': 
                        hdict["start_line_btl_data"] = nline_next
                        break

                    # If not, this is part of the header, and we add info 
                    # onto the hdict["col_names"] field.
                    else:
                        hdict["col_names"] = [
                            f'{col_name} {next_part}' for col_name, next_part in 
                            zip_longest(hdict["col_names"], line_next.split(), fillvalue='')
                                ]
                    nline_next += 1
                
                # Remove trailing whitespaces from col_names
                col_names_stripped = [col_name.rstrip() for col_name in hdict["col_names"]]
                hdict["col_names"] = col_names_stripped

                # Store the end line
                hdict['hdr_end_line'] = n_line-1
                
                # Break the loop through all lines
                break

        # Remove the first ('</Sensors>') and last ('*END*') lines from the SBE history string.
        hdict['SBEproc_hist'] = hdict['SBEproc_hist'] [1:-1]

        # Set empty fields to None
        for hkey in hkeys:
            if isinstance(hdict[hkey], list):
                if len(hdict[hkey])==0:
                    hdict[hkey] = None

        # Assign the file name without the directory path
        for suffix in ['cnv', 'hdr', 'btl']:
            if filename.endswith(suffix):
                hdict['source_file'] = filename[filename.rfind('/')+1:]
                hdict['source_file_type'] = suffix

        return hdict


def to_nc(
    ds: xr.Dataset,
    where: str = '.',
    filename: bool = False,
    suffix: str = None
) -> None:
    """

    TBW!!

    Export a dataset to a NetCDF file.

    Parameters:
    ----------
    ds : xr.Dataset
        The dataset produced by read_cnv().
    where : str, optional
        Path in which the NetCDF file will be located.
    filename : bool or str, optional
        The name of the NetCDF file. If False (default),
        the name of the cnv file will be used (no ".nc" suffix necessary).
    suffix : str, optional
        Option to add a suffix (e.g., "_prelim") to the file name.
    """


## INTERNAL FUNCTIONS: PARSING

def _read_column_data_xr(source_file, header_info):
    '''
    Reads columnar data from a single .cnv to an xarray Dataset.

    (By way of a pandas DataFrame)

    '''
    df = pd.read_csv(source_file, header = header_info['hdr_end_line']+1,
                 delim_whitespace=True, encoding = 'latin-1',
                 names = header_info['col_names'])
    
    # Convert to xarray DataFrame
    ds = xr.Dataset(df).rename({'dim_0':'scan_count'})
    # Will update "binning" later if we discover that it is binned 
    ds.attrs['binned'] = 'no' 

    return ds




def _read_btl_column_data_xr(source_file, header_info, verbose = False):
    '''
    Read the columnar data from a .btl file during information (previously)
    parsed from the header.

    A little clunky as the data are distributed over two rows (subrows), a la:

    Bottle        Date    Sal00     T090C     C0S/m  
    Position      Time                                                                                                   
      1       Jan 04 2021    34.6715   2.963803  1.5859    (avg)
               15:27:59                0.0001    0.000015  (sdev)
      2       Jan 04 2021    34.6733   2.530134  1.5663    (avg)
               15:31:33                0.0002    0.000012  (sdev)

    '''

    # Read column data to dataframe
    df_all = pd.read_fwf(source_file, skiprows = header_info['start_line_btl_data'], 
        delim_whitespace = True, names = header_info['col_names'] + ['avg_std'], 
        encoding = 'latin-1')
    
    # Read the primary (avg) and second (std rows)
    # + Reindexing and dropping the "index" and "avg_std" columns

    # Parse the first subrow
    df_first_subrow = (df_all.iloc[0::2].reset_index(drop = False)
                       .drop(['index', 'avg_std'], axis = 1))

    # Parse the second subrow
    df_second_subrow = (df_all.iloc[1::2].reset_index()
                        .drop(['index', 'avg_std'], axis = 1))

    # Build a datafra with all the information combined
    df_combined = pd.DataFrame()

    ## Loop through all columns except date and bottle number:
    # - Add the variables from the first subrow if they are not empty
    # - Replace column names the nice names (if available)
    # - Add variables from the second subrow with _std as suffix 
    # - Also adding units, sensors as variable attributes (.attr)

    for sbe_name_ in df_all.keys():
        # Skip "bottle number" and "time" rows, will deal with those separately
        if 'Bottle' in sbe_name_ or 'Date' in sbe_name_:
            pass
        else:
            # Read first subrow
            try:
                df_combined[sbe_name_.replace('/', '_')] = (
                    df_first_subrow[sbe_name_].astype(float))
            except:
                if verbose:
                    print(f'Could not read {sbe_name_} as float - > dropping')

            # Read second subrow
            try:
                df_combined[f'{sbe_name_.replace("/", "_")}_std'] = (
                    df_second_subrow[sbe_name_].astype(float))
            except:
                if verbose:
                    print(f'Could not read {sbe_name_}_std as float - > dropping')

    ## Add bottle number (assuming it is the first column)
    bottle_num_name = df_first_subrow.keys()[0] # Name of the bottle column
    df_combined['NISKIN_NUMBER'] = df_first_subrow[bottle_num_name].astype(float)


    # Parse TIME_SAMPLE from first + second subrows (assuming it is the second column)
    # Using the nice pandas function to_datetime (for parsing)
    # Then converting to datenum (days since 1970-01-01)
    # NOTE: Should put this general functionality in util.time!)

    time_name = df_first_subrow.keys()[1] # Name of the time column

    TIME_SAMPLE = []
    epoch = '1970-01-01'

    for datestr, timestr in zip(df_first_subrow[time_name], df_second_subrow[time_name]):
        time_string = ', '.join([datestr, timestr])
        
        # Parse the time string to pandas Timestamp
        time_pd = pd.to_datetime(time_string)

        # Convert to days since epoch
        time_DSE = ((time_pd - pd.Timestamp(epoch)) / pd.to_timedelta(1, unit='D'))
        
        TIME_SAMPLE += [time_DSE]

    df_combined['TIME_SAMPLE'] = TIME_SAMPLE
    df_combined['TIME_SAMPLE'].attrs = {'units':f'Days since {epoch}',                        
            'long_name': 'Time stamp of bottle closing',
            'coverage_content_type':'coordinate',
            'SBE_source_variable' : time_name,}


    # Convert DataFrame to xarray Dataset
    ds = xr.Dataset()

    variable_list = list(df_combined.keys())
    variable_list.remove('NISKIN_NUMBER')

    for varnm in variable_list:
        ds[varnm] = xr.DataArray(df_combined[varnm], dims=['NISKIN_NUMBER'], 
                                coords={'NISKIN_NUMBER': df_combined.NISKIN_NUMBER})

        # Preserve metadata attributes for the 'Value' variable
        ds[varnm].attrs = df_combined[varnm].attrs

    ds['NISKIN_NUMBER'].attrs = {'long_name':'Niskin bottle number',
        'comment': 'Designated number for each physical Niskin bottle '
            'on the CTD rosette (typically e.g. 1-24, 1-11).'
            ' Bottles may be closed at different depths at different stations. '}

    return ds



def _read_SBE_proc_steps(ds, header_info):
    '''
    Parse the information about SBE processing steps from the cnv header into 
    a more easily readable format and storing the information as the global
    variable *SBE_processing*.

    This is a long and clunky function. This is mainly because the input
    format is clunky. 

    Also:
    - Adds a *SBE_processing_date* global variable (is this useful?)
    - Adds a *source_files* variable with the names of the .hex, .xmlcon,
      and .cnv files
    - Appends SBE processing history line to the *history* attribute.
    '''
    SBElines = header_info['SBEproc_hist']
    ct = 1 # Step counter, SBE steps
    dmy_fmt = '%Y-%m-%d'

    sbe_proc_str = ['SBE SOFTWARE PROCESSING STEPS (extracted'
                f' from .{header_info["source_file_type"]} file header):',]

    for line in SBElines:
        # Get processing date
        if 'datcnv_date' in line:
            proc_date_str = re.search(r'(\w{3} \d{2} \d{4} \d{2}:\d{2}:\d{2})',
                                line).group(1)
            proc_date = pd.to_datetime(proc_date_str) 
            proc_date_ISO8601 = time.datetime_to_ISO8601(proc_date) 
            proc_date_dmy = proc_date.strftime(dmy_fmt)
            history_str = (f'{proc_date_dmy}: Processed to '
                           f'.{header_info["source_file_type"]} using'
                            ' SBE software (details in "SBE_processing").')

        # Get input file names (without paths)
        if 'datcnv_in' in line:
            hex_fn = re.search(r'\\([^\\]+\.HEX)', line.upper()).group(1)
            try:
                xmlcon_fn = re.search(r'\\([^\\]+\.XMLCON)', line.upper()).group(1)
            except:
                xmlcon_fn = re.search(r'\\([^\\]+\.CON)', line.upper()).group(1)

            
            src_files_raw = f'{hex_fn}, {xmlcon_fn}'
            sbe_proc_str += [f'{ct}. Raw data read from {hex_fn}, {xmlcon_fn}.']
            ct += 1

        # SBE processing details
        # Get skipover scans
        if 'datcnv_skipover' in line:
            skipover_scans = int(re.search(r'= (\d+)', line).group(1))
            if skipover_scans != 0:
                sbe_proc_str += [f'{ct}. Skipped over {skipover_scans} initial scans.']
                ct += 1

        # Get ox hysteresis correction 
        if 'datcnv_ox_hysteresis_correction' in line:
            ox_hyst_yn = re.search(r'= (\w+)', line).group(1)
            if ox_hyst_yn == 'yes':
                sbe_proc_str += [f'{ct}. Oxygen hysteresis correction applied.']
                ct += 1

        # Get ox tau correction 
        if 'datcnv_ox_tau_correction' in line:
            ox_hyst_yn = re.search(r'= (\w+)', line).group(1)
            if ox_hyst_yn == 'yes':
                sbe_proc_str += [f'{ct}. Oxygen tau correction applied.']
                ct += 1

        # Get low pass filter details
        if 'filter_low_pass_tc_A' in line:
            lp_A = float(re.search(r' = (\d+\.\d+)', line).group(1))
        if 'filter_low_pass_tc_B' in line:
            lp_B = float(re.search(r' = (\d+\.\d+)', line).group(1))
        if 'filter_low_pass_A_vars' in line:
            try:
                lp_vars_A = re.search(r' = (.+)$', line).group(1).split()
                sbe_proc_str += [f'{ct}. Low-pass filter with time constant {lp_A}'
                    + f' seconds applied to: {" ".join(lp_vars_A)}.']
                ct += 1
            except: 
                print('FYI: Looks like filter A was not applied to any variables.')
        if 'filter_low_pass_B_vars' in line:
            try:
                lp_vars_B = re.search(r' = (.+)$', line).group(1).split()
                sbe_proc_str += [f'{ct}. Low-pass filter with time constant {lp_B}'
                    + f' seconds applied to: {" ".join(lp_vars_B)}.']
                ct += 1
            except:
                print('FYI: Looks like filter A was not applied to any variables.')

        # Get cell thermal mass correction details 
        if 'celltm_alpha' in line:
            celltm_alpha = re.search(r'= (.+)$', line).group(1)
        if 'celltm_tau' in line:
            celltm_tau= re.search(r'= (.+)$', line).group(1)
        if 'celltm_temp_sensor_use_for_cond' in line:
            celltm_sensors = re.search(r'= (.+)$', line).group(1)
            sbe_proc_str += [f'{ct}. Cell thermal mass correction applied to conductivity' 
                 f' from sensors: [{celltm_sensors}]. ',
                 f'   > Parameters ALPHA: [{celltm_alpha}], TAU: [{celltm_tau}].']
            ct += 1

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
            sbe_proc_str += [f'{ct}. Loop editing applied.',
                 (f'   > Parameters: Minimum velocity (ms-1): {loop_minvel}, '
                  f'Soak depth range (m): {loop_ss_mindep} to {loop_ss_maxdep}, '
                  + f'\n   > {loop_excl_str}. '
                  + f'Deck pressure offset: {loop_ss_deckpress_str}.')]
            ct += 1

        # Get wild edit details
        if 'wildedit_date' in line:
            sbe_proc_str += [f'{ct}. Wild editing applied.']
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
            ct += 1

        if 'Derive_in' in line or 'derive_in' in line:
            sbe_proc_str += [
                f'{ct}. Derived EOS-8 salinity and other variables.']
            ct += 1

        # Get window filter details
        if 'wfilter_excl_bad_scans' in line:
            wf_bad_scans = re.search(r'= (.+)', line).group(1)
            if wf_bad_scans == 'yes':
                wf_bad_scans = 'Bad scans excluded'
            else:
                wf_bad_scans = 'Bad scans not excluded'
        if 'wfilter_action' in line:

            wf_variable = re.search(r'(.+) =', line).group(0).split()[-2]
            wf_filter_type = (re.search(r'= (.+)', line).group(0).split()[1]
                              .replace(',', ''))
            wf_filter_param = re.search(r'(\d+)', line).group(0)
            sbe_proc_str += [f'{ct}. Window filter ({wf_filter_type}, '
                f'{wf_filter_param}) applied to {wf_variable} ({wf_bad_scans}).']
            ct += 1

        # Get align CTD details
        if 'alignctd_adv' in line:
            # Find all matches in the string
            matches = re.findall(r'(\w+)\s+([0-9.]+)', line)
            # Rerutn a list of tuples with (variable, advance time in seconds)
            align_tuples = [(key, float(value)) for key, value in matches]
            sbe_proc_str += [f'{ct}. Misalignment correction applied.']
            sbe_proc_str += [f'   > Parameters [variable (advance time, sec)]:']
            align_str = []
            for align_tuple in align_tuples:
                align_str += [f'{align_tuple[0]} ({align_tuple[1]})'] 
            sbe_proc_str += [f'   > {", ".join(align_str)}']
            ct += 1

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
            sbe_proc_str += [f'{ct}. Bin averaged ({bin_size} {bin_unit}).']
            sbe_proc_str += [f'   > {binavg_excl_str}{bin_skipover_str}.']
            sbe_proc_str += [f'   > {surfbin_str}.']
            SBE_binned = f'{bin_size} {bin_unit} (SBE software)'
            ct += 1
    try:
        ds.attrs['binned'] = SBE_binned
    except: pass
    ds.attrs['SBE_processing'] = '\n'.join(sbe_proc_str)
    ds.attrs['SBE_processing_date'] = proc_date_ISO8601
    ds.attrs['history'] += f'\n{history_str}'
    ds.attrs['source_files'] = f'{src_files_raw} -> {header_info["source_file"].upper()}'

    return ds

def _read_sensor_info(source_file, verbose = False):
    '''
    Look through the header for information about sensors:
        - Serial numbers
        - Calibration dates  
    '''

    sensor_dict = {}

    # Define stuff we want to remove from strings
    drop_str_patterns = ['<.*?>', '\n', ' NPI']
    drop_str_pattern_comb = '|'.join(drop_str_patterns)

    with open(source_file, 'r', encoding = 'latin-1') as f:
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

                    for sensor_str, var_key in \
                        vardef.sensor_info_dict_SBE.items():
                    
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
                        serial_number_index = line.rindex('<SerialNumber>')+14

                        SN_instr = re.sub(drop_str_pattern_comb, '', 
                                               line[serial_number_index:])

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


            # Stop reading after the END string (.cnv, .hdr)
            if '*END*' in line:
                return sensor_dict
    # For .btl files (no end string) - just read the whole file before 
    # before returning

    return sensor_dict

## INTERNAL FUNCTIONS: MODIFY THE DATASET

def _assign_specified_lat_lon_station(ds, lat, lon, station):
    '''
    Assign values to latitude, longitude, station attributes
    if specified by the user (not None).
    '''

    if lat:
        ds.attrs['latitude'] = lat 
    if lon:
        ds.attrs['longitude'] = lon
    if station:
        ds.attrs['station'] = station 

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

def _add_time_dim_profile(ds, epoch = '1970-01-01', 
                          time_source = 'sample_time',
                          suppress_latlon_warning = False):
    '''
    Add a 0-dimensional TIME coordinate to a profile.
    Also adds the 0-d variables STATION, LATITUDE, and LONGITUDE.

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
            'long_name':'Average time of measurements',
            'SBE_source_variable':ds.TIME_SAMPLE.SBE_source_variable}
    else:
        start_time_num = time.ISO8601_to_datenum(ds.attrs['start_time'])
        ds = ds.assign_coords({'TIME':[start_time_num]})
        ds.TIME.attrs = {'units' : f'Days since {epoch} 00:00',
                        'standard_name' : 'time',
                        'long_name':'Start time of profile',
                        'comment':f'Source: {ds.start_time_source}'}
        
    
    # Add STATION
    ds = _add_station_variable(ds)
    ds = _add_latlon_variables(ds, suppress_latlon_warning)
    
    return ds

def _add_station_variable(ds):
    '''
    Adds a 0-d STATION variable to a profile.

    Grabs the value from the station attribute.

    (Requires that a 0-d TIME dimension exists, see 
    _add_time_dim_profile).

    '''

    station_array = xr.DataArray([ds.station], dims = 'TIME', coords={'TIME': ds.TIME},
        attrs = {'long_name' : 'CTD station ID',  'cf_role':'profile_id'})
    ds['STATION'] = station_array

    return ds

def _add_latlon_variables(ds, suppress_latlon_warning = False):
    '''
    Adds a 0-d STATION variable to a profile.

    Grabs the value from the station attribute.

    (Requires that a 0-d TIME dimension exists, see 
    _add_time_dim_profile).

    '''

    missing = False

    if 'latitude' in ds.attrs:
        if ds.latitude: # If not "None"
            lat_value = ds.latitude
        else:
            lat_value = np.nan
            missing = 'latitude'

    elif 'LATITUDE_SAMPLE' in ds:
        lat_value = ds.LATITUDE_SAMPLE.mean()
    else:
        lat_value = np.nan
        missing = 'latitude'

    lat_array = xr.DataArray([lat_value], dims = 'TIME', 
        coords={'TIME': ds.TIME},
        attrs = {'standard_name': 'latitude','units': 'degree_north', 
                'long_name': 'latitude',}) 
                
    ds['LATITUDE'] = lat_array

    if 'longitude' in ds.attrs:
        if ds.longitude: # If not "None"
            lon_value = ds.longitude
        else:
            lon_value = np.nan
            if missing:
                missing += ', longitude'
            else:
                missing = 'longitude'

    elif  'LONGITUDE_SAMPLE' in ds:
        lon_value = ds.LONGITUDE_SAMPLE.mean()
    else:
        lon_value = np.nan
        if missing:
            missing += ', longitude'
        else:
            missing = 'longitude'

    lon_array = xr.DataArray([lon_value], dims = 'TIME', 
        coords={'TIME': ds.TIME},
        attrs = {'standard_name': 'longitude','units': 'degree_east', 
                 'long_name': 'longitude',}) 
    ds['LONGITUDE'] = lon_array


    if missing and suppress_latlon_warning==False:
        warn_str = (f'{ds.STATION.values[0]}: Unable to find [{missing}] ' 
            'in .cnv file --> Assigning NaN values.')
        print(f'NOTE!: {warn_str}')
    
    return ds

def _add_header_attrs(ds, header_info, station_from_filename = False,
                      decimals_latlon = 4):
    '''
    Add the following as attributes if they are available from the header:

        ship, cruise, station, latitude, longitude

    If the attribute is already assigned, we don't change it 

    If we don't have a station, we use the cnv file name base.
    (can be forced by setting station_from_filename = True)
    '''
    for key in ['ship', 'cruise_name', 'station', 'latitude', 'longitude']:

        if key in header_info and key not in ds.attrs:

            ds.attrs[key] = header_info[key]
            if key in ['latitude', 'longitude'] and ds.attrs[key]!=None:
                ds.attrs[key] = np.round(ds.attrs[key], decimals_latlon)
    
    # Grab station from filename (stripping away .cnv and _bin)
    if 'station' not in ds.attrs or station_from_filename:
        station_from_filename = (
            header_info['source_file'].replace(
            '.cnv', '').replace('.CNV', '')
            .replace('_bin', '').replace('_BIN', '')
            .replace('.btl', '').replace('.BTL', ''))
        
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

def _update_variables(ds, source_file):
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
    
    sensor_info = _read_sensor_info(source_file)

    for old_name in ds.keys():
        old_name_cap = old_name.upper()
        
        # For .btl-files we can have valiables with a _std suffix 
        if old_name_cap.endswith('_STD'):
            old_name_cap = old_name_cap.replace('_STD', '')
            std_suffix = True
        else:
            std_suffix = False

        if old_name_cap in vardef.SBE_name_map:

            var_dict = vardef.SBE_name_map[old_name_cap]

            if not std_suffix:
                new_name = var_dict['name']
            else:
                new_name = var_dict['name'] + '_std'

            unit = var_dict['units']
            ds = ds.rename({old_name:new_name})
            ds[new_name].attrs['units'] = unit

            if 'sensors' in var_dict:
                sensor_SNs = []
                sensor_caldates = []

                for sensor in var_dict['sensors']:  
                    try:
                        sensor_SNs += [sensor_info[sensor]['SN']]
                        sensor_caldates += [sensor_info[sensor]['cal_date']]

                    except:
                        print(f'*NOTE*: Failed to find sensor {sensor}'
                            f' associated with variable {old_name_cap}.\n'
                            f'(file: {source_file})')
                        sensor_SNs += ['N/A']
                        sensor_caldates += ['N/A']

                ds[new_name].attrs['sensor_serial_number'] = (
                    ', '.join(sensor_SNs))

                ds[new_name].attrs['sensor_calibration_date'] = (
                    ', '.join(sensor_caldates)) 

    return ds


def _change_dims_scan_to_pres(ds):
    '''
    Change coordinate / dimension from scan_count to 
    PRES. A good idea for pressure binned data (bad idea otherwise).
    '''
    ds = ds.set_coords('PRES')
    ds = ds.swap_dims({'scan_count': 'PRES'})
    ds = ds.drop('scan_count')

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

## INTERNAL FUNCTIONS: INSPECT

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

## INTERNAL FUNCTIONS: FORMAT CONVERSION

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

def _decdeg_from_line(line):
    '''
    From a line line  '** Latitude: 081 16.1347'
    return decimal degrees, e.g. 81.26891166

    NOTE: This is to parse ad-hoc formatted files. May not work for everything.
    The default behaviour is to look for NMEA Lat/Lon - this is preferable and
    should be present in a well formatted header!
    '''
    deg_min_str = line[(line.rfind(':')+2):].replace('\n','').split()
    if len(deg_min_str)==0: # If there is no actual lat/lon string..
        return None
    
    # Occasionally written without colon; as e.g. "Latitude N 081 12.33" 
    # -> Need to remove the strings before parsing degrees

    if isinstance(deg_min_str[0], str):
        is_number = False
        while is_number == False:
            try: 
                # Test if we have a number
                float(deg_min_str[0])
                is_number = True
            except:
                # Flip the sign is S or W (convention is positive N/E)
                if deg_min_str[0] in ['S', 'W']:
                    deg_min_str[1] = str(-float(deg_min_str[1]))
                deg_min_str = deg_min_str[1:]

    deg = float(deg_min_str[0])
    min = float(deg_min_str[1])
    min_decdeg = min/60
    sign_deg = np.sign(deg)
    decdeg = deg + sign_deg*min_decdeg

    return decdeg

def _convert_time(ds, header_info, epoch = '1970-01-01', 
                  suppress_time_warning = False, start_time_NMEA = False):
    '''
    Convert time either from julian days (TIME_JULD extracted from 'timeJ') 
    or from time elapsed in seconds (TIME_ELAPSED extracted from timeS). 

    suppress_time_warning: Don't show a warning if there are no 
    timeJ or timeS fields (useful for loops etc).

    '''

    if 'TIME_ELAPSED' in ds.keys():
        ds = _convert_time_from_elapsed(ds, header_info, 
            epoch = epoch, start_time_NMEA = start_time_NMEA)
        ds.TIME_SAMPLE.attrs['SBE_source_variable'] = 'timeS' 
    elif 'TIME_JULD' in ds.keys():
        ds = _convert_time_from_juld(ds, header_info, 
            epoch = epoch, start_time_NMEA = start_time_NMEA)
        ds.TIME_SAMPLE.attrs['SBE_source_variable'] = 'timeJ' 

    else:
        if not suppress_time_warning:
            print('\nNOTE: Failed to extract sample time info '
                      '(no timeS or timeJ in .cnv file).'
                      '\n(Not a big problem, the start_time '
                      'can be used to assign a profile time).')
    return ds

def _convert_time_from_elapsed(ds, header_info, 
        epoch = '1970-01-01', start_time_NMEA = False):
    '''
    Convert TIME_ELAPSED (sec)
    to TIME_SAMPLE (days since 1970-01-01)
    
    Only sensible reference I could find is here;
    https://search.r-project.org/CRAN/refmans/oce/html/read.ctd.sbe.html

    (_DSE: time since epoch)
    '''

    if start_time_NMEA:
        ref_time = header_info['NMEA_time'] 
    else:
        ref_time = header_info['start_time'] 

    start_time_DSE = ((ref_time- pd.Timestamp(epoch))
                                    / pd.to_timedelta(1, unit='D'))
    
    elapsed_time_days = ds.TIME_ELAPSED/86400

    time_stamp_DSE = start_time_DSE + elapsed_time_days

    ds['TIME_SAMPLE'] = time_stamp_DSE
    ds.TIME_SAMPLE.attrs['units'] = f'Days since {epoch} 00:00:00'
    ds = ds.drop('TIME_ELAPSED')

    return ds


def _convert_time_from_juld(ds, header_info, epoch = '1970-01-01',
            start_time_NMEA = False):
    '''
    Convert TIME_ELAPSED (sec)
    to TIME (days since 1970-01-01)
    
    Only sensible reference I could fnd is here;
    https://search.r-project.org/CRAN/refmans/oce/html/read.ctd.sbe.html

    (_DSE: time since epoch)
    '''

    if start_time_NMEA:
        ref_time = header_info['NMEA_time'] 
    else:
        ref_time = header_info['start_time'] 

    year_start = ref_time.replace(month=1, day=1, 
                        hour=0, minute=0, second=0)
    time_stamp = pd.to_datetime(ds.TIME_JULD-1, origin=year_start, 
                            unit='D', yearfirst=True, ).round('1s')    
    time_stamp_DSE = ((time_stamp- pd.Timestamp(epoch))
                                    / pd.to_timedelta(1, unit='D'))

    ds['TIME_SAMPLE'] = (('scan_count'), time_stamp_DSE)
    ds.TIME_SAMPLE.attrs['units'] = f'Days since {epoch} 00:00:00'
    ds = ds.drop('TIME_JULD')

    return ds

