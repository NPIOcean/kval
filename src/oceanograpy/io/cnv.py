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

import matplotlib.pyplot as plt
import re
from typing import Optional

def read_cnv(cnvfile: str,
             apply_flags: Optional[bool] = True,
             profile: Optional[str] = 'downcast',
             inspect_plot: Optional[bool] = False,
             start_scan: Optional[int] = None,
             end_scan: Optional[int] = None) -> xr.Dataset:
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
     

    Returns
    -------
    xarray.Dataset
        A dataset containing CTD data and associated attributes.

    TO DO
    ----- 
    - Checking and testing
    - Assign sensor metadata as variable attributes
    - Testing (why did a TIME coordinate appear for some files?)
    - Better docs 
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
    ds = _convert_time(ds, header_info)
    ds.attrs['history'] = header_info['start_history']
    ds = _read_SBE_proc_steps(ds, header_info)
    ds0 = ds.copy()


    sensor_dict = _read_sensor_info(cnvfile, verbose = False):

    sensor_dict 

    if apply_flags:
        ds = _apply_flag(ds)
        ds.attrs['SBE flags applied'] = True
    else:
        ds.attrs['SBE flags applied'] = False


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

    return ds


def _convert_time(ds, header_info, epoch = '1970-01-01'):
    '''
    Convert time either from julian days (timeJ) or from  time elapsed
     in seconds (timeS). 
    '''

    if 'TIME_ELAPSED' in ds.keys():
        ds = _convert_time_from_elapsed(ds, header_info, epoch = epoch)
    elif 'timeJ' in ds.keys():
        ds = _convert_time_from_juld(ds, header_info, epoch = epoch)
    else:
        raise Warning('Failed to extract time info (no timeS or timeJ in source)')
    
    return ds


def _convert_time_from_elapsed(ds, header_info, epoch = '1970-01-01'):
    '''
    Convert TIME_ELAPSED (sec)
    to TIME (days since 1970-01-01)
    
    Only sensible reference I could fnd is here;
    https://search.r-project.org/CRAN/refmans/oce/html/read.ctd.sbe.html

    (_DSE: time since epoch)
    '''

    start_time_DSE = ((header_info['start_time'] 
                                    - pd.Timestamp(epoch))
                                    / pd.to_timedelta(1, unit='D'))
    
    elapsed_time_days = ds.TIME_ELAPSED/86400

    TIME_ELAPSED_DSE = start_time_DSE + elapsed_time_days

    ds['TIME'] = TIME_ELAPSED_DSE
    ds.TIME.attrs['units'] = f'Days since {epoch} 00:00:00'
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
    time_stamp = pd.to_datetime(ds.timeJ-1, origin=year_start, unit='D', 
                        yearfirst=True, ).round('1s')    
    time_stamp_epoch = ((time_stamp- pd.Timestamp(epoch))
                                    / pd.to_timedelta(1, unit='D'))
    ds['TIME'] = time_stamp_epoch
    ds.TIME.attrs['units'] = f'Days since {epoch} 00:00:00'
    ds = ds.drop('timeJ')

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

    return ds



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
    
    sensor_info = read_sensor_info(cnvfile)

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
                hdict['col_nums'] += [int(line.split()[2])]
                hdict['col_names'] += [line.split()[4].replace(':', '')]
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
                    + ': Start of data collection.')
                
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
        hdict['cnvfile'] = cnvfile

        return hdict


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
    '''
    SBElines = header_info['SBEproc_hist']

    dmy_fmt = '%Y-%m-%d'

    sbe_proc_str = ['SBE SOFTWARE PROCESSING STEPS (extracted'
                    ' from .cnv file header)', ' '*110]

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





        # binavg_surface_bin = yes, min = 0.000, max = 2.000, value = 0.000

# binavg_skipover = 0
# binavg_surface_bin = yes, min = 0.000, max = 2.000, value = 0.000


    ds.attrs['SBE_processing'] = '\n'.join(sbe_proc_str)
    ds.attrs['SBE_processing_date'] = proc_date_ISO8601
    ds.attrs['history'] += f'\n{history_str}'
    ds.attrs['source_files'] = src_files_raw

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