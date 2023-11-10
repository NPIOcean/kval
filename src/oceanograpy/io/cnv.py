### OceanograPy.io.cnv.py ###

'''
CNV parsing.

Including pressure binning -> does that belong here or elsewhere? 
'''

import pandas as pd
import xarray as xr
import numpy as np
from oceanograpy.io import _variable_defs as vardef
import matplotlib.pyplot as plt
import re


def read_cnv(cnvfile, 
             apply_flags = True, 
             remove_upcast = True,
             inspect_plot = False,
             start_scan = None,
             end_scan = None):
    '''
    Read a .cnv file to netCDF. 

    Both some info from the header and the column data.

    Transfer variable names, time, etc from native format to our preferred
    format.

    TBD:
    - Better docs
    - Add other metadata from header
    - Apply to some other datasets for testing
    - pyTests?
    '''

    header_info = read_header(cnvfile)

    ds = _read_column_data_xr(cnvfile, header_info)
    ds = _update_variables(ds, cnvfile)
    ds = _convert_time(ds, header_info)
    ds0 = ds.copy()

    if apply_flags:
        ds = _apply_flag(ds)
        ds.attrs['SBE flags applied'] = True
    else:
        ds.attrs['SBE flags applied'] = False


    if remove_upcast:
        ds = _remove_upcast(ds)

    if start_scan:
        ds = _remove_start_scan(ds, start_scan)
    if end_scan:
        ds = _remove_end_scan(ds, end_scan)

    if inspect_plot:
        _inspect_downcast(ds, ds0, start_scan, end_scan)

    return ds



def bin_to_pressure(ds, dp = 1):
    '''
    Apply pressure binning into bins of *dp* dbar.

    Reproducing the SBE algorithm as documented in:
    https://www.seabird.com/cms-portals/seabird_com/
    cms/documents/training/Module13_AdvancedDataProcessing.pdf

    (See page 13 for the formula used)

    Equivalent to this in SBE terms
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
    # = not bin *average* but *linear estimate of variable at bin pressure*
    # (in practice a small but difference)

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

    # Set xarray option "keep_attrs" back to whatever it was
    xr.set_options(keep_attrs=_keep_attr_value)

    return ds_binned



def _convert_time(ds, header_info, epoch = '1970-01-01'):
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

    ds['TIME_ELAPSED'] = TIME_ELAPSED_DSE
    ds.TIME_ELAPSED.attrs['units'] = f'Days since {epoch} 00:00:00'

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


def _inspect_downcast(ds, ds0, start_scan=None, end_scan=None):
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



def _remove_upcast(ds):
    '''
    
    '''
    # Index of max pressure, taken as "end of downcast"
    max_pres_index = int(ds.PRES.argmax().data)

    # If the max index is a single value following invalid values,
    # we interpret it as the start of the upcast and use the preceding 
    # point as the "end of upcast index" 
    print()
    if (ds.scan_count[max_pres_index] 
        - ds.scan_count[max_pres_index-1]) > 1:
        
        max_pres_index -= 1

    # Remove everything after pressure max
    ds = ds.isel({'scan_count':slice(None, max_pres_index+1)})

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
                 delim_whitespace=True,
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
    
    with open(cnvfile, 'r') as f:

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

            # Read serial numbers
            if ' SN' in line:
                sn_info_str = ' '.join(line.split()[1:])
                hdict['SN_info'] += [sn_info_str]

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
             
        return hdict


def read_sensor_info(cnvfile, verbose = False):
    '''
    Look through the header for information about sensors:
        - Serial numbers
        - Calibration dates  
    '''

    sensor_dict = {}

    # Define stuff we want to remove from strings
    drop_str_patterns = ['<.*?>', '\n', ' NPI']
    drop_str_pattern_comb = '|'.join(drop_str_patterns)

    with open(cnvfile, 'r') as f:
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

    


#

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