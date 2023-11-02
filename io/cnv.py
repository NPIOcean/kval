### OceanograPy.io.cnv.py ###

'''
CNV parsing functions.
'''

import pandas as pd
import xarray as xr

def _read_column_data_pandas(startline):
    '''
    Reads columnar data from a single .cnv to a pandas array.

    TBD: 
     - Standardize variable names and attributes. 
     - Add relevant attributes from header

    '''
    da = pd.read_csv(fn, header = hd['hdr_end_line']+1,
                 delim_whitespace=True,
                 names = hd['col_names'])


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
                 'latitude', 'longitude', 'time']
        hdict = {hkey:[] for hkey in hkeys}

        # Flag that will be turned on when we read the SBE history section 
        start_read_history = False 

        # Go through the header line by line and extract specific information 
        # when we encounter specific terms dictated by the format  
        
        for n_line, line in enumerate(f.readlines()):

            # Read the column header info (which variable is in which data column)
            if '# name' in line:
                hdict['col_nums'] += [int(line.split()[2])]
                hdict['col_names'] += [line.split()[4]]
                hdict['col_longnames'] += [' '.join(line.split()[5:])]            

            # Read NMEA lat/lon/time
            if 'NMEA Latitude' in line:
                hdict['latitude'] = _nmea_lat_to_decdeg(*line.split()[-3:])
            if 'NMEA Longitude' in line:
                hdict['longitude'] = _nmea_lon_to_decdeg(*line.split()[-3:])
            if 'NMEA UTC' in line:
                nmea_time_split = line.split()[-4:]
                hdict['time'] = _nmea_time_to_datetime(*nmea_time_split)
                
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