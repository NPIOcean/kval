from matplotlib.dates import num2date
import pandas as pd

def datetime_to_ISO8601(time_dt, zone = 'Z'):
    '''
    Convert datetime to YYYY-MM-DDThh:mm:ss<zone>

    *zone* defines the time zone. Default: 'Z' (UTC).

    (see https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3)
    '''

    time_fmt = f'%Y-%m-%dT%H:%M:%S{zone}'
    iso8601_time = time_dt.strftime(time_fmt)

    return iso8601_time

def datenum_to_ISO8601(datenum, zone = 'Z'):
    '''
    Convert datenum (time since epoch, e.g. 18634.11) to YYYY-MM-DDThh:mm:ss<zone>.

    *zone* defines the time zone. Default: 'Z' (UTC).

    (see https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3)
    '''

    time_dt = num2date(datenum)
    iso8601_time = datetime_to_ISO8601(time_dt)

    return iso8601_time

def ISO8601_to_datetime(time_str, to_UTC = True):
    '''
    Convert YYYY-MM-DDThh:mm:ss<zone> or
    to pandas datetime.

    Converting to UTC if 
        to_UTC = True

    
    (see https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3)
    '''

    iso8601_time = pd.to_datetime(time_str)
    if to_UTC:
        iso8601_time_utc = iso8601_time.tz_convert(tz = 'UTC')
        return iso8601_time_utc
    else:
        return iso8601_time

def ISO8601_to_datenum(time_str, epoch = '1970-01-01'):
    '''
    Convert YYYY-MM-DDThh:mm:ss<zone> or
    to days since *epoch*.
    
    (see https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3)
    '''

    iso8601_time = ISO8601_to_datetime(time_str, to_UTC = True)
    start_time_DSE = ((iso8601_time- pd.Timestamp(epoch, tz = 'UTC'))
                                    / pd.to_timedelta(1, unit='D'))
    return start_time_DSE


def start_end_times_to_duration(start_time, end_time):
    '''
    *TBW*

    Calculate time difference beetween two timestamps
    and convert to ISO8601 (P[YYYY]-[MM]-[DD]T[hh]:[mm]:[ss])
    '''

    pass