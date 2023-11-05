from matplotlib.dates import num2date

def datetime_to_ISO8601(time_dt, zone = 'Z'):
    '''
    Convert datetime to YYYY-MM-DDThh:mm:ss<zone>

    *zone* defines the time zone. Default: 'Z' (UTC).

    (see https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3)
    '''

    time_fmt = f'%Y-%M-%dT%H:%M:%S{zone}'
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


def start_end_times_to_duration(start_time, end_time):
    '''
    *TBW*

    Calculate time difference beetween two timestamps
    and convert to ISO8601 (P[YYYY]-[MM]-[DD]T[hh]:[mm]:[ss])
    '''

    pass