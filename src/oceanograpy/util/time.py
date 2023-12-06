from matplotlib.dates import num2date
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np


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


def start_end_times_cftime_to_duration(start_cftime, end_cftime):
    '''
    Calculate time difference beetween two cftime timestamps
    and convert to ISO8601 (P[YYYY]-[MM]-[DD]T[hh]:[mm]:[ss])
    '''

    # Convert cftime objects to datetime objects
    start_dt = datetime(start_cftime.year, start_cftime.month, start_cftime.day, start_cftime.hour, 
                        start_cftime.minute, start_cftime.second)
    end_dt = datetime(end_cftime.year, end_cftime.month, end_cftime.day, end_cftime.hour, 
                      end_cftime.minute, end_cftime.second)

    # Calculate the time difference
    delta = relativedelta(end_dt, start_dt)

    # Format the delta as a string with leading zeros
    formatted_difference = (
        f"P{delta.years:04}-{delta.months:02}-{(delta.days):02}T"
        f"{delta.hours:02}:{delta.minutes:02}:{delta.seconds:02}"
    )

    return formatted_difference



def matlab_time_to_timestamp(matlab_time):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    if isinstance(matlab_time, (int, float)):
        # Single value case
        days = matlab_time % 1
        return datetime.fromordinal(
            int(matlab_time)) + timedelta(days=days) - timedelta(days=366)
    elif isinstance(matlab_time, (list, tuple, np.ndarray)):
        # Array case
        result = []
        for time in matlab_time:
            days = time % 1
            result.append(
                datetime.fromordinal(int(time)) + timedelta(days=days) 
                                     - timedelta(days=366))
        return np.array(result)
    else:
        raise ValueError("Input must be a single value or an array of Matlab datenums.")



def timestamp_to_matlab_time(timestamp):
    """
    Convert Python datetime into Matlab datenum.
    :param timestamp: Datetime object or an array of datetime objects
    :return:         Matlab datenum or an array of datenums corresponding 
                     to timestamp(s).
    """
    if isinstance(timestamp, datetime):
        # Single value case
        days = (timestamp - datetime.fromordinal(1)).days + 366
        matlab_time_stamp = (days + (timestamp - datetime.fromordinal(days))
                             .total_seconds() / (24 * 60 * 60)) + 366
        return matlab_time_stamp
    elif isinstance(timestamp, (list, tuple, np.ndarray)):
        # Array case
        result = []
        for ts in timestamp:
            days = (ts - datetime.fromordinal(1)).days + 366
            result.append(days + 
                (ts - datetime.fromordinal(days)).total_seconds() 
                / (24 * 60 * 60))

        matlab_time_stamp = np.array(result) + 366
        return matlab_time_stamp
    else:
        raise ValueError(
            "Input must be a single value or an array of Python datetimes.")


def timestamp_to_datenum(timestamps, epoch='1970-01-01'):
    """
    Convert a timestamp or an array of timestamps to the number of days since the epoch.

    Parameters:
    - timestamps (datetime, np.ndarray): A single datetime object or an array of datetime objects.
    - epoch (str): The reference epoch as a string in the format 'YYYY-MM-DD'. Defaults to '1970-01-01'.

    Returns:
    - np.ndarray: An array of floats representing the number of days since the epoch.
    """
    # Convert the epoch to a datetime object
    epoch_datetime = datetime.strptime(epoch, '%Y-%m-%d')

    # Ensure timestamps is an array
    timestamps = np.array(timestamps)

    # Calculate the differences between the timestamps and the epoch
    deltas = timestamps - epoch_datetime

    # Handle both datetime and numpy.datetime64 objects
    if isinstance(deltas[0], np.timedelta64):
        seconds_since_epoch = deltas.astype('timedelta64[s]').astype(float)
    else:
        seconds_since_epoch = np.array([delta.total_seconds() for delta in deltas])

    # Convert the time differences to days
    days_since_epoch = seconds_since_epoch / (60 * 60 * 24)

    return days_since_epoch


def convert_timenum_to_datetime(TIME, units, out_fmt='%d-%b-%Y %H:%M'):
    """
    Convert a numeric time value to a formatted datetime string.

    Parameters:
    - TIME (float): Numeric time value.
    - units (str): String specifying the time units.
    - out_fmt (str): Output format for the datetime string (default: '%d-%b-%Y %H:%M').

    Returns:
    - str: Formatted datetime string.

    """
    # Extract the reference date from the units
    reference_date_str = units.split('since')[-1].strip()
    reference_date = datetime.strptime(reference_date_str, '%Y-%m-%d %H:%M')

    # Calculate the datetime from the reference date and numeric time
    date_time = reference_date + timedelta(days=float(TIME))

    # Format the datetime string according to the specified format
    reference_date_string = date_time.strftime(out_fmt)

    return reference_date_string