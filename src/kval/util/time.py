from typing import Union, List, Tuple
from matplotlib.dates import num2date, date2num
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import numbers


def datetime_to_ISO8601(time_dt: datetime, zone: str = "Z") -> str:
    """
    Convert datetime to YYYY-MM-DDThh:mm:ss<zone>
    """
    time_fmt = f"%Y-%m-%dT%H:%M:%S{zone}"
    iso8601_time = time_dt.strftime(time_fmt)
    return iso8601_time


def datenum_to_ISO8601(datenum: float, zone: str = "Z") -> str:
    """
    Convert datenum (time since epoch, e.g. 18634.11) to
    YYYY-MM-DDThh:mm:ss<zone>.
    """
    time_dt = num2date(datenum)
    iso8601_time = datetime_to_ISO8601(time_dt)
    return iso8601_time


def ISO8601_to_datetime(time_str: str, to_UTC: bool = True) -> pd.Timestamp:
    """
    Convert YYYY-MM-DDThh:mm:ss<zone> to pandas datetime.
    Converting to UTC if to_UTC = True
    """
    iso8601_time = pd.to_datetime(time_str)
    if to_UTC:
        iso8601_time_utc = iso8601_time.tz_convert(tz="UTC")
        return iso8601_time_utc
    else:
        return iso8601_time

def dt64_to_datenum(dt64: np.datetime64, epoch: str = "1970-01-01") -> float:
    '''
    Convert numpy datetime64 to timenum (days since epoch)
    '''
    days_since_epoch = (
        (dt64 - np.datetime64(epoch)) / np.timedelta64(1, 'D'))
    return days_since_epoch



def ISO8601_to_datenum(time_str: str, epoch: str = "1970-01-01") -> float:
    """
    Convert YYYY-MM-DDThh:mm:ss<zone> to days since *epoch*.
    """
    iso8601_time = ISO8601_to_datetime(time_str, to_UTC=True)
    start_time_DSE = (
        iso8601_time - pd.Timestamp(epoch, tz="UTC")
    ) / pd.to_timedelta(1, unit="D")
    return start_time_DSE


def start_end_times_cftime_to_duration(
    start_cftime: datetime, end_cftime: datetime
) -> str:
    """
    Calculate time difference between two cftime timestamps
    and convert to ISO8601 (P[YYYY]-[MM]-[DD]T[hh]:[mm]:[ss])
    """
    start_dt = datetime(
        start_cftime.year,
        start_cftime.month,
        start_cftime.day,
        start_cftime.hour,
        start_cftime.minute,
        start_cftime.second,
    )
    end_dt = datetime(
        end_cftime.year,
        end_cftime.month,
        end_cftime.day,
        end_cftime.hour,
        end_cftime.minute,
        end_cftime.second,
    )
    delta = relativedelta(end_dt, start_dt)
    formatted_difference = (
        f"P{delta.years:04}-{delta.months:02}-{delta.days:02}T"
        f"{delta.hours:02}:{delta.minutes:02}:{delta.seconds:02}"
    )
    return formatted_difference


def seconds_to_ISO8601(seconds: int) -> str:
    """
    Takes an integer number of seconds (e.g. a sampling rate) and
    returns a ISO8601 string (P[YYYY]-[MM]-[DD]T[hh]:[mm]:[ss]).
    """
    seconds = int(seconds)
    seconds_residual = 0
    if seconds > 86400 - 1:
        days = int(seconds // 86400)
        seconds_residual = seconds - days * 86400
    else:
        days = 0
        seconds_residual = seconds
    if seconds_residual > 24 * 60 - 1:
        hours = int(seconds_residual // (3600))
        seconds_residual = seconds_residual - hours * 3600
    else:
        hours = 0
    if seconds_residual > 59:
        minutes = int(seconds_residual // (60))
        seconds_residual = seconds_residual - minutes * 60
    else:
        minutes = 0
    iso_str = (
        f"P0000-00-{days:02}T{hours:02}:{minutes:02}:{seconds_residual:02}"
    )
    return iso_str


def days_to_ISO8601(days: float) -> str:
    """
    Takes an number of days (e.g. a sampling rate) and
    returns a ISO8601 string (P[YYYY]-[MM]-[DD]T[hh]:[mm]:[ss]).
    """
    seconds = int(np.round(days*86400))
    iso_str = seconds_to_ISO8601(seconds=seconds)
    return iso_str


def matlab_time_to_datetime(
    matlab_time: Union[float, List[float], Tuple[float], np.ndarray]
) -> Union[datetime, np.ndarray]:
    """
    Convert Matlab datenum into Python datetime.
    """
    if isinstance(matlab_time, (int, float)):
        days = np.float64(matlab_time % 1)
        return (
            datetime.fromordinal(int(matlab_time))
            + timedelta(days=days)
            - timedelta(days=366)
        )
    elif isinstance(matlab_time, (list, tuple, np.ndarray)):
        result = []
        for time in matlab_time:
            days = np.float64(time % 1)
            result.append(
                datetime.fromordinal(int(time))
                + timedelta(days=days)
                - timedelta(days=366)
            )
        return np.array(result)
    else:
        raise ValueError(
            "Input must be a single value or an array of Matlab datenums."
        )


def matlab_time_to_python_time(
    matlab_time: Union[float, List[float], Tuple[float], np.ndarray]
) -> Union[datetime, np.ndarray]:
    """
    Convert MATLAB datenum (days) to Matplotlib dates (days).

    This function converts a time value from MATLAB's datenum format,
    which counts days from 00-Jan-0000, to Matplotlib's date format,
    which counts days from 01-Jan-1970.

    Args:
        mattime (float): MATLAB datenum in days.

    Returns:
        float: Corresponding Matplotlib date in days.    """
    time_stamp = matlab_time_to_datetime(matlab_time)
    python_time = date2num(time_stamp)

    return python_time


def timestamp_to_matlab_time(
    timestamp: Union[datetime, List[datetime], Tuple[datetime], np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert Python datetime into Matlab datenum.
    """
    if isinstance(timestamp, datetime):
        days = (timestamp - datetime.fromordinal(1)).days + 366
        matlab_time_stamp = (
            days
            + (timestamp - datetime.fromordinal(days)).total_seconds()
            / (24 * 60 * 60)
        ) + 366
        return matlab_time_stamp
    elif isinstance(timestamp, (list, tuple, np.ndarray)):
        result = []
        for ts in timestamp:
            days = (ts - datetime.fromordinal(1)).days + 366
            result.append(
                days
                + (ts - datetime.fromordinal(days)).total_seconds()
                / (24 * 60 * 60)
            )
        matlab_time_stamp = np.array(result) + 366
        return matlab_time_stamp
    else:
        raise ValueError(
            "Input must be a single value or an array of Python datetimes."
        )


def timestamp_to_datenum(
    timestamps: Union[datetime, np.ndarray], epoch: str = "1970-01-01"
) -> np.ndarray:
    """
    Convert a timestamp or an array of timestamps to the number of days since
    the epoch.
    """
    try:
        epoch_datetime = datetime.strptime(epoch, "%Y-%m-%d")
    except ValueError:
        try:
            epoch_datetime = datetime.strptime(epoch, "Days since %Y-%m-%d")
        except ValueError:
            epoch_datetime = datetime.strptime(
                epoch, "Days since %Y-%m-%d %H:%M"
            )
    timestamps = np.array(timestamps)
    deltas = timestamps - epoch_datetime
    if isinstance(deltas[0], np.timedelta64):
        seconds_since_epoch = deltas.astype("timedelta64[s]").astype(float)
    else:
        seconds_since_epoch = np.array(
            [delta.total_seconds() for delta in deltas]
        )
    days_since_epoch = seconds_since_epoch / (60 * 60 * 24)
    return days_since_epoch


def datenum_to_timestamp(
    datenum: Union[float, np.ndarray], epoch: str = "1970-01-01"
) -> Union[datetime, np.ndarray]:
    """
    Convert the number of days since the epoch to a timestamp or an array of
    timestamps.
    """
    try:
        epoch_datetime = datetime.strptime(epoch, "%Y-%m-%d")
    except ValueError:
        try:
            epoch_datetime = datetime.strptime(epoch, "Days since %Y-%m-%d")
        except ValueError:
            try:
                epoch_datetime = datetime.strptime(
                    epoch, "Days since %Y-%m-%d %H:%M"
                )
            except ValueError:
                epoch_datetime = datetime.strptime(
                    epoch, "Days since %Y-%m-%d %H:%M:%S"
                )
    datenum = np.array(datenum)
    seconds_since_epoch = datenum * (60 * 60 * 24)
    if isinstance(seconds_since_epoch, (numbers.Number)):
        timedelta_object = timedelta(seconds=int(seconds_since_epoch))
        timestamp = epoch_datetime + timedelta_object
        return timestamp
    else:
        timedelta_objects = np.array(
            [timedelta(seconds=int(sec)) for sec in seconds_since_epoch]
        )
        timestamps = np.array(
            [epoch_datetime + td for td in timedelta_objects]
        )
        return timestamps



def convert_timenum_to_datetime(
        TIME: float, units: str, ) -> str:
    """
    Convert a numeric time value to a datetime object.
    """
    reference_date_str = units.split("since")[-1].strip()
    try:
        reference_date = datetime.strptime(
            reference_date_str, "%Y-%m-%d %H:%M"
        )
    except ValueError:
        try:
            reference_date = datetime.strptime(
                reference_date_str, "%Y-%m-%d %H:%M:%S"
            )
        except ValueError:
            reference_date = datetime.strptime(reference_date_str, "%Y-%m-%d")


    # Calculate the datetime from the reference date and numeric time
    date_time = reference_date + timedelta(days=float(TIME))
    return date_time


def convert_timenum_to_datestring(
    TIME: float, units: str, out_fmt: str = "%d-%b-%Y %H:%M"
) -> str:
    """
    Convert a numeric time value to a formatted datetime string.
    """

    # Calculate the datetime from the reference date and numeric time
    date_time = convert_timenum_to_datetime(TIME=TIME, units=units)
    # Format the datetime string according to the specified format
    reference_date_string = date_time.strftime(out_fmt)

    return reference_date_string


def time_to_decimal_year(
        time: Union[datetime, np.datetime64, str, Union[int, float]]
        ) -> float:
    """
    Convert various time formats to a decimal year.

    Args:
        time (Union[datetime.datetime, np.datetime64, str, int, float]):
            The time input which can be:
            - A `datetime.datetime` object.
            - A `numpy.datetime64` object.
            - A string in ISO format (e.g., '2021-02-01' or
              '2021-02-01 23:30:00').
            - A numeric value representing days since '1970-01-01'.

    Returns:
        float: The corresponding decimal year as a floating-point number.

    Raises:
        ValueError: If the string cannot be parsed into a datetime object.
        TypeError: If the time input is of an unexpected type.
    """

    # If time is a string, try to parse it into a datetime object
    if isinstance(time, str):
        try:
            time = datetime.fromisoformat(time)
        except ValueError:
            raise ValueError(f"Invalid time format: {time}")

    # If time is a numpy datetime64, convert it to a datetime object
    elif isinstance(time, np.datetime64):
        time = pd.to_datetime(time).to_pydatetime()

    # If time is a numeric value, interpret it as days since 1970-01-01
    elif isinstance(time, (int, float)):
        base_date = datetime(1970, 1, 1)
        time = base_date + timedelta(days=time)

    # Ensure time is now a datetime object
    if not isinstance(time, datetime):
        raise TypeError("Expected time to be datetime, string, or numeric,"
                        f" got {type(time)} instead.")

    # Get the start and end of the year
    year_start = datetime(time.year, 1, 1)
    next_year_start = datetime(time.year + 1, 1, 1)

    # Calculate the length of the year and the time elapsed
    year_length = (next_year_start - year_start).total_seconds()
    time_elapsed = (time - year_start).total_seconds()

    # Compute the decimal year
    decimal_year = time.year + time_elapsed / year_length

    return decimal_year
