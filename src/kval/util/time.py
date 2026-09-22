"""
KVAL.UTIL.TIME

Time conversions. This module deals with several genuinely different
"kinds" of time, which is the main source of past confusion here:

- ISO8601 string      e.g. '2021-06-01T12:00:00Z' -- human-readable,
                       timezone-aware.
- datetime.datetime    Python's standard library time object.
- numpy.datetime64     numpy's vectorized time type.
- pandas.Timestamp     pandas' time object (adds timezone handling).
- CF numeric time      a plain number + a 'units' string, e.g.
                       (18628.5, 'days since 1970-01-01'). This is how
                       kval datasets store TIME on disk and whenever
                       loaded with decode_cf=False -- the epoch is
                       whatever the file says, NOT fixed.
- matplotlib datenum   matplotlib's own num2date/date2num convention --
                       also "days since an epoch", but matplotlib's OWN
                       fixed epoch, not the file's. Easy to confuse with
                       CF numeric time above since both get called
                       "datenum" informally.
- MATLAB datenum       MATLAB's own convention: days since 0000-01-00. A
                       THIRD, different epoch, unrelated to either of the
                       above despite sharing the name "datenum". Only
                       relevant when reading MATLAB-originated files (see
                       matfile.py). Named explicitly as "matlab_datenum"
                       throughout this module so it's never confused with
                       the other two.
- decimal year         e.g. 2021.415 -- not an epoch offset at all, a
                       fraction-of-the-way-through-the-year
                       representation.

For converting a whole xr.Dataset's TIME coordinate between CF numeric
and datetime64, prefer kval.util.xr_funcs.time_as_datetime / time_as_float
instead of anything in this module -- those handle a Dataset's TIME
coordinate as a whole (including remembering/restoring the original
units). This module is for converting bare values/arrays, and for the
MATLAB- and ISO8601/duration-specific cases xr_funcs doesn't cover.
"""

from matplotlib.dates import num2date, date2num
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np


#### CANONICAL CF NUMERIC TIME <-> DATETIME CONVERSION
#
# Everything else in this module that converts between CF numeric time
# (a number + a 'units' string) and datetime goes through these two
# functions. They're thin wrappers around xarray's own CF-time coding,
# which is more robust than hand-rolled string parsing (many more units/
# calendar variations handled correctly).

def numeric_time_to_datetime(
    values: float | np.ndarray,
    units: str,
    calendar: str = "standard",
) -> np.ndarray:
    """
    Convert CF numeric time (a number or array + a 'units' string, e.g.
    'days since 1970-01-01') to datetime64.

    Args:
        values: Numeric time value(s).
        units: CF units string, e.g. 'days since 1970-01-01'.
        calendar: CF calendar. Defaults to 'standard'.

    Returns:
        np.ndarray of datetime64 (0-d array for scalar input).
    """
    return xr.coding.times.decode_cf_datetime(values, units, calendar=calendar)


def datetime_to_numeric_time(
    values,
    units: str,
    calendar: str = "standard",
) -> np.ndarray:
    """
    Convert datetime-like value(s) to CF numeric time under the given
    units/calendar.

    Args:
        values: datetime-like value(s) (datetime, datetime64, Timestamp,
            or an array of these).
        units: CF units string to encode to, e.g. 'days since 1970-01-01'.
        calendar: CF calendar. Defaults to 'standard'.

    Returns:
        np.ndarray of numeric time value(s).
    """
    num, _, _ = xr.coding.times.encode_cf_datetime(
        np.asarray(values), units=units, calendar=calendar
    )
    return num


def numeric_time_to_datestring(
    value: float, units: str, out_fmt: str = "%d-%b-%Y %H:%M", calendar: str = "standard"
) -> str:
    """
    Convert a single CF numeric time value to a formatted datetime string.

    Args:
        value: Numeric time value.
        units: CF units string, e.g. 'days since 1970-01-01'.
        out_fmt: strftime format for the output string.
        calendar: CF calendar. Defaults to 'standard'.

    Returns:
        Formatted datetime string.
    """
    dt = pd.Timestamp(numeric_time_to_datetime(value, units, calendar=calendar).item())
    return dt.strftime(out_fmt)


#### ISO8601 FORMATTING

def datetime_to_ISO8601(time_dt: datetime, zone: str = "Z") -> str:
    """
    Convert datetime to YYYY-MM-DDThh:mm:ss<zone>
    """
    time_fmt = f"%Y-%m-%dT%H:%M:%S{zone}"
    iso8601_time = time_dt.strftime(time_fmt)
    return iso8601_time


def datenum_to_ISO8601(datenum: float, zone: str = "Z") -> str:
    """
    Convert a matplotlib datenum (matplotlib's own num2date/date2num
    convention, NOT CF numeric time or MATLAB datenum) to
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
    Convert numpy datetime64 to CF numeric time (days since epoch).
    '''
    return datetime_to_numeric_time(dt64, units=f"days since {epoch}")


def ISO8601_to_datenum(time_str: str, epoch: str = "1970-01-01") -> float:
    """
    Convert YYYY-MM-DDThh:mm:ss<zone> to days since *epoch*.
    """
    iso8601_time = ISO8601_to_datetime(time_str, to_UTC=True)
    return datetime_to_numeric_time(
        iso8601_time.tz_localize(None), units=f"days since {epoch}"
    )


#### DURATIONS (distinct from a point in time)

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


def seconds_to_ISO8601(seconds: float) -> str:
    """
    Takes a number of seconds (e.g. a sampling rate) and
    returns a ISO8601 string (P[YYYY]-[MM]-[DD]T[hh]:[mm]:[ss]).
    """
    if seconds >= 86400:
        days = int(seconds // 86400)
        seconds_residual = seconds - days * 86400
    else:
        days = 0
        seconds_residual = seconds

    if seconds_residual >= 3600:
        hours = int(seconds_residual // 3600)
        seconds_residual = seconds_residual - hours * 3600
    else:
        hours = 0

    if seconds_residual >= 60:
        minutes = int(seconds_residual // 60)
        seconds_residual = seconds_residual - minutes * 60
    else:
        minutes = 0

    # Only show fractional seconds if the residual actually has a
    # fractional part; otherwise format as a plain integer.
    if seconds_residual == int(seconds_residual):
        seconds_str = f"{int(seconds_residual):02}"
    else:
        seconds_str = f"{seconds_residual:05.2f}"

    iso_str = (
        f"P0000-00-{days:02}T{hours:02}:{minutes:02}:{seconds_str}"
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


#### MATLAB DATENUM (MATLAB's own epoch: days since 0000-01-00 -- a
#### different convention from both CF numeric time and matplotlib's
#### datenum above. Only relevant for reading MATLAB-originated files.)

def matlab_datenum_to_datetime(
    matlab_datenum: float | list[float] | tuple[float] | np.ndarray
) -> datetime | np.ndarray:
    """
    Convert a MATLAB datenum into Python datetime.
    """
    if isinstance(matlab_datenum, (int, float)):
        days = np.float64(matlab_datenum % 1)
        return (
            datetime.fromordinal(int(matlab_datenum))
            + timedelta(days=days)
            - timedelta(days=366)
        )
    elif isinstance(matlab_datenum, (list, tuple, np.ndarray)):
        result = []
        for time in matlab_datenum:
            days = np.float64(time % 1)
            result.append(
                datetime.fromordinal(int(time))
                + timedelta(days=days)
                - timedelta(days=366)
            )
        return np.array(result)
    else:
        raise ValueError(
            "Input must be a single value or an array of MATLAB datenums."
        )


def matlab_datenum_to_mpl_datenum(
    matlab_datenum: float | list[float] | tuple[float] | np.ndarray
) -> datetime | np.ndarray:
    """
    Convert MATLAB datenum (days since 0000-01-00) to a matplotlib
    datenum (days since matplotlib's own epoch) -- useful for plotting
    MATLAB-originated time values directly.

    Args:
        matlab_datenum (float): MATLAB datenum in days.

    Returns:
        float: Corresponding matplotlib datenum in days.
    """
    time_stamp = matlab_datenum_to_datetime(matlab_datenum)
    mpl_datenum = date2num(time_stamp)

    return mpl_datenum


def datetime_to_matlab_datenum(
    timestamp: datetime | np.datetime64 | list | tuple | np.ndarray
) -> float | np.ndarray:
    """
    Convert Python datetime, numpy datetime64, or pandas Timestamp (single
    value or array-like) into MATLAB datenum.
    """
    is_scalar = isinstance(timestamp, (datetime, np.datetime64, pd.Timestamp))
    # Normalize everything to a pandas DatetimeIndex of Python datetimes
    # first -- this makes the function robust to datetime, datetime64,
    # Timestamp, and arrays/lists of any of these, rather than requiring
    # the caller to pre-convert to one specific type.
    pd_index = pd.DatetimeIndex(np.atleast_1d(timestamp))
    py_datetimes = pd_index.to_pydatetime()

    result = []
    for ts in py_datetimes:
        days = (ts - datetime.fromordinal(1)).days + 366
        result.append(
            days
            + (ts - datetime.fromordinal(days)).total_seconds()
            / (24 * 60 * 60)
        )
    matlab_datenum = np.array(result) + 366
    return matlab_datenum[0] if is_scalar else matlab_datenum


#### DECIMAL YEAR

def time_to_decimal_year(
        time: datetime | np.datetime64 | str | int | float
) -> float:
    """
    Convert various time formats to a decimal year.

    Args:
        time (datetime.datetime | np.datetime64 | str | int | float):
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
        time = pd.Timestamp(
            numeric_time_to_datetime(time, units="days since 1970-01-01").item()
        ).to_pydatetime()

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