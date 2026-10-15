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

Time zone: kval treats all times as UTC. Times are stored without a time
zone (datetime64, or CF numeric time "<units> since <epoch>" with no
offset, which CF reads as UTC), and ISO 8601 strings are written with a
"Z". Instrument times are taken as they are in the files; kval cannot
tell whether an instrument clock was set to UTC, so check that it was.

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
from kval.util.netcdf import bounds_map


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
    values = np.asarray(values)
    # Normalise to a flat datetime64[ns] array (handles datetime, Timestamp
    # and datetime64 of any resolution), encode, and restore the shape.
    # Older xarray versions (before ~2025) only encode datetime64[ns].
    flat = pd.to_datetime(values.ravel()).values.astype("datetime64[ns]")
    num, _, _ = xr.coding.times.encode_cf_datetime(
        flat, units=units, calendar=calendar, dtype=np.float64
    )
    return np.asarray(num, dtype=np.float64).reshape(values.shape)


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

def _seconds_str(seconds: float, width: int = 0) -> str:
    """Seconds rounded to the microsecond, without trailing zeros:
    5 -> '5' (or '05' with width=2), 0.5 -> '0.5' (or '00.5')."""
    seconds = round(float(seconds), 6)
    if seconds.is_integer():
        return f"{int(seconds):0{width}d}"
    whole, frac = f"{seconds:.6f}".rstrip("0").split(".")
    return f"{int(whole):0{width}d}.{frac}"


def _iso8601_duration(years=0, months=0, days=0, hours=0, minutes=0,
                      seconds=0.0) -> str:
    """
    ISO 8601 duration.

    Uses the extended alternative format P[YYYY]-[MM]-[DD]T[hh]:[mm]:[ss],
    which ACDD 1.3 recommends for time_coverage_duration and
    time_coverage_resolution, e.g. 'P0000-00-00T00:15:00'. Sub-second
    durations get a decimal fraction on the seconds, e.g.
    'P0000-00-00T00:00:00.5'.

    The alternative format can not hold values above the carry-over points
    (12 months, 30 days, 24 hours, 60 minutes, 60 seconds). Durations that
    do (e.g. 150 days, which can not be written in months without a
    calendar) are written in the designator format instead, e.g. 'P150D'
    or 'P400DT1M1S'. ACDD accepts this format too.
    """
    years, months, days = int(years), int(months), int(days)
    hours, minutes = int(hours), int(minutes)
    seconds = round(float(seconds), 6)

    if (years <= 9999 and months <= 12 and days <= 30 and hours <= 24
            and minutes <= 60 and seconds <= 60):
        return (f"P{years:04d}-{months:02d}-{days:02d}T"
                f"{hours:02d}:{minutes:02d}:{_seconds_str(seconds, 2)}")

    # Designator format; zero components are left out.
    date_part = "".join(f"{value}{unit}" for value, unit in
                        ((years, "Y"), (months, "M"), (days, "D")) if value)
    time_part = "".join(f"{value}{unit}" for value, unit in
                        ((hours, "H"), (minutes, "M"),
                         (_seconds_str(seconds), "S")) if float(value) != 0)
    return "P" + date_part + ("T" + time_part if time_part else "")


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
    return _iso8601_duration(delta.years, delta.months, delta.days,
                             delta.hours, delta.minutes, delta.seconds)


def seconds_to_ISO8601(seconds: float) -> str:
    """
    Takes a number of seconds (e.g. a sampling interval) and returns an
    ISO 8601 duration, in days, hours, minutes and seconds.

    Uses the format P[YYYY]-[MM]-[DD]T[hh]:[mm]:[ss] (e.g. 900 ->
    'P0000-00-00T00:15:00', 0.5 -> 'P0000-00-00T00:00:00.5'). Durations
    of more than 30 days do not fit that format and are written in the
    designator format (e.g. 150 days -> 'P150D').
    """
    if seconds < 0:
        raise ValueError("seconds_to_ISO8601: duration must not be negative.")
    seconds = round(float(seconds), 6)
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, rest = divmod(rest, 60)
    return _iso8601_duration(days=int(days), hours=int(hours),
                             minutes=int(minutes), seconds=rest)


def days_to_ISO8601(days: float) -> str:
    """
    Takes a number of days (e.g. a sampling interval) and returns an
    ISO 8601 duration (see seconds_to_ISO8601). Rounded to the millisecond.
    """
    seconds = round(float(days) * 86400, 3)
    return seconds_to_ISO8601(seconds=seconds)


#### MATLAB DATENUM (MATLAB's own epoch: days since 0000-01-00 -- a
#### different convention from both CF numeric time and matplotlib's
#### datenum above. Only relevant for reading MATLAB-originated files.)

def matlab_datenum_to_datetime(
    matlab_datenum: float | list[float] | tuple[float] | np.ndarray
) -> datetime | np.ndarray:
    """
    Convert a MATLAB datenum into Python datetime.
    """
    if isinstance(matlab_datenum, (int, float, np.integer, np.floating)):
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
        ).round("us").to_pydatetime()

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




# Seconds per CF time unit (for judging resolution of numeric time variables).
_CF_UNIT_SECONDS = {
    "second": 1.0, "seconds": 1.0, "s": 1.0,
    "minute": 60.0, "minutes": 60.0, "min": 60.0,
    "hour": 3600.0, "hours": 3600.0, "h": 3600.0,
    "day": 86400.0, "days": 86400.0, "d": 86400.0,
}


def is_time_like(name, var, ds=None) -> bool:
    """True if a variable is a time variable (by name or CF metadata).

    Time variables must never be down-cast: values are large offsets from an
    epoch (e.g. ~19 000 days since 1970), so float32 cannot resolve the
    sampling interval (see `numeric_time_resolution_seconds`).

    If the dataset `ds` is given, the boundary variable of a time variable
    (e.g. TIME_bnds, named by TIME's 'bounds' attribute) also counts as
    time-like. CF boundary variables carry no units of their own (they
    inherit TIME's), so they can't be recognised from their own attributes.
    """
    attrs = getattr(var, "attrs", {}) or {}
    units = str(attrs.get("units", "")).lower()
    if (
        str(name).upper() == "TIME"
        or " since " in units
        or attrs.get("standard_name") == "time"
        or attrs.get("axis") == "T"
        or getattr(var.dtype, "kind", "") in ("M", "m")
    ):
        return True
    if ds is not None:
        parent = bounds_map(ds).get(name)
        if parent is not None and parent != name:
            return is_time_like(parent, ds[parent])
    return False


def numeric_time_resolution_seconds(values, units: str, dtype=None) -> float:
    """Smallest step [s] that `dtype` can represent at the largest time value.

    For float32 and ~19 000 days since 1970 this is ~170 s.
    Returns np.nan if the unit cannot be interpreted.
    """
    unit = str(units).split(" since ")[0].strip().lower()
    factor = _CF_UNIT_SECONDS.get(unit)
    values = np.asarray(values)
    if factor is None or values.size == 0 or np.isnan(values).all():
        return np.nan
    dtype = np.dtype(dtype or values.dtype)
    if dtype.kind != "f":
        return factor  # integer time: one unit
    vmax = np.nanmax(np.abs(values))
    return float(np.spacing(dtype.type(vmax))) * factor