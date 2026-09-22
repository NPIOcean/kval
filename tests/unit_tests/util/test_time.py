"""
Tests for kval.util.time.

Covers the two canonical CF numeric<->datetime functions (which everything
else in the module now delegates to), the MATLAB datenum trio (renamed,
and datetime_to_matlab_datenum was fixed to handle datetime64 input during
the time.py cleanup -- tested explicitly here), and the remaining ISO8601/
duration/decimal-year utilities.
"""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from kval.util import time


# ---------------------------------------------------------------------
# Canonical pair: numeric_time_to_datetime / datetime_to_numeric_time
# ---------------------------------------------------------------------

def test_numeric_time_to_datetime_scalar():
    result = time.numeric_time_to_datetime(18628.5, 'days since 1970-01-01')
    assert pd.Timestamp(result.item()) == pd.Timestamp('2021-01-01T12:00:00')


def test_numeric_time_to_datetime_array():
    result = time.numeric_time_to_datetime(
        np.array([0, 1, 2]), 'days since 2021-01-01'
    )
    expected = pd.date_range('2021-01-01', periods=3)
    np.testing.assert_array_equal(result, expected.values)


def test_datetime_to_numeric_time_scalar():
    result = time.datetime_to_numeric_time(
        np.datetime64('2021-01-01T12:00:00'), 'days since 1970-01-01'
    )
    assert float(result) == pytest.approx(18628.5)


def test_datetime_to_numeric_time_uses_float64_not_int64():
    """Regression check for the dtype fix -- half-day offsets must not
    trigger xarray's int64-can't-represent-this warning/fallback dance;
    dtype=np.float64 should be requested explicitly up front."""
    result = time.datetime_to_numeric_time(
        np.datetime64('2021-01-01T12:00:00'), 'days since 1970-01-01'
    )
    assert np.issubdtype(np.asarray(result).dtype, np.floating)


def test_numeric_datetime_round_trip():
    original = np.array([0.0, 1.5, 3.25])
    units = 'days since 2021-01-01'
    dt = time.numeric_time_to_datetime(original, units)
    back = time.datetime_to_numeric_time(dt, units)
    np.testing.assert_allclose(back, original)


# ---------------------------------------------------------------------
# numeric_time_to_datestring
# ---------------------------------------------------------------------

def test_numeric_time_to_datestring_default_format():
    result = time.numeric_time_to_datestring(18628.5, 'days since 1970-01-01')
    assert result == '01-Jan-2021 12:00'


def test_numeric_time_to_datestring_custom_format():
    result = time.numeric_time_to_datestring(
        18628.5, 'days since 1970-01-01', out_fmt='%Y-%m-%d'
    )
    assert result == '2021-01-01'


# ---------------------------------------------------------------------
# ISO8601 utilities
# ---------------------------------------------------------------------

def test_datetime_to_ISO8601():
    result = time.datetime_to_ISO8601(datetime(2021, 6, 1, 12, 0, 0))
    assert result == '2021-06-01T12:00:00Z'


def test_ISO8601_to_datetime_utc():
    result = time.ISO8601_to_datetime('2021-06-01T12:00:00Z', to_UTC=True)
    assert result == pd.Timestamp('2021-06-01T12:00:00', tz='UTC')


def test_datenum_to_ISO8601_and_back_are_consistent():
    """datenum_to_ISO8601 uses matplotlib's own epoch (via num2date), so
    round-trip it through matplotlib's date2num rather than assuming any
    particular numeric epoch."""
    from matplotlib.dates import date2num
    mpl_num = date2num(datetime(2021, 6, 1, 12, 0, 0))
    result = time.datenum_to_ISO8601(mpl_num)
    assert result == '2021-06-01T12:00:00Z'


def test_ISO8601_to_datenum_matches_dt64_to_datenum():
    """Both should agree on the same date via the same default epoch,
    since they now both delegate to the same canonical function."""
    a = time.ISO8601_to_datenum('2021-06-01T00:00:00Z')
    b = time.dt64_to_datenum(np.datetime64('2021-06-01'))
    assert a == pytest.approx(b)


# ---------------------------------------------------------------------
# Durations (distinct from a point in time)
# ---------------------------------------------------------------------

class _FakeCftime:
    """Minimal stand-in for a cftime object, exposing the same
    year/month/day/hour/minute/second attributes."""
    def __init__(self, dt):
        (self.year, self.month, self.day,
         self.hour, self.minute, self.second) = (
            dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
        )


def test_start_end_times_cftime_to_duration():
    start = _FakeCftime(datetime(2021, 1, 1, 0, 0, 0))
    end = _FakeCftime(datetime(2021, 1, 2, 6, 30, 0))
    result = time.start_end_times_cftime_to_duration(start, end)
    assert result == 'P0000-00-01T06:30:00'


def test_seconds_to_ISO8601_whole_seconds():
    assert time.seconds_to_ISO8601(3661) == 'P0000-00-00T01:01:01'


def test_seconds_to_ISO8601_more_than_a_day():
    assert time.seconds_to_ISO8601(90000) == 'P0000-00-01T01:00:00'


def test_days_to_ISO8601():
    assert time.days_to_ISO8601(1.5) == 'P0000-00-01T12:00:00'


# ---------------------------------------------------------------------
# MATLAB datenum trio
# ---------------------------------------------------------------------

def test_matlab_datenum_to_datetime_known_reference():
    """730486 is the well-known MATLAB datenum for 2000-01-01 00:00:00."""
    result = time.matlab_datenum_to_datetime(730486.0)
    assert result == datetime(2000, 1, 1)


def test_datetime_to_matlab_datenum_known_reference():
    result = time.datetime_to_matlab_datenum(datetime(2000, 1, 1))
    assert result == pytest.approx(730486.0)


def test_matlab_datenum_round_trip_scalar():
    original = 738521.5
    back = time.datetime_to_matlab_datenum(time.matlab_datenum_to_datetime(original))
    assert back == pytest.approx(original)


def test_matlab_datenum_round_trip_array():
    original = np.array([738521.5, 738522.5, 738523.5])
    back = time.datetime_to_matlab_datenum(time.matlab_datenum_to_datetime(original))
    np.testing.assert_allclose(back, original)


def test_datetime_to_matlab_datenum_accepts_datetime64():
    """Regression test: datetime_to_matlab_datenum previously broke on
    numpy datetime64 input (a real bug caught while updating callers
    during the time.py cleanup) -- this is exactly the shape of input
    kval.util.time.numeric_time_to_datetime produces."""
    dt64_array = time.numeric_time_to_datetime(
        np.array([18628.5, 18629.5]), 'days since 1970-01-01'
    )
    result = time.datetime_to_matlab_datenum(dt64_array)
    assert len(result) == 2
    assert np.all(np.isfinite(result))


def test_datetime_to_matlab_datenum_accepts_datetime64_scalar():
    dt64_scalar = np.datetime64('2021-01-01T12:00:00')
    result = time.datetime_to_matlab_datenum(dt64_scalar)
    assert np.isscalar(result) or result.ndim == 0


def test_datetime_to_matlab_datenum_accepts_list_of_datetimes():
    result = time.datetime_to_matlab_datenum(
        [datetime(2021, 1, 1), datetime(2021, 1, 2)]
    )
    assert len(result) == 2
    np.testing.assert_allclose(np.diff(result), [1.0])


def test_matlab_datenum_to_mpl_datenum_is_plottable_number():
    from matplotlib.dates import num2date
    result = time.matlab_datenum_to_mpl_datenum(730486.0)
    assert num2date(result).replace(tzinfo=None) == datetime(2000, 1, 1)


# ---------------------------------------------------------------------
# time_to_decimal_year (existing tests kept separately in this file;
# these add a couple of cases not already covered)
# ---------------------------------------------------------------------

def test_time_to_decimal_year_start_of_year():
    result = time.time_to_decimal_year(datetime(2021, 1, 1))
    assert result == pytest.approx(2021.0, abs=1e-6)


def test_time_to_decimal_year_end_of_year():
    result = time.time_to_decimal_year(datetime(2021, 12, 31, 23, 59, 59))
    assert result == pytest.approx(2022.0, abs=1e-4)