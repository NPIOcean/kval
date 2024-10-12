import pytest
from datetime import datetime
import numpy as np
from kval.util import time


def test_datetime_input():
    """Test time_to_dec_year with a datetime object."""
    dt = datetime(2023, 8, 24)
    result = time.time_to_decimal_year(dt)
    expected = 2023 + (235 / 365.0)  # August 24th is the 235th day of 2023
    assert pytest.approx(result, 0.001) == expected


def test_string_input():
    """Test time_to_dec_year with a string date."""
    result = time.time_to_decimal_year('2023-08-24')
    expected = 2023 + (235 / 365.0)
    assert pytest.approx(result, 0.001) == expected


def test_numeric_input():
    """Test time_to_dec_year with a numeric value representing days since 1970-01-01."""
    result = time.time_to_decimal_year(19600)  # Should correspond to '2023-08-24'
    expected = 2023 + (235 / 365.0)
    assert pytest.approx(result, 0.001) == expected


def test_numpy_datetime64_input():
    """Test time_to_dec_year with a numpy.datetime64 object."""
    dt64 = np.datetime64('2023-08-24T00:00:00')
    result = time.time_to_decimal_year(dt64)
    expected = 2023 + (235 / 365.0)
    assert pytest.approx(result, 0.001) == expected


def test_invalid_string():
    """Test time_to_dec_year with an invalid string."""
    with pytest.raises(ValueError):
        time.time_to_decimal_year('invalid-date')


def test_invalid_type():
    """Test time_to_dec_year with an invalid type."""
    with pytest.raises(TypeError):
        time.time_to_decimal_year(['2023-08-24'])  # Passing a list instead of a string


# Test cases
def test_dt64_to_datenum():
    # Test case 1: Check conversion for a known date
    dt = np.datetime64('2022-08-08')
    expected = (dt - np.datetime64('1970-01-01')) / np.timedelta64(1, 'D')
    assert time.dt64_to_datenum(dt) == pytest.approx(expected)

    # Test case 2: Test for epoch date
    dt_epoch = np.datetime64('1970-01-01')
    assert time.dt64_to_datenum(dt_epoch) == 0.0

    # Test case 3: Check conversion for a date before epoch
    dt_before = np.datetime64('1960-01-01')
    expected_before = (dt_before - np.datetime64('1970-01-01')) / np.timedelta64(1, 'D')
    assert time.dt64_to_datenum(dt_before) == pytest.approx(expected_before)

    # Test case 4: Check conversion for a date after epoch
    dt_after = np.datetime64('2025-01-01')
    expected_after = (dt_after - np.datetime64('1970-01-01')) / np.timedelta64(1, 'D')
    assert time.dt64_to_datenum(dt_after) == pytest.approx(expected_after)

    # Test case 5: Test with a custom epoch
    custom_epoch = '2000-01-01'
    dt_custom = np.datetime64('2022-08-08')
    expected_custom = (dt_custom - np.datetime64(custom_epoch)) / np.timedelta64(1, 'D')
    assert time.dt64_to_datenum(dt_custom, epoch=custom_epoch) == pytest.approx(expected_custom)
