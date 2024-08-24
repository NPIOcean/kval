import pytest
from datetime import datetime
import numpy as np
from kval.util.time import time_to_decimal_year


def test_datetime_input():
    """Test time_to_dec_year with a datetime object."""
    dt = datetime(2023, 8, 24)
    result = time_to_decimal_year(dt)
    expected = 2023 + (235 / 365.0)  # August 24th is the 235th day of 2023
    assert pytest.approx(result, 0.001) == expected


def test_string_input():
    """Test time_to_dec_year with a string date."""
    result = time_to_decimal_year('2023-08-24')
    expected = 2023 + (235 / 365.0)
    assert pytest.approx(result, 0.001) == expected


def test_numeric_input():
    """Test time_to_dec_year with a numeric value representing days since 1970-01-01."""
    result = time_to_decimal_year(19600)  # Should correspond to '2023-08-24'
    expected = 2023 + (235 / 365.0)
    assert pytest.approx(result, 0.001) == expected


def test_numpy_datetime64_input():
    """Test time_to_dec_year with a numpy.datetime64 object."""
    dt64 = np.datetime64('2023-08-24T00:00:00')
    result = time_to_decimal_year(dt64)
    expected = 2023 + (235 / 365.0)
    assert pytest.approx(result, 0.001) == expected


def test_invalid_string():
    """Test time_to_dec_year with an invalid string."""
    with pytest.raises(ValueError):
        time_to_decimal_year('invalid-date')


def test_invalid_type():
    """Test time_to_dec_year with an invalid type."""
    with pytest.raises(TypeError):
        time_to_decimal_year(['2023-08-24'])  # Passing a list instead of a string