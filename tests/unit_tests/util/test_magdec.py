import pytest
import numpy as np
from datetime import datetime
from kval.util.magdec import (
    get_declination, get_declination_point, choose_wmm_model)


# Known values for testing
known_values = [
    # (latitude, longitude, time, expected_declination,)
    (-70, -130, datetime(2024, 8, 24), 56.037),
    (80.0, 30.2, datetime(2020, 8, 24), 21.792),
    (20, -80.0, datetime(2022, 1, 29), -6.337),
    (0, 0, datetime(2026, 4, 16), -3.86),

]
@pytest.mark.parametrize("latitude, longitude, time, expected_declination",
                        known_values)




def test_get_declination_point(latitude, longitude, time, expected_declination,):
    """Test get_declination_point with known values."""
    declination = get_declination_point(latitude, longitude, time)

    assert isinstance(declination, float)
    assert -180 <= declination <= 180
    assert pytest.approx(declination, rel=1e-3) == expected_declination


# Test for get_declination function with various date formats
def test_get_declination():
    dates = [
        datetime(2021, 2, 1),
        np.datetime64('2021-02-01'),
        '2021-02-01T00:00:00',
    ]
    result = get_declination(45.0, -93.0, dates)

    # Validate the result
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(dates),)
    # Magnetic declination should be in this range
    assert all(-180 <= x <= 180 for x in result)

    # Check that all declination values are the same
    assert np.all(result == result[0]), (
        "Declination values for different date formats should be the same")


# Test for choose_wmm_model function
def test_choose_wmm_model():
    assert choose_wmm_model(2010) == 'wmm/WMM_2010.COF'
    assert choose_wmm_model(2015) == 'wmm/WMM_2015.COF'
    assert choose_wmm_model(2020) == 'wmm/WMM_2020.COF'
    assert choose_wmm_model(2025) == 'wmm/WMM_2025.COF'

    with pytest.raises(ValueError):
            choose_wmm_model(2031)



# Test for handling invalid date formats
def test_get_declination_invalid_date():
    with pytest.raises(TypeError):
        get_declination(45.0, -93.0, [object()])


# Test get_declination with different models
def test_get_declination_varied_model():
    dates = [datetime(2020, 1, 1), datetime(2022, 1, 1)]

    # Run tests with different models
    result_auto = get_declination(45.0, -93.0, dates, model='auto')
    result_2010 = get_declination(45.0, -93.0, dates, model='2010')
    result_2015 = get_declination(45.0, -93.0, dates, model='2015')
    result_2020 = get_declination(45.0, -93.0, dates, model='2020')

    assert isinstance(result_auto, np.ndarray)
    assert isinstance(result_2010, np.ndarray)
    assert isinstance(result_2015, np.ndarray)
    assert isinstance(result_2020, np.ndarray)

    # For dates in 2020 and 2022, the 'auto' model should use the 2020 model
    assert np.all(result_auto == result_2020), (
        "Results for auto and 2020 model should be the same")

    # Check that 'auto' results should not match models that are out of date
    # range e.g., results for the 2010 and 2015 models
    assert not np.all(result_auto == result_2010), (
        "Results for auto and 2010 model should differ")
    assert not np.all(result_auto == result_2015), (
        "Results for auto and 2015 model should differ")
