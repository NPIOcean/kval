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


# ---------------------------------------------------------------------
# Official WMM2025 test values (NOAA NCEI, WMM2025testvalues.pdf):
# (decimal year, height km, lat, lon, D in degrees, given to 0.01)
# 2025.0 = 2025-01-01 00:00; 2027.5 = 2027-07-02 12:00 (2027 has 365 days)

wmm2025_test_values = [
    (datetime(2025, 1, 1), 0, 80, 0, 1.28),
    (datetime(2025, 1, 1), 0, 0, 120, -0.16),
    (datetime(2025, 1, 1), 0, -80, -120, 68.78),   # lon 240 E
    (datetime(2025, 1, 1), 100, 80, 0, 0.85),
    (datetime(2025, 1, 1), 100, 0, 120, -0.15),
    (datetime(2025, 1, 1), 100, -80, -120, 68.21),
    (datetime(2027, 7, 2, 12), 0, 80, 0, 2.59),
    (datetime(2027, 7, 2, 12), 0, 0, 120, -0.24),
    (datetime(2027, 7, 2, 12), 0, -80, -120, 68.49),
    (datetime(2027, 7, 2, 12), 100, 80, 0, 2.16),
    (datetime(2027, 7, 2, 12), 100, 0, 120, -0.23),
    (datetime(2027, 7, 2, 12), 100, -80, -120, 67.93),
]


@pytest.mark.parametrize("date, height_km, lat, lon, expected",
                         wmm2025_test_values)
def test_declination_matches_official_wmm2025_test_values(
        date, height_km, lat, lon, expected):
    """Altitude is in metres in kval. It used to be passed on to pygeomag,
    which wants km, as it was: 100 m was taken as 100 km."""
    altitude_m = height_km * 1000.0
    point = get_declination_point(lat, lon, date, altitude=altitude_m)
    assert point == pytest.approx(expected, abs=0.005)
    for daily in (True, False):
        series = get_declination(lat, lon, [date], altitude=altitude_m,
                                 daily=daily)
        assert series[0] == pytest.approx(expected, abs=0.005)


def test_get_declination_daily_matches_exact():
    """Hourly times for 10 days: daily calculation + interpolation gives
    the same values as calculating every time, to far below 0.001 deg."""
    import pandas as pd
    times = pd.date_range("2021-03-01", periods=240, freq="1h").values
    daily = get_declination(78.9, 11.9, times)
    exact = get_declination(78.9, 11.9, times, daily=False)
    np.testing.assert_allclose(daily, exact, atol=1e-4)


def test_get_declination_numeric_days_since_1970():
    t = np.datetime64("2021-02-01T06:00")
    days = (t - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
    a = get_declination(45.0, -93.0, [t])
    b = get_declination(45.0, -93.0, np.array([days]))
    c = get_declination(45.0, -93.0, [np.int64(18659)])  # 2021-02-01
    assert a[0] == pytest.approx(b[0], abs=1e-9)
    assert np.isfinite(c[0])


def test_get_declination_empty():
    assert get_declination(45.0, -93.0, []).shape == (0,)


def test_get_declination_reads_each_model_file_once(monkeypatch):
    from kval.util import magdec
    magdec._geomag.cache_clear()
    calls = []
    real = magdec.GeoMag

    def counting_geomag(*args, **kwargs):
        calls.append(kwargs.get("coefficients_file"))
        return real(*args, **kwargs)

    monkeypatch.setattr(magdec, "GeoMag", counting_geomag)
    get_declination(45.0, -93.0, [datetime(2021, 2, d) for d in range(1, 20)],
                    daily=False)
    assert calls == ["wmm/WMM_2020.COF"]
    magdec._geomag.cache_clear()
