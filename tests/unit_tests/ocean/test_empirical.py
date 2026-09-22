"""
Tests for kval.ocean.empirical.

Reference values for windstress were independently verified against the
published Large and Pond (1981) coefficients (Cd=1.2e-3 for 4<=U<=11 m/s,
Cd=(0.49+0.065*U)*1e-3 for 11<U<=25 m/s) before being pinned here.
Reference value for coriolis_parameter uses Earth's angular rotation rate
Omega = 7.292115e-5 rad/s (sidereal day).
"""

import numpy as np
import pytest

from kval.ocean import empirical


@pytest.mark.parametrize(
    "u10,expected",
    [
        (5, 0.0360),
        (10, 0.1440),
        (15, 0.39555),
        (20, 0.8592),
        (25, 1.58625),
    ],
)
def test_windstress_matches_reference_values(u10, expected):
    assert empirical.windstress(u10) == pytest.approx(expected, rel=1e-4)


def test_windstress_accepts_array_input():
    result = empirical.windstress(np.array([5, 10, 15]))
    expected = np.array([0.0360, 0.1440, 0.39555])
    np.testing.assert_allclose(result, expected, rtol=1e-4)


def test_windstress_uses_custom_rho_air():
    default_result = empirical.windstress(10)
    custom_result = empirical.windstress(10, rho_air=1.225)
    assert custom_result == pytest.approx(default_result * 1.225 / 1.2, rel=1e-6)


def test_windstress_warns_below_validated_range():
    with pytest.warns(UserWarning, match="4-25 m/s"):
        empirical.windstress(2)


def test_windstress_warns_above_validated_range():
    with pytest.warns(UserWarning, match="4-25 m/s"):
        empirical.windstress(30)


def test_windstress_does_not_warn_within_validated_range():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        empirical.windstress(10)  # should not raise/warn


def test_coriolis_parameter_zero_at_equator():
    assert empirical.coriolis_parameter(0) == pytest.approx(0.0, abs=1e-12)


def test_coriolis_parameter_maximal_at_pole():
    expected = 2 * 7.292115e-5
    assert empirical.coriolis_parameter(90) == pytest.approx(expected, rel=1e-6)


def test_coriolis_parameter_negative_in_southern_hemisphere():
    north = empirical.coriolis_parameter(60)
    south = empirical.coriolis_parameter(-60)
    assert south == pytest.approx(-north)


def test_coriolis_parameter_accepts_array_input():
    result = empirical.coriolis_parameter(np.array([0, 45, 90]))
    assert result[0] == pytest.approx(0.0, abs=1e-12)
    assert result[2] == pytest.approx(2 * 7.292115e-5, rel=1e-6)


def test_coriolis_parameter_raises_for_invalid_latitude():
    with pytest.raises(ValueError, match="between -90 and 90"):
        empirical.coriolis_parameter(100)