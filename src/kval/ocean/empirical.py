'''
EMPIRICAL.PY

Collection of empirical/standard formulas used in oceanography.
'''

import numpy as np

# Earth's angular rotation rate (rad/s), based on the sidereal day
# (23h 56m 4.1s). Source: IERS / standard geodetic value, ~7.292115e-5 rad/s.
EARTH_ANGULAR_VELOCITY = 7.292115e-5


def coriolis_parameter(latitude: float | np.ndarray) -> float | np.ndarray:
    '''
    Calculate the Coriolis parameter (Coriolis frequency) f = 2 * Omega *
    sin(latitude), where Omega is Earth's angular rotation rate.

    f is the frequency of inertial oscillations at a given latitude, and
    appears throughout geophysical fluid dynamics (e.g. Ekman transport,
    Rossby radius of deformation, geostrophic balance).

    Args:
        latitude: Latitude in decimal degrees (-90 to 90). Positive in the
            Northern Hemisphere, negative in the Southern Hemisphere.

    Returns:
        Coriolis parameter f, in rad/s. Positive in the Northern
        Hemisphere, negative in the Southern Hemisphere, zero at the
        equator.

    Examples
    --------
    >>> coriolis_parameter(70)  # Typical high-latitude value
    1.3690...e-04
    '''
    latitude = np.asarray(latitude, dtype=float)
    if np.any(np.abs(latitude) > 90):
        raise ValueError('latitude must be between -90 and 90 degrees.')
    return 2 * EARTH_ANGULAR_VELOCITY * np.sin(np.deg2rad(latitude))


def _drag_coefficient_large_pond(u10: float | np.ndarray) -> np.ndarray:
    '''
    Drag coefficient Cd as a function of 10 m wind speed, following the
    bulk formula of Large and Pond (1981), valid for 4 <= U10 <= 25 m/s:

        Cd = 1.2e-3                    for  4 <= U10 <= 11 m/s
        Cd = (0.49 + 0.065 * U10) * 1e-3   for 11 <  U10 <= 25 m/s

    Reference: Large, W. G., and S. Pond (1981), Open ocean momentum flux
    measurements in moderate to strong winds, J. Phys. Oceanogr., 11,
    324-336.

    Outside the range in which the formula was validated (< 4 or > 25 m/s),
    the nearest end-point value is used with a warning, rather than
    extrapolating silently.
    '''
    u10 = np.asarray(u10, dtype=float)

    below_range = u10 < 4
    above_range = u10 > 25
    if np.any(below_range) or np.any(above_range):
        import warnings
        warnings.warn(
            'windstress: wind speed outside the 4-25 m/s range Large and '
            'Pond (1981) was validated for -- clamping Cd to the nearest '
            'end-point value rather than extrapolating.', UserWarning
        )

    u10_clamped = np.clip(u10, 4, 25)
    cd = np.where(
        u10_clamped <= 11,
        1.2e-3,
        (0.49 + 0.065 * u10_clamped) * 1e-3,
    )
    return cd


def windstress(
    u10: float | np.ndarray, rho_air: float = 1.2
) -> float | np.ndarray:
    '''
    Calculate the magnitude of wind stress from 10 m wind speed, using the
    bulk formula of Large and Pond (1981):

        tau = rho_air * Cd(U10) * U10**2

    Reference: Large, W. G., and S. Pond (1981), Open ocean momentum flux
    measurements in moderate to strong winds, J. Phys. Oceanogr., 11,
    324-336.

    Note: this assumes u10 is already the wind speed at 10 m height
    (the standard reference height this formula is defined for); no
    height adjustment (e.g. log-law extrapolation from another
    measurement height) is applied.

    Args:
        u10: 10 m wind speed [m/s].
        rho_air: Air density [kg m-3]. Defaults to 1.2 (typical sea-level
            value).

    Returns:
        Wind stress magnitude [Pa].

    Examples
    --------
    >>> windstress(10)
    0.144
    '''
    cd = _drag_coefficient_large_pond(u10)
    return rho_air * cd * np.asarray(u10, dtype=float) ** 2