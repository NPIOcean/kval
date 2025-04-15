"""
Functions for manipulating vector datasets.

Belongs here:

- Principal axis analysis
- Rotate UV by angle
- Rotary spectra? (may possible belong in a spectral module)

"""

from typing import Tuple, Union, Optional
import numpy as np


def rotate_uv(
    u: np.ndarray, v: np.ndarray, angle: float, in_degrees: bool = False,
    decimals: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotates the vector (u, v) CLOCKWISE by the specified angle.

    Parameters:
    ----------
    u : np.ndarray
        A 1D array representing the x-components of the vector.
    v : np.ndarray
        A 1D array representing the y-components of the vector.
    angle : float
        The angle by which to rotate the vector. If `in_degrees` is True,
        the angle should be provided in degrees. Otherwise, it is assumed
        to be in radians.
    in_degrees : bool, optional
        If True, the `angle` is interpreted as being in degrees. If False
        (default), the `angle` is in radians.
    decimals:
        To avoid numeriucal noise, round to this level of precision.

    Returns:
    -------
    Tuple[np.ndarray, np.ndarray]
        - The x-components of the rotated vector.
        - The y-components of the rotated vector.
    """

    # Convert angle to radians if it is in degrees
    angle_rad = np.radians(angle) if in_degrees else angle

    # Perform the rotation
    uvc_rot = (u + 1j * v) * np.exp(-1j * angle_rad)

    # Extract the real and imaginary parts
    # Round away numerical noise)
    u_rot = np.round(uvc_rot.real, decimals)
    v_rot = np.round(uvc_rot.imag, decimals)

    return u_rot, v_rot


def principal_angle(
    u: np.ndarray,
    v: np.ndarray,
    in_degrees: bool = False,
    return_std: bool = False,
) -> Tuple[Union[float, float], Optional[float], Optional[float]]:
    """
    Computes the principal angle between two vectors `u` and `v` in the range
    [-π/2, π/2], where the squared distances to `u` and `v` are maximized. The
    angle is calculated using the method described by Emery and Thompson
    (p327). Optionally, the angle can be returned in degrees, and the function
    can also return the standard deviations along the major and minor axes.

    Parameters:
    ----------
    u : np.ndarray
        A 1D array representing the first vector.
    v : np.ndarray
        A 1D array representing the second vector.
    in_degrees : bool, optional
        If True, the principal angle is returned in degrees. Default is False
        (radians).
    return_std : bool, optional
        If True, also return the standard deviation along the major and minor
        axes. Default is False (only return the principal angle).

    Returns:
    -------
    Tuple[Union[float, float], Optional[float], Optional[float]]
        - The principal angle, in radians by default or degrees if `in_degrees`
          is True. Provided in the [-pi/2, pi/2] or [-180, 180] range.
        - Standard deviation along the major axis if `return_std` is True,
          otherwise None.
        - Standard deviation along the minor axis if `return_std` is True,
          otherwise None.

    Notes:
    -----
    If the mean of either `u` or `v` is non-zero, it will be removed before
    computing the principal angle.
    """
    if np.nanmean(u) > 1e-7 or np.nanmean(v) > 1e-7:
        u -= np.nanmean(u)
        v -= np.nanmean(v)

    # Compute the principal angle using the given formula
    principal_angle_rad = 0.5 * np.arctan2(
        2 * np.nanmean(u * v), np.nanmean(u**2) - np.nanmean(v**2)
    )

    # Ensure the angle is within (-π/2, π/2]
    if principal_angle_rad < -np.pi / 2:
        principal_angle_rad += np.pi
    if principal_angle_rad >= np.pi / 2:
        principal_angle_rad -= np.pi

    # Optionally convert the angle to degrees
    principal_angle = (
        np.degrees(principal_angle_rad) if in_degrees else principal_angle_rad
    )

    if return_std:
        # Rotate the vector clockwiseby the principal angle
        uvc_rotated = (u + 1j * v) * np.exp(-1j * principal_angle_rad)

        # Compute standard deviations along the major and minor axes
        major_axis_std = np.nanstd(uvc_rotated.real)
        minor_axis_std = np.nanstd(uvc_rotated.imag)

        return principal_angle, major_axis_std, minor_axis_std
    else:
        return principal_angle
