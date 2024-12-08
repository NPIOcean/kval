import xarray as xr
import pandas as pd
from typing import Union
import numpy as np

"""
This module provides functions to extract and process moored dataset values
around a CTD profile, including time-based subsetting, nearest point selection,
and profile comparison.

TO DO:
- X Find time slice around CTD
- X Find moored subset around CTD
- X Grab nearest profile point (based on pressure)
- X Grab nearest moored point (based on time)
- Get numerical difference moored-profile.
- Plot profile with points (one or several moorings)
- Plot mooring time series with CTD (p/m some dbar if specified)
- Wrapper to apply difference directly? (Maybe not?)
- Pytests for code testing
"""


def moored_around_ctd(
    ds_moor: xr.Dataset, ds_ctd: xr.Dataset, hours: int = 1
) -> xr.Dataset:
    """
    Extract a subset of the moored dataset within a specified time window
    around a CTD profile.

    Args:
        ds_moor (xr.Dataset):
            Moored dataset with a 'TIME' coordinate.
        ds_ctd (xr.Dataset):
            CTD dataset with a single timestamp in its 'TIME' variable.
        hours (int, optional):
            Time window (in hours) before and after the CTD time for
            subsetting.
            Defaults to 1 hour.

    Returns:
        xr.Dataset:
            Subset of moored dataset restricted to the time window, with time
            converted back to its original format.

    Raises:
        ValueError: If no valid timestamps are found within the specified time
        indow.
    """
    original_time_dtype = ds_moor["TIME"].dtype
    ds_moor_cp = ds_moor.copy()

    # Handle conversion from fractional days to datetime64 if needed
    if np.issubdtype(original_time_dtype, np.floating):
        time_attrs = ds_moor.TIME.attrs.copy()
        epoch_str = time_attrs["units"].lower().replace("days since ", "")
        epoch = np.datetime64(epoch_str)
        ds_moor_cp["TIME"] = epoch + pd.to_timedelta(ds_moor_cp["TIME"].values,
                                                     "D")

    # Create a time slice around the CTD time
    time_slice_prof = slice_around_time_stamp(ds_ctd, hours=hours)
    ds_moor_sel = ds_moor_cp.sel(TIME=time_slice_prof)

    if len(ds_moor_sel["TIME"]) == 0:
        raise ValueError(
            f"No valid timestamps found within the {hours}-hour window")

    # Convert 'TIME' back to original fractional days format if necessary
    if np.issubdtype(original_time_dtype, np.floating):
        delta = ds_moor_sel["TIME"] - epoch
        ds_moor_sel["TIME"] = delta / np.timedelta64(1, "D")
        ds_moor_sel["TIME"].attrs = time_attrs

    return ds_moor_sel


def moored_nearest_ctd(ds_moor: xr.Dataset, ds_ctd: xr.Dataset) -> xr.Dataset:
    """
    Select the nearest mooring data point to the time of a CTD profile.

    Args:
        ds_moor (xr.Dataset):
            Moored dataset with a 'TIME' coordinate.
        ds_ctd (xr.Dataset):
            CTD dataset with a single timestamp in its 'TIME' variable.

    Returns:
        xr.Dataset:
            Nearest moored dataset entry to the CTD time.
    """
    original_time_dtype = ds_moor["TIME"].dtype
    ds_moor_cp = ds_moor.copy()

    if np.issubdtype(original_time_dtype, np.floating):
        time_attrs = ds_moor.TIME.attrs.copy()
        epoch_str = time_attrs["units"].lower().replace("days since ", "")
        epoch = np.datetime64(epoch_str)
        ds_moor_cp["TIME"] = epoch + pd.to_timedelta(ds_moor_cp["TIME"].values,
                                                     "D")

    # Get the exact CTD timestamp
    ctd_time_stamp = single_time_stamp(ds_ctd)

    # Select the nearest moored data point
    ds_moor_sel = ds_moor_cp.sel(TIME=ctd_time_stamp, method="nearest")

    # Revert 'TIME' to fractional days if it was originally in that format
    if np.issubdtype(original_time_dtype, np.floating):
        delta = ds_moor_sel["TIME"] - epoch
        ds_moor_sel["TIME"] = delta / np.timedelta64(1, "D")
        ds_moor_sel["TIME"].attrs = time_attrs

    return ds_moor_sel


def slice_around_time_stamp(ds: xr.Dataset, hours: int = 1) -> slice:
    """
    Create a time slice spanning a specified number of hours around a single
    timestamp.

    Args:
        ds (xr.Dataset):
            Dataset with a 'TIME' variable containing one timestamp.
        hours (int, optional):
            Number of hours before and after the timestamp for the slice.
            Defaults to 1 hour.

    Returns:
        slice:
            Time slice from `hours` before to `hours` after the timestamp.

    Raises:
        ValueError: If the 'TIME' variable is missing or does not contain
        exactly one timestamp.
    """
    if "TIME" not in ds:
        raise ValueError("The dataset does not contain a 'TIME' variable.")
    if ds["TIME"].size != 1:
        raise ValueError("'TIME' must contain exactly one timestamp.")

    time_stamp = pd.Timestamp(ds["TIME"].values.item())
    start_time = time_stamp - pd.Timedelta(hours=hours)
    end_time = time_stamp + pd.Timedelta(hours=hours)

    return slice(start_time, end_time)


def single_time_stamp(ds: xr.Dataset) -> pd.Timestamp:
    """
    Extract the single timestamp from a dataset.

    Args:
        ds (xr.Dataset):
            Dataset with a 'TIME' variable containing one timestamp.

    Returns:
        pd.Timestamp:
            Extracted timestamp.

    Raises:
        ValueError: If the 'TIME' variable is missing or does not contain
        exactly one timestamp.
    """
    if "TIME" not in ds:
        raise ValueError("The dataset does not contain a 'TIME' variable.")
    if ds["TIME"].size != 1:
        raise ValueError("'TIME' must contain exactly one timestamp.")

    return pd.Timestamp(ds["TIME"].values.item())


def closest_profile_point(ds_ctd: xr.Dataset, ds: xr.Dataset) -> xr.Dataset:
    """
    Select the profile point in the CTD dataset closest in pressure to the
    moored dataset.

    Args:
        ds_ctd (xr.Dataset):
            CTD dataset containing pressure ('PRES') variable.
        ds (xr.Dataset):
            Moored dataset with 'PRES' variable. If multiple times exist, the
            mean pressure is used.

    Returns:
        xr.Dataset:
            CTD profile at the closest pressure point to the moored dataset.
    """
    if ds.TIME.ndim == 0:
        pres_target = ds.PRES.item()
    else:
        pres_target = ds.PRES.mean(dim="TIME").item()

    ds_ctd_closest = ds_ctd.sel(PRES=pres_target, method="nearest")

    return ds_ctd_closest


def diff_numerical(ds_moor, ds_ctd, varnm, varnm_ctd=None):
    '''


    (CTD minus moored)

    '''

    if varnm_ctd==None:
        varnm_ctd = varnm

    # (check that we have these variables)

    ds_sel_moor = moored_nearest_ctd(ds_moor, ds_ctd)
    ds_sel_ctd = closest_profile_point(ds_ctd, ds_moor)

    print(ds_sel_ctd[varnm_ctd].item() )
    print( ds_sel_moor[varnm].item())
    diff = ds_sel_ctd[varnm_ctd].item() - ds_sel_moor[varnm].item()

    return diff

