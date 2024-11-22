import xarray as xr
import pandas as pd
from typing import Union
import numpy as np



def moored_around_ctd(ds_moor: xr.Dataset,
                      ds_ctd: xr.Dataset,
                      hours: int = 1) -> xr.Dataset:
    """
    Extract values from a moored dataset within a specified time window around
    a CTD profile, and revert the 'TIME' coordinate to its original format if
    necessary.

    Args:
        ds_moor (xr.Dataset):
            The moored dataset, which contains a time dimension.
        ds_ctd (xr.Dataset):
            The CTD dataset, which contains a single timestamp in the TIME
            variable.
        hours (int, optional):
            The number of hours before and after the CTD time to extract from
            the moored dataset. Defaults to 1 hour.

    Returns:
        xr.Dataset:
            A subset of the moored dataset, limited to the time window around
            the CTD timestamp, with 'TIME' converted back to its original
            format (either datetime64 or fractional days if originally in that
            format).

    Raises:
        ValueError: If no valid time stamps are found within the specified
        range.
    """

    # Check if the 'TIME' coordinate is not in datetime64 format (assumed to be
    # fractional days)
    original_time_dtype = ds_moor['TIME'].dtype
    ds_moor_cp = ds_moor.copy()

    if np.issubdtype(original_time_dtype, np.floating):
        # If it's fractional days (float), convert it to datetime64 for slicing
        # Convert fractional days to datetime64[ns]
        time_attrs = ds_moor.TIME.attrs.copy()
        epoch_str = time_attrs['units'].lower().replace('days since ', '')
        epoch = np.datetime64(epoch_str)
        ds_moor_cp['TIME'] = (
            epoch + pd.to_timedelta(ds_moor_cp['TIME'].values, 'D'))

    # Get the time slice around the CTD time using the helper function
    time_slice_prof = slice_around_time_stamp(ds_ctd, hours=hours)

    # Select the moored dataset around the CTD time
    ds_moor_sel = ds_moor_cp.sel(TIME=time_slice_prof)

    # Check if any time stamps exist within the range
    if len(ds_moor_sel['TIME']) == 0:
        raise ValueError(
            f"No valid time stamps found within the {hours}-hour window")

    # Convert 'TIME' back to the original fractional days format if it was
    # originally in that format
    if np.issubdtype(original_time_dtype, np.floating):
        # Convert datetime64[ns] back to fractional days since the epoch
        delta = ds_moor_sel['TIME'] - epoch
        ds_moor_sel['TIME'] = delta / np.timedelta64(1, 'D')  # Convert to days
        ds_moor_sel['TIME'].attrs = time_attrs

    return ds_moor_sel


def slice_around_time_stamp(ds: xr.Dataset, hours: int = 1) -> slice:
    """
    Creates a time slice around a single timestamp in the dataset.

    This function extracts a time slice from an `xarray.Dataset` around a
    single timestamp (found in the `TIME` variable). The slice spans a
    specified number of hours before and after the timestamp.

    Args:
        ds (xr.Dataset):
            The input dataset, which should contain a `TIME` variable with a
            single timestamp. hours (int, optional): The number of hours before
            and after the `TIME` timestamp to include in the slice.
            Defaults to 1 hour.

    Returns:
        slice:
            A time slice spanning from `hours` before to `hours` after the
            `TIME` timestamp.

    Raises:
        ValueError: If the dataset does not contain a `TIME` variable, or if
        `TIME` does not have exactly one timestamp.
    """
    # Check if 'TIME' exists in the dataset
    if 'TIME' not in ds:
        raise ValueError("The dataset does not contain a 'TIME' variable.")

    # Ensure that TIME contains only one timestamp
    if ds['TIME'].size != 1:
        raise ValueError("'TIME' should contain exactly one timestamp.")

    # Convert the TIME variable to a pandas Timestamp
    time_stamp = pd.Timestamp(ds['TIME'].values.item())  # .item() to extract the scalar value

    # Define the start and end times for the slice
    start_time = time_stamp - pd.Timedelta(hours=hours)
    end_time = time_stamp + pd.Timedelta(hours=hours)

    # Create and return the time slice
    return slice(start_time, end_time)


def closest_profile_point(ds_ctd, ds):

    pres_target = ds.PRES.mean()
    ds_ctd_closest = ds_ctd.sel(PRES=pres_target, method = 'nearest')

