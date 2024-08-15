"""
EDIT.PY

Functions for editing (generalized) datasets.
"""

import numpy as np
import xarray as xr
from typing import Optional

def remove_points_profile(ds: xr.Dataset, varnm: str, TIME_index: int, remove_inds) -> xr.Dataset:
    """
    Remove specified points from a profile in the dataset by setting them to NaN.
    
    Parameters:
    - ds: xarray.Dataset
      The dataset containing the variable to modify.
    - varnm: str
      The name of the variable to modify.
    - TIME_index: int
      The index along the TIME dimension to modify.
    - remove_inds: list or array-like
      Indices of points to remove (set to NaN) within the specified TIME profile.
    
    Returns:
    - ds: xarray.Dataset
      The dataset with specified points removed (set to NaN).
    """

    # Convert remove_inds to a list if it's not already
    remove_inds = np.asarray(remove_inds)
    
    # Create a boolean array for removal
    remove_bool = np.zeros(len(ds[varnm].isel(TIME=TIME_index)), dtype=bool)
    remove_bool[remove_inds] = True

    # Use the `where` method to set the selected points to NaN
    ds[varnm].isel(TIME=TIME_index).values[:] = np.where(remove_bool, 
                                                         np.nan, 
                                                         ds[varnm].isel(TIME=TIME_index).values)

    return ds

def offset(ds: xr.Dataset, variable: str, offset: float) -> xr.Dataset:
    """
    Apply a fixed offset to a specified variable in an xarray Dataset.

    This function modifies the values of the specified variable by adding a fixed 
    offset to them. The `valid_min` and `valid_max` attributes are updated to reflect
    the new range of values after applying the offset.

    Parameters
    ----------
    ds : xr.Dataset
        The input xarray Dataset.
    variable : str
        The name of the variable within the Dataset to which the offset will be applied.
    offset : float
        The fixed offset value to add to the variable.

    Returns
    -------
    xr.Dataset
        A new xarray Dataset with the offset applied to the specified variable. The
        `valid_min` and `valid_max` attributes are updated accordingly.

    Examples
    --------
    # Apply an offset of 5 to the 'TEMP' variable
    ds_offset = offset(ds, 'TEMP', offset=5)
    """

    ds_new = ds.copy()

    if variable not in ds_new:
        raise ValueError(f"Variable '{variable}' not found in the Dataset.")

    # Apply the offset
    ds_new[variable] = ds_new[variable] + offset
    ds_new[variable].attrs = ds[variable].attrs

    # Update the valid_min and valid_max attributes if they exist
    if 'valid_min' in ds_new[variable].attrs:
        ds_new[variable].attrs['valid_min'] += offset
    
    if 'valid_max' in ds_new[variable].attrs:
        ds_new[variable].attrs['valid_max'] += offset

    return ds_new

def threshold(ds: xr.Dataset, variable: str, 
              max_val: Optional[float] = None, 
              min_val: Optional[float] = None) -> xr.Dataset:
    """
    Apply a threshold to a specified variable in an xarray Dataset, setting 
    values outside the specified range (min_val, max_val) to NaN.

    Also modifies the valid_min and valid_max variable attributes.

    Parameters
    ----------
    ds : xr.Dataset
        The input xarray Dataset.
    variable : str
        The name of the variable within the Dataset to be thresholded.
    max_val : Optional[float], default=None
        The maximum allowed value for the variable. Values greater than 
        this will be set to NaN.
        If None, no upper threshold is applied.
    min_val : Optional[float], default=None
        The minimum allowed value for the variable. Values less than 
        this will be set to NaN.
        If None, no lower threshold is applied.

    Returns
    -------
    xr.Dataset
        A new xarray Dataset with the thresholded variable. The `valid_min` 
        and `valid_max` attributes are updated accordingly.

    Examples
    --------
    # Reject temperatures below -1 and above 3
    ds_thresholded = threshold(ds, 'TEMP', min_val=-1, max_val=3)
    """

    ds_new = ds.copy()

    if max_val is not None:
        ds_new[variable] = ds_new[variable].where(ds_new[variable] <= max_val)
        ds_new[variable].attrs['valid_max'] = max_val

    if min_val is not None:
        ds_new[variable] = ds_new[variable].where(ds_new[variable] >= min_val)
        ds_new[variable].attrs['valid_min'] = min_val

        if max_val is not None and max_val <= min_val:
            raise ValueError(f'Threshold editing: max_val ({max_val}) must be greater than min_val ({min_val}).')

    return ds_new
