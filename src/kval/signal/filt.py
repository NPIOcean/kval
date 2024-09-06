'''
UTIL.FILT.PY

Functions for filtering data.
'''
import xarray as xr
import numpy as np
from typing import Union

def rolling(ds: xr.Dataset, var_name: str, dim: str,
            window_size: int, filter_type: str = 'mean',
            min_periods: Union[bool, int] = None,
            nan_edges: bool = True) -> xr.Dataset:
    """
    Apply a running mean or median filter on a variable of an xarray Dataset
    along a specific dimension, with options to handle NaNs and edge values.

    Parameters:
    - ds: xarray.Dataset
        The dataset containing the variable to filter.
    - var_name: str
        The name of the variable in the dataset to apply the filter on.
    - dim: str
        The dimension along which to apply the filter.
    - window_size: int
        The size of the rolling window.
    - filter_type: str, optional
        The type of filter to apply: 'mean' for running mean or 'median' for
        running median. Defaults to 'mean'.
    - min_periods: Union[bool, int], optional
        Minimum number of observations in the window required to have a value.
        If an integer, it specifies the minimum number of observations in a
        rolling window.
        If `None` (default), all windows with a NaN will be set to Nan.
    - nan_edges: bool, optional
        Whether to set edge values (half of the window length) to NaN.
        (Redundant if min_periods is set to `None`)
        Defaults to `True`.

    Returns:
    - ds_filt: xarray.Dataset
        The dataset with the filtered variable, where edge values may be NaN if `nan_edges` is `True`.
    """
    # Extract the variable to be filtered
    data_var = ds[var_name]

    # Apply the appropriate rolling filter
    if filter_type == 'mean':
        filtered_var = data_var.rolling(
            {dim: window_size}, center=True, min_periods=min_periods
            ).reduce(np.nanmean)
    elif filter_type == 'median':
        filtered_var = data_var.rolling(
            {dim: window_size}, center=True, min_periods=min_periods
            ).reduce(np.nanmedian)
    else:
        raise ValueError("Invalid filter_type. Choose 'mean' or 'median'.")

    # Handle edge values
    if nan_edges:
        halfwidth = int(np.ceil(window_size / 2))
        filtered_var.isel({dim: slice(0, halfwidth)})[:] = np.nan
        filtered_var.isel({dim: slice(-halfwidth, None)})[:] = np.nan

    # Replace the variable in the original dataset with the filtered result
    ds_filt = ds.copy()
    ds_filt[var_name] = filtered_var

    return ds_filt
