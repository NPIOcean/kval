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
        The type of filter to apply: 'mean' for running mean, 'median' for
        running median, 'sd' for standard deviation. Defaults to 'mean'.
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
    elif filter_type == 'sd':
        filtered_var = data_var.rolling(
            {dim: window_size}, center=True, min_periods=min_periods
            ).reduce(np.nanstd)
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


def rolling_sd(ds: xr.Dataset, var_name: str, dim: str,
               window_size: int,
               min_periods: Union[int, None] = None,
               nan_edges: bool = True,) -> xr.DataArray:
    """
    Compute a rolling standard deviation on a specified variable from an
    xarray Dataset along a given dimension.

    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset containing the variable to apply the rolling standard
        deviation.
    var_name : str
        The name of the variable in the dataset on which to calculate the
        rolling standard deviation.
    dim : str
        The dimension along which the rolling operation is performed.
    window_size : int
        The size of the rolling window.
    min_periods : int or None, optional
        The minimum number of observations in the window required to return
        a valid result. If set to `None` (default), the window will return
        NaN when there are insufficient data points (e.g., due to NaNs).
    nan_edges : bool, optional
        If `True`, the edge values of the output (half the window length)
        will be set to NaN, which is useful to avoid edge effects when
        rolling. Ignored if `min_periods` is set to a non-None value.
        Defaults to `True`.

    Returns:
    --------
    xarray.DataArray
        A DataArray containing the rolling standard deviation of the
        specified variable.
    """
    # Extract the variable to be filtered
    data_var = ds[var_name]

    def nan_std(var, **kwargs):
        '''
        Wrapper for np.nanstd that returns NaN for all-NaN slices and,
        crucially, does not return an annoying warning.
        '''

        var_sd = np.ones(var.shape[0])*np.nan

        for nn in np.arange(var.shape[0]):
            if not np.isnan(var[nn]).all():
                var_sd[nn] = np.nanstd(var[nn])

        return var_sd

    # Apply the rolling standard deviation
    sd_var = data_var.rolling(
        {dim: window_size}, center=True, min_periods=min_periods
    ).reduce(nan_std)

    # Handle edge values if required
    if nan_edges:
        halfwidth = int(np.ceil(window_size / 2))
        sd_var.isel({dim: slice(0, halfwidth)})[:] = np.nan
        sd_var.isel({dim: slice(-halfwidth, None)})[:] = np.nan

    return sd_var
