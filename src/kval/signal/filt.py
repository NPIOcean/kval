'''
SIGNAL.FILT.PY

Functions for filtering data.
'''
import warnings
import xarray as xr
import numpy as np


def _complete_window(data_var, dim, window_size):
    """
    True where the centred rolling window along `dim` lies entirely inside
    the record (i.e. not at the edges). Computed with the same rolling call,
    so it matches xarray's placement of odd and even windows exactly.
    """
    n = data_var.sizes[dim]
    ones = xr.DataArray(np.ones(n), dims=[dim])
    n_in_window = ones.rolling({dim: window_size}, center=True,
                               min_periods=1).sum()
    return n_in_window == window_size


def rolling(ds: xr.Dataset, variable: str, dim: str,
            window_size: int, filter_type: str = 'mean',
            min_periods: bool | int = None,
            nan_edges: bool = True) -> xr.Dataset:
    """
    Apply a running mean or median filter on a variable of an xarray Dataset
    along a specific dimension, with options to handle NaNs and edge values.

    Parameters:
    - ds: xarray.Dataset
        The dataset containing the variable to filter.
    - variable: str
        The name of the variable in the dataset to apply the filter on.
    - dim: str
        The dimension along which to apply the filter.
    - window_size: int
        The size of the rolling window.
    - filter_type: str, optional
        The type of filter to apply: 'mean' for running mean, 'median' for
        running median, 'sd' for standard deviation. Defaults to 'mean'.
    - min_periods: bool | int, optional
        Minimum number of observations in the window required to have a value.
        If an integer, it specifies the minimum number of observations in a
        rolling window.
        If `None` (default), all windows with a NaN will be set to Nan.
    - nan_edges: bool, optional
        Whether to set edge values to NaN: the points whose window extends
        past the start or end of the record (about half a window length).
        Applies whether or not `min_periods` is set (with `min_periods=None`
        the edge windows are incomplete and already NaN).
        Defaults to `True`.

    Returns:
    - ds_filt: xarray.Dataset
        The dataset with the filtered variable, where edge values may be NaN if `nan_edges` is `True`.
    """
    # Extract the variable to be filtered
    data_var = ds[variable]

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
        filtered_var = filtered_var.where(
            _complete_window(data_var, dim, window_size))

    # Replace the variable in the original dataset with the filtered result
    ds_filt = ds.copy()
    ds_filt[variable] = filtered_var

    return ds_filt


def rolling_sd(ds: xr.Dataset, variable: str, dim: str,
               window_size: int,
               min_periods: int | None = None,
               nan_edges: bool = True,) -> xr.DataArray:
    """
    Compute a rolling standard deviation on a specified variable from an
    xarray Dataset along a given dimension.

    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset containing the variable to apply the rolling standard
        deviation.
    variable : str
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
        If `True`, the edge values of the output (the points whose window
        extends past the start or end of the record, about half a window
        length) will be set to NaN, which is useful to avoid edge effects when
        rolling. Applies whether or not `min_periods` is set.
        Defaults to `True`.

    Returns:
    --------
    xarray.DataArray
        A DataArray containing the rolling standard deviation of the
        specified variable.
    """
    # Extract the variable to be filtered
    data_var = ds[variable]

    def nan_std(var, axis=-1, **kwargs):
        '''
        np.nanstd over the window axis, returning NaN for all-NaN windows
        without the "Degrees of freedom <= 0" warning. Works for variables
        with any number of dimensions.
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanstd(var, axis=axis)

    # Apply the rolling standard deviation
    sd_var = data_var.rolling(
        {dim: window_size}, center=True, min_periods=min_periods
    ).reduce(nan_std)

    # Handle edge values if required
    if nan_edges:
        sd_var = sd_var.where(_complete_window(data_var, dim, window_size))

    return sd_var
