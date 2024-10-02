'''
KVAL.SIGNAL.INTERP

Functions for interpolation
'''


import xarray as xr

def interpolate_nans(ds, dim, method='linear', limit=None):
    """
    Interpolate over NaN values along a given dimension in the dataset.

    Straight wrapper for the xr.interpolate_na function.

    Parameters:
    ds (xr.Dataset or xr.DataArray): The xarray Dataset or DataArray to interpolate.
    method (str): The interpolation method (default is 'linear').
                  Other options include 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', etc.
    limit (int, optional): Maximum number of consecutive NaNs to fill.
                           If None, all NaNs will be interpolated.

    Returns:
    xr.Dataset or xr.DataArray: Interpolated Dataset or DataArray.
    """
    return ds.interpolate_na(dim=dim, method=method, limit=limit)
