'''
KVAL.SIGNAL.DESPIKE

Routines for basic outlier editing.
'''


import numpy as np
import xarray as xr
from kval.signal import filt
from typing import Union, Tuple
import matplotlib.pyplot as plt


def despike_rolling(
    ds: xr.Dataset,
    var_name: str,
    window_size: int,
    n_std: float,
    dim: str,
    filter_type: str = 'median',
    min_periods: Union[int, None] = None,
    return_ds: bool = True,
    return_index: bool = False,
    plot: bool = False,
    verbose: bool = False,

) -> Union[xr.Dataset,
           xr.DataArray,
           Tuple[xr.Dataset, xr.DataArray],
           Tuple[xr.DataArray, xr.DataArray]]:
    """
    Despike a variable in a dataset by identifying and removing outliers
    based on a rolling mean/median and standard deviation.

    Outliers are data points where the variable deviates from the rolling
    mean/median by a number of standard deviations. Both the mean/median
    and standard deviation are calculated within a rolling window centered
    on each data point.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the variable to despike.
    var_name : str
        The name of the variable to despike.
    window_size : int
        The size of the rolling window for calculating the mean/median
        and standard deviation.
    n_std : float
        The number of standard deviations used as the threshold to identify
        outliers.
    dim : str
        The dimension along which to calculate the rolling statistics.
    filter_type : str, optional
        The type of filter to apply ('mean' or 'median'). Default is 'mean'.
    min_periods : int or None, optional
        The minimum number of observations in the window required to return
        a valid result. Default is None.
    return_ds : bool, optional
        If True, returns the updated dataset with the despiked variable.
        If False, returns only the despiked variable. Default is True.
    return_index : bool, optional
        If True, returns a mask indicating which points were flagged as
        outliers. Default is False.
    plot : bool, optional
        If True, plots the original data and the despiking results.
        Default is False.
    verbise : bool, optional
        If True, print some basic info about the result of the despiking.
        Default is False.
    Returns
    -------

        - If `return_ds` is True and `return_index` is False: returns the
          updated dataset with the despiked variable.
        - If `return_ds` is False and `return_index` is False: returns the
          despiked variable as a DataArray.
        - If `return_ds` is True and `return_index` is True: returns a tuple
          of the updated dataset and a mask of outliers.
        - If `return_ds` is False and `return_index` is True: returns a tuple
          of the despiked variable and a mask of outliers.
    """
    # Calculate the rolling mean using the specified filter type
    var_mean = filt.rolling(
        ds, var_name=var_name, dim=dim,
        window_size=window_size, filter_type=filter_type,
        min_periods=min_periods
    )[var_name]

    # Calculate the rolling standard deviation
    var_sd = filt.rolling_sd(
        ds, var_name=var_name, dim=dim,
        window_size=window_size, min_periods=min_periods,
    )

    # Identify the outliers based on the standard deviation threshold
    is_outside_criterion = (
        np.abs(var_mean - ds[var_name].values) > n_std * var_sd)

    # Apply the mask to remove outliers
    var_despiked = ds[var_name].where(~is_outside_criterion)

    # Optional plotting
    if plot:
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(ds[dim], ds[var_name], '.', color = 'tab:red', ms = 2,
                   label=f'Original {var_name} data', alpha=0.6)
        ax[0].plot(var_mean[dim], var_despiked, 'k',
                   label=f'Despiked {var_name} data')
        ax[1].plot(var_mean[dim], np.abs(var_mean - ds[var_name]),
                   label='| Data$-$rolling mean |', lw=1)
        ax[1].plot(var_mean[dim], n_std * var_sd, 'k', lw=0.4,
                   label=f'{n_std} $\\times$ Rolling std')
        ax[1].plot(var_mean[dim][is_outside_criterion],
                   np.abs(var_mean - ds[var_name])[is_outside_criterion],
                   '.r', label='Labelled as outlier')
        for axn in ax:
            leg = axn.legend(fontsize=9, ncol=1, handlelength = 1, bbox_to_anchor = (1, 0.5))
            leg.set_zorder(0)
            axn.set_xlabel('Index')
            if 'units' in ds[var_name].attrs:
                axn.set_ylabel(ds[var_name].attrs['units'])
        fig.suptitle(f'Despiking `{var_name}` along the dimension `{dim}`:')
        plt.tight_layout()

    # Optional printing
    if verbose:
        n_removed = np.sum(is_outside_criterion).item()
        print(
            f'Removed {n_removed} points from {var_name} after despiking'
            f' along dimension {dim}.'
        )

    # Return options based on flags
    if return_ds:
        ds_updated = ds.copy()
        ds_updated[var_name] = var_despiked
        if return_index:
            return ds_updated, is_outside_criterion
        return ds_updated
    else:
        if return_index:
            return var_despiked, is_outside_criterion
        return var_despiked