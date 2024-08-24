'''
KVAL.UTIL.ERA5

Helper function to quickly grab ERA5 data from the APDRC OpenDap servers.

Can read hourly or monthly values now - may implement daily as well
eventually.

Implemented:
- Sea surface pressure
- Surface winds

To be implemented:
- Air temperature
- Surface heat fluxes
'''

import xarray as xr
import warnings
from typing import Literal, Union
from matplotlib.dates import num2date
from datetime import datetime
import pandas as pd

era5_base_url = (
    'http://apdrc.soest.hawaii.edu:80/dods/public_data/Reanalysis_Data/ERA5/')
era5_monthly_url = f'{era5_base_url}monthly_2d/Surface/'

era_var_dict = {
    'SLP':  {'era_name': 'sp',
             'hourly_url': f'{era5_base_url}hourly/Surface_pressure'},
    'UWND': {'era_name': 'u10',
             'hourly_url': f'{era5_base_url}hourly/U_wind_component_10m'},
    'VWND': {'era_name': 'v10',
             'hourly_url': f'{era5_base_url}hourly/V_wind_component_10m'},
    'T2M':  {'era_name': 't2',
             'hourly_url': f'{era5_base_url}hourly/2m_temperature'}, }


def open_era5_dataset(
    variable: Literal['SLP', 'UW10M', 'VW10M', 'T2M'],
    time_resolution: Literal['monthly', 'hourly'],
    verbose: bool = True
) -> xr.Dataset:
    """
    Open an ERA-5 dataset remotely over OpenDAP.

    This function lazily loads the dataset from the Asia-Pacific Data Research
    Center (APDRC), which hosts a large number of climate datasets.

    Args:
        variable (Literal['SLP', 'UW10M', 'VW10M', 'T2M']):
            The variable to retrieve:
                - 'SLP': Sea Level Pressure.
                - 'UW10M': U-component (zonal) wind at 10 meters.
                - 'VW10M': V-component (meridional) wind at 10 meters.
                - 'T2M': Air temperature at 2 meters.
        time_resolution (Literal['monthly', 'hourly']):
            Temporal resolution of the dataset:
                - 'monthly': Monthly aggregated data.
                - 'hourly': Hourly data.
        verbose (bool):
            Whether or not to print status.

    Returns:
        xr.Dataset: The requested ERA-5 dataset loaded lazily.

    Raises:
        ValueError: If an unsupported time_resolution is provided.
    """

    # Select the dataset URL based on the time resolution
    if time_resolution == 'monthly':
        dataset_url = era5_monthly_url
    elif time_resolution == 'hourly':
        dataset_url = era_var_dict[variable]['hourly_url']
    else:
        raise ValueError(f"Erroneous time_resolution {time_resolution}. "
                         "Must be 'monthly' or 'hourly' (daily not "
                         "implemented yet).")

    # Load the dataset, suppressing specific xarray warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=xr.coding.times.SerializationWarning)
        if verbose:
            print(f"Loading ERA5 file from APDRC servers ({dataset_url}) ..",
                  end='')
        ds = xr.open_dataset(dataset_url)
        if verbose:
            print(" done.")

    return ds


def get_era5_time_series_point(
    variable: Literal['SLP', 'UW10M', 'VW10M', 'T2M'],
    time_resolution: Literal['monthly', 'hourly'],
    lat: float, lon: float,
    time_start: Union[str, datetime, float],
    time_end: Union[str, datetime, float],
    method: str = 'nearest',
    center_time: bool = True,
    verbose: bool = True
) -> xr.Dataset:
    """
    Retrieve a time series from the ERA-5 dataset for a specified
    latitude and longitude, within a given time range.

    Data are obrained over OpenDAP from the Asia-Pacific Data Research
    Center (APDRC), which hosts a large number of climate datasets.

    Args:
        variable (Literal['SLP', 'UW10M', 'VW10M', 'T2M']):
            The variable to retrieve:
                - 'SLP': Sea Level Pressure.
                - 'UW10M': U-component (zonal) wind at 10 meters.
                - 'VW10M': V-component (meridional) wind at 10 meters.
                - 'T2M': Air temperature at 2 meters.
        time_resolution (Literal['monthly', 'hourly']):
            Temporal resolution of the dataset:
                - 'monthly': Monthly aggregated data.
                - 'hourly': Hourly data.
        lat (float): Latitude in decimal degrees.
        lon (float): Longitude in decimal degrees. Negative values will be
                     adjusted to the [0, 360] range to matche ERA5.
        time_start (Union[str, datetime, float]):
            Start time for the time series. Can be:
                - A string in 'YYYY-MM-DD' format.
                - A `datetime` object.
                - A numeric value representing days since '1970-01-01'.
        time_end (Union[str, datetime, float]):
            End time for the time series. Same format as `time_start`.
        method (str, optional):
            Method for spatial interpolation.
            Defaults to 'nearest'.
        center_time (bool, optional):
            If `True`, set time stamp to center of time interval
            (mid-month for monthly, mid.hour for hourly).
        verbose (bool, optional): If `True`, prints progress messages.
                                  Defaults to `True`.

    Returns:
        xr.Dataset: The extracted ERA-5 time series dataset.

    Raises:
        ValueError: If an unsupported `time_resolution` is provided.
    """

    if lon < 0:
        lon += 360

    # If these times are python numerals (days since 1970-1-1),
    # convert them to datetime objects.
    # Have to make them time zone unaware in orded to compare with ERA5
    # (technical - all times are always assumed to be UTC in any event).
    if isinstance(time_start, float):
        time_start = num2date(time_start)
        time_start = time_start.replace(tzinfo=None)
    if isinstance(time_end, float):
        time_end = num2date(time_end)
        time_end = time_end.replace(tzinfo=None)

    time_slice = slice(time_start, time_end)

    ds_full = open_era5_dataset(variable=variable,
                                time_resolution=time_resolution,
                                verbose=verbose)

    ds_subset_lazy = (ds_full.sel(lat=lat, lon=lon, method=method)
                             .sel(time=time_slice))
    if verbose:
        print('Evaluating local ERA5 time series ..', end='')

    ds_subset = ds_subset_lazy.compute()

    if verbose:
        print(' done.')

    # Rename variable
    era5_varname = era_var_dict[variable]['era_name']
    # Account for T2M variable names being different in daily ("t2") ands
    # monthly ERA ("t2m")
    if time_resolution == 'monthly' and variable == 'T2M':
        era5_varname = 't2m'

    ds_subset = ds_subset.rename_vars({era5_varname: variable})
    ds_subset[variable].attrs['era5_variable_name'] = era5_varname

    # Convert K to C if temperature
    if variable == 'T2M':
        ds_subset[variable] -= 273.15
        ds_subset[variable].attrs['long_name'] = (
            ds_subset[variable].long_name.replace('[k]', '[C]')
        )

    # Center time
    if center_time:
        if time_resolution == 'monthly':
            # Calculate the middle of the month
            ds_subset['time'] = (
                ds_subset.indexes['time']
                + pd.to_timedelta(
                    ds_subset.indexes['time'].days_in_month / 2,
                    unit='D'))
        elif time_resolution == 'hourly':
            # Calculate the middle of the hour (30 minutes past the hour)
            ds_subset['time'] = (ds_subset.indexes['time']
                                 + pd.to_timedelta(30, unit='m'))

        ds_subset.time.attrs['note'] = '*Center* of time interval.'
    else:
        ds_subset.time.attrs['note'] = '*Start* of time interval.'

    ds_subset.time.attrs['long_name'] = 'time'

    return ds_subset
