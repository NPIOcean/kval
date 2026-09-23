"""
kval.data.dataset

Various functions for working with generalized datasets.
"""

import os
import warnings
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
from kval.metadata import compliance, conventionalize
from kval.util import time
import gsw
from kval.util.xr_funcs import append_processing_history

#### ADD VARIABLES


def add_latlon(ds, lon=None, lat=None, suppress_latlon_warning=False):
    """
    Adds LATITUDE and LONGITUDE coordinate variables to a dataset.

    Either lon or lat may be omitted -- only the one(s) actually
    provided get assigned. If a value is omitted, a warning is issued
    unless suppress_latlon_warning=True.
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    if lat is not None:
        ds['LATITUDE'] = ((), (lat), {
                "standard_name": "latitude",
                "units": "degree_north",
                "long_name": "latitude",
            },)
        ds = ds.set_coords('LATITUDE')
    elif not suppress_latlon_warning:
        warnings.warn('No latitude value provided -- LATITUDE not set.')

    if lon is not None:
        ds['LONGITUDE'] = ((), (lon), {
                "standard_name": "longitude",
                "units": "degree_east",
                "long_name": "longitude",
            },)
        ds = ds.set_coords('LONGITUDE')
    elif not suppress_latlon_warning:
        warnings.warn('No longitude value provided -- LONGITUDE not set.')

    return ds



# Recalculate sal
def calculate_PSAL(
    ds: xr.Dataset,
    cndc_var: str = "CNDC",
    temp_var: str = "TEMP",
    pres_var: str = "PRES",
    psal_var: str = "PSAL",
    retain_nans: bool = True,
) -> xr.Dataset:
    """(Re)calculate Practical Salinity (PSAL) from conductivity, temperature,
    and pressure using the GSW-Python module
    (https://teos-10.github.io/GSW-Python/).

    This function updates the PSAL variable in the dataset with newly computed
    salinity values while preserving the metadata attributes of PSAL.

    Args:
        ds (xr.Dataset):
            The input dataset containing conductivity, temperature, and
            pressure variables.
        cndc_var (str):
            The name of the conductivity variable in the dataset.
            Defaults to 'CNDC'.
        temp_var (str):
            The name of the temperature variable in the dataset.
            Defaults to 'TEMP'.
        pres_var (str):
            The name of the pressure variable in the dataset.
            Defaults to 'PRES'.
        psal_var (str):
            The name of the salinity variable in the dataset.
            Defaults to 'PSAL'.
        retain_nans (bool):
            If there is already a PSAL field: Retain the Nan values from
            old to new PSAL field.
            Defaults to True.

    Returns:
        xr.Dataset: The updated dataset with recalculated PSAL values.

    Notes:
        The operation preserves PSAL metadata attributes. If the input sensors
        change (e.g., if a different temperature sensor is used), the PSAL
        metadata attributes should be updated accordingly.
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    CNDC_ = ds[cndc_var].copy()
    if 'units' in CNDC_.attrs:
        if CNDC_.units == 'S m-1':
            print('Detected S m-1 unit - applying an x10 factor to CNDC.')
            CNDC_.values *= 10



    # Calculate PSAL
    PSAL = gsw.SP_from_C(
        CNDC_.values, ds[temp_var].values, ds[pres_var].values)

    # Retain NaNs if applicable
    if retain_nans and psal_var in ds:
       # PSAL = PSAL.where(~np.isnan(ds[psal_var]), np.nan)
        PSAL = np.where(np.isnan(ds[psal_var]), np.nan, PSAL)
    if psal_var in ds:
        ds[psal_var][:] = PSAL
    else:
        ds[psal_var] = (ds[cndc_var].dims, PSAL.data,
                        {'units': '1',})
        if ('sensor_calibration_date' in ds[temp_var].attrs
           and 'sensor_calibration_date' in ds[cndc_var].attrs):
            ds[psal_var].attrs['sensor_calibration_date'] = (
                f'{ds[temp_var].sensor_calibration_date} (TEMP), '
                f'{ds[cndc_var].sensor_calibration_date} (CNDC)')

    note = (
        f"Computed from {cndc_var}, {temp_var}, {pres_var} "
        "using the Python gsw module."
    )
    ds = append_processing_history(ds, psal_var, note, deep_copy=False)

    return ds


# Recalculate SA & CT
def calculate_SA_CT(
    ds: xr.Dataset,
    cndc_var: str = "CNDC",
    temp_var: str = "TEMP",
    pres_var: str = "PRES",
    psal_var: str = "PSAL",
) -> xr.Dataset:
    """Recalculate absolute salinity (SA) and coneservative temperature (CT)
    from conductivity, temperature, and pressure using the GSW-Python module
    (https://teos-10.github.io/GSW-Python/).

    This function adds or updates the SA and CT variables in the dataset with
    newly computed values.

    Args:
        ds (xr.Dataset):
            The input dataset containing conductivity, temperature, and
            pressure variables.
        cndc_var (str):
            The name of the conductivity variable in the dataset.
            Defaults to 'CNDC'.
        temp_var (str):
            The name of the temperature variable in the dataset.
            Defaults to 'TEMP'.
        pres_var (str):
            The name of the pressure variable in the dataset.
            Defaults to 'PRES'.
        psal_var (str):
            The name of the salinity variable in the dataset.
            Defaults to 'PSAL'.

    Returns:
        xr.Dataset: The updated dataset with SA, CT values.

    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    # Calculate absolute salinity
    SA = gsw.SA_from_SP(ds[psal_var], ds[pres_var], ds.LONGITUDE, ds.LATITUDE)
    # Calculate conservative temperature
    CT = gsw.CT_from_t(SA, ds[temp_var], ds[pres_var])

    ds['SA'] = (ds[psal_var].dims, SA.values,
                {'units': 'g kg-1',
                 'standard_name': 'sea_water_absolute_salinity',
                 'long_name': 'Absolute Salinity'})

    ds['CT'] = (ds[psal_var].dims, CT.values,
                {'units': 'degree_C',
                 'standard_name':'sea_water_conservative_temperature',
                 'long_name': 'Conservative Temperature'})

    note = (
        f"Computed from {cndc_var}, {temp_var}, {pres_var} "
        "using the Python gsw module."
    )
    for varname in ['CT', 'SA']:
        ds = append_processing_history(ds, varname, note, deep_copy=False)

    return ds



# Recalculate RHO
def calculate_rho(
    ds: xr.Dataset,
    cndc_var: str = "CNDC",
    temp_var: str = "TEMP",
    pres_var: str = "PRES",
    psal_var: str = "PSAL",
) -> xr.Dataset:
    """Recalculate Density (RHO) from conductivity, temperature,
    and pressure using the GSW-Python module
    (https://teos-10.github.io/GSW-Python/).

    This function adds or updates the RHO variable in the dataset with newly
    computed density value.

    Args:
        ds (xr.Dataset):
            The input dataset containing conductivity, temperature, and
            pressure variables.
        cndc_var (str):
            The name of the conductivity variable in the dataset.
            Defaults to 'CNDC'.
        temp_var (str):
            The name of the temperature variable in the dataset.
            Defaults to 'TEMP'.
        pres_var (str):
            The name of the pressure variable in the dataset.
            Defaults to 'PRES'.
        psal_var (str):
            The name of the salinity variable in the dataset.
            Defaults to 'PSAL'.

    Returns:
        xr.Dataset: The updated dataset with RHO values.

    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    # Calculate absolute salinity
    SA = gsw.SA_from_SP(ds[psal_var], ds[pres_var], ds.LONGITUDE, ds.LATITUDE)
    # Calculate conservative temperature
    CT = gsw.CT_from_t(SA, ds[temp_var], ds[pres_var])
    # Calculate density
    RHO = gsw.rho(SA, CT, ds[pres_var])

    ds['RHO'] = (ds[psal_var].dims, RHO.values,
                 {'units': 'kg m-3', 'standard_name':'sea_water_density',
                  'long_name' : 'In-situ seawater density'})


    note = (
        f"Computed from {cndc_var}, {temp_var}, {pres_var} "
        "using the Python gsw module."
    )
    ds = append_processing_history(ds, 'RHO', note, deep_copy=False)

    return ds


# Recalculate sigma0
def calculate_sig0(
    ds: xr.Dataset,
    temp_var: str = "TEMP",
    pres_var: str = "PRES",
    psal_var: str = "PSAL",
) -> xr.Dataset:
    """Recalculate potential density anomaly (SIG0) from
    conductivity, temperature, and pressure using the GSW-Python module
    (https://teos-10.github.io/GSW-Python/).

    This function adds or updates the SIG0 variable in the dataset with newly
    computed density value.

    Args:
        ds (xr.Dataset):
            The input dataset containing conductivity, temperature, and
            pressure variables.
        temp_var (str):
            The name of the temperature variable in the dataset.
            Defaults to 'TEMP'.
        pres_var (str):
            The name of the pressure variable in the dataset.
            Defaults to 'PRES'.
        psal_var (str):
            The name of the salinity variable in the dataset.
            Defaults to 'PSAL'.

    Returns:
        xr.Dataset: The updated dataset with SIG0 values.

    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    # Calculate absolute salinity
    SA = gsw.SA_from_SP(ds[psal_var], ds[pres_var], ds.LONGITUDE, ds.LATITUDE)
    # Calculate conservative temperature
    CT = gsw.CT_from_t(SA, ds[temp_var], ds[pres_var])
    # Calculate sigma 0
    SIG0 = gsw.sigma0(SA, CT)

    ds['SIG0'] = (ds[psal_var].dims, SIG0.values,
                 {'units': 'kg m-3', 'standard_name': 'sea_water_sigma_theta',
                  'long_name': ('Potential density of water '
                                'minus 1000 kg m-3.')})

    note = (
        f"Computed from {temp_var}, {pres_var} "
        "using the Python gsw module."
    )
    ds = append_processing_history(ds, 'SIG0', note, deep_copy=False)
    return ds


# Recalculate cndc
def calculate_CNDC(
    ds: xr.Dataset,
    cndc_var: str = "CNDC",
    temp_var: str = "TEMP",
    pres_var: str = "PRES",
    psal_var: str = "PSAL",
    retain_nans: bool = True,
) -> xr.Dataset:
    """(Re)calculate Conductivity (CNDC) from practical salinity, temperature,
    and pressure using the GSW-Python module
    (https://teos-10.github.io/GSW-Python/).

    This function updates the CNDC variable in the dataset with newly computed
    conductivity values while preserving the metadata attributes of CNDC.

    Args:
        ds (xr.Dataset):
            The input dataset containing salinity, temperature, and
            pressure variables.
        psal_var (str):
            The name of the salinity variable in the dataset.
            Defaults to 'PSAL'.
        temp_var (str):
            The name of the temperature variable in the dataset.
            Defaults to 'TEMP'.
        pres_var (str):
            The name of the pressure variable in the dataset.
            Defaults to 'PRES'.
        cndc_var (str):
            The name of the conductivity variable in the dataset.
            Defaults to 'CNDC'.
        retain_nans (bool):
            If there is already a CNDC field: Retain the NaN values from
            old to new CNDC field.
            Defaults to True.

    Returns:
        xr.Dataset: The updated dataset with recalculated CNDC values.

    Notes:
        The operation preserves CNDC metadata attributes. If the input sensors
        change (e.g., if a different temperature sensor is used), the CNDC
        metadata attributes should be updated accordingly.
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    # Calculate CNDC (in mS/cm)
    CNDC = gsw.C_from_SP(
        ds[psal_var].values, ds[temp_var].values, ds[pres_var].values
    )

    # Retain NaNs if applicable
    if retain_nans and cndc_var in ds:
        CNDC = np.where(np.isnan(ds[cndc_var]), np.nan, CNDC)

    # Overwrite or create CNDC variable
    if cndc_var in ds:
        ds[cndc_var][:] = CNDC
    else:
        ds[cndc_var] = (ds[psal_var].dims, CNDC.data, {'units': 'mS/cm'})
        if ('sensor_calibration_date' in ds[temp_var].attrs
           and 'sensor_calibration_date' in ds[psal_var].attrs):
            ds[cndc_var].attrs['sensor_calibration_date'] = (
                f'{ds[temp_var].sensor_calibration_date} (TEMP), '
                f'{ds[psal_var].sensor_calibration_date} (PSAL)'
            )

    note = (
        f"Computed from {psal_var}, {temp_var}, {pres_var} "
        "using the Python gsw module."
    )
    ds = append_processing_history(ds, cndc_var, note, deep_copy=False)

    return ds


def calculate_ss(
    ds: xr.Dataset,
    cndc_var: str = "CNDC",
    temp_var: str = "TEMP",
    pres_var: str = "PRES",
    psal_var: str = "PSAL",
) -> xr.Dataset:
    """Calculate sound speed (SVEL) from conductivity, temperature, and
    pressure using the GSW-Python module
    (https://teos-10.github.io/GSW-Python/).

    If SA and CT are already present in the dataset (e.g. from a prior
    call to calculate_SA_CT), they are reused rather than recomputed.
    Otherwise they are computed internally first.

    Args:
        ds (xr.Dataset):
            The input dataset containing conductivity, temperature, and
            pressure variables.
        cndc_var (str): Defaults to 'CNDC'.
        temp_var (str): Defaults to 'TEMP'.
        pres_var (str): Defaults to 'PRES'.
        psal_var (str): Defaults to 'PSAL'.

    Returns:
        xr.Dataset: The updated dataset with SVEL values.
    """

    ds = ds.copy(deep=True)

    if "SA" in ds and "CT" in ds:
        SA, CT = ds["SA"], ds["CT"]
    else:
        SA = gsw.SA_from_SP(ds[psal_var], ds[pres_var], ds.LONGITUDE, ds.LATITUDE)
        CT = gsw.CT_from_t(SA, ds[temp_var], ds[pres_var])

    SS = gsw.sound_speed(SA, CT, ds[pres_var])

    ds["SVEL"] = (ds[psal_var].dims, np.asarray(SS),
                  {"units": "m s-1",
                   "standard_name": "speed_of_sound_in_sea_water",
                   "long_name": "Sound velocity"})

    note = (
        f"Computed from {cndc_var}, {temp_var}, {pres_var} "
        "using the Python gsw module."
    )
    ds = append_processing_history(ds, "SVEL", note, deep_copy=False)

    return ds


#### MODIFY METADATA

def add_now_as_date_created(ds: xr.Dataset) -> xr.Dataset:
    """
    Add a global attribute "date_created" with today's date.

    Parameters:
    - D: The xarray.Dataset to which the attribute will be added.

    Returns:
    - The modified xarray.Dataset with the 'date_created' attribute.
    """
    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds
    now_time = pd.Timestamp.now()
    now_str = time.datetime_to_ISO8601(now_time)
    ds.attrs['date_created'] = now_str
    return ds

#### HELPER FUNCTIONS

#### EXPORT

def to_netcdf(
    ds: xr.Dataset,
    path: str,
    file_name: str = None,
    convention_check: bool = False,
    add_to_history: bool = True,
    verbose: bool = True
) -> None:
    """
    Export xarray Dataset to NetCDF format.

    Parameters:
    - ds: The xarray.Dataset to export.
    - path: Directory where the file will be saved.
    - file_name: Name of the NetCDF file.
    - convention_check: If True, check file conventions.
    - add_to_history: If True, update the history attribute.
    - verbose: If True, print information about the export process.
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    path = Path(path)

    ds = add_now_as_date_created(ds)
    ds = conventionalize.reorder_attrs(ds)

    if file_name is None:
        file_name = getattr(ds, 'id', 'DATASET_NO_NAME')

    if not file_name.endswith('.nc'):
        file_name += '.nc'

    file_path = path / file_name

    if add_to_history:
        if 'history' not in ds.attrs:
            ds.attrs['history'] = ''

        if 'Creation of this netcdf file' in ds.attrs['history']:
            history_lines = ds.attrs['history'].split('\n')
            updated_history = [
                line for line in history_lines
                if "Creation of this netcdf file" not in line
            ]
            ds.attrs['history'] = '\n'.join(updated_history)

        now_time = pd.Timestamp.now().strftime('%Y-%m-%d')
        ds.attrs['history'] += f'\n{now_time}: Creation of this netcdf file.'

        if verbose:
            print(f'Updated history attribute. Current content:\n---')
            print(ds.attrs['history'])
            print('---')

    try:
        ds.to_netcdf(file_path)
    except PermissionError:
        user_input = input(f"The file {file_path} already exists. Overwrite? (y/n): ")
        if user_input.lower() in ['yes', 'y']:
            os.remove(file_path)
            ds.to_netcdf(file_path)
            print(f"File {file_path} overwritten.")
        else:
            print("Operation canceled. File not overwritten.")

    if verbose:
        print(f'Exported NetCDF file as: {file_path}')

    if convention_check:
        print('Running convention checker:')
        compliance.compliance_checks_ioos(file_path)