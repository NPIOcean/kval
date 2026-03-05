"""
ctd.data.dataset

Various functions for working with generalized datasets.
"""

import os
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
from kval.metadata import check_conventions, conventionalize
from kval.util import time

#### ADD VARIABLES


def add_latlon(ds, lon, lat, suppress_latlon_warning=False):
    """
    Adds 0-d LATITUDE and LONGITUDE variables to a dataset.

    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    ds['LATITUDE'] = ((), (lat), {
            "standard_name": "latitude",
            "units": "degree_north",
            "long_name": "latitude",
        },)

    ds['LONGITUDE'] = ((), (lon), {
            "standard_name": "longitude",
            "units": "degree_east",
            "long_name": "longitude",
        },)

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


def add_processing_history_var_ctd(
    ds: xr.Dataset,
    source_file: str | list[str] | np.ndarray | None = None,
    post_processing: bool = True,
    py_script: bool = True
) -> xr.Dataset:
    """
    Add a `PROCESSING` variable to store metadata about the dataset's processing history.

    This variable tracks the source files, post-processing steps, and the Python
    script used for processing.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to which the `PROCESSING` variable will be added.
    source_file : str | list[str] | np.ndarray | None, default=None
        Path(s) to source file(s) from which data were loaded.
    post_processing : bool, default=True
        Include information about post-processing steps if True.
    py_script : bool, default=True
        Include the Python script used for processing if True.

    Returns
    -------
    xr.Dataset
        The modified dataset with the added `PROCESSING` variable.
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    ds['PROCESSING'] = xr.DataArray(
        data=None, dims=[],
        attrs={
            'long_name': 'Empty variable whose attributes describe processing '
                         'history of the dataset.',
            'comment': ''
        })

    if 'SBE_processing' in ds.attrs:
        ds['PROCESSING'].attrs['SBE_processing'] = ds.attrs.pop('SBE_processing')
        ds['PROCESSING'].attrs['comment'] += (
            '# SBE_processing #: Description of automated editing applied using '
            'SeaBird software before post-processing in Python.\n'
        )

    if source_file is not None:
        if isinstance(source_file, str):
            source_file_string = os.path.basename(source_file)
        elif isinstance(source_file, (list, np.ndarray)):
            file_names = sorted(
                [os.path.basename(path) for path in np.sort(source_file)]
            )
            source_file_string = ', '.join(file_names)
        ds['PROCESSING'].attrs['source_file'] = source_file_string
        ds['PROCESSING'].attrs['comment'] = (
            '# source_file #: List of files produced by SBE processing.\n'
        )

    if post_processing:
        ds['PROCESSING'].attrs['post_processing'] = ''
        ds['PROCESSING'].attrs['comment'] += (
            '# post_processing #:\nDescription of post-processing starting with '
            '`source_file`.\n(Note: Indexing in the PRES dimension starts at 0 - '
            'for MATLAB add 1 to the index).\n'
        )

    if py_script:
        ds['PROCESSING'].attrs['python_script'] = ''
        ds['PROCESSING'].attrs['comment'] += (
            '# python_script #: Python script for reproducing post-processing '
            'from `source_file`.\n'
        )

    return ds


def add_processing_history_var_moored(
    ds: xr.Dataset,
    source_file: str = None,
    post_processing: bool = True,
    py_script: bool = True
) -> xr.Dataset:
    """
    Add a `PROCESSING` variable to store metadata about processing history.

    Parameters:
    - ds: The xarray.Dataset to which the variable will be added.
    - source_file: Source data file (e.g. .cnv, .rsk).
    - post_processing: If True, include post-processing information.
    - py_script: If True, include the Python script used for processing.

    Returns:
    - The modified xarray.Dataset with the `PROCESSING` variable.
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    ds['PROCESSING'] = xr.DataArray(
        data=None, dims=[],
        attrs={
            'long_name': 'Empty variable whose attributes describe processing '
                         'history of the dataset.',
        })

    if source_file is not None:
        if isinstance(source_file, str):
            source_file_string = os.path.basename(source_file)
        else:
            raise ValueError('Invalid `source_file` (should be a string).')

        ds['PROCESSING'].attrs['source_file'] = source_file_string

    if post_processing:
        ds['PROCESSING'].attrs['post_processing'] = ''

    if py_script:
        ds['PROCESSING'].attrs['python_script'] = ''

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
        check_conventions.check_file(file_path)

#### OCEANOGRAPHIC CALCULATIONS


def calculate_PSAL(
    ds: xr.Dataset,
    cndc_var: str = "CNDC",
    temp_var: str = "TEMP",
    pres_var: str = "PRES",
    psal_var: str = "PSAL",
    retain_nans: bool = True,
) -> xr.Dataset:
    """
    (Re)calculate Practical Salinity (PSAL) from conductivity, temperature,
    and pressure using the GSW-Python module.

    Parameters
    ----------
    ds : xr.Dataset
    cndc_var, temp_var, pres_var, psal_var : str
        Variable names. Defaults: 'CNDC', 'TEMP', 'PRES', 'PSAL'.
    retain_nans : bool
        If PSAL already exists, retain its NaN mask. Default True.

    Returns
    -------
    xr.Dataset
        Dataset with recalculated PSAL.
    """
    import gsw

    ds = ds.copy(deep=True)

    CNDC_ = ds[cndc_var].copy()
    if 'units' in CNDC_.attrs and CNDC_.units == 'S m-1':
        print('Detected S m-1 unit - applying an x10 factor to CNDC.')
        CNDC_.values *= 10

    PSAL = gsw.SP_from_C(CNDC_.values, ds[temp_var].values, ds[pres_var].values)

    if retain_nans and psal_var in ds:
        PSAL = np.where(np.isnan(ds[psal_var]), np.nan, PSAL)

    if psal_var in ds:
        ds[psal_var][:] = PSAL
    else:
        ds[psal_var] = (ds[cndc_var].dims, PSAL, {'units': '1'})
        if ('sensor_calibration_date' in ds[temp_var].attrs
                and 'sensor_calibration_date' in ds[cndc_var].attrs):
            ds[psal_var].attrs['sensor_calibration_date'] = (
                f'{ds[temp_var].sensor_calibration_date} (TEMP), '
                f'{ds[cndc_var].sensor_calibration_date} (CNDC)')

    ds[psal_var].attrs['note'] = (
        f'Computed from {cndc_var}, {temp_var}, {pres_var} '
        'using the Python gsw module.')

    return ds


def calculate_SA_CT(
    ds: xr.Dataset,
    cndc_var: str = "CNDC",
    temp_var: str = "TEMP",
    pres_var: str = "PRES",
    psal_var: str = "PSAL",
) -> xr.Dataset:
    """
    Calculate Absolute Salinity (SA) and Conservative Temperature (CT)
    using the GSW-Python module.

    Parameters
    ----------
    ds : xr.Dataset
    cndc_var, temp_var, pres_var, psal_var : str
        Variable names. Defaults: 'CNDC', 'TEMP', 'PRES', 'PSAL'.

    Returns
    -------
    xr.Dataset
        Dataset with SA and CT added.
    """
    import gsw

    ds = ds.copy(deep=True)

    SA = gsw.SA_from_SP(ds[psal_var], ds[pres_var], ds.LONGITUDE, ds.LATITUDE)
    CT = gsw.CT_from_t(SA, ds[temp_var], ds[pres_var])

    ds['SA'] = (ds[psal_var].dims, SA.values,
                {'units': 'g kg-1',
                 'standard_name': 'sea_water_absolute_salinity',
                 'long_name': 'Absolute Salinity'})
    ds['CT'] = (ds[psal_var].dims, CT.values,
                {'units': 'degree_C',
                 'standard_name': 'sea_water_conservative_temperature',
                 'long_name': 'Conservative Temperature'})

    for varname in ['CT', 'SA']:
        ds[varname].attrs['note'] = (
            f'Computed from {cndc_var}, {temp_var}, {pres_var} '
            'using the Python gsw module.')

    return ds


def calculate_rho(
    ds: xr.Dataset,
    cndc_var: str = "CNDC",
    temp_var: str = "TEMP",
    pres_var: str = "PRES",
    psal_var: str = "PSAL",
) -> xr.Dataset:
    """
    Calculate in-situ seawater density (RHO) using the GSW-Python module.

    Parameters
    ----------
    ds : xr.Dataset
    cndc_var, temp_var, pres_var, psal_var : str
        Variable names. Defaults: 'CNDC', 'TEMP', 'PRES', 'PSAL'.

    Returns
    -------
    xr.Dataset
        Dataset with RHO added.
    """
    import gsw

    ds = ds.copy(deep=True)

    SA = gsw.SA_from_SP(ds[psal_var], ds[pres_var], ds.LONGITUDE, ds.LATITUDE)
    CT = gsw.CT_from_t(SA, ds[temp_var], ds[pres_var])
    RHO = gsw.rho(SA, CT, ds[pres_var])

    ds['RHO'] = (ds[psal_var].dims, RHO.values,
                 {'units': 'kg m-3',
                  'standard_name': 'sea_water_density',
                  'long_name': 'In-situ seawater density',
                  'note': (f'Computed from {cndc_var}, {temp_var}, {pres_var} '
                           'using the Python gsw module.')})
    return ds


def calculate_sig0(
    ds: xr.Dataset,
    temp_var: str = "TEMP",
    pres_var: str = "PRES",
    psal_var: str = "PSAL",
) -> xr.Dataset:
    """
    Calculate potential density anomaly (SIG0) using the GSW-Python module.

    Parameters
    ----------
    ds : xr.Dataset
    temp_var, pres_var, psal_var : str
        Variable names. Defaults: 'TEMP', 'PRES', 'PSAL'.

    Returns
    -------
    xr.Dataset
        Dataset with SIG0 added.
    """
    import gsw

    ds = ds.copy(deep=True)

    SA = gsw.SA_from_SP(ds[psal_var], ds[pres_var], ds.LONGITUDE, ds.LATITUDE)
    CT = gsw.CT_from_t(SA, ds[temp_var], ds[pres_var])
    SIG0 = gsw.sigma0(SA, CT)

    ds['SIG0'] = (ds[psal_var].dims, SIG0.values,
                  {'units': 'kg m-3',
                   'standard_name': 'sea_water_sigma_theta',
                   'long_name': 'Potential density minus 1000 kg m-3',
                   'note': (f'Computed from {temp_var}, {pres_var} '
                            'using the Python gsw module.')})
    return ds


def calculate_CNDC(
    ds: xr.Dataset,
    cndc_var: str = "CNDC",
    temp_var: str = "TEMP",
    pres_var: str = "PRES",
    psal_var: str = "PSAL",
    retain_nans: bool = True,
) -> xr.Dataset:
    """
    (Re)calculate Conductivity (CNDC) from practical salinity, temperature,
    and pressure using the GSW-Python module.

    Parameters
    ----------
    ds : xr.Dataset
    cndc_var, temp_var, pres_var, psal_var : str
        Variable names. Defaults: 'CNDC', 'TEMP', 'PRES', 'PSAL'.
    retain_nans : bool
        If CNDC already exists, retain its NaN mask. Default True.

    Returns
    -------
    xr.Dataset
        Dataset with recalculated CNDC.
    """
    import gsw

    ds = ds.copy(deep=True)

    CNDC = gsw.C_from_SP(ds[psal_var].values, ds[temp_var].values, ds[pres_var].values)

    if retain_nans and cndc_var in ds:
        CNDC = np.where(np.isnan(ds[cndc_var]), np.nan, CNDC)

    if cndc_var in ds:
        ds[cndc_var][:] = CNDC
    else:
        ds[cndc_var] = (ds[psal_var].dims, CNDC, {'units': 'mS/cm'})
        if ('sensor_calibration_date' in ds[temp_var].attrs
                and 'sensor_calibration_date' in ds[psal_var].attrs):
            ds[cndc_var].attrs['sensor_calibration_date'] = (
                f'{ds[temp_var].sensor_calibration_date} (TEMP), '
                f'{ds[psal_var].sensor_calibration_date} (PSAL)')

    ds[cndc_var].attrs['note'] = (
        f'Computed from {psal_var}, {temp_var}, {pres_var} '
        'using the Python gsw module.')

    return ds
