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