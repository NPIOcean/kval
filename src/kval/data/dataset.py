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