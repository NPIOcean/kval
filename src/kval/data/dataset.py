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

#### MODIFY METADATA

def add_now_as_date_created(D: xr.Dataset) -> xr.Dataset:
    """
    Add a global attribute "date_created" with today's date.

    Parameters:
    - D: The xarray.Dataset to which the attribute will be added.

    Returns:
    - The modified xarray.Dataset with the 'date_created' attribute.
    """
    now_time = pd.Timestamp.now()
    now_str = time.datetime_to_ISO8601(now_time)
    D.attrs['date_created'] = now_str
    return D

def add_processing_history_var(
    D: xr.Dataset,
    source_files: str | list[str] | np.ndarray = None,
    post_processing: bool = True,
    py_script: bool = True
) -> xr.Dataset:
    """
    Add a `PROCESSING` variable to store metadata about processing history.

    Parameters:
    - D: The xarray.Dataset to which the variable will be added.
    - source_files: A single file or list of files used in processing.
    - post_processing: If True, include post-processing information.
    - py_script: If True, include the Python script used for processing.

    Returns:
    - The modified xarray.Dataset with the `PROCESSING` variable.
    """
    D['PROCESSING'] = xr.DataArray(
        data=None, dims=[],
        attrs={
            'long_name': 'Empty variable whose attributes describe processing '
                         'history of the dataset.',
            'comment': '** NOTE: Experimental - testing this for documentation. **\n\n'
        }
    )

    if 'SBE_processing' in D.attrs:
        D['PROCESSING'].attrs['SBE_processing'] = D.attrs.pop('SBE_processing')
        D['PROCESSING'].attrs['comment'] += (
            '# SBE_processing #:\nDescription of automated editing applied using '
            'SeaBird software before post-processing in Python.\n'
        )

    if source_files is not None:
        if isinstance(source_files, str):
            source_file_string = os.path.basename(source_files)
        elif isinstance(source_files, (list, np.ndarray)):
            file_names = sorted(
                [os.path.basename(path) for path in np.sort(source_files)]
            )
            source_file_string = ', '.join(file_names)
        D['PROCESSING'].attrs['source_files'] = source_file_string
        D['PROCESSING'].attrs['comment'] += (
            '# source_files #:\nList of files produced by SBE processing.\n'
        )

    if post_processing:
        D['PROCESSING'].attrs['post_processing'] = ''
        D['PROCESSING'].attrs['comment'] += (
            '# post_processing #:\nDescription of post-processing starting with '
            '*source_files*.\n(Note: Indexing in the PRES dimension starts at 0 - '
            'for MATLAB add 1 to the index).\n'
        )

    if py_script:
        D['PROCESSING'].attrs['python_script'] = ''
        D['PROCESSING'].attrs['comment'] += (
            '# python_script #:\nPython script for reproducing post-processing '
            'from *source_files*.\n'
        )

    return D

#### HELPER FUNCTIONS   

#### EXPORT

def to_netcdf(
    D: xr.Dataset,
    path: str,
    file_name: str = None,
    convention_check: bool = False,
    add_to_history: bool = True,
    verbose: bool = True
) -> None:
    """
    Export xarray Dataset to NetCDF format.

    Parameters:
    - D: The xarray.Dataset to export.
    - path: Directory where the file will be saved.
    - file_name: Name of the NetCDF file.
    - convention_check: If True, check file conventions.
    - add_to_history: If True, update the history attribute.
    - verbose: If True, print information about the export process.
    """
    path = Path(path)

    D = add_now_as_date_created(D)
    D = conventionalize.reorder_attrs(D)

    if file_name is None:
        file_name = getattr(D, 'id', 'DATASET_NO_NAME')

    if not file_name.endswith('.nc'):
        file_name += '.nc'

    file_path = path / file_name

    if add_to_history:
        if 'history' not in D.attrs:
            D.attrs['history'] = ''

        if 'Creation of this netcdf file' in D.attrs['history']:
            history_lines = D.attrs['history'].split('\n')
            updated_history = [
                line for line in history_lines if "Creation of this netcdf file" not in line
            ]
            D.attrs['history'] = '\n'.join(updated_history)

        now_time = pd.Timestamp.now().strftime('%Y-%m-%d')
        D.attrs['history'] += f'\n{now_time}: Creation of this netcdf file.'

        if verbose:
            print(f'Updated history attribute. Current content:\n---')
            print(D.attrs['history'])
            print('---')

    try:
        D.to_netcdf(file_path)
    except PermissionError:
        user_input = input(f"The file {file_path} already exists. Overwrite? (y/n): ")
        if user_input.lower() in ['yes', 'y']:
            os.remove(file_path)
            D.to_netcdf(file_path)
            print(f"File {file_path} overwritten.")
        else:
            print("Operation canceled. File not overwritten.")

    if verbose:
        print(f'Exported NetCDF file as: {file_path}')

    if convention_check:
        print('Running convention checker:')
        check_conventions.check_file(file_path)

def metadata_to_txt(D: xr.Dataset, outfile: str) -> None:
    """
    Write metadata from an xarray.Dataset to a text file.

    Parameters:
    - D: The dataset containing metadata to write.
    - outfile: Path for the output text file. Extension '.txt' will be added if not provided.

    Returns:
    - None
    """
    if not outfile.lower().endswith('.txt'):
        outfile += '.txt'

    with open(outfile, 'w') as f:
        file_header = f'FILE METADATA FROM: {D.attrs.get("id", "Unknown")}'
        f.write('#' * 80 + '\n')
        f.write(f'####  {file_header:<68}  ####\n')
        f.write('#' * 80 + '\n')
        f.write('\n' + '#' * 27 + '\n')
        f.write('### GLOBAL ATTRIBUTES   ###\n')
        f.write('#' * 27 + '\n')
        f.write('\n')

        for key, item in D.attrs.items():
            f.write(f'# {key}:\n')
            f.write(f'{item}\n')

        f.write('\n' + '#' * 27 + '\n')
        f.write('### VARIABLE ATTRIBUTES ###\n')
        f.write('#' * 27 + '\n')

        all_vars = list(D.coords) + list(D.data_vars)

        for varnm in all_vars:
            f.write('\n' + '-' * 50 + '\n')
            f.write(f'{varnm} (coordinate)\n' if varnm in D.coords else f'{varnm}\n')
            f.write('-' * 50 + '\n')

            for key, item in D[varnm].attrs.items():
                f.write(f'# {key}:\n')
                f.write(f'{item}\n')
