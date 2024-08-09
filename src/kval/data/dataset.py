'''
DATASET.PY

Various functions to be applied to generalized datasets.
'''

from kval.util import time
import pandas as pd
import xarray as xr
from kval.metadata import check_conventions, conventionalize
import os
from pathlib import Path


#### MODIFY METADATA

def add_now_as_date_created(D):
    '''
    Add a global attribute "date_created" with todays date.
    '''
    now_time = pd.Timestamp.now()
    now_str = time.datetime_to_ISO8601(now_time)

    D.attrs['date_created'] = now_str

    return D




#### HELPER FUNCTIONS   

#### EXPORT

def to_netcdf(D: xr.Dataset, path: str, file_name: str = None, convention_check: bool = False, add_to_history: bool = True, verbose: bool = True):
    """
    Export xarray Dataset to netCDF.
    Using the 'id' attribute as file name if file_name not specified (if that doesn't exist, use 'CTD_DATASET_NO_NAME').
    """
    # Ensure path is a Path object
    path = Path(path)

    # Add current date as creation date
    D = add_now_as_date_created(D)
    D = conventionalize.reorder_attrs(D)

    # Determine file name
    if file_name is None:
        file_name = getattr(D, 'id', 'DATASET_NO_NAME')

    # Ensure file name ends with '.nc'
    if not file_name.endswith('.nc'):
        file_name += '.nc'

    # Construct file path
    file_path = path / file_name

    if add_to_history:
        # Add or update the history attribute
        if 'history' not in D.attrs:
            D.attrs['history'] = ''
        
        if 'Creation of this netcdf file' in D.attrs['history']:
            history_lines = D.attrs['history'].split('\n')
            updated_history = [line for line in history_lines if "Creation of this netcdf file" not in line]
            D.attrs['history'] = '\n'.join(updated_history)

        now_time = pd.Timestamp.now().strftime('%Y-%m-%d')
        D.attrs['history'] += f'\n{now_time}: Creation of this netcdf file.'
        
        if verbose:
            print(f'Updated history attribute. Current content:\n---')
            print(D.attrs['history'])
            print('---')

    # Save the dataset to NetCDF
    try:
        D.to_netcdf(file_path)
    
    except PermissionError:
        # Handle file overwrite if permission error occurs
        user_input = input(f"The file {file_path} already exists. Do you want to overwrite it? (y/n): ")
        if user_input.lower() in ['yes', 'y']:
            os.remove(file_path)
            D.to_netcdf(file_path)
            print(f"File {file_path} overwritten.")
        else:
            print("Operation canceled. File not overwritten.")

    if verbose:
        print(f'Exported netCDF file as: {file_path}')

    if convention_check:
        print('Running convention checker:')
        check_conventions.check_file(file_path)


def metadata_to_txt(D: xr.Dataset, outfile: str) -> None:
    """
    Write metadata information from an xarray.Dataset to a text file.

    Parameters:
    - D (xr.Dataset): The dataset containing metadata to be written.
    - outfile (str): Path for the output text file. The file extension will be appended if not provided.

    Returns:
    - None: Writes metadata to the specified text file.

    Example:
    >>> metadata_to_txt(D, 'metadata_output')
    """

    # Ensure the output file has a '.txt' extension
    if not outfile.lower().endswith('.txt'):
        outfile += '.txt'

    # Open the text file for writing
    with open(outfile, 'w') as f:
        # Create the file header based on the presence of 'id' attribute
        file_header = f'FILE METADATA FROM: {D.attrs.get("id", "Unknown")}'
        
        # Print the file header with formatting
        f.write('#' * 80 + '\n')
        f.write(f'####  {file_header:<68}  ####\n')
        f.write('#' * 80 + '\n')
        f.write('\n' + '#' * 27 + '\n')
        f.write('### GLOBAL ATTRIBUTES   ###\n')
        f.write('#' * 27 + '\n')
        f.write('\n')

        # Print global attributes
        for key, item in D.attrs.items():
            f.write(f'# {key}:\n')
            f.write(f'{item}\n')

        f.write('\n')
        f.write('#' * 27 + '\n')
        f.write('### VARIABLE ATTRIBUTES ###\n')
        f.write('#' * 27 + '\n')

        # Get all variable names (coordinates and data variables)
        all_vars = list(D.coords) + list(D.data_vars)

        # Iterate through variables
        for varnm in all_vars:
            f.write('\n' + '-' * 50 + '\n')

            # Print variable name with indication of coordinate status
            if varnm in D.coords:
                f.write(f'{varnm} (coordinate)\n')
            else:
                f.write(f'{varnm}\n')

            f.write('-' * 50 + '\n')

            # Print variable attributes
            for key, item in D[varnm].attrs.items():
                f.write(f'# {key}:\n')
                f.write(f'{item}\n')
