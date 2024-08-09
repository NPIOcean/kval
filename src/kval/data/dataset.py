'''
DATASET.PY

Various functions to be applied to generalized datasets.
'''

from kval.util import time
import pandas as pd
import xarray as xr



#### MODIFY METADATA

def add_now_as_date_created(D):
    '''
    Add a global attribute "date_created" with todays date.
    '''
    now_time = pd.Timestamp.now()
    now_str = time.datetime_to_ISO8601(now_time)

    D.attrs['date_created'] = now_str

    return D




def reorder_attrs(D):
    """
    Reorder global and variable attributes of a dataset based on the 
    specified order in _standard_attrs.

    Parameters:
        ds (xarray.Dataset): The dataset containing global attributes.
        ordered_list (list): The desired order of global attributes.

    Returns:
        xarray.Dataset: The dataset with reordered global attributes.
    """
    ### GLOBAL
    reordered_list = _reorder_list(D.attrs, 
                                  _standard_attrs.global_attrs_ordered)
    attrs_dict = D.attrs
    D.attrs = {}
    for attr_name in reordered_list:
        D.attrs[attr_name] = attrs_dict[attr_name]

    ### VARIABLE
    for varnm in D.data_vars:
        reordered_list_var = _reorder_list(D[varnm].attrs, 
                      _standard_attrs.variable_attrs_ordered)
        var_attrs_dict = D[varnm].attrs
        D[varnm].attrs = {}
        for attr_name in reordered_list_var:
            D[varnm].attrs[attr_name] = var_attrs_dict[attr_name]
    return D



    #### HELPER FUNCTIONS   

    #### EXPORT


##### Export

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
