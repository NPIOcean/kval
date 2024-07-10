'''
XR_FUNCS.PY

Various generalized wrapper functions for working with xarray Datasets
'''

import xarray as xr

def rename_attr(D, old_name, new_name, explicit=True):
    """
    Rename an attribute in an xarray Dataset.

    Parameters:
    -----------
    D : xarray.Dataset
        The xarray Dataset containing attributes.
    old_name : str
        The current name of the attribute to be renamed.
    new_name : str
        The new name to assign to the attribute.
    explicit : bool, optional
        If True, print a message confirming the attribute rename (default is True).

    Notes:
    ------
    - For renaming global attributes of the Dataset, use `rename_attr(D, old_name, new_name)`.
    - For renaming attributes of a specific variable within the Dataset, use `rename_attr(D[var_name], old_name, new_name)`.

    Example:
    --------
    Suppose D is an xarray Dataset with global attributes:
    D.attrs = {'units': 'meters', 'description': 'Sample dataset'}

    To rename 'units' to 'length':
    rename_attr(D, 'units', 'length')

    """
    if old_name in D.attrs:
        D.attrs[new_name] = D.attrs.pop(old_name)
        if explicit:
            print(f"Renamed attribute '{old_name}' to '{new_name}'.")
    else:
        if explicit:
            print(f"Could not rename attribute '{old_name}' to '{new_name}'."
                  " (Original attribute not found)")



