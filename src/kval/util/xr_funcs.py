'''
XR_FUNCS.PY

Various generalized wrapper functions for working with xarray Datasets
'''

import xarray as xr

def rename_attr(D, old_name, new_name, verbose=True):
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
        if verbose:
            print(f"Renamed attribute '{old_name}' to '{new_name}'.")
    else:
        if verbose:
            print(f"Could not rename attribute '{old_name}' to '{new_name}'."
                  " (Original attribute not found)")


def add_attrs_from_dict(D, attr_dict, override=True):
    """
    Assign attributes to an xarray.Dataset from a dictionary.

    `attr_dict` should map {attr_key: attr_value}.

    Examples:
    - For global attributes of the Dataset, use 
      `add_attrs_from_dict(dataset, attr_dict)`.
    - For variable-specific attributes within the Dataset, 
      use `add_attrs_from_dict(dataset[var_name], attr_dict)`.

    Parameters:
    - dataset (xarray.Dataset): The dataset to which attributes will be added.
    - attr_dict (dict): Dictionary mapping attribute keys to their values.
    - override (bool, optional): If False, existing attributes will not be 
                                 overridden. Defaults to True.
    """

    for key, value in attr_dict.items():
        if key in D.attrs and not override:
            continue  # Skip if attribute exists and override is False
        else:
            D.attrs[key] = value  # Assign attribute to the dataset
