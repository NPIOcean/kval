'''
XR_FUNCS.PY

Various generalized wrapper functions for working with xarray Datasets
'''

import xarray as xr
import numpy as np



###### INDEXING
def pick(ds, squeeze=True, **conditions):
    """
    Filter an xarray.Dataset based on conditions applied to its one-dimensional variables.

    This function is equivalent to `.isel()` but works with non-coordinate variables.
    For example, if we have a variable `STATION(TIME)`, we can select by station: 
    `pick(ds, STATION='sta01')` or by multiple stations: `pick(ds, STATION=['sta01', 'sta02'])`.

    The function selects and returns the subset of the dataset where the specified 
    condition(s) on the given variable(s) are met. The dimension along which the filtering 
    occurs is determined dynamically based on the variable(s) provided in `conditions`.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset to be filtered.
    **conditions : dict
        Key-value pairs where the key is the name of a one-dimensional variable in the 
        dataset, and the value is the condition. The condition can be a single value 
        (e.g., STATION='sta02') or a list of values (e.g., STATION=['sta02', 'sta03']).
    squeeze : bool, optional
        If True (default), the returned dataset will be squeezed to remove any singleton 
        dimensions. If False, the original dimensions will be preserved.

    Returns
    -------
    xarray.Dataset
        A dataset filtered to only include the indices that match the condition(s). The 
        dimension along which filtering is applied is inferred from the condition variable.
        The returned dataset may be squeezed depending on the `squeeze` parameter.

    Raises
    ------
    ValueError
        If the specified variable does not exist in the dataset or is not one-dimensional.

    Examples
    --------
    >>> ds = xr.Dataset(
    ...     {
    ...         'TEMP': (['TIME', 'PRES'], temp_data),
    ...         'OCEAN': (['TIME'], ocean_data),
    ...         'STATION': (['TIME'], station_data)
    ...     },
    ...     coords={
    ...         'TIME': time,
    ...         'PRES': pres
    ...     }
    ... )
    >>> pick(ds, STATION='st02')
    <xarray.Dataset>
    Dimensions:  (TIME: 1, PRES: 5)
    Coordinates:
      * TIME     (TIME) datetime64[ns] 2024-01-02
      * PRES     (PRES) float64 1e+03 875.0 750.0 625.0 500.0
    Data variables:
        TEMP     (TIME, PRES) float64 14.5 15.3 12.7 17.6 8.67
        OCEAN    (TIME) <U13 'Arctic'
        STATION  (TIME) <U3 'st02'

    >>> pick(ds, STATION=['st02', 'st03'])
    <xarray.Dataset>
    Dimensions:  (TIME: 2, PRES: 5)
    Coordinates:
      * TIME     (TIME) datetime64[ns] 2024-01-02 2024-01-03
      * PRES     (PRES) float64 1e+03 875.0 750.0 625.0 500.0
    Data variables:
        TEMP     (TIME, PRES) float64 14.5 15.3 12.7 17.6 8.67 16.8 12.4 ...
        OCEAN    (TIME) <U13 'Arctic' 'Pacific'
        STATION  (TIME) <U3 'st02' 'st03'
    """

    # Iterate over the conditions
    for var_name, value in conditions.items():
        # Check if the variable exists in the dataset
        if var_name not in ds:
            raise ValueError(f"Variable '{var_name}' not found in the dataset.")
        
        # Find the dimension that the variable depends on
        var_dims = ds[var_name].dims
        
        # Ensure the variable is one-dimensional
        if len(var_dims) != 1:
            raise ValueError(f"Variable '{var_name}' must be one-dimensional.")
        
        dim = var_dims[0]  # Get the dimension name
        
        # Handle cases where 'value' is a list or array of values
        if isinstance(value, (list, np.ndarray)):
            indices = ds[var_name].isin(value)
        else:
            indices = ds[var_name] == value
        
        # Filter the dataset using the indices
        ds = ds.isel({dim: indices})
    
    if squeeze:
        ds = ds.squeeze()

    return ds



###### DATA MANIPULATION


###### ATTRIBUTE MANIPULATION

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


