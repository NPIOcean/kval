"""
XR_FUNCS.PY

Various generalized wrapper functions for working with xarray Datasets
"""

import xarray as xr
import numpy as np


# INDEXING


def pick(ds, squeeze=True, **conditions):
    """
    Filter an xarray.Dataset based on conditions applied to its one-dimensional
    variables.

    This function is equivalent to `.isel()` but works with non-coordinate
    variables. For example, if we have a variable `STATION(TIME)`, we can
    select by station: `pick(ds, STATION='sta01')` or by multiple stations:
    `pick(ds, STATION=['sta01', 'sta02'])`.

    The function selects and returns the subset of the dataset where the
    specified condition(s) on the given variable(s) are met. The dimension
    along which the filtering occurs is determined dynamically based on the
    variable(s) provided in `conditions`.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset to be filtered.
    **conditions : dict
        Key-value pairs where the key is the name of a one-dimensional variable
        in the dataset, and the value is the condition. The condition can be a
        single value (e.g., STATION='sta02') or a list of values (e.g.,
        STATION=['sta02', 'sta03']).
    squeeze : bool, optional
        If True (default), the returned dataset will be squeezed to remove any
        single dimensions. If False, the original dimensions will be preserved.

    Returns
    -------
    xarray.Dataset
        A dataset filtered to only include the indices that match the
        condition(s). The dimension along which filtering is applied is
        inferred from the condition variable. The returned dataset may be
        squeezed depending on the `squeeze` parameter.

    Raises
    ------
    ValueError
        If the specified variable does not exist in the dataset or is not
        one-dimensional.

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
            raise ValueError(
                f"Variable '{var_name}' not found in the dataset."
            )

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


# DATA MANIPULATION


# ATTRIBUTE MANIPULATION


def rename_attr(ds, old_name, new_name, verbose=True):
    """
    Rename an attribute in an xarray Dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        The xarray Dataset containing attributes.
    old_name : str
        The current name of the attribute to be renameds.
    new_name : str
        The new name to assign to the attribute.
    explicit : bool, optional
        If True, print a message confirming the attribute rename (default is
        True).

    Notes:
    ------
    - For renaming global attributes of the Dataset, use `rename_attr(ds,
      old_name, new_name)`.
    - For renaming attributes of a specific variable within the Dataset, use
      `rename_attr(D[var_name], old_name, new_name)`.

    Example:
    --------
    Suppose ds is an xarray Dataset with global attributes: ds.attrs =
    {'units': 'meters', 'description': 'Sample dataset'}

    To rename 'units' to 'length': rename_attr(ds, 'units', 'length')

    """
    if old_name in ds.attrs:
        ds.attrs[new_name] = ds.attrs.pop(old_name)
        if verbose:
            print(f"Renamed attribute '{old_name}' to '{new_name}'.")
    else:
        if verbose:
            print(
                f"Could not rename attribute '{old_name}' to '{new_name}'."
                " (Original attribute not found)"
            )


def add_attrs_from_dict(ds, attr_dict, override=True):
    """
    Assign attributes to an xarray.Dataset from a dictionary.

    `attr_dict` should map {attr_key: attr_value}.

    Examples:
    - For global attributes of the Dataset, use
      `add_attrs_from_dict(dataset, attr_dict)`.
    - For variable-specific attributes within the Dataset,
      use `add_attrs_from_dict(dataset[var_name], attr_dict)`.

    Parameters:
    - dataset (xarray.Dataset): The dataset to which attributes will be addeds.
    - attr_dict (dict): Dictionary mapping attribute keys to their values.
    - override (bool, optional): If False, existing attributes will not be
                                 overridden. Defaults to True.
    """

    for key, value in attr_dict.items():
        if key in ds.attrs and not override:
            continue  # Skip if attribute exists and override is False
        else:
            ds.attrs[key] = value  # Assign attribute to the dataset


# STRUCTURE MANIPULATION


def swap_var_coord(
    ds: xr.Dataset, coordinate: str, variable: str, drop_original: bool = False
) -> xr.Dataset:
    """
    Swap a coordinate variable with a non-coordinate variable in an
    xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        The input xarray Dataset.
    coordinate : str
        The name of the variable currently used as a coordinate, which will
        become a non-coordinate variable.
    variable : str
        The name of the variable to be promoted to a coordinate and used as
        a dimension.
    drop_original : bool, optional
        If True, the original coordinate variable will be dropped from the
        Dataset.
        Default is False.

    Returns:
    --------
    xr.Dataset
        The modified Dataset with the specified coordinate and variable
        swapped.

    Raises:
    -------
    ValueError:
        If `coordinate` is not a coordinate in the Dataset.
        If `variable` is not a non-coordinate variable in the Dataset.
    """
    # Check that the coordinate is actually a coordinate variable
    if coordinate not in ds.coords:
        raise ValueError(f"'{coordinate}' is not a coordinate in the Dataset.")

    # Check that the variable is actually a non-coordinate variable
    if variable in ds.coords:
        raise ValueError(
            f"'{variable}' is already a coordinate in the Dataset."
        )

    # Set the variable as a coordinate
    ds = ds.set_coords(variable)

    # Swap the dimension from the original coordinate to the new variable
    ds = ds.swap_dims({coordinate: variable})

    # Reset the original coordinate to a non-coordinate variable or drop it
    if drop_original:
        ds = ds.drop_vars(coordinate)
    else:
        ds = ds.reset_coords(coordinate, drop=False)

    return ds




def promote_cf_coordinates(ds):
    """
    Promote all variables listed in any variable's 'coordinates' attribute
    to auxiliary coordinates, if present in the dataset.
    """
    # collect all coordinate names mentioned in 'coordinates' attributes
    coord_names = set()
    for var in ds.data_vars:
        coords_attr = ds[var].attrs.get("coordinates", "")
        coord_names.update(coords_attr.split())

    # keep only existing variables that aren't already coords
    to_promote = [c for c in coord_names if c in ds and c not in ds.coords]

    if to_promote:
        ds = ds.set_coords(to_promote)

    return ds
