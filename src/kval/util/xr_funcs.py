"""
XR_FUNCS.PY

Various generalized wrapper functions for working with xarray Datasets
"""

import xarray as xr
import numpy as np
import pandas as pd
from xarray.coding.times import encode_cf_datetime

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
    # A single match squeezes away the TIME dimension entirely by default
    # (squeeze=True), since it has length 1; TIME becomes a scalar coordinate
    # rather than a dimension. Use squeeze=False to keep it as TIME: 1.
    <xarray.Dataset>
    Dimensions:  (PRES: 5)
    Coordinates:
        TIME     datetime64[ns] 2024-01-02
      * PRES     (PRES) float64 1e+03 875.0 750.0 625.0 500.0
    Data variables:
        TEMP     (PRES) float64 14.5 15.3 12.7 17.6 8.67
        OCEAN    <U13 'Arctic'
        STATION  <U3 'st02'

    >>> pick(ds, STATION=['st02', 'st03'])
    # Multiple matches keep TIME as a real (length > 1) dimension, so
    # squeezing has no effect here.
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
      `rename_attr(ds[var_name], old_name, new_name)`.

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
      use `add_attrs_from_dict(ds[var_name], attr_dict)`.

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


def append_processing_history(ds, variable, note, key="processing_history", deep_copy=True):
    """
    Append a note to a variable's processing-history attribute, creating it
    if it doesn't already exist. Used by editing functions (offset,
    threshold, filters, drift corrections, etc.) to build up a plain-text,
    human-readable record of what's been done to a variable.

    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset containing the variable.
    variable : str
        Name of the variable within `ds` to attach the note to.
    note : str
        Description of the operation performed, written to read naturally
        when appended after any existing notes (e.g. a full sentence).
    key : str, optional
        The attribute name to use. Default is 'processing_history'.
    deep_copy : bool, optional
        If True (default), returns a fresh deep copy of `ds`, leaving the
        dataset passed in untouched. Set to False when calling from inside
        a function that has already made its own deep copy (e.g. most
        kval editing functions), to avoid copying the same dataset twice.

    Returns:
    --------
    xarray.Dataset
        `ds` (or a deep copy of it, if deep_copy=True), with the note
        appended to `ds[variable].attrs[key]`.

    Raises:
    -------
    ValueError
        If `variable` is not found in `ds`.

    Example:
    --------
    ds = append_processing_history(ds, 'TEMP', 'Applied offset of 5.2 degC.')
    """
    if variable not in ds:
        raise ValueError(f"Variable '{variable}' not found in the Dataset.")

    if deep_copy:
        ds = ds.copy(deep=True)
    existing = ds[variable].attrs.get(key)
    ds[variable].attrs[key] = f"{existing} {note}" if existing else note
    return ds


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

def time_average(
    ds: xr.Dataset,
    interval: str = '1D',
    label: str = 'center',
    origin: str | None = None,
    time_dim: str = 'TIME',
    **resample_kwargs,
) -> xr.Dataset:
    """
    Average an xarray Dataset along a time dimension over a specified interval.

    Only numeric data variables with a `time_dim` dimension are averaged;
    non-numeric variables with a `time_dim` dimension (e.g. string variables
    like STATION) are dropped, since a mean is not well-defined for them.
    Variables without a `time_dim` dimension at all (e.g. ZONE(PRES)) are
    preserved unchanged and unbroadcast.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with a time dimension coordinate.
    interval : str, default='1D'
        Averaging interval, as a pandas offset alias (e.g. '1D', '6h', '30min').
    label : {'center', 'left', 'right'}, default='center'
        Where to place the timestamp for each averaging bin:
        - 'center': the midpoint of the bin (only valid for fixed-duration
          intervals; raises an error for calendar-based intervals like 'ME'
          or 'YE', since those don't have a fixed duration to center within).
        - 'left': the start of the bin (xarray's default resample behavior).
        - 'right': the end of the bin.
    origin : str, optional
        A timestamp (e.g. '2020-01-01 00:00') at which to anchor the bin
        edges, if you want bins to start at a specific time rather than the
        default alignment. Passed through to xr.Dataset.resample. Only has
        an effect for fixed-frequency ("Tick-like") intervals (e.g. '1D',
        '6h'); it is silently ignored by xarray for calendar-based intervals
        (e.g. 'ME', 'YE').
    time_dim : str, default='TIME'
        Name of the time dimension/coordinate to resample along.
    **resample_kwargs
        Additional keyword arguments passed through to xr.Dataset.resample
        (e.g. `closed`).

    Returns
    -------
    xr.Dataset
        Dataset averaged over the specified interval along `time_dim`.
        Non-numeric variables with a `time_dim` dimension are dropped (see
        printed message); variables without `time_dim` are preserved as-is.

    Raises
    ------
    ValueError
        If `time_dim` is not a dimension in the dataset, if `label` is not
        one of 'center', 'left', 'right', or if label='center' is used with
        a calendar-based (non-fixed-duration) interval.
    """
    if time_dim not in ds.dims:
        raise ValueError(f"'{time_dim}' is not a dimension in the dataset")
    if label not in ('center', 'left', 'right'):
        raise ValueError("label must be one of 'center', 'left', or 'right'")


    was_encoded = np.issubdtype(ds[time_dim].dtype, np.number)
    if was_encoded:
        original_units = ds[time_dim].attrs.get('units')
        original_calendar = ds[time_dim].attrs.get('calendar', 'standard')
        if original_units is None:
            raise ValueError(
                f"'{time_dim}' is numeric but has no 'units' attribute, so "
                "it can't be decoded as CF time. Expected e.g. "
                "'days since 1970-01-01'."
            )
        ds = xr.decode_cf(ds, decode_timedelta=True)


    # xarray's resample only natively supports 'left'/'right' labeling;
    # for 'center' we resample as 'left' and shift the result afterward
    resample_label = 'left' if label == 'center' else label

    resample_kwargs_full = dict(label=resample_label)
    if origin is not None:
        resample_kwargs_full['origin'] = origin
    resample_kwargs_full.update(resample_kwargs)

    # Split off variables that don't depend on time_dim at all -- xarray's
    # resample().mean() otherwise broadcasts them across the new time bins
    # (e.g. ZONE(PRES) becomes ZONE(TIME, PRES)), which is not desired.
    time_indep_vars = [v for v in ds.data_vars if time_dim not in ds[v].dims]
    ds_time_indep = ds[time_indep_vars]

    # Non-numeric variables with a time_dim dimension can't be meaningfully
    # averaged (e.g. STATION strings) -- drop them and say so.
    dropped_vars = [
        v for v in ds.data_vars
        if time_dim in ds[v].dims and not np.issubdtype(ds[v].dtype, np.number)
    ]
    ds_numeric = ds.drop_vars(dropped_vars + time_indep_vars)

    ds_out = ds_numeric.resample(
        {time_dim: interval}, **resample_kwargs_full
    ).mean()

    # Re-merge the time-independent variables, preserved exactly as they were
    ds_out = ds_out.merge(ds_time_indep)

    if label == 'center':
        try:
            offset = pd.Timedelta(interval) / 2
        except ValueError as e:
            raise ValueError(
                f"Could not center timestamps for interval '{interval}': "
                f"{e}. Calendar-based intervals (e.g. months, years) don't "
                "have a fixed duration, so centering is not well-defined -- "
                "use label='left' or label='right' instead."
            )
        ds_out = ds_out.assign_coords(
            {time_dim: ds_out[time_dim].values + offset}
        )

    if dropped_vars:
        print(f"time_average: dropped non-numeric {time_dim}-dependent "
              f"variable(s) {dropped_vars} (mean is not defined for "
              "non-numeric data)")


    if was_encoded:
        num, units, calendar = encode_cf_datetime(
            ds_out[time_dim].values, units=original_units, calendar=original_calendar
        )
        ds_out = ds_out.assign_coords({time_dim: num})
        ds_out[time_dim].attrs['units'] = units
        ds_out[time_dim].attrs['calendar'] = calendar

    return ds_out