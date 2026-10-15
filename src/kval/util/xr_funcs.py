"""
XR_FUNCS.PY

Various generalized wrapper functions for working with xarray Datasets
"""

import warnings
import xarray as xr
import numpy as np
import pandas as pd
from xarray.coding.times import encode_cf_datetime
from pandas.tseries.frequencies import to_offset
from kval.util.netcdf import bounds_map, bounds_vars


def time_as_datetime(ds, time_dim='TIME'):
    """Ensure ds[time_dim] is decoded to datetime64.

    If it's already datetime64, returns ds unchanged (no-op). If it's raw
    CF-encoded numeric time (e.g. from load_moored, which loads with
    decode_cf=False), decodes it using its 'units'/'calendar' attributes.
    The original units/calendar are remembered in .encoding, so a later
    call to time_as_float can round-trip back to the same representation.

    Only variables with "<units> since <date>" units are decoded to datetimes.
    Variables whose units merely look like a duration ("seconds", "days", ...)
    are left exactly as they are. Note that xarray's CF decoding also replaces
    values equal to a variable's `_FillValue` by NaN in all variables, and
    `time_as_float` does not put them back.

    Args:
        ds (xr.Dataset): Dataset containing the time coordinate.
        time_dim (str): Name of the time coordinate. Defaults to 'TIME'.

    Returns:
        xr.Dataset: Dataset with time_dim as datetime64.

    Raises:
        TypeError: If time_dim is neither datetime64 nor numeric.
        ValueError: If time_dim is numeric but has no 'units' attribute.
    """
    ds = ds.copy(deep=True)
    if np.issubdtype(ds[time_dim].dtype, np.datetime64):
        return ds
    if not np.issubdtype(ds[time_dim].dtype, np.number):
        raise TypeError(
            f"'{time_dim}' is neither datetime64 nor numeric "
            f"(dtype={ds[time_dim].dtype}) -- cannot interpret as time."
        )
    units = ds[time_dim].attrs.get('units')
    if units is None:
        raise ValueError(
            f"'{time_dim}' is numeric but has no 'units' attribute, so "
            "it can't be decoded as CF time. Expected e.g. "
            "'days since 1970-01-01 00:00'."
        )
    calendar = ds[time_dim].attrs.get('calendar', 'standard')
    ds_decoded = xr.decode_cf(ds, decode_timedelta=False)
    ds_decoded[time_dim].encoding['units'] = units
    ds_decoded[time_dim].encoding['calendar'] = calendar
    return ds_decoded


def time_as_float(ds, time_dim='TIME', units=None, calendar=None):
    """Ensure ds[time_dim] is raw CF-encoded numeric time.

    If it's already numeric, returns ds unchanged (no-op). If it's decoded
    datetime64, encodes it back to numeric -- using units/calendar passed
    explicitly, or falling back to whatever time_as_datetime last
    remembered in .encoding, or (if neither is available) defaulting to
    'days since 1970-01-01 00:00' with a warning.

    Args:
        ds (xr.Dataset): Dataset containing the time coordinate.
        time_dim (str): Name of the time coordinate. Defaults to 'TIME'.
        units (str, optional): CF units to encode to, e.g.
            'days since 1970-01-01 00:00'. Overrides any remembered units.
        calendar (str, optional): CF calendar. Defaults to 'standard' if
            not remembered or passed explicitly.

    If time_dim has a CF boundary variable (named by its 'bounds' attribute,
    e.g. TIME_bnds), that is encoded too, with the same units and calendar
    (CF 7.1: boundary variables take their units from the parent variable).

    Returns:
        xr.Dataset: Dataset with time_dim as raw numeric CF time.
    """
    ds = ds.copy(deep=True)
    bnds = ds[time_dim].attrs.get('bounds')
    bnds_needs_encoding = (
        isinstance(bnds, str) and bnds in ds.variables
        and np.issubdtype(ds[bnds].dtype, np.datetime64))

    if np.issubdtype(ds[time_dim].dtype, np.number):
        if not bnds_needs_encoding:
            return ds
        # TIME is already numeric: encode the bounds in TIME's own units
        units = ds[time_dim].attrs.get('units')
        calendar = ds[time_dim].attrs.get('calendar', 'standard')
        if units is None:
            raise ValueError(
                f"'{time_dim}' is numeric but has no 'units' attribute, so "
                f"its bounds variable '{bnds}' cannot be encoded.")
    else:
        if units is None:
            units = ds[time_dim].encoding.get('units')
        if calendar is None:
            calendar = ds[time_dim].encoding.get('calendar', 'standard')
        if units is None:
            units = 'days since 1970-01-01 00:00'
            warnings.warn(
                f"No known original units for '{time_dim}' -- defaulting to "
                f"'{units}'. Pass units= explicitly to silence this."
            )
        num, units, calendar = encode_cf_datetime(
            ds[time_dim].values, units=units, calendar=calendar,
            dtype=np.float64
        )
        # (older xarray versions ignore dtype= and may return integers)
        num = np.asarray(num, dtype=np.float64)
        # Keep TIME's other attributes (standard_name, axis, bounds, ...)
        time_attrs = dict(ds[time_dim].attrs)
        ds = ds.assign_coords({time_dim: num})
        ds[time_dim].attrs.update(time_attrs)
        ds[time_dim].attrs['units'] = units
        ds[time_dim].attrs['calendar'] = calendar

    if bnds_needs_encoding:
        bnum, _, _ = encode_cf_datetime(
            ds[bnds].values, units=units, calendar=calendar, dtype=np.float64)
        bnum = np.asarray(bnum, dtype=np.float64)
        bnds_var = xr.Variable(ds[bnds].dims, bnum, attrs=dict(ds[bnds].attrs))
        if bnds in ds.coords:
            ds = ds.assign_coords({bnds: bnds_var})
        else:
            ds[bnds] = bnds_var
    return ds




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


def reorder_coords_to_end(ds, coord_names):
    """
    Reorder a dataset's coordinates so that the given coord_names appear
    last (in the order given), after all other coordinates and
    variables. Purely cosmetic (affects repr/coords iteration order
    only) -- no functional effect otherwise.

    Args:
        ds (xr.Dataset): Dataset to reorder.
        coord_names (str | list[str]): Coordinate name(s) to move to the
            end. Names not present in ds are silently skipped.

    Returns:
        xr.Dataset: The reordered dataset.
    """
    if isinstance(coord_names, str):
        coord_names = [coord_names]
    existing_order = list(ds.variables)
    reordered = (
        [v for v in existing_order if v not in coord_names]
        + [v for v in coord_names if v in existing_order]
    )
    return ds[reordered]


def promote_cf_coordinates(ds):
    """
    Promote all variables listed in any variable's 'coordinates' attribute
    to auxiliary coordinates, if present in the dataset. CF boundary
    variables (named by a 'bounds' attribute, e.g. TIME.bounds =
    'TIME_bnds') are made coordinates too, so they stay attached to the
    variable they describe. Newly-promoted coordinates are moved to the
    end (see reorder_coords_to_end) so they consistently appear after
    dimension coordinates like TIME, rather than wherever they happened to
    sit in the file's own variable order.
    """
    # collect all coordinate names mentioned in 'coordinates' attributes,
    # preserving first-seen order (not a plain set, which would give
    # LAT/LON etc. an arbitrary relative order among themselves)
    coord_names = dict()
    for var in ds.data_vars:
        coords_attr = ds[var].attrs.get("coordinates", "")
        for name in coords_attr.split():
            coord_names.setdefault(name, None)
    for bnds in bounds_map(ds):
        coord_names.setdefault(bnds, None)
    # xarray writes coordinates that share no dimensions with any data
    # variable (e.g. TIME_bnds, with its extra 'bnds' dimension) to a
    # *global* 'coordinates' attribute. Decoding with xarray removes that
    # attribute again; do the same here.
    global_coords = ds.attrs.get("coordinates")
    if isinstance(global_coords, str):
        for name in global_coords.split():
            coord_names.setdefault(name, None)
        ds = ds.copy()
        ds.attrs.pop("coordinates")

    # keep only existing variables that aren't already coords
    to_promote = [c for c in coord_names if c in ds and c not in ds.coords]

    if to_promote:
        ds = ds.set_coords(to_promote)
        ds = reorder_coords_to_end(ds, to_promote)

    return ds

# --- Helpers for describing averaging bins (used by time_average and
# kval.data.moored.combine_datasets) ---

def _bin_edges(tidx: pd.DatetimeIndex, freq, resample_kwargs: dict):
    """
    Start and end time of every resample bin, and which edge is closed.

    Returns (edges, closed): `edges` is a DataFrame with 'start' and 'end'
    columns, indexed by the bin's pandas label with label='left'; `closed`
    is 'left' if each bin is [start, end) and 'right' if (start, end].

    The edges are the *real* bin edges, which are not always the labels
    pandas gives the bins. For end-anchored calendar intervals (e.g. 'ME',
    'YE', 'W'), pandas bins whole calendar periods, but labels them with
    the day *before* the period starts (label='left') or the last day
    of the period at 00:00 (label='right'). The result is checked against
    where pandas actually put the samples, so a change in pandas' binning
    raises an error rather than giving wrong bounds.
    """
    kw = {k: v for k, v in resample_kwargs.items() if k != 'label'}
    positions = pd.Series(np.arange(len(tidx)), index=tidx)
    resampled = positions.resample(freq, label='left', **kw)
    left = resampled.count().index
    right = positions.resample(freq, label='right', **kw).count().index
    closed = kw.get('closed') or pd.Grouper(freq=freq).closed

    starts, ends = left, right
    name = to_offset(freq).name
    end_anchored = (name in ("BME", "ME", "W")
                    or name.split("-")[0] in ("BQE", "BYE", "QE", "YE", "W"))
    if closed == 'right' and end_anchored:
        # pandas moves these right-closed edges to the end of the day
        # (00:00 the next day, minus 1 ns), so the bins are effectively
        # [start, end) with both edges at midnight.
        starts = left + pd.Timedelta(days=1)
        ends = right + pd.Timedelta(days=1)
        closed = 'left'

    # Check against the samples pandas actually put in each bin
    first, last = resampled.min(), resampled.max()
    has_data = first.notna().values
    t_first = tidx[first[has_data].astype(int).values]
    t_last = tidx[last[has_data].astype(int).values]
    s, e = starts[has_data], ends[has_data]
    if closed == 'left':
        consistent = (t_first >= s) & (t_last < e)
    else:
        consistent = (t_first > s) & (t_last <= e)
    if not np.all(consistent):
        raise RuntimeError(
            f"Could not determine the averaging bin edges for interval "
            f"'{freq}': the samples pandas put in each bin do not match the "
            "expected edges. Please report this as a kval bug.")

    return pd.DataFrame({'start': starts, 'end': ends}, index=left), closed


def _describe_interval(interval, interval_td) -> str:
    """Interval in words, e.g. '1-day', '6-hour', 'calendar-month'."""
    if interval_td is not None:
        secs = interval_td.total_seconds()
        for unit, size in (("day", 86400), ("hour", 3600),
                           ("minute", 60), ("second", 1)):
            if secs >= size and secs % size == 0:
                return f"{int(secs // size)}-{unit}"
        return f"'{interval}'"
    try:
        off = to_offset(interval)
    except ValueError:
        return f"'{interval}'"
    units = {"MS": "month", "ME": "month", "QS": "quarter", "QE": "quarter",
             "YS": "year", "YE": "year", "W": "week"}
    unit = units.get(off.name.split("-")[0])
    if unit is None:
        return f"'{interval}'"
    return f"calendar-{unit}" if off.n == 1 else f"{off.n}-calendar-{unit}"


def _time_averaging_attr(label: str, interval_desc: str, closed: str,
                         bounds_name: str) -> str:
    """Plain-language description of what the averaged timestamps mean."""
    position = {'left': 'START', 'center': 'CENTRE', 'right': 'END'}[label]
    if closed == 'left':
        inclusion = ("Each bin includes samples from its start time up to, "
                     "but not including, its end time")
    else:
        inclusion = ("Each bin includes samples after its start time, up to "
                     "and including its end time")
    return (f"Time stamp at the {position} of each {interval_desc} averaging "
            f"bin. {inclusion}; bin start and end times are in "
            f"{bounds_name}.")


def _averaged_time_long_name(label: str) -> str:
    """long_name for the time coordinate of averaged data, e.g.
    'Time (centre of averaging bin)'."""
    position = {'left': 'start', 'center': 'centre', 'right': 'end'}[label]
    return f"Time ({position} of averaging bin)"


def _add_cell_method(existing, dim: str, method: str = 'mean') -> str:
    """
    Append e.g. 'TIME: mean' to a CF cell_methods string (CF 7.3: methods
    are listed in the order they were applied). Not repeated if it is
    already the last entry, e.g. when averaging data that were already
    averaged.
    """
    entry = f"{dim}: {method}"
    existing = (existing or "").strip()
    if not existing:
        return entry
    if existing == entry or existing.endswith(" " + entry):
        return existing
    return f"{existing} {entry}"


def time_average(
    ds: xr.Dataset,
    interval: str = '1D',
    label: str = 'center',
    origin: str | None = 'epoch',
    time_dim: str = 'TIME',
    min_fraction: float | None = None,     
    **resample_kwargs,
) -> xr.Dataset:
    """
    Average an xarray Dataset along a time dimension over a specified interval.

    Only numeric data variables with a `time_dim` dimension are averaged;
    non-numeric variables with a `time_dim` dimension (e.g. string variables
    like STATION) are dropped, since a mean is not well-defined for them.
    Variables without a `time_dim` dimension at all (e.g. ZONE(PRES)) are
    preserved unchanged and unbroadcast.

    The output records what the averaging did, following the CF conventions:

    - Bin edges: a coordinate `<time_dim>_bnds` (e.g. TIME_bnds) with
      dimensions (`time_dim`, 'bnds') holds the start and end time of each
      bin, and `time_dim` gets the attribute bounds='TIME_bnds' (CF 7.1).
      Each bin includes samples from its start time up to, but not
      including, its end time (unless `closed='right'` is passed on to
      resample). The bounds are stored in the same way as `time_dim`
      (datetime64, or numeric in the same units and calendar), and carry
      no attributes of their own: CF has them inherit those of `time_dim`.
    - Method: each averaged variable gets cell_methods='TIME: mean' (CF
      7.3), appended to any existing cell_methods.
    - In words: `time_dim` gets an attribute 'time_averaging', e.g. "Time
      stamp at the CENTRE of each 1-day averaging bin. ...", and its
      long_name is set to e.g. 'Time (centre of averaging bin)' (replacing
      e.g. 'Time of measurement').
    - Any existing bounds variable of `time_dim` (e.g. from averaging data
      that were already averaged) is replaced.
    - The global attribute 'time_coverage_resolution' (if any) is removed,
      since it described the input sampling; `add_range_attrs` sets it
      again from the averaged data.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with a time dimension coordinate.
    interval : str, default='1D'
        Averaging interval, as a pandas offset alias (e.g. '1D', '6h', '30min').
        Calendar-based intervals are also supported: 'MS' and 'ME' both give
        calendar months, 'YS'/'YE' calendar years, 'W' weeks (Monday to
        Sunday).
    label : {'center', 'left', 'right'}, default='center'
        Where to place the timestamp for each averaging bin:
        - 'center': the midpoint of the bin. For calendar-based intervals
          the midpoint of each bin is used, so e.g. monthly timestamps are
          not evenly spaced.
        - 'left': the start of the bin.
        - 'right': the end of the bin.
        The timestamps always follow from the bin edges in TIME_bnds. (For
        'ME', 'YE' and 'W', pandas itself labels a bin with the day before
        it starts or with the last day at 00:00; those labels are not used.)
    origin : str or None, default='epoch'
        Where to anchor the bin edges. The default 'epoch' (1970-01-01
        00:00) puts daily bins on midnight, 6-hourly bins on 00/06/12/18,
        and so on, and makes the bin edges independent of the data: two
        datasets averaged separately share the same edges at every
        fixed-duration interval, including multi-day ones like '7D'.
        Can be any value pandas accepts for `origin` (e.g. a timestamp such
        as '2020-01-01 03:00', or 'start_day'); None leaves it to pandas.
        Only used for fixed-duration intervals (e.g. '1D', '6h'); for
        calendar-based intervals (e.g. 'ME', 'YE') the bins follow the
        calendar and the default is not passed on.
    time_dim : str, default='TIME'
        Name of the time dimension/coordinate to resample along.
    min_fraction : float in (0, 1], optional
        Drop averaging bins that contain fewer than this fraction of the
        samples expected for a full bin (expected = interval divided by the
        median sampling interval; samples are counted by timestamp, whether
        or not their values are NaN). E.g. `min_fraction=0.9` removes the
        incomplete first/last bin and interior bins with >10 % missing
        samples. Only for fixed-duration intervals. Default None: keep all
        bins, but warn if the first or last bin is less than 95 % full,
        because its mean is then not representative of the full interval
        (and with label='center' it is stamped at the bin centre, not at the
        mean time of its samples).
    **resample_kwargs
        Additional keyword arguments passed through to xr.Dataset.resample
        (e.g. `closed`).

    Returns
    -------
    xr.Dataset
        Dataset averaged over the specified interval along `time_dim`.
        Non-numeric variables with a `time_dim` dimension are dropped (see
        printed message); variables without `time_dim` are preserved as-is.
        Includes the bin-edge coordinate `<time_dim>_bnds` (see above).

    Raises
    ------
    ValueError
        If `time_dim` is not a dimension in the dataset, or if `label` is
        not one of 'center', 'left', 'right'.
    """
    if time_dim not in ds.dims:
        raise ValueError(f"'{time_dim}' is not a dimension in the dataset")
    if label not in ('center', 'left', 'right'):
        raise ValueError("label must be one of 'center', 'left', or 'right'")

    # If TIME is still CF-encoded (raw numeric, e.g. from load_moored, which
    # loads with decode_cf=False), decode it so resample() has a real
    # DatetimeIndex to work with. We convert back to the original numeric
    # units/calendar before returning (via time_as_float below), so the
    # output matches whatever encoding state the input was in -- this
    # function shouldn't silently change that on the caller.
    #
    # We capture units/calendar explicitly here (rather than relying on
    # time_as_datetime's .encoding-based memory) because .encoding does
    # not survive resample()/merge() below -- it gets dropped, so passing
    # it through explicitly is the robust option for this particular
    # decode-then-recombine-then-encode pipeline.
    was_encoded = np.issubdtype(ds[time_dim].dtype, np.number)
    original_units = ds[time_dim].attrs.get('units') if was_encoded else None
    original_calendar = ds[time_dim].attrs.get('calendar', 'standard') if was_encoded else None
    ds = time_as_datetime(ds, time_dim)

    # TIME's own attributes are carried over to the output (minus the ones
    # set below). Any existing bounds of TIME (e.g. when averaging data that
    # were already averaged) would otherwise be averaged themselves, so they
    # are dropped here and rebuilt for the new bins.
    time_attrs = {k: v for k, v in ds[time_dim].attrs.items()
                  if k not in ('units', 'calendar', 'bounds', 'time_averaging')}
    old_bounds = [b for b, parent in bounds_map(ds).items() if parent == time_dim]
    if old_bounds:
        ds = ds.drop_vars(old_bounds)
    ds[time_dim].attrs.pop('bounds', None)

    # Resample with the bin start as label; the requested timestamps are
    # set from the bin edges afterwards (see _bin_edges for why pandas'
    # own labels are not used directly).
    resample_label = 'left'

    # --- How full is each bin? (fixed-duration intervals only) ---
    try:
        interval_td = pd.Timedelta(interval)
    except ValueError:
        interval_td = None
    if min_fraction is not None:
        if not (0 < min_fraction <= 1):
            raise ValueError("min_fraction must be in (0, 1]")
        if interval_td is None:
            raise ValueError(
                f"min_fraction needs a fixed-duration interval; '{interval}' "
                "is calendar-based.")

    # Fixed-duration intervals are handed to pandas as an exact number of
    # nanoseconds. In pandas >= 3 a '1D' (or '2D', ...) frequency is a
    # *calendar* day, and `origin=` is then silently ignored, so bins would no
    # longer line up with the origin that combine_datasets relies on (all-NaN
    # result). A fixed nanosecond frequency behaves the same in pandas 2 and 3.
    resample_freq = interval if interval_td is None else f"{interval_td.value}ns"

    resample_kwargs_full = dict(label=resample_label)
    # origin only applies to fixed-duration intervals. For calendar-based
    # ones pandas ignores it and warns, so don't forward the *default* there
    # (an origin the caller chose explicitly is still passed, warning and all).
    if origin is not None and (interval_td is not None or origin != 'epoch'):
        resample_kwargs_full['origin'] = origin
    resample_kwargs_full.update(resample_kwargs)

    tidx = pd.DatetimeIndex(ds[time_dim].values)
    bin_edges, closed = _bin_edges(tidx, resample_freq, resample_kwargs_full)

    bin_fraction = None
    if interval_td is not None and len(tidx) > 1:
        dt_med = pd.Series(tidx).diff().median()
        if pd.notna(dt_med) and dt_med > pd.Timedelta(0):
            expected = interval_td / dt_med
            bin_counts = pd.Series(1, index=tidx).resample(
                resample_freq, **resample_kwargs_full).sum()
            bin_fraction = bin_counts / expected
            if min_fraction is None:
                for which, pos in (('first', 0), ('last', -1)):
                    frac = bin_fraction.iloc[pos]
                    if frac < 0.95:
                        warnings.warn(
                            f"time_average: the {which} '{interval}' bin "
                            f"({bin_fraction.index[pos]}) contains only "
                            f"{int(bin_counts.iloc[pos])} of ~{expected:.0f} "
                            f"expected samples ({100 * frac:.0f} %). Its mean "
                            "is not representative of the full interval; use "
                            "min_fraction=... to drop incomplete bins.",
                            UserWarning, stacklevel=2)

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
        {time_dim: resample_freq}, **resample_kwargs_full
    ).mean()

    # Drop incomplete bins if requested
    if min_fraction is not None and bin_fraction is not None:
        frac_out = bin_fraction.reindex(ds_out[time_dim].values).fillna(0.0)
        keep = (frac_out >= min_fraction).values
        n_drop = int((~keep).sum())
        ds_out = ds_out.isel({time_dim: keep})
        if n_drop:
            print(f"time_average: dropped {n_drop} of {len(keep)} bin(s) with "
                  f"less than {100 * min_fraction:.0f} % of the expected samples.")

    # Re-merge the time-independent variables, preserved exactly as they were
    ds_out = ds_out.merge(ds_time_indep, compat="override")

    # --- Timestamps and bin edges (CF bounds) ---
    left_labels = pd.DatetimeIndex(ds_out[time_dim].values)
    starts = bin_edges['start'].reindex(left_labels).values
    ends = bin_edges['end'].reindex(left_labels).values
    if label == 'left':
        stamps = starts
    elif label == 'right':
        stamps = ends
    else:
        stamps = starts + (ends - starts) / 2

    bounds_name = f"{time_dim}_bnds"
    ds_out = ds_out.assign_coords({time_dim: stamps})
    ds_out = ds_out.assign_coords(
        {bounds_name: ((time_dim, 'bnds'), np.stack([starts, ends], axis=1))})
    ds_out[time_dim].attrs.update(time_attrs)
    ds_out[time_dim].attrs['long_name'] = _averaged_time_long_name(label)
    ds_out[time_dim].attrs['bounds'] = bounds_name
    ds_out[time_dim].attrs['time_averaging'] = _time_averaging_attr(
        label, _describe_interval(interval, interval_td), closed, bounds_name)

    for var in ds_numeric.data_vars:
        if var in ds_out.data_vars:
            ds_out[var].attrs['cell_methods'] = _add_cell_method(
                ds_out[var].attrs.get('cell_methods'), time_dim, 'mean')

    # A time_coverage_resolution global attribute (ACDD) describes the
    # sampling of the input (e.g. 15 min from an instrument file), not of
    # the averaged output. Remove it; add_range_attrs() sets it again from
    # the averaged TIME.
    ds_out.attrs.pop('time_coverage_resolution', None)

    if dropped_vars:
        print(f"time_average: dropped non-numeric {time_dim}-dependent "
              f"variable(s) {dropped_vars} (mean is not defined for "
              "non-numeric data)")

    if was_encoded:
        ds_out = time_as_float(ds_out, time_dim, units=original_units, calendar=original_calendar)

    return ds_out