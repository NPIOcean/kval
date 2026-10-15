"""
KVAL.UTIL.NETCDF

Small helpers for writing CF-compliant NetCDF files.

These deal with the difference between what a dataset looks like in memory
and what ends up on disk. `xarray` makes some encoding choices of its own
when writing, and not all of them are CF-compliant; the functions here
correct for that at write time.

Intended for internal use by the various `to_netcdf` paths in `kval`, rather
than by users directly.
"""

import numpy as np
import xarray as xr


def bounds_map(ds: xr.Dataset) -> dict:
    """
    Map each CF boundary variable in `ds` to the variable it belongs to.

    A boundary variable (CF section 7.1, e.g. TIME_bnds) is found through
    the 'bounds' attribute of its parent variable (e.g. TIME.bounds =
    'TIME_bnds'), not by its name. Boundary variables need special handling
    in several places: they take their units and calendar from the parent,
    should carry no attributes of their own, and must never get a
    _FillValue.

    Parameters
    ----------
    ds : xr.Dataset

    Returns
    -------
    dict
        ``{boundary_variable_name: parent_variable_name}`` for every
        'bounds' attribute that names a variable present in `ds`.
    """
    out = {}
    for name in list(ds.coords) + list(ds.data_vars):
        bnds = ds[name].attrs.get("bounds")
        if isinstance(bnds, str) and bnds in ds.variables:
            out[bnds] = name
    return out


def bounds_vars(ds: xr.Dataset) -> set:
    """
    Names of the CF boundary variables in `ds` (see `bounds_map`).
    """
    return set(bounds_map(ds))


def strip_coord_fill_values(ds: xr.Dataset) -> xr.Dataset:
    """
    Remove any '_FillValue' attribute from coordinate variables and CF
    boundary variables (e.g. TIME_bnds).

    CF section 2.5.1: a coordinate variable defines an axis, so every element
    must have a value, and _FillValue is not allowed. The same goes for
    boundary variables (CF section 7.1), which may also be stored as data
    variables. A _FillValue sitting in a coordinate's `.attrs` is written to
    file regardless of any encoding we pass, so it has to be removed from the
    dataset itself.

    This most often shows up on data read back from a file with
    `decode_cf=False`, where `_FillValue` stays in `.attrs` rather than being
    moved into `.encoding`.

    Parameters
    ----------
    ds : xr.Dataset

    Returns
    -------
    xr.Dataset
        A copy with '_FillValue' removed from all coordinates and boundary
        variables. The input dataset is not modified.
    """
    ds = ds.copy()
    for name in _no_fill_value_vars(ds):
        ds[name].attrs.pop("_FillValue", None)
    return ds


def coord_fill_value_encoding(ds: xr.Dataset) -> dict:
    """
    Build a `to_netcdf()` encoding dict suppressing '_FillValue' on all
    coordinate variables and CF boundary variables.

    `xarray` writes `_FillValue = NaN` onto every float variable by default,
    coordinates included, and does so even when the data contain no NaNs at
    all. Passing ``{"_FillValue": None}`` tells it to write no _FillValue for
    that variable (which is not the same as leaving the key out).

    This governs *encoding* only. Apply `strip_coord_fill_values()` to the
    dataset first, or a _FillValue in `.attrs` will be written anyway.

    Parameters
    ----------
    ds : xr.Dataset

    Returns
    -------
    dict
        ``{name: {"_FillValue": None}}`` for every coordinate and boundary
        variable, suitable for passing as the `encoding` argument of
        `Dataset.to_netcdf()`.
    """
    return {str(name): {"_FillValue": None} for name in _no_fill_value_vars(ds)}


def _no_fill_value_vars(ds: xr.Dataset) -> list:
    """Coordinates plus boundary variables (which may be data variables)."""
    names = list(ds.coords)
    names += [b for b in sorted(bounds_vars(ds)) if b not in ds.coords]
    return names


def time_bounds_encoding(ds: xr.Dataset) -> dict:
    """
    Build a `to_netcdf()` encoding dict that writes a datetime64 time
    variable and its datetime64 bounds (e.g. TIME and TIME_bnds) with the
    same units and calendar.

    CF section 7.1: a boundary variable takes its units from its parent and
    should not carry a different 'units' attribute. Left to itself, xarray
    picks the units for each datetime64 variable separately (e.g. "days
    since <first value>"), so TIME and TIME_bnds end up with different
    units. Here both get the parent's encoding units if it has any, and
    otherwise 'days since 1970-01-01' (as float64). Numeric time variables
    are already written as they are and need nothing here.

    Parameters
    ----------
    ds : xr.Dataset

    Returns
    -------
    dict
        ``{name: {"units": ..., "calendar": ..., "dtype": "float64"}}`` for
        each such time variable and its bounds (empty if there are none).
    """
    encoding = {}
    for bnds, parent in bounds_map(ds).items():
        if (np.issubdtype(ds[parent].dtype, np.datetime64)
                and np.issubdtype(ds[bnds].dtype, np.datetime64)):
            units = ds[parent].encoding.get("units", "days since 1970-01-01")
            calendar = ds[parent].encoding.get("calendar", "standard")
            for name in (parent, bnds):
                encoding[str(name)] = {"units": units, "calendar": calendar,
                                       "dtype": "float64"}
    return encoding


def prepare_for_export(ds: xr.Dataset) -> tuple:
    """
    Convenience wrapper: apply `strip_coord_fill_values()` and build the
    matching encoding dict in one step (including `time_bounds_encoding()`).

    Returns
    -------
    (xr.Dataset, dict)
        The dataset to write, and the `encoding` dict to write it with, e.g.:

        >>> ds_out, encoding = prepare_for_export(ds)
        >>> ds_out.to_netcdf(path, encoding=encoding)
    """
    ds = strip_coord_fill_values(ds)
    encoding = coord_fill_value_encoding(ds)
    for name, enc in time_bounds_encoding(ds).items():
        encoding.setdefault(name, {}).update(enc)
    return ds, encoding