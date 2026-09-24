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

import xarray as xr


def strip_coord_fill_values(ds: xr.Dataset) -> xr.Dataset:
    """
    Remove any '_FillValue' attribute from coordinate variables.

    CF section 2.5.1: a coordinate variable defines an axis, so every element
    must have a value, and _FillValue is not allowed. A _FillValue sitting in
    a coordinate's `.attrs` is written to file regardless of any encoding we
    pass, so it has to be removed from the dataset itself.

    This most often shows up on data read back from a file with
    `decode_cf=False`, where `_FillValue` stays in `.attrs` rather than being
    moved into `.encoding`.

    Parameters
    ----------
    ds : xr.Dataset

    Returns
    -------
    xr.Dataset
        A copy with '_FillValue' removed from all coordinates. The input
        dataset is not modified.
    """
    ds = ds.copy()
    for coord in ds.coords:
        ds[coord].attrs.pop("_FillValue", None)
    return ds


def coord_fill_value_encoding(ds: xr.Dataset) -> dict:
    """
    Build a `to_netcdf()` encoding dict suppressing '_FillValue' on all
    coordinate variables.

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
        ``{coord_name: {"_FillValue": None}}`` for every coordinate, suitable
        for passing as the `encoding` argument of `Dataset.to_netcdf()`.
    """
    return {str(coord): {"_FillValue": None} for coord in ds.coords}


def prepare_for_export(ds: xr.Dataset) -> tuple:
    """
    Convenience wrapper: apply `strip_coord_fill_values()` and build the
    matching encoding dict in one step.

    Returns
    -------
    (xr.Dataset, dict)
        The dataset to write, and the `encoding` dict to write it with, e.g.:

        >>> ds_out, encoding = prepare_for_export(ds)
        >>> ds_out.to_netcdf(path, encoding=encoding)
    """
    ds = strip_coord_fill_values(ds)
    return ds, coord_fill_value_encoding(ds)