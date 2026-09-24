"""
Tests for kval.util.netcdf -- write-time CF encoding helpers.

The behaviour being pinned down here: xarray stamps `_FillValue = NaN` onto
every float variable it writes, coordinates included, even when the data
contain no NaNs. CF section 2.5.1 forbids _FillValue on coordinate variables,
so these helpers strip and suppress it at write time.
"""

import os

import numpy as np
import pytest
import xarray as xr
import netCDF4 as nc

from kval.util import netcdf


def _make_ds(coord_fill_value=None):
    """Small dataset with a float dimension coordinate and a scalar coordinate."""
    ds = xr.Dataset(
        {"TEMP": ("TIME", np.array([1.0, 2.0, np.nan], dtype="float32"))},
        coords={
            "TIME": ("TIME", np.array([0.0, 1.0, 2.0]),
                     {"units": "days since 1970-01-01"}),
            "LATITUDE": ((), np.float32(81.0)),
        },
    )
    if coord_fill_value is not None:
        ds["TIME"].attrs["_FillValue"] = coord_fill_value
    return ds


def _file_attrs(path, varname):
    with nc.Dataset(path) as f:
        var = f.variables[varname]
        return {a: var.getncattr(a) for a in var.ncattrs()}


# ---------------------------------------------------------------------
# strip_coord_fill_values
# ---------------------------------------------------------------------

def test_strip_removes_fill_value_from_coordinates():
    ds = _make_ds(coord_fill_value=-9999.0)
    out = netcdf.strip_coord_fill_values(ds)
    assert "_FillValue" not in out["TIME"].attrs


def test_strip_does_not_modify_the_input_dataset():
    ds = _make_ds(coord_fill_value=-9999.0)
    netcdf.strip_coord_fill_values(ds)
    assert ds["TIME"].attrs["_FillValue"] == -9999.0


def test_strip_leaves_data_variables_alone():
    ds = _make_ds()
    ds["TEMP"].attrs["_FillValue"] = np.float32(-9999.0)
    out = netcdf.strip_coord_fill_values(ds)
    assert out["TEMP"].attrs["_FillValue"] == np.float32(-9999.0)


def test_strip_is_a_no_op_when_there_is_nothing_to_strip():
    ds = _make_ds()
    out = netcdf.strip_coord_fill_values(ds)
    assert "_FillValue" not in out["TIME"].attrs
    assert out["TIME"].attrs["units"] == "days since 1970-01-01"


# ---------------------------------------------------------------------
# coord_fill_value_encoding
# ---------------------------------------------------------------------

def test_encoding_covers_every_coordinate_and_nothing_else():
    ds = _make_ds()
    enc = netcdf.coord_fill_value_encoding(ds)
    assert set(enc) == {"TIME", "LATITUDE"}
    assert all(v == {"_FillValue": None} for v in enc.values())


def test_encoding_keys_are_plain_strings():
    """to_netcdf() wants str keys, not numpy or Hashable objects."""
    ds = _make_ds()
    enc = netcdf.coord_fill_value_encoding(ds)
    assert all(isinstance(k, str) for k in enc)


# ---------------------------------------------------------------------
# Round trip: what actually lands in the file
# ---------------------------------------------------------------------

def test_written_file_has_no_fill_value_on_coordinates(tmp_path):
    ds = _make_ds()
    out, enc = netcdf.prepare_for_export(ds)
    path = os.path.join(tmp_path, "out.nc")
    out.to_netcdf(path, encoding=enc)

    assert "_FillValue" not in _file_attrs(path, "TIME")
    assert "_FillValue" not in _file_attrs(path, "LATITUDE")


def test_written_file_keeps_fill_value_on_data_variables(tmp_path):
    ds = _make_ds()
    out, enc = netcdf.prepare_for_export(ds)
    path = os.path.join(tmp_path, "out.nc")
    out.to_netcdf(path, encoding=enc)

    assert "_FillValue" in _file_attrs(path, "TEMP")


def test_fill_value_in_attrs_would_otherwise_survive_the_encoding(tmp_path):
    """
    Encoding alone is not enough: a _FillValue in .attrs wins and is written
    anyway. This is why prepare_for_export() strips as well as encodes.
    """
    ds = _make_ds(coord_fill_value=-9999.0)

    # Encoding only -- the attribute still makes it to file
    path_a = os.path.join(tmp_path, "encoding_only.nc")
    ds.to_netcdf(path_a, encoding=netcdf.coord_fill_value_encoding(ds))
    assert _file_attrs(path_a, "TIME")["_FillValue"] == -9999.0

    # Strip + encoding -- clean
    out, enc = netcdf.prepare_for_export(ds)
    path_b = os.path.join(tmp_path, "stripped.nc")
    out.to_netcdf(path_b, encoding=enc)
    assert "_FillValue" not in _file_attrs(path_b, "TIME")


def test_coordinate_values_are_unchanged_by_the_round_trip(tmp_path):
    ds = _make_ds()
    out, enc = netcdf.prepare_for_export(ds)
    path = os.path.join(tmp_path, "out.nc")
    out.to_netcdf(path, encoding=enc)

    back = xr.open_dataset(path, decode_cf=False)
    np.testing.assert_array_equal(back["TIME"].values, ds["TIME"].values)


# ---------------------------------------------------------------------
# Integration with the to_netcdf() path users actually call
# ---------------------------------------------------------------------

def test_dataset_to_netcdf_writes_clean_coordinates(tmp_path):
    from kval.data import dataset

    ds = _make_ds()
    ds.attrs["id"] = "test_dataset"
    dataset.to_netcdf(ds, str(tmp_path), "out.nc", verbose=False)

    path = os.path.join(tmp_path, "out.nc")
    assert "_FillValue" not in _file_attrs(path, "TIME")
    assert "_FillValue" not in _file_attrs(path, "LATITUDE")