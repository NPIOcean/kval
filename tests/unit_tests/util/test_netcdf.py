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

# ---------------------------------------------------------------------
# CF boundary variables (e.g. TIME_bnds)
# ---------------------------------------------------------------------

def _ds_with_bounds(as_coord=True):
    ds = xr.Dataset(
        {"TEMP": ("TIME", np.array([1.0, 2.0]))},
        coords={"TIME": ("TIME", np.array([0.5, 1.5]),
                         {"units": "days since 1970-01-01", "bounds": "TIME_bnds"})})
    bnds = (("TIME", "bnds"), np.array([[0.0, 1.0], [1.0, 2.0]]),
            {"_FillValue": np.nan})
    if as_coord:
        ds = ds.assign_coords(TIME_bnds=bnds)
    else:
        ds["TIME_bnds"] = bnds
    return ds


def test_bounds_map_finds_bounds_from_the_bounds_attribute():
    ds = _ds_with_bounds()
    assert netcdf.bounds_map(ds) == {"TIME_bnds": "TIME"}
    assert netcdf.bounds_vars(ds) == {"TIME_bnds"}
    ds["TIME"].attrs["bounds"] = "NOT_THERE"
    assert netcdf.bounds_map(ds) == {}


@pytest.mark.parametrize("as_coord", [True, False])
def test_prepare_for_export_removes_fill_value_from_bounds(as_coord, tmp_path):
    ds, encoding = netcdf.prepare_for_export(_ds_with_bounds(as_coord))
    assert "_FillValue" not in ds["TIME_bnds"].attrs
    assert encoding["TIME_bnds"] == {"_FillValue": None}
    path = tmp_path / "bnds.nc"
    ds.to_netcdf(path, encoding=encoding)
    with nc.Dataset(path) as f:
        assert "_FillValue" not in f["TIME_bnds"].ncattrs()
        assert "_FillValue" in f["TEMP"].ncattrs()  # data still get one


def test_datetime_time_and_bounds_are_written_with_the_same_units(tmp_path):
    """Left to itself xarray picks separate units for TIME and TIME_bnds
    (e.g. 'days since <first value>'), giving the bounds a 'units' attribute
    that differs from TIME's (not allowed by CF 7.1)."""
    import pandas as pd
    t0 = pd.date_range("2021-11-09", periods=3)
    ds = xr.Dataset({"TEMP": ("TIME", np.array([1.0, 2.0, 3.0]))},
                    coords={"TIME": t0 + pd.Timedelta("12h")})
    ds = ds.assign_coords(TIME_bnds=(("TIME", "bnds"),
                                     np.stack([t0, t0 + pd.Timedelta("1D")], 1)))
    ds["TIME"].attrs["bounds"] = "TIME_bnds"

    ds_out, encoding = netcdf.prepare_for_export(ds)
    path = tmp_path / "dt_bnds.nc"
    ds_out.to_netcdf(path, encoding=encoding)
    with nc.Dataset(path) as f:
        assert f["TIME"].units == "days since 1970-01-01"
        assert f["TIME_bnds"].ncattrs() == []  # takes TIME's units
        assert f["TIME"].dtype == np.float64
        np.testing.assert_array_equal(f["TIME_bnds"][0], [18940.0, 18941.0])
    re = xr.open_dataset(path)
    np.testing.assert_array_equal(re.TIME_bnds.values, ds.TIME_bnds.values)
    re.close()
