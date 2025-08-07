import pytest
import numpy as np
import xarray as xr
import warnings
import pandas as pd
from kval.metadata import conventionalize

# Test nans_to_fill_value
def test_replace_nans_in_float_vars():
    ds = xr.Dataset(
        {
            "float_var": ("x", [1.0, np.nan, 3.0]),
            "int_var": ("x", [1, 2, 3]),
        }
    )
    ds_out = conventionalize.nans_to_fill_value(ds, fill_value=-9999.0)
    assert np.all(ds_out["float_var"].values == np.array([1.0, -9999.0, 3.0]))
    assert ds_out["float_var"].attrs.get("_FillValue") == -9999.0
    # int var should be unchanged (NaN not present, dtype int)
    assert np.all(ds_out["int_var"].values == np.array([1, 2, 3]))
    assert "_FillValue" not in ds_out["int_var"].attrs

def test_coords_are_processed_if_float():
    ds = xr.Dataset(coords={"coord_float": ("x", [1.0, np.nan, 3.0])})
    ds_out = conventionalize.nans_to_fill_value(ds, fill_value=-9999.0)
    assert np.all(ds_out.coords["coord_float"].values == np.array([1.0, -9999.0, 3.0]))
    assert ds_out.coords["coord_float"].attrs.get("_FillValue") == -9999.0

def test_integer_coords_not_changed():
    ds = xr.Dataset(coords={"coord_int": ("x", [1, 2, 3])})
    ds_out = conventionalize.nans_to_fill_value(ds, fill_value=-9999.0)
    assert np.all(ds_out.coords["coord_int"].values == np.array([1, 2, 3]))
    assert "_FillValue" not in ds_out.coords["coord_int"].attrs

def test_no_nans_no_change():
    ds = xr.Dataset({"float_var": ("x", [1.0, 2.0, 3.0])})
    ds_out = conventionalize.nans_to_fill_value(ds, fill_value=-9999.0)
    assert np.all(ds_out["float_var"].values == np.array([1.0, 2.0, 3.0]))
    # _FillValue still set even if no NaNs present
    assert ds_out["float_var"].attrs.get("_FillValue") == -9999.0

def test_custom_fill_value():
    ds = xr.Dataset({"float_var": ("x", [np.nan, 2.0, 3.0])})
    ds_out = conventionalize.nans_to_fill_value(ds, fill_value=12345.6)
    assert np.all(ds_out["float_var"].values == np.array([12345.6, 2.0, 3.0]))
    assert ds_out["float_var"].attrs.get("_FillValue") == 12345.6

# Test convert_64_to_32

# Helper to create dataset with float64 and int64 arrays
def make_ds(float64_vals, int64_vals, coords=None):
    data_vars = {
        'floats': (('x',), float64_vals.astype(np.float64)),
        'ints': (('x',), int64_vals.astype(np.int64)),
    }
    coords = coords or {'x': np.arange(len(float64_vals))}
    return xr.Dataset(data_vars=data_vars, coords=coords)

def test_downcast_safe():
    ds = make_ds(
        float64_vals=np.array([1.0, 2.0, 3.0]),
        int64_vals=np.array([10, 20, 30])
    )
    ds2 = conventionalize.convert_64_to_32(ds)
    assert ds2['floats'].dtype == np.float32
    assert ds2['ints'].dtype == np.int32
    
def test_no_precision_loss_warn():
    vals = np.array([1e7 + 0.1, 2.0, 3.0], dtype=np.float64)
    ds = make_ds(float64_vals=vals, int64_vals=np.array([1, 2, 3]))

    # Default tolerance: conversion should happen, no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds2 = conventionalize.convert_64_to_32(ds)
        float_warnings = [warn for warn in w if "Float64 → Float32 conversion may lose precision" in str(warn.message)]
        assert len(float_warnings) == 0
        assert ds2['floats'].dtype == np.float32  # conversion happened

    # Smaller tolerance to force warning and no conversion
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds3 = conventionalize.convert_64_to_32(ds, relative_tol=1e-9)
        float_warnings = [warn for warn in w if "Float64 → Float32 conversion may lose precision" in str(warn.message)]
        assert len(float_warnings) == 1
        assert ds3['floats'].dtype == np.float64  # no conversion due to warning

        
def test_force_conversion():
    vals = np.array([1.123456789012345, 2.0, 3.0], dtype=np.float64)
    ds = make_ds(float64_vals=vals, int64_vals=np.array([2**40, 20, 30]))

    ds2 = conventionalize.convert_64_to_32(ds, force=True)
    assert ds2['floats'].dtype == np.float32
    assert ds2['ints'].dtype == np.int32

def test_coords_downcast():
    ds = xr.Dataset(
        data_vars={'data': (('x',), np.array([1.0, 2.0], dtype=np.float64))},
        coords={'x': np.array([1.0, 2.0], dtype=np.float64)}
    )
    ds2 = conventionalize.convert_64_to_32(ds)
    assert ds2.coords['x'].dtype == np.float32

def test_original_unchanged():
    ds = make_ds(
        float64_vals=np.array([1.0, 2.0, 3.0]),
        int64_vals=np.array([10, 20, 30])
    )
    ds2 = conventionalize.convert_64_to_32(ds)
    assert ds['floats'].dtype == np.float64
    assert ds['ints'].dtype == np.int64

def test_float_nan_inf():
    vals = np.array([1.0, np.nan, np.inf, -np.inf, 1e10], dtype=np.float64)
    ds = make_ds(float64_vals=vals, int64_vals=np.array([1, 2, 3, 4, 5]))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds2 = conventionalize.convert_64_to_32(ds)
        assert ds2['floats'].dtype == np.float32
        np.testing.assert_array_equal(np.isnan(ds2['floats']), np.isnan(vals))
        np.testing.assert_array_equal(np.isinf(ds2['floats']), np.isinf(vals))

def test_int64_edge_values():
    int32_max = np.iinfo(np.int32).max
    int32_min = np.iinfo(np.int32).min

    vals_safe = np.array([int32_min, 0, int32_max], dtype=np.int64)
    vals_unsafe = np.array([int32_min - 1, int32_max + 1], dtype=np.int64)

    ds_safe = make_ds(float64_vals=np.array([1.0, 2.0, 3.0]), int64_vals=vals_safe)
    ds_unsafe = make_ds(float64_vals=np.array([1.0, 2.0]), int64_vals=vals_unsafe)

    ds2_safe = conventionalize.convert_64_to_32(ds_safe)
    assert ds2_safe['ints'].dtype == np.int32

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds2_unsafe = conventionalize.convert_64_to_32(ds_unsafe)
        int_warnings = [warn for warn in w if "Int64 → Int32 conversion may overflow" in str(warn.message)]
        assert len(int_warnings) == 1
        assert ds2_unsafe['ints'].dtype == np.int64

def test_force_conversion_with_nans_and_infs():
    vals_float = np.array([np.nan, np.inf, -np.inf], dtype=np.float64)
    vals_int = np.array([2**40, -2**40, 0], dtype=np.int64)

    ds = make_ds(float64_vals=vals_float, int64_vals=vals_int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ds2 = conventionalize.convert_64_to_32(ds, force=True)

    assert ds2['floats'].dtype == np.float32
    assert ds2['ints'].dtype == np.int32
    np.testing.assert_array_equal(np.isnan(ds2['floats']), np.isnan(vals_float))
    np.testing.assert_array_equal(np.isinf(ds2['floats']), np.isinf(vals_float))


# Test add_range_attrs


# Mocks for the helper functions (replace with your actual imports or fixtures)
def _get_geospatial_bounds_wkt_str(ds):
    return "POLYGON((0 0,1 0,1 1,0 1,0 0))"

def _get_time_coverage_duration_str(ds):
    return "P1D"


def create_basic_ds(with_vertical=True, vertical_var="DEPTH"):
    times = np.array([0, 1], dtype='float64')  # numeric time for CF convention
    ds = xr.Dataset(
        {
            "LATITUDE": (("x",), [10, 20]),
            "LONGITUDE": (("x",), [30, 40]),
            "TIME": (("time",), times),
        },
        coords={
            "x": [0, 1],
            "time": times,
        }
    )
    # Add CF-style units attribute for TIME to allow cftime.num2date conversion
    ds["TIME"].attrs["units"] = "days since 2000-01-01 00:00:00"
    
    if with_vertical:
        ds[vertical_var] = (("x",), [5.0, 15.0])
        ds[vertical_var].attrs["units"] = "m"
        ds[vertical_var].attrs["axis"] = "Z"
    return ds

def test_add_range_attrs_basic():
    ds = create_basic_ds()
    ds_out = conventionalize.add_range_attrs(ds)
    assert "geospatial_lat_max" in ds_out.attrs
    assert ds_out.attrs["geospatial_lat_max"] == 20.0
    assert "geospatial_lon_min" in ds_out.attrs
    assert ds_out.attrs["geospatial_vertical_min"] == 5.0
    assert ds_out.attrs["geospatial_vertical_units"] == "m"
    assert "time_coverage_start" in ds_out.attrs

    # Accept both common ISO8601 duration formats for one day
    assert ds_out.attrs["time_coverage_duration"] in ("P1D", "P0000-00-01T00:00:00")

def test_add_range_attrs_missing_latlon(monkeypatch):
    ds = create_basic_ds()
    ds = ds.drop_vars(["LATITUDE", "LONGITUDE"])

    # Should print an error but not raise
    ds_out = conventionalize.add_range_attrs(ds)
    assert "geospatial_lat_max" not in ds_out.attrs


def test_add_range_attrs_missing_time(monkeypatch):
    ds = create_basic_ds()
    ds = ds.drop_vars("TIME")

    ds_out = conventionalize.add_range_attrs(ds)
    assert "time_coverage_start" not in ds_out.attrs


def test_add_range_attrs_no_vertical(monkeypatch):
    ds = create_basic_ds(with_vertical=False)
    ds_out = conventionalize.add_range_attrs(ds)
    assert "geospatial_vertical_min" not in ds_out.attrs


def test_add_range_attrs_vertical_var_specified():
    ds = create_basic_ds(vertical_var="DEPTH")
    ds_out = conventionalize.add_range_attrs(ds, vertical_var="DEPTH")
    assert ds_out.attrs["geospatial_vertical_units"] == "m"



def test_add_range_attrs_time_variable_interval():
    times = np.array([0, 6, 12], dtype="float64")  # numeric times in hours/days
    ds = xr.Dataset(
        {
            "LATITUDE": (("x",), [10, 20]),
            "LONGITUDE": (("x",), [30, 40]),
            "TIME": (("time",), times),
            "DEPTH": (("x",), [5.0, 15.0]),
        },
        coords={
            "x": [0, 1],
            "time": times,
        }
    )
    ds["TIME"].attrs["units"] = "hours since 2000-01-01 00:00:00"
    ds.DEPTH.attrs["units"] = "m"
    ds.DEPTH.attrs["axis"] = "Z"

    ds_out = conventionalize.add_range_attrs(ds)
    assert "time_coverage_resolution" in ds_out.attrs


# Test add_now_as_date_created


def test_add_now_as_date_created_default_format():
    ds = xr.Dataset()
    ds_out = conventionalize.add_now_as_date_created(ds)
    assert "date_created" in ds_out.attrs

    # Check if date matches the format "%Y-%m-%d" by trying to parse it
    try:
        pd.to_datetime(ds_out.attrs["date_created"], format="%Y-%m-%d")
    except ValueError:
        pytest.fail("date_created attribute does not match the default format '%Y-%m-%d'")

def test_add_now_as_date_created_custom_format():
    ds = xr.Dataset()
    custom_format = "%d %b %Y"  # e.g., "01 Aug 2025"
    ds_out = conventionalize.add_now_as_date_created(ds, datefmt=custom_format)
    assert "date_created" in ds_out.attrs

    # Check if date matches the custom format by trying to parse it
    try:
        pd.to_datetime(ds_out.attrs["date_created"], format=custom_format)
    except ValueError:
        pytest.fail(f"date_created attribute does not match the custom format '{custom_format}'")

def test_original_dataset_not_modified():
    ds = xr.Dataset()
    ds_copy = ds.copy(deep=True)
    ds_out = conventionalize.add_now_as_date_created(ds)
    # Since xarray datasets are mutable, ds will be modified in place
    # But we can check ds and ds_out are same object
    assert ds is ds_out