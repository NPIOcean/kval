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

def test_coords_are_not_processed():
    """Coordinates are deliberately left alone: CF 2.5.1 forbids _FillValue on
    coordinate variables, and filling a dimension coordinate would break
    monotonicity."""
    ds = xr.Dataset(coords={"coord_float": ("x", [1.0, np.nan, 3.0])})
    ds_out = conventionalize.nans_to_fill_value(ds, fill_value=-9999.0)

    assert np.isnan(ds_out.coords["coord_float"].values[1])
    assert "_FillValue" not in ds_out.coords["coord_float"].attrs

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

# --- Tests for add_standard_glob_attrs_org ---

def test_add_standard_glob_attrs_org_default_none_does_nothing():
    """No org specified -> no organization-specific attributes added."""
    ds = xr.Dataset(attrs={'title': 'test'})
    ds_out = conventionalize.add_standard_glob_attrs_org(ds)
    assert 'institution' not in ds_out.attrs
    assert ds_out.attrs == {'title': 'test'}


def test_add_standard_glob_attrs_org_npi():
    """Explicit org='npi' adds the expected NPI attributes."""
    ds = xr.Dataset()
    ds_out = conventionalize.add_standard_glob_attrs_org(ds, org='npi')
    assert ds_out.attrs['institution'] == 'Norwegian Polar Institute (NPI)'
    assert ds_out.attrs['creator_email'] == 'post@npolar.no'
    assert ds_out.attrs['publisher_name'] == 'Norwegian Polar Institute (NPI)'


def test_add_standard_glob_attrs_org_case_insensitive():
    """Org name matching should be case-insensitive."""
    ds = xr.Dataset()
    ds_out = conventionalize.add_standard_glob_attrs_org(ds, org='NPI')
    assert ds_out.attrs['institution'] == 'Norwegian Polar Institute (NPI)'


def test_add_standard_glob_attrs_org_unknown_raises_clear_error():
    """An unrecognized org name should raise ValueError listing what's
    actually available, not a bare KeyError."""
    ds = xr.Dataset()
    with pytest.raises(ValueError, match="Unknown org 'not_a_real_org'"):
        conventionalize.add_standard_glob_attrs_org(ds, org='not_a_real_org')


def test_add_standard_glob_attrs_org_does_not_override_existing_by_default():
    """override=False (the default) should not touch an attribute the
    dataset already has."""
    ds = xr.Dataset(attrs={'institution': 'My Own Institution'})
    ds_out = conventionalize.add_standard_glob_attrs_org(ds, org='npi')
    assert ds_out.attrs['institution'] == 'My Own Institution'


def test_add_standard_glob_attrs_org_override_true_replaces_existing():
    """override=True should replace an existing attribute with the
    organization's standard value."""
    ds = xr.Dataset(attrs={'institution': 'My Own Institution'})
    ds_out = conventionalize.add_standard_glob_attrs_org(
        ds, org='npi', override=True)
    assert ds_out.attrs['institution'] == 'Norwegian Polar Institute (NPI)'


# ---------------------------------------------------------------------
# 0.5.1: convert_64_to_32 must never round the time coordinate
# ---------------------------------------------------------------------
def _ds_with_numeric_time(freq="15min", n=31000, tname="TIME",
                          units="days since 1970-01-01"):
    t = pd.date_range("2021-11-07", periods=n, freq=freq)
    days = (t - pd.Timestamp("1970-01-01")).total_seconds().values / 86400
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {"TEMP": (tname, 1 + 0.5 * rng.standard_normal(n), {"units": "degC"}),
         "PRES": (tname, 49 + 0.3 * rng.standard_normal(n).round(3), {"units": "dbar"}),
         "CNDC": (tname, 29 + 0.2 * rng.standard_normal(n).round(4), {"units": "mS/cm"})},
        coords={tname: (tname, days, {"units": units, "calendar": "standard"})})


@pytest.mark.parametrize("freq", ["1s", "1min", "15min", "1h"])
def test_convert_64_to_32_keeps_time_as_exact_float64(freq):
    ds = _ds_with_numeric_time(freq)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # keeping time as float64 is silent
        out = conventionalize.convert_64_to_32(ds)
    assert out.TIME.dtype == np.float64
    np.testing.assert_array_equal(out.TIME.values, ds.TIME.values)


def test_convert_64_to_32_keeps_time_even_with_force():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = conventionalize.convert_64_to_32(_ds_with_numeric_time(), force=True)
    assert out.TIME.dtype == np.float64


def test_convert_64_to_32_detects_time_by_units_not_name():
    ds = _ds_with_numeric_time(tname="t_obs")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert conventionalize.convert_64_to_32(ds).t_obs.dtype == np.float64


def test_convert_64_to_32_keeps_int64_epoch_seconds():
    ds = xr.Dataset(
        {"X": ("TIME", np.arange(5.0))},
        coords={"TIME": ("TIME", (np.arange(5) + 1_700_000_000).astype("int64"),
                         {"units": "seconds since 1970-01-01"})})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert conventionalize.convert_64_to_32(ds).TIME.dtype == np.int64


def test_convert_64_to_32_still_converts_ordinary_variables():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = conventionalize.convert_64_to_32(_ds_with_numeric_time())
    assert {out[v].dtype for v in ("TEMP", "PRES", "CNDC")} == {np.dtype("float32")}


def test_convert_64_to_32_step_guard_keeps_offset_dominated_variable():
    x = 1e6 + np.arange(1000) * 0.01  # float32 spacing at 1e6 is 0.0625
    ds = xr.Dataset({"X": ("N", x)})
    with pytest.warns(UserWarning, match="lose resolution"):
        assert conventionalize.convert_64_to_32(ds).X.dtype == np.float64

# ---------------------------------------------------------------------
# CF boundary variables (e.g. TIME_bnds)
# ---------------------------------------------------------------------

def _daily_ds_with_bounds():
    """Daily means with numeric TIME and TIME_bnds (no attributes of their
    own, as CF recommends)."""
    days = 18940.0 + np.arange(30)
    return xr.Dataset(
        {"TEMP": ("TIME", np.linspace(1, 2, 30))},
        coords={"TIME": ("TIME", days + 0.5,
                         {"units": "days since 1970-01-01", "bounds": "TIME_bnds"}),
                "TIME_bnds": (("TIME", "bnds"), np.stack([days, days + 1], 1))})


def test_convert_64_to_32_never_converts_time_bounds():
    """TIME_bnds has no units of its own, and at a 1-day spacing float32
    rounding (~170 s) passes the resolution check -- so without knowing it
    belongs to TIME it would be silently converted."""
    out = conventionalize.convert_64_to_32(_daily_ds_with_bounds())
    assert out.TIME_bnds.dtype == np.float64
    assert out.TEMP.dtype == np.float32


def test_nans_to_fill_value_skips_bounds_data_variable():
    ds = _daily_ds_with_bounds().reset_coords("TIME_bnds")  # as data variable
    out = conventionalize.nans_to_fill_value(ds)
    assert "_FillValue" not in out.TIME_bnds.attrs
    assert "_FillValue" in out.TEMP.attrs


def test_add_standard_var_attrs_leaves_bounds_alone():
    """'TIME_bnds' must not pick up the standard attributes of 'TIME'."""
    out = conventionalize.add_standard_var_attrs(_daily_ds_with_bounds())
    assert out.TIME_bnds.attrs == {}
    assert out.TIME.attrs["standard_name"] == "time"


# ---------------------------------------------------------------------
# remove_numbers_in_var_names: only sensor numbers are removed
# ---------------------------------------------------------------------

def _ds_with_vars(*names):
    return xr.Dataset({n: ("TIME", np.ones(3)) for n in names},
                      coords={"TIME": [0.0, 1.0, 2.0]})


def test_remove_numbers_strips_single_sensor_numbers():
    ds = _ds_with_vars("TEMP1", "CNDC1", "PSAL1", "CHLA1_fluorescence",
                       "DOXY1_instr")
    out = conventionalize.remove_numbers_in_var_names(ds)
    assert set(out.data_vars) == {"TEMP", "CNDC", "PSAL", "CHLA_fluorescence",
                                  "DOXY_instr"}


def test_remove_numbers_keeps_digits_that_are_part_of_the_name():
    """Used to strip every digit: SIG0 -> SIG, DO18 -> DO, T090C -> TC."""
    names = ("SIG0", "DO18", "TEMP_SBE37", "T090C", "POTEMP090C", "SAL00")
    out = conventionalize.remove_numbers_in_var_names(_ds_with_vars(*names))
    assert set(out.data_vars) == set(names)


def test_remove_numbers_leaves_dual_sensors_alone():
    out = conventionalize.remove_numbers_in_var_names(
        _ds_with_vars("TEMP1", "TEMP2", "PSAL1"))
    assert set(out.data_vars) == {"TEMP1", "TEMP2", "PSAL"}


def test_remove_numbers_does_not_clash_with_existing_name():
    out = conventionalize.remove_numbers_in_var_names(
        _ds_with_vars("TEMP", "TEMP1"))
    assert set(out.data_vars) == {"TEMP", "TEMP1"}


def test_metadata_auto_keeps_sig0():
    import warnings
    from kval.data import moored
    ds = _ds_with_vars("SIG0", "PSAL1")
    ds.TIME.attrs["units"] = "days since 1970-01-01"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = moored.metadata_auto(ds)
    assert set(out.data_vars) == {"SIG0", "PSAL"}


# ---------------------------------------------------------------------
# geospatial_bounds (ACDD 1.3: WKT, "lat lon" order for EPSG:4326)

def _ds_with_ranges(lat_min, lat_max, lon_min, lon_max):
    ds = xr.Dataset()
    ds.attrs.update(geospatial_lat_min=lat_min, geospatial_lat_max=lat_max,
                    geospatial_lon_min=lon_min, geospatial_lon_max=lon_max)
    return ds


def test_geospatial_bounds_polygon_is_lat_lon_like_the_acdd_example():
    """Used to be written lon lat, i.e. with the axes swapped for EPSG:4326."""
    ds = _ds_with_ranges(40.26, 41.26, -111.29, -110.29)
    assert conventionalize._get_geospatial_bounds_wkt_str(ds) == (
        "POLYGON ((40.26 -111.29, 41.26 -111.29, 41.26 -110.29, "
        "40.26 -110.29, 40.26 -111.29))")


def test_geospatial_bounds_polygon_rounds_outwards():
    ds = _ds_with_ranges(78.123, 79.871, -5.555, 10.001)
    assert conventionalize._get_geospatial_bounds_wkt_str(ds) == (
        "POLYGON ((78.12 -5.56, 79.88 -5.56, 79.88 10.01, "
        "78.12 10.01, 78.12 -5.56))")


def test_geospatial_bounds_single_position_is_a_point():
    ds = _ds_with_ranges(81.0, 81.0, 31.25, 31.25)
    assert conventionalize._get_geospatial_bounds_wkt_str(ds) == "POINT (81 31.25)"
    ds = _ds_with_ranges(78.9583333, 78.9583333, -0.123456, -0.123456)
    assert (conventionalize._get_geospatial_bounds_wkt_str(ds)
            == "POINT (78.95833 -0.12346)")


def test_geospatial_bounds_one_varying_coordinate_is_a_line():
    ds = _ds_with_ranges(80.0, 81.5, 30.0, 30.0)
    assert (conventionalize._get_geospatial_bounds_wkt_str(ds)
            == "LINESTRING (80 30, 81.5 30)")


def test_add_range_attrs_mooring_gets_point_bounds():
    ds = create_basic_ds()
    ds["LATITUDE"] = (("x",), [79.5, 79.5])
    ds["LONGITUDE"] = (("x",), [-3.25, -3.25])
    out = conventionalize.add_range_attrs(ds)
    assert out.attrs["geospatial_bounds"] == "POINT (79.5 -3.25)"
    assert out.attrs["geospatial_bounds_crs"] == "EPSG:4326"


# ---------------------------------------------------------------------
# GCMD keywords

def _kw_ds(*names):
    return xr.Dataset({name: ("x", [1.0]) for name in names})


def test_gmdc_keywords_are_in_dictionary_order_without_duplicates():
    """The order used to come from a set, so it changed from run to run."""
    ds = conventionalize.add_gmdc_keywords_ctd(
        _kw_ds("VCUR", "PSAL", "TEMP", "UCUR"), moored=True)
    assert ds.attrs["keywords"] == ",".join([
        "OCEANS>OCEAN TEMPERATURE>WATER TEMPERATURE",
        "OCEANS>SALINITY/DENSITY>SALINITY",
        "OCEANS>OCEAN CIRCULATION>OCEAN CURRENTS",
    ])
    ds2 = conventionalize.add_gmdc_keywords_moor(
        _kw_ds("UCUR", "TEMP", "PSAL", "VCUR"))
    assert ds2.attrs["keywords"] == ds.attrs["keywords"]


def test_gmdc_keywords_existing_keywords_are_left_alone_by_default():
    """Existing keywords used to be overwritten whatever `reset` was."""
    ds = _kw_ds("TEMP")
    ds.attrs["keywords"] = "my own keyword"
    out = conventionalize.add_gmdc_keywords_ctd(ds)
    assert out.attrs["keywords"] == "my own keyword"
    out = conventionalize.add_gmdc_keywords_moor(ds)
    assert out.attrs["keywords"] == "my own keyword"


def test_gmdc_keywords_override_replaces_existing_keywords():
    ds = _kw_ds("TEMP")
    ds.attrs["keywords"] = "my own keyword"
    out = conventionalize.add_gmdc_keywords_ctd(ds, override=True)
    assert out.attrs["keywords"] == (
        "OCEANS>OCEAN TEMPERATURE>OCEAN TEMPERATURE PROFILES")


def test_gmdc_keywords_empty_keywords_are_filled():
    ds = _kw_ds("TEMP")
    ds.attrs["keywords"] = ""
    out = conventionalize.add_gmdc_keywords_moor(ds)
    assert out.attrs["keywords"] == "OCEANS>OCEAN TEMPERATURE>WATER TEMPERATURE"


@pytest.mark.parametrize("reset, expected", [
    (True, "OCEANS>OCEAN TEMPERATURE>WATER TEMPERATURE"),
    (False, "my own keyword"),
])
def test_gmdc_keywords_reset_is_deprecated_but_honoured(reset, expected):
    ds = _kw_ds("TEMP")
    ds.attrs["keywords"] = "my own keyword"
    with pytest.warns(DeprecationWarning, match="override"):
        out = conventionalize.add_gmdc_keywords_moor(ds, reset=reset)
    assert out.attrs["keywords"] == expected


def test_gmdc_keywords_with_a_comma_are_quoted(monkeypatch):
    """ACDD: keywords are comma-separated, so one containing a comma is
    put in quotes."""
    monkeypatch.setattr(conventionalize._standard_attrs,
                        "gmdc_keyword_dict_moored",
                        {"TEMP": "A>B, C", "PSAL": "D>E"})
    out = conventionalize.add_gmdc_keywords_moor(_kw_ds("TEMP", "PSAL"))
    assert out.attrs["keywords"] == '"A>B, C",D>E'


# ---------------------------------------------------------------------
# add_range_attrs: time coverage

def _time_ds(time_values, units=None):
    ds = xr.Dataset({"LATITUDE": 80.0, "LONGITUDE": 30.0},
                    coords={"TIME": ("TIME", time_values)})
    if units is not None:
        ds["TIME"].attrs["units"] = units
    return ds


def test_add_range_attrs_datetime_time():
    """datetime64 TIME used to fail in the resolution/duration part (it
    needed TIME.units), so these attributes were not set."""
    times = pd.date_range("2024-01-01", periods=97, freq="15min").values
    out = conventionalize.add_range_attrs(_time_ds(times))
    assert out.attrs["time_coverage_start"] == "2024-01-01T00:00:00Z"
    assert out.attrs["time_coverage_end"] == "2024-01-02T00:00:00Z"
    assert out.attrs["time_coverage_resolution"] == "P0000-00-00T00:15:00"
    assert out.attrs["time_coverage_duration"] == "P0000-00-01T00:00:00"


def test_add_range_attrs_resolution_uses_the_time_units():
    """The step of numeric TIME used to be taken as days whatever the
    units were: 6-hourly data in 'hours since' got a resolution of 6 days."""
    out = conventionalize.add_range_attrs(
        _time_ds(np.array([0.0, 6.0, 12.0]), "hours since 2000-01-01"))
    assert out.attrs["time_coverage_resolution"] == "P0000-00-00T06:00:00"
    out = conventionalize.add_range_attrs(
        _time_ds(np.array([0.0, 600.0, 1200.0]), "seconds since 2000-01-01"))
    assert out.attrs["time_coverage_resolution"] == "P0000-00-00T00:10:00"


@pytest.mark.parametrize("numeric", [True, False])
def test_add_range_attrs_time_coverage_from_time_bounds(numeric):
    """With CF time bounds (e.g. daily averages) and
    time_coverage_from_bounds=True, the coverage runs from the start of the
    first to the end of the last interval. By default it runs from the first
    to the last time stamp (ACDD; what the IOOS checker tests)."""
    times = pd.to_datetime(["2024-03-01 12:00", "2024-03-02 12:00",
                            "2024-03-03 12:00"]).values
    bnds = np.stack([times - np.timedelta64(12, "h"),
                     times + np.timedelta64(12, "h")], axis=1)
    if numeric:
        epoch = np.datetime64("1970-01-01")
        times = (times - epoch) / np.timedelta64(1, "D")
        bnds = (bnds - epoch) / np.timedelta64(1, "D")
        ds = _time_ds(times, "days since 1970-01-01")
    else:
        ds = _time_ds(times)
    ds["TIME"].attrs["bounds"] = "TIME_bnds"
    ds["TIME_bnds"] = (("TIME", "bnds"), bnds)

    out = conventionalize.add_range_attrs(ds.copy())
    assert out.attrs["time_coverage_start"] == "2024-03-01T12:00:00Z"
    assert out.attrs["time_coverage_end"] == "2024-03-03T12:00:00Z"
    assert out.attrs["time_coverage_duration"] == "P0000-00-02T00:00:00"
    assert out.attrs["time_coverage_resolution"] == "P0000-00-01T00:00:00"

    out = conventionalize.add_range_attrs(ds.copy(),
                                          time_coverage_from_bounds=True)
    assert out.attrs["time_coverage_start"] == "2024-03-01T00:00:00Z"
    assert out.attrs["time_coverage_end"] == "2024-03-04T00:00:00Z"
    assert out.attrs["time_coverage_duration"] == "P0000-00-03T00:00:00"
    assert out.attrs["time_coverage_resolution"] == "P0000-00-01T00:00:00"


def test_add_range_attrs_after_kval_time_average():
    from kval.util.xr_funcs import time_average
    times = pd.date_range("2024-03-01 00:30", periods=72, freq="1h")
    ds = _time_ds(times.values)
    ds["TEMP"] = ("TIME", np.arange(72.0))
    out = conventionalize.add_range_attrs(time_average(ds, "1D"),
                                          time_coverage_from_bounds=True)
    assert out.attrs["time_coverage_start"] == "2024-03-01T00:00:00Z"
    assert out.attrs["time_coverage_end"] == "2024-03-04T00:00:00Z"


def test_add_range_attrs_does_not_set_vertical_crs():
    """geospatial_bounds is 2-D, so ACDD's geospatial_bounds_vertical_crs
    (the CRS of its Z coordinates) does not apply. It used to be set to
    EPSG:5831 (depth in m), also for a vertical range in dbar."""
    ds = create_basic_ds(vertical_var="PRES")
    ds["PRES"].attrs["units"] = "dbar"
    out = conventionalize.add_range_attrs(ds)
    assert out.attrs["geospatial_vertical_units"] == "dbar"
    assert "geospatial_bounds_vertical_crs" not in out.attrs
