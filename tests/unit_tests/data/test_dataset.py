import pytest
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from kval.data import dataset

# Define a fixture for a mock dataset
@pytest.fixture
def mock_dataset() -> xr.Dataset:
    """
    Fixture to create a mock xarray.Dataset with metadata and variables for testing.
    """
    # Define dimensions
    Nt = 10
    time = pd.date_range('2024-01-01', periods=Nt, freq='D')
    pres = [100, 600, 2000, 6000, 11000]  # Pressure levels

    # Create data for TEMP(TIME, PRES)
    temp_data = 15 + 8 * np.random.randn(Nt, len(pres))  # Example temperature data

    # Create data for STATION(TIME) and OCEAN(TIME)
    station_data = [f'st{stnum:02.0f}' for stnum in np.arange(1, Nt + 1)]
    ocean_data = ['Atlantic', 'Arctic', 'Pacific', 'Mediterranean', 'Southern',
                 'Baltic', 'Indian', 'Caribbean', 'Weddell', 'Ross']

    # Create data for ZONE(PRES)
    zone_data = ['epipelagic', 'mesopelagic', 'bathypelagic', 'abyssopelagic', 'hadopelagic']

    # Create the Dataset
    ds = xr.Dataset(
        {
            'TEMP': (['TIME', 'PRES'], temp_data),
            'OCEAN': (['TIME'], ocean_data),
            'STATION': (['TIME'], station_data),
            'ZONE': (['PRES'], zone_data)
        },
        coords={
            'TIME': time,
            'PRES': pres
        },
        # Add some metadata
        attrs={
            'id': 'test_dataset',
            'description': 'This is a test dataset.',
            'author': 'Test Author'
        }
    )

    # Add variable attributes
    ds['TEMP'].attrs = {
        'units': 'degC',
        'long_name': 'Test Temperature'
    }

    return ds

## TEST NETCDF EXPORT

def test_to_netcdf_default_filename(mock_dataset):
    """
    Test the to_netcdf function to ensure default file naming works correctly.
    """
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / 'test_dataset.nc'
        print(file_path, 'AAAA')
        # Call the function with no file_name
        dataset.to_netcdf(mock_dataset, tmpdir)

        # Check that the file was created with the default name
        assert file_path.exists()


def test_to_netcdf_custom_filename(mock_dataset):
    """
    Test the to_netcdf function to ensure custom file naming works correctly.
    """
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / 'custom_file.nc'

        # Call the function with a custom file_name
        dataset.to_netcdf(mock_dataset, tmpdir, file_name='custom_file')

        # Check that the file was created with the custom name
        assert file_path.exists()


def test_to_netcdf_file_overwrite(mock_dataset):
    """
    Test the to_netcdf function to ensure file overwrite behavior works correctly.
    """
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / 'overwrite_test.nc'

        # Create an initial file with arbitrary content
        with open(file_path, 'wb') as f:
            f.write(b'Initial content')

        # Mock user input to automatically overwrite
        with patch('builtins.input', return_value='y'):
            dataset.to_netcdf(mock_dataset, tmpdir, file_name='overwrite_test')

        # Check that the file was overwritten
        assert file_path.exists()

        # Optionally, check the file's content length or other characteristics
        # This check depends on what you expect in the file.
        # For NetCDF files, you might check specific metadata or attributes.
        # Here, we just ensure the file size is greater than the initial file size.
        assert file_path.stat().st_size > len(b'Initial content')


def test_to_netcdf_verbose_output(mock_dataset):
    """
    Test the to_netcdf function to ensure verbose output works correctly.
    """
    with TemporaryDirectory() as tmpdir:
        with patch('builtins.print') as mock_print:
            dataset.to_netcdf(mock_dataset, tmpdir, verbose=True)
            assert mock_print.called

def test_to_netcdf_convention_check(mock_dataset):
    """
    Test the to_netcdf function to ensure convention checker is called.
    """
    with TemporaryDirectory() as tmpdir:
        with patch('kval.data.dataset.compliance.compliance_checks_ioos') as mock_check:
            dataset.to_netcdf(mock_dataset, tmpdir, convention_check=True)
            mock_check.assert_called_once_with(Path(tmpdir) / 'test_dataset.nc')



# ===================================================================
# Tests for the gsw-based derived-parameter functions:
# calculate_PSAL, calculate_CNDC, calculate_SA_CT, calculate_rho,
# calculate_sig0, calculate_ss.
#
# These check numerical correctness against official TEOS-10 reference
# ("check") values -- the same reference values gsw-python's own test
# suite uses to validate against the reference MATLAB toolbox. A test
# that only checks "the function runs and returns a finite number" would
# not catch a real bug like a swapped argument order or a wrong input
# unit, since gsw would still return *some* number in that case.
# Comparing against independently-known-correct values is what actually
# catches that class of bug.
#
# Reference values were extracted from gsw's own check-value dataset
# (gsw/tests/gsw_cv_v3_0.npz, TEOS-10 v3.0) at two points:
#   - point A: shallow, warm (p=0 dbar, t=27.962 C)
#   - point B: intermediate depth, cooler (p=909 dbar, t=5.195 C)
#
# calculate_rho, calculate_sig0, and calculate_ss use gsw's "computationally
# efficient" polynomial approximation (Roquet et al., 2015) rather than the
# fully exact Gibbs-function computation the reference values use, so these
# three are checked with a small tolerance rather than an exact match. This
# was verified deliberately (not just assumed) -- the approximation's known
# discrepancy against the exact reference is on the order of 1e-4 to 1e-1
# in absolute terms, many orders of magnitude below any real CTD sensor's
# measurement precision. calculate_PSAL, calculate_CNDC, and calculate_SA_CT
# use exact algorithms and match the reference to machine precision.
# ===================================================================

from kval.data.dataset import (
    calculate_PSAL,
    calculate_CNDC,
    calculate_SA_CT,
    calculate_rho,
    calculate_sig0,
    calculate_ss,
)

_POINT_A = dict(
    SP=34.306287392599714, t=27.962, p=0.0, lat=11.0, lon=142.0,
    SA_ref=34.468236430490606, CT_ref=27.996436412058213,
    RHO_ref=1021.8866110446018, SIG0_ref=21.886611044601636,
    SVEL_ref=1540.4098538961257, C_ref=55.19754712635529,
)
_POINT_B = dict(
    SP=34.5463600000422, t=5.194999999999999, p=909.0, lat=9.5, lon=183.0,
    SA_ref=34.718723829149454, CT_ref=5.117058424764436,
    RHO_ref=1031.4704840056474, SIG0_ref=27.309757772675766,
    SVEL_ref=1485.6746748734474, C_ref=33.6425208551767,
)
_GSW_POINTS = [_POINT_A, _POINT_B]


def _make_gsw_ds(point, with_cndc=True, with_psal=True, with_latlon=True):
    """Build a minimal single-point xr.Dataset with the variable names
    and structure kval's calculate_* functions expect."""
    ds = xr.Dataset(
        {
            "TEMP": ("TIME", [point["t"]]),
            "PRES": ("TIME", [point["p"]]),
        },
        coords={"TIME": [0]},
    )
    if with_cndc:
        ds["CNDC"] = ("TIME", [point["C_ref"]])
        ds["CNDC"].attrs["units"] = "mS/cm"
    if with_psal:
        ds["PSAL"] = ("TIME", [point["SP"]])
    if with_latlon:
        ds["LATITUDE"] = ((), point["lat"])
        ds["LONGITUDE"] = ((), point["lon"])
    return ds


@pytest.mark.parametrize("point", _GSW_POINTS)
def test_calculate_PSAL_matches_reference(point):
    ds = _make_gsw_ds(point, with_psal=False, with_latlon=False)
    result = calculate_PSAL(ds)
    assert result["PSAL"].values[0] == pytest.approx(point["SP"], rel=1e-10)


@pytest.mark.parametrize("point", _GSW_POINTS)
def test_calculate_CNDC_matches_reference(point):
    ds = _make_gsw_ds(point, with_cndc=False, with_latlon=False)
    result = calculate_CNDC(ds)
    assert result["CNDC"].values[0] == pytest.approx(point["C_ref"], rel=1e-10)


@pytest.mark.parametrize("point", _GSW_POINTS)
def test_calculate_PSAL_CNDC_round_trip(point):
    """Recomputing CNDC from the recomputed PSAL should recover the
    original CNDC -- a sanity check that these two functions are
    consistent inverses of each other, independent of the reference data."""
    ds = _make_gsw_ds(point, with_psal=False, with_latlon=False)
    ds = calculate_PSAL(ds)
    ds = calculate_CNDC(ds)
    assert ds["CNDC"].values[0] == pytest.approx(point["C_ref"], rel=1e-8)


def test_calculate_PSAL_converts_S_per_m_units():
    """calculate_PSAL should detect CNDC given in S/m and apply the x10
    conversion to mS/cm before calling gsw -- gsw.SP_from_C explicitly
    requires input conductivity in mS/cm."""
    point = _POINT_A
    ds = _make_gsw_ds(point, with_psal=False, with_latlon=False)
    ds["CNDC"].values[:] = ds["CNDC"].values / 10.0  # express as S/m instead
    ds["CNDC"].attrs["units"] = "S m-1"
    result = calculate_PSAL(ds)
    assert result["PSAL"].values[0] == pytest.approx(point["SP"], rel=1e-6)


@pytest.mark.parametrize("point", _GSW_POINTS)
def test_calculate_SA_CT_matches_reference(point):
    ds = _make_gsw_ds(point)
    result = calculate_SA_CT(ds)
    assert result["SA"].values[0] == pytest.approx(point["SA_ref"], rel=1e-8)
    assert result["CT"].values[0] == pytest.approx(point["CT_ref"], rel=1e-8)


@pytest.mark.parametrize("point", _GSW_POINTS)
def test_calculate_rho_matches_reference(point):
    ds = _make_gsw_ds(point)
    result = calculate_rho(ds)
    assert result["RHO"].values[0] == pytest.approx(point["RHO_ref"], rel=1e-5)


@pytest.mark.parametrize("point", _GSW_POINTS)
def test_calculate_sig0_matches_reference(point):
    ds = _make_gsw_ds(point)
    result = calculate_sig0(ds)
    # Absolute tolerance here, not relative: SIG0 is a density *anomaly*
    # (~20-30 kg/m3) rather than full density (~1030 kg/m3), so the same
    # absolute algorithmic discrepancy looks like a much larger relative
    # error here purely due to the smaller baseline magnitude.
    assert result["SIG0"].values[0] == pytest.approx(point["SIG0_ref"], abs=5e-3)


@pytest.mark.parametrize("point", _GSW_POINTS)
def test_calculate_ss_matches_reference_without_existing_SA_CT(point):
    ds = _make_gsw_ds(point)
    assert "SA" not in ds and "CT" not in ds
    result = calculate_ss(ds)
    assert result["SVEL"].values[0] == pytest.approx(point["SVEL_ref"], abs=0.5)


@pytest.mark.parametrize("point", _GSW_POINTS)
def test_calculate_ss_matches_reference_with_existing_SA_CT(point):
    ds = _make_gsw_ds(point)
    ds = calculate_SA_CT(ds)
    assert "SA" in ds and "CT" in ds
    result = calculate_ss(ds)
    assert result["SVEL"].values[0] == pytest.approx(point["SVEL_ref"], abs=0.5)


def test_calculate_ss_reuses_existing_SA_CT_rather_than_recomputing():
    """If SA/CT are already present, calculate_ss should use those values
    directly rather than recomputing them from CNDC/TEMP/PRES -- this is
    the documented, deliberate branch that distinguishes calculate_ss from
    calculate_rho/calculate_sig0 (which always recompute). We prove the
    branch is actually taken by planting deliberately wrong SA/CT values
    and confirming the (now deliberately wrong) output reflects them."""
    point = _POINT_A
    ds = _make_gsw_ds(point)

    correct_result = calculate_ss(ds.copy(deep=True))
    correct_svel = correct_result["SVEL"].values[0]

    ds_wrong = ds.copy(deep=True)
    ds_wrong["SA"] = ("TIME", [point["SA_ref"] + 5.0])
    ds_wrong["CT"] = ("TIME", [point["CT_ref"] + 5.0])
    wrong_result = calculate_ss(ds_wrong)
    wrong_svel = wrong_result["SVEL"].values[0]

    assert abs(wrong_svel - correct_svel) > 1.0
    assert wrong_svel != pytest.approx(point["SVEL_ref"], abs=0.5)


@pytest.mark.parametrize(
    "func", [calculate_PSAL, calculate_CNDC, calculate_SA_CT, calculate_rho, calculate_sig0, calculate_ss]
)
def test_gsw_functions_do_not_mutate_input_dataset(func):
    ds = _make_gsw_ds(_POINT_A)
    ds_original = ds.copy(deep=True)
    _ = func(ds)
    xr.testing.assert_identical(ds, ds_original)

# ===================================================================
# add_latlon
# ===================================================================

def test_add_latlon_assigns_as_coordinates_not_data_vars(mock_dataset):
    """LATITUDE/LONGITUDE should be coordinate variables, not plain data
    variables -- required for compliance_checks_custom's coordinate
    checks, and for CF/ACDD compliance generally (they're spatial
    identifiers, not measured data)."""
    ds = dataset.add_latlon(mock_dataset, lon=30.0, lat=80.0)
    assert 'LATITUDE' in ds.coords
    assert 'LONGITUDE' in ds.coords
    assert 'LATITUDE' not in ds.data_vars
    assert 'LONGITUDE' not in ds.data_vars


def test_add_latlon_correct_values_and_attrs(mock_dataset):
    ds = dataset.add_latlon(mock_dataset, lon=30.0, lat=80.0)
    assert float(ds['LATITUDE'].values) == 80.0
    assert float(ds['LONGITUDE'].values) == 30.0
    assert ds['LATITUDE'].attrs['units'] == 'degree_north'
    assert ds['LONGITUDE'].attrs['units'] == 'degree_east'
    assert ds['LATITUDE'].attrs['standard_name'] == 'latitude'
    assert ds['LONGITUDE'].attrs['standard_name'] == 'longitude'


def test_add_latlon_does_not_mutate_input(mock_dataset):
    ds_original = mock_dataset.copy(deep=True)
    _ = dataset.add_latlon(mock_dataset, lon=30.0, lat=80.0)
    xr.testing.assert_identical(mock_dataset, ds_original)


def test_add_latlon_warns_if_lat_missing():
    ds = xr.Dataset({'TEMP': ('TIME', np.array([1.0]))}, coords={'TIME': [0]})
    with pytest.warns(UserWarning, match="latitude"):
        result = dataset.add_latlon(ds, lon=30.0)
    assert 'LONGITUDE' in result.coords
    assert 'LATITUDE' not in result.coords


def test_add_latlon_warns_if_lon_missing():
    ds = xr.Dataset({'TEMP': ('TIME', np.array([1.0]))}, coords={'TIME': [0]})
    with pytest.warns(UserWarning, match="longitude"):
        result = dataset.add_latlon(ds, lat=80.0)
    assert 'LATITUDE' in result.coords
    assert 'LONGITUDE' not in result.coords


def test_add_latlon_suppress_warning():
    ds = xr.Dataset({'TEMP': ('TIME', np.array([1.0]))}, coords={'TIME': [0]})
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> test failure
        dataset.add_latlon(ds, lat=80.0, suppress_latlon_warning=True)


def test_add_latlon_zero_values_are_not_dropped():
    """lat=0/lon=0 (equator / prime meridian) are valid coordinates and
    must not be treated as falsy/missing."""
    ds = xr.Dataset({'TEMP': ('TIME', np.array([1.0]))}, coords={'TIME': [0]})
    result = dataset.add_latlon(ds, lon=0.0, lat=0.0, suppress_latlon_warning=True)
    assert 'LATITUDE' in result.coords
    assert 'LONGITUDE' in result.coords
    assert float(result.LATITUDE) == 0.0
    assert float(result.LONGITUDE) == 0.0


def test_add_latlon_forces_float32():
    """Integer input (e.g. lat=82) previously inferred as int64; now
    explicitly cast to float32 regardless of input type."""
    ds = xr.Dataset({'TEMP': ('TIME', np.array([1.0]))}, coords={'TIME': [0]})
    result = dataset.add_latlon(ds, lon=30, lat=82)  # plain ints
    assert result.LATITUDE.dtype == np.float32
    assert result.LONGITUDE.dtype == np.float32


def test_add_latlon_places_coords_after_time():
    """Purely cosmetic, but LATITUDE/LONGITUDE should consistently
    appear after TIME in the coords listing, not wherever they happen
    to land based on assignment order."""
    ds = xr.Dataset({'TEMP': ('TIME', np.array([1.0]))}, coords={'TIME': [0]})
    result = dataset.add_latlon(ds, lon=30.0, lat=80.0)
    coord_order = list(result.coords)
    assert coord_order.index('TIME') < coord_order.index('LATITUDE')
    assert coord_order.index('TIME') < coord_order.index('LONGITUDE')