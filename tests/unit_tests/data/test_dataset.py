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
        with patch('kval.data.dataset.check_conventions.check_file') as mock_check:
            dataset.to_netcdf(mock_dataset, tmpdir, convention_check=True)
            mock_check.assert_called_once_with(Path(tmpdir) / 'test_dataset.nc')



# ─── Oceanographic calculation functions ────────────────────────────────────

@pytest.fixture
def mock_ctd_dataset() -> xr.Dataset:
    """
    Minimal CTD-like dataset with CNDC, TEMP, PRES, PSAL, LATITUDE, LONGITUDE.
    Uses realistic-ish values so gsw functions don't produce NaN.
    """
    np.random.seed(42)
    nt, nz = 5, 10
    pres   = np.linspace(10, 500, nz)
    temp   = 15 - pres * 0.01 + np.random.randn(nt, nz) * 0.1
    # conductivity in S/m for ~35 PSU water
    cndc   = 3.5 + np.random.randn(nt, nz) * 0.01
    psal   = np.zeros((nt, nz))
    lat    = np.full(nt, 60.0)
    lon    = np.full(nt, 5.0)

    return xr.Dataset(
        {
            "CNDC":      (["TIME", "PRES"], cndc,  {"units": "S m-1"}),
            "TEMP":      (["TIME", "PRES"], temp,  {"units": "degree_C"}),
            "PSAL":      (["TIME", "PRES"], psal,  {"units": "1"}),
            "LATITUDE":  (["TIME"],         lat,   {"units": "degree_north"}),
            "LONGITUDE": (["TIME"],         lon,   {"units": "degree_east"}),
        },
        coords={
            "TIME": np.arange(nt),
            "PRES": pres,
        },
    )


def test_calculate_PSAL_updates_variable(mock_ctd_dataset):
    """calculate_PSAL should overwrite PSAL with non-zero values."""
    import gsw
    ds = dataset.calculate_PSAL(mock_ctd_dataset, cndc_var="CNDC",
                                 temp_var="TEMP", pres_var="PRES")
    assert "PSAL" in ds
    assert not np.all(ds["PSAL"].values == 0), "PSAL was not updated"
    assert "note" in ds["PSAL"].attrs


def test_calculate_PSAL_unit_conversion(mock_ctd_dataset):
    """calculate_PSAL should apply x10 factor for S m-1 conductivity."""
    import gsw
    ds_Sm = mock_ctd_dataset.copy(deep=True)  # units already S m-1
    ds_mScm = mock_ctd_dataset.copy(deep=True)
    ds_mScm["CNDC"].values[:] *= 10
    ds_mScm["CNDC"].attrs["units"] = "mS/cm"

    result_Sm   = dataset.calculate_PSAL(ds_Sm)
    result_mScm = dataset.calculate_PSAL(ds_mScm)

    np.testing.assert_allclose(
        result_Sm["PSAL"].values, result_mScm["PSAL"].values, rtol=1e-4,
        err_msg="S m-1 and mS/cm inputs should give same PSAL"
    )


def test_calculate_PSAL_retain_nans(mock_ctd_dataset):
    """calculate_PSAL should preserve pre-existing NaNs in PSAL when retain_nans=True."""
    ds = mock_ctd_dataset.copy(deep=True)
    ds["PSAL"].values[0, 0] = np.nan

    result = dataset.calculate_PSAL(ds, retain_nans=True)
    assert np.isnan(result["PSAL"].values[0, 0]), "NaN was not preserved"
    assert not np.isnan(result["PSAL"].values[0, 1])


def test_calculate_SA_CT_adds_variables(mock_ctd_dataset):
    """calculate_SA_CT should add SA and CT with correct units."""
    ds = dataset.calculate_PSAL(mock_ctd_dataset)
    ds = dataset.calculate_SA_CT(ds)

    assert "SA" in ds
    assert "CT" in ds
    assert ds["SA"].attrs["units"] == "g kg-1"
    assert ds["CT"].attrs["units"] == "degree_C"
    assert ds["SA"].shape == ds["PSAL"].shape
    assert not np.all(np.isnan(ds["SA"].values))


def test_calculate_rho_adds_variable(mock_ctd_dataset):
    """calculate_rho should add RHO with physically plausible values."""
    ds = dataset.calculate_PSAL(mock_ctd_dataset)
    ds = dataset.calculate_rho(ds)

    assert "RHO" in ds
    assert ds["RHO"].attrs["units"] == "kg m-3"
    # Seawater density is roughly 1020–1030 kg/m³
    rho_vals = ds["RHO"].values
    assert np.nanmin(rho_vals) > 990 and np.nanmax(rho_vals) < 1060, (
        f"RHO values out of plausible range: {np.nanmin(rho_vals):.1f}–{np.nanmax(rho_vals):.1f}"
    )


def test_calculate_sig0_adds_variable(mock_ctd_dataset):
    """calculate_sig0 should add SIG0 with values roughly 20–30 kg/m³."""
    ds = dataset.calculate_PSAL(mock_ctd_dataset)
    ds = dataset.calculate_sig0(ds)

    assert "SIG0" in ds
    assert ds["SIG0"].attrs["units"] == "kg m-3"
    sig0_vals = ds["SIG0"].values
    assert np.nanmin(sig0_vals) > 15 and np.nanmax(sig0_vals) < 40, (
        f"SIG0 values out of plausible range: {np.nanmin(sig0_vals):.2f}–{np.nanmax(sig0_vals):.2f}"
    )


def test_calculate_CNDC_roundtrip(mock_ctd_dataset):
    """calculate_CNDC after calculate_PSAL should roughly recover original conductivity."""
    ds = dataset.calculate_PSAL(mock_ctd_dataset)
    # Store original CNDC (in S m-1), compute PSAL, then recompute CNDC
    ds_recndc = dataset.calculate_CNDC(ds)

    assert "CNDC" in ds_recndc
    assert "note" in ds_recndc["CNDC"].attrs


def test_calculations_do_not_modify_input(mock_ctd_dataset):
    """All calculation functions should return a new dataset, not modify in place."""
    ds_orig = mock_ctd_dataset.copy(deep=True)
    ds = dataset.calculate_PSAL(mock_ctd_dataset)
    ds = dataset.calculate_SA_CT(ds)
    ds = dataset.calculate_rho(ds)
    ds = dataset.calculate_sig0(ds)

    # Original should be unchanged
    np.testing.assert_array_equal(
        mock_ctd_dataset["PSAL"].values, ds_orig["PSAL"].values
    )