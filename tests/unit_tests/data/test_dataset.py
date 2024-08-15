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


# TEST METADATA EXPORT

def test_metadata_to_txt(mock_dataset):
    """
    Test the metadata_to_txt function using a mock dataset and temporary file.
    """
    # Define the output file path
    outfile = 'metadata_output.txt'

    # Call the function to create the metadata file
    dataset.metadata_to_txt(mock_dataset, outfile)

    # Read the content of the file and verify its correctness
    with open(outfile, 'r') as f:
        content = f.read()

    # Check for expected strings in the output file
    assert 'FILE METADATA FROM: test_dataset' in content
    assert '### GLOBAL ATTRIBUTES   ###' in content
    assert 'description:\nThis is a test dataset.' in content
    assert 'author:\nTest Author' in content
    assert '### VARIABLE ATTRIBUTES ###' in content
    assert 'TEMP' in content
    assert 'units:\ndegC' in content
    assert 'long_name:\nTest Temperature' in content

    # Verify that attributes appear directly after the correct variable name
    temp_index = content.index('TEMP')
    ocean_index = content.index('OCEAN')
    station_index = content.index('STATION')

    # Check that the order is right
    assert temp_index < ocean_index
    assert ocean_index < station_index

    # Check that we have one unit entry and one long_name entry under TEMP
    assert content[temp_index:ocean_index].count('units:') == 1
    assert content[temp_index:ocean_index].count('long_name:') == 1

    # Check that we have no unit entry or one long_name entry under OCEAN
    assert content[ocean_index:station_index].count('units:') == 0
    assert content[ocean_index:station_index].count('long_name:') == 0

    # Clean up the file after test
    Path(outfile).unlink()