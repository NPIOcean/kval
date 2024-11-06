import pytest
import xarray as xr
import os
import time
import requests
from pathlib import Path
from kval.file.rbr import read_rsk
from kval.file._variable_defs import RBR_name_map, RBR_units_map

# Define the URLs for the files you want to test
FILE_URLS = {
    "conc_chl_par_example.rsk": "https://zenodo.org/records/13321317/files/conc_chl_par_example.rsk?download=1",
    "conc_example.rsk": "https://zenodo.org/records/13321317/files/conc_example.rsk?download=1",
    "solo_example.rsk": "https://zenodo.org/records/13321317/files/solo_example.rsk?download=1",
}

# Define the directory where files should be stored
FILE_DIR = Path("tests/test_data/rbr_files")

@pytest.fixture(scope="module", autouse=True)
def setup_files():
    """
    Fixture to ensure test files are downloaded and available for testing.
    Local files are used without deletion, while downloaded files are deleted after testing.
    """
    # Create the directory for storing test files if it doesn't already exist.
    FILE_DIR.mkdir(parents=True, exist_ok=True)

    # Track which files are downloaded
    downloaded_files = set()

    # Download each file in FILE_URLS if it doesn't already exist locally.
    for file_name, url in FILE_URLS.items():
        file_path = FILE_DIR / file_name
        if not file_path.exists():  # Check if the file is already present
            response = requests.get(url)  # Download the file
            response.raise_for_status()  # Raise an error if download fails
            with open(file_path, 'wb') as file:  # Write the file to disk
                file.write(response.content)
            downloaded_files.add(file_name)  # Mark the file as downloaded

    # Yield control back to the test functions. This pauses the fixture here.
    yield

    # Teardown: This code runs after the tests are complete.
    # Only delete files that were downloaded
    for file_name in FILE_URLS.keys():
        file_path = FILE_DIR / file_name
        if file_path.exists() and file_name in downloaded_files:
            # Only delete files that were downloaded
            for _ in range(3):  # Try up to 3 times to delete the file
                try:
                    file_path.unlink()  # Attempt to delete the file
                    break  # Exit the loop if the file is successfully deleted
                except PermissionError:  # Handle the case of locked file
                    time.sleep(0.5)  # Wait for a short time before trying again

@pytest.mark.parametrize("file_name", [
    "solo_example.rsk",
    "conc_example.rsk",
    "conc_chl_par_example.rsk",
])

def test_read(file_name):

    # Construct the full path to the test file
    file_path = FILE_DIR / file_name

    # Read the dataset using the read function
    ds_rsk = read_rsk(file_path)

    # Check if the dataset is of type xarray.Dataset
    assert isinstance(ds_rsk, xr.Dataset), "Output is not an xarray.Dataset"

    # Check that the TIME dimension exists and is of correct type
    assert 'TIME' in ds_rsk.dims, "TIME dimension not found"
    assert ds_rsk['TIME'].dtype == 'float64', "TIME variable is not of type float64"

    # Check for presence of expected variables in the dataset
    assert 'TIME' in ds_rsk.variables, "TIME variable not found"
    assert len(ds_rsk.data_vars) > 0, "No data variables found in dataset"

    # Check if units are correctly assigned
    for var in ds_rsk.data_vars:
        assert 'units' in ds_rsk[var].attrs, f"Units attribute not found for variable {var}"
        expected_unit = RBR_units_map.get(ds_rsk[var].attrs['units'], ds_rsk[var].attrs['units'])
        assert ds_rsk[var].attrs['units'] == expected_unit, f"Units attribute for {var} is incorrect"

    # Check if the TIME variable has correct attributes
    assert ds_rsk['TIME'].attrs['units'] == 'days since 1970-01-01', "TIME units attribute is incorrect"
    assert ds_rsk['TIME'].attrs['axis'] == 'T', "TIME axis attribute is incorrect"

    # Check for presence of metadata
    assert 'instrument_model' in ds_rsk.attrs, "Instrument model metadata not found"
    assert 'instrument_serial_number' in ds_rsk.attrs, "Instrument serial number metadata not found"
    assert 'time_coverage_resolution' in ds_rsk.attrs, "Time coverage resolution metadata not found"

    # Validate calibration dates
    for var in ds_rsk.data_vars:
        if 'calibration_date' in ds_rsk[var].attrs:
            assert ds_rsk[var].attrs['calibration_date'] == ds_rsk[var].attrs.get('calibration_date'), f"Calibration date for {var} is incorrect"

    # Check for no extra variables
    expected_var_names = set(RBR_name_map.values())
    actual_var_names = set(ds_rsk.variables) - {'TIME'}
    assert actual_var_names.issubset(expected_var_names), "Unexpected variables found in the dataset"

    # Check if the TIME variable has the correct type
    assert ds_rsk['TIME'].dtype == 'float64', "TIME variable type is incorrect"
