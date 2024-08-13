import pytest
import xarray as xr
import os
import requests
from pathlib import Path
from kval.file.rbr import read
from kval.file._variable_defs import RBR_name_map, RBR_units_map

# Define the URLs for the files you want to test
FILE_URLS = {
    "conc_chl_par_example.rsk": "https://example.com/path/to/conc_bio.rsk",
    "conc_example.rsk": "https://example.com/path/to/conc_full.rsk",
    "solo_example.rsk": "https://example.com/path/to/solo_full.rsk",
}

# Define the directory where files should be stored
FILE_DIR = Path("tests/test_data/rbr_files")

@pytest.fixture(scope="module", autouse=True)
def setup_files():
    """Fixture to ensure test files are downloaded."""
    FILE_DIR.mkdir(parents=True, exist_ok=True)
    
    for file_name, url in FILE_URLS.items():
        file_path = FILE_DIR / file_name
        if not file_path.exists():
            # Download the file if it does not exist
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            with open(file_path, 'wb') as file:
                file.write(response.content)
    
    # Yield control back to the test functions
    yield
    
    # Teardown code, if needed (e.g., cleanup files after tests)
    # For now, no teardown is required

@pytest.mark.parametrize("file_name", [
    "solo_example.rsk",
    "conc_example.rsk",
    "conc_chl_par_example.rsk",
])

def test_read(file_name):

    # Construct the full path to the test file
    file_path = FILE_DIR / file_name

    # Read the dataset using the read function
    ds_rsk = read(file_path)

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
    assert 'instrument' in ds_rsk.attrs, "Instrument metadata not found"
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
