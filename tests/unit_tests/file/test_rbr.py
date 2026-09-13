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
def _download_with_retry(url: str, retries: int = 3, backoff: float = 2.0) -> bytes:
    """
    GET a URL, retrying on transient server errors (e.g. Zenodo occasionally
    returning a 504 under load). Raises the last error if all attempts fail.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return response.content
        except requests.exceptions.HTTPError as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise last_exc


@pytest.fixture
def rbr_file(request):
    """
    Ensure a single named RBR test file is downloaded and available locally.
    """
    file_name = request.param
    FILE_DIR.mkdir(parents=True, exist_ok=True)
    file_path = FILE_DIR / file_name
    was_downloaded = False

    if not file_path.exists():
        content = _download_with_retry(FILE_URLS[file_name])
        with open(file_path, "wb") as file:
            file.write(content)
        was_downloaded = True

    yield file_path

    if was_downloaded and file_path.exists():
        for _ in range(3):
            try:
                file_path.unlink()
                break
            except PermissionError:
                time.sleep(0.5)

@pytest.mark.parametrize("rbr_file", [
    "solo_example.rsk",
    "conc_example.rsk",
    "conc_chl_par_example.rsk",
], indirect=True)

def test_read(rbr_file):

    # Read the dataset using the read function
    ds_rsk = read_rsk(rbr_file)

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
