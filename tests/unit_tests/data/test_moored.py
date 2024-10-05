import pytest
import xarray as xr
import requests
from pathlib import Path
from kval.data.moored import load_moored, assign_pressure
from unittest import mock
from unittest import mock
import numpy as np

# Define the URLs for the files you want to test
RBR_FILE_URLS = {
    "conc_chl_par_example.rsk": "https://zenodo.org/records/13321317/files/conc_chl_par_example.rsk?download=1",
    "conc_example.rsk": "https://zenodo.org/records/13321317/files/conc_example.rsk?download=1",
    "solo_example.rsk": "https://zenodo.org/records/13321317/files/solo_example.rsk?download=1",
}

# Define the directory where files should be stored
RBR_FILE_DIR = Path("tests/test_data/rbr_files")
SBE37_FILE_PATH = Path("tests/test_data/sbe_files/sbe37/cnv/test_sbe37.cnv")

@pytest.fixture(scope="module", autouse=True)
def setup_files():
    """
    Fixture to ensure test files are downloaded and available for testing.
    Local files are used without deletion, while downloaded files are deleted after testing.
    """
    # Create the directory for storing test files if it doesn't already exist.
    RBR_FILE_DIR.mkdir(parents=True, exist_ok=True)

    # Track which files are downloaded
    downloaded_files = set()

    # Download each file in RBR_FILE_URLS if it doesn't already exist locally.
    for file_name, url in RBR_FILE_URLS.items():
        file_path = RBR_FILE_DIR / file_name
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
    for file_name in RBR_FILE_URLS.keys():
        file_path = RBR_FILE_DIR / file_name
        if file_path.exists() and file_name in downloaded_files:
            try:
                file_path.unlink()  # Attempt to delete the file
            except PermissionError:  # Handle the case of locked file
                print(f"Failed to delete {file_name}. File might be in use.")

@pytest.mark.parametrize("file_name", [
    "solo_example.rsk",
    "conc_example.rsk",
    "conc_chl_par_example.rsk",
])
def test_load_moored_rbr(file_name, setup_files):
    """
    Test the load_moored function with RBR files.
    The setup_files fixture ensures files are available locally before testing.
    """
    # Full path to the file
    file_path = RBR_FILE_DIR / file_name

    # Call the load_moored function with the file
    ds = load_moored(str(file_path))

    # Perform basic checks on the returned xarray.Dataset
    assert isinstance(ds, xr.Dataset), f"Expected xarray.Dataset, got {type(ds)}"
    assert "PROCESSING" in ds, "PROCESSING variable is missing from the dataset"
    assert "python_script" in ds.PROCESSING.attrs, "python_script attribute is missing from PROCESSING variable"
    assert ds.PROCESSING.attrs["python_script"].strip(), "python_script attribute should not be empty"

    # Additional checks can be added here as needed

def test_load_moored_sbe37(setup_files):
    """
    Test the load_moored function with the SBE37 file.
    """
    # Ensure the SBE37 file exists before running the test
    assert SBE37_FILE_PATH.exists(), f"SBE37 file not found at {SBE37_FILE_PATH}"

    # Call the load_moored function with the SBE37 file
    ds = load_moored(str(SBE37_FILE_PATH))

    # Perform basic checks on the returned xarray.Dataset
    assert isinstance(ds, xr.Dataset), f"Expected xarray.Dataset, got {type(ds)}"
    assert "PROCESSING" in ds, "PROCESSING variable is missing from the dataset"
    assert "python_script" in ds.PROCESSING.attrs, "python_script attribute is missing from PROCESSING variable"
    assert ds.PROCESSING.attrs["python_script"].strip(), "python_script attribute should not be empty"




@pytest.fixture
def example_datasets_assign_pres():
    """Fixture to create example datasets for testing"""
    time = np.array(['2022-01-01T00:00:00', '2022-01-01T01:00:00'], dtype='datetime64[ns]')
    pres_above = xr.DataArray([100.0, 105.0], dims="TIME", coords={"TIME": time})
    pres_below = xr.DataArray([150.0, 155.0], dims="TIME", coords={"TIME": time})
    no_pres = xr.DataArray([np.nan, np.nan], dims="TIME", coords={"TIME": time})

    ds_main = xr.Dataset({"TEMP": no_pres, "LATITUDE": xr.DataArray(60.0)}, coords={"TIME": time})
    ds_above = xr.Dataset({"PRES": pres_above, "LATITUDE": xr.DataArray(60.0)}, coords={"TIME": time})
    ds_below = xr.Dataset({"PRES": pres_below, "LATITUDE": xr.DataArray(60.0)}, coords={"TIME": time})

    return ds_main, ds_above, ds_below

def test_assign_pressure_basic(example_datasets_assign_pres):
    """Test basic functionality of assign_pressure"""
    ds_main, ds_above, ds_below = example_datasets_assign_pres

    nom_dep_main = 50.0
    nom_dep_above = 30.0
    nom_dep_below = 70.0

    result = assign_pressure(ds_main, ds_above, ds_below, nom_dep_main, nom_dep_above, nom_dep_below, auto_accept=True, plot=False)

    # Assert that the pressure is now in the main dataset
    assert "PRES" in result
    assert result["PRES"].dims == ("TIME",)
    assert result["PRES"].shape == ds_main["TIME"].shape
    assert not np.isnan(result["PRES"]).all()  # Ensure pressure was assigned

def test_assign_pressure_missing_lat(example_datasets_assign_pres):
    """Test handling of missing latitude in ds_main"""
    ds_main, ds_above, ds_below = example_datasets_assign_pres

    # Remove latitude from ds_main
    ds_main = ds_main.drop_vars("LATITUDE")

    nom_dep_main = 50.0
    nom_dep_above = 30.0
    nom_dep_below = 70.0

    # Expect an exception since latitude is not provided
    with pytest.raises(Exception, match="Could not find latitude for depth->pressure calculation"):
        assign_pressure(ds_main, ds_above, ds_below, nom_dep_main, nom_dep_above, nom_dep_below, auto_accept=True, plot=False)

@mock.patch('builtins.input', return_value='y')
def test_assign_pressure_manual_accept(mock_input, example_datasets_assign_pres):
    """Test pressure assignment with manual user acceptance (accepts)"""
    ds_main, ds_above, ds_below = example_datasets_assign_pres

    nom_dep_main = 50.0
    nom_dep_above = 30.0
    nom_dep_below = 70.0

    result = assign_pressure(ds_main, ds_above, ds_below, nom_dep_main, nom_dep_above, nom_dep_below, auto_accept=False, plot=False)

    # Assert that the pressure is now in the main dataset
    assert "PRES" in result

@mock.patch('builtins.input', return_value='n')
def test_assign_pressure_manual_reject(mock_input, example_datasets_assign_pres):
    """Test pressure assignment with manual user acceptance (rejects)"""
    ds_main, ds_above, ds_below = example_datasets_assign_pres

    nom_dep_main = 50.0
    nom_dep_above = 30.0
    nom_dep_below = 70.0

    result = assign_pressure(ds_main, ds_above, ds_below, nom_dep_main, nom_dep_above, nom_dep_below, auto_accept=False, plot=False)

    # Assert that the pressure is NOT assigned to the main dataset
    assert "PRES" not in result
