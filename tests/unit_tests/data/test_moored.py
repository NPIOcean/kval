import pytest
import xarray as xr
import requests
from pathlib import Path
from kval.data.moored import load_moored

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

    # Additional checks can be added here as needed
