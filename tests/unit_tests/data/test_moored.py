import pytest
import xarray as xr
import requests
from pathlib import Path
import gsw
from kval.data.moored import load_moored, assign_pressure, drop_variables, calculate_PSAL, adjust_time_for_drift, chop_by_time
from unittest import mock
import numpy as np
import re
import pandas as pd

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


# Test assign_pressure

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


# Test drop_variables

@pytest.fixture
def sample_dataset_dropvars():
    """Fixture to create a sample xarray Dataset for testing."""
    time = np.arange(10)
    var1 = np.random.rand(10)
    var2 = np.random.rand(10)
    static_var = 42  # A variable without TIME dimension

    ds = xr.Dataset(
        {
            "var1": (("TIME"), var1),
            "var2": (("TIME"), var2),
            "static_var": ((), static_var)  # Static variable with no TIME dimension
        },
        coords={"TIME": time}
    )
    return ds

def test_drop_vars(sample_dataset_dropvars):
    """Test dropping specific variables using the drop_vars argument."""
    ds = drop_variables(sample_dataset_dropvars, drop_vars=["var1"])
    assert "var1" not in ds
    assert "var2" in ds
    assert "static_var" in ds  # static_var should not be dropped

def test_retain_vars(sample_dataset_dropvars):
    """Test retaining specific variables using the retain_vars argument."""
    ds = drop_variables(sample_dataset_dropvars, retain_vars=["var1"])
    assert "var1" in ds
    assert "var2" not in ds
    assert "static_var" in ds  # static_var should not be dropped

def test_retain_all_vars(sample_dataset_dropvars):
    """Test retaining all variables by setting retain_vars to True."""
    ds = drop_variables(sample_dataset_dropvars, retain_vars=True)
    assert "var1" in ds
    assert "var2" in ds
    assert "static_var" in ds  # All variables should be retained

def test_no_retain_vars(sample_dataset_dropvars):
    """Test that no variables are retained when retain_vars is an empty list."""
    ds = drop_variables(sample_dataset_dropvars, retain_vars=[])
    assert "var1" not in ds
    assert "var2" not in ds
    assert "static_var" in ds  # static_var should not be dropped since it has no TIME dimension

def test_error_if_both_retain_and_drop(sample_dataset_dropvars):
    """Test that an error is raised if both retain_vars and drop_vars are specified."""
    with pytest.raises(ValueError):
        drop_variables(sample_dataset_dropvars, retain_vars=["var1"], drop_vars=["var2"])

def test_verbose_output(capfd, sample_dataset_dropvars):
    """Test verbose output when dropping variables."""
    drop_variables(sample_dataset_dropvars, drop_vars=["var1"], verbose=True)
    captured = capfd.readouterr()
    assert "Dropped variables from the Dataset: ['var1']" in captured.out

def test_no_action(sample_dataset_dropvars):
    """Test that nothing happens if neither retain_vars nor drop_vars is specified."""
    ds = drop_variables(sample_dataset_dropvars)
    assert ds.equals(sample_dataset_dropvars)  # The dataset should remain unchanged

# Test Calculate_psal

def test_calculate_psal():
    # Create a mock dataset with CNDC, TEMP, and PRES variables
    data = {
        "CNDC": (["TIME", "DEPTH"], np.random.rand(10, 5) * 3.0),  # Random conductivity data
        "TEMP": (["TIME", "DEPTH"], np.random.rand(10, 5) * 20.0),  # Random temperature data (°C)
        "PRES": (["TIME", "DEPTH"], np.random.rand(10, 5) * 500.0), # Random pressure data (dbar)
        "PSAL": (["TIME", "DEPTH"], np.zeros((10, 5))),             # Placeholder salinity (PSAL)
    }

    ds = xr.Dataset(
        data,
        coords={"TIME": np.arange(10), "DEPTH": np.arange(5)},
        attrs={"title": "Test Dataset"}
    )

    # Set expected salinity using gsw directly for comparison
    expected_psal = gsw.SP_from_C(ds["CNDC"], ds["TEMP"], ds["PRES"])

    # Call the function to calculate PSAL
    ds_updated = calculate_PSAL(ds)

    # Assert that the PSAL variable was updated correctly
    np.testing.assert_allclose(ds_updated["PSAL"], expected_psal, rtol=1e-5)

    # Check that the attributes are updated
    assert "note" in ds_updated["PSAL"].attrs
    assert "Python gsw module" in ds_updated["PSAL"].attrs["note"]

    # Ensure no other variables were altered
    assert ds["CNDC"].equals(ds_updated["CNDC"])
    assert ds["TEMP"].equals(ds_updated["TEMP"])
    assert ds["PRES"].equals(ds_updated["PRES"])

    # Ensure PSAL has the same shape and type as before
    assert ds_updated["PSAL"].shape == (10, 5)
    assert isinstance(ds_updated["PSAL"].values, np.ndarray)

# Test adjust_time_for_drift


@pytest.fixture
def sample_dataset_drift():
    """Create a sample xarray Dataset for testing."""
    time_values = np.arange(0, 10)  # Example time values
    data = np.random.rand(10)  # Random data for testing
    ds = xr.Dataset({
        'data_var': ('TIME', data)
    })
    ds.coords['TIME'] = ('TIME', time_values)
    ds['TIME'].attrs['units'] = 'days since 1970-01-01'
    return ds

@pytest.fixture
def sample_dataset_drift_bad_time_units():
    """Create a sample xarray Dataset for testing."""
    time_values = np.arange(0, 10)  # Example time values
    data = np.random.rand(10)  # Random data for testing
    ds = xr.Dataset({
        'data_var': ('TIME', data)
    })
    ds.coords['TIME'] = ('TIME', time_values)
    ds['TIME'].attrs['units'] = 'not a valid unit'
    return ds






def test_adjust_time_for_positive_drift(sample_dataset_drift):
    """Test positive clock drift adjustment."""
    ds = adjust_time_for_drift(sample_dataset_drift, seconds=30)
    assert ds['TIME'].values[-1] == pytest.approx(sample_dataset_drift['TIME'].values[-1] - (30 / 86400), rel=1e-2)
    assert ds['TIME'].values[0] == sample_dataset_drift['TIME'].values[0]

def test_adjust_time_for_negative_drift(sample_dataset_drift):
    """Test negative clock drift adjustment."""
    ds = adjust_time_for_drift(sample_dataset_drift, seconds=-30)
    assert ds['TIME'].values[-1] == pytest.approx(sample_dataset_drift['TIME'].values[-1] + (30 / 86400), rel=1e-2)
    assert ds['TIME'].values[0] == sample_dataset_drift['TIME'].values[0]

def test_adjust_time_for_combined_drift(sample_dataset_drift):
    """Test combined clock drift adjustment."""
    ds = adjust_time_for_drift(sample_dataset_drift, minutes=1, seconds=-30)
    total_adjustment = (60 / 86400) - (30 / 86400)  # 1 minute in seconds + (30 seconds adjustment)
    assert ds['TIME'].values[-1] == pytest.approx(sample_dataset_drift['TIME'].values[-1] - total_adjustment, rel=1e-2)
    assert ds['TIME'].values[0] == sample_dataset_drift['TIME'].values[0]


def test_zero_drift_prints_warning(sample_dataset_drift):
    """Test that an exception is raised for zero drift."""
    nz_drift_msg = (
        'To adjust for clock drift, a non-zero clock drift'
        ' has to be specified -> Doing nothing')

    with pytest.warns(UserWarning, match=nz_drift_msg):
        adjust_time_for_drift(sample_dataset_drift, seconds=0)

def test_invalid_time_units(sample_dataset_drift_bad_time_units):
    """Test that an exception is raised for non-numerical TIME."""
    with pytest.raises(Exception,
                       match=re.escape(
            'Could not add drift because TIME is non-numerical'
            ' or has unknown units (should be "Days since..")')):

        adjust_time_for_drift(sample_dataset_drift_bad_time_units,
                              seconds=10)

def test_empty_dataset():
    """Test adjustment on an empty dataset."""
    ds = xr.Dataset()
    ds.coords['TIME'] = ('TIME', [])
    ds['TIME'].attrs['units'] = 'days since 1970-01-01'

    adjusted_ds = adjust_time_for_drift(ds, seconds=10)
    assert adjusted_ds['TIME'].size == 0  # The size should still be zero



#### Test chop_by_time
@pytest.fixture
def sample_dataset_chopbytime():
    """Fixture to create a sample xarray Dataset for chop_by_time testing."""
    time = np.array([
        '2022-01-01T00:00:00', '2022-01-01T01:00:00', '2022-01-01T02:00:00',
        '2022-01-01T03:00:00', '2022-01-01T04:00:00'
    ], dtype='datetime64[ns]')
    data_var = np.random.rand(5)
    ds = xr.Dataset(
        {"data_var": ("TIME", data_var)},
        coords={"TIME": time}
    )
    return ds

def test_chop_by_time_basic(sample_dataset_chopbytime):
    """Test basic functionality of chop_by_time."""
    # Define start_time and end_time for chopping
    start_time = '2022-01-01T01:00:00'
    end_time = '2022-01-01T03:00:00'

    # Call chop_by_time
    result = chop_by_time(sample_dataset_chopbytime, start_time=start_time, end_time=end_time)

    # Assert that the dataset was properly cropped
    expected_times = np.array([
        '2022-01-01T01:00:00', '2022-01-01T02:00:00', '2022-01-01T03:00:00'
    ], dtype='datetime64[ns]')

    assert np.array_equal(result.TIME.values, expected_times), "The time range was not cropped correctly."
    assert result.sizes['TIME'] == 3, "Expected 3 time steps after chopping."

def test_chop_by_time_no_start_time(sample_dataset_chopbytime):
    """Test chop_by_time with no start_time."""
    end_time = '2022-01-01T02:00:00'

    # Call chop_by_time with no start_time (removes all data before end_time)
    result = chop_by_time(sample_dataset_chopbytime, end_time=end_time)

    expected_times = np.array([
        '2022-01-01T00:00:00', '2022-01-01T01:00:00', '2022-01-01T02:00:00'
    ], dtype='datetime64[ns]')

    assert np.array_equal(result.TIME.values, expected_times), "The dataset was not chopped correctly when no start_time was provided."
    assert result.sizes['TIME'] == 3, "Expected 3 time steps after chopping."


def test_chop_by_time_no_end_time(sample_dataset_chopbytime):
    """Test chop_by_time with no end_time."""
    start_time = '2022-01-01T02:00:00'

    # Call chop_by_time with no end_time (removes all data after start_time)
    result = chop_by_time(sample_dataset_chopbytime, start_time=start_time)

    expected_times = np.array([
        '2022-01-01T02:00:00', '2022-01-01T03:00:00', '2022-01-01T04:00:00'
    ], dtype='datetime64[ns]')

    assert np.array_equal(result.TIME.values, expected_times), "The dataset was not chopped correctly when no end_time was provided."
    assert result.sizes['TIME'] == 3, "Expected 3 time steps after chopping."

def test_chop_by_time_no_times(sample_dataset_chopbytime):
    """Test chop_by_time with no start_time and end_time (should return the same dataset)."""
    result = chop_by_time(sample_dataset_chopbytime)

    assert result.equals(sample_dataset_chopbytime), "The dataset should remain unchanged when no start_time and end_time are provided."
