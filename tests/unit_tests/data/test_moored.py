import pytest
import xarray as xr
import requests
from pathlib import Path
import gsw
from kval.data.moored import load_moored, assign_pressure, drop_variables, calculate_PSAL, adjust_time_for_drift, chop_by_time, combine_datasets
from unittest import mock
import numpy as np
import re
import pandas as pd
import time

# Define the URLs for the files you want to test
RBR_FILE_URLS = {
    "conc_chl_par_example.rsk": "https://zenodo.org/records/13321317/files/conc_chl_par_example.rsk?download=1",
    "conc_example.rsk": "https://zenodo.org/records/13321317/files/conc_example.rsk?download=1",
    "solo_example.rsk": "https://zenodo.org/records/13321317/files/solo_example.rsk?download=1",
}

# Define the directory where files should be stored
RBR_FILE_DIR = Path("tests/test_data/rbr_files")
SBE37_FILE_PATH = Path("tests/test_data/sbe_files/sbe37/cnv/test_sbe37.cnv")
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
    Only downloads the specific file the requesting test needs, so a
    transient failure fetching one file doesn't fail tests that don't
    need it. Local files are reused without re-downloading; only files
    this fixture itself downloaded are cleaned up afterward.
    """
    file_name = request.param
    RBR_FILE_DIR.mkdir(parents=True, exist_ok=True)
    file_path = RBR_FILE_DIR / file_name
    was_downloaded = False

    if not file_path.exists():
        try:
            content = _download_with_retry(RBR_FILE_URLS[file_name])
        except requests.exceptions.HTTPError as e:
            pytest.skip(f"Could not download {file_name} from Zenodo: {e}")
        with open(file_path, "wb") as file:
            file.write(content)
        was_downloaded = True

    yield file_path

    if was_downloaded and file_path.exists():
        try:
            file_path.unlink()
        except PermissionError:
            print(f"Failed to delete {file_name}. File might be in use.")


@pytest.mark.parametrize("rbr_file", [
    "solo_example.rsk",
    "conc_example.rsk",
    "conc_chl_par_example.rsk",
], indirect=True)
def test_load_moored_rbr(rbr_file):
    """
    Test the load_moored function with RBR files.
    The rbr_file fixture ensures only the specific file needed is
    downloaded before this test runs.
    """
    ds = load_moored(str(rbr_file))
    assert isinstance(ds, xr.Dataset), f"Expected xarray.Dataset, got {type(ds)}"

def test_load_moored_sbe37():
    """
    Test the load_moored function with the SBE37 file.
    """
    assert SBE37_FILE_PATH.exists(), f"SBE37 file not found at {SBE37_FILE_PATH}"
    ds = load_moored(str(SBE37_FILE_PATH))
    assert isinstance(ds, xr.Dataset), f"Expected xarray.Dataset, got {type(ds)}"


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
    ds = drop_variables(sample_dataset_dropvars, drop=["var1"])
    assert "var1" not in ds
    assert "var2" in ds
    assert "static_var" in ds  # static_var should not be dropped

def test_retain_vars(sample_dataset_dropvars):
    """Test retaining specific variables using the retain_vars argument."""
    ds = drop_variables(sample_dataset_dropvars, retain=["var1"])
    assert "var1" in ds
    assert "var2" not in ds
    assert "static_var" in ds  # static_var should not be dropped

def test_retain_all_vars(sample_dataset_dropvars):
    """Test retaining all variables by setting retain_vars to True."""
    ds = drop_variables(sample_dataset_dropvars, retain=True)
    assert "var1" in ds
    assert "var2" in ds
    assert "static_var" in ds  # All variables should be retained

def test_no_retain_vars(sample_dataset_dropvars):
    """Test that no variables are retained when retain_vars is an empty list."""
    ds = drop_variables(sample_dataset_dropvars, retain=[])
    assert "var1" not in ds
    assert "var2" not in ds
    assert "static_var" in ds  # static_var should not be dropped since it has no TIME dimension

def test_error_if_both_retain_and_drop(sample_dataset_dropvars):
    """Test that an error is raised if both retain_vars and drop_vars are specified."""
    with pytest.raises(ValueError):
        drop_variables(sample_dataset_dropvars, retain=["var1"], drop=["var2"])

def test_verbose_output(capfd, sample_dataset_dropvars):
    """Test verbose output when dropping variables."""
    drop_variables(sample_dataset_dropvars, drop=["var1"], verbose=True)
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
    assert "processing_history" in ds_updated["PSAL"].attrs
    assert "Python gsw module" in ds_updated["PSAL"].attrs["processing_history"]

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


# Note: ignoring the warning - don't want to see it when testing
@pytest.mark.filterwarnings("ignore:TIME coordinate is empty -> Doing nothing:UserWarning")
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


## Soime tests of the timea djoustment function..

@pytest.fixture
def sample_dataset_drift_uneven():
    """Dataset with gaps/uneven spacing in TIME, to test that drift
    correction is linear in elapsed time rather than sample index."""
    # Uneven spacing: gaps between points 1 and 2
    time_values = np.array([0.0, 1.0, 5.0, 6.0])  # days
    data = np.random.rand(4)
    ds = xr.Dataset({'data_var': ('TIME', data)})
    ds.coords['TIME'] = ('TIME', time_values)
    ds['TIME'].attrs['units'] = 'days since 1970-01-01'
    return ds


@pytest.fixture
def sample_dataset_drift_unsorted():
    """Dataset with TIME out of order."""
    time_values = np.array([0.0, 2.0, 1.0, 3.0])
    data = np.random.rand(4)
    ds = xr.Dataset({'data_var': ('TIME', data)})
    ds.coords['TIME'] = ('TIME', time_values)
    ds['TIME'].attrs['units'] = 'days since 1970-01-01'
    return ds


@pytest.fixture
def sample_dataset_drift_single_time():
    """Dataset with only one (or duplicate) TIME value."""
    time_values = np.array([5.0, 5.0])
    data = np.random.rand(2)
    ds = xr.Dataset({'data_var': ('TIME', data)})
    ds.coords['TIME'] = ('TIME', time_values)
    ds['TIME'].attrs['units'] = 'days since 1970-01-01'
    return ds


@pytest.fixture
def sample_dataset_drift_no_units():
    """Dataset with TIME missing the 'units' attribute entirely."""
    time_values = np.arange(0, 10)
    data = np.random.rand(10)
    ds = xr.Dataset({'data_var': ('TIME', data)})
    ds.coords['TIME'] = ('TIME', time_values)
    return ds


def test_adjust_time_for_drift_uneven_spacing(sample_dataset_drift_uneven):
    """Drift correction should scale with elapsed TIME, not sample index --
    the midpoint here (index 1) is NOT at the midpoint in time, so the
    correction at index 1 should reflect that."""
    total_drift = 60  # seconds
    ds = adjust_time_for_drift(sample_dataset_drift_uneven, seconds=total_drift)

    orig_time = sample_dataset_drift_uneven['TIME'].values
    span = orig_time[-1] - orig_time[0]  # 6.0 days

    for i in range(len(orig_time)):
        frac = (orig_time[i] - orig_time[0]) / span
        expected = orig_time[i] - (frac * total_drift) / 86400
        assert ds['TIME'].values[i] == pytest.approx(expected, rel=1e-6)

    # Explicitly confirm index-1 correction does NOT match a naive
    # index-based (1/3 of total) calculation, since elapsed time to
    # index 1 is only 1/6 of the total span, not 1/3
    naive_index_based = orig_time[1] - ((1 / 3) * total_drift) / 86400
    assert ds['TIME'].values[1] != pytest.approx(naive_index_based, rel=1e-6)


def test_adjust_time_for_drift_missing_units_raises(sample_dataset_drift_no_units):
    """TIME with no 'units' attribute should raise a clear error."""
    with pytest.raises(Exception, match='has no "units" attribute'):
        adjust_time_for_drift(sample_dataset_drift_no_units, seconds=10)


def test_adjust_time_for_drift_leaves_existing_comment_untouched(sample_dataset_drift):
    """A pre-existing 'comment' attribute is unrelated to processing_history
    and should be left completely alone."""
    sample_dataset_drift['TIME'].attrs['comment'] = 'Pre-existing comment'
    ds = adjust_time_for_drift(sample_dataset_drift, seconds=30)
    assert ds['TIME'].attrs['comment'] == 'Pre-existing comment'

def test_adjust_time_for_drift_processing_history_created(sample_dataset_drift):
    ds = adjust_time_for_drift(sample_dataset_drift, seconds=30)
    assert 'Adjusted for observed clock drift' in ds['TIME'].attrs['processing_history']

def test_adjust_time_for_drift_processing_history_accumulates(sample_dataset_drift):
    ds_step1 = adjust_time_for_drift(sample_dataset_drift, seconds=30)
    ds_step2 = adjust_time_for_drift(ds_step1, seconds=-10)
    history = ds_step2['TIME'].attrs['processing_history']
    assert history.count('Adjusted for observed clock drift') == 2

def test_adjust_time_for_drift_unsorted_raises(sample_dataset_drift_unsorted):
    with pytest.raises(Exception, match="not sorted in non-decreasing order"):
        adjust_time_for_drift(sample_dataset_drift_unsorted, seconds=10)

def test_adjust_time_for_drift_zero_span_raises(sample_dataset_drift_single_time):
    with pytest.raises(Exception, match="are identical"):
        adjust_time_for_drift(sample_dataset_drift_single_time, seconds=10)


@pytest.fixture
def sample_dataset_drift_datetime_units():
    """Dataset with TIME as days-since-epoch floats, spanning several days,
    for testing custom start_time/end_time."""
    # 6 points, daily, starting 2020-01-01
    time_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])  # days since 2020-01-01
    data = np.random.rand(6)
    ds = xr.Dataset({'data_var': ('TIME', data)})
    ds.coords['TIME'] = ('TIME', time_values)
    ds['TIME'].attrs['units'] = 'days since 2020-01-01'
    return ds


def test_adjust_time_for_drift_custom_start_end_within_range(
        sample_dataset_drift_datetime_units):
    """start_time/end_time set to the 2nd and 5th points (not the actual
    first/last TIME values) -- drift should be 0 at start_time, full offset
    at end_time, and linearly interpolated/extrapolated elsewhere."""
    total_drift = 120  # seconds

    ds = adjust_time_for_drift(
        sample_dataset_drift_datetime_units,
        seconds=total_drift,
        start_time='2020-01-02 00:00',  # = TIME index 1 (1.0 days)
        end_time='2020-01-05 00:00',    # = TIME index 4 (4.0 days)
    )

    orig_time = sample_dataset_drift_datetime_units['TIME'].values
    start_num, end_num = 1.0, 4.0
    span = end_num - start_num  # 3.0 days

    for i in range(len(orig_time)):
        frac = (orig_time[i] - start_num) / span
        expected = orig_time[i] - (frac * total_drift) / 86400
        assert ds['TIME'].values[i] == pytest.approx(expected, rel=1e-6)

    # Sanity: drift at start_time (index 1) should be exactly zero, i.e.
    # TIME unchanged there
    assert ds['TIME'].values[1] == pytest.approx(orig_time[1], abs=1e-9)

    # Sanity: drift at end_time (index 4) should be the full offset
    assert ds['TIME'].values[4] == pytest.approx(
        orig_time[4] - total_drift / 86400, rel=1e-6)


def test_adjust_time_for_drift_extrapolates_outside_window(
        sample_dataset_drift_datetime_units):
    """Points before start_time or after end_time should extrapolate
    linearly at the same drift rate, not clamp to 0 or the full offset."""
    total_drift = 120  # seconds

    ds = adjust_time_for_drift(
        sample_dataset_drift_datetime_units,
        seconds=total_drift,
        start_time='2020-01-02 00:00',  # index 1
        end_time='2020-01-05 00:00',    # index 4
    )

    orig_time = sample_dataset_drift_datetime_units['TIME'].values
    start_num, end_num = 1.0, 4.0
    span = end_num - start_num

    # index 0 is before start_time -> frac should be negative (extrapolated
    # backward), not clamped to 0
    frac_0 = (orig_time[0] - start_num) / span
    assert frac_0 < 0
    expected_0 = orig_time[0] - (frac_0 * total_drift) / 86400
    assert ds['TIME'].values[0] == pytest.approx(expected_0, rel=1e-6)
    assert ds['TIME'].values[0] != pytest.approx(orig_time[0], abs=1e-9)  # not zero drift

    # index 5 is after end_time -> frac should exceed 1 (extrapolated
    # forward), not clamped to the full offset
    frac_5 = (orig_time[5] - start_num) / span
    assert frac_5 > 1
    expected_5 = orig_time[5] - (frac_5 * total_drift) / 86400
    assert ds['TIME'].values[5] == pytest.approx(expected_5, rel=1e-6)


def test_adjust_time_for_drift_default_start_end_matches_first_last(
        sample_dataset_drift_datetime_units):
    """With no start_time/end_time given, behavior should match using the
    first/last TIME values directly (i.e. same as before this feature)."""
    total_drift = 90

    ds_explicit = adjust_time_for_drift(
        sample_dataset_drift_datetime_units.copy(deep=True),
        seconds=total_drift,
        start_time='2020-01-01 00:00',  # = first TIME value
        end_time='2020-01-06 00:00',    # = last TIME value
    )
    ds_default = adjust_time_for_drift(
        sample_dataset_drift_datetime_units.copy(deep=True),
        seconds=total_drift,
    )

    np.testing.assert_allclose(
        ds_explicit['TIME'].values, ds_default['TIME'].values, rtol=1e-6)


def test_adjust_time_for_drift_bad_start_time_string_raises(
        sample_dataset_drift_datetime_units):
    """An unparseable start_time string should raise a clear error."""
    with pytest.raises(Exception, match='Could not parse start_time'):
        adjust_time_for_drift(
            sample_dataset_drift_datetime_units,
            seconds=10,
            start_time='not a real timestamp',
        )


def test_adjust_time_for_drift_bad_end_time_string_raises(
        sample_dataset_drift_datetime_units):
    """An unparseable end_time string should raise a clear error."""
    with pytest.raises(Exception, match='Could not parse end_time'):
        adjust_time_for_drift(
            sample_dataset_drift_datetime_units,
            seconds=10,
            end_time='also not a timestamp',
        )


def test_adjust_time_for_drift_identical_start_end_raises(
        sample_dataset_drift_datetime_units):
    """start_time == end_time should raise (zero anchor span)."""
    with pytest.raises(Exception, match='are identical'):
        adjust_time_for_drift(
            sample_dataset_drift_datetime_units,
            seconds=10,
            start_time='2020-01-02 00:00',
            end_time='2020-01-02 00:00',
        )


#### Test combine_datasets

@pytest.fixture
def ds_instr1_combine():
    """Higher-frequency instrument (every 2h), has TEMP and PRES."""
    t = pd.date_range('2024-01-01 00:00', periods=10, freq='2h')
    ds = xr.Dataset(
        {
            'TEMP': ('TIME', np.linspace(10, 12, len(t))),
            'PRES': ('TIME', np.linspace(100, 105, len(t))),
        },
        coords={'TIME': t},
        attrs={'instrument_serial_number': '12345', 'mooring_name': 'M1'},
    )
    ds['TEMP'].attrs = {'units': 'degC'}
    ds['PRES'].attrs = {'units': 'dbar'}
    return ds


@pytest.fixture
def ds_instr2_combine():
    """Lower-frequency instrument (every 3h), different start/end, has
    TEMP only (no PRES)."""
    t = pd.date_range('2024-01-01 01:00', periods=8, freq='3h')
    ds = xr.Dataset(
        {'TEMP': ('TIME', np.linspace(9, 11, len(t)))},
        coords={'TIME': t},
        attrs={'instrument_serial_number': '67890', 'mooring_name': 'M1'},
    )
    ds['TEMP'].attrs = {'units': 'degC'}
    return ds


def test_combine_datasets_basic_shape(ds_instr1_combine, ds_instr2_combine):
    """Combined dataset should have an INSTR dim of length 2 and TEMP
    stacked as (INSTR, TIME)."""
    out = combine_datasets(ds_instr1_combine, ds_instr2_combine, interval='1h')
    assert out.sizes['INSTR'] == 2
    assert out['TEMP'].dims == ('INSTR', 'TIME')


def test_combine_datasets_auto_labels_from_serial_number(ds_instr1_combine, ds_instr2_combine):
    out = combine_datasets(ds_instr1_combine, ds_instr2_combine, interval='1h')
    assert list(out.INSTR.values) == ['12345', '67890']


def test_combine_datasets_default_method_leaves_sampling_gaps_nan(ds_instr1_combine, ds_instr2_combine):
    """ds_instr1 samples every 2h starting at 00:00, so on an hourly grid
    (default method='time_average') the 01:00 bin has no sample -> NaN."""
    out = combine_datasets(ds_instr1_combine, ds_instr2_combine, interval='1h')
    val = out['TEMP'].sel(INSTR='12345', TIME='2024-01-01 01:00').item()
    assert np.isnan(val)


def test_combine_datasets_interpolate_fills_sampling_gaps(ds_instr1_combine, ds_instr2_combine):
    """The same slot should be filled under method='interpolate', since it
    has real neighboring samples."""
    out = combine_datasets(ds_instr1_combine, ds_instr2_combine, interval='1h', method='interpolate')
    val = out['TEMP'].sel(INSTR='12345', TIME='2024-01-01 01:00').item()
    assert not np.isnan(val)


def test_combine_datasets_missing_variable_is_nan_for_other_instrument(ds_instr1_combine, ds_instr2_combine):
    """PRES only exists in ds_instr1 -> should be all-NaN for ds_instr2."""
    out = combine_datasets(ds_instr1_combine, ds_instr2_combine, interval='1h')
    assert np.all(np.isnan(out['PRES'].sel(INSTR='67890').values))
    assert not np.all(np.isnan(out['PRES'].sel(INSTR='12345').values))


def test_combine_datasets_shared_attr_kept_even_if_var_missing_elsewhere(ds_instr1_combine, ds_instr2_combine):
    """PRES.units should stay a shared attribute even though only one of
    the two instruments has a PRES variable."""
    out = combine_datasets(ds_instr1_combine, ds_instr2_combine, interval='1h')
    assert out['PRES'].attrs.get('units') == 'dbar'
    assert 'PRES_units' not in out.coords


def test_combine_datasets_shared_global_attr_kept(ds_instr1_combine, ds_instr2_combine):
    out = combine_datasets(ds_instr1_combine, ds_instr2_combine, interval='1h')
    assert out.attrs.get('mooring_name') == 'M1'
    assert 'instrument_serial_number' in out.coords  # differs -> per-instrument coord


def test_combine_datasets_explicit_instr_names(ds_instr1_combine, ds_instr2_combine):
    out = combine_datasets(
        ds_instr1_combine, ds_instr2_combine, interval='1h',
        instr_names=['upper', 'lower'])
    assert list(out.INSTR.values) == ['upper', 'lower']


def test_combine_datasets_too_few_datasets_raises(ds_instr1_combine):
    with pytest.raises(ValueError, match='at least 2'):
        combine_datasets(ds_instr1_combine, interval='1h')


def test_combine_datasets_invalid_method_raises(ds_instr1_combine, ds_instr2_combine):
    with pytest.raises(ValueError, match='method must be'):
        combine_datasets(ds_instr1_combine, ds_instr2_combine, interval='1h', method='bogus')

def test_combine_datasets_duplicate_labels_warns_not_raises(
        ds_instr1_combine, ds_instr2_combine):
    """Duplicate INSTR labels should warn, not raise -- e.g. useful for
    testing self-correlation by combining a dataset with itself."""
    with pytest.warns(UserWarning, match='not unique'):
        out = combine_datasets(
            ds_instr1_combine, ds_instr2_combine, interval='1h',
            instr_names=['dup', 'dup'])
    assert out.sizes['INSTR'] == 2
    assert list(out.INSTR.values) == ['dup', 'dup']


def test_combine_datasets_instr_names_wrong_length_raises(
        ds_instr1_combine, ds_instr2_combine):
    with pytest.raises(ValueError, match='entries but'):
        combine_datasets(
            ds_instr1_combine, ds_instr2_combine, interval='1h',
            instr_names=['only_one'])


def test_combine_datasets_missing_time_coord_raises(
        ds_instr1_combine, ds_instr2_combine):
    ds_no_time = ds_instr1_combine.rename({'TIME': 'OTHER_TIME'})
    with pytest.raises(ValueError, match="missing 'TIME'"):
        combine_datasets(ds_no_time, ds_instr2_combine, interval='1h')


def test_combine_datasets_numeric_time_no_units_raises(
        ds_instr1_combine):
    ds_bad = xr.Dataset(
        {'TEMP': ('TIME', np.linspace(10, 12, 5))},
        coords={'TIME': np.arange(5, dtype=float)},
    )
    with pytest.raises(ValueError, match="no 'units' attribute"):
        combine_datasets(ds_instr1_combine, ds_bad, interval='1h')


def test_combine_datasets_interpolate_drops_non_numeric_var(
        ds_instr1_combine, ds_instr2_combine, capsys):
    ds1 = ds_instr1_combine.copy(deep=True)
    ds1['FLAG'] = ('TIME', ['a'] * ds1.sizes['TIME'])
    out = combine_datasets(
        ds1, ds_instr2_combine, interval='1h', method='interpolate')
    assert 'FLAG' not in out.data_vars
    captured = capsys.readouterr()
    assert 'dropped non-numeric' in captured.out


def test_combine_datasets_varying_var_attr_becomes_coordinate(
        ds_instr1_combine, ds_instr2_combine):
    ds1 = ds_instr1_combine.copy(deep=True)
    ds1['TEMP'].attrs['sensor_calibration_date'] = '2022-01-01'
    ds2 = ds_instr2_combine.copy(deep=True)
    ds2['TEMP'].attrs['sensor_calibration_date'] = '2023-06-01'
    out = combine_datasets(ds1, ds2, interval='1h')
    assert 'TEMP_sensor_calibration_date' in out.coords
    assert list(out['TEMP_sensor_calibration_date'].values) == [
        '2022-01-01', '2023-06-01']


def test_combine_datasets_scalar_variable_stacked_by_instr(
        ds_instr1_combine, ds_instr2_combine):
    ds1 = ds_instr1_combine.copy(deep=True)
    ds1['LATITUDE'] = 78.5
    ds2 = ds_instr2_combine.copy(deep=True)
    ds2['LATITUDE'] = 78.6
    out = combine_datasets(ds1, ds2, interval='1h')
    assert out['LATITUDE'].dims == ('INSTR',)
    np.testing.assert_allclose(out['LATITUDE'].values, [78.5, 78.6])