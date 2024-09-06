import pytest
import xarray as xr
import numpy as np
from kval.signal import filt  # Update this path to your actual import

@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    time = np.arange(10)
    temp = np.sin(time)  # Some synthetic data (a sine wave)

    ds = xr.Dataset(
        {
            'temperature': (['time'], temp)
        },
        coords={
            'time': time
        }
    )
    return ds

@pytest.fixture
def sample_dataset_with_nan():
    """Create a sample dataset for testing with NaNs."""
    time = np.arange(10)
    temp = np.sin(time)
    temp[3] = np.nan  # Introduce a NaN value

    ds = xr.Dataset(
        {
            'temperature': (['time'], temp)
        },
        coords={
            'time': time
        }
    )
    return ds



@pytest.fixture
def sample_dataset_for_sd():
    """Create a sample xarray Dataset for testing."""
    np.random.seed(0)
    time = np.arange(100)
    data = np.sin(time / 10) + np.random.normal(0, 0.1, size=time.shape)
    ds = xr.Dataset({"signal": (["time"], data)}, coords={"time": time})
    return ds



def test_rolling_mean(sample_dataset):
    """Test the rolling mean filter."""
    window_size = 3
    filtered_ds = filt.rolling(sample_dataset, var_name='temperature', dim='time', window_size=window_size, filter_type='mean')

    # Assert that the dimensions are the same
    assert filtered_ds['temperature'].dims == sample_dataset['temperature'].dims

    # Assert that the size of the filtered variable is the same
    assert filtered_ds['temperature'].shape == sample_dataset['temperature'].shape

    # Assert that the rolling operation was applied (check for non-NaN values in the middle)
    middle_idx = len(sample_dataset['time']) // 2
    assert not np.isnan(filtered_ds['temperature'].values[middle_idx])

    # Test values for correctness
    test_idxs = [4, 7]
    for idx in test_idxs:
        expected = np.nanmean(sample_dataset['temperature'].values[idx-1:idx+2])
        assert np.isclose(filtered_ds['temperature'][idx].values, expected)

def test_rolling_median(sample_dataset):
    """Test the rolling median filter."""
    window_size = 3
    filtered_ds = filt.rolling(sample_dataset, var_name='temperature', dim='time', window_size=window_size, filter_type='median')

    # Assert that the dimensions are the same
    assert filtered_ds['temperature'].dims == sample_dataset['temperature'].dims

    # Assert that the size of the filtered variable is the same
    assert filtered_ds['temperature'].shape == sample_dataset['temperature'].shape

    # Assert that the rolling operation was applied (check for non-NaN values in the middle)
    middle_idx = len(sample_dataset['time']) // 2
    assert not np.isnan(filtered_ds['temperature'].values[middle_idx])

    # Test values for correctness
    test_idxs = [4, 7]
    for idx in test_idxs:
        expected = np.nanmedian(sample_dataset['temperature'].values[idx-1:idx+2])
        assert np.isclose(filtered_ds['temperature'][idx].values, expected)

def test_rolling_mean_with_nans(sample_dataset_with_nan):
    """Test the rolling nanmean filter with NaNs."""
    filtered_ds_nonan = filt.rolling(sample_dataset_with_nan, var_name='temperature', dim='time', window_size=3, filter_type='mean', nan_edges=True)
    filtered_ds_yesnan = filt.rolling(sample_dataset_with_nan, var_name='temperature', dim='time', window_size=3, min_periods=1, filter_type='mean', nan_edges=True)

    # Assert that the dimensions are the same
    assert filtered_ds_nonan['temperature'].dims == sample_dataset_with_nan['temperature'].dims
    assert filtered_ds_yesnan['temperature'].dims == sample_dataset_with_nan['temperature'].dims

    # Assert that the size of the filtered variable is the same
    assert filtered_ds_nonan['temperature'].shape == sample_dataset_with_nan['temperature'].shape
    assert filtered_ds_yesnan['temperature'].shape == sample_dataset_with_nan['temperature'].shape

    # Ensure NaNs are handled
    assert np.isnan(filtered_ds_nonan['temperature'].values[3])  # Expect NaN where input has NaN
    assert not np.isnan(filtered_ds_yesnan['temperature'].values[3])  # .. but not if min_periods = 1

    # Test values for correctness (ignoring NaNs)
    idx = 4
    assert np.isnan(filtered_ds_nonan['temperature'][idx].values)
    expected_yesnan = np.nanmean(sample_dataset_with_nan['temperature'].values[idx-1:idx+2])
    assert np.isclose(filtered_ds_yesnan['temperature'][idx].values, expected_yesnan)


def test_rolling_nanmedian(sample_dataset_with_nan):
    """Test the rolling nanmedian filter with NaNs."""
    filtered_ds_nonan = filt.rolling(sample_dataset_with_nan, var_name='temperature', dim='time', window_size=3, filter_type='median', nan_edges=True)
    filtered_ds_yesnan = filt.rolling(sample_dataset_with_nan, var_name='temperature', dim='time', window_size=3, min_periods=1, filter_type='median', nan_edges=True)

    # Assert that the dimensions are the same
    assert filtered_ds_nonan['temperature'].dims == sample_dataset_with_nan['temperature'].dims
    assert filtered_ds_yesnan['temperature'].dims == sample_dataset_with_nan['temperature'].dims

    # Assert that the size of the filtered variable is the same
    assert filtered_ds_nonan['temperature'].shape == sample_dataset_with_nan['temperature'].shape
    assert filtered_ds_yesnan['temperature'].shape == sample_dataset_with_nan['temperature'].shape

    # Ensure NaNs are handled
    assert np.isnan(filtered_ds_nonan['temperature'].values[3])  # Expect NaN where input has NaN
    assert not np.isnan(filtered_ds_yesnan['temperature'].values[3])  # .. but not if min_periods = 1

    # Test values for correctness (ignoring NaNs)
    idx = 4
    assert np.isnan(filtered_ds_nonan['temperature'][idx].values)
    expected_yesnan = np.nanmedian(sample_dataset_with_nan['temperature'].values[idx-1:idx+2])
    assert np.isclose(filtered_ds_yesnan['temperature'][idx].values, expected_yesnan)






def test_rolling_sd_basic(sample_dataset_for_sd):
    """Test basic rolling standard deviation computation."""
    ds = sample_dataset_for_sd
    result = filt.rolling_sd(ds, var_name="signal", dim="time", window_size=5)

    assert isinstance(result, xr.DataArray), "Result should be a DataArray."
    assert result.count() > 90, "Most values should be non-NaN."

def test_rolling_sd_nan_edges(sample_dataset_for_sd):
    """Test rolling standard deviation with nan_edges=True."""
    ds = sample_dataset_for_sd
    window_size = 5
    result = filt.rolling_sd(ds, var_name="signal", dim="time", window_size=window_size, nan_edges=True)

    halfwidth = int(np.ceil(window_size / 2))

    # Check that the edges are NaN
    assert result.isel(time=slice(0, halfwidth)).isnull().all(), "The first half-window should be NaN."
    assert result.isel(time=slice(-halfwidth, None)).isnull().all(), "The last half-window should be NaN."

def test_rolling_sd_nan_handling(sample_dataset_for_sd):
    """Test handling of NaNs in the dataset."""
    ds = sample_dataset_for_sd.copy()
    ds["signal"][10:20] = np.nan  # Introduce NaNs in the dataset
    result = filt.rolling_sd(ds, var_name="signal", dim="time", window_size=5)

    # Ensure the result contains NaNs where there is insufficient data
    assert result.isel(time=slice(10, 20)).isnull().all(), "NaN values should propagate in rolling calculation."
    assert result.count() > 70, "There should still be valid values outside NaN sections."

def test_rolling_sd_large_window(sample_dataset_for_sd):
    """Test rolling standard deviation with a large window size."""
    ds = sample_dataset_for_sd
    window_size = 50
    result = filt.rolling_sd(ds, var_name="signal", dim="time",
                             nan_edges=False,
                             window_size=window_size, min_periods=1)
    assert isinstance(result, xr.DataArray), "Result should be a DataArray."
    assert result.count() > 50, "Most values should be non-NaN despite the large window size."

def test_rolling_sd_small_dataset():
    """Test the rolling standard deviation on a small dataset."""
    time = np.arange(5)
    data = np.random.random(5)
    ds = xr.Dataset({"signal": (["time"], data)}, coords={"time": time})

    result = filt.rolling_sd(ds, var_name="signal", dim="time", window_size=3)

    assert isinstance(result, xr.DataArray), "Result should be a DataArray."
    assert result.count() == 1, "Only one valid value should be present due to the small dataset size."