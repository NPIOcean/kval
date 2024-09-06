import pytest
import xarray as xr
import numpy as np
from kval.util import filt  # Update this path to your actual import

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


