import pytest
import xarray as xr
from unittest.mock import patch
from kval.util.era5 import open_era5_dataset, get_era5_time_series_point
import pandas as pd

# Constants for testing
LAT = 45.0
LON = -93.0
TIME_START = '2022-01-01'
TIME_END = '2022-01-31'
VARIABLE = 'SLP'
TIME_RESOLUTION = 'monthly'

# Mock dataset
@pytest.fixture
def mock_dataset():
    # Create a mock xarray.Dataset
    time_index = pd.date_range(start='2022-01-01', periods=1, freq='ME')
    data = xr.Dataset(
        {
            'sp': (['time', 'lat', 'lon'], [[[1000.0]]]),
        },
        coords={
            'time': time_index,
            'lat': [LAT],
            'lon': [LON],
        }
    )
    return data

# Test with mocking
@patch('xarray.open_dataset')
def test_open_era5_dataset(mock_open_dataset, mock_dataset):
    # Setup the mock to return the mock dataset
    mock_open_dataset.return_value = mock_dataset

    ds = open_era5_dataset(variable='SLP', time_resolution='monthly', verbose=False)

    assert isinstance(ds, xr.Dataset)
    assert 'sp' in ds
    assert ds['sp'].values.item() == 1000.0

@patch('xarray.open_dataset')
def test_get_era5_time_series_point(mock_open_dataset, mock_dataset):
    # Setup the mock to return the mock dataset
    mock_open_dataset.return_value = mock_dataset

    ds_subset = get_era5_time_series_point(
        variable=VARIABLE,
        time_resolution=TIME_RESOLUTION,
        lat=LAT,
        lon=LON,
        time_start=TIME_START,
        time_end=TIME_END,
        verbose=False
    )

    assert isinstance(ds_subset, xr.Dataset)
    assert 'SLP' in ds_subset
    assert ds_subset['SLP'].values.item() == 1000.0

# Real call (to be run manually or commented out during routine testing)
@pytest.mark.skip(
        reason="This test makes a real call to the APDRC servers. "
               "Comment it out for routine testing.")
def test_real_call():
    ds = open_era5_dataset(variable='SLP', time_resolution='monthly', verbose=False)
    assert isinstance(ds, xr.Dataset)
    assert 'sp' in ds

    ds_subset = get_era5_time_series_point(
        variable='SLP',
        time_resolution='monthly',
        lat=LAT,
        lon=LON,
        time_start=TIME_START,
        time_end=TIME_END,
        verbose=False
    )

    assert isinstance(ds_subset, xr.Dataset)
    assert 'SLP' in ds_subset
