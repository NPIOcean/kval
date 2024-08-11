import pytest
import xarray as xr
import numpy as np
from typing import Optional
import pandas as pd
from kval.data import edit

# Define a fixture for a mock dataset
@pytest.fixture
def mock_dataset() -> xr.Dataset:
    """
    Fixture to create a mock xarray.Dataset with metadata and variables for testing.
    """
    # Define dimensions
    Nt = 10
    time = pd.date_range('2024-01-01', periods=Nt, freq='D')
    pres = [100, 600, 2000, 6000, 11000]  # Pressure levels

    # Create data for TEMP(TIME, PRES)
    temp_data = 15 + 8 * np.random.randn(Nt, len(pres))  # Example temperature data

    # Create data for STATION(TIME) and OCEAN(TIME)
    station_data = [f'st{stnum:02.0f}' for stnum in np.arange(1, Nt + 1)]
    ocean_data = ['Atlantic', 'Arctic', 'Pacific', 'Mediterranean', 'Southern', 
                 'Baltic', 'Indian', 'Caribbean', 'Weddell', 'Ross']

    # Create data for ZONE(PRES)
    zone_data = ['epipelagic', 'mesopelagic', 'bathypelagic', 'abyssopelagic', 'hadopelagic']

    # Create the Dataset
    ds = xr.Dataset(
        {
            'TEMP': (['TIME', 'PRES'], temp_data),
            'OCEAN': (['TIME'], ocean_data),
            'STATION': (['TIME'], station_data),
            'ZONE': (['PRES'], zone_data)
        },
        coords={
            'TIME': time,
            'PRES': pres
        },
        # Add some metadata
        attrs={
            'id': 'test_dataset',
            'description': 'This is a test dataset.',
            'author': 'Test Author'
        }
    )

    # Add variable attributes
    ds['TEMP'].attrs = {
        'units': 'degC',
        'long_name': 'Test Temperature'
    }
    
    return ds

# Test cases using the mock dataset fixture
def test_no_thresholds(mock_dataset):
    """Test when no thresholds are applied."""
    ds_new = edit.threshold(mock_dataset, 'TEMP')
    
    assert np.array_equal(ds_new['TEMP'], mock_dataset['TEMP'])
    assert 'valid_min' not in ds_new['TEMP'].attrs
    assert 'valid_max' not in ds_new['TEMP'].attrs

def test_max_threshold(mock_dataset):
    """Test applying only the maximum threshold."""
    ds_new = edit.threshold(mock_dataset, 'TEMP', max_val=18)
    
    expected = mock_dataset['TEMP'].where(mock_dataset['TEMP'] <= 18)
    assert np.array_equal(ds_new['TEMP'].values, expected.values, equal_nan=True)
    assert ds_new['TEMP'].attrs['valid_max'] == 18
    assert 'valid_min' not in ds_new['TEMP'].attrs

def test_min_threshold(mock_dataset):
    """Test applying only the minimum threshold."""
    ds_new = edit.threshold(mock_dataset, 'TEMP', min_val=10)
    
    expected = mock_dataset['TEMP'].where(mock_dataset['TEMP'] >= 10)
    assert np.array_equal(ds_new['TEMP'].values, expected.values, equal_nan=True)
    assert ds_new['TEMP'].attrs['valid_min'] == 10
    assert 'valid_max' not in ds_new['TEMP'].attrs

def test_min_and_max_threshold(mock_dataset):
    """Test applying both minimum and maximum thresholds."""
    ds_new = edit.threshold(mock_dataset, 'TEMP', max_val=18, min_val=10)
    
    expected = mock_dataset['TEMP'].where((mock_dataset['TEMP'] >= 10) & (mock_dataset['TEMP'] <= 18))
    assert np.array_equal(ds_new['TEMP'].values, expected.values, equal_nan=True)
    assert ds_new['TEMP'].attrs['valid_min'] == 10
    assert ds_new['TEMP'].attrs['valid_max'] == 18

def test_empty_dataset(mock_dataset):
    """Test when the dataset is empty."""
    ds = xr.Dataset({'var': (['TIME', 'PRES'], np.array([[]]))})
    
    ds_new = edit.threshold(ds, 'var', max_val=18, min_val=10)
    
    expected = np.array([[]])
    assert np.array_equal(ds_new['var'].values, expected, equal_nan=True)
    assert ds_new['var'].attrs['valid_min'] == 10
    assert ds_new['var'].attrs['valid_max'] == 18

def test_no_modification_needed(mock_dataset):
    """Test when no modification is needed due to thresholds."""
    ds_new = edit.threshold(mock_dataset, 'TEMP', max_val=40, min_val=-10)
    
    assert np.array_equal(ds_new['TEMP'], mock_dataset['TEMP'])
    assert ds_new['TEMP'].attrs['valid_min'] == -10
    assert ds_new['TEMP'].attrs['valid_max'] == 40