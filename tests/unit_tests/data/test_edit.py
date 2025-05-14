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
    np.random.seed(42)  # Use a fixed seed
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
def test_threshold_no_thresholds(mock_dataset):
    """Test when no thresholds are applied."""
    ds_new = edit.threshold(mock_dataset, 'TEMP')

    assert np.array_equal(ds_new['TEMP'], mock_dataset['TEMP'])
    assert 'valid_min' not in ds_new['TEMP'].attrs
    assert 'valid_max' not in ds_new['TEMP'].attrs

def test_threshold_max_threshold(mock_dataset):
    """Test applying only the maximum threshold."""
    ds_new = edit.threshold(mock_dataset, 'TEMP', max_val=18)

    expected = mock_dataset['TEMP'].where(mock_dataset['TEMP'] <= 18)
    assert np.array_equal(ds_new['TEMP'].values, expected.values, equal_nan=True)
    assert ds_new['TEMP'].attrs['valid_max'] == 18
    assert 'valid_min' not in ds_new['TEMP'].attrs

def test_threshold_min_threshold(mock_dataset):
    """Test applying only the minimum threshold."""
    ds_new = edit.threshold(mock_dataset, 'TEMP', min_val=10)

    expected = mock_dataset['TEMP'].where(mock_dataset['TEMP'] >= 10)
    assert np.array_equal(ds_new['TEMP'].values, expected.values, equal_nan=True)
    assert ds_new['TEMP'].attrs['valid_min'] == 10
    assert 'valid_max' not in ds_new['TEMP'].attrs

def test_threshold_min_and_max_threshold(mock_dataset):
    """Test applying both minimum and maximum thresholds."""
    ds_new = edit.threshold(mock_dataset, 'TEMP', max_val=18, min_val=10)

    expected = mock_dataset['TEMP'].where((mock_dataset['TEMP'] >= 10) & (mock_dataset['TEMP'] <= 18))
    assert np.array_equal(ds_new['TEMP'].values, expected.values, equal_nan=True)
    assert ds_new['TEMP'].attrs['valid_min'] == 10
    assert ds_new['TEMP'].attrs['valid_max'] == 18

def test_threshold_empty_dataset(mock_dataset):
    """Test when the dataset is empty."""
    ds = xr.Dataset({'var': (['TIME', 'PRES'], np.array([[]]))})

    ds_new = edit.threshold(ds, 'var', max_val=18, min_val=10)

    expected = np.array([[]])
    assert np.array_equal(ds_new['var'].values, expected, equal_nan=True)
    assert ds_new['var'].attrs['valid_min'] == 10
    assert ds_new['var'].attrs['valid_max'] == 18

def test_threshold_no_modification_needed(mock_dataset):
    """Test when no modification is needed due to thresholds."""
    ds_new = edit.threshold(mock_dataset, 'TEMP', max_val=40, min_val=-10)

    assert np.array_equal(ds_new['TEMP'].values,
                          mock_dataset['TEMP'].values)

    assert ds_new['TEMP'].attrs['valid_min'] == -10
    assert ds_new['TEMP'].attrs['valid_max'] == 40




# Test cases for the offset function
def test_offset_apply_fixed_offset(mock_dataset):
    """Test applying a fixed offset to the dataset."""
    offset = 5

    ds_new = edit.offset(mock_dataset, 'TEMP', offset)

    expected = mock_dataset['TEMP'] + offset
    assert np.array_equal(ds_new['TEMP'].values, expected.values, equal_nan=True)
    assert ds_new['TEMP'].attrs['units'] == 'degC'
    assert ds_new['TEMP'].attrs['long_name'] == 'Test Temperature'
#    assert ds_new['TEMP'].attrs['valid_min'] == mock_dataset['TEMP'].attrs.get('valid_min', 0) + offset
#    assert ds_new['TEMP'].attrs['valid_max'] == mock_dataset['TEMP'].attrs.get('valid_max', 0) + offset

def test_offset_valid_min(mock_dataset):
    """Test that applying a fixed offset to the dataset
    also offsets valid_min, valid_max."""

    offset = 5
    mock_dataset.TEMP.attrs['valid_min'] = -10
    mock_dataset.TEMP.attrs['valid_max'] = 40

    ds_new = edit.offset(mock_dataset, 'TEMP', offset)

    assert (ds_new['TEMP'].attrs['valid_min']
            == mock_dataset['TEMP'].attrs.get('valid_min', 0) + offset)
    assert (ds_new['TEMP'].attrs['valid_max']
            == mock_dataset['TEMP'].attrs.get('valid_max', 0) + offset)

def test_offset_no_variable(mock_dataset):
    """Test when applying offset to a non-existent variable
    (should raise an error)."""
    offset = 5
    with pytest.raises(ValueError, match=("Variable 'NON_EXISTENT_VAR'"
                                          " not found in the Dataset")):
        edit.offset(mock_dataset, 'NON_EXISTENT_VAR', offset)


def test_offset_zero_offset(mock_dataset):
    """Test when applying an offset that doesn't change the data."""
    offset = 0
    ds_new = edit.offset(mock_dataset, 'TEMP', offset)

    assert np.array_equal(ds_new['TEMP'].values, mock_dataset['TEMP'].values, equal_nan=True)
    assert ds_new['TEMP'].attrs == mock_dataset['TEMP'].attrs


# Test the linear drift function
# (Pretty basic right now)

def test_linear_drift_offset(mock_dataset):
    """Test linear_drift function with default offset."""
    # Use the mock dataset for testing
    ds_out = edit.linear_drift(mock_dataset, 'TEMP', end_val=5, start_val=2,)

    # Check the output dataset's variable values
    expected_values = np.linspace(2, 5, num=len(mock_dataset.TIME))
    np.testing.assert_almost_equal(ds_out['TEMP'].values[:, 0] - mock_dataset['TEMP'].values[:, 0], expected_values, decimal=5)


def test_linear_drift_factor(mock_dataset):
    """Test linear_drift function with factor option."""
    # Use the mock dataset for testing
    ds_out = edit.linear_drift(mock_dataset, 'TEMP', end_val=2, factor=True, start_val=1, )

    # Check the output dataset's variable values
    expected_values = mock_dataset['TEMP'].values[:, 0] * np.linspace(1, 2, num=len(mock_dataset.TIME))
    np.testing.assert_almost_equal(ds_out['TEMP'].values[:, 0], expected_values, decimal=5)

    # Check if the comment attribute was added

def test_no_dates(mock_dataset):
    """Test linear_drift function without start and end dates."""
    # Use the mock dataset for testing
    ds_out = edit.linear_drift(mock_dataset, 'TEMP', end_val=3)

    # Check if the drift value is applied over the entire dataset
    expected_values = np.linspace(0, 3, num=len(mock_dataset.TIME))
    np.testing.assert_almost_equal(ds_out['TEMP'].values[:, 0]- mock_dataset['TEMP'].values[:, 0], expected_values, decimal=5)




@pytest.fixture
def sample_dataset_remove_pts():
    # Create a simple xarray dataset with time series data
    time = np.arange(10)  # Example time dimension
    var_data = np.random.random(10)  # Random data for testing
    ds = xr.Dataset(
        {
            'var': (['time'], var_data)
        },
        coords={
            'time': time
        }
    )
    return ds

def test_remove_single_point(sample_dataset_remove_pts):
    ds = sample_dataset_remove_pts.copy()
    varnm = 'var'

    # Remove the 3rd point
    remove_inds = [2]
    modified_ds = edit.remove_points_timeseries(ds, varnm, remove_inds)

    # Check if the 3rd point is NaN
    assert np.isnan(modified_ds[varnm].values[2]), "The 3rd point should be NaN."
    assert not np.isnan(modified_ds[varnm].values[1]), "The 2nd point should not be NaN."

def test_remove_multiple_points(sample_dataset_remove_pts):
    ds = sample_dataset_remove_pts.copy()
    varnm = 'var'

    # Remove the 1st and 5th points
    remove_inds = [0, 4]
    modified_ds = edit.remove_points_timeseries(ds, varnm, remove_inds)

    # Check if the correct points are NaN
    assert np.isnan(modified_ds[varnm].values[0]), "The 1st point should be NaN."
    assert np.isnan(modified_ds[varnm].values[4]), "The 5th point should be NaN."
    assert not np.isnan(modified_ds[varnm].values[3]), "The 4th point should not be NaN."

def test_remove_with_slice(sample_dataset_remove_pts):
    ds = sample_dataset_remove_pts.copy()
    varnm = 'var'

    # Remove points from index 2 to 5
    remove_inds = slice(2, 6)
    modified_ds = edit.remove_points_timeseries(ds, varnm, remove_inds)

    # Check if points 2 to 5 are NaN
    assert np.isnan(modified_ds[varnm].values[2]), "The 2nd point should be NaN."
    assert np.isnan(modified_ds[varnm].values[5]), "The 5th point should be NaN."
    assert not np.isnan(modified_ds[varnm].values[6]), "The 6th point should not be NaN."

def test_remove_no_points(sample_dataset_remove_pts):
    ds = sample_dataset_remove_pts.copy()
    varnm = 'var'

    # Remove no points
    remove_inds = []
    modified_ds = edit.remove_points_timeseries(ds, varnm, remove_inds)

    # Ensure no points were modified (nothing should be NaN)
    assert not np.any(np.isnan(modified_ds[varnm].values)), "No points should be NaN."

def test_remove_invalid_index(sample_dataset_remove_pts):
    ds = sample_dataset_remove_pts.copy()
    varnm = 'var'

    # Attempt to remove an invalid index (out of bounds)
    remove_inds = [100]  # Invalid index

    with pytest.raises(IndexError):
        edit.remove_points_timeseries(ds, varnm, remove_inds)




# -------------------------
# Fixtures for replace tests
# -------------------------
def _create_1d_dataset():
    return xr.Dataset(
        coords={'TIME': np.arange(100) + 1000},
        data_vars={
            'TEMP1': ('TIME', np.random.rand(100)),
            'TEMP2': ('TIME', np.random.rand(100) + 1),
            'flag': ('TIME', np.zeros(100, dtype=int))
        }
    )

@pytest.fixture
def ds_1d():
    np.random.seed(0)
    return _create_1d_dataset()


def _create_2d_dataset():
    return xr.Dataset(
        coords={
            'TIME': np.arange(100) + 1000,
            'DEPTH': [300, 400, 500]
        },
        data_vars={
            'TEMP1': (('TIME', 'DEPTH'), np.random.rand(100, 3)),
            'TEMP2': (('TIME', 'DEPTH'), np.random.rand(100, 3) + 1),
            'flag':  (('TIME', 'DEPTH'), np.zeros((100, 3), dtype=int))
        }
    )

@pytest.fixture
def ds_2d():
    np.random.seed(1)
    return _create_2d_dataset()

# -------------------------
# Tests for the replace function
# -------------------------
def test_replace_1d_sel(ds_1d):
    """Test replace for 1D dataset using .sel"""
    ds = ds_1d
    time_sel = ds.TIME.values[10]
    ds_new = edit.replace(
        ds, 'TEMP1', 'TEMP2',
        use_values=True, flag_value=5, var_flag='flag',
        TIME=time_sel
    )
    expected = ds['TEMP2'].sel(TIME=time_sel)
    assert ds_new['TEMP1'].sel(TIME=time_sel).item() == pytest.approx(expected.item())
    assert ds_new['flag'].sel(TIME=time_sel).item() == 5

def test_replace_1d_isel(ds_1d):
    """Test replace for 1D dataset using .isel"""
    ds = ds_1d
    idx = 20
    ds_new = edit.replace(
        ds,
        var_target='TEMP1',
        var_source='TEMP2',
        use_values=False,
        flag_value=3,
        var_flag='flag',
        TIME=idx
    )
    assert ds_new['TEMP1'].isel(TIME=idx).item() == pytest.approx(ds['TEMP2'].isel(TIME=idx).item())
    assert ds_new['flag'].isel(TIME=idx).item() == 3


def test_replace_2d_mixed(ds_2d):
    """Test replace for 2D dataset with mixed use_values dict"""
    ds = ds_2d
    time_sel = ds.TIME.values[5]
    depth_idx = 1
    ds_new = edit.replace(
        ds,
        var_target='TEMP1',
        var_source='TEMP2',
        use_values={'TIME': True, 'DEPTH': False},
        flag_value=7,
        var_flag='flag',
        TIME=time_sel,
        DEPTH=depth_idx,
        sel_method='nearest'
    )
    assert ds_new['TEMP1'].sel(TIME=time_sel).isel(DEPTH=depth_idx).item() == pytest.approx(
        ds['TEMP2'].sel(TIME=time_sel, method='nearest').isel(DEPTH=depth_idx).item())
    assert ds_new['flag'].sel(TIME=time_sel, method='nearest').isel(DEPTH=depth_idx).item() == 7

def test_replace_2d_sel_method_global(ds_2d):
    """Test replace with sel_method applied globally across all sel dimensions"""
    ds = ds_2d
    time_sel = ds.TIME.values[3] + 0.4  # fractional to trigger nearest
    depth_sel = ds.DEPTH.values[1] + 1  # will resolve to nearest=400
    ds_new = edit.replace(
        ds,
        var_target='TEMP1',
        var_source='TEMP2',
        use_values=True,
        flag_value=9,
        var_flag='flag',
        sel_method='nearest',
        TIME=time_sel,
        DEPTH=depth_sel,
    )
    expected = ds['TEMP2'].sel(TIME=time_sel, DEPTH=depth_sel, method='nearest')
    result = ds_new['TEMP1'].sel(TIME=time_sel, DEPTH=depth_sel, method='nearest')
    assert result.item() == pytest.approx(expected.item())
    assert ds_new['flag'].sel(TIME=time_sel, DEPTH=depth_sel, method='nearest').item() == 9


def test_replace_errors(ds_1d):
    """Test replace errors for missing variables or invalid use_values"""
    ds = ds_1d
    with pytest.raises(ValueError):
        edit.replace(ds, var_target='NO', var_source='TEMP2')
    with pytest.raises(ValueError):
        edit.replace(ds, var_target='TEMP1', var_source='NO')
    with pytest.raises(TypeError):
        edit.replace(ds, var_target='TEMP1', var_source='TEMP2', use_values='bad')
