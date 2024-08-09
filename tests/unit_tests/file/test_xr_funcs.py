import pytest
import xarray as xr
import pandas as pd
import numpy as np

from kval.file import xr_funcs  # Replace 'your_module' with the actual name of your Python file

# Define a fixture for a mock dataset
@pytest.fixture
def mock_dataset():
    
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
        }
    )
    
    return ds

def test_pick_single_TIME(mock_dataset):
    result = xr_funcs.pick(mock_dataset, STATION='st02')
    assert list(result.dims)== ['PRES']
    assert 'st02' in result.STATION.values
    assert 'Arctic' in result.OCEAN.values

def test_pick_multiple_TIME(mock_dataset):
    result = xr_funcs.pick(mock_dataset, STATION=['st02', 'st03'])
    assert result.sizes['TIME'] == 2
    assert 'st02' in result.STATION.values
    assert 'st03' in result.STATION.values

def test_pick_single_PRES(mock_dataset):
    result = xr_funcs.pick(mock_dataset, ZONE='epipelagic')
    assert list(result.dims)== ['TIME']
    assert 'epipelagic' in result.ZONE.values
    assert result.PRES.item() == 100

def test_pick_multiple_PRES(mock_dataset):
    result = xr_funcs.pick(mock_dataset, ZONE=['epipelagic', 'hadopelagic'])
    assert result.sizes['PRES'] == 2
    assert 'epipelagic' in result.ZONE.values
    assert 'hadopelagic' in result.ZONE.values

def test_pick_nonexistent_value(mock_dataset):
    result = xr_funcs.pick(mock_dataset, STATION='st11')
    assert list(result.dims) == ['TIME', 'PRES']
    assert result.sizes['TIME'] == 0

def test_pick_squeeze_false(mock_dataset):
    result = xr_funcs.pick(mock_dataset, STATION='st02', squeeze=False)
    assert result.sizes['TIME'] == 1
    assert result.sizes['PRES'] == 5  # PRES dimension should remain unchanged

def test_pick_with_squeeze(mock_dataset):
    result = xr_funcs.pick(mock_dataset, STATION='st02')
    assert 'TIME' not in result.dims

def test_pick_invalid_dimension(mock_dataset):
    with pytest.raises(ValueError):
        xr_funcs.pick(mock_dataset, TEMP=15)


def test_pick_multiple_conditions(mock_dataset):
    # Use multiple conditions: both STATION and ZONE
    result = xr_funcs.pick(mock_dataset, 
                           STATION=['st02', 'st03', 'st05'], 
                           ZONE=['epipelagic', 'bathypelagic'])
    
    # Check that only the entries that match both conditions are present
    assert result.sizes['TIME'] == 3  # Only one TIME index should match the condition
    assert result.sizes['PRES'] == 2  # Only one PRES index should match the condition
    
    # Verify the results
    assert ['epipelagic', 'bathypelagic'] in result.ZONE.values
    assert ['st02', 'st03', 'st05'] in result.STATION.values
    assert 'st01' not in result.STATION.values    
    assert 'abyssopelagic' not in result.ZONE.values
    
    # Check if `TIME` dimension is correctly filtered
    assert len(result.TIME) == 3
    assert pd.Timestamp('2024-01-02') in result.TIME
    assert pd.Timestamp('2024-01-03') in result.TIME
    assert pd.Timestamp('2024-01-05') in result.TIME
    assert pd.Timestamp('2024-01-04') not in result.TIME


    # Check if `PRES` dimension is correctly filtered
    assert (result.PRES.values == [100, 2000]).all()