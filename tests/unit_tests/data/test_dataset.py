import pytest
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from kval.data import dataset  # Assuming the function is in xr_funcs module

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






def test_metadata_to_txt(mock_dataset):
    """
    Test the metadata_to_txt function using a mock dataset and temporary file.
    """
    # Define the output file path
    outfile = 'metadata_output.txt'
    
    # Call the function to create the metadata file
    dataset.metadata_to_txt(mock_dataset, outfile)
    
    # Read the content of the file and verify its correctness
    with open(outfile, 'r') as f:
        content = f.read()
    
    # Check for expected strings in the output file
    assert 'FILE METADATA FROM: test_dataset' in content
    assert '### GLOBAL ATTRIBUTES   ###' in content
    assert 'description:\nThis is a test dataset.' in content
    assert 'author:\nTest Author' in content
    assert '### VARIABLE ATTRIBUTES ###' in content
    assert 'TEMP' in content
    assert 'units:\ndegC' in content
    assert 'long_name:\nTest Temperature' in content
    

    # Verify that attributes appear directly after the correct variable name
    temp_index = content.index('TEMP')
    ocean_index = content.index('OCEAN')
    station_index = content.index('STATION')

    # Check that the order is right
    assert temp_index < ocean_index
    assert ocean_index < station_index

    # Check that we have one unit entry and one long_name entry under TEMP 
    assert content[temp_index:ocean_index].count('units:') == 1
    assert content[temp_index:ocean_index].count('long_name:') == 1

    # Check that we have no unit entry or one long_name entry under OCEAN 
    assert content[ocean_index:station_index].count('units:') == 0
    assert content[ocean_index:station_index].count('long_name:') == 0


    # Clean up the file after test
    Path(outfile).unlink()