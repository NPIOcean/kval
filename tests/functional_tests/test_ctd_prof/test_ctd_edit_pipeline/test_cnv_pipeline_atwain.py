'''
Want full test of CTD processing pipeline here. 

Do:

- Generalize from ATWAIN to many datasets
- Once we have non-graphic editing functions: Test those
- Once we have migrated the attribute editing functions from ctd to something more general: test those

'''


import pytest
from kval.data import ctd
import xarray as xr

@pytest.fixture
def atwain_dir():
    '''
    Returns the directory containing the atwain test files.
    '''
    test_data_dir = 'tests/test_data/sbe_files/sbe911plus/'
 
    # Grab the directory
    atwain_dir = f'{test_data_dir}atwain_cruise_ctds/'

    return atwain_dir

def test_cnv_pipeline(atwain_dir):
    '''
    Test the cnv pipeline (to be extended..)
    '''

    # Loading
    ds = ctd.ctds_from_cnv_dir(atwain_dir)

    # Test that we are making an xr dataset
    assert isinstance(ds, xr.Dataset) , "Failed to load all ATWAIN .cnv files to xarray.Dataset"

    # Test that the coordinates TIME, PRES are present
    assert list(ds.coords) == ['STATION', 'LATITUDE', 'LONGITUDE', 'PRES', 'TIME'], "Coordinates not read as ['STATION', 'LATITUDE', 'LONGITUDE', 'PRES', 'TIME']"

