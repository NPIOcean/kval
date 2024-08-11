import xarray as xr
import pytest
from kval.data import ctd
import glob2
import numpy as np

@pytest.fixture
def dir_list_test_cnvs():
    '''
    Returns a list of directories containing test .cnv files from different projects.
    '''
    test_data_dir = 'tests/test_data/sbe_files/sbe911plus/'
    cruises =['atwain_cruise_ctds', 'dml_2020', 'kongsfjorden_ctds', 
              'pirata_ctd', 'troll_transect_22_23']

    # Grab the first .cnv file for each of these cruises
    file_dirs_cnv = [f'{test_data_dir}{cruise}/' for cruise in cruises] 

    return file_dirs_cnv


def flist_list_test_cnvs():
    '''
    Returns a list of directories containing test .cnv files from different projects.
    '''
    test_data_dir = 'tests/test_data/sbe_files/sbe911plus/'
    cruises =['atwain_cruise_ctds', 'dml_2020', 'kongsfjorden_ctds', 
              'pirata_ctd', 'troll_transect_22_23']

    # Grab the first .cnv file for each of these cruises
    flists_cnv = [glob2.glob(f'{test_data_dir}{cruise}/*.cnv') for cruise in cruises] 

    return flists_cnv



def test_ctds_from_cnv_dir_returns_dataset(dir_list_test_cnvs):

    datasets = []
    
    for index, cnvlist in enumerate(dir_list_test_cnvs):
        print(f"Testing ctd.ctds_from_cnv_dir function : CTD dataset  {index + 1}/{len(dir_list_test_cnvs)}")
        dataset = ctd.ctds_from_cnv_dir(cnvlist)
        datasets.append(dataset)

    assert all(isinstance(ds, xr.Dataset) for ds in datasets), "Failed to load all test .cnv file collections to xarray.Dataset"



# Test cases for the offset function
def test_offset_apply_fixed_offset(dir_list_test_cnvs):
    """Test applying a fixed offset to the dataset."""
    ds = ctd.ctds_from_cnv_dir(dir_list_test_cnvs[0])
    ds = ctd.metadata_auto(ds)
    ds0 = ds.copy()

    offset = 5

    ds = ctd.offset(ds0, 'TEMP', offset)
    
    expected = ds0['TEMP'] + offset
    assert np.array_equal(ds['TEMP'].values, 
                          expected.values, equal_nan=True)
    assert ds['TEMP'].attrs['units'] == ds0['TEMP'].attrs['units']
    assert (ds['TEMP'].attrs['valid_max'] 
            == ds0['TEMP'].attrs['valid_max']+offset)
