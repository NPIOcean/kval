import numpy as np
import xarray as xr
import pytest
from kval.file import sbe
import glob2
import os

@pytest.fixture
def file_list_test_cnvs_single():
    '''
    Returns a list of .cnv files from different projects used in subsequent 
    tests.
    '''
    test_data_dir = 'tests/test_data/sbe_files/sbe911plus/'
    cruises =['atwain_cruise_ctds', 'dml_2020', 'kongsfjorden_ctds', 
              'pirata_ctd', 'troll_transect_22_23']

    flist_cnv = [glob2.glob(f'{test_data_dir}{cruise}/*cnv')[0] for cruise in cruises] 

    return flist_cnv


def test_read_cnv_returns_xarray_dataset(file_list_test_cnvs_single):
    '''
    Test that the read_cnv function returns an xr Dataset for a bunch of 
    different input files. 

    NOTE: Probably not great to do te globbin in here - will look into
    the best way of going about it (probably fixtures). 
    '''

    datasets = [sbe.read_cnv(fn) for fn in file_list_test_cnvs_single] 
    assert all(isinstance(ds, xr.Dataset) for ds in datasets), "Not all elements are xarray.Dataset"