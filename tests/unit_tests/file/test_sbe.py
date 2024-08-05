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

    # Grab the first .cnv file for each of these cruises
    flist_cnv = [glob2.glob(f'{test_data_dir}{cruise}/*.cnv')[0] for cruise in cruises] 

    return flist_cnv


@pytest.fixture
def file_list_test_btls_single():
    '''
    Returns a list of .btl files from different projects used in subsequent 
    tests.
    '''
    test_data_dir = 'tests/test_data/sbe_files/sbe911plus/'
    cruises =['dml_2020', 'kongsfjorden_ctds', 'troll_transect_22_23']

    # Grab the first .btl file for each of these cruises
    flist_cnv = [glob2.glob(f'{test_data_dir}{cruise}/*.btl')[0] for cruise in cruises] 

    return flist_cnv

################


def test_read_cnv_returns_xarray_dataset(file_list_test_cnvs_single):
    '''
    Test that the read_cnv function returns an xr Dataset for a bunch of 
    different input files. 
    '''

    datasets = [sbe.read_cnv(fn) for fn in file_list_test_cnvs_single] 
    assert all(isinstance(ds, xr.Dataset) for ds in datasets), "Failed to load all test .cnv files to xarray.Dataset"



def test_read_btl_returns_xarray_dataset(file_list_test_btls_single):
    '''
    Test that the read_btl function returns an xr Dataset for a bunch of 
    different input files. 
    '''

    datasets = [sbe.read_btl(fn) for fn in file_list_test_btls_single] 
    assert all(isinstance(ds, xr.Dataset) for ds in datasets), "Failed to load all test .btl files to xarray.Dataset"



