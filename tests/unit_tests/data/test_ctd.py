import xarray as xr
import pytest
from kval.data import ctd
import glob2
import numpy as np
from kval.data.ctd import (
    calculate_PSAL,
    calculate_CNDC,
    calculate_SA_CT,
    calculate_rho,
    calculate_sig0,
    calculate_ss,
)


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


def test_metadata_auto_default_no_org_attrs(dir_list_test_cnvs):
    """Without org specified, no organization-specific global attrs
    (e.g. institution) should be added."""
    ds = ctd.ctds_from_cnv_dir(dir_list_test_cnvs[0])
    ds = ctd.metadata_auto(ds)
    assert 'institution' not in ds.attrs


def test_metadata_auto_explicit_org_npi(dir_list_test_cnvs):
    """org='npi' should add the standard NPI global attributes."""
    ds = ctd.ctds_from_cnv_dir(dir_list_test_cnvs[0])
    ds = ctd.metadata_auto(ds, org='npi')
    assert ds.attrs.get('institution') == 'Norwegian Polar Institute (NPI)'



# Testing reachability of the calculate_* functions imported from data.dataset

@pytest.fixture
def small_ctd_profile_ds():
    n = 5
    return xr.Dataset(
        {
            "TEMP": ("PRES", 10.0 - 0.5 * np.arange(n)),
            "PSAL": ("PRES", 34.5 + 0.05 * np.arange(n)),
            "CNDC": ("PRES", 40.0 + 0.1 * np.arange(n)),
        },
        coords={"PRES": np.arange(n) * 50.0},
    ).pipe(lambda ds: ds.assign(LATITUDE=((), 70.0), LONGITUDE=((), 10.0)))
 
 
@pytest.mark.parametrize(
    "func,expected_var",
    [
        (calculate_PSAL, "PSAL"),
        (calculate_CNDC, "CNDC"),
        (calculate_SA_CT, "SA"),
        (calculate_rho, "RHO"),
        (calculate_sig0, "SIG0"),
        (calculate_ss, "SVEL"),
    ],
)
def test_ctd_can_reach_dataset_functions(small_ctd_profile_ds, func, expected_var):
    """ctd.<func> should be reachable and work on profile-shaped (PRES-
    dimensioned) data -- this is new functionality, ctd.py had none of
    these six functions before."""
    result = func(small_ctd_profile_ds)
    assert expected_var in result
    assert np.all(np.isfinite(result[expected_var].values))
 
