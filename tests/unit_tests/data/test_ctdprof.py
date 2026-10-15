import xarray as xr
import pytest
import glob2
import numpy as np
from kval.data import ctdprof 
from kval.data.ctdprof import (
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
        print(f"Testing ctdprof.ctds_from_cnv_dir function : CTD dataset  {index + 1}/{len(dir_list_test_cnvs)}")
        dataset = ctdprof.ctds_from_cnv_dir(cnvlist)
        datasets.append(dataset)

    assert all(isinstance(ds, xr.Dataset) for ds in datasets), "Failed to load all test .cnv file collections to xarray.Dataset"



# Test cases for the offset function
def test_offset_apply_fixed_offset(dir_list_test_cnvs):
    """Test applying a fixed offset to the dataset."""
    ds = ctdprof.ctds_from_cnv_dir(dir_list_test_cnvs[0])
    ds = ctdprof.metadata_auto(ds)
    ds0 = ds.copy()

    offset = 5

    ds = ctdprof.offset(ds0, 'TEMP', offset)
    
    expected = ds0['TEMP'] + offset
    assert np.array_equal(ds['TEMP'].values, 
                          expected.values, equal_nan=True)
    assert ds['TEMP'].attrs['units'] == ds0['TEMP'].attrs['units']
    assert (ds['TEMP'].attrs['valid_max'] 
            == ds0['TEMP'].attrs['valid_max']+offset)


def test_metadata_auto_default_no_org_attrs(dir_list_test_cnvs):
    """Without org specified, no organization-specific global attrs
    (e.g. institution) should be added."""
    ds = ctdprof.ctds_from_cnv_dir(dir_list_test_cnvs[0])
    ds = ctdprof.metadata_auto(ds)
    assert 'institution' not in ds.attrs


def test_metadata_auto_explicit_org_npi(dir_list_test_cnvs):
    """org='npi' should add the standard NPI global attributes."""
    ds = ctdprof.ctds_from_cnv_dir(dir_list_test_cnvs[0])
    ds = ctdprof.metadata_auto(ds, org='npi')
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
    """ctdprof.<func> should be reachable and work on profile-shaped (PRES-
    dimensioned) data -- this is new functionality, ctdprof.py had none of
    these six functions before."""
    result = func(small_ctd_profile_ds)
    assert expected_var in result
    assert np.all(np.isfinite(result[expected_var].values))
 


# ---------------------------------------------------------------------
# calibrate_chl: valid_min/valid_max

def _chl_ds(valid_min, valid_max):
    ds = xr.Dataset({"CHLA_fluorescence": ("PRES", [0.1, 0.5, 2.0])},
                    coords={"PRES": [1.0, 2.0, 3.0]})
    ds.CHLA_fluorescence.attrs.update(units="mg m-3", valid_min=valid_min,
                                      valid_max=valid_max)
    return ds


def test_calibrate_chl_converts_data_based_valid_limits():
    """The limits of the uncalibrated variable used to be copied as they
    were."""
    out = ctdprof.calibrate_chl(_chl_ds(0.0, 3.0), A=2.0, B=0.5,
                                verbose=False)
    assert out.CHLA.attrs["valid_min"] == pytest.approx(0.5)
    assert out.CHLA.attrs["valid_max"] == pytest.approx(6.5)


def test_calibrate_chl_negative_slope_swaps_valid_limits():
    out = ctdprof.calibrate_chl(_chl_ds(0.0, 3.0), A=-1.0, B=10.0,
                                verbose=False)
    assert out.CHLA.attrs["valid_min"] == pytest.approx(7.0)
    assert out.CHLA.attrs["valid_max"] == pytest.approx(10.0)


def test_calibrate_chl_generic_valid_limits_become_the_standard_ones():
    from kval.metadata import _standard_attrs
    std = _standard_attrs.standard_var_attrs["CHLA"]
    out = ctdprof.calibrate_chl(_chl_ds(-1.9, 100.0), A=2.0, B=0.5,
                                verbose=False)
    assert out.CHLA.attrs["valid_min"] == std["valid_min"]
    assert out.CHLA.attrs["valid_max"] == std["valid_max"]
    # The input variable is not changed
    assert out.CHLA_fluorescence.attrs["valid_max"] == 100.0


def test_calibrate_chl_generic_valid_limits_dropped_without_standard():
    out = ctdprof.calibrate_chl(_chl_ds(-1.9, 100.0), A=2.0, B=0.5,
                                chl_name_out="MY_CHL", verbose=False)
    assert "valid_min" not in out.MY_CHL.attrs
    assert "valid_max" not in out.MY_CHL.attrs


# ---------------------------------------------------------------------
# to_csv

def _csv_ds(datetime_time):
    times = np.array(["2024-05-01T10:00:00", "2024-05-02T11:30:00"],
                     dtype="datetime64[ns]")
    ds = xr.Dataset(
        {"TEMP": (("TIME", "PRES"), [[1.0, 2.0], [3.0, 4.0]]),
         "STATION": ("TIME", ["st01", "st02"]),
         "LATITUDE": ("TIME", [70.0, 71.0]),
         "LONGITUDE": ("TIME", [10.0, 11.0])},
        coords={"TIME": times, "PRES": [1.0, 2.0]})
    if not datetime_time:
        ds = ds.assign_coords(TIME=(times - np.datetime64("1970-01-01"))
                              / np.timedelta64(1, "D"))
        ds.TIME.attrs["units"] = "days since 1970-01-01"
    return ds


@pytest.mark.parametrize("datetime_time", [True, False])
def test_to_csv_numeric_and_datetime_time(tmp_path, datetime_time):
    """datetime64 TIME used to raise AttributeError (needed TIME.units)."""
    outfile = str(tmp_path / "out")
    ctdprof.to_csv(_csv_ds(datetime_time), outfile)
    text = open(outfile + ".csv").read()
    assert "2024-05-01 10:00:00" in text
    assert "2024-05-02 11:30:00" in text
    assert "st02" in text
