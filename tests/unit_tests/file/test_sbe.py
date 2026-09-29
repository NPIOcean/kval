import numpy as np
import xarray as xr
import pytest
from kval.file import sbe
import glob2
import os

# The SBE loader warns when it drops a column it cannot read as float
# (e.g. avg_std in bottle files). Expected for the test files.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Could not read .* as float:UserWarning")

@pytest.fixture
def file_list_test_cnvs_single():
    '''
    Returns a list of .cnv files from different projects used in subsequent
    tests.
    '''
    test_data_dir = 'tests/test_data/sbe_files/'

    # Grab some test files from 911 data
    cruises =['atwain_cruise_ctds', 'dml_2020', 'kongsfjorden_ctds',
              'pirata_ctd', 'troll_transect_22_23']

    # Grab the first .cnv file for each of these cruises
    flist_cnv_profile = [glob2.glob(f'{test_data_dir}sbe911plus/{cruise}/*.cnv')[0] for cruise in cruises]



    flist_cnv_sbe37 = glob2.glob(test_data_dir + 'sbe37/cnv/*cnv')
    flist_cnv_sbe56 = glob2.glob(test_data_dir + 'sbe56/*cnv')


    flist_cnv = flist_cnv_profile + flist_cnv_sbe37 + flist_cnv_sbe56

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



## Test the read_cnv function

def test_read_csv_valid_file():
    # Test reading all valid CSV files in the directory
    test_data_dir = 'tests/test_data/sbe_files/sbe56/'

    # Use glob2 to find all CSV files in the directory
    csv_files = glob2.glob(os.path.join(test_data_dir, '*.csv'))

    for filename in csv_files:
        # Check that the function does not raise an error and returns an xarray Dataset
        ds = sbe.read_csv(filename)

        # Assertions to check if the dataset has the expected structure and attributes
        assert isinstance(ds, xr.Dataset), "Output should be an xarray Dataset"
        assert 'TIME' in ds, "Dataset should contain a 'TIME' variable"
        assert 'TEMP' in ds, "Dataset should contain a 'TEMP' variable"

        # Check the attributes of the dataset
        # Replace with expected values based on the specific CSV file being read
        assert ds.attrs['instrument_model'] == 'SBE56', f"Failed for {filename}"
        assert ds.attrs['filename'] == os.path.basename(filename), f"Failed for {filename}"
        assert ds['TEMP'].attrs['units'] == 'degree_Celsius', f"Failed for {filename}"

        # Additional checks for dimensions, data values, etc. can be added here
        assert ds.sizes['TIME'] > 0, "Dataset should contain time dimension"

def test_read_csv_file_not_found():
    # Test behavior when file is not found
    test_data_dir = 'tests/test_data/sbe_files/sbe56/'

    invalid_filename = os.path.join(test_data_dir, 'invalid_file.csv')

    with pytest.raises(FileNotFoundError) as excinfo:
        sbe.read_csv(invalid_filename)

    assert "File not found" in str(excinfo.value)



# ===================================================================
# _add_latlon_variables
# ===================================================================

def _make_ds_for_latlon(with_attrs=True, with_sample_vars=False):
    ds = xr.Dataset({'STATION': ('TIME', ['st01'])}, coords={'TIME': [0]})
    if with_attrs:
        ds.attrs['latitude'] = 80.0
        ds.attrs['longitude'] = 30.0
    if with_sample_vars:
        ds['LATITUDE_SAMPLE'] = ('TIME', [80.5])
        ds['LONGITUDE_SAMPLE'] = ('TIME', [30.5])
    return ds


def test_add_latlon_variables_assigns_as_coordinates_not_data_vars():
    """LATITUDE/LONGITUDE should be coordinate variables, not plain data
    variables -- matches dataset.add_latlon's convention and required
    for compliance_checks_custom's coordinate checks."""
    ds = _make_ds_for_latlon()
    result = sbe._add_latlon_variables(ds)
    assert 'LATITUDE' in result.coords
    assert 'LONGITUDE' in result.coords
    assert 'LATITUDE' not in result.data_vars
    assert 'LONGITUDE' not in result.data_vars


def test_add_latlon_variables_reads_from_attrs():
    ds = _make_ds_for_latlon(with_attrs=True)
    result = sbe._add_latlon_variables(ds)
    assert float(result.LATITUDE.values[0]) == 80.0
    assert float(result.LONGITUDE.values[0]) == 30.0


def test_add_latlon_variables_falls_back_to_sample_vars():
    ds = _make_ds_for_latlon(with_attrs=False, with_sample_vars=True)
    result = sbe._add_latlon_variables(ds)
    assert float(result.LATITUDE.values[0]) == 80.5
    assert float(result.LONGITUDE.values[0]) == 30.5


def test_add_latlon_variables_assigns_nan_and_warns_if_missing(capsys):
    ds = _make_ds_for_latlon(with_attrs=False, with_sample_vars=False)
    result = sbe._add_latlon_variables(ds)
    assert np.isnan(result.LATITUDE.values[0])
    assert np.isnan(result.LONGITUDE.values[0])
    out = capsys.readouterr().out
    assert 'latitude' in out and 'longitude' in out


def test_add_latlon_variables_suppress_warning():
    ds = _make_ds_for_latlon(with_attrs=False, with_sample_vars=False)
    # Should not raise/print when suppressed -- just confirm it runs
    # cleanly and still produces the (NaN) coordinates.
    result = sbe._add_latlon_variables(ds, suppress_latlon_warning=True)
    assert 'LATITUDE' in result.coords

# ===================================================================
# Regression tests: truthy-check bugs that silently dropped lat=0/lon=0
# (equator / prime meridian -- valid real coordinates)
# ===================================================================

def test_assign_specified_lat_lon_station_preserves_zero_values():
    """_assign_specified_lat_lon_station's own docstring says 'if
    specified by the user (not None)', but the code used `if lat:`
    (plain truthiness), which silently dropped lat=0/lon=0."""
    ds = xr.Dataset()
    result = sbe._assign_specified_lat_lon_station(ds, lat=0.0, lon=0.0, station='EQ01')
    assert result.attrs['latitude'] == 0.0
    assert result.attrs['longitude'] == 0.0
    assert result.attrs['station'] == 'EQ01'


def test_add_latlon_variables_preserves_zero_from_attrs():
    ds = xr.Dataset({'STATION': ('TIME', ['st01'])}, coords={'TIME': [0]})
    ds.attrs['latitude'] = 0.0
    ds.attrs['longitude'] = 0.0
    result = sbe._add_latlon_variables(ds)
    assert float(result.LATITUDE.values[0]) == 0.0
    assert float(result.LONGITUDE.values[0]) == 0.0


def test_decdeg_from_line_zero_value_not_dropped_by_caller():
    """A header line parsing to exactly 0.0 (equator/prime meridian)
    should not be treated the same as a parse failure (None)."""
    from kval.file.sbe import _decdeg_from_line
    # A line that genuinely parses to 0.0
    result = _decdeg_from_line('** Latitude: 000 00.0000')
    assert result == 0.0


# ---------------------------------------------------------------------
# 0.5.1: robustness of the processing-step parsing of the .cnv header
# ---------------------------------------------------------------------
def _proc_steps(lines):
    ds = xr.Dataset(attrs={"history": ""})
    header_info = {"SBEproc_hist": lines, "source_file_type": "cnv",
                   "source_file": "cast001.cnv"}
    return sbe._read_SBE_proc_steps(ds, header_info, _is_moored=False)


_BASE = ["# datcnv_date = Jan 05 2022 10:11:12, 7.2.5"]


@pytest.mark.parametrize("datcnv_in, hex_name, con_name", [
    (r"C:\Data\cast001.hex C:\Data\cast001.XMLCON", "cast001.hex", "CAST001.XMLCON"),
    ("/Users/me/data/cast001.hex /Users/me/data/cast001.XMLCON", "cast001.hex", "CAST001.XMLCON"),
    ("cast001.hex cast001.XMLCON", "cast001.hex", "CAST001.XMLCON"),
    (r"C:\Data\cast001.hex C:\Data\cast001.CON", "cast001.hex", "CAST001.CON"),
])
def test_read_SBE_proc_steps_file_names_any_path_style(datcnv_in, hex_name, con_name):
    ds = _proc_steps(_BASE + [f"# datcnv_in = {datcnv_in}"])
    assert hex_name in ds.attrs["source_file"]
    assert con_name in ds.attrs["source_file"]


def test_read_SBE_proc_steps_unparseable_file_names_do_not_crash():
    ds = _proc_steps(_BASE + ["# datcnv_in = something odd"])
    assert "N/A" in ds.attrs["source_file"]


def _binavg_lines(excl="yes", skip="0",
                  surf="yes, min = 0.000, max = 1.000, value = 0.500"):
    return _BASE + [
        "# binavg_bintype = decibars", "# binavg_binsize = 1",
        f"# binavg_excl_bad_scans = {excl}", f"# binavg_skipover = {skip}",
        f"# binavg_surface_bin = {surf}"]


def test_read_SBE_proc_steps_binavg_text_reflects_header():
    ds = _proc_steps(_binavg_lines())
    text = ds.attrs["SBE_processing"]
    assert "Bad scans excluded" in text and "skipped over" not in text
    assert "Surface bin parameters" in text and "min = 0.000" in text
    assert "binned" in ds.attrs


def test_read_SBE_proc_steps_binavg_other_options():
    text = _proc_steps(
        _binavg_lines(excl="no", skip="5", surf="no")).attrs["SBE_processing"]
    assert "Bad scans not excluded" in text
    assert "skipped over 5 initial scans" in text
    assert "No surface bin" in text


def test_read_SBE_proc_steps_not_binned_has_no_binned_attribute():
    ds = _proc_steps(_BASE + [r"# datcnv_in = C:\a\b.hex C:\a\b.XMLCON"])
    assert "binned" not in ds.attrs


def test_read_SBE_proc_steps_lowpass_time_constant_without_decimals():
    lines = _BASE + ["# filter_low_pass_tc_A = 1",
                     "# filter_low_pass_A_vars = c0S/m"]
    assert "time constant 1.0" in _proc_steps(lines).attrs["SBE_processing"]