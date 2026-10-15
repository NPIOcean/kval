import numpy as np
import xarray as xr
import pytest
from kval.file import sbe
import glob
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
    flist_cnv_profile = [glob.glob(f'{test_data_dir}sbe911plus/{cruise}/*.cnv')[0] for cruise in cruises]



    flist_cnv_sbe37 = glob.glob(test_data_dir + 'sbe37/cnv/*cnv')
    flist_cnv_sbe56 = glob.glob(test_data_dir + 'sbe56/*cnv')


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
    flist_cnv = [glob.glob(f'{test_data_dir}{cruise}/*.btl')[0] for cruise in cruises]

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

    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(test_data_dir, '*.csv'))

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

# ===================================================================
# History dates (independent of matplotlib's date epoch)
# ===================================================================

def _first_last_dates(ds):
    t = xr.coding.times.decode_cf_datetime(ds.TIME.values, ds.TIME.units)
    return (np.datetime_as_string(t.min(), unit='D'),
            np.datetime_as_string(t.max(), unit='D'))


def test_read_csv_history_dates_do_not_depend_on_matplotlib_epoch():
    """The "Data collection" dates in the history attribute used to come
    from matplotlib's num2date, which ignores TIME's units and uses
    matplotlib's own epoch setting."""
    import matplotlib
    filename = 'tests/test_data/sbe_files/sbe56/SBE56-1232_test.csv'
    with matplotlib.rc_context({"date.epoch": "2000-01-01T00:00:00"}):
        ds = sbe.read_csv(filename)
    first, last = _first_last_dates(ds)
    assert ds.attrs['history'].startswith(
        f"{first} - {last}: Data collection.")


def test_read_cnv_moored_history_dates_do_not_depend_on_matplotlib_epoch():
    import matplotlib
    filename = 'tests/test_data/sbe_files/sbe37/cnv/test_sbe37.cnv'
    with matplotlib.rc_context({"date.epoch": "2000-01-01T00:00:00"}):
        ds = sbe.read_cnv(filename)
    first, last = _first_last_dates(ds)
    assert ds.attrs['history'].startswith(f"{first} - {last}:")


# ===================================================================
# read_csv: order of day/month/year in the Date column
# ===================================================================

_SBE56_CSV = 'tests/test_data/sbe_files/sbe56/SBE05602355_NABOS-test.csv'


def _write_sbe56_csv(path, dates, times=None):
    """An SBE56 .csv with the header of a real file and the given dates."""
    with open(_SBE56_CSV, encoding='latin-1') as f:
        header = [line for line in f.read().splitlines() if line.startswith('%')]
    times = times or ['12:00:00'] * len(dates)
    rows = [f'"{d}","{t}",0.1' for d, t in zip(dates, times)]
    with open(path, 'w', encoding='latin-1') as f:
        f.write('\n'.join(header + ['"Date","Time","Temperature"'] + rows) + '\n')
    return str(path)


def _decoded_times(ds):
    return xr.coding.times.decode_cf_datetime(ds.TIME.values, ds.TIME.units)


def test_read_csv_real_files_dates():
    ds = sbe.read_csv('tests/test_data/sbe_files/sbe56/SBE56-1232_test.csv')
    assert _decoded_times(ds)[0] == np.datetime64('2022-08-08T22:00:00')
    ds = sbe.read_csv(_SBE56_CSV)  # day-first, starts 28-08-2013
    assert _decoded_times(ds)[0] == np.datetime64('2013-08-28T12:00:00')


def test_read_csv_day_first_with_small_days_is_not_read_month_first(tmp_path):
    """Used to be read as 9 Jan - 2 Feb (month-first), silently."""
    path = _write_sbe56_csv(tmp_path / 'SBE05602355_a.csv',
                            ['01-09-2013', '02-09-2013'])
    with pytest.warns(UserWarning, match='cannot be told from the file'):
        ds = sbe.read_csv(path)
    np.testing.assert_array_equal(
        _decoded_times(ds),
        np.array(['2013-09-01T12:00', '2013-09-02T12:00'], dtype='datetime64[ns]'))


def test_read_csv_day_first_past_day_12_reads_and_does_not_crash(tmp_path):
    dates = [f'{d:02d}-09-2013' for d in range(1, 20)]
    path = _write_sbe56_csv(tmp_path / 'SBE05602355_b.csv', dates)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)  # not ambiguous: no warning
        ds = sbe.read_csv(path)
    t = _decoded_times(ds)
    assert t[0] == np.datetime64('2013-09-01T12:00')
    assert t[-1] == np.datetime64('2013-09-19T12:00')


def test_read_csv_month_first_is_detected(tmp_path):
    path = _write_sbe56_csv(tmp_path / 'SBE05602355_c.csv',
                            ['09-12-2013', '09-13-2013'])
    ds = sbe.read_csv(path)
    assert _decoded_times(ds)[0] == np.datetime64('2013-09-12T12:00')


def test_read_csv_explicit_date_format(tmp_path):
    path = _write_sbe56_csv(tmp_path / 'SBE05602355_d.csv',
                            ['01-09-2013', '02-09-2013'])
    ds = sbe.read_csv(path, date_format='MDY')
    assert _decoded_times(ds)[0] == np.datetime64('2013-01-09T12:00')
    with pytest.raises(ValueError, match='does not match'):
        sbe.read_csv(path, date_format='YMD')


@pytest.mark.parametrize('dates, match', [
    (['13-01-2013', '01-13-2013'], 'both the first and the second field'),
    (['Sep 1 2013', 'Sep 2 2013'], 'unrecognised format'),
])
def test_read_csv_unreadable_dates_raise(tmp_path, dates, match):
    path = _write_sbe56_csv(tmp_path / 'SBE05602355_e.csv', dates)
    with pytest.raises(ValueError, match=match):
        sbe.read_csv(path)


def test_read_cnv_header_with_blank_line(tmp_path):
    """A blank line in the .cnv header used to raise IndexError."""
    src = 'tests/test_data/sbe_files/sbe37/cnv/test_sbe37.cnv'
    with open(src, encoding='latin-1') as f:
        lines = f.read().splitlines(True)
    reference = sbe.read_cnv(src)
    lines.insert(3, '\n')
    path = tmp_path / 'blank_line_in_header.cnv'
    with open(path, 'w', encoding='latin-1') as f:
        f.write(''.join(lines))
    ds = sbe.read_cnv(str(path))
    np.testing.assert_array_equal(ds.TIME.values, reference.TIME.values)
    np.testing.assert_array_equal(ds.TEMP.values, reference.TEMP.values)


def _cnv_data_lines(path):
    """The data lines of a .cnv file (everything after *END*)."""
    with open(path, encoding='latin-1') as f:
        lines = f.read().splitlines()
    end = next(i for i, line in enumerate(lines) if line.startswith('*END*'))
    return [line for line in lines[end + 1:] if line.strip()]


_ALL_CNVS = sorted(glob.glob('tests/test_data/sbe_files/**/*.cnv', recursive=True))


@pytest.mark.parametrize('path', _ALL_CNVS, ids=os.path.basename)
def test_cnv_column_reader_keeps_every_data_row(path):
    """The first data row used to be read as a header row and dropped
    (e.g. the shallowest bin of a bin-averaged profile)."""
    data_lines = _cnv_data_lines(path)
    header_info = sbe.read_header(path)
    ds = sbe._read_column_data_xr(path, header_info)
    assert ds.sizes['scan_count'] == len(data_lines)
    first_values = [float(v) for v in data_lines[0].split()]
    first_col = header_info['col_names'][0]
    assert float(ds[first_col].values[0]) == pytest.approx(first_values[0])


def test_read_cnv_profile_keeps_shallowest_bin():
    ds = sbe.read_cnv('tests/test_data/sbe_files/sbe911plus/dml_2020/sta013_01.cnv')
    assert float(ds.PRES.min()) == 4.0  # first data line is the 4 dbar bin


# ===================================================================
# timeJ rounding, start time, serial numbers
# ===================================================================

import pandas as pd


def _timej_times(step_s, n=200, start='2021-03-01 12:00:00', decimals=6):
    """Times as they come out of a .cnv timeJ column: true sample times,
    written as Julian days with `decimals` decimals, converted back."""
    true = pd.date_range(start, periods=n, freq=pd.Timedelta(seconds=step_s))
    year_start = pd.Timestamp(true[0].year, 1, 1)
    juld = np.round((true - year_start) / pd.Timedelta('1D') + 1, decimals)
    return true, pd.to_datetime(juld - 1, origin=year_start, unit='D')


@pytest.mark.parametrize('step_s', [1, 10, 15, 600])
def test_round_sbe_times_whole_second_sampling_is_exact(step_s):
    true, raw = _timej_times(step_s)
    np.testing.assert_array_equal(sbe._round_sbe_times(raw), true)


@pytest.mark.parametrize('step_s', [0.25, 0.5, 1.5])
def test_round_sbe_times_keeps_sub_second_and_fractional_steps(step_s):
    """Used to round everything to whole seconds: 4 Hz sampling collapsed
    onto duplicate timestamps, and 1.5 s steps became 1 s / 2 s steps."""
    true, raw = _timej_times(step_s)
    out = sbe._round_sbe_times(raw)
    assert (np.diff(out.asi8) > 0).all()  # strictly increasing
    # within the precision of 6-decimal timeJ (~0.09 s)
    assert np.abs((out - true).total_seconds()).max() < 0.05


def test_round_sbe_times_falls_back_if_seconds_would_merge_samples():
    t = pd.to_datetime(['2021-01-01 00:00:00.000', '2021-01-01 00:00:01.000',
                        '2021-01-01 00:00:01.400', '2021-01-01 00:00:02.000',
                        '2021-01-01 00:00:03.000'])
    out = sbe._round_sbe_times(t)
    assert (np.diff(out.asi8) > 0).all()


def test_reference_start_time_nmea_falls_back_to_start_time():
    st = pd.Timestamp('2021-01-05 15:58:23')
    t, source = sbe._reference_start_time({'start_time': st, 'NMEA_time': None},
                                          start_time_NMEA=True)
    assert t == st and 'start_time' in source


def test_missing_start_time_gives_clear_error():
    """Used to raise `Warning(...)` as an exception, or an AttributeError."""
    with pytest.raises(ValueError, match='Did not find a start time'):
        sbe._reference_start_time({'start_time': None, 'NMEA_time': None})
    ds = xr.Dataset()
    with pytest.raises(ValueError, match='Did not find a start time'):
        sbe._add_start_time(ds, {'start_time': None, 'NMEA_time': None},
                            start_time_NMEA=True)


def test_read_header_non_numeric_serial_number(tmp_path):
    """int() on the serial number used to crash the read."""
    src = 'tests/test_data/sbe_files/sbe56/AMU22_SBE56-1225_test.cnv'
    with open(src, encoding='latin-1') as f:
        text = f.read()
    path = tmp_path / 'serial.cnv'
    with open(path, 'w', encoding='latin-1') as f:
        f.write(text.replace("SerialNumber='05611225'", "SerialNumber='0561A225'"))
    assert sbe.read_header(str(path))['instrument_serial_number'] == 'A225'
    assert sbe.read_header(src)['instrument_serial_number'] == '1225'


def test_cnv_time_units_lower_case():
    ds = sbe.read_cnv('tests/test_data/sbe_files/sbe37/cnv/test_sbe37.cnv')
    assert ds.TIME.attrs['units'] == 'days since 1970-01-01 00:00:00'
