"""
Tests for kval.data.ship_ctd_tools._ctd_tools.

Focused on join_cruise_btl and its supporting helpers (_btl_files_from_path,
_datasets_from_btllist), which previously had almost no test coverage
(~15% of join_cruise_btl's own lines were ever exercised, vs. ~75% for its
.cnv sibling join_cruise, which gets substantial incidental coverage via
ctd.ctds_from_cnv_dir's existing tests).
"""
import pytest
import xarray as xr
import numpy as np
import warnings

from kval.data import ctd
from kval.data.ship_ctd_tools import _ctd_tools as tools


DML_2020_DIR = 'tests/test_data/sbe_files/sbe911plus/dml_2020'
KONGSFJORDEN_DIR = 'tests/test_data/sbe_files/sbe911plus/kongsfjorden_ctds'


# --- dataset_from_btl_dir (full pipeline: _btl_files_from_path ->
# _datasets_from_btllist -> join_cruise_btl) ---

class TestDatasetFromBtlDir:

    def test_basic_multi_file_join(self):
        """Joining 3 real .btl profiles should produce one Dataset with
        TIME length 3, real station/lat/lon values, and a CRUISE
        variable (even if just the placeholder, since these files don't
        have a 'cruise' attribute set)."""
        ds = ctd.dataset_from_btl_dir(DML_2020_DIR, verbose=False)
        assert isinstance(ds, xr.Dataset)
        assert ds.sizes['TIME'] == 3
        assert 'STATION' in ds
        assert 'CRUISE' in ds
        assert 'PSAL1' in ds.data_vars

    def test_time_is_sorted(self):
        """join_cruise_btl explicitly sorts by TIME -- confirm the
        output actually is, regardless of input file order."""
        ds = ctd.dataset_from_btl_dir(DML_2020_DIR, verbose=False)
        assert list(ds.TIME.values) == sorted(ds.TIME.values)

    def test_single_file_directory(self):
        """A directory with just one .btl file should still work (the
        join loop's first-iteration-only path)."""
        ds = ctd.dataset_from_btl_dir(KONGSFJORDEN_DIR, verbose=False)
        assert isinstance(ds, xr.Dataset)
        assert ds.sizes['TIME'] == 1

    def test_no_btl_files_raises(self, tmp_path):
        """An existing but empty directory should raise FileNotFoundError
        with a clear message, not fail some other way."""
        with pytest.raises(FileNotFoundError, match='Did not find any .btl'):
            ctd.dataset_from_btl_dir(str(tmp_path), verbose=False)

    def test_no_future_warning_from_concat(self):
        """Regression test: join_cruise_btl's own xr.concat call used to
        emit a FutureWarning about the default 'join' behavior changing
        in a future xarray version (the same issue already fixed once
        for join_cruise's separate, near-duplicate implementation, but
        not this one)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ctd.dataset_from_btl_dir(DML_2020_DIR, verbose=False)
            future_warnings = [x for x in w
                              if issubclass(x.category, FutureWarning)]
        assert not future_warnings, (
            f"Unexpected FutureWarning(s): {[str(x.message) for x in future_warnings]}")


# --- join_cruise_btl itself, called directly ---

class TestJoinCruiseBtl:

    def test_invalid_input_type_raises(self):
        """A completely invalid input type (not a path string or a list
        of Datasets) should raise a clear Exception."""
        with pytest.raises(Exception, match='Input \\*datasets\\* invalid'):
            tools.join_cruise_btl(12345)

    def test_accepts_list_of_datasets_directly(self):
        """join_cruise_btl's second documented input form -- a plain list
        of xr.Dataset objects, not a path -- is what ctd.dataset_from_btl_dir
        actually uses internally, so it needs its own direct coverage."""
        btl_files = tools._btl_files_from_path(DML_2020_DIR)
        profile_datasets = tools._datasets_from_btllist(
            btl_files, verbose=False, start_time_NMEA=False,
            time_adjust_NMEA=False, station_from_filename=False)

        ds = tools.join_cruise_btl(profile_datasets, verbose=False)
        assert ds.sizes['TIME'] == 3
        assert list(ds['STATION'].values) == ['012', '014', '015']

    def test_invalid_list_contents_raises(self):
        """A list that isn't all xr.Dataset objects should also raise,
        not silently misbehave."""
        with pytest.raises(Exception, match='Input \\*datasets\\* invalid'):
            tools.join_cruise_btl(['not', 'a', 'dataset'])


# --- Helper functions ---

class TestBtlFilesFromPath:

    def test_finds_all_btl_files(self):
        files = tools._btl_files_from_path(DML_2020_DIR)
        assert isinstance(files, list)
        assert len(files) == 3
        assert all(f.endswith('.btl') for f in files)

    def test_empty_directory_returns_empty_list(self, tmp_path):
        files = tools._btl_files_from_path(str(tmp_path))
        assert files == []


class TestDatasetsFromBtllist:

    def test_returns_one_dataset_per_file(self):
        btl_files = tools._btl_files_from_path(DML_2020_DIR)
        profile_datasets = tools._datasets_from_btllist(
            btl_files, verbose=False, start_time_NMEA=False,
            time_adjust_NMEA=False, station_from_filename=False)
        assert isinstance(profile_datasets, list)
        assert len(profile_datasets) == len(btl_files)
        assert all(isinstance(d, xr.Dataset) for d in profile_datasets)