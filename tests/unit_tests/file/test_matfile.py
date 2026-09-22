"""
Tests for kval.file.matfile.

Note: xr_to_mat's default output (simplify=False) and mat_to_xr_1D/2D are
NOT a matched read/write pair -- verified empirically before writing these
tests. xr_to_mat's default output nests data under top-level 'coords'/
'attrs'/'dims'/'data_vars' keys (from ds.to_dict()), while mat_to_xr_1D/2D
expect a single MATLAB struct with the data variables as direct fields
(matching the module's own docstring: "a copy of a function used on a
specific project... rather hardcoded"). So each side is tested against
its own actual contract here, not against each other.
"""

import os
import numpy as np
import pytest
import xarray as xr
from datetime import datetime
from scipy.io import savemat, loadmat

from kval.file import matfile
from kval.util.time import datetime_to_matlab_datenum


# ---------------------------------------------------------------------
# mat_to_xr_1D
# ---------------------------------------------------------------------

def _make_struct_matfile(path, extra_vars=None, time_dt=None, struct_name="mystruct"):
    """Write a .mat file shaped the way mat_to_xr_1D/2D actually expect:
    a single top-level MATLAB struct with named fields."""
    if time_dt is None:
        time_dt = [datetime(2021, 1, 2), datetime(2021, 1, 1), datetime(2021, 1, 3)]
    fields = {"time": datetime_to_matlab_datenum(time_dt)}
    if extra_vars:
        fields.update(extra_vars)
    savemat(path, {struct_name: fields})


def test_mat_to_xr_1D_reads_data_and_sorts_chronologically(tmp_path):
    path = str(tmp_path / "test.mat")
    _make_struct_matfile(
        path,
        extra_vars={"TEMP": np.array([2.0, 1.0, 3.0]), "title": "test dataset"},
    )
    ds = matfile.mat_to_xr_1D(path)

    # Input time was [Jan 2, Jan 1, Jan 3] with TEMP [2, 1, 3] -- after
    # sorting chronologically, TEMP should now read [1, 2, 3]
    np.testing.assert_allclose(ds["TEMP"].values, [1.0, 2.0, 3.0])
    assert ds.attrs["title"] == "test dataset"
    assert "TIME" in ds.coords


def test_mat_to_xr_1D_raises_clearly_for_missing_time_variable(tmp_path):
    path = str(tmp_path / "test.mat")
    savemat(path, {"mystruct": {"TEMP": np.array([1.0, 2.0])}})  # no 'time' field
    with pytest.raises(Exception, match="Unable to parse time"):
        matfile.mat_to_xr_1D(path)


def test_mat_to_xr_1D_requires_field_name_for_multiple_data_keys(tmp_path):
    path = str(tmp_path / "test.mat")
    savemat(path, {
        "struct_a": {"time": datetime_to_matlab_datenum([datetime(2021, 1, 1)]), "TEMP": np.array([1.0])},
        "struct_b": {"time": datetime_to_matlab_datenum([datetime(2021, 1, 1)]), "TEMP": np.array([2.0])},
    })
    with pytest.raises(Exception, match="multiple data fields"):
        matfile.mat_to_xr_1D(path)


def test_mat_to_xr_1D_field_name_selects_correct_struct(tmp_path):
    path = str(tmp_path / "test.mat")
    two_times = datetime_to_matlab_datenum([datetime(2021, 1, 1), datetime(2021, 1, 2)])
    savemat(path, {
        "struct_a": {"time": two_times, "TEMP": np.array([1.0, 1.5])},
        "struct_b": {"time": two_times, "TEMP": np.array([99.0, 99.5])},
    })
    ds = matfile.mat_to_xr_1D(path, field_name="struct_b")
    assert ds["TEMP"].values[0] == 99.0


def test_mat_to_xr_1D_custom_time_name(tmp_path):
    path = str(tmp_path / "test.mat")
    savemat(path, {"mystruct": {
        "my_custom_time": datetime_to_matlab_datenum([datetime(2021, 1, 1), datetime(2021, 1, 2)]),
        "TEMP": np.array([1.0, 2.0]),
    }})
    ds = matfile.mat_to_xr_1D(path, time_name="my_custom_time")
    assert "TEMP" in ds
    assert ds.sizes["TIME"] == 2


# ---------------------------------------------------------------------
# mat_to_xr_2D
# ---------------------------------------------------------------------

def test_mat_to_xr_2D_assigns_dimensions_correctly(tmp_path):
    path = str(tmp_path / "test2d.mat")
    time_dt = [datetime(2021, 1, 1), datetime(2021, 1, 2)]
    savemat(path, {"mystruct": {
        "time": datetime_to_matlab_datenum(time_dt),
        "PRES": np.array([10.0, 20.0, 30.0]),
        "TEMP": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # (TIME, PRES)
    }})
    ds = matfile.mat_to_xr_2D(path)

    assert ds.sizes == {"TIME": 2, "PRES": 3}
    np.testing.assert_allclose(ds["TEMP"].values, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_mat_to_xr_2D_transposes_when_dims_are_reversed(tmp_path):
    """TEMP shaped (PRES, TIME) instead of (TIME, PRES) should be
    transposed automatically to (TIME, PRES)."""
    path = str(tmp_path / "test2d_transposed.mat")
    time_dt = [datetime(2021, 1, 1), datetime(2021, 1, 2)]
    temp_pres_time = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])  # (PRES=3, TIME=2)
    savemat(path, {"mystruct": {
        "time": datetime_to_matlab_datenum(time_dt),
        "PRES": np.array([10.0, 20.0, 30.0]),
        "TEMP": temp_pres_time,
    }})
    ds = matfile.mat_to_xr_2D(path)
    assert ds["TEMP"].dims == ("TIME", "PRES")
    np.testing.assert_allclose(ds["TEMP"].values, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_mat_to_xr_2D_raises_for_missing_dim2_variable(tmp_path):
    path = str(tmp_path / "test2d.mat")
    savemat(path, {"mystruct": {
        "time": datetime_to_matlab_datenum([datetime(2021, 1, 1), datetime(2021, 1, 2)]),
        "TEMP": np.array([1.0, 2.0]),
    }})  # no 'PRES' field
    with pytest.raises(Exception, match="PRES"):
        matfile.mat_to_xr_2D(path)


# ---------------------------------------------------------------------
# xr_to_mat
# ---------------------------------------------------------------------

def _make_test_ds():
    ds = xr.Dataset(
        {"TEMP": ("TIME", np.array([1.0, 2.0, 3.0]))},
        coords={"TIME": ("TIME", np.array([0.0, 1.0, 2.0]))},
    )
    ds["TIME"].attrs["units"] = "days since 2021-01-01"
    ds.attrs["title"] = "test dataset"
    return ds


def test_xr_to_mat_simplified_output_is_flat_and_readable(tmp_path):
    ds = _make_test_ds()
    path = str(tmp_path / "simple.mat")
    matfile.xr_to_mat(ds, path, simplify=True)

    loaded = loadmat(path, squeeze_me=True)
    keys = {k for k in loaded.keys() if not k.startswith("__")}
    assert keys == {"TIME", "TEMP", "TIME_mat"}
    np.testing.assert_allclose(loaded["TEMP"], [1.0, 2.0, 3.0])


def test_xr_to_mat_adds_matlab_datenum_time(tmp_path):
    ds = _make_test_ds()
    path = str(tmp_path / "simple.mat")
    matfile.xr_to_mat(ds, path, simplify=True)

    loaded = loadmat(path, squeeze_me=True)
    # 2021-01-01 as MATLAB datenum, verified via the same conversion
    # function used elsewhere in this session (738521 range for 2021)
    assert loaded["TIME_mat"][0] > 738000  # sanity: genuinely in 2021, not 1970 or 0


def test_xr_to_mat_full_output_contains_expected_top_level_structure(tmp_path):
    """Default (simplify=False) output nests everything under
    coords/attrs/dims/data_vars, per ds.to_dict()'s own structure --
    this is NOT the flat format mat_to_xr_1D/2D expect to read (see
    module docstring above); this test documents/pins that structure,
    not a round trip."""
    ds = _make_test_ds()
    path = str(tmp_path / "full.mat")
    matfile.xr_to_mat(ds, path, simplify=False)

    loaded = loadmat(path, squeeze_me=True)
    keys = {k for k in loaded.keys() if not k.startswith("__")}
    assert keys == {"coords", "attrs", "dims", "data_vars"}


def test_xr_to_mat_appends_mat_extension_if_missing(tmp_path):
    ds = _make_test_ds()
    path_no_ext = str(tmp_path / "no_extension")
    matfile.xr_to_mat(ds, path_no_ext, simplify=True)
    assert os.path.exists(path_no_ext + ".mat")