"""
Tests for the interactive widget classes in edit.py, _ctdprof_edit.py, and
_moored_tools.py.

These deliberately do NOT test the matplotlib/RectangleSelector mouse-drag
UI itself (not practical or valuable to test headlessly). Instead they
instantiate each class directly (confirmed internals.check_interactive()
is a safe no-op outside a notebook) and manually drive the underlying
state (e.g. setting .remove_bool or .selected_options directly, as if
points/checkboxes had already been selected), then invoke the button
callback methods directly with a synthetic button/change argument.

The specific thing under test is the resilience concern this was written
for: that after a simulated "apply"/"remove" click, self.ds actually
reflects the change -- and keeps reflecting it -- rather than being lost
to a copy (the historical failure mode referenced in GitHub issue #47).
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import warnings

from kval.data.edit import drop_vars_pick
from kval.data.ctdprof_tools._ctdprof_edit import hand_remove_points as ctd_hand_remove_points
from kval.data.moored_tools._moored_tools import hand_remove_points as moored_hand_remove_points


@pytest.fixture(autouse=True)
def _suppress_interactive_warnings():
    """internals.check_interactive() warns when not in a notebook; that
    warning is expected and irrelevant to what these tests check."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ---------------------------------------------------------------------
# edit.drop_vars_pick
# ---------------------------------------------------------------------

def _make_drop_vars_ds():
    return xr.Dataset({
        "TEMP": ("TIME", np.array([1.0, 2.0, 3.0])),
        "PSAL": ("TIME", np.array([34.0, 34.1, 34.2])),
    }, coords={"TIME": np.array([0, 1, 2])})


def test_drop_vars_pick_removes_selected_variable_ctd_branch():
    ds = _make_drop_vars_ds()
    picker = drop_vars_pick(ds, moored=False)

    picker.selected_options = ["PSAL"]
    picker.on_remove_button_click({"new": True}, ds)

    assert "PSAL" not in picker.ds.data_vars
    assert "TEMP" in picker.ds.data_vars


def test_drop_vars_pick_removes_selected_variable_moored_branch():
    ds = _make_drop_vars_ds()
    picker = drop_vars_pick(ds, moored=True)

    picker.selected_options = ["TEMP"]
    picker.on_remove_button_click({"new": True}, ds)

    assert "TEMP" not in picker.ds.data_vars
    assert "PSAL" in picker.ds.data_vars


def test_drop_vars_pick_change_persists_after_click_completes():
    """The specific resilience check: the change should still be visible
    on picker.ds after the callback returns, not just transiently during
    it (this is exactly the class of bug referenced by GitHub issue #47
    -- a function silently operating on/returning a copy that then gets
    discarded)."""
    ds = _make_drop_vars_ds()
    picker = drop_vars_pick(ds, moored=False)
    picker.selected_options = ["PSAL"]
    picker.on_remove_button_click({"new": True}, ds)

    # Simulate some time passing / other code running -- re-check the
    # same attribute again to confirm it wasn't a transient state.
    assert "PSAL" not in picker.ds.data_vars


def test_drop_vars_pick_exit_button_does_not_modify_dataset():
    ds = _make_drop_vars_ds()
    picker = drop_vars_pick(ds, moored=False)
    original_vars = set(picker.ds.data_vars)

    picker.on_exit_button_click({"new": True})

    assert set(picker.ds.data_vars) == original_vars


def test_drop_vars_pick_does_not_mutate_caller_input_dataset():
    """picker.ds should be an internal copy -- the dataset the caller
    passed in should be untouched."""
    ds = _make_drop_vars_ds()
    original_vars = set(ds.data_vars)
    picker = drop_vars_pick(ds, moored=False)
    picker.selected_options = ["PSAL"]
    picker.on_remove_button_click({"new": True}, ds)

    assert set(ds.data_vars) == original_vars  # caller's ds unchanged
    assert "PSAL" not in picker.ds.data_vars    # picker's internal copy changed


# ---------------------------------------------------------------------
# ctdprof_tools._ctdprof_edit.hand_remove_points (profile: TIME x PRES)
# ---------------------------------------------------------------------

def _make_ctd_profile_ds():
    n_pres = 5
    ds = xr.Dataset(
        {"TEMP": (("TIME", "PRES"), np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]),
                  {"units": "degree_C"})},
        coords={"TIME": ("TIME", np.array([0.0])),
                "PRES": ("PRES", np.arange(n_pres, dtype=float))},
    )
    ds["TIME"].attrs["units"] = "days since 2021-01-01"
    ds["PRES"].attrs["units"] = "dbar"
    return ds


def test_ctd_hand_remove_points_apply_sets_selected_points_to_nan():
    ds = _make_ctd_profile_ds()
    editor = ctd_hand_remove_points(ds, "TEMP", TIME_index=0)

    editor.remove_bool = np.array([False, True, False, True, False])
    editor.exit_and_apply_var(button=None)

    result = editor.ds["TEMP"].isel(TIME=0).values
    np.testing.assert_array_equal(np.isnan(result), [False, True, False, True, False])
    np.testing.assert_allclose(result[~np.isnan(result)], [1.0, 3.0, 5.0])


def test_ctd_hand_remove_points_records_manual_editing_attribute():
    ds = _make_ctd_profile_ds()
    editor = ctd_hand_remove_points(ds, "TEMP", TIME_index=0)
    editor.remove_bool = np.array([False, True, False, True, False])
    editor.exit_and_apply_var(button=None)

    assert "2" in editor.ds["TEMP"].attrs["manual_editing"]


def test_ctd_hand_remove_points_change_persists_on_self_ds():
    """Resilience check, same concern as the drop_vars_pick test above."""
    ds = _make_ctd_profile_ds()
    editor = ctd_hand_remove_points(ds, "TEMP", TIME_index=0)
    editor.remove_bool = np.array([True, False, False, False, False])
    editor.exit_and_apply_var(button=None)

    # Re-access after the call, from the object's own state
    result_again = editor.ds["TEMP"].isel(TIME=0).values
    assert np.isnan(result_again[0])


def test_ctd_hand_remove_points_raises_for_invalid_variable():
    ds = _make_ctd_profile_ds()
    with pytest.raises(Exception, match="Invalid variable"):
        ctd_hand_remove_points(ds, "NOT_A_VARIABLE", TIME_index=0)


# ---------------------------------------------------------------------
# moored_tools._moored_tools.hand_remove_points (time series)
# ---------------------------------------------------------------------

def _make_moored_ds():
    n = 5
    ds = xr.Dataset(
        {"TEMP": ("TIME", np.array([1.0, 2.0, 3.0, 4.0, 5.0]), {"units": "degree_C"})},
        coords={"TIME": pd.date_range("2021-01-01", periods=n)},
    )
    return ds


def test_moored_hand_remove_points_apply_sets_selected_points_to_nan():
    ds = _make_moored_ds()
    editor = moored_hand_remove_points(ds, "TEMP")

    editor.remove_bool = np.array([False, True, False, True, False])
    editor.exit_and_apply_var(button=None)

    result = editor.ds["TEMP"].values
    np.testing.assert_array_equal(np.isnan(result), [False, True, False, True, False])


def test_moored_hand_remove_points_change_persists_on_self_ds():
    ds = _make_moored_ds()
    editor = moored_hand_remove_points(ds, "TEMP")
    editor.remove_bool = np.array([True, False, False, False, False])
    editor.exit_and_apply_var(button=None)

    assert np.isnan(editor.ds["TEMP"].values[0])


def test_moored_hand_remove_points_records_manual_editing_attribute():
    ds = _make_moored_ds()
    editor = moored_hand_remove_points(ds, "TEMP")
    editor.remove_bool = np.array([False, True, True, False, False])
    editor.exit_and_apply_var(button=None)

    assert "2" in editor.ds["TEMP"].attrs["manual_editing"]


def test_moored_hand_remove_points_raises_for_invalid_variable():
    ds = _make_moored_ds()
    with pytest.raises(Exception, match="Invalid variable"):
        moored_hand_remove_points(ds, "NOT_A_VARIABLE")


def test_moored_hand_remove_points_supports_separate_edit_variable():
    """varnm_edit lets you visualize one variable but edit a different
    one (e.g. inspect TEMP1 but actually edit a derived/QC variable)."""
    ds = _make_moored_ds()
    ds["TEMP_QC"] = ("TIME", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    editor = moored_hand_remove_points(ds, "TEMP", varnm_edit="TEMP_QC")

    editor.remove_bool = np.array([False, False, True, False, False])
    editor.exit_and_apply_var(button=None)

    assert np.isnan(editor.ds["TEMP_QC"].values[2])
    assert not np.isnan(editor.ds["TEMP"].values[2])  # visualized var untouched

def test_ctd_hand_remove_points_forget_selection_does_not_crash():
    """Regression test: forget_selection had `np.bool_(len())` -- len()
    called with zero arguments -- which raised TypeError immediately on
    every click of the 'Forget selection' button."""
    ds = _make_ctd_profile_ds()
    editor = ctd_hand_remove_points(ds, "TEMP", TIME_index=0)
    editor.forget_selection(button=None)  # should not raise
    assert editor.TF_indixes_selected.shape == (5,)
    assert not editor.TF_indixes_selected.any()