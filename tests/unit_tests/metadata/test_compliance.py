"""
Tests for kval.metadata.compliance.

compliance_checks_custom prints its report rather than returning a value,
so these tests capture stdout (via pytest's capsys) and check for the
expected pass/fail markers rather than a return value.

compliance_checks_ioos wraps the third-party compliance-checker package;
those tests mock ComplianceChecker.run_checker rather than depending on
the real external checker, so they run reliably without network access
or a fully configured checker environment.
"""

import contextlib
import io
import os
import sys
import numpy as np
import pytest
import xarray as xr

from kval.metadata import compliance
from unittest.mock import patch, MagicMock

from kval.metadata.compliance import compliance_checks_custom, compliance_checks_ioos


def _make_compliant_ds():
    """A dataset that should pass every required/recommended check in
    compliance_checks_custom except the (deliberately extensive) list of
    recommended global attributes -- verified to genuinely produce
    'All checks passed (0 issues)' before being used as a baseline here."""
    n = 5
    ds = xr.Dataset(
        {
            "TEMP": (("TIME",), np.arange(n, dtype="float32"),
                     {"units": "degree_C", "standard_name": "sea_water_temperature",
                      "_FillValue": np.float32(-999), "processing_level": "L2",
                      "QC_indicator": "good_data"}),
        },
        coords={
            "TIME": (("TIME",), np.arange(n, dtype="float32"),
                      {"units": "days since 1970-01-01", "axis": "T",
                       "coverage_content_type": "coordinate",
                       "standard_name": "time", "_FillValue": np.float32(-999)}),
            "LATITUDE": ((), np.float32(80.0),
                         {"axis": "Y", "coverage_content_type": "coordinate",
                          "units": "degree_north", "standard_name": "latitude",
                          "_FillValue": np.float32(-999)}),
            "LONGITUDE": ((), np.float32(30.0),
                          {"axis": "X", "coverage_content_type": "coordinate",
                           "units": "degree_east", "standard_name": "longitude",
                           "_FillValue": np.float32(-999)}),
        },
    )
    ds.attrs.update({
        "title": "Test dataset", "summary": "A test dataset", "Conventions": "CF-1.8, ACDD-1.3",
        "creator_name": "Test", "creator_email": "test@example.com", "institution": "NPI",
        "keywords": "ocean", "date_created": "2024-01-01", "featureType": "timeSeries",
    })
    return ds


# ---------------------------------------------------------------------
# compliance_checks_custom
# ---------------------------------------------------------------------

def test_compliant_dataset_reports_zero_issues(capsys):
    compliance_checks_custom(_make_compliant_ds())
    out = capsys.readouterr().out
    assert "All checks passed (0 issues)" in out


def test_64bit_dtype_is_flagged(capsys):
    ds = _make_compliant_ds()
    ds["TEMP"] = ds["TEMP"].astype("float64")
    compliance_checks_custom(ds)
    out = capsys.readouterr().out
    assert "64-bit types" in out
    assert "TEMP" in out


def test_TBW_placeholder_attribute_is_flagged(capsys):
    ds = _make_compliant_ds()
    ds.attrs["summary"] = "TBW"
    compliance_checks_custom(ds)
    out = capsys.readouterr().out
    assert "TBW placeholders" in out
    assert "summary" in out


def test_missing_fill_value_is_flagged(capsys):
    ds = _make_compliant_ds()
    del ds["TEMP"].attrs["_FillValue"]
    compliance_checks_custom(ds)
    out = capsys.readouterr().out
    assert "_FillValue" in out
    assert "TEMP" in out


def test_missing_units_is_flagged(capsys):
    ds = _make_compliant_ds()
    del ds["TEMP"].attrs["units"]
    compliance_checks_custom(ds)
    out = capsys.readouterr().out
    assert "Missing 'units'" in out
    assert "TEMP" in out


def test_missing_standard_name_and_long_name_is_flagged(capsys):
    ds = _make_compliant_ds()
    del ds["TEMP"].attrs["standard_name"]
    compliance_checks_custom(ds)
    out = capsys.readouterr().out
    assert "standard_name" in out


def test_sbe_flag_variable_is_flagged(capsys):
    ds = _make_compliant_ds()
    ds["SBE_FLAG"] = ("TIME", np.zeros(5))
    compliance_checks_custom(ds)
    out = capsys.readouterr().out
    assert "SBE_FLAG" in out


def test_non_monotonic_time_is_flagged(capsys):
    ds = _make_compliant_ds()
    ds["TIME"] = ("TIME", np.array([0, 2, 1, 3, 4], dtype="float32"))
    compliance_checks_custom(ds)
    out = capsys.readouterr().out
    assert "monotonic" in out.lower()


def test_missing_required_global_attribute_does_not_raise_and_is_reported_first(capsys):
    """Regression test: this used to raise ValueError and abort before any
    other check ran (fixed earlier this session). It should now print a
    report -- with the missing-required-attrs banner first -- rather than
    raising, so the rest of the checks still run and get reported."""
    ds = _make_compliant_ds()
    del ds.attrs["title"]
    del ds.attrs["creator_email"]
    # Should not raise:
    compliance_checks_custom(ds)
    out = capsys.readouterr().out
    assert "MISSING REQUIRED GLOBAL ATTRIBUTES" in out
    assert "title" in out
    assert "creator_email" in out
    # Confirm other checks still ran and were reported (proof execution
    # continued past the missing-attrs check):
    assert "Passed checks" in out


def test_conventions_missing_cf_is_flagged(capsys):
    ds = _make_compliant_ds()
    ds.attrs["Conventions"] = "ACDD-1.3"
    compliance_checks_custom(ds)
    out = capsys.readouterr().out
    assert "does not mention CF" in out


def test_flag_variable_missing_attrs_is_flagged(capsys):
    ds = _make_compliant_ds()
    ds["TEMP_FLAG"] = ("TIME", np.zeros(5, dtype="int32"))
    compliance_checks_custom(ds)
    out = capsys.readouterr().out
    assert "flag_meanings" in out or "flag_values" in out


def test_processing_level_both_global_and_variable_is_flagged(capsys):
    ds = _make_compliant_ds()
    ds.attrs["processing_level"] = "L2"  # already on TEMP too -> both places
    compliance_checks_custom(ds)
    out = capsys.readouterr().out
    assert "processing_level" in out
    assert "exists globally and on vars" in out


# ---------------------------------------------------------------------
# compliance_checks_ioos (mocked -- no dependency on the real
# compliance-checker package or network access)
# ---------------------------------------------------------------------

def test_compliance_checks_ioos_raises_clearly_if_checker_not_installed():
    with patch("kval.metadata.compliance.COMPLIANCE_CHECKER_AVAILABLE", False):
        with pytest.raises(ImportError, match="IOOS Compliance Checker is not installed"):
            compliance_checks_ioos("some_file.nc")


def test_compliance_checks_ioos_calls_run_checker_with_cf_and_acdd(tmp_path):
    ds = _make_compliant_ds()
    nc_path = tmp_path / "test.nc"
    ds.to_netcdf(nc_path)

    with patch("kval.metadata.compliance.COMPLIANCE_CHECKER_AVAILABLE", True), \
         patch("kval.metadata.compliance._in_notebook", return_value=False), \
         patch("kval.metadata.compliance.CheckSuite") as mock_suite, \
         patch("kval.metadata.compliance.ComplianceChecker") as mock_checker:
        mock_checker.run_checker.return_value = (True, [])
        compliance_checks_ioos(str(nc_path))

        mock_suite.return_value.load_all_available_checkers.assert_called_once()
        args, kwargs = mock_checker.run_checker.call_args
        assert args[0] == str(nc_path)
        assert args[1] == ["cf", "acdd"]


def test_compliance_checks_ioos_accepts_dataset_and_cleans_up_temp_file(tmp_path, monkeypatch):
    ds = _make_compliant_ds()
    monkeypatch.chdir(tmp_path)

    with patch("kval.metadata.compliance.COMPLIANCE_CHECKER_AVAILABLE", True), \
         patch("kval.metadata.compliance._in_notebook", return_value=False), \
         patch("kval.metadata.compliance.CheckSuite") as mock_suite, \
         patch("kval.metadata.compliance.ComplianceChecker") as mock_checker:
        mock_checker.run_checker.return_value = (True, [])
        compliance_checks_ioos(ds)

        # The temp file used for the in-memory Dataset should be cleaned
        # up afterwards, not left behind.
        assert not (tmp_path / "temp.nc").exists()

# ---------------------------------------------------------------------
# Temporary-file handling for Dataset input.
#
# The checker only works on files, so a Dataset is written to a temporary
# copy first. These tests pin down that the copy (a) never lands in the
# user's working directory, and (b) is removed even when the check does
# not finish -- an exception, or a user interrupting a slow/hanging run.
# ---------------------------------------------------------------------

def _patched_ioos(**kwargs):
    """Common patches for exercising the IOOS path without the real checker."""
    return patch.multiple(
        "kval.metadata.compliance",
        COMPLIANCE_CHECKER_AVAILABLE=True,
        **kwargs,
    )


def test_dataset_temp_file_is_not_written_to_working_directory(tmp_path, monkeypatch):
    ds = _make_compliant_ds()
    monkeypatch.chdir(tmp_path)

    seen = {}

    def fake_run(path):
        # Record what the checker was handed, and what the working
        # directory looks like *while* the check is running.
        seen["path"] = path
        seen["cwd_contents"] = sorted(os.listdir(os.getcwd()))

    with _patched_ioos(_in_notebook=lambda: False, _run_ioos_checkers=fake_run):
        compliance_checks_ioos(ds)

    assert os.path.basename(seen["path"]) == "temp.nc"
    # The temp copy exists somewhere, but not in the working directory
    assert os.path.dirname(seen["path"]) != str(tmp_path)
    assert seen["cwd_contents"] == []
    assert sorted(os.listdir(tmp_path)) == []


def test_dataset_temp_file_removed_when_checker_raises(tmp_path, monkeypatch):
    ds = _make_compliant_ds()
    monkeypatch.chdir(tmp_path)

    temp_dirs = []

    def boom(path):
        temp_dirs.append(os.path.dirname(path))
        assert os.path.exists(path)  # it really was written
        raise RuntimeError("checker blew up")

    with _patched_ioos(_in_notebook=lambda: False, _run_ioos_checkers=boom):
        with pytest.raises(RuntimeError, match="checker blew up"):
            compliance_checks_ioos(ds)

    assert not os.path.exists(temp_dirs[0])
    assert sorted(os.listdir(tmp_path)) == []


def test_dataset_temp_file_removed_on_keyboard_interrupt(tmp_path, monkeypatch):
    """A hanging check that the user aborts must not leave a stale file."""
    ds = _make_compliant_ds()
    monkeypatch.chdir(tmp_path)

    temp_dirs = []

    def interrupt(path):
        temp_dirs.append(os.path.dirname(path))
        raise KeyboardInterrupt

    with _patched_ioos(_in_notebook=lambda: False, _run_ioos_checkers=interrupt):
        with pytest.raises(KeyboardInterrupt):
            compliance_checks_ioos(ds)

    assert not os.path.exists(temp_dirs[0])
    assert sorted(os.listdir(tmp_path)) == []


def test_file_path_input_writes_no_temp_file(tmp_path, monkeypatch):
    """A path is passed straight through -- nothing should be written."""
    ds = _make_compliant_ds()
    nc_path = tmp_path / "source.nc"
    ds.to_netcdf(nc_path)

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    seen = {}
    with _patched_ioos(
        _in_notebook=lambda: False,
        _run_ioos_checkers=lambda path: seen.update(path=path),
    ):
        compliance_checks_ioos(str(nc_path))

    assert seen["path"] == str(nc_path)
    assert sorted(os.listdir(work_dir)) == []


# ---------------------------------------------------------------------
# Dispatch and the optional-dependency guard
# ---------------------------------------------------------------------

def test_notebook_context_uses_the_button_wrapper():
    with _patched_ioos(
        _in_notebook=lambda: True,
        _compliance_checks_ioos_with_button=MagicMock(),
        _compliance_checks_ioos_plain=MagicMock(),
    ):
        from kval.metadata import compliance

        compliance.compliance_checks_ioos("some_file.nc")
        compliance._compliance_checks_ioos_with_button.assert_called_once_with(
            "some_file.nc"
        )
        compliance._compliance_checks_ioos_plain.assert_not_called()


def test_terminal_context_uses_the_plain_wrapper():
    with _patched_ioos(
        _in_notebook=lambda: False,
        _compliance_checks_ioos_with_button=MagicMock(),
        _compliance_checks_ioos_plain=MagicMock(),
    ):
        from kval.metadata import compliance

        compliance.compliance_checks_ioos("some_file.nc")
        compliance._compliance_checks_ioos_plain.assert_called_once_with(
            "some_file.nc"
        )
        compliance._compliance_checks_ioos_with_button.assert_not_called()


@pytest.mark.parametrize(
    "func_name",
    ["compliance_checks_ioos",
     "_compliance_checks_ioos_plain",
     "_compliance_checks_ioos_with_button"],
)
def test_all_ioos_entry_points_guard_the_optional_dependency(func_name):
    from kval.metadata import compliance

    with patch("kval.metadata.compliance.COMPLIANCE_CHECKER_AVAILABLE", False):
        with pytest.raises(ImportError, match="IOOS Compliance Checker is not installed"):
            getattr(compliance, func_name)("some_file.nc")


# ---------------------------------------------------------------------
# Small regressions in compliance_checks_custom
# ---------------------------------------------------------------------

def test_recommended_global_attributes_are_not_listed_twice(capsys):
    """'platform_vocabulary' used to appear twice in the recommended list."""
    ds = _make_compliant_ds()
    compliance_checks_custom(ds)
    out = capsys.readouterr().out
    assert out.count("platform_vocabulary") <= 1


# ---------------------------------------------------------------------
# Suppressing known compliance-checker bugs
#
# The checker reports crashed checks by printing to stderr, not through
# the warnings machinery, so these have to be filtered out of the stream.
# ---------------------------------------------------------------------

_HEADER = ("WARNING: The following exceptions occurred during the cf checker "
           "(possibly indicate compliance checker issues):\n")


def test_known_bug_and_its_header_are_both_removed():
    text = _HEADER + "cf.check_domain_variables: list index out of range\n"
    out = compliance._filter_checker_stderr(text, ("cf.check_domain_variables",))
    assert out.strip() == ""


def test_a_genuine_failure_still_gets_reported():
    text = (_HEADER
            + "cf.check_domain_variables: list index out of range\n"
            + "cf.check_something_real: a real problem\n")
    out = compliance._filter_checker_stderr(text, ("cf.check_domain_variables",))

    assert "check_something_real" in out
    assert "check_domain_variables" not in out
    assert "WARNING" in out          # header kept, since something survived


def test_unrelated_stderr_output_passes_through_untouched():
    text = "Using packaged standard name table v93\n"
    out = compliance._filter_checker_stderr(text, ("cf.check_domain_variables",))
    assert out == text


def test_nothing_is_suppressed_when_the_list_is_empty():
    text = _HEADER + "cf.check_domain_variables: list index out of range\n"
    out = compliance._filter_checker_stderr(text, ())
    assert "check_domain_variables" in out


def test_suppression_can_be_switched_off(tmp_path):
    """suppress_known_bugs=False must not touch stderr at all."""
    calls = {}

    def fake_run_checker(path, names, verbose, criteria):
        calls["ran"] = True
        sys.stderr.write(_HEADER)
        sys.stderr.write("cf.check_domain_variables: list index out of range\n")

    err = io.StringIO()
    with patch.multiple("kval.metadata.compliance",
                        COMPLIANCE_CHECKER_AVAILABLE=True,
                        CheckSuite=MagicMock(),
                        ComplianceChecker=MagicMock(run_checker=fake_run_checker)):
        with contextlib.redirect_stderr(err):
            compliance._run_ioos_checkers("dummy.nc", suppress_known_bugs=False)

    assert calls["ran"]
    assert "check_domain_variables" in err.getvalue()


def test_suppression_is_on_by_default(tmp_path):
    def fake_run_checker(path, names, verbose, criteria):
        sys.stderr.write(_HEADER)
        sys.stderr.write("cf.check_domain_variables: list index out of range\n")

    err = io.StringIO()
    with patch.multiple("kval.metadata.compliance",
                        COMPLIANCE_CHECKER_AVAILABLE=True,
                        CheckSuite=MagicMock(),
                        ComplianceChecker=MagicMock(run_checker=fake_run_checker)):
        with contextlib.redirect_stderr(err):
            compliance._run_ioos_checkers("dummy.nc")

    assert err.getvalue().strip() == ""


def test_stderr_is_not_swallowed_when_the_checker_raises():
    """A crash must not eat the output captured before it."""
    def exploding_run_checker(path, names, verbose, criteria):
        sys.stderr.write("something important\n")
        raise RuntimeError("boom")

    err = io.StringIO()
    with patch.multiple("kval.metadata.compliance",
                        COMPLIANCE_CHECKER_AVAILABLE=True,
                        CheckSuite=MagicMock(),
                        ComplianceChecker=MagicMock(run_checker=exploding_run_checker)):
        with contextlib.redirect_stderr(err):
            with pytest.raises(RuntimeError, match="boom"):
                compliance._run_ioos_checkers("dummy.nc")

    assert "something important" in err.getvalue()