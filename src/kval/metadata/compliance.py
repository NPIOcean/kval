"""
kval.metadata.compliance
========================

Checks that a dataset is ready for publication:

- ``compliance_checks_ioos``  -- runs the external IOOS compliance-checker
  (CF and ACDD) on a netCDF file or an in-memory ``xarray.Dataset``.
- ``compliance_checks_custom`` -- ad-hoc checks for CF/ACDD compatibility
  and NPI practice, printed as a human-readable report.

``compliance-checker`` is an optional dependency (it is awkward to install
on some platforms, and is not installed on Windows), so it is imported
conditionally and the IOOS functions raise a clear ImportError if it is
missing.
"""

import contextlib
import io
import os
import re
import sys
import tempfile

import numpy as np
import xarray as xr
from IPython.display import display, clear_output
import ipywidgets as widgets

from kval.util import netcdf


# Conditional import for compliance-checker
try:
    from compliance_checker.runner import ComplianceChecker, CheckSuite

    COMPLIANCE_CHECKER_AVAILABLE = True
except ImportError:
    COMPLIANCE_CHECKER_AVAILABLE = False


_CHECKER_MISSING_MESSAGE = (
    "IOOS Compliance Checker is not installed. "
    "Please install it to use this functionality, e.g.:\n"
    "$ conda install -c conda-forge compliance-checker\nor\n"
    "$ pip install compliance-checker\n\n"
    "(Note: There are some issues with getting the dependencies "
    " of this library to work on Py3.12, MacOS and Windows..)"
)


def _require_compliance_checker() -> None:
    """
    Raise a helpful ImportError if the optional compliance-checker
    dependency is not available.
    """
    if not COMPLIANCE_CHECKER_AVAILABLE:
        raise ImportError(_CHECKER_MISSING_MESSAGE)


def _in_notebook() -> bool:
    """Check whether we're running inside a Jupyter notebook/lab (not a
    plain terminal or script), where ipywidgets will actually render."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return (
            shell is not None
            and shell.__class__.__name__ == "ZMQInteractiveShell"
        )
    except ImportError:
        return False


def compliance_checks_ioos(file):
    """
    Use the IOOS compliance checker to check an nc file (CF and ACDD
    conventions). Can take a file path or an xr.Dataset as input.

    If running in a Jupyter notebook, results are shown with a "close"
    button; otherwise, results print directly.
    """
    _require_compliance_checker()

    if _in_notebook():
        _compliance_checks_ioos_with_button(file)
    else:
        _compliance_checks_ioos_plain(file)


def _run_ioos_checkers(path: str,
                      suppress_known_bugs: bool = True) -> None:
    """
    Run the IOOS CF and ACDD checkers on the netCDF file at *path*.

    Results are printed by the checker itself; nothing is returned.

    Parameters
    ----------
    path : str
        netCDF file to check.
    suppress_known_bugs : bool, default=True
        Hide reports of checks that are known to crash inside
        compliance-checker regardless of the file (see
        _KNOWN_CHECKER_BUGS). Genuine failures are always shown. Set
        False to see everything the checker reports.
    """
    check_suite = CheckSuite()
    check_suite.load_all_available_checkers()

    if not suppress_known_bugs:
        ComplianceChecker.run_checker(path, ["cf", "acdd"], 0, "normal")
        return

    # The checker reports crashed checks by printing to stderr, not via
    # the warnings machinery, so filtering means capturing the stream.
    captured = io.StringIO()
    try:
        with contextlib.redirect_stderr(captured):
            ComplianceChecker.run_checker(path, ["cf", "acdd"], 0, "normal")
    finally:
        # Re-emit whatever we captured, minus the known-bug lines --
        # including when the checker raised, so nothing is swallowed.
        remaining = _filter_checker_stderr(captured.getvalue(),
                                           _KNOWN_CHECKER_BUGS)
        if remaining.strip():
            sys.stderr.write(remaining)


# Checks that are known to raise inside compliance-checker itself, for
# reasons that have nothing to do with the file being checked. Each entry
# is a "<checker>.<check_name>" as it appears in the checker's own error
# report, with a note on why it is safe to hide.
#
# - cf.check_domain_variables: raises "list index out of range" whenever
#   featureType is set (6.1.0). The check is for CF 5.8 domain variables,
#   which are irrelevant to a moored time series, and featureType is
#   required by ACDD -- so the sensible response is to keep featureType
#   and ignore the noise. Verified on a minimal dataset carrying nothing
#   but featureType.
_KNOWN_CHECKER_BUGS = ("cf.check_domain_variables",)

_ERROR_HEADER = re.compile(
    r"^WARNING: The following exceptions occurred during the .* checker")


def _filter_checker_stderr(text: str, suppress: tuple) -> str:
    """
    Drop reports of known compliance-checker bugs from its stderr output.

    The checker prints a header line followed by one "<checker>.<check>:
    <message>" line per failed check. We remove the suppressed lines, and
    the header too if nothing is left under it -- so a genuine new failure
    still gets reported, header and all.
    """
    lines = text.splitlines()
    kept, block = [], []

    def flush():
        # A header with no surviving error lines under it is dropped.
        if len(block) > 1:
            kept.extend(block)
        block.clear()

    for line in lines:
        if _ERROR_HEADER.match(line):
            flush()
            block.append(line)
        elif block and any(line.startswith(name + ":") for name in suppress):
            continue  # a known bug -- drop it
        elif block and re.match(r"^\S+\.\S+: ", line):
            block.append(line)
        else:
            flush()
            kept.append(line)
    flush()

    return "\n".join(kept) + ("\n" if text.endswith("\n") and kept else "")


def _compliance_checks_ioos_plain(file):
    """
    Use the IOOS compliance checker
    (https://github.com/ioos/compliance-checker-web)
    to check an nc file (CF and ACDD conventions).

    Can take a file path or an xr.Dataset as input.
    """
    _require_compliance_checker()

    if isinstance(file, xr.Dataset):
        # The checker works on files, so write a temporary copy of the
        # dataset. Using a temporary directory means (a) the copy is
        # removed even if the check raises or the user interrupts it
        # mid-run, and (b) it never lands in the user's working
        # directory.
        with tempfile.TemporaryDirectory() as tempdir:
            temp_file = os.path.join(tempdir, "temp.nc")
            to_write, coord_encoding = netcdf.prepare_for_export(file)
            to_write.to_netcdf(temp_file, encoding=coord_encoding)
            _run_ioos_checkers(temp_file)
    else:
        _run_ioos_checkers(file)


def _compliance_checks_ioos_with_button(file):
    """
    (Wrapper for _compliance_checks_ioos_plain() with a "close" button)

    Use the IOOS compliance checker
    (https://github.com/ioos/compliance-checker-web)
    to check an nc file (CF and ACDD conventions).

    Can take a file path or an xr.Dataset as input.
    """
    _require_compliance_checker()

    output_widget = widgets.Output()

    def on_button_click(b):
        with output_widget:
            # Clear the output widget
            clear_output(wait=True)

        # Remove the button and the output widget
        close_button.close()
        output_widget.close()

    close_button = widgets.Button(description="Close")
    close_button.on_click(on_button_click)

    display(widgets.VBox([close_button, output_widget]))

    with output_widget:
        _compliance_checks_ioos_plain(file)


def compliance_checks_custom(ds: xr.Dataset) -> None:
    """
    Various ad-hoc checks for CF/ACDD compatibility and following good NPI practice. 
    Inspects an xarray.Dataset and prints a summary with red flags.

    NOTE: We should add new checks here as we think of useful ones.

    Checks:
        1. Variable data types — warns if any are int64 or float64.
        2. Attribute placeholders — warns if any global or variable attributes
           contain 'TBW'.
        3. Fill values — warns if any contain 'nn' or are missing _FillValue.
        4. Required attribute — 'processing_level' must exist either globally
           or on all relevant variables, but not both.
        5. Recommended attribute — 'QC_indicator' should exist globally or on
           all relevant variables, but not both.
        6. Variable metadata completeness — all relevant variables must have
           'units' and either 'standard_name' or 'long_name'.
        7. SBE_FLAG variable check — suggest removing if present.
        8. Coordinate checks — ensure coordinates are finite, monotonic,
           have coverage_content_type='coordinate', and recommended axis
           attributes; also ensure LATITUDE, LONGITUDE, STATION are coordinates.
        9. Global ACDD metadata presence and Conventions.
       10. Variable dimension consistency — all dims must be coordinates.
       11. FLAG variable structure — must have flag_meanings / flag_values.
    """
    try:
        check, warn, cross, arrow = "✅", "⚠️", "❌", "⮕"
    except Exception:
        check, warn, cross, arrow = "[OK]", "[!]", "[X]", "->"

    problems, passed, issues = [], [], 0
    skip_vars = {"STATION", "CRUISE", "DEPTH_INDEX", "NISKIN_NUMBER", "CAST"}
    vars_relevant = [v for v in ds.variables if v not in skip_vars]
    data_vars_relevant = [v for v in ds.data_vars if v not in skip_vars]

    # 1. dtype check
    bad_types = [v for v in ds.variables if ds[v].dtype in (np.int64, np.float64)]
    if bad_types:
        problems.append(
            f"{warn} 64-bit types (32-bit is recommended):\n{', '.join(bad_types)}\n   {arrow} "
            "Suggestion: use kval.conventionalize.convert_64_to_32(ds)"
        )
        issues += 1
    else:
        passed.append(f"{check} No 64-bit variable types")

    # 2. TBW attributes
    bad_g = [k for k, v in ds.attrs.items()
            if isinstance(v, str) and v.strip().upper() == "TBW"]

    bad_v = [f"{var}:{k}" for var in ds.variables
            for k, v in ds[var].attrs.items()
            if isinstance(v, str) and v.strip().upper() == "TBW"]
    if bad_g or bad_v:
        msg = f"{cross} Attributes with TBW placeholders:"
        if bad_g:
            msg += f"\n   – Global:\n    {', '.join(bad_g)}"
        if bad_v:
            msg += f"\n   – Variables:\n    {', '.join(bad_v)}"
        msg += f"\n   {arrow} Suggestion: Replace with actual metadata"
        problems.append(msg)
        issues += 1
    else:
        passed.append(f"{check} No obvious placeholder attributes")

    # 3. fill value check
    bad_fill = []
    for v in vars_relevant:
        fv = ds[v].attrs.get("_FillValue")
        if fv is None or (isinstance(fv, (float, np.floating)) and np.isnan(fv)):
            bad_fill.append(v)

    if bad_fill:
        problems.append(
            f"{warn} Suspicious/missing _FillValue (including NaNs, which are discouraged):\n{', '.join(bad_fill)}\n   "
            f"{arrow} Suggestion: use kval.conventionalize.nans_to_fill_value(ds)"
        )
        issues += 1
    else:
        passed.append(f"{check} _FillValue nicely defined for all relevant variables")

    # 4. processing_level
    g_proc = "processing_level" in ds.attrs
    v_proc = [v for v in data_vars_relevant if "processing_level" in ds[v].attrs]
    if g_proc and v_proc:
        problems.append(f"{cross} 'processing_level' exists globally and on vars")
        issues += 1
    elif not g_proc and len(v_proc) != len(data_vars_relevant):
        missing = [v for v in data_vars_relevant if "processing_level" not in ds[v].attrs]
        problems.append(
            f"{cross} Missing 'processing_level':\n{', '.join(missing)}\n   "
            f"{arrow} Suggestion: add globally or on all relevant variables"
        )
        issues += 1
    else:
        passed.append(f"{check} Required 'processing_level' present correctly")

    # 5. QC_indicator
    g_q = "QC_indicator" in ds.attrs
    v_q = [v for v in data_vars_relevant if "QC_indicator" in ds[v].attrs]
    if g_q and v_q:
        problems.append(f"{warn} 'QC_indicator' exists globally and on vars")
        issues += 1
    elif not g_q and len(v_q) != len(data_vars_relevant):
        missing = [v for v in data_vars_relevant if "QC_indicator" not in ds[v].attrs]
        problems.append(f"{warn} Missing 'QC_indicator' (not strictly required):\n{', '.join(missing)}")
        issues += 1
    else:
        passed.append(f"{check} Recommended 'QC_indicator' present correctly")

    # 6. variable metadata completeness
    vars_units_check = [v for v in vars_relevant if not v.endswith("_FLAG")]
    missing_units = [v for v in vars_units_check if "units" not in ds[v].attrs]
    missing_name = [v for v in vars_relevant
                    if not ("standard_name" in ds[v].attrs or
                            "long_name" in ds[v].attrs)]
    if missing_units or missing_name:
        msg = ""
        if missing_units:
            msg += f"{cross} Missing 'units':\n{', '.join(missing_units)}\n"
        if missing_name:
            msg += f"{cross} Missing 'standard_name'/'long_name':\n{', '.join(missing_name)}"
        msg += f"\n   {arrow} Suggestion: Add 'units' and 'standard_name'/'long_name'"
        problems.append(msg)
        issues += 1
    else:
        passed.append(f"{check} All relevant variables have 'units' and 'standard_name' or 'long_name' attributes")

    # 7. SBE_FLAG variable
    if "SBE_FLAG" in ds.variables:
        problems.append(f"{warn} 'SBE_FLAG' present\n   {arrow} Consider removing")
        issues += 1
    else:
        passed.append(f"{check} No 'SBE_FLAG' variable present")

    # 8. Coordinate checks
    coord_vars = list(ds.coords)
    if not coord_vars:
        problems.append(f"{cross} No coordinate variables found")
        issues += 1
    else:
        # (a) Check monotonicity and NaNs for *dimension* coords only
        bad_order, bad_nans = [], []
        for c in coord_vars:
            arr = ds[c].values
            if c in ds.dims and np.issubdtype(arr.dtype, np.number) and arr.ndim == 1:
                if np.any(np.isnan(arr)):
                    bad_nans.append(c)
                elif not (np.all(np.diff(arr) > 0) or np.all(np.diff(arr) < 0)):
                    bad_order.append(c)
        if bad_nans:
            problems.append(
                f"{cross} Dimension coordinate(s) contain NaNs: {', '.join(bad_nans)}\n   "
                f"{arrow} Suggestion: remove or interpolate missing coordinate values"
            )
            issues += 1
        elif bad_order:
            problems.append(
                f"{warn} Non-monotonic dimension coordinate(s): {', '.join(bad_order)}\n   "
                f"{arrow} Suggestion: ensure dimensional coordinates are uniformly increasing or decreasing"
            )
            issues += 1
        else:
            passed.append(f"{check} Dimension coordinates are finite and monotonic")

        # (b) coverage_content_type
        missing_cov = [
            c for c in coord_vars
            if c.lower() != "station"
            and ds[c].attrs.get("coverage_content_type", "").lower() != "coordinate"
        ]
        if missing_cov:
            problems.append(
                f"{warn} Missing or incorrect 'coverage_content_type' (should be 'coordinate'):\n"
                f"{', '.join(missing_cov)}"
            )
            issues += 1
        else:
            passed.append(f"{check} All coordinates (except STATION) have coverage_content_type='coordinate'")

        # (c) axis attribute presence
        missing_axis = [
            c for c in coord_vars
            if c.lower() != "station" and "axis" not in ds[c].attrs
        ]
        if missing_axis:
            problems.append(
                f"{warn} Missing recommended 'axis' attribute for coordinates:\n{', '.join(missing_axis)}"
            )
            issues += 1
        else:
            passed.append(f"{check} All coordinates (except STATION) have an 'axis' attribute")


        # (d) Ensure key spatial coords are marked as coordinates
        rec_coords = {"LATITUDE", "LONGITUDE", "STATION"}
        missing_from_coords = [v for v in rec_coords if v in ds.variables and v not in ds.coords]
        if missing_from_coords:
            problems.append(
                f"{warn} Variables you may want to set as (non-dimensional) coordinate variables but aren't:\n{', '.join(missing_from_coords)}\n   "
                f"{arrow} Suggestion: promote them to coordinates (ds = ds.set_coords([...]))"
            )
            issues += 1
        else:
            passed.append(f"{check} Spatial identifiers correctly treated as coordinates")

        # (e) Axis sanity check
        axis_expected = {
            "LATITUDE": "Y",
            "LONGITUDE": "X",
            "DEPTH": "Z",
            "TIME": "T"
        }
        axis_mismatch = []
        for c, expected in axis_expected.items():
            if c in ds.coords:
                found = ds[c].attrs.get("axis")
                if found and found.upper() != expected:
                    axis_mismatch.append(f"{c}: expected axis='{expected}', found '{found}'")
        if axis_mismatch:
            problems.append(f"{warn} Axis attribute mismatches:\n" + "\n".join(f"   – {i}" for i in axis_mismatch))
            issues += 1
        else:
            passed.append(f"{check} Coordinate axis attributes consistent with CF expectations")

    # 9. Global attributes check (required vs recommended)
    required_global_attrs = [
        "title",
        "summary",
        "Conventions",
        "creator_name",
        "creator_email",
        "institution",
        "keywords",
        "date_created",
        "featureType",
    ]

    recommended_global_attrs = [
        "product_version",
        "id",
        "license",
        "project",
        "doi",
        "acknowledgment",
        "references",
        "data_set_progress",
        "cruise_name",
        "ship",
        "area",
        "location",
        "source",
        "data_set_language",
        "keywords_vocabulary",
        "standard_name_vocabulary",
        "iso_topic_category",
        "platform",
        "platform_vocabulary",
        "data_assembly_center",
        "creator_type",
        "creator_url",
        "creator_institution",
        "publisher_name",
        "publisher_url",
        "publisher_email",
        "publisher_type",
        "publisher_institution",
        "instrument_vocabulary",
        "naming_authority"

     #   "processing_level"  # optional at global level if present on variables
    ]

    # Check required globals
    missing_required = [k for k in required_global_attrs if k not in ds.attrs]
    if missing_required:
        problems.insert(0,
            f"{cross}{cross}{cross} MISSING REQUIRED GLOBAL ATTRIBUTES {cross}{cross}{cross}\n"
            f"{', '.join(missing_required)}\n"
            f"{arrow} These MUST be added for CF/ACDD compliance"
        )
        issues += 1
    else:
        passed.append(f"{check} All required global attributes present")

    # Check recommended globals
    missing_recommended = [k for k in recommended_global_attrs if k not in ds.attrs]
    if missing_recommended:
        problems.append(
            f"{warn} Missing recommended global attributes:\n{', '.join(missing_recommended)}\n"
            f"{arrow} Consider adding them for better metadata completeness"
        )
    else:
        passed.append(f"{check} All recommended global attributes present")

    # Check that Conventions mentions CF/ACDD
    if "Conventions" in ds.attrs:
        conv = str(ds.attrs["Conventions"]).lower()
        if "cf" not in conv:
            problems.append(f"{warn} 'Conventions' does not mention CF")
            issues += 1
        elif "acdd" not in conv:
            problems.append(f"{warn} 'Conventions' does not mention ACDD")
            issues += 1
        else:
            passed.append(f"{check} 'Conventions' mentions both CF and ACDD")


    # 10. Dimension consistency
    dim_issues = []
    for v in vars_relevant:
        for d in ds[v].dims:
            if d not in ds.coords:
                dim_issues.append(f"{v}: dimension '{d}' not a coordinate variable")
    if dim_issues:
        problems.append(f"{warn} Variables reference non-coordinate dimensions:\n" +
                        "\n".join(f"   – {i}" for i in dim_issues))
        issues += 1
    else:
        passed.append(f"{check} All variable dimensions correspond to coordinate variables")

    # 11. FLAG variable structure
    flag_vars = [v for v in ds.variables if v.endswith("_FLAG")]
    flag_issues = []
    for v in flag_vars:
        attrs = ds[v].attrs
        if not all(k in attrs for k in ("flag_meanings", "flag_values")):
            flag_issues.append(v)
    if flag_issues:
        problems.append(f"{warn} FLAG variables missing 'flag_meanings' or 'flag_values':\n{', '.join(flag_issues)}")
        issues += 1
    else:
        passed.append(f"{check} All FLAG variables have CF-compliant attributes")

    # Summary
    sum_line = (f"{check} All checks passed ({issues} issues)" if issues == 0 else
                f"----------------------------------\n{warn} Dataset has {issues} issue(s)\n----------------------------------")
    print("\n" + sum_line + "\n")
    if problems:
        print("\n\n".join(problems) + "\n")
    if passed:
        print(f"----------------------------------\n{check} Passed checks\n----------------------------------\n")
        print("\n\n".join(passed) + "\n")