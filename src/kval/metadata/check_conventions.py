import os
import xarray as xr
from IPython.display import display, clear_output
import ipywidgets as widgets
import numpy as np


# Conditional import for compliance-checker
try:
    from compliance_checker.runner import ComplianceChecker, CheckSuite

    COMPLIANCE_CHECKER_AVAILABLE = True
except ImportError:
    COMPLIANCE_CHECKER_AVAILABLE = False


def check_file(file):
    """
    Use the IOOS compliance checker
    (https://github.com/ioos/compliance-checker-web)
    to check an nc file (CF and ACDD conventions).

    Can take a file path or an xr.Dataset as input
    """
    if not COMPLIANCE_CHECKER_AVAILABLE:
        raise ImportError(
            "IOOS Compliance Checker is not installed. "
            "Please install it to use this functionality, e.g.:\n"
            "$ conda install -c conda-forge compliance-checker\nor\n"
            "$ pip install compliance-checker\n\n"
            "(Note: There are some issues with getting the dependencies "
            " of this library to work on Py3.12, MacOS and Windows..)"
        )

    # Load all available checker classes
    temp = False
    if isinstance(file, xr.Dataset):
        # Store a temp copy for checking
        temp_file = "./temp.nc"
        file.to_netcdf(temp_file)
        file = temp_file
        temp = True

    check_suite = CheckSuite()
    check_suite.load_all_available_checkers()

    # Run cf and adcc checks
    path = file
    checker_names = ["cf", "acdd"]
    verbose = 0
    criteria = "normal"

    return_value, errors = ComplianceChecker.run_checker(
        path,
        checker_names,
        verbose,
        criteria,
    )

    if temp:
        os.remove(temp_file)


def check_file_with_button(file):
    """
    (Wrapper for check_file() with a "close" button)

    Use the IOOS compliance checker
    (https://github.com/ioos/compliance-checker-web)
    to check an nc file (CF and ACDD conventions).

    Can take a file path or an xr.Dataset as input
    """
    if not COMPLIANCE_CHECKER_AVAILABLE:
        raise ImportError(
            "IOOS Compliance Checker is not installed. "
            "Please install it to use this functionality, e.g.:\n"
            "$ conda install -c conda-forge compliance-checker\nor\n"
            "$ pip install compliance-checker\n\n"
            "(Note: There are some issues with getting the dependencies "
            " of this library to work on Py3.12, MacOS and Windows..)"
        )

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

    # Your existing code
    with output_widget:
        check_file(file)


def custom_checks(ds: xr.Dataset) -> None:
    """
    Various ad-hoc checks for CF/ACDD compatibility and following good NPI practice. 
    Inspects an xarray.Dataset and print a summary with red flags.

    NOTE: We should add new checks here as we think of useful ones.

    Checks:
        1. Variable data types — warns if any are int64 or float64.
        2. Attribute placeholders — warns if any global or variable attributes
           contain 'OPT' or 'TBW'.
        3. Fill values — warns if any contain 'nn' or are missing _FillValue.
        4. Required attribute — 'processing_level' must exist either globally
           or on all relevant variables, but not both.
        5. Recommended attribute — 'QC_indicator' should exist globally or on
           all relevant variables, but not both.
        6. Variable metadata completeness — all relevant variables must have
           'units' and either 'standard_name' or 'long_name'.
        7. SBE_FLAG variable check — suggest removing if present.
    """
    try:
        check, warn, cross, arrow = "✅","⚠️","❌","⮕"
    except Exception:
        check, warn, cross, arrow = "[OK]","[!]","[X]","->"

    warnings, passed, issues = [], [], 0
    skip_vars = {"STATION", "PROCESSING", "CRUISE", "DEPTH_INDEX", "NISKIN_NUMBER"}
    vars_relevant = [v for v in ds.variables if v not in skip_vars]

    # 1. dtype check
    bad_types = [v for v in ds.variables if ds[v].dtype in (np.int64, np.float64)]
    if bad_types:
        warnings.append(
            f"{warn} 64-bit types (32-bit is recommended):\n{', '.join(bad_types)}\n   {arrow} "
            "Suggestion: use kval.conventionalize.convert_64_to_32(ds)"
        )
        issues += 1
    else:
        passed.append(f"{check} No 64-bit variable types")

    # 2. OPT/TBW attributes
    bad_g = [k for k, v in ds.attrs.items()
             if isinstance(v, str) and v.strip().upper() in ("OPT","TBW",)]
    bad_v = [f"{v}:{k}" for v in ds.variables
             for k, v in ds[v].attrs.items()
             if isinstance(v, str) and v.strip().upper() in ("OPT","TBW")]
    if bad_g or bad_v:
        msg = f"{cross} Attributes with OPT/TBW placeholders:"
        if bad_g:
            msg += f"\n   – Global:\n    {', '.join(bad_g)}"
        if bad_v:
            msg += f"\n   – Variables:\n    {', '.join(bad_v)}"
        msg += f"\n   {arrow} Suggestion: Replace with actual metadata"
        warnings.append(msg)
        issues += 1
    else:
        passed.append(f"{check} No obvious placeholder attributes")

    # 3. fill value check
    bad_fill = [v for v in vars_relevant if ds[v].attrs.get("_FillValue") in (None, "nn")]
    if bad_fill:
        warnings.append(
            f"{warn} Suspicious/missing _FillValue (including NaNs, which are discouraged):\n{', '.join(bad_fill)}\n   "
            f"{arrow} Suggestion: use kval.conventionalize.nans_to_fill_value(ds)"
        )
        issues += 1
    else:
        passed.append(f"{check} _FillValue nicely defined for all relevant variables")

    # 4. processing_level
    g_proc = "processing_level" in ds.attrs
    v_proc = [v for v in vars_relevant if "processing_level" in ds[v].attrs]
    if g_proc and v_proc:
        warnings.append(f"{cross} 'processing_level' exists globally and on vars")
        issues += 1
    elif not g_proc and len(v_proc) != len(vars_relevant):
        missing = [v for v in vars_relevant if "processing_level" not in ds[v].attrs]
        warnings.append(
            f"{cross} Missing 'processing_level':\n{', '.join(missing)}\n   "
            f"{arrow} Suggestion: add globally or on all relevant variables"
        )
        issues += 1
    else:
        passed.append(f"{check} Required 'processing_level' present correctly")

    # 5. QC_indicator
    g_q = "QC_indicator" in ds.attrs
    v_q = [v for v in vars_relevant if "QC_indicator" in ds[v].attrs]
    if g_q and v_q:
        warnings.append(f"{warn} 'QC_indicator' exists globally and on vars")
        issues += 1
    elif not g_q and len(v_q) != len(vars_relevant):
        missing = [v for v in vars_relevant if "QC_indicator" not in ds[v].attrs]
        warnings.append(f"{warn} Missing 'QC_indicator':\n{', '.join(missing)}")
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
        warnings.append(msg)
        issues += 1
    else:
        passed.append(f"{check} All relevant variables have 'units' and 'standard_name' or 'long_name' attributes")

    # 7. SBE_FLAG variable
    if "SBE_FLAG" in ds.variables:
        warnings.append(f"{warn} 'SBE_FLAG' present\n   {arrow} Consider removing")
        issues += 1
    else:
        passed.append(f"{check} No 'SBE_FLAG' variable present")

    # Summary
    sum_line = (f"{check} All checks passed ({issues} issues)" if issues == 0 else
                f"----------------------------------\n{warn} Dataset has {issues} issue(s)\n----------------------------------")
    print("\n"+sum_line+"\n")
    if warnings:
        print("\n\n".join(warnings)+"\n")
    if passed:
        print(f"----------------------------------\n{check} Passed checks\n----------------------------------\n")
        print("\n\n".join(passed)+"\n")
