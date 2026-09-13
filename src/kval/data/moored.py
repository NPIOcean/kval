"""
KVAL.DATA.MOORED

Loading and processing data from fixed instruments.

Currently works for SBE and RBR CTD sensors - may want to broaden functionality
for other moored sensors.

- Loading data from .cnv or .rsk
- QC (!)
    - Compare with CTD (!)
    - Deck PRES values (!)
    - Quicklook functions (!)
- Editing
    - Chop deck time from the record
    - Despike*
    - Rolling filter*
    - Threshold editing*
    - Drift corr (!)
    - Remove points by index
    - Remove pick by hand picking
    - Drop variables
    - Drop variables (interactive)
- Calculations
    - Recalculate PSAL
    - Calculate depth (!)
    - Calculate all TEOS-10 (!)
- Standard metadata fixes*
- Saving
    - To matfile*

(!) To be written
* Simple wrappers

"""

import xarray as xr
import numpy as np
import pandas as pd

import os
import gsw
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib.dates import DateFormatter, HourLocator
import matplotlib as mpl

from scipy import signal

from kval.file import sbe, rbr, matfile
from kval.data import dataset, edit
from kval.data.moored_tools import _moored_tools

from kval.util import internals, index, time
from kval.signal import despike, filt
from kval.metadata import conventionalize
from kval.metadata.check_conventions import check_file_with_button
import warnings

# Want to be able to use these functions directly..
from kval.data.dataset import  to_netcdf, add_latlon
from kval.data.edit import threshold, offset, linear_drift
from kval.util.xr_funcs import time_average

if internals.is_notebook():
    from IPython.display import display


def load_moored(
    file: str,
    processing_variable=True,
    lat=None,
    lon=None,
) -> xr.Dataset:
    """
    Load moored instrument data from a file into an xarray Dataset, preserving
    metadata whevever possible.

    Should be able to read instruments from RBR (Concerto, Solo..) and
    SBE (SBE37, SBE16). Mileage may vary for older file types.

    Parameters:
    - file (str):
        Path to the file.
    - processing_variable (bool):
        Whether to add processing history to the dataset.

    Returns:
    - xr.Dataset:
        The loaded dataset.
    """
    # Check file type and return an error if invalid
    if file.endswith(".nc"):
        ds = load_nc(file)
        return ds
    elif file.endswith(".rsk"):
        instr_type = "RBR"
    elif file.endswith(".cnv"):
        instr_type = "SBE"
    elif file.endswith(".csv"):
        instr_type = "SBE_csv"
    elif file.endswith(".asc"):
        instr_type = "SBE_asc"

    else:
        raise ValueError(
            f"Unable to load moored instrument {os.path.basename(file)}.\n"
            "Supported files are .cnv (SBE) and .rsk (RBR) - will also try to"
            " read .csv and .asc as SBE files."
        )

    # Load data
    if instr_type == "RBR":
        ds = rbr.read_rsk(file)
    elif instr_type in ("SBE", "SBE_asc"):
        ds = sbe.read_cnv(file)
    elif instr_type in ("SBE_csv"):
        ds = sbe.read_csv(file)

    # Assign lat/lon if we have specified them
    if lat:
        ds["LATITUDE"] = ((), lat)
    if lon:
        ds["LONGITUDE"] = ((), lon)

    return ds


def load_nc(
    file: str,
    decode_cf = False
) -> xr.Dataset:
    '''
    Wrapper for loading nc file.
    '''
    ds = xr.open_dataset(file, decode_cf=decode_cf)

    return ds


def chop_deck(
    ds: xr.Dataset,
    variable: str = "PRES",
    sd_thr: float = 1.0,
    indices: tuple[int, int] | None = None,
    auto_accept: bool = False,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Chop away start and end parts of a time series (xarray Dataset).
    Default behaviour is to look for the indices to remove. Looks for
    the indices where `variable` (e.g. temperature or pressure) is a
    specified number of standard deviations away from the mean (the user can
    also specify the indices to cut).

    Typical application: Remove data from a mooring record during which the
    instrument was on deck or being lowered/raised through the water column.

    The function can automatically suggest chopping based on the standard
    deviation of the specified variable or allow the user to manually input
    the desired range for chopping. A plot is displayed showing the proposed
    chop unless `auto_accept` is set to True.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the data to be chopped. The dataset
        should include the field corresponding to the specified `variable`.
    variable : str, optional
        The variable in the dataset used for determining chop boundaries.
        Defaults to 'PRES'.
    sd_thr : float, optional
        The standard deviation threshold for determining the chop boundaries
        when `indices` is not provided. Defaults to 3.0.
    indices : Optional[Tuple[int, int]], optional
        A tuple specifying the (start, stop) indices for manually chopping the
        dataset along the TIME dimension. If not provided, the function will
        use the standard deviation threshold to determine the range
        automatically. Defaults to None.
    auto_accept : bool, optional
        If `True`, automatically accepts the suggested chop based on the
        pressure record without prompting the user. Defaults to False.

    Returns
    -------
    xr.Dataset
        The chopped xarray Dataset, with the range outside the specified or
        computed boundaries removed.

    Raises
    ------
    ValueError
        If the specified `variable` is not present in the dataset or if the
        user input during the chop confirmation is invalid.
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    # Confirm that `variable` exists in ds
    if variable not in ds:
        raise ValueError(
            f"Error: Cannot do chopping based on {variable} because it "
            "is not a variable in the datset."
        )

    if indices is None:
        # Calculate the mean and standard deviation
        chop_var = ds[variable].data
        chop_var_mean = np.ma.median(chop_var)
        chop_var_sd = np.ma.std(chop_var)

        indices = [None, None]

        # If we detect deck time at start of time series:
        # find a start index
        if chop_var[0] < chop_var_mean - sd_thr * chop_var_sd:
            indices[0] = (
                np.where(
                    np.diff(chop_var < chop_var_mean - sd_thr * chop_var_sd)
                )[0][0]
                + 1
            )
        # If we detect deck time at end of time series:
        # find an end index
        if chop_var[-1] < chop_var_mean - sd_thr * chop_var_sd:
            indices[1] = np.where(
                np.diff(chop_var < chop_var_mean - sd_thr * chop_var_sd)
            )[0][-1]

        # A slice defining the suggested "good" range
        keep_slice = slice(*indices)

        if auto_accept:
            accept = "y"
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            index = np.arange(len(chop_var))

            ylab = variable
            if hasattr(ds[variable], "units"):
                ylab = f"{ylab} [{ds[variable].units}]"

            ax.plot(index, chop_var, "k", label=variable)
            ax.plot(
                index[keep_slice],
                chop_var[keep_slice],
                "r",
                label="Chopped Range",
            )
            ax.set_xlabel("Index")
            ax.set_ylabel(ylab)
            ax.invert_yaxis()
            ax.set_title(
                f"Suggested chop: [{keep_slice.start}, "
                f"{keep_slice.stop}] (to red curve)."
            )
            ax.legend()

            # Ensure plot updates and displays (different within notebook with
            # widget backend..)
            if internals.is_notebook():
                if mpl.get_backend() != "tkagg":
                    display(fig)
                else:
                    plt.ion()
                    plt.show()
            else:
                plt.show(block=False)

            print(
                f"Suggested chop: [{keep_slice.start}, "
                f"{keep_slice.stop}] (to red curve)."
            )
            accept = input("Accept (y/n)?: ")

            # Close the plot after input tochop_var avoid re-display
            plt.close(fig)

        if accept.lower() == "n":
            print("Not accepted -> Not chopping anything now.")
            print("NOTE: run chop(ds, indices =[A, B]) to manually set chop.")
            return ds

        elif accept.lower() == "y":
            pass
        else:
            raise ValueError(
                f'I do not understand your input "{accept}"'
                '. Only "y" or "n" works. -> Exiting.'
            )
    else:
        keep_slice = slice(indices[0], indices[1] + 1)

    L0 = ds.sizes["TIME"]

    ds = ds.isel(TIME=keep_slice)

    L1 = ds.sizes["TIME"]
    net_str = (
        f"Chopped {L0 - L1} samples using -> {indices} "
        f"(total samples {L0} -> {L1})"
    )
    if verbose:
        print(f"Chopping to index: {indices}")
        print(net_str)


    return ds

def chop_by_time(
    ds: xr.Dataset,
    start_time: str | None = None,
    end_time: str | None = None,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Chop away parts of a time series (xarray Dataset) based on an optional
    start and end time. If neither is provided, no chopping is performed.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the data to be chopped. It must include
        a 'TIME' coordinate, which can be either CF-compliant or numerical.
    start_time : str | None, optional
        The starting time for chopping in the format 'YYYY-MM-DD' or
        'YYYY-MM-DD HH:MM'. If not provided, the dataset will not be chopped
        from the start (i.e., retains all earlier data).
    end_time : str | None, optional
        The ending time for chopping in the format 'YYYY-MM-DD' or
        'YYYY-MM-DD HH:MM'. If not provided, the dataset will retain all data
        beyond this point (i.e., not chopped at the end).
    verbose : bool, optional
        Whether to print detailed information about the chop process.
        Defaults to True.

    Returns
    -------
    xr.Dataset
        The chopped xarray Dataset. If no `start_time` or `end_time` is provided,
        the original dataset is returned.

    Raises
    ------
    ValueError
        If the `start_time` and `end_time` result in an invalid or empty slice.
        Also raised if the 'TIME' coordinate is missing.
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    # Ensure the 'TIME' coordinate exists
    if 'TIME' not in ds.coords:
        raise ValueError(f"Dataset does not contain a 'TIME' coordinate.")

    # Decode CF-compliant time if TIME is numerical
    if isinstance(ds.TIME.values[0], float):
        time_units = ds.TIME.units
        ds = xr.decode_cf(ds, decode_timedelta=True)
    else:
        time_units = None

    # Handle the time range
    if start_time is None and end_time is None:
        if verbose:
            print("No start or end time specified, returning the original dataset.")
        return ds

    # Define the slice based on the optional times
    time_slice = slice(start_time, end_time)
    ds_chopped = ds.sel(TIME=time_slice)

    # Error handling: check if the chop resulted in an empty dataset
    if ds_chopped.sizes["TIME"] == 0:
        raise ValueError(
            "Invalid time range selection. The resulting dataset has no samples."
        )

    # Get length before and after chopping
    L0 = ds.sizes["TIME"]
    L1 = ds_chopped.sizes["TIME"]

    if verbose:
        chop_info = (
            f"Chopped dataset to time range {start_time or 'start'} to "
            f"{end_time or 'end'}, reducing from {L0} to {L1} samples."
        )
        print(chop_info)


    # If initial TIME was numerical: Convert back to numerical format
    if time_units:
        time_attrs = ds_chopped['TIME'].attrs
        ds_chopped['TIME'] = date2num(ds_chopped['TIME'])
        ds_chopped['TIME'].attrs = {'units': 'Days since 1970-01-01 00:00:00'} | time_attrs
        if 'DAYS SINCE 1970-01-01' not in time_units.upper():
            print(f'NOTE: time units have changed from {time_units} '
                  f'to {ds_chopped["TIME"].units}')

    return ds_chopped


# Despike

def despike_rolling(
    ds: xr.Dataset,
    var_name: str,
    window_size: int,
    n_std: float,
    dim: str = "TIME",
    filter_type: str = "median",
    min_periods: int | None = None,
    plot: bool = False,
    verbose: bool = False,
) -> xr.Dataset:
    """
    Despike a variable in a dataset using a rolling mean or median filter.

    Outliers are identified as points deviating from the rolling mean/median by
    more than `n_std` standard deviations. Both the rolling statistic and 
    standard deviation are computed within a centered window along a specified 
    dimension.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the variable to despike.
    var_name : str
        Name of the variable to despike.
    window_size : int
        Size of the rolling window.
    n_std : float
        Number of standard deviations for thresholding outliers.
    dim : str, default='TIME'
        Dimension along which to compute rolling statistics.
    filter_type : str, default='median'
        Rolling filter type, either 'mean' or 'median'.
    min_periods : int or None, optional
        Minimum number of observations required in a window to compute a value.
        Default is None.
    plot : bool, default=False
        If True, plots the original and despiked data.
    verbose : bool, default=False
        If True, prints summary information about the despiking.

    Returns
    -------
    xr.Dataset
        Dataset with the despiked variable. Optionally, depending on internal
        flags, a mask of outliers may also be returned.

    Notes
    -----
    The function modifies the variable in place in a copy of the dataset,
    leaving the original dataset unchanged.
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    ds, is_outside_criterion = despike.despike_rolling(
        ds,
        var_name,
        window_size,
        n_std,
        dim,
        filter_type,
        min_periods,
        True,
        True,
        plot,
        verbose,
    )

    #n_removed = np.sum(is_outside_criterion).item()


    var_comment = (f"Despiking: Values exceeding the {window_size}-point rolling {filter_type} by more than {n_std} (rolling) standard deviations have been removed.")

    if 'comment' in ds[var_name].attrs:
        var_comment = ds[var_name].comment + '\n' + var_comment

    ds[var_name].attrs['comment'] = var_comment


    return ds

def adjust_time_for_drift(
    ds: xr.Dataset,
    seconds: float = 0,
    minutes: float = 0,
    hours: float = 0,
    days: float = 0,
    start_time: str = None,
    end_time: str = None,
) -> xr.Dataset:
    """
    Adjust the TIME coordinate of an xarray Dataset to correct for instrument clock drift.

    Applies a linear drift correction in time: zero correction at start_time,
    ramping linearly (in elapsed time, not sample index) to the full specified
    offset at end_time. This is robust to gaps or uneven sampling intervals.

    By default, start_time and end_time are the first and last TIME values in
    the dataset (i.e., the drift ramps from 0 at deployment to the full offset
    at recovery). If explicitly specified, TIME values outside the
    [start_time, end_time] window are extrapolated linearly at the same drift
    rate, rather than clamped.

    The offset can be specified in seconds, minutes, hours, or days.
    Negative drift values indicate the instrument lags true time (offset is added),
    positive values indicate the instrument leads true time (offset is subtracted).

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with a TIME coordinate. TIME must be numeric (not
        datetime64) with a "days since..." units attribute, and must be
        sorted in non-decreasing order.
    seconds : float, default=0
        Clock drift in seconds.
    minutes : float, default=0
        Clock drift in minutes.
    hours : float, default=0
        Clock drift in hours.
    days : float, default=0
        Clock drift in days.
    start_time : str, optional
        Timestamp at which the drift correction is zero, e.g. '2020-01-02 00:33'.
        Defaults to the first TIME value if not specified.
    end_time : str, optional
        Timestamp at which the drift correction equals the full specified
        offset, e.g. '2020-01-15 08:00'. Defaults to the last TIME value if
        not specified.

    Returns
    -------
    xr.Dataset
        A new dataset with the adjusted TIME coordinate.
    """

    ds = ds.copy(deep=True)  # Make sure we're not modifying the input ds

    # Convert all drift values to seconds
    total_drift_seconds = (
        seconds +
        (minutes * 60) +
        (hours * 3600) +
        (days * 86400)  # 86400 seconds in a day
    )

    if total_drift_seconds > 0:
        drift_operation = 'subtracted'
    elif total_drift_seconds < 0:
        drift_operation = 'added'
    elif total_drift_seconds == 0:
        warnings.warn('To adjust for clock drift, a non-zero clock drift has'
                      ' to be specified -> Doing nothing', UserWarning)
        return ds

    # Check TIME units before touching values
    if 'units' not in ds.TIME.attrs:
        raise Exception('Could not add drift because TIME has no "units" '
                        'attribute (expected numerical "Days since..")')
    units_str = ds.TIME.attrs['units']
    if 'DAYS SINCE' not in units_str.upper():
        raise Exception('Could not add drift because TIME is non-numerical'
                        ' or has unknown units (should be "Days since..")')

    # Get the TIME coordinate as float; fail loudly if that's not possible
    try:
        time = ds.coords['TIME'].values.astype(float)
    except (TypeError, ValueError) as e:
        raise Exception('Could not add drift because TIME values could not '
                        f'be cast to float: {e}')

    # Nothing to do (and time[-1]/time[0] below would raise) on empty TIME
    if len(time) == 0:
        warnings.warn('TIME coordinate is empty -> Doing nothing', UserWarning)
        return ds

    # Drift correction assumes TIME is non-decreasing from deployment
    # (index 0) to recovery (index -1). Duplicate timestamps are allowed;
    # reversed/out-of-order TIME is not.
    if not np.all(np.diff(time) >= 0):
        raise Exception('Could not add drift because TIME is not sorted in '
                        'non-decreasing order')

    # Resolve the reference date from the "days since <ref>" units string,
    # needed to convert start_time/end_time strings into the same numeric
    # scale as the TIME coordinate
    ref_date_str = units_str.upper().split('DAYS SINCE')[-1].strip()
    try:
        ref_date = pd.Timestamp(ref_date_str)
    except (ValueError, TypeError) as e:
        raise Exception('Could not parse reference date from TIME units '
                        f'"{units_str}": {e}')

    def _time_str_to_num(time_str, label):
        try:
            dt = pd.Timestamp(time_str)
        except (ValueError, TypeError) as e:
            raise Exception(f'Could not parse {label}="{time_str}" as a '
                            f'timestamp (expected e.g. "2020-01-02 00:33"): {e}')
        return (dt - ref_date).total_seconds() / 86400  # convert to days

    start_num = _time_str_to_num(start_time, 'start_time') if start_time is not None else time[0]
    end_num = _time_str_to_num(end_time, 'end_time') if end_time is not None else time[-1]

    # Need a nonzero span to define a fractional position within it
    anchor_span = end_num - start_num
    if anchor_span == 0:
        raise Exception('Could not add drift because start_time and end_time '
                        '(or first/last TIME values) are identical')

    # Fractional position of each TIME point relative to [start_num, end_num],
    # linear in elapsed time (not sample index). Points outside this window
    # (if start_time/end_time were set explicitly) extrapolate linearly.
    frac_elapsed = (time - start_num) / anchor_span
    drift_adjustments_sec = frac_elapsed * total_drift_seconds
    adjusted_time = time - drift_adjustments_sec / 86400

    # Update the TIME coordinate in the dataset
    time_attrs = ds['TIME'].attrs
    drift_comment = (
        f'Adjusted for observed clock drift ({drift_operation} '
        f'from 0 to {abs(total_drift_seconds)} sec)')
    if 'comment' in time_attrs and time_attrs['comment']:
        time_attrs['comment'] = time_attrs['comment'] + '; ' + drift_comment
    else:
        time_attrs['comment'] = drift_comment
    ds['TIME'] = ('TIME', adjusted_time, time_attrs)

    return ds


# Filtering
def rolling_mean(
    ds: xr.Dataset,
    var_name: str,
    window_size: int,
    filter_type: str = "mean",
    dim: str = "TIME",
    min_periods: int | bool | None = None,
    nan_edges: bool = True,
) -> xr.Dataset:
    """
    Apply a rolling mean, median, or standard deviation filter to a variable
    in an xarray Dataset along a specified dimension.

    Edge handling and minimum valid observations can be controlled.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the variable to filter.
    var_name : str
        Name of the variable to apply the filter on.
    window_size : int
        Size of the rolling window.
    filter_type : str, optional
        Filter type: 'mean', 'median', or 'sd' (standard deviation). Default is 'mean'.
    dim : str, optional
        Dimension along which to apply the rolling filter. Default is 'TIME'.
    min_periods : int, bool, or None, optional
        Minimum number of observations in the window required to compute a value.
        If None, windows containing NaN values are set to NaN. Default is None.
    nan_edges : bool, optional
        If True, sets edge values (half the window length) to NaN. Default is True.

    Returns
    -------
    xr.Dataset
        The dataset with the filtered variable. Edge values may be NaN if `nan_edges` is True.
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    ds = filt.rolling(
        ds=ds,
        var_name=var_name,
        window_size=window_size,
        filter_type=filter_type,
        dim=dim,
        min_periods=min_periods,
        nan_edges=nan_edges,
    )

    var_comment = (f"A {window_size}-point rolling {filter_type} has been applied.")

    if 'comment' in ds[var_name].attrs:
        var_comment = ds[var_name].comment + '\n' + var_comment

    ds[var_name].attrs['comment'] = var_comment

    return ds


# Drift




# Interactive threshold edit
def threshold_pick(ds: xr.Dataset) -> xr.Dataset:
    """
    Interactively select a valid range for data variables and apply thresholds
    to the data.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to modify.

    Returns
    -------
    xr.Dataset
        The dataset with thresholds applied.

    Notes
    -----
    Utilizes interactive widgets for selecting thresholds within a Jupyter
    environment.
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    data_variables = []

    for varnm in ds.data_vars:
        if "TIME" in ds[varnm].dims:
            data_variables += [varnm]

    edit.threshold_edit(ds, variables=data_variables)
    return ds


# Remove points by index
def remove_points(
    ds: xr.Dataset, varnm: str, remove_inds, time_var="TIME",
    deep_copy = True,
) -> xr.Dataset:
    """
    Remove specified points from a time series in the dataset by setting them
    to NaN.

    Parameters:
    - ds: xarray.Dataset
      The dataset containing the variable to modify.
    - varnm: str
      The name of the variable to modify.
    - remove_inds: list or array-like
      Indices of points to remove (set to NaN).

    Returns:
    - ds: xarray.Dataset
      The dataset with specified points removed (set to NaN).
    """
    if deep_copy:
        ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    ds = edit.remove_points_timeseries(
        ds=ds, varnm=varnm, remove_inds=remove_inds, time_var=time_var,
        deep_copy = deep_copy
    )

    return ds


# Remove points (hand pick)
def hand_remove_points(
    ds: xr.Dataset,
    variable: str,
    variable_edit: str | None = None,
) -> xr.Dataset:
    """
    Interactively remove data points from CTD profiles.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the CTD data.
    variable : str
        The name of the variable to visualize (e.g., 'TEMP1', 'CHLA').
    variable_edit : str or None, optional
        The name of the variable to edit (e.g., 'TEMP1', 'CHLA').
        If None, the same variable as `variable` will be edited. Default is None.

    Returns
    -------
    xr.Dataset
        The dataset with points removed based on interactive input.

    Examples
    --------
    >>> ds = hand_remove_points(ds, 'TEMP1', 'StationA')

    Notes
    -----
    Use the interactive plot to select points for removal, then click the
    corresponding buttons for actions.

    Unlike most kval functions, this modifies `ds` in place (in addition
    to returning it), since the interactive editing session needs to act
    on the same object the user is clicking on.
    """

    # Note: deliberately *not* doing a reep copy here-
    # we to modify it in place for this to work.
    # ds = ds.copy(deep=True) 

    if not variable_edit:
        variable_edit = variable

    hand_remove = _moored_tools.hand_remove_points(
        ds, variable, varnm_edit=variable_edit)
    ds = hand_remove.ds

    return ds


# Recalculate sal
def calculate_PSAL(
    ds: xr.Dataset,
    cndc_var: str = "CNDC",
    temp_var: str = "TEMP",
    pres_var: str = "PRES",
    psal_var: str = "PSAL",
    retain_nans: bool = True,
) -> xr.Dataset:
    """(Re)calculate Practical Salinity (PSAL) from conductivity, temperature,
    and pressure using the GSW-Python module
    (https://teos-10.github.io/GSW-Python/).

    This function updates the PSAL variable in the dataset with newly computed
    salinity values while preserving the metadata attributes of PSAL.

    Args:
        ds (xr.Dataset):
            The input dataset containing conductivity, temperature, and
            pressure variables.
        cndc_var (str):
            The name of the conductivity variable in the dataset.
            Defaults to 'CNDC'.
        temp_var (str):
            The name of the temperature variable in the dataset.
            Defaults to 'TEMP'.
        pres_var (str):
            The name of the pressure variable in the dataset.
            Defaults to 'PRES'.
        psal_var (str):
            The name of the salinity variable in the dataset.
            Defaults to 'PSAL'.
        retain_nans (bool):
            If there is already a PSAL field: Retain the Nan values from
            old to new PSAL field.
            Defaults to True.

    Returns:
        xr.Dataset: The updated dataset with recalculated PSAL values.

    Notes:
        The operation preserves PSAL metadata attributes. If the input sensors
        change (e.g., if a different temperature sensor is used), the PSAL
        metadata attributes should be updated accordingly.
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    CNDC_ = ds[cndc_var].copy()
    if 'units' in CNDC_.attrs:
        if CNDC_.units == 'S m-1':
            print('Detected S m-1 unit - applying an x10 factor to CNDC.')
            CNDC_.values *= 10



    # Calculate PSAL
    PSAL = gsw.SP_from_C(
        CNDC_.values, ds[temp_var].values, ds[pres_var].values)

    # Retain NaNs if applicable
    if retain_nans and psal_var in ds:
       # PSAL = PSAL.where(~np.isnan(ds[psal_var]), np.nan)
        PSAL = np.where(np.isnan(ds[psal_var]), np.nan, PSAL)
    if psal_var in ds:
        ds[psal_var][:] = PSAL
    else:
        ds[psal_var] = (ds[cndc_var].dims, PSAL.data,
                        {'units': '1',})
        if ('sensor_calibration_date' in ds[temp_var].attrs
           and 'sensor_calibration_date' in ds[cndc_var].attrs):
            ds[psal_var].attrs['sensor_calibration_date'] = (
                f'{ds[temp_var].sensor_calibration_date} (TEMP), '
                f'{ds[cndc_var].sensor_calibration_date} (CNDC)')

    ds[psal_var].attrs["note"] = (
        f"Computed from {cndc_var}, {temp_var}, {pres_var} "
        "using the Python gsw module."
    )

    return ds


# Recalculate SA & CT
def calculate_SA_CT(
    ds: xr.Dataset,
    cndc_var: str = "CNDC",
    temp_var: str = "TEMP",
    pres_var: str = "PRES",
    psal_var: str = "PSAL",
) -> xr.Dataset:
    """Recalculate absolute salinity (SA) and coneservative temperature (CT)
    from conductivity, temperature, and pressure using the GSW-Python module
    (https://teos-10.github.io/GSW-Python/).

    This function adds or updates the SA and CT variables in the dataset with
    newly computed values.

    Args:
        ds (xr.Dataset):
            The input dataset containing conductivity, temperature, and
            pressure variables.
        cndc_var (str):
            The name of the conductivity variable in the dataset.
            Defaults to 'CNDC'.
        temp_var (str):
            The name of the temperature variable in the dataset.
            Defaults to 'TEMP'.
        pres_var (str):
            The name of the pressure variable in the dataset.
            Defaults to 'PRES'.
        psal_var (str):
            The name of the salinity variable in the dataset.
            Defaults to 'PSAL'.

    Returns:
        xr.Dataset: The updated dataset with SA, CT values.

    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    # Calculate absolute salinity
    SA = gsw.SA_from_SP(ds[psal_var], ds[pres_var], ds.LONGITUDE, ds.LATITUDE)
    # Calculate conservative temperature
    CT = gsw.CT_from_t(SA, ds[temp_var], ds[pres_var])

    ds['SA'] = (ds[psal_var].dims, SA.values,
                {'units': 'g kg-1',
                 'standard_name': 'sea_water_absolute_salinity',
                 'long_name': 'Absolute Salinity'})

    ds['CT'] = (ds[psal_var].dims, CT.values,
                {'units': 'degree_C',
                 'standard_name':'sea_water_conservative_temperature',
                 'long_name': 'Conservative Temperature'})

    for varname in ['CT', 'SA']:
        ds[varname].attrs["note"] = (
            f"Computed from {cndc_var}, {temp_var}, {pres_var} "
            "using the Python gsw module."
        )

    return ds



# Recalculate RHO
def calculate_rho(
    ds: xr.Dataset,
    cndc_var: str = "CNDC",
    temp_var: str = "TEMP",
    pres_var: str = "PRES",
    psal_var: str = "PSAL",
) -> xr.Dataset:
    """Recalculate Density (RHO) from conductivity, temperature,
    and pressure using the GSW-Python module
    (https://teos-10.github.io/GSW-Python/).

    This function adds or updates the RHO variable in the dataset with newly
    computed density value.

    Args:
        ds (xr.Dataset):
            The input dataset containing conductivity, temperature, and
            pressure variables.
        cndc_var (str):
            The name of the conductivity variable in the dataset.
            Defaults to 'CNDC'.
        temp_var (str):
            The name of the temperature variable in the dataset.
            Defaults to 'TEMP'.
        pres_var (str):
            The name of the pressure variable in the dataset.
            Defaults to 'PRES'.
        psal_var (str):
            The name of the salinity variable in the dataset.
            Defaults to 'PSAL'.

    Returns:
        xr.Dataset: The updated dataset with RHO values.

    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    # Calculate absolute salinity
    SA = gsw.SA_from_SP(ds[psal_var], ds[pres_var], ds.LONGITUDE, ds.LATITUDE)
    # Calculate conservative temperature
    CT = gsw.CT_from_t(SA, ds[temp_var], ds[pres_var])
    # Calculate density
    RHO = gsw.rho(SA, CT, ds[pres_var])

    ds['RHO'] = (ds[psal_var].dims, RHO.values,
                 {'units': 'kg m-3', 'standard_name':'sea_water_density',
                  'long_name' : 'In-situ seawater density'})


    ds['RHO'].attrs["note"] = (
        f"Computed from {cndc_var}, {temp_var}, {pres_var} "
        "using the Python gsw module."
    )


    return ds


# Recalculate sigma0
def calculate_sig0(
    ds: xr.Dataset,
    temp_var: str = "TEMP",
    pres_var: str = "PRES",
    psal_var: str = "PSAL",
) -> xr.Dataset:
    """Recalculate potential density anomaly (SIG0) from
    conductivity, temperature, and pressure using the GSW-Python module
    (https://teos-10.github.io/GSW-Python/).

    This function adds or updates the SIG0 variable in the dataset with newly
    computed density value.

    Args:
        ds (xr.Dataset):
            The input dataset containing conductivity, temperature, and
            pressure variables.
        temp_var (str):
            The name of the temperature variable in the dataset.
            Defaults to 'TEMP'.
        pres_var (str):
            The name of the pressure variable in the dataset.
            Defaults to 'PRES'.
        psal_var (str):
            The name of the salinity variable in the dataset.
            Defaults to 'PSAL'.

    Returns:
        xr.Dataset: The updated dataset with SIG0 values.

    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    # Calculate absolute salinity
    SA = gsw.SA_from_SP(ds[psal_var], ds[pres_var], ds.LONGITUDE, ds.LATITUDE)
    # Calculate conservative temperature
    CT = gsw.CT_from_t(SA, ds[temp_var], ds[pres_var])
    # Calculate sigma 0
    SIG0 = gsw.sigma0(SA, CT)

    ds['SIG0'] = (ds[psal_var].dims, SIG0.values,
                 {'units': 'kg m-3', 'standard_name': 'sea_water_sigma_theta',
                  'long_name': ('Potential density of water '
                                'minus 1000 kg m-3.')})

    ds['SIG0'].attrs["note"] = (
        f"Computed from {temp_var}, {pres_var} "
        "using the Python gsw module."
    )
    return ds


# Recalculate cndc
def calculate_CNDC(
    ds: xr.Dataset,
    cndc_var: str = "CNDC",
    temp_var: str = "TEMP",
    pres_var: str = "PRES",
    psal_var: str = "PSAL",
    retain_nans: bool = True,
) -> xr.Dataset:
    """(Re)calculate Conductivity (CNDC) from practical salinity, temperature,
    and pressure using the GSW-Python module
    (https://teos-10.github.io/GSW-Python/).

    This function updates the CNDC variable in the dataset with newly computed
    conductivity values while preserving the metadata attributes of CNDC.

    Args:
        ds (xr.Dataset):
            The input dataset containing salinity, temperature, and
            pressure variables.
        psal_var (str):
            The name of the salinity variable in the dataset.
            Defaults to 'PSAL'.
        temp_var (str):
            The name of the temperature variable in the dataset.
            Defaults to 'TEMP'.
        pres_var (str):
            The name of the pressure variable in the dataset.
            Defaults to 'PRES'.
        cndc_var (str):
            The name of the conductivity variable in the dataset.
            Defaults to 'CNDC'.
        retain_nans (bool):
            If there is already a CNDC field: Retain the NaN values from
            old to new CNDC field.
            Defaults to True.

    Returns:
        xr.Dataset: The updated dataset with recalculated CNDC values.

    Notes:
        The operation preserves CNDC metadata attributes. If the input sensors
        change (e.g., if a different temperature sensor is used), the CNDC
        metadata attributes should be updated accordingly.
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    # Calculate CNDC (in mS/cm)
    CNDC = gsw.C_from_SP(
        ds[psal_var].values, ds[temp_var].values, ds[pres_var].values
    )

    # Retain NaNs if applicable
    if retain_nans and cndc_var in ds:
        CNDC = np.where(np.isnan(ds[cndc_var]), np.nan, CNDC)

    # Overwrite or create CNDC variable
    if cndc_var in ds:
        ds[cndc_var][:] = CNDC
    else:
        ds[cndc_var] = (ds[psal_var].dims, CNDC.data, {'units': 'mS/cm'})
        if ('sensor_calibration_date' in ds[temp_var].attrs
           and 'sensor_calibration_date' in ds[psal_var].attrs):
            ds[cndc_var].attrs['sensor_calibration_date'] = (
                f'{ds[temp_var].sensor_calibration_date} (TEMP), '
                f'{ds[psal_var].sensor_calibration_date} (PSAL)'
            )

    ds[cndc_var].attrs["note"] = (
        f"Computed from {psal_var}, {temp_var}, {pres_var} "
        "using the Python gsw module."
    )

    return ds



# Assign pressure from adjacent instruments
def assign_pressure(
    ds_main: xr.Dataset,
    ds_above: xr.Dataset,
    ds_below: xr.Dataset,
    nom_dep_main: float,
    nom_dep_above: float,
    nom_dep_below: float,
    auto_accept: bool = False,
    plot: bool = True,
    lat: float = None,
    return_fig: bool = False,
) -> xr.Dataset:
    """
    Estimate and assign sea pressure to an instrument without a pressure record
    by interpolating between pressure sensors located above and below the
    instrument.

    This method is useful for instruments like an RBR Solo (temperature-only)
    located between two instruments (e.g., RBR Concertos with pressure sensors)
    on a mooring. The function interpolates pressure from the adjacent sensors
    and can display a plot comparing estimated and nominal pressures.

    Parameters
    ----------
    ds_main : xarray.Dataset
        Dataset for the instrument without pressure record, which will receive
        the estimated pressure.
    ds_above : xarray.Dataset
        Dataset for the instrument with pressure sensor located above the main
        instrument.
    ds_below : xarray.Dataset
        Dataset for the instrument with pressure sensor located below the main
        instrument.
    nom_dep_main : float
        Nominal (planned) depth of the main instrument [meters].
    nom_dep_above : float
        Nominal depth of the above sensor [meters].
    nom_dep_below : float
        Nominal depth of the below sensor [meters].
    auto_accept : bool, optional
        Automatically accept the pressure estimate without user confirmation.
        Default is False.
    plot : bool, optional
        Display a plot comparing the interpolated pressure to the recorded
        pressures of the adjacent sensors. Default is True.
    lat : float, optional
        Latitude for converting depth to pressure. If not provided, it will be
        inferred from `ds_main`.
    return_fig : bool, optional
        Return the figure object with the plot
    Returns
    -------
    xarray.Dataset
        Updated dataset `ds_main` with an added 'PRES' variable containing the
        estimated pressures [dbar].

    (if return_fig is True);
    (xarray.Dataset, matplotlib.figure.Figure)
        Tuple containing
            1. Updated dataset
            2. Matplotlib figure showing the adjustment

    Raises
    ------
    Exception
        If latitude (`lat`) is not provided and cannot be inferred from
        `ds_main`.
    """

    ds_main = ds_main.copy(deep=True) # Make sure we're not modifying the input ds
    ds_above = ds_above.copy(deep=True) 
    ds_below = ds_below.copy(deep=True) 

    # Ensure we have latitude for depth-to-pressure conversion
    if lat is None:
        try:
            lat = ds_main.LATITUDE.item()
        except AttributeError:
            raise Exception(
                "Could not find latitude for depth->pressure calculation. "
                "Specify `lat` in `assign_pressure`."
            )

    # return_fig=True not allowed if plot=False
    if return_fig and not plot:

        raise Exception(
            "Cannot return the figure (`return_fig = True`) if we are  "
            "not creating a plot (`plot = False`).."
            )

    # Convert nominal depth to nominal pressure
    nom_pres_main = gsw.p_from_z(-nom_dep_main, lat=lat)
    nom_pres_above = gsw.p_from_z(-nom_dep_above, lat=lat)
    nom_pres_below = gsw.p_from_z(-nom_dep_below, lat=lat)

    # Interpolate pressure records of above/below sensors onto main sensor's
    # time grid
    pres_above = ds_above.interp_like(ds_main).PRES
    pres_below = ds_below.interp_like(ds_main).PRES

    # Calculate interpolation weights based on nominal depths
    above_weight = (nom_pres_below - nom_pres_main) / (
        nom_pres_below - nom_pres_above
    )
    below_weight = (nom_pres_main - nom_pres_above) / (
        nom_pres_below - nom_pres_above
    )

    # Calculate interpolated pressure for main sensor
    pres_main = pres_above * above_weight + pres_below * below_weight
    pres_main_median = np.nanmedian(pres_main)
    dep_main_median = -gsw.z_from_p(pres_main_median, lat=lat)

    # Set default instrument and serial number if they don't exist
    instr_main = getattr(ds_main, "instrument", "Main instrument")
    serial_main = getattr(ds_main, "instrument_serial_number",
                          "Unknown serial")

    if plot:

        # For figure legends: Set above/below instrument and serial number if
        # they don't exist
        instr_above = getattr(ds_above, "instrument", "Above instrument")
        serial_above = getattr(
            ds_above, "instrument_serial_number", "Unknown serial"
        )
        instr_below = getattr(ds_below, "instrument", "Below instrument")
        serial_below = getattr(
            ds_below, "instrument_serial_number", "Unknown serial"
        )

        fig, ax = plt.subplots()
        ax.plot(
            ds_above.TIME, ds_above.PRES, label=f"{instr_above} {serial_above}"
        )
        ax.plot(
            ds_below.TIME, ds_below.PRES, label=f"{instr_below} {serial_below}"
        )
        ax.plot(
            ds_main.TIME,
            pres_main,
            label=f"**Estimate**: {instr_main} {serial_main}\n"
            f"(Median: {pres_main_median:.1f} dbar / {dep_main_median:.1f} m)",
        )
        hline_args = {"ls": "--", "color": "k", "zorder": 0, "lw": 0.7}
        ax.axhline(nom_pres_main, **hline_args, label="Nominal pressures")
        ax.axhline(nom_pres_above, **hline_args)
        ax.axhline(nom_pres_below, **hline_args)
        ax.invert_yaxis()
        ax.set_ylabel("Pressure [dbar]")
        ax.legend(fontsize=8)

        if internals.is_notebook() and mpl.get_backend() != "tkagg":
            display(fig)
        else:
            plt.show()

    if not auto_accept:
        accept = input(
            f"Estimated offset: {pres_main_median - nom_pres_main:.2f} dbar. "
            f"Assign to {instr_main} {serial_main}? (y/n):"
        )
        if plot:
            plt.close(fig)
        if accept.lower() not in ["y", "yes"]:
            print("No -> `Not` assigning pressure to the dataset.")

            if return_fig:
                return ds_main, fig
            else:
                return ds_main
    # Assign interpolated pressure to main dataset
    ds_main["PRES"] = (
        ("TIME"),
        pres_main.data,
        {
            "units": "dbar",
            "long_name": "Sea pressure (estimate from interpolation)",
            "processing_level": "Data interpolated",
            "coverage_content_type": "referenceInformation",
            "comment": ("Estimated by interpolating between adjacent"
                        " instruments with pressure sensors."),
        },
    )

    if return_fig:
        return ds_main, fig
    else:
        return ds_main


# Recalculate sal
def linear_drift_offset(
    ds: xr.Dataset,
    variable: str,
    end_val: float,
    start_val: float = 0,
    start_date: str | None = None,
    end_date: str | None = None
) -> xr.Dataset:
    """
    Apply a linearly increasing drift offset to a variable in an xarray Dataset.

    This function adds a linearly increasing additive offset over time to the
    specified variable. The drift is applied between `start_date` and `end_date`
    if provided, or over the entire time range of the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray Dataset containing the time series data.
    variable : str
        The name of the variable in the dataset to which the drift will be applied.
    end_val : float
        The value of the drift offset at the end of the period.
    start_val : float, optional
        The starting value of the drift offset. Default is 0.
    start_date : str or None, optional
        Start date in 'YYYY-MM-DD' format. If None, uses the first time value.
    end_date : str or None, optional
        End date in 'YYYY-MM-DD' format. If None, uses the last time value.

    Returns
    -------
    xr.Dataset
        A new dataset with the drift offset applied to the specified variable.

    Notes
    -----
    This is a wrapper for `kval.data.edit.linear_drift`.
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    ds = edit.linear_drift(
        ds, variable, end_val, start_val=start_val, start_date=start_date,
        end_date=end_date, factor=False)


    # Record to PSAL metadata field
    if start_date is None:
        start_date = 'the first data entry'
    if end_date is None:
        end_date = 'the last data entry'

    drift_comment = (f'Adjusted for drift by applying am *offset* linearly '
                     f'evolving from {start_val} on {start_date} to {end_val}'
                     ' on {end_date}.')

    if 'comment' in ds[variable].attrs:
        ds[variable].attrs['comment' ] += '\n' + drift_comment
    else:
        ds[variable].attrs['comment' ] = drift_comment

    return ds


# Recalculate sal
def linear_drift_factor(
    ds: xr.Dataset,
    variable: str,
    end_val: float,
    start_val: float = 1,
    start_date: str | None = None,
    end_date: str | None = None
) -> xr.Dataset:
    """
    Apply a linearly increasing drift factor to a variable in an xarray Dataset.

    This function applies a linearly increasing multiplicative drift over time
    to a specified variable. The drift is applied between `start_date` and
    `end_date` if provided, or over the entire time range of the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray Dataset containing the time series data.
    variable : str
        The name of the variable to which the drift will be applied.
    end_val : float
        The value of the drift factor at the end of the period.
    start_val : float, optional
        The starting value of the drift factor. Default is 1.
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format. If None, uses the first time value.
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, uses the last time value.

    Returns
    -------
    xr.Dataset
        A new dataset with the drift applied to the specified variable.

    Notes
    -----
    This is a wrapper for `kval.data.edit.linear_drift`.
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    # Apply drift
    ds = edit.linear_drift(
        ds, variable, end_val, start_val=start_val, start_date=start_date,
        end_date=end_date, factor=True)

    # Record to PSAL metadata field
    if start_date is None:
        start_date = 'the first data entry'
    if end_date is None:
        end_date = 'the last data entry'

    drift_comment = (f'Adjusted for drift by applying a *factor*'
                     f' linearly evolving from {start_val} on'
                    f' {start_date} to {end_val} on {end_date}.')

    if 'comment' in ds[variable].attrs:
        ds[variable].attrs['comment' ] += '\n' + drift_comment
    else:
        ds[variable].attrs['comment' ] = drift_comment

    return ds


# Drop variables
def drop_variables(
    ds: xr.Dataset,
    drop: list[str] | None = None,
    retain: list[str] | bool | None = None,
    verbose: bool = True,
    dims_to_check: list[str] = ["TIME"],    
) -> xr.Dataset:
    """
    Drop or retain variables in an xarray Dataset based on specified criteria.

    Exactly one of `drop` or `retain` can be provided. Variables without a TIME 
    dimension are always retained. If neither is provided, the dataset is returned unchanged.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset from which variables will be dropped.
    drop : list[str], optional
        Variables to remove from the dataset. Overrides `retain` if provided.
    retain : list[str] or bool, optional
        Variables to keep. If True, all variables are retained. Ignored if `drop` is provided.
    verbose : bool, default=True
        If True, prints information about dropped variables.
    dims_to_check : list[str], optional
        Dimensions to consider when dropping variables. Only variables that
        have at least one of these dimensions are eligible for dropping.
        Defaults to ["TIME"].
    Returns
    -------
    xr.Dataset
        A new dataset with the specified variables dropped or retained, or unchanged if neither is supplied.

    Raises
    ------
    ValueError
        If both `drop` and `retain` are provided.

    Examples
    --------
    >>> ds_new = drop_variables(ds, drop=['TEMP1', 'SAL'])
    >>> ds_new = drop_variables(ds, retain=['TEMP1', 'SAL'])
    """
    ds = ds.copy(deep=True)  # Ensure input dataset is not modified

    ds = edit.drop_variables(ds, drop= drop, retain=retain, verbose=verbose, 
                   dims_to_check=dims_to_check)

    return ds

# Drop variables (interactive)
def drop_vars_pick(ds: xr.Dataset) -> xr.Dataset:
    """
    Interactively drop (remove) selected variables from an xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset from which variables will be dropped.

    Returns
    -------
    xr.Dataset
        The dataset with the selected variables removed.

    Notes
    -----
    Displays an interactive widget with checkboxes for each variable, allowing
    users to select variables to remove. The removal is performed by clicking
    the "Drop variables" button. The removed variables are also printed to the
    output.
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    edit_obj = edit.drop_vars_pick(ds, moored=True)
    return edit_obj.ds



def metadata_auto(ds: xr.Dataset, NPI: bool = True) -> xr.Dataset:
    """
    Various modifications to the metadata to standardize the dataset for
    publication.

    This function applies several standardizations and conventions to the
    dataset's metadata, including renaming variables, adding standard
    attributes, and ensuring the metadata is consistent.

    Parameters
    ----------
    ds : xr.Dataset
        The input xarray Dataset whose metadata is to be standardized.
    NPI : bool, optional
        Not used in this function. Default is True.

    Returns
    -------
    xr.Dataset
        The dataset with updated metadata.

    Notes
    -----
    This function calls multiple sub-functions to update the metadata:
    -  `remove_numbers_in_var_names`:
        Removes numbers from variable names.
    - `add_standard_var_attrs`:
        Adds standard variable attributes.
    - `add_standard_glob_attrs_ctd`:
        Adds standard global attributes specific to CTD data.
    - `add_standard_glob_attrs_org`:
        Adds standard global attributes for the organization.
    - `add_gmdc_keywords_ctd`:
        Adds GMDC keywords for CTD data.
    - `add_range_attrs`:
        Adds range attributes.
    - `reorder_attrs`:
        Reorders attributes for consistency.
    """
    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    ds = conventionalize.remove_numbers_in_var_names(ds)
    ds = conventionalize.add_standard_var_attrs(ds, data_type='moored')
    ds = conventionalize.add_standard_glob_attrs_moor(ds, override=False)
    ds = conventionalize.add_standard_glob_attrs_org(ds)
    ds = conventionalize.add_gmdc_keywords_ctd(ds, moored = True)
    ds = conventionalize.add_range_attrs(ds)
    ds = conventionalize.reorder_attrs(ds)

    return ds

def to_mat(ds: xr.Dataset, outfile: str, simplify: bool = False) -> None:
    """
    Convert a CTD xarray.Dataset to a MATLAB .mat file.

    Adds a field `TIME_mat` containing MATLAB datenums along with the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to be converted.
    outfile : str
        Output path for the MATLAB .mat file. If the path does not end with
        '.mat', the extension will be appended automatically.
    simplify : bool, optional
        If True, only coordinate and data variables are included (no metadata).
        If False, the .mat file will include a struct containing attrs, data_vars,
        coords, and dims. Default is False.

    Returns
    -------
    None
        Saves the dataset as a MATLAB .mat file.

    Examples
    --------
    >>> to_mat(ds, 'output_matfile', simplify=True)
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    # Also transposing dimensions to PRES, TIME for ease of plotting etc in
    # MATLAB.
    matfile.xr_to_mat(ds_wo_proc.transpose(), outfile, simplify=simplify)


def check_metadata(ds: xr.Dataset | str) -> None:
    """
    Run the IOOS compliance checker on a dataset or NetCDF file.

    Checks for compliance with CF and ACDD conventions and displays the results
    interactively with a "Close" button.

    Parameters
    ----------
    ds : xr.Dataset or str
        The dataset or path to a NetCDF file to check.

    Notes
    -----
    This function is intended for interactive use in a Jupyter environment.
    """


def plot(ds: xr.Dataset) -> None:
    """
    Interactively visualize time series data from an xarray Dataset.

    Supports applying hourly or daily mean filters to variables with a TIME dimension.

    Assumptions
    -----------
    - 1-D time series data along the TIME dimension.
    - Running in a Jupyter notebook with the matplotlib widget backend.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing variables with a TIME dimension.

    Notes
    -----
    Displays the plot interactively with a "Close" button to dismiss the figure.
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    # Make sure we have datetime TIME
    ds_cf = xr.decode_cf(ds, decode_timedelta=True)

    _moored_tools.inspect_time_series(ds_cf)


# Standardize metadata
def adjust_PSAL_from_CNDC_TEMP(
        ds: xr.Dataset,
        window: int = None,
        max_diff: float = 0.15,
        min_periods: int = 1,
        plot: bool = False) -> xr.Dataset:
    '''
    Recompute PSAL from CNDC and TEMP in the dataset, with options to apply a
    rolling mean to CNDC and TEMP first, and reject samples based on the
    difference between scaled CNDC and TEMP.

    Args:
        ds (xr.Dataset):
            Input dataset containing CNDC, TEMP, and PRES variables.
        window (int, optional):
            Rolling mean window size to apply on CNDC and TEMP. If None, no
            rolling mean is applied. Defaults to None.
        max_diff (float, optional):
            Maximum allowed difference between scaled CNDC and TEMP
            for rejecting PSAL samples. Defaults to 0.15.
        min_periods (int, optional):
            Minimum number of observations in the window required to calculate
            the rolling mean. Defaults to 1.
        plot (bool, optional):
            If True, plot the scaled CNDC, TEMP, and PSAL data. Defaults to
            False.

    Returns:
        xr.Dataset: A new dataset with updated PSAL values and a comment
        attribute.
    '''

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    CNDC_mS_cm = ds.CNDC.copy()

    if 'units' in CNDC_mS_cm.attrs:
        if CNDC_mS_cm.units == 'S m-1':
            CNDC_mS_cm.values *= 10

    # Rolling mean (applied on CNDC and TEMP for computing PSAL):
    # Parameters
    if window:
        roll_params = {'dim': {'TIME': window},
                       'center': True,
                       'min_periods': min_periods}
        # Apply to CNDC and TEMP
        CNDC_ = CNDC_mS_cm.rolling(**roll_params).mean()
        TEMP_ = ds.TEMP.rolling(**roll_params).mean()
        comment = (
            f'Computed from CNDC and TEMP using the gsw module '
            f'after applying a {window}-point running mean to both. ')
        post_proc_comment = (
            f'Recomputed PSAL from CNDC and TEMP '
            f'after applying a {window}-point running mean to both. ')
    else:
        CNDC_ = CNDC_mS_cm.copy()
        TEMP_ = ds.TEMP.copy()
        comment = ''
        post_proc_comment = ''

    # Recalculate PSAL
    PSAL_ = gsw.SP_from_C(CNDC_, TEMP_, ds.PRES)

    # Compute scaled TEMP and CNDC (for comparing the two quantitatively)
    TEMP_scaled = (TEMP_ - TEMP_.mean()) / np.std(TEMP_ - TEMP_.mean())
    CNDC_scaled = (CNDC_ - CNDC_.mean()) / np.std(CNDC_ - CNDC_.mean())

    # Detrend
    TEMP_scaled = xr.apply_ufunc(signal.detrend, TEMP_scaled)
    CNDC_scaled = xr.apply_ufunc(signal.detrend, CNDC_scaled)

    # Flag for PSAL samples to reject
    reject_sal = np.bool_(np.abs(CNDC_scaled - TEMP_scaled) > max_diff)
    n_samp = np.sum(reject_sal)
    comment += (
        'Samples where SD-normalized scaled temperature and conductivity '
        f'anomalies diverge by >{max_diff} were rejected.')
    post_proc_comment = (
        'Rejected samples where normalized TEMP and CNDC '
        f'anomalies diverged by >{max_diff} ({n_samp} samples).')

    # Apply flag
    PSAL_ = PSAL_.where(~reject_sal)

    # Make a copy of the dataset and update with the new PSAL
    ds1 = ds.copy()
    ds1.PSAL.values = PSAL_
    if 'comment' in ds1.PSAL.attrs:
        ds1.PSAL.attrs['comment'] += f'\n\n{comment}'
    else:
        ds1.PSAL.attrs['comment'] = comment


    if plot:
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(num2date(ds.TIME.values), CNDC_scaled, color='tab:orange',
                   label='Scaled CNDC')
        ax[0].plot(ds.TIME, TEMP_scaled, alpha=0.6, lw=1,
                   label='Scaled TEMP')
        ax[0].plot(ds.TIME.values[reject_sal], CNDC_scaled.values[reject_sal],
                   'or', ms=3, label='PSAL rejected')

        ax[1].plot(ds.TIME, np.abs(CNDC_scaled - TEMP_scaled),
                   label='|Scaled CNDC - Scaled TEMP|')
        ax[1].plot(ds.TIME.values[reject_sal],
                   np.abs(CNDC_scaled - TEMP_scaled).values[reject_sal],
                   'or', ms=3, label='PSAL rejected')
        ax[1].axhline(max_diff, linestyle=':', color='k', label='max_diff')

        ax[2].plot(ds.TIME, ds.PSAL, 'k', alpha=0.7, label='Original')
        ax[2].plot(ds.TIME, PSAL_, color='tab:orange', label='Updated')
        ax[2].set_ylabel('PSAL')

        for axn in ax:
            leg = axn.legend(fontsize=9)
            leg.set_zorder(0)

        ax[2].xaxis.set_major_formatter(DateFormatter('%H:%M'))

    return ds1


def get_median_depth(ds: xr.Dataset, lat: float = None, decimals: int = 1) -> float:
    """
    Calculate the median depth from an xarray dataset based on the pressure record.

    Args:
        ds (xr.Dataset): Input dataset containing a pressure (PRES) variable.
        lat (float, optional): Latitude in degrees. If not provided, the function attempts to extract
                               it from the dataset's LATITUDE variable. Defaults to None.
        decimals (int, optional): Number of decimal places to round the depth. Defaults to 1.

    Returns:
        float: The median depth in meters, rounded to the specified number of decimal places.

    Raises:
        ValueError: If latitude is not provided and cannot be extracted from the dataset.
        KeyError: If the dataset does not contain a 'PRES' variable.
    """

    

    # Ensure the dataset contains the 'PRES' variable
    if 'PRES' not in ds:
        raise KeyError("The dataset must contain a 'PRES' variable (pressure in decibars).")

    # Attempt to get latitude from the dataset if not provided
    if lat is None:
        try:
            lat = ds.LATITUDE.values.item()
        except AttributeError:
            raise ValueError("No latitude provided and no 'LATITUDE' variable found in the dataset.")

    # Calculate the median pressure, ignoring NaN values
    median_pres = np.nanmedian(ds.PRES)

    # Convert the median pressure to depth using the gsw.z_from_p function
    median_depth = np.round(-gsw.z_from_p(median_pres, lat), decimals)

    return median_depth






def _split_attrs(var_attr_sets, all_labels):
    """
    var_attr_sets: list of (label, attrs_dict) for instruments that are
    ELIGIBLE for this attribute (e.g. instruments that have the variable
    at all, for variable-level attrs; all instruments, for global attrs).
    all_labels: full list of INSTR labels (used to build the per-instrument
    coordinate arrays, including ineligible instruments as None).
 
    An attribute is "shared" if every ELIGIBLE instrument specifies it and
    they all agree -- instruments that don't have the variable at all do
    NOT count against sharing (e.g. PRES.units can be 'dbar' even if only
    one of several instruments has a PRES variable).
 
    Returns (shared_attrs, per_instr_attrs):
    - shared_attrs: {attr_key: value} for attrs that are consistent across
      all eligible instruments.
    - per_instr_attrs: {attr_key: [values in all_labels order]} for attrs
      that vary (or aren't specified by every eligible instrument); value
      is None for instruments that are ineligible or don't specify the key.
    """
    keys = set()
    for _, attrs in var_attr_sets:
        keys.update(attrs.keys())
 
    label_to_attrs = {label: attrs for label, attrs in var_attr_sets}
    eligible_labels = [label for label, _ in var_attr_sets]
 
    shared = {}
    per_instr = {}
    for key in sorted(keys):
        eligible_values = [label_to_attrs[lbl].get(key, None) for lbl in eligible_labels]
        non_none_eligible = [v for v in eligible_values if v is not None]
        if (len(non_none_eligible) == len(eligible_labels)
                and all(v == non_none_eligible[0] for v in non_none_eligible)):
            shared[key] = non_none_eligible[0]
        else:
            per_instr[key] = [
                label_to_attrs.get(lbl, {}).get(key, None) for lbl in all_labels
            ]
    return shared, per_instr
 
 
def combine_datasets(
    *datasets: xr.Dataset,
    interval: str,
    method: str = 'time_average',
    instr_dim: str = 'INSTR',
    instr_names: list = None,
    time_dim: str = 'TIME',
) -> xr.Dataset:
    """
    Combine multiple xarray Datasets (e.g. different instruments on a
    mooring) onto a single common TIME grid, stacked along a new
    dimension (`instr_dim`, default 'INSTR').
 
    Each dataset is put onto a common regular TIME grid, anchored to the
    same origin across all datasets so grid points line up exactly. Two
    methods are available (`method`):
    - 'time_average' (default): each grid point is the mean of samples
      falling in [t, t + interval) -- see `kval.util.xr_funcs.time_average`.
      If a dataset has no sample in a given interval (e.g. its own
      sampling rate is coarser than `interval`), that point is NaN.
    - 'interpolate': each grid point is linearly interpolated from the
      dataset's own samples. Outside a dataset's own time range, values
      are NaN (no extrapolation).
 
    Either way, a variable missing from some datasets, or a dataset not
    covering part of the combined time range, results in NaN entries for
    those instruments/times. Each numeric TIME-dependent variable becomes
    2D (INSTR, TIME); scalar (non-TIME) numeric variables (e.g. LATITUDE)
    become 1D (INSTR,). Non-numeric TIME-dependent variables can't be
    averaged or interpolated and are dropped (reported via print, once
    per dataset that has one).
 
    Metadata handling: for both global (dataset-level) and variable-level
    attributes, a value that is identical across every instrument that
    actually has the variable is kept as a shared attribute -- instruments
    lacking the variable entirely don't count against sharing (e.g. a
    PRES.units of 'dbar' stays shared even if only one instrument has
    PRES). Otherwise, the attribute becomes a new coordinate variable
    along `instr_dim`:
    - Global attrs that vary -> coordinate named after the attr key
      itself (e.g. 'instrument_serial_number').
    - Variable-level attrs that vary -> coordinate named
      '{variable}_{attr_key}' (e.g. 'TEMP_sensor_calibration_date').
 
    Parameters
    ----------
    *datasets : xr.Dataset
        Two or more datasets to combine. Each must have a `time_dim`
        coordinate, either datetime64 or numeric with a "<units> since
        <ref>" units attribute.
    interval : str
        Spacing of the common TIME grid, as a pandas frequency string
        (e.g. '1h', '30min', '1D'). No default -- must be chosen based on
        the data being combined. Must be a fixed-duration interval (not
        calendar-based like 'M'/'Y').
    method : {'time_average', 'interpolate'}, default='time_average'
        How to put each dataset onto the common grid; see above.
    instr_dim : str, default='INSTR'
        Name of the new dimension along which datasets are stacked.
    instr_names : list of str, optional
        Labels for each dataset along `instr_dim`, in the same order as
        `datasets`. Must be unique. If not given, labels are auto-detected
        from each dataset's 'instrument_serial_number' global attribute,
        falling back to an integer index for any dataset lacking it.
    time_dim : str, default='TIME'
        Name of the time dimension/coordinate in the input datasets.
 
    Returns
    -------
    xr.Dataset
        Combined dataset with dimensions (`instr_dim`, `time_dim`) for
        TIME-dependent variables and (`instr_dim`,) for scalar variables.
 
    Raises
    ------
    ValueError
        If fewer than 2 datasets are given, if `instr_names` has the wrong
        length or contains duplicates, if any dataset lacks `time_dim`, or
        if numeric TIME has no 'units' attribute.
    """
    if len(datasets) < 2:
        raise ValueError("combine_datasets requires at least 2 datasets")
    if method not in ('time_average', 'interpolate'):
        raise ValueError("method must be 'time_average' or 'interpolate'")
 
    n = len(datasets)
 
    # --- Resolve instrument labels ---
    if instr_names is not None:
        if len(instr_names) != n:
            raise ValueError(
                f"instr_names has {len(instr_names)} entries but "
                f"{n} datasets were provided")
        labels = list(instr_names)
    else:
        labels = []
        for i, ds in enumerate(datasets):
            serial = ds.attrs.get('instrument_serial_number')
            labels.append(str(serial) if serial is not None else i)
 
    if len(set(labels)) != len(labels):
        raise ValueError(f"INSTR labels must be unique, got: {labels}")
 
    # --- Decode TIME to datetime64 for each dataset ---
    decoded = []
    for ds in datasets:
        if time_dim not in ds.coords:
            raise ValueError(f"Dataset missing '{time_dim}' coordinate")
        tvals = ds[time_dim].values
        if np.issubdtype(tvals.dtype, np.datetime64):
            decoded.append(ds)
        else:
            if 'units' not in ds[time_dim].attrs:
                raise ValueError(
                    f"Dataset {time_dim} is numeric but has no 'units' "
                    "attribute needed to decode to datetime")
            decoded.append(xr.decode_cf(ds, decode_timedelta=True))
 
    # --- Common origin, so all datasets land on exactly the same grid ---
    global_min = min(ds[time_dim].values.min() for ds in decoded)
    global_max = max(ds[time_dim].values.max() for ds in decoded)
    origin = pd.Timestamp(global_min)
 
    # Common TIME grid, spanning the combined range. Each grid point is
    # either a bin start (method='time_average', label='left') or a
    # direct interpolation point (method='interpolate') -- using the same
    # grid point convention for both keeps the two methods' outputs
    # directly comparable for the same `interval`.
    common_time = pd.date_range(
        start=origin, end=pd.Timestamp(global_max), freq=interval)
 
    averaged = []
    if method == 'time_average':
        for ds in decoded:
            ds_avg = time_average(
                ds, interval=interval, label='left', origin=origin,
                time_dim=time_dim)
            ds_avg = ds_avg.reindex({time_dim: common_time})
            averaged.append(ds_avg)
    else:  # method == 'interpolate'
        for ds in decoded:
            # Match time_average's handling of non-numeric TIME-dependent
            # variables: they can't be interpolated either, so drop them
            # (reporting it, same as time_average does).
            dropped_vars = [
                v for v in ds.data_vars
                if time_dim in ds[v].dims and not np.issubdtype(ds[v].dtype, np.number)
            ]
            time_indep_vars = [v for v in ds.data_vars if time_dim not in ds[v].dims]
            time_dep_vars = [
                v for v in ds.data_vars
                if time_dim in ds[v].dims and v not in dropped_vars
            ]
            ds_interp = ds[time_dep_vars].interp({time_dim: common_time})
            ds_interp = ds_interp.merge(ds[time_indep_vars])
            if dropped_vars:
                print(f"combine_datasets: dropped non-numeric {time_dim}-"
                      f"dependent variable(s) {dropped_vars} (cannot "
                      "interpolate non-numeric data)")
            averaged.append(ds_interp)
 
    # --- Identify variable groups across all (averaged) datasets ---
    time_vars = set()
    scalar_vars = set()
    for ds in averaged:
        for v in ds.data_vars:
            is_time_dep = time_dim in ds[v].dims
            if is_time_dep:
                time_vars.add(v)
            elif np.issubdtype(ds[v].dtype, np.number):
                scalar_vars.add(v)
            # non-numeric, non-time-dependent variables: rare for mooring
            # instrument metadata; not handled, silently skipped
 
    data_vars_out = {}
    coords_out = {instr_dim: labels, time_dim: common_time}
 
    # --- TIME-dependent (already time-averaged) variables: stack ---
    for var in sorted(time_vars):
        arr = np.full((n, len(common_time)), np.nan)
        var_attr_sets = []
        for i, ds in enumerate(averaged):
            if var in ds and time_dim in ds[var].dims:
                da = ds[var]
                arr[i, :] = da.values
                # attrs come from the *original* (pre-averaging) dataset,
                # since resample/mean doesn't necessarily preserve them
                orig_attrs = dict(decoded[i][var].attrs) if var in decoded[i] else {}
                var_attr_sets.append((labels[i], orig_attrs))
 
        shared_attrs, per_instr_attrs = _split_attrs(var_attr_sets, labels)
        data_vars_out[var] = ((instr_dim, time_dim), arr, shared_attrs)
        for attr_key, per_instr_vals in per_instr_attrs.items():
            coord_name = f'{var}_{attr_key}'
            coords_out[coord_name] = (instr_dim, per_instr_vals)
 
    # --- Scalar (non-TIME) numeric variables: stack along INSTR only ---
    for var in sorted(scalar_vars):
        arr = np.full(n, np.nan)
        var_attr_sets = []
        for i, ds in enumerate(averaged):
            if var in ds and time_dim not in ds[var].dims:
                val = ds[var].values
                arr[i] = val.item() if np.ndim(val) == 0 else val
                var_attr_sets.append((labels[i], dict(ds[var].attrs)))
 
        shared_attrs, per_instr_attrs = _split_attrs(var_attr_sets, labels)
        data_vars_out[var] = ((instr_dim,), arr, shared_attrs)
        for attr_key, per_instr_vals in per_instr_attrs.items():
            coord_name = f'{var}_{attr_key}'
            coords_out[coord_name] = (instr_dim, per_instr_vals)
 
    # --- Global attrs: shared vs per-instrument (every dataset is
    # "eligible" here, so this reduces to requiring all datasets to agree) ---
    global_attr_sets = [(labels[i], dict(decoded[i].attrs)) for i in range(n)]
    combined_global_attrs, per_instr_global = _split_attrs(global_attr_sets, labels)
    for attr_key, per_instr_vals in per_instr_global.items():
        coords_out[attr_key] = (instr_dim, per_instr_vals)
 
    ds_out = xr.Dataset(data_vars_out, coords=coords_out, attrs=combined_global_attrs)
 
    return ds_out
 
