"""
KVAL.DATA.MOORED

Loadin and processing data from fixed instruments.

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
from typing import Optional, Tuple, Union, List
import numpy as np

import os
import gsw
import matplotlib.pyplot as plt
import matplotlib as mpl

from kval.file import sbe, rbr, matfile
from kval.data import dataset, edit
from kval.data.moored_tools import _moored_tools
from kval.data.moored_tools._moored_decorator import record_processing

from kval.util import internals, index, time
from kval.signal import despike, filt
from kval.metadata import conventionalize
from kval.metadata.check_conventions import check_file_with_button
import warnings

# Want to be able to use these functions directly..
from kval.data.dataset import metadata_to_txt, to_netcdf

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
    if file.endswith(".rsk"):
        instr_type = "RBR"
    elif file.endswith(".cnv"):
        instr_type = "SBE"
    elif file.endswith(".csv"):
        instr_type = "SBE_csv"
    else:
        raise ValueError(
            f"Unable to load moored instrument {os.path.basename(file)}.\n"
            "Supported files are .cnv (SBE) and .rsk (RBR)."
        )

    # Load data
    if instr_type == "RBR":
        ds = rbr.read_rsk(file)
    elif instr_type == "SBE":
        ds = sbe.read_cnv(file)
    elif instr_type == "SBE_csv":
        ds = sbe.read_csv(file)

    # Assign lat/lon if we have specified them
    if lat:
        ds["LATITUDE"] = ((), lat)
    if lon:
        ds["LONGITUDE"] = ((), lon)

    # Add PROCESSING variable with useful metadata
    # ( + remove some excessive global attributes)
    if processing_variable:
        ds = dataset.add_processing_history_var_moored(
            ds,
        )
        # Add a source_file attribute (and remove from the global attrs)
        ds.PROCESSING.attrs["source_file"] = ds.source_files

        # Remove some unwanted global atributes
        for attr_name in [
            "source_files",
            "filename",
            "SBE_flags_applied",
            "SBE_processing_date",
        ]:
            if attr_name in ds.attrs:
                del ds.attrs[attr_name]

        # For SBE: Move the SBE_processing attribute to PROCESSING.
        if instr_type == "SBE" and "SBE_processing" in ds.attrs:
            ds.PROCESSING.attrs["SBE_processing"] = ds.SBE_processing
            del ds.attrs["SBE_processing"]

            # Adde exlanation of `SBE_processing` to the `comment` attribute
            ds.PROCESSING.attrs["comment"] += (
                "# SBE_processing #:\nSummary of the post-processing applied "
                "within SeaBird software to produce the .cnv file."
            )

        # Add python scipt snipped to reproduce this operation
        ds.PROCESSING.attrs[
            "python_script"
        ] += f"""from kval import data
data_dir = "./" # Directory containing `filename` (MUST BE SET BY THE USER!)
filename = "{os.path.basename(file)}"

# Load file into an xarray Dataset:
ds = data.moored.load_moored(
    data_dir + filename,
    processing_variable={processing_variable})
    """

    return ds


# Chop record
# Note: We do the recording to PROCESSING inside the function, not in the
# decorator. (Too complex otherwise)
def chop_deck(
    ds: xr.Dataset,
    variable: str = "PRES",
    sd_thr: float = 1.0,
    indices: Optional[Tuple[int, int]] = None,
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
    # Make sure we are working with a copy
    ds = ds.copy()

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

    # Record to PROCESSING metadata variable
    if "PROCESSING" in ds:

        if keep_slice.start is None and keep_slice.stop is not None:
            start_end_str = "end"
            indices_str = f"None, {keep_slice.stop-1}"
        elif keep_slice.start is not None and keep_slice.stop is None:
            start_end_str = "start"
            indices_str = f"{keep_slice.start}, None"
        elif keep_slice.start is not None and keep_slice.stop is not None:
            start_end_str = "start and end"
            indices_str = f"{keep_slice.start}, {keep_slice.stop-1}"

        if keep_slice.start is None and keep_slice.stop is None:
            pass
        else:
            ds["PROCESSING"].attrs["post_processing"] += (
                f"Chopped {L0 - L1} samples at the {start_end_str} "
                "of the time series.\n"
            )

            ds["PROCESSING"].attrs["python_script"] += (
                f"\n\n# Chopping away samples from the {start_end_str}"
                " of the time series\n"
                f"ds = data.moored.chop_deck(ds, indices = [{indices_str}])"
            )

    return ds


# Despike
@record_processing(
    "",
    py_comment=(
        "Find/reject {var_name} outliers (points exceeding {window_size}-"
        "pt rolling {filter_type} by>{n_std} SDs."
    ),
)
def despike_rolling(
    ds: xr.Dataset,
    var_name: str,
    window_size: int,
    n_std: float,
    dim: str = "TIME",
    filter_type: str = "median",
    min_periods: Union[int, None] = None,
    plot: bool = False,
    verbose: bool = False,
) -> xr.Dataset:
    """

    Despike a variable in a dataset by identifying and removing outliers
    based on a rolling mean/median and standard deviation.

    Outliers are data points where the variable deviates from the rolling
    mean/median by a number of standard deviations. Both the mean/median
    and standard deviation are calculated within a rolling window centered
    on each data point.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the variable to despike.
    var_name : str
        The name of the variable to despike.
    window_size : int
        The size of the rolling window for calculating the mean/median
        and standard deviation.
    n_std : float
        The number of standard deviations used as the threshold to identify
        outliers.
    dim : str
        The dimension along which to calculate the rolling statistics.
        Default: 'TIME'.
    filter_type : str, optional
        The type of filter to apply ('mean' or 'median'). Default is 'mean'.
    min_periods : int or None, optional
        The minimum number of observations in the window required to return
        a valid result. Default is None.
    plot : bool, optional
        If True, plots the original data and the despiking results.
        Default is False.
    verbose : bool, optional
        If True, print some basic info about the result of the despiking.
        Default is False.
    Returns
    -------

        - If `return_ds` is True and `return_index` is False: returns the
          updated dataset with the despiked variable.
        - If `return_ds` is False and `return_index` is False: returns the
          despiked variable as a DataArray.
        - If `return_ds` is True and `return_index` is True: returns a tuple
          of the updated dataset and a mask of outliers.
        - If `return_ds` is False and `return_index` is True: returns a tuple
          of the despiked variable and a mask of outliers.
    """

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

    n_removed = np.sum(is_outside_criterion).item()
    if "PROCESSING" in ds:
        ds.PROCESSING.attrs["post_processing"] += (
            f"Edited out spikes {var_name} using a rolling window criterion. "
            f"Values exceeding the {window_size}-point rolling {filter_type} "
            f"by more than {n_std} (rolling) standard deviations were "
            f"interpreted as outliers and masked (found {n_removed} "
            "outliers)."
        )

    return ds



# Adjust for clock drift
@record_processing(
    "",
    py_comment=(
        "Adjust for clock drift"
    ))

def adjust_time_for_drift(
    ds: xr.Dataset,
    seconds: Optional[float] = 0,
    minutes: Optional[float] = 0,
    hours: Optional[float] = 0,
    days: Optional[float] = 0) -> xr.Dataset:
    """
    Adjust the TIME coordinate of an xarray Dataset for instrument clock drift.

    Clock offset is specified in sec, min, hrs, days.

    *Negative* clock values: Instrument *lags* true -> *adding* offset.
    *Positive* clock values: Instrument *leads* true -> *subtracting* offset.

    Parameters:
    ds (xarray.Dataset): Input dataset with a TIME coordinate.
    seconds (float): Clock drift, seconds.
    minutes (float): Clock drift, minutes.
    hours (float): Clock drift, hours.
    days (float): Clock drift, days.

    Returns:
    xarray.Dataset: Dataset with adjusted TIME coordinate.
    """

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

    # Get the TIME coordinate
    time = ds.coords['TIME'].values
    drift_adjustments_sec = (
        np.arange(len(time)) / (len(time)-1))*total_drift_seconds


    # Check if TIME is in numerical format (days since epoch)
    if 'DAYS SINCE' in ds.TIME.units.upper():
        adjusted_time = time - drift_adjustments_sec / 86400
    else:
        raise Exception('Could not add drift because TIME is non-numerical'
                        ' or has unknown units (should be "Days since..")')

    # Update the TIME coordinate in the dataset
    time_attrs = ds['TIME'].attrs
    time_attrs['comment'] = (
        f'Adjusted for observed clock drift ({drift_operation} '
        f'from 0 to {abs(total_drift_seconds)} sec)')
    ds['TIME'] = ('TIME', adjusted_time, time_attrs)

    if "PROCESSING" in ds:
        ds.PROCESSING.attrs["post_processing"] += (
            f"Adjusted for clock offset: {drift_operation}"
            f" from 0 to {abs(total_drift_seconds)} s assuming linear drift"
        )

    return ds


# Filtering
@record_processing(
    "Ran a {window_size}-point rolling {filter_type} filter "
    "on the variable {dim}.",
    py_comment=(
        "Run a {window_size}-point rolling {filter_type} filter " "on {dim}"
    ),
)
def rolling_mean(
    ds: xr.Dataset,
    var_name: str,
    window_size: int,
    filter_type: str = "mean",
    dim: str = "TIME",
    min_periods: Union[bool, int] = None,
    nan_edges: bool = True,
) -> xr.Dataset:
    """
    Apply a running mean or median filter on a variable of an xarray Dataset
    along a specific dimension, with options to handle NaNs and edge values.

    Parameters:
    - ds: xarray.Dataset
        The dataset containing the variable to filter.
    - var_name: str
        The name of the variable in the dataset to apply the filter on.
    - dim: str
        The dimension along which to apply the filter.
        Default: TIME
    - window_size: int
        The size of the rolling window.
    - filter_type: str, optional
        The type of filter to apply: 'mean' for running mean, 'median' for
        running median, 'sd' for standard deviation. Defaults to 'mean'.
    - min_periods: Union[bool, int], optional
        Minimum number of observations in the window required to have a value.
        If an integer, it specifies the minimum number of observations in a
        rolling window.
        If `None` (default), all windows with a NaN will be set to Nan.
    - nan_edges: bool, optional
        Whether to set edge values (half of the window length) to NaN.
        (Redundant if min_periods is set to `None`)
        Defaults to `True`.

    Returns:
    - ds_filt: xarray.Dataset
        The dataset with the filtered variable, where edge values may be NaN
        if `nan_edges` is `True`.
    """

    ds = filt.rolling(
        ds=ds,
        var_name=var_name,
        window_size=window_size,
        filter_type=filter_type,
        dim=dim,
        min_periods=min_periods,
        nan_edges=nan_edges,
    )

    return ds


# Drift


# Threshold edit
@record_processing(
    "Rejected values of {variable} outside the range ({min_val}, {max_val})",
    py_comment=(
        "Rejecting values of {variable} outside the range "
        "({min_val}, {max_val}):"
    ),
)
def threshold(
    ds: xr.Dataset,
    variable: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> xr.Dataset:
    """
    Apply a threshold to a specified variable in an xarray Dataset, setting
    values outside the specified range (min_val, max_val) to NaN.

    Also modifies the valid_min and valid_max variable attributes.

    Parameters
    ----------
    ds : xr.Dataset
        The input xarray Dataset.
    variable : str
        The name of the variable within the Dataset to be thresholded.
    min_val : Optional[float], default=None
        The minimum allowed value for the variable. Values less than
        this will be set to NaN. If None, no lower threshold is applied.
    max_val : Optional[float], default=None
        The maximum allowed value for the variable. Values greater than
        this will be set to NaN. If None, no upper threshold is applied.

    Returns
    -------
    xr.Dataset
        A new xarray Dataset with the thresholded variable. The `valid_min`
        and `valid_max` attributes are updated accordingly.

    Examples
    --------
    # Reject temperatures below -1 and above 3
    ds_thresholded = threshold(ds, 'TEMP', max_val=3, min_val=-1)
    """
    ds = edit.threshold(
        ds=ds, variable=variable, max_val=max_val, min_val=min_val
    )

    return ds


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

    data_variables = []

    for varnm in ds.data_vars:
        if "TIME" in ds[varnm].dims:
            data_variables += [varnm]

    edit.threshold_edit(ds, variables=data_variables)
    return ds


# Remove points by index
@record_processing(
    "Rejecting (setting to NaN) the following time indices from"
    " {varnm}:\n{remove_inds}.",
    py_comment=("Reject {varnm} values at specific points"),
)
def remove_points(
    ds: xr.Dataset, varnm: str, remove_inds, time_var="TIME"
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

    ds = edit.remove_points_timeseries(
        ds=ds, varnm=varnm, remove_inds=remove_inds, time_var=time_var
    )

    return ds


# Remove points (hand pick)
def hand_remove_points(
    ds: xr.Dataset,
    variable: str,
) -> xr.Dataset:
    """
    Interactively remove data points from CTD profiles.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the CTD data.
    variable : str
        The name of the variable to visualize and edit (e.g., 'TEMP1', 'CHLA').
    TIME_index : str
        The name of the station (e.g., '003', '012_01', 'AT290', 'StationA').

    Returns
    -------
    xr.Dataset
        The dataset with data points removed based on interactive input.

    Examples
    --------
    >>> ds = hand_remove_points(ds, 'TEMP1', 'StationA')

    Notes
    -----
    Use the interactive plot to select points for removal, then click the
    corresponding buttons for actions.
    """

    hand_remove = _moored_tools.hand_remove_points(ds, variable)
    ds = hand_remove.ds

    return ds


# Recalculate sal
@record_processing(
    "Recalculated PSAL using the GSW-Python module.",
    py_comment="Recalculating PSAL",
)
def calculate_PSAL(
    ds: xr.Dataset,
    cndc_var: str = "CNDC",
    temp_var: str = "TEMP",
    pres_var: str = "PRES",
    psal_var: str = "PSAL",
) -> xr.Dataset:
    """Recalculate Practical Salinity (PSAL) from conductivity, temperature,
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

    Returns:
        xr.Dataset: The updated dataset with recalculated PSAL values.

    Notes:
        The operation preserves PSAL metadata attributes. If the input sensors
        change (e.g., if a different temperature sensor is used), the PSAL
        metadata attributes should be updated accordingly.
    """
    PSAL = gsw.SP_from_C(ds[cndc_var], ds[temp_var], ds[pres_var])
    ds[psal_var][:] = PSAL

    ds[psal_var].attrs["note"] = (
        f"Computed from {cndc_var}, {temp_var}, {pres_var} "
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

    Returns
    -------
    xarray.Dataset
        Updated dataset `ds_main` with an added 'PRES' variable containing the
        estimated pressures [dbar].

    Raises
    ------
    Exception
        If latitude (`lat`) is not provided and cannot be inferred from
        `ds_main`.
    """

    # Ensure we have latitude for depth-to-pressure conversion
    if lat is None:
        try:
            lat = ds_main.LATITUDE.item()
        except AttributeError:
            raise Exception(
                "Could not find latitude for depth->pressure calculation. "
                "Specify `lat` in `assign_pressure`."
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

    return ds_main


# Recalculate sal
@record_processing(
    "Applied an offset to {variable} linearly changing from {start_val} to {end_val}.",
    py_comment="Apply linear drift to {variable}",
)
def linear_drift_offset(
    ds: xr.Dataset,
    variable: str,
    end_val: float,
    start_val: float = 0,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None) -> xr.Dataset:
    '''
    (Wrapper for kval.data.edit.linear_drift)

    Apply a linearly increasing drift offset to a variable in the
    dataset.

    This function applies a linearly increasing drift over time to a specified
    variable in an xarray Dataset. The drift is added as an additive factor.

    The drift is applied between `start_date` and `end_date` (if provided), or
    over the entire time range of the dataset.

    Args:
        ds (xr.Dataset):
            Input xarray Dataset containing the time series data.
        variable (str):
            The name of the variable in the dataset to which the drift will be
            applied.
        end_val (float):
            The value of the drift offset at the end of the period.
        start_val (float, optional):
            The starting value of the drift offset. Default is 0.
        start_date (str, optional):
            The starting date for applying the drift in 'YYYY-MM-DD' format. If
            None, the drift starts at the first time value in the dataset.
            Default is None.
        end_date (str, optional):
            The ending date for applying the drift in 'YYYY-MM-DD' format. If
            None, the drift ends at the last time value in the dataset. Default
            is None.

    Returns:
        xr.Dataset: A new dataset with the drift applied to the specified
        variable.
        '''

    ds = edit.linear_drift(
        ds, variable, end_val, start_val=start_val, start_date=start_date,
        end_date=end_date, factor=False)

    return ds


# Recalculate sal
@record_processing(
    "Applied an correctional factor to {variable} linearly changing from {start_val} to {end_val}.",
    py_comment="Apply linear drift to {variable}",
)
def linear_drift_factor(
    ds: xr.Dataset,
    variable: str,
    end_val: float,
    start_val: float = 1,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None) -> xr.Dataset:
    '''
    (Wrapper for kval.data.edit.linear_drift)

    Apply a linearly increasing drift factor to a variable in the
    dataset.

    This function applies a linearly increasing drift over time to a specified
    variable in an xarray Dataset. The drift is added as a multiplicative
    factor.

    The drift is applied between `start_date` and `end_date` (if provided), or
    over the entire time range of the dataset.

    Args:
        ds (xr.Dataset):
            Input xarray Dataset containing the time series data.
        variable (str):
            The name of the variable in the dataset to which the drift will be
            applied.
        end_val (float):
            The value of the drift factor at the end of the period .
        start_val (float, optional):
            The starting value of the drift factor. Default is 1.
        start_date (str, optional):
            The starting date for applying the drift in 'YYYY-MM-DD' format. If
            None, the drift starts at the first time value in the dataset.
            Default is None.
        end_date (str, optional):
            The ending date for applying the drift in 'YYYY-MM-DD' format. If
            None, the drift ends at the last time value in the dataset. Default
            is None.

    Returns:
        xr.Dataset: A new dataset with the drift applied to the specified
        variable.
        '''

    ds = edit.linear_drift(
        ds, variable, end_val, start_val=start_val, start_date=start_date,
        end_date=end_date, factor=True)

    return ds


# Drop variables
@record_processing("", py_comment="Dropping some variables")
# Note: Doing PROCESSING.post_processing record keeping within the
# drop_variables() function because we want to access the *dropped* list.
def drop_variables(
    ds: xr.Dataset,
    drop_vars: Optional[List[str]] = None,
    retain_vars: Optional[Union[List[str], bool]] = None,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Drop measurement variables from the dataset based on specified criteria.

    This function retains or drops variables from an xarray.Dataset based on
    provided lists of variables to retain or drop. If `retain_vars` is True,
    no variables will be dropped.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset from which variables will be dropped.
    retain_vars : Optional[Union[List[str], bool]], default=None
        List of variables to retain. If a boolean `True` is provided, all
        variables are retained (no changes made). This parameter is ignored if
        `drop_vars` is specified.
    drop_vars : Optional[List[str]], default=None
        List of variables to drop from the dataset. If specified, this will
        override `retain_vars`.
    verbose : bool, default=True
        Whether to print information about the dropped variables.

    Returns
    -------
    xr.Dataset
        The modified dataset with specified variables dropped or retained.

    Notes
    -----
    Provide *either* `retain_vars` or `drop_vars`, but not both.
    Variables without a TIME dimension are always retained.
    """

    if retain_vars is not None and drop_vars is not None:
        raise ValueError(
            "Specify either `retain_vars` or `drop_vars`, but not both."
        )

    if retain_vars is None and drop_vars is None:
        return ds

    dropped = []  # To keep track of dropped variables

    # Case: drop variables by explicitly provided drop_vars
    if drop_vars is not None:
        ds = ds.drop_vars(drop_vars)
        dropped = drop_vars
    # Case: retain variables based on retain_vars list
    else:
        if isinstance(retain_vars, bool):
            if retain_vars:  # If retain_vars is True, return unchanged dataset
                return ds
            retain_vars = []  # If False, treat it as an empty retain list

        all_vars = list(ds.data_vars)
        for varnm in all_vars:
            # Drop variables not in retain_vars, and those having "TIME" in
            # dimensions
            if varnm not in retain_vars and "TIME" in ds[varnm].dims:
                ds = ds.drop_vars(varnm)
                dropped.append(varnm)

    # Inform and log dropped variables
    if dropped:
        drop_str = f"Dropped variables from the Dataset: {dropped}."
        if verbose:
            print(drop_str)
        if "PROCESSING" in ds:
            ds["PROCESSING"].attrs["post_processing"] = (
                ds["PROCESSING"].attrs.get("post_processing", "")
                + f"{drop_str}\n")

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
    edit_obj = edit.drop_vars_pick(ds, moored=True)
    return edit_obj.D


# Standardize metadata
@record_processing(
    "Applied automatic standardization of metadata.",
    py_comment="Applying standard metadata (global+variable attributes):",
)
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
    ds = conventionalize.remove_numbers_in_var_names(ds)
    ds = conventionalize.add_standard_var_attrs(ds, data_type='moored')
    ds = conventionalize.add_standard_glob_attrs_moor(ds, override=False)
    ds = conventionalize.add_standard_glob_attrs_org(ds)
    ds = conventionalize.add_gmdc_keywords_ctd(ds)
    ds = conventionalize.add_range_attrs(ds)
    ds = conventionalize.reorder_attrs(ds)

    return ds


# Export to matfile
def to_mat(ds, outfile, simplify=False):
    """
    Convert the CTD data (xarray.Dataset) to a MATLAB .mat file.

    A field 'TIME_mat' with Matlab datenums is added along with the data.

    Parameters:
    - ds (xarray.Dataset): Input dataset to be converted.
    - outfile (str): Output file path for the MATLAB .mat file. If the path
      doesn't end with '.mat', it will be appended.
    - simplify (bool, optional): If True, simplify the dataset by extracting
      only coordinate and data variables (no metadata attributes). If False,
      the matfile will be a struct containing [attrs, data_vars, coords, dims].
      Defaults to False.

    Returns:
    None: The function saves the dataset as a MATLAB .mat file.

    Example:
    >>> moored.xr_to_mat(ds, 'output_matfile', simplify=True)
    """
    # Drop the empty PROCESSING variable (doesn't work well with MATLAB)
    ds_wo_proc = drop_variables(ds, drop_vars="PROCESSING")

    # Also transposing dimensions to PRES, TIME for ease of plotting etc in
    # MATLAB.
    matfile.xr_to_mat(ds_wo_proc.transpose(), outfile, simplify=simplify)


def check_metadata(ds: Union[xr.Dataset, str]) -> None:
    """
    Use the IOOS compliance checker to check an NetCDF file (CF and ACDD
    conventions).

    Parameters
    ----------
    ds : Union[xr.Dataset, str]
        The dataset or file path to check. Can be either an xarray Dataset or a
        file path.

    Displays the compliance check results with a "Close" button.
    """
    check_file_with_button(ds)
