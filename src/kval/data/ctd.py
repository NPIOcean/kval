"""
kval.ctd

--------------------------------------------------------------
A note about maintaining a metadata record of processing steps
--------------------------------------------------------------

We want to maintain a record in the file metadata of all operations
that modify the file in significant ways.

This is done by populating the variable attributes of the
PROCESSING variable of the dataset. Specifically:

- *ds.PROCESSING.post_processing* should contain an algorithmic
  description of steps that were applied. Should be human readable
  but contain all necessary details to reproduce the processing step.
- *ds.PROCESSING.python_script* should contain a python script
  reproducing the processing procedure. In cases where data are changed
  based on interactive user input (e.g. hand selecting points), the
  corresponding line of code in ds.PROCESSING.python_script should be
  a call to a corresponding non-interactive function performing the exact
  equivalent modifications to the data.

The preferred method of updating the these metadata attributes is using
the decorator function defined at the start of the script. The decorator
is defined below in record_processing(). An example of how it is used can
be found above the function metadata_auto().

In cases with interactive input, it is not always feasible to use the
decorator approach. In such cases, it may be necessary to update
ds.PROCESSING.post_processing and ds.PROCESSING.python_script
more directly.

"""

import xarray as xr
from kval.data.ship_ctd_tools import _ctd_tools as tools
from kval.data.ship_ctd_tools import _ctd_visualize as viz
from kval.data.ship_ctd_tools import _ctd_edit as ctd_edit
from kval.data.ship_ctd_tools._ctd_decorator import record_processing
from kval.file import matfile
from kval.data import dataset, edit
from kval.util import time, xr_funcs
from kval.metadata import conventionalize, _standard_attrs
from kval.metadata.check_conventions import check_file_with_button, custom_checks
from kval.metadata.conventionalize import convert_64_to_32, add_now_as_date_created
from kval.metadata.io import import_metadata

from typing import List, Optional, Union
import numpy as np
from pathlib import Path


# Want to be able to use these functions directly..
from kval.data.dataset import metadata_to_txt, to_netcdf

# DECORATOR TO PRESERVE PROCESSING STEPS IN METADATA


# LOADING AND SAVING DATA


def ctds_from_cnv_dir(
    path: str,
    station_from_filename: bool = False,
    verbose: bool = False,
    start_time_NMEA=False,
    profile="downcast",
    processing_variable=True,
    remove_duplicates=True,

) -> xr.Dataset:
    """
    Create CTD datasets from CNV files in the specified path.

    Parameters:
    - path (str): Path to the CNV files.
    - station_from_filename (bool): Whether to extract station information
                                    from filenames.
    - time_warnings (bool): Enable/disable time-related warnings.
    - verbose: If False, suppress some prints output.
    - start_time_NMEA (bool, optional)
      If True: get start_time attribute from the "NMEA UTC (Time)"
      header line. Default (False) is to grab it from the "start_time" line.
      (That seems to occasionally cause problems).
    - profile : str, optional
        Specify the profile type (only relevant for unbinned input data).
        Options are ['upcast', 'downcast', 'none'].
    - remove_duplicates : bool, optional
        Remove duplicate columns (identical name). If not removed,
        duplicate comumns will be assigned suffices, DUPLICATE,
        DUPLICATE2, etc. Default is True.

    Returns:
    - ds (xarray.Dataset): Joined CTD dataset.

    """
    cnv_files = tools._cnv_files_from_path(path)
    number_of_cnv_files = len(cnv_files)
    if number_of_cnv_files == 0:
        raise Exception(
            "Did not find any .cnv files in the specified "
            f'directory ("{path}"). Is there an error in the path?'
        )
    else:
        print(f'Found {number_of_cnv_files} .cnv files in  "{path}".')

    profile_datasets = tools._datasets_from_cnvlist(
        cnv_files,
        profile=profile,
        station_from_filename=station_from_filename,
        verbose=verbose,
        start_time_NMEA=start_time_NMEA,
        remove_duplicates=remove_duplicates,
    )

    ds = tools.join_cruise(profile_datasets, verbose=verbose)

    # Add PROCESSING variable
    if processing_variable:
        ds = dataset.add_processing_history_var_ctd(
            ds, source_file=np.sort(cnv_files)
        )
        ds.attrs["history"] = ds.history.replace(
            '"SBE_processing"', '"PROCESSING.SBE_processing"'
        )

        # Add python scipt snipped to reproduce this operation
        ds.PROCESSING.attrs[
            "python_script"
        ] += f"""from kval import data

# Path to directory containing *source_file* (MUST BE SET BY THE USER!)
cnv_dir = "./"

# Load all .cnv files and join together into a single xarray Dataset:
ds = data.ctd.ctds_from_cnv_dir(
    cnv_dir,
    station_from_filename={station_from_filename},
    start_time_NMEA={start_time_NMEA},
    processing_variable={processing_variable}
    )"""

    return ds


@record_processing(
    "Created CTD dataset from CNV list: {cnv_list}. Station info from "
    "filenames: {station_from_filename}. Time warnings: {time_warnings}. "
    "Start time from NMEA: {start_time_NMEA}. "
    "Processing variable: {processing_variable}.",
    "Loaded and combined CNV files from list into a single dataset.",
)
def ctds_from_cnv_list(
    cnv_list: list,
    station_from_filename: bool = False,
    profile="downcast",
    time_warnings: bool = True,
    verbose: bool = True,
    start_time_NMEA=False,
    processing_variable=True,
    remove_duplicates=True,

) -> xr.Dataset:
    """
    Create CTD datasets from CNV files in the specified list.

    Parameters:
    - cnv_list (list): List of CNV file paths.
    - station_from_filename (bool): Whether to extract station
                                    information from filenames.
    - time_warnings (bool): Enable/disable time-related warnings.
    - verbose: If False, suppress some prints output.
    - start_time_NMEA (bool, optional): If True, get start_time attribute from
      the "NMEA UTC (Time)" header line. Default is to grab it from the
      "start_time" line.
    - processing_variable (bool): Whether to add a processing history variable.
    - profile : str, optional
            Specify the profile type (only relevant for unbinned input data).
            Options are ['upcast', 'downcast', 'none'].
    - remove_duplicates : bool, optional
        Remove duplicate columns (identical name). If not removed,
        duplicate comumns will be assigned suffices, DUPLICATE,
        DUPLICATE2, etc. Default is True.

    Returns:
    - ds (xarray.Dataset): Joined CTD dataset.
    """
    profile_datasets = tools._datasets_from_cnvlist(
        cnv_list,
        verbose=verbose,
        profile=profile,
        start_time_NMEA=start_time_NMEA,
        station_from_filename=station_from_filename,
        remove_duplicates=remove_duplicates,
    )
    ds = tools.join_cruise(profile_datasets, verbose=verbose)

    # Add PROCESSING variable
    if processing_variable:
        ds = dataset.add_processing_history_var_ctd(
            ds, source_file=np.sort(cnv_list)
        )
        ds.attrs["history"] = ds.history.replace(
            '"SBE_processing"', '"PROCESSING.SBE_processing"'
        )

        # Add python script snippet to reproduce this operation
        ds.PROCESSING.attrs["python_script"] += (
            "from kval import data\n"
            "cnv_list = [{files}] # A list of strings specifying paths to all"
            " files in *source_file*.\n\n"
            "# Load all .cnv files and join together into a single xarray"
            " Dataset:\n"
            "ds = data.ctd.ctds_from_cnv_list(cnv_list,\n"
            f"    station_from_filename={station_from_filename},\n"
            f"    start_time_NMEA={start_time_NMEA},\n"
            f"    processing_variable={processing_variable})"
        )

    return ds


@record_processing(
    (
        "Created CTD dataset from BTL files in directory '{path}'. Station "
        "info from filenames: {station_from_filename}. Start time from NMEA: "
        "{start_time_NMEA}. Time adjust from NMEA: {time_adjust_NMEA}."
    ),
    "Loaded and combined BTL files from directory into a single dataset.",
)
def dataset_from_btl_dir(
    path: str | Path,
    station_from_filename: bool = False,
    start_time_NMEA: bool = False,
    time_adjust_NMEA: bool = False,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Create CTD datasets from BTL files in the specified path.

    Parameters:
        path (str or Path): Directory containing .btl files.
        station_from_filename (bool): Extract station info from filenames if True.
        start_time_NMEA (bool): Use start time from NMEA if True.
        time_adjust_NMEA (bool): Adjust time using NMEA data if True.
        verbose (bool): If False, suppress some printed output.

    Returns:
        xr.Dataset: Joined CTD dataset.

    Raises:
        FileNotFoundError: If no .btl files are found in the path.
    """
    path = Path(path)  # Ensure Path object

    btl_files = tools._btl_files_from_path(path)
    number_of_btl_files = len(btl_files)
    if number_of_btl_files == 0:
        raise FileNotFoundError(
            f'Did not find any .btl files in the specified directory ("{path}").'
        )
    if verbose:
        print(f'Found {number_of_btl_files} .btl files in "{path}".')

    profile_datasets = tools._datasets_from_btllist(
        btl_files,
        verbose=verbose,
        start_time_NMEA=start_time_NMEA,
        time_adjust_NMEA=time_adjust_NMEA,
        station_from_filename=station_from_filename,
    )

    ds = tools.join_cruise_btl(profile_datasets, verbose=verbose)
    ds = ds.transpose()

    return ds



def from_netcdf(path_to_file):
    """
    Import a netCDF file - e.g. one previously generated
    with these tools.

    Skipping cf decoding, and re-promoting aux coordinates
    (doesn't matter but nice for clarity).
    """
    ds = xr.open_dataset(path_to_file, decode_cf=False)
    ds = xr_funcs.promote_cf_coordinates(ds)
    return ds


# No need to document this!
#@record_processing(
#    "Converted dataset to MATLAB .mat file '{outfile}'. Simplify: {simplify}.",
#    "Converted dataset to MATLAB .mat file '{outfile}' with "
#    "simplify={simplify}.",
#)
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
    >>> ctd.xr_to_mat(ds, 'output_matfile', simplify=True)
    """
    # Drop the empty PROCESSING variable (doesn't work well with MATLAB)
    if "PROCESSING" in ds:
        ds_wo_proc = drop_variables(ds, drop_vars="PROCESSING")
    else:
        ds_wo_proc = ds

    # Also transposing dimensions to PRES, TIME for ease of plotting etc
    # in MATLAB.
    matfile.xr_to_mat(ds_wo_proc.transpose(), outfile, simplify=simplify)


@record_processing(
    "Converted dataset to CSV file '{outfile}'.",
    "Converted dataset to CSV file '{outfile}'.",
)
def to_csv(ds, outfile):
    """
    Convert the CTD data (xarray.Dataset) to a human-readable .csv file.

    The file shows columnar data for all data parameters for all stations.
    Stations are separated by a header with the station name/time/lat/lon.

    Parameters:
    - ds (xarray.Dataset): Input dataset to be converted.
    - outfile (str): Output file path for the .csv file. If the path
      doesn't end with '.csv', it will be appended.

    Returns:
    None: The function saves the dataset as a .csv file.

    Example:
    >>> ctd.to_csv(ds, 'output_cnvfile')
    """
    prof_vars = ["PRES"]

    for key in ds.data_vars.keys():
        if "TIME" in ds[key].dims:
            if "PRES" in ds[key].dims:
                prof_vars += [key]

    if not outfile.endswith(".csv"):
        outfile += ".csv"

    with open(outfile, "w") as f:
        for time_ in ds.TIME.values:
            ds_prof = ds.sel(TIME=time_)
            time_str = time.datenum_to_timestamp(time_).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            print("#" * 88, file=f)
            print(
                f"#####  {ds_prof.STATION.values:<8} ###  {time_str}  "
                f"###  LAT: {ds_prof.LATITUDE.values:<10}"
                f" ### LON: {ds_prof.LONGITUDE.values:<10} #####",
                file=f,
            )
            print("#" * 88 + "\n", file=f)

            ds_pd = ds_prof[prof_vars].to_pandas()
            ds_pd = ds_pd.drop("TIME", axis=1)

            ds_pd = ds_pd.dropna(
                subset=ds_pd.columns.difference(["PRES"]), how="all"
            )
            print(ds_pd.to_csv(), file=f)


# MODIFYING DATA


@record_processing(
    "Rejected values of {variable} outside the range ({min_val}, {max_val})",
    py_comment="Rejecting values of {variable} outside the range "
    "({min_val}, {max_val}):",
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


@record_processing(
    "Applied offset ={offset} to the variable {variable}.",
    py_comment="Applied offset {offset} to variable {variable}:",
)
def offset(ds: xr.Dataset, variable: str, offset: float) -> xr.Dataset:
    """
    Apply a fixed offset to a specified variable in an xarray Dataset.

    This function modifies the values of the specified variable by adding a
    fixed offset to them. The `valid_min` and `valid_max` attributes are
    updated to reflect the new range of values after applying the offset.

    Parameters
    ----------
    ds : xr.Dataset
        The input xarray Dataset.
    variable : str
        The name of the variable within the Dataset to which the offset
        will be applied.
    offset : float
        The fixed offset value to add to the variable.

    Returns
    -------
    xr.Dataset
        A new xarray Dataset with the offset applied to the specified
        variable. The `valid_min` and `valid_max` attributes are updated
        accordingly.

    Examples
    --------
    # Apply an offset of 5 to the 'TEMP' variable
    ds_offset = offset(ds, 'TEMP', offset=5)
    """
    ds = edit.offset(ds=ds, variable=variable, offset=offset)
    return ds


# APPLYING CORRECTIONS ETC


@record_processing(
    "Applied a calibration to chlorophyll: "
    "{chl_name_out} = {A} * {chl_name_in} + {B}.",
    py_comment="Applying chlorophyll calibration based on fit to lab values:",
)
def calibrate_chl(
    ds: xr.Dataset,
    A: float,
    B: float,
    chl_name_in: Optional[str] = "CHLA_fluorescence",
    chl_name_out: Optional[str] = "CHLA",
    verbose: Optional[bool] = True,
    remove_uncal: Optional[bool] = False,
) -> xr.Dataset:
    """
    Apply a calibration to chlorophyll based on a fit to water samples.

    Converts uncalibrated chlorophyll (CHLA_fluorescence) to calibrated
    chlorophyll (CHLA) using the formula:

    CHLA = A * CHLA_fluorescence + B

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing A, B: Linear coefficients based on fitting to chl
        samples.
    A : float
        Linear coefficient for calibration.
    B : float
        Linear coefficient for calibration.
    chl_name_in : str, optional
        Name of the variable containing uncalibrated chlorophyll from the
        instrument. Default is 'CHLA_fluorescence'. Will look for
        'CHLA1_fluorescence' if that doesn't exist.
    chl_name_out : str, optional
        Name of the calibrated chlorophyll variable. If not provided, it is
        derived from chl_name_in. Default is 'CHLA'.
    verbose : bool, optional
        If True, print messages about the calibration process. Default is True.
    remove_uncal : bool, optional
        If True, remove the uncalibrated chlorophyll from the dataset.
        Default is False.

    Returns
    -------
    xr.Dataset
        Updated dataset with the calibrated chlorophyll variable.

    Examples
    --------
    # Apply calibration with coefficients A=0.5 and B=2
    ds_calibrated = calibrate_chl(ds, A=0.5, B=2,
          chl_name_in='CHLA_fluorescence', chl_name_out='CHLA')
    """

    # Determine the input variable name
    if chl_name_in not in ds:
        if "CHLA1_fluorescence" in ds:
            chl_name_in = "CHLA1_fluorescence"
        else:
            raise Exception(
                f'Did not find {chl_name_in} or "CHLA1_fluorescence" '
                "in the dataset. Please specify the variable name of "
                "uncalibrated chlorophyll using the *chl_name_in* flag."
            )

    # Determine the output variable name for calibrated chlorophyll
    if not chl_name_out:
        if "_instr" in chl_name_in or "_fluorescence" in chl_name_in:
            chl_name_out = chl_name_in.replace("_instr", "").replace(
                "_fluorescence", ""
            )
        else:
            chl_name_out = f"{chl_name_in}_cal"

    # Create a new variable with the coefficients applied
    ds[chl_name_out] = A * ds[chl_name_in] + B
    ds[chl_name_out].attrs = {
        key: item for key, item in ds[chl_name_in].attrs.items()
    }

    # Add suitable attributes
    new_attrs = {
        "long_name": ("Chlorophyll-A concentration calibrated "
                      "against water sample measurements"),
        "calibration_formula": f"{chl_name_out} = {A} * {chl_name_in} + {B}",
        "coefficient_A": A,
        "coefficient_B": B,
        "comment": (
            "No correction for near-surface fluorescence quenching "
            "(see e.g. https://doi.org/10.4319/lom.2012.10.483) "
            "has been applied."
        ),
        "processing_level": "Post-recovery calibrations have been applied",
        "QC_indicator": "good data",
    }

    for key, item in new_attrs.items():
        ds[chl_name_out].attrs[key] = item

    # Remove the uncalibrated chl
    if remove_uncal:
        remove_str = (
            f' Removed uncalibrated Chl-A ("{chl_name_in}") from the dataset.'
        )
        ds = ds.drop_vars(chl_name_in)
    else:
        remove_str = ""

    # Print
    if verbose:
        print(
            f'Added calibrated Chl-A ("{chl_name_out}") calculated from'
            f' variable "{chl_name_in}".{remove_str}'
        )

    return ds


# MODIFYING METADATA


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
    - `remove_numbers_in_var_names`:
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
    ds = conventionalize.add_standard_var_attrs(ds, data_type='ctdprof')
    ds = conventionalize.add_standard_glob_attrs_ctd(ds, override=False)
    ds = conventionalize.add_standard_glob_attrs_org(ds)
    ds = conventionalize.add_gmdc_keywords_ctd(ds)
    ds = conventionalize.add_range_attrs(ds)
    ds = conventionalize.reorder_attrs(ds)

    return ds


# Note: Doing PROCESSING.post_processing record keeping within the
# drop_variables() function because we want to access the *dropped* list.
@record_processing("", py_comment="Dropping some variables")
def drop_variables(
    ds: xr.Dataset,
    retain_vars: Optional[Union[List[str], bool]] = None,
    drop_vars: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Drop measurement variables from the dataset based on specified criteria.

    This function retains or drops variables from an xarray.Dataset based on
    provided lists of variables to retain or drop. If `retain_vars` is True, no
    variables will be dropped.

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

    Returns
    -------
    xr.Dataset
        The modified dataset with specified variables dropped or retained.

    Notes
    -----
    Provide *either* `retain_vars` or `drop_vars`, but not both. Variables that
    do not have a 'PRES' or 'NISKIN_NUMBER' dimension will always be retained.
    """
    if retain_vars is None and drop_vars is None:
        return ds

    if drop_vars is not None:
        ds = ds.drop_vars(drop_vars)
        dropped = drop_vars
    else:
        if retain_vars is None:
            raise ValueError(
                "Either `drop_vars` or `retain_vars` must be specified,"
                " not both."
            )

        if isinstance(retain_vars, bool):
            if retain_vars:
                return ds
            retain_vars = []

        all_vars = list(ds.data_vars)
        dropped = []
        for varnm in all_vars:
            if varnm not in retain_vars and (
                "PRES" in ds[varnm].dims or "NISKIN_NUMBER" in ds[varnm].dims
            ):
                ds = ds.drop_vars(varnm)
                dropped.append(varnm)

    if dropped:
        drop_str = f"Dropped these variables from the Dataset: {dropped}."
        print(drop_str)
        if "PROCESSING" in ds:
            ds["PROCESSING"].attrs["post_processing"] += f"{drop_str}\n"

    return ds


# VISUALIZATION (WRAPPER FOR FUNCTIONS IN THE
# data.ship_ctd_tools._ctd_visualize.py MODULE)

def map(
    ds: xr.Dataset,
    station_labels: bool = False,
    station_label_alpha: float = 0.5,
) -> None:
    """
    Generate a quick map of the cruise CTD stations.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing latitude (`LATITUDE`) and longitude
        (`LONGITUDE`).
    station_labels : bool, optional
        Whether to display labels for the stations on the map. Default is
        False.
    station_label_alpha : float, optional
        The transparency level of the station labels, between 0 and 1. Default
        is 0.5.

    Displays a quick map using the provided xarray Dataset with latitude and
    longitude information. The map includes a plot of the cruise track and red
    dots at data points.

    Additionally, the function provides interactive buttons: - "Close"
    minimizes and closes the plot. - "Original Size" restores the plot to its
    original size. - "Larger" increases the plot size.

    Examples
    --------
    >>> map(ds)
    >>> fig, ax = map(ds, station_labels=True, station_label_alpha=0.7)

    Notes
    -----
    This function utilizes the `quickmap` module for generating a stereographic
    map. It is designed to come up with reasonable autoscaling and produce grid
    lines.
    """
    viz.map(
        ds,
        station_labels=station_labels,
        station_label_alpha=station_label_alpha,
    )


def inspect_profiles(ds: xr.Dataset) -> None:
    """
    Interactively inspect individual CTD profiles in an xarray dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray dataset containing variables such as `PRES`, `STATION`, and
        other profile variables.

    This function creates an interactive plot that allows users to explore
    profiles within the given xarray dataset. It displays a slider to choose a
    profile by its index, a dropdown menu to select a variable for
    visualization, and another dropdown to pick a specific station. The
    selected profile is highlighted in color, while others are shown in the
    background.

    Examples
    --------
    >>> inspect_profiles(ds)

    Notes
    -----
    This function utilizes Matplotlib for plotting and ipywidgets for
    interactive controls.
    """
    viz.inspect_profiles(ds)

def inspect_phase_space(ds: xr.Dataset) -> None:
    """
    Interactively inspect phase space plots of two variables in an xarray dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray dataset containing variables such as `PRES`, `STATION`, and
        other profile variables.

    This function creates an interactive plot that allows users to explore
    profiles within the given xarray dataset. It displays a slider to choose a
    profile by its index, a dropdown menu to select a variable for
    visualization, and another dropdown to pick a specific station. The
    selected profile is highlighted in color, while others are shown in the
    background.

    Examples
    --------
    >>> inspect_phase_sapce(ds)

    Notes
    -----
    This function utilizes Matplotlib for plotting and ipywidgets for
    interactive controls.
    """
    viz.inspect_phase_space(ds)


def inspect_dual_sensors(ds: xr.Dataset) -> None:
    """
    Interactively inspect profiles of sensor pairs (e.g., PSAL1 and PSAL2).

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the variables for dual sensors.

    Usage
    -----
    Call `inspect_dual_sensors(ds)` to interactively inspect profiles of sensor
    pairs.
    """
    viz.inspect_dual_sensors(ds)


def contour(ds: xr.Dataset) -> None:
    """
    Create interactive contour plots based on an xarray dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray dataset containing profile variables and coordinates.

    This function generates interactive contour plots for two selected profile
    variables from the given xarray dataset. It provides dropdown menus to
    choose the variables, select the x-axis variable (e.g., 'TIME',
    'LONGITUDE', 'LATITUDE', 'Profile #'), and set the maximum depth for the
    y-axis.

    Additionally, the function includes a button to close the plot.

    Examples
    --------
    >>> contour(ds)

    Notes
    -----
    This function uses Matplotlib for creating contour plots and ipywidgets for
    interactive elements.
    """
    viz.ctd_contours(ds)


# INSPECTING METADATA


def quick_metadata_check(ds: xr.Dataset) -> None:
    """
    Perform a quick metadata check on the dataset.

    This function checks for the presence of required global and variable
    attributes. It is a preliminary check; a more comprehensive verification is
    done on export to NetCDF.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray dataset to check.

    Notes
    -----
    This function is intended for CTD-specific datasets. Consider moving it to
    `conventionalize.py` if it is generalized.
    """
    print("--  QUICK METADATA CHECK --")
    print("NOTE: Not comprehensive! A true check is done on export to NetCDF.")

    print("\n# GLOBAL #")

    # Global attributes
    attrs_dict_ref = _standard_attrs.global_attrs_ordered.copy()
    attrs_dict_ref.remove("date_created")
    attrs_dict_ref.remove("processing_level")

    for attr in attrs_dict_ref:
        if attr not in ds.attrs:
            print(f"- Possibly missing {attr}")

    print("\n# VARIABLE #")

    # Variable attributes
    attrs_dict_ref_var = _standard_attrs.variable_attrs_necessary

    for varnm in ds.variables:
        if "PRES" in ds[varnm].dims:
            _attrs_dict_ref_var = attrs_dict_ref_var.copy()

            if varnm == "CHLA":
                _attrs_dict_ref_var += [
                    "calibration_formula",
                    "coefficient_A",
                    "coefficient_B",
                ]
            if varnm == "PRES":
                _attrs_dict_ref_var += [
                    "axis",
                    "positive",
                ]
                _attrs_dict_ref_var.remove("processing_level")
                _attrs_dict_ref_var.remove("QC_indicator")

            any_missing = False
            for var_attr in _attrs_dict_ref_var:
                if var_attr not in ds[varnm].attrs:
                    print(f"- {varnm}: Possibly missing {var_attr}")
                    any_missing = True
            if not any_missing:
                print(f"- {varnm}: OK")


############


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


############

# SMALL FUNCTIONS FOR MODIFYING METADATA ETC

# Consider moving some (all?) of these to nc_attrs.conventionalize?

def set_attr_glob(ds: xr.Dataset, attr: str) -> xr.Dataset:
    """
    Set a global attribute (metadata) for the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to modify.
    attr : str
        The global attribute name (e.g., "title").

    Returns
    -------
    xr.Dataset
        The updated dataset with the global attribute set.

    Examples
    --------
    >>> ds = set_attr_glob(ds, 'title')
    """
    ds = conventionalize.set_glob_attr(ds, attr)
    return ds


def set_attr_var(ds: xr.Dataset, variable: str, attr: str) -> xr.Dataset:
    """
    Set a variable attribute (metadata) for a specific variable in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to modify.
    variable : str
        The variable name for which the attribute will be set (e.g., "PRES").
    attr : str
        The attribute name (e.g., "long_name").

    Returns
    -------
    xr.Dataset
        The updated dataset with the variable attribute set.

    Examples
    --------
    >>> ds = set_attr_var(ds, 'TEMP1', 'units')
    """
    ds = conventionalize.set_var_attr(ds, variable, attr)
    return ds


# EDITING
# (Wrappers for functions in the data.edit and data.ship_ctd_tools._ctd_edit.py
#  module)


def hand_remove_points(
    ds: xr.Dataset, variable: str, TIME_index: str
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
        The index of the station/profile number, i.e. index along the TIME
        dimension (starting at 0).

    Returns
    -------
    xr.Dataset
        The dataset with data points removed based on interactive input.

    Examples
    --------
    >>> ds = hand_remove_points(ds, 'TEMP1', 11)

    Notes
    -----
    Use the interactive plot to select points for removal, then click the
    corresponding buttons for actions.
    """

    hand_remove = ctd_edit.hand_remove_points(ds, variable, TIME_index)
    ds = hand_remove.d

    return ds


def apply_threshold(ds: xr.Dataset) -> xr.Dataset:
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

    variables = tools._get_profile_variables(ds)
    edit.threshold_edit(ds, variables=variables)
    return ds


def apply_offset(ds: xr.Dataset) -> xr.Dataset:
    """
    Apply an offset to a selected variable in a given xarray CTD Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The CTD dataset to which the offset will be applied.

    Returns
    -------
    xr.Dataset
        The dataset with the offset applied.

    Examples
    --------
    >>> ds = apply_offset(my_dataset)

    Notes
    -----
    Utilizes IPython widgets for interactive use within a Jupyter environment.
    """
    ctd_edit.apply_offset(ds)
    return ds


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
    edit_obj = edit.drop_vars_pick(ds)
    return edit_obj.D


# TABLED/UNFINISHED/COULD PERHAPS BECOME USEFUL

if False:

    def _drop_stations_pick(ds: xr.Dataset) -> xr.Dataset:
        """
        UNFINISHED! Tabled for fixing.

        Interactive class for dropping selected time points from an xarray
        Dataset based on the value of STATION(TIME).

        Parameters
        ----------
        ds : xr.Dataset
            The dataset from which time points will be dropped.

        Returns
        -------
        xr.Dataset
            The dataset with selected time points removed.

        Notes
        -----
        Displays an interactive widget with checkboxes for each time point,
        showing the associated STATION. Users can select time points to remove.
        The removal is performed by clicking the "Drop time points" button.
        """
        edit_obj = ctd_edit.drop_stations_pick(ds)
        return edit_obj.D
