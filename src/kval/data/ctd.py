"""
kval.ctd

"""

import xarray as xr
from kval.data.ship_ctd_tools import _ctd_tools as tools
from kval.data.ship_ctd_tools import _ctd_visualize as viz
from kval.data.ship_ctd_tools import _ctd_edit as ctd_edit
from kval.file import matfile
from kval.data import dataset, edit
from kval.util import time, xr_funcs
from kval.metadata import conventionalize, _standard_attrs
from kval.metadata.check_conventions import check_file_with_button, custom_checks
from kval.metadata.conventionalize import convert_64_to_32, add_now_as_date_created, nans_to_fill_value
from kval.metadata.io import import_metadata, export_metadata

import numpy as np
from pathlib import Path

# Want to be able to use these functions directly..
from kval.data.dataset import to_netcdf
from kval.data.edit import threshold, offset


# LOADING AND SAVING DATA

def ctds_from_cnv_dir(
    path: str,
    station_from_filename: bool = False,
    verbose: bool = False,
    start_time_NMEA: bool = False,
    profile: str = "downcast",
    processing_variable: bool = True,
    remove_duplicates: bool = True,
) -> xr.Dataset:
    """
    Create a joined CTD dataset from CNV files in the specified directory.

    Parameters
    ----------
    path : str
        Path to the directory containing CNV files.
    station_from_filename : bool, default=False
        Whether to extract station information from filenames.
    verbose : bool, default=False
        If False, suppress some printed output.
    start_time_NMEA : bool, default=False
        If True, get the `start_time` attribute from the "NMEA UTC (Time)" 
        header line. Otherwise, use the "start_time" line (may occasionally cause issues).
    profile : str, default='downcast'
        Specify the profile type (only relevant for unbinned input data). 
        Options: 'upcast', 'downcast', 'none'.
    processing_variable : bool, default=True
        Whether to include processing variables in the output dataset.
    remove_duplicates : bool, default=True
        Remove duplicate columns (identical names). If not removed, duplicate columns 
        will be assigned suffixes: DUPLICATE, DUPLICATE2, etc.

    Returns
    -------
    ds : xarray.Dataset
        The joined CTD dataset.
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

    return ds


def ctds_from_cnv_list(
    cnv_list: list[str],
    station_from_filename: bool = False,
    profile: str = "downcast",
    time_warnings: bool = True,
    verbose: bool = True,
    start_time_NMEA: bool = False,
    processing_variable: bool = True,
    remove_duplicates: bool = True,
) -> xr.Dataset:
    """
    Create a joined CTD dataset from a list of CNV files.

    Parameters
    ----------
    cnv_list : list[str]
        List of paths to CNV files.
    station_from_filename : bool, default=False
        Whether to extract station information from filenames.
    profile : str, default='downcast'
        Profile type (only relevant for unbinned input data). 
        Options: 'upcast', 'downcast', 'none'.
    time_warnings : bool, default=True
        Enable or disable time-related warnings.
    verbose : bool, default=True
        If False, suppress some printed output.
    start_time_NMEA : bool, default=False
        If True, get the `start_time` attribute from the "NMEA UTC (Time)" 
        header line. Otherwise, use the "start_time" line (may occasionally cause issues).
    processing_variable : bool, default=True
        Whether to include a processing history variable in the dataset.
    remove_duplicates : bool, default=True
        Remove duplicate columns (identical names). If not removed, duplicates 
        will be assigned suffixes: DUPLICATE, DUPLICATE2, etc.

    Returns
    -------
    ds : xarray.Dataset
        The joined CTD dataset.
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

    return ds


def dataset_from_btl_dir(
    path: str | Path,
    station_from_filename: bool = False,
    start_time_NMEA: bool = False,
    time_adjust_NMEA: bool = False,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Create a joined CTD dataset from BTL files in a specified directory.

    Parameters
    ----------
    path : str | Path
        Directory containing .btl files.
    station_from_filename : bool, default=False
        Extract station information from filenames if True.
    start_time_NMEA : bool, default=False
        Use the start time from the NMEA header line if True.
    time_adjust_NMEA : bool, default=False
        Adjust timestamps using NMEA data if True.
    verbose : bool, default=True
        If False, suppress some printed output.

    Returns
    -------
    xr.Dataset
        The joined CTD dataset.

    Raises
    ------
    FileNotFoundError
        If no .btl files are found in the specified path.
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



def from_netcdf(path_to_file: str | Path) -> xr.Dataset:
    """
    Load a netCDF file into an xarray Dataset.

    Skips CF decoding and preserves auxiliary coordinates.
    """
    ds = xr.open_dataset(path_to_file, decode_cf=False)
    ds = xr_funcs.promote_cf_coordinates(ds)
    return ds



def to_mat(ds: xr.Dataset, outfile: str, simplify: bool = False) -> None:
    """
    Convert a CTD xarray Dataset to a MATLAB .mat file.

    Adds a 'TIME_mat' field with MATLAB datenums.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to convert.
    outfile : str
        Output file path for the MATLAB .mat file. '.mat' will be appended if missing.
    simplify : bool, default=False
        If True, include only coordinates and data variables (no metadata). 
        If False, the mat file will include attrs, data_vars, coords, and dims.

    Returns
    -------
    None
        The function writes the dataset to the specified .mat file.

    Examples
    --------
    >>> to_mat(ds, 'output_matfile', simplify=True)
    """

    ds_cp = ds.copy(deep=True) # Make sure we're not modifying the input ds

    # Also transposing dimensions to PRES, TIME for ease of plotting etc
    # in MATLAB.
    matfile.xr_to_mat(ds_cp.transpose(), outfile, simplify=simplify)


def to_csv(ds: xr.Dataset, outfile: str) -> None:
    """
    Convert a CTD xarray Dataset to a human-readable CSV file.

    The CSV shows columnar data for all parameters and all stations.
    Stations are separated by a header containing station name, time, latitude, and longitude.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to convert.
    outfile : str
        Output CSV file path. '.csv' will be appended if missing.

    Returns
    -------
    None
        The function writes the dataset to the specified CSV file.

    Examples
    --------
    >>> to_csv(ds, 'output_cnvfile')
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

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

# (deleted old locatl functions (threshold, offset), now importing from universal ones in edit.py)

# APPLYING CORRECTIONS ETC

def calibrate_chl(
    ds: xr.Dataset,
    A: float,
    B: float,
    chl_name_in: str = "CHLA_fluorescence",
    chl_name_out: str = "CHLA",
    verbose: bool = True,
    remove_uncal: bool = False,
) -> xr.Dataset:
    """
    Calibrate chlorophyll based on a linear fit to water samples.

    Converts uncalibrated chlorophyll to calibrated chlorophyll using:

        CHLA = A * CHLA_fluorescence + B

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the chlorophyll variable to calibrate.
    A : float
        Linear coefficient from calibration.
    B : float
        Offset coefficient from calibration.
    chl_name_in : str, default='CHLA_fluorescence'
        Name of the uncalibrated chlorophyll variable. Will try 'CHLA1_fluorescence'
        if the specified name is not found.
    chl_name_out : str, default='CHLA'
        Name for the calibrated chlorophyll variable.
    verbose : bool, default=True
        Print messages about the calibration process if True.
    remove_uncal : bool, default=False
        Remove the uncalibrated variable from the dataset if True.

    Returns
    -------
    xr.Dataset
        Dataset with the calibrated chlorophyll variable added.

    Examples
    --------
    >>> ds_calibrated = calibrate_chl(ds, A=0.5, B=2,
    ...     chl_name_in='CHLA_fluorescence', chl_name_out='CHLA')
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

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

    note = f"Calibrated from {chl_name_in} using {chl_name_out} = {A} * {chl_name_in} + {B}."
    ds = xr_funcs.append_processing_history(ds, chl_name_out, note, deep_copy=False)

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


def metadata_auto(ds: xr.Dataset, NPI: bool = True) -> xr.Dataset:
    """
    Standardize and enrich metadata in a CTD xarray Dataset.

    Applies common conventions to variable and global attributes to prepare
    the dataset for publication or sharing. This includes renaming variables,
    adding standard attributes, and ensuring consistent metadata structure.

    NOTE: This should provide a good start, but you will still have to 
    manually work with metadata to get to CF/ACDD compliance!

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset whose metadata will be standardized.
    NPI : bool, default=True
        Not used; retained for API compatibility.

    Returns
    -------
    xr.Dataset
        Dataset with updated and standardized metadata.

    Notes
    -----
    This function calls multiple sub-functions to update the metadata:
    - `remove_numbers_in_var_names`: remove numbers from variable names
    - `add_standard_var_attrs`: add standard variable attributes
    - `add_standard_glob_attrs_ctd`: add CTD-specific global attributes
    - `add_standard_glob_attrs_org`: add organizational global attributes
    - `add_gmdc_keywords_ctd`: add GMDC keywords for CTD data
    - `add_range_attrs`: add range attributes
    - `reorder_attrs`: reorder attributes for consistency
    """
    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    ds = conventionalize.remove_numbers_in_var_names(ds)
    ds = conventionalize.add_standard_var_attrs(ds, data_type='ctdprof')
    ds = conventionalize.add_standard_glob_attrs_ctd(ds, override=False)
    ds = conventionalize.add_standard_glob_attrs_org(ds)
    ds = conventionalize.add_gmdc_keywords_ctd(ds)
    ds = conventionalize.add_range_attrs(ds)
    ds = conventionalize.reorder_attrs(ds)

    return ds


def drop_variables(
    ds: xr.Dataset,
    retain: list[str] | bool | None = None,
    drop: list[str] | None = None,
    verbose: bool = True,
    dims_to_check: list[str] = ["PRES", "NISKIN_NUMBER"],
) -> xr.Dataset:
    """
    Drop or retain variables from an xarray Dataset.

    Allows selective removal or retention of variables in a dataset. 
    If `retain_vars` is True, no variables are dropped. If `drop_vars` 
    is provided, it overrides `retain_vars`.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from which variables will be dropped or retained.
    retain : list[str] | bool | None, default=None
        List of variables to retain. If True, all variables are kept. Ignored
        if `drop_vars` is specified.
    drop : list[str] | None, default=None
        List of variables to drop. Overrides `retain_vars` if provided.
    verbose : bool, default=True
        If True, prints the list of dropped variables.
    dims_to_check : list[str], optional
        Dimensions to consider when dropping variables. Only variables that
        have at least one of these dimensions are eligible for dropping.
        Defaults to ["PRES", "NISKIN_NUMBER"].

    Returns
    -------
    xr.Dataset
        Dataset with specified variables dropped or retained.

    Notes
    -----
    Only one of `retain` or `drop` should be provided. 
    Variables without 'PRES' or 'NISKIN_NUMBER' dimensions are always retained.
    """

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    ds = edit.drop_variables(ds, retain=retain, drop=drop, verbose=verbose, 
                             dims_to_check= dims_to_check)

    return ds


# VISUALIZATION (WRAPPER FOR FUNCTIONS IN THE
# data.ship_ctd_tools._ctd_visualize.py MODULE)

def map(
    ds: xr.Dataset,
    station_labels: bool = False,
    station_label_alpha: float = 0.5,
) -> None:
    """
    Quick map of CTD stations from a cruise dataset.

    Plots latitude and longitude points from the dataset, showing the cruise track
    with red dots. Optionally displays station labels with adjustable transparency.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing `LATITUDE` and `LONGITUDE` variables.
    station_labels : bool, default=False
        Show labels for the stations if True.
    station_label_alpha : float, default=0.5
        Transparency of the station labels (0=transparent, 1=opaque).

    Returns
    -------
    None
        Displays the map directly.

    Notes
    -----
    Uses the `quickmap` module to generate a stereographic map with autoscaling 
    and grid lines. Interactive buttons allow resizing or closing the figure.

    Examples
    --------
    >>> map(ds)
    >>> map(ds, station_labels=True, station_label_alpha=0.7)
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
    >>> inspect_phase_space(ds)

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


############


def check_metadata(ds: xr.Dataset | str) -> None:
    """
    Check a dataset or NetCDF file for CF and ACDD compliance.

    Uses the IOOS compliance checker to validate conventions in a dataset 
    or a NetCDF file.

    Parameters
    ----------
    ds : xr.Dataset | str
        Dataset or path to a NetCDF file to be checked.

    Returns
    -------
    None
        Displays the compliance results interactively, including a "Close" button.
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
    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds
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
    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds
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

    Unlike most kval functions, this modifies `ds` in place (in addition
    to returning it), since the interactive editing session needs to act
    on the same object the user is clicking on.

    """

    # Deliberately skip using a deep copy - we need to change this object in place.
    #ds = ds.copy(deep=True) 

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
   
    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds
    
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

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds
    
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

    ds = ds.copy(deep=True) # Make sure we're not modifying the input ds

    edit_obj = edit.drop_vars_pick(ds)

    return edit_obj.ds


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
