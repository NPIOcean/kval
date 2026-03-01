"""
### KVAL.IO.SBE.py ###

Parsing data from seabird format (.cnv, .hdr, .btl) to xarray Datasets.

Some of these functions are rather long and clunky. This is mainly because the
input files are clunky and the format changes quite a lot.

Key functions
-------------

read_cnv:
  Loading single profiles. Reads CTD data and metadata from a .cnv file into a
  Dataset (-> netCDF file) with any potentially useful metadata we can extract
  from the .cnv header.

read_btl:
  Similar: Load single .btl files.

join_cruise:
  Joing several profiles (e.g. from an entire cruise) into a single Dataset (->
  netCDF file).

read_header:
  Parse useful information from a .cnv header into a dictionary. Mainly used
  within read_cnv, but may have its additional uses.

May eventually want to look into the organization: -> Do all these things
belong here or elsewhere? -> This is a massive script, should it be split up?

TBD: - Go through documentation and PEP8 formatting
    - Pretty decent already
- Consider usign the *logging module*

"""

# IMPORTS

import pandas as pd
import xarray as xr
import numpy as np
from kval.file import _variable_defs as vardef
from kval.util import time, xr_funcs
from kval.metadata.conventionalize import remove_numbers_in_var_names
from kval.data import dataset

import matplotlib.pyplot as plt
import re
import os
from typing import Optional
from itertools import zip_longest
from matplotlib.dates import num2date
import warnings

# KEY FUNCTIONS


def read_cnv(
    source_file: str,
    apply_flags: Optional[bool] = True,
    profile: Optional[str] = "downcast",
    remove_surface_soak: Optional[bool] = True,
    time_dim: Optional[bool] = False,
    inspect_plot: Optional[bool] = False,
    start_scan: Optional[int] = None,
    end_scan: Optional[int] = None,
    suppress_time_warning: Optional[bool] = False,
    suppress_latlon_warning: Optional[bool] = False,
    start_time_NMEA: Optional[bool] = False,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    station: Optional[str] = None,
    station_from_filename: Optional[bool] = False,
    remove_duplicates: Optional[bool] = True,

) -> xr.Dataset:
    """
        Reads CTD data and metadata from a .cnv file into a more handy format.
        (I.e. an xarray Dataframe with any potentially useful metadata we can
        extract from the .cnv header.)

        -> This does not mean that the profile is CF/ACDD compliant (we will
           need more metadata not found in the file for that). This should be a
           very good start, however.

        -> Profiles are assigned the coordinates ['scan_count'], or, if
           time_dim is set to True, ['scan_count', 'TIME'], where TIME is
           one-dimensional.

        -> Profiles created using this function can be joined together using
           the join_cruise() function. For this, we need to set time_dim=True
           when using read_cnv().

        Parameters
        ----------
        source_file : str
            Path to a .cnv file containing the data

        apply_flags : bool, optional
            If True, flags (from the SBE *flag* column) are applied as NaNs
            across all variables (recommended). Default is True.

        profile : str, optional
            Specify the profile type. Options are ['upcast', 'downcast',
            'none'].

        time_dim : bool, optional
            Choose whether to include a 0-D TIME coordinate. Useful if
            combining several profiles. Default is False,.

        inspect_plot : bool, optional
            Return a plot of the whole pressure time series, showing the part
            of the profile we extracted (useful for inspecting up/downcast
            extraction and SBE flags). Default is False.

        start_scan : int, optional
            Manually specify the scan at which to start the profile (in
            *addition* to profile detection and flags). Default is None.

        end_scan : int, optional
            Manually specify the scan at which to end the profile (in
            *addition* to profile detection and flags). Default is None.

        suppress_time_warning : bool, optional
            Don't show a warning if there are no timeJ or timeS fields. Default
            is False.

        suppress_latlon_warning : bool, optional
            Don't show a warning if there is no lat/lon information. Default is
            False. some
        station_from_filename : bool, optional
            Option to read the station name from the file name, e.g. "STA001"
            from a file "STA001.cnv". Otherwise, we try to grab it from the
            header. Default is False.
        remove_duplicates : bool, optional
            Remove duplicate columns (identical name). If not removed,
            duplicate comumns will be assigned suffices, DUPLICATE,
            DUPLICATE2, etc. Default is True.

    """
    # Parse useful information from the file header
    header_info = read_header(source_file)
    _is_moored = header_info["moored_sensor"]

    # Not looking for down/upcast if this is a moored sensor..
    if _is_moored:
        profile = "none"

    # Read the columnar data to an xarray Dataset
    ds = _read_column_data_xr(source_file, header_info)

    # Remove duplicate varaiables
    if any("_DUPLICATE" in var for var in ds.data_vars):
        if remove_duplicates:
            ds = _remove_duplicate_variables(ds)
            remove_dup_str = 'removing duplicates'
        else:
            remove_dup_str = 'preserving duplicates'

        print(f'Note: Duplicate variables found ({remove_dup_str}).')


    # Update variable names (e.g. t090C -> TEMP1)
    ds = _update_variables(ds, source_file, _is_moored)

    # Assign lat/lon/statin if specified in the function call
    ds = _assign_specified_lat_lon_station(ds, lat, lon, station)

    # Parse time from "timeJ" or "timeS" fields
    ds = _convert_time(
        ds,
        header_info,
        suppress_time_warning=suppress_time_warning,
        start_time_NMEA=start_time_NMEA,
    )

    # Start a history attribute tracking the post-processing history
    ds.attrs["history"] = header_info["start_history"]

    # Add various attributes read from the header
    ds = _add_header_attrs(ds, header_info, station_from_filename)

    # Add a start_time attribute
    ds = _add_start_time(ds, header_info, start_time_NMEA=start_time_NMEA)

    # Parse the SBE processing steps to a human-readble string
    try:
        ds = _read_SBE_proc_steps(ds, header_info, _is_moored)
    except Exception as err:
        ds.attrs["SBE_processing"] = (
            f"Unable to parse from file.\n(Error: {err})."
        )
        raise Warning(f"Unable to parse from file.\n(Error: {err}).")

    # Create a copy of the dataset before we apply any flags
    ds0 = ds.copy()

    # Apply the flags specified in the "flag" column of the cnv file
    if apply_flags:
        ds = _apply_flag(ds)
        ds = _apply_flag_variables(
            ds,
            bad_flag_value=header_info["bad_flag_value"])
        ds.attrs["SBE_flags_applied"] = "yes"

    else:
        ds.attrs["SBE_flags_applied"] = "no"


    # Add a time dimension to a profile dataset
    if time_dim and not _is_moored:
        ds = _add_time_dim_profile(
            ds, suppress_latlon_warning=suppress_latlon_warning
        )

    # Isolate up- or downcast (before/after pressure maximum)
    if profile in ["upcast", "downcast", "dncast"]:
        if ds.binned == "no":
            ds = _remove_up_dncast(ds, keep=profile)
    else:
        ds.attrs["profile_extracted"] = "All available good data"

    # If we have specified a start/end scan: Isolate only the desired range
    if start_scan:
        ds = _remove_start_scan(ds, start_scan)
    if end_scan:
        ds = _remove_end_scan(ds, end_scan)

    # Plot the pressure track, showing which data were flagged
    if inspect_plot:
        _inspect_extracted(ds, ds0, start_scan, end_scan)

    # Set e.g. TEMP1 -> TEMP if there is no TEMP2
    ds = remove_numbers_in_var_names(ds)

    # Some custom modifications for moored sensors
    if _is_moored:
        ds = _modify_moored(ds, header_info)

    # For SBE56s with a slightly unusual format: custom
    # paring of calibtaion date etc
    if 'sbe56' in ds.instrument_model.lower():
       ds = _add_meta_info_sbe56_cnv(ds, source_file)

    # Record the start of post-processing as the now time stamp
    now_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    ds.attrs["history"] += f"\n{now_str}: Post-processing."

    return ds


def read_btl(
    source_file,
    verbose=False,
    time_dim=True,
    station_from_filename=False,
    start_time_NMEA=True,
    time_adjust_NMEA=False,
):
    """
    time_adjust_NMEA: Use this if the "start time" in the file
    header is incorrect (occasionally seems to be reset to 2000-1-1)
    """

    # Parse the header
    header_info = read_header(source_file)

    # Parse the columnar data
    ds = _read_btl_column_data_xr(source_file, header_info, verbose=verbose)
    # Add nice variable names and attributes
    ds = _update_variables(ds, source_file)
    # Add a history string
    ds.attrs["history"] = header_info["start_history"]
    # Add ship, cruise, station, latitude, longitude attributes from the header
    ds = _add_header_attrs(
        ds, header_info, station_from_filename=station_from_filename
    )
    # Add a start time attribute from the header
    ds = _add_start_time(ds, header_info, start_time_NMEA=start_time_NMEA)
    # Add the SBE processing string from the header
    ds = _read_SBE_proc_steps(ds, header_info)
    # Add a (0-D) TIME dimension
    time_dim = True
    if time_dim:

        ds = _add_time_dim_profile(ds)

        if time_adjust_NMEA:

            # Calculate the difference bewteen nmae and start time in days
            diff_days = (
                header_info["NMEA_time"] - header_info["start_time"]
            ).total_seconds() / (24 * 3600)

            # Hold on to variable attributes
            time_attrs = ds.TIME.attrs

            # Apply the offset to TIME
            ds = ds.assign_coords(TIME=ds.TIME.values + diff_days)

            # Reapply variable attributes
            ds.TIME.attrs = time_attrs

    return ds


def read_header(filename: str) -> dict:
    """
    Reads a SBE .cnv (or .hdr, .btl) file and returns a dictionary with various
    metadata parameters extracted from the header.

    NOTE: This is a very bulky and ad-hoc function - but it does seem to
          do a reasonable job in the test cases tried so far.

    NOTE: Only tested for .cnv and .btl.

    Parameters:
    ----------
    source_file : str
        The path to the SBE .cnv, .hdr, or .btl file.

    Returns:
    -------
    dict
        A dictionary containing various metadata parameters extracted from the
        header.
    """

    with open(filename, "r", encoding="latin-1") as f:

        # Empty dictionary: Will fill these parameters up as we go
        hkeys = [
            "col_nums",
            "col_names",
            "col_longnames",
            "SN_info",
            "moon_pool",
            "SBEproc_hist",
            "hdr_end_line",
            "latitude",
            "longitude",
            "NMEA_time",
            "start_time",
            "start_history",
            "bad_flag_value",
        ]
        hdict = {hkey: [] for hkey in hkeys}

        # Flag that will be turned on when we read the SBE history section
        start_read_history = False

        # Go through the header line by line and extract specific information
        # when we encounter specific terms dictated by the format

        lines = f.readlines()
        # return lines
        for n_line, line in enumerate(lines):

            # Read the instrument type (usually the first line)
            if "Data File:" in line:
                hdict["instrument_model"] = " ".join(line.split()[1:-2])

                # If this is a SBE37 or SBE56, we will assume that
                # this is a moored sensor.
                if "SBE37" in line or "SBE56" in line or "SBE16" in line:
                    hdict["moored_sensor"] = True

            # Read the instrument type and SN from an alternative header
            # structure which seems common for SBE56s

            if "StatusData" in line:

                # Define a regex pattern to extract DeviceType and SerialNumber
                pattern = r"DeviceType='([^']+)'\s+SerialNumber='([^']+)'"

                # Search for the pattern in the line
                match = re.search(pattern, line)

                if match and "instrument_model" not in hdict:
                    hdict["instrument_model"] = match.group(1)
                if match and "instrument_serial_number" not in hdict:
                    serial_no = match.group(2)
                    # Remove some instrument info that is occasionally included
                    # in the serial number
                    for instr_key in ['0561', '037', '016']:
                        if serial_no.startswith(instr_key):
                            serial_no = serial_no[len(instr_key):]

                    # Remove leading zeros
                    serial_no = str(int(serial_no))

                    hdict["instrument_serial_number"] = serial_no


                # If this is a SBE37 or SBE56, we will assume that
                # this is a moored sensor.
                if "SBE37" in line or "SBE56" in line or "SBE16" in line:
                    hdict["moored_sensor"] = True

            # Read the column header info (which variable is in which data
            # column)
            if "# name" in line and not filename.endswith(".btl"):
                # Read column number
                hdict["col_nums"] += [int(line.split()[2])]
                # Read column header
                col_name = line.split()[4].replace(":", "")
                col_name = col_name.replace(
                    "/", "_"
                )  # "/" in varnames not allowed in netcdf
                hdict["col_names"] += [col_name]
                # Read column longname
                hdict["col_longnames"] += [" ".join(line.split()[5:])]

            # Read sample interval
            if "* sample interval =" in line:

                hdict["sample_interval"] = (
                    line[(line.rfind("interval = ") + 10):]
                    .replace("\n", "")
                    .strip()
                )

            # Read NMEA lat/lon/time
            if "NMEA Latitude" in line:
                hdict["latitude"] = _nmea_lat_to_decdeg(*line.split()[-3:])
            if "NMEA Longitude" in line:
                hdict["longitude"] = _nmea_lon_to_decdeg(*line.split()[-3:])
            if "NMEA UTC" in line:
                nmea_time_split = line.split()[-4:]
                hdict["NMEA_time"] = _nmea_time_to_datetime(*nmea_time_split)

            # If no NMEA lat/lon: Look for "** Latitude/Longitude"
            # (for some poorly structured .cnv files)
            if "** LATITUDE" in line.upper() and isinstance(
                hdict["latitude"], list
            ):
                if len(hdict["latitude"]) == 0:
                    lat_value = _decdeg_from_line(line)
                    if lat_value:
                        hdict["latitude"] = lat_value

            if "** LONGITUDE" in line.upper() and isinstance(
                hdict["longitude"], list
            ):
                if len(hdict["longitude"]) == 0:
                    lon_value = _decdeg_from_line(line)
                    if lon_value:
                        hdict["longitude"] = lon_value

            # Read start time
            if "start_time" in line:
                hdict["start_time"] = _nmea_time_to_datetime(
                    *line.split()[3:7]
                )

                hdict["start_history"] = (
                    hdict["start_time"].strftime("%Y-%m-%d")
                    + ": Data collection."
                )

            # Read bad_flag value
            if "# bad_flag" in line:

                match = re.search(r"=\s*([-\d.eE]+)", line)
                if match:
                    hdict['bad_flag_value']= float(match.group(1))
                else:
                    print('Had trouble reading the `bad_flag`'
                          ' value - assigning nan')
                    hdict['bad_flag_value'] = np.nan

            # Read cruise/ship/station/bottom depth/operator if available
            if "** CRUISE" in line.upper():
                hdict["cruise_name"] = (
                    line[(line.rfind(": ") + 2):].replace("\n", "").strip()
                )
            if "** STATION" in line.upper():
                hdict["station"] = (
                    line[(line.rfind(": ") + 2):].replace("\n", "").strip()
                )
            if "** SHIP" in line.upper():
                hdict["ship"] = (
                    line[(line.rfind(": ") + 2):].replace("\n", "").strip()
                )
            if "** BOTTOM DEPTH" in line.upper():
                hdict["bdep"] = (
                    line[(line.rfind(": ") + 2):].replace("\n", "").strip()
                )

            # Read moon pool info
            if "Skuteside" in line:
                mp_str = line.split()[-1]
                if mp_str == "M":
                    hdict["moon_pool"] = True
                elif mp_str == "S":
                    hdict["moon_pool"] = False

            # At the end of the SENSORS section: read the history lines
            if "</Sensors>" in line:
                start_read_history = True

            if start_read_history:
                hdict["SBEproc_hist"] += [line]

            # For .cnvs:
            # Read the line containing the END string
            # (and stop reading the file after that)
            if "*END*" in line and filename.endswith(".cnv"):
                hdict["hdr_end_line"] = n_line

                # Break the loop through all lines
                break

            # For .btls:
            # Read the line containing the header information string
            # (and stop reading the file after that)

            if line.split()[0] == "Bottle" and filename.endswith(".btl"):
                # Grab the header names
                hdict["col_names"] = line.split()

                # The header is typically wrapped across two lines. -> adding
                # info from the next line
                # NOTE: Assuming that 2-row column
                # names cannot follow 1-row ones. There *could* be cases where
                # this is otherwise, but I haven't encountered this.

                # Loop through following lines (usually just one)
                nline_next = n_line + 1
                # Arbitrary clause, we will break this loop when getting to the
                # data:
                while (True):

                    # Look at the following line
                    line_next = lines[nline_next]

                    # If line_next starts with "1", this is the beginning of
                    # the data and we stop looking for additional column name
                    # information, but store the start line.
                    if line_next.split()[0] == "1":
                        hdict["start_line_btl_data"] = nline_next
                        break

                    # If not, this is part of the header, and we add info
                    # onto the hdict["col_names"] field.
                    else:
                        hdict["col_names"] = [
                            f"{col_name} {next_part}"
                            for col_name, next_part in zip_longest(
                                hdict["col_names"],
                                line_next.split(),
                                fillvalue="",
                            )
                        ]
                    nline_next += 1

                # Remove trailing whitespaces from col_names
                col_names_stripped = [
                    col_name.rstrip() for col_name in hdict["col_names"]
                ]
                hdict["col_names"] = col_names_stripped

                # Store the end line
                hdict["hdr_end_line"] = n_line - 1

                # Break the loop through all lines
                break

        # Deal with duplicate columns (append DUPLICATE to strings..)

        seen = set()
        duplicate_columns_in_cnv = False

        for i, item in enumerate(hdict["col_names"]):
            original_item = item
            dup_str = 'DUPLICATE'
            dup_count = 1

            while item in seen:
                duplicate_columns_in_cnv = True
                item = (f"{original_item}_{dup_str}" if dup_count == 1
                        else f"{original_item}_DUPLICATE{dup_count}")
                dup_count += 1

            hdict["col_names"][i] = item
            if dup_count > 1:  # Only update longname if it was changed

                hdict["col_longnames"][i] = (
                    f"{hdict['col_longnames'][i]} [{item.split('_', -1)[-1]}]")
            seen.add(item)


        # Remove the first ('</Sensors>') and last ('*END*') lines from the SBE
        # history string.
        hdict["SBEproc_hist"] = hdict["SBEproc_hist"][1:-1]

        # Set empty fields to None
        for hkey in hkeys:
            if isinstance(hdict[hkey], list):
                if len(hdict[hkey]) == 0:
                    hdict[hkey] = None

        # Assign the file name without the directory path
        for suffix in ["cnv", "hdr", "btl"]:
            if filename.endswith(suffix):
                hdict["source_file"] = filename[filename.rfind("/") + 1:]
                hdict["source_file_type"] = suffix

        # If we did not decide that this is a moored instrument, we will asume
        # it is a profiling one.
        if "moored_sensor" not in hdict:
            hdict["moored_sensor"] = False

        return hdict


def read_csv(filename: str) -> xr.Dataset:
    """
    Read a Seabird .csv file, often output from SBE56 sensors, and convert it
    to an xarray Dataset (preserving metadata from the header).

    Parameters:
    -----------
    filename : str
        The path to the .csv file to read.

    Returns:
    --------
    xr.Dataset
        An xarray Dataset containing the data from the .csv file.

    """

    # Loop through the header to:
    # - Extract useful metadata
    # - Find the start of the data column headers
    meta_dict = {
       'instrument_model':'N/A','SN':'N/A','cal_date':'N/A',
       'conv_date':'','source_file':'N/A',}
    try:
        with open(filename, "r", encoding="latin-1") as f:
            lines = f.readlines()
            for n_line, line in enumerate(lines):
                # Look for specific metadata
                if 'Instrument type' in line:
                    meta_dict['instrument_model'] = line.split()[-1]
                if 'Serial Number' in line:
                    meta_dict['SN'] = line.split()[-1]
                if 'Conversion Date' in line:
                    meta_dict['conv_date'] = line.split()[-1]
                if 'Calibration Date' in line:
                    meta_dict['cal_date'] = line.split()[-1]
                if 'Source file' in line:
                    # Store only the file name (not the whole path)
                    src_path = line.split()[-1].replace('\\', "/")
                    meta_dict['source_file'] = os.path.basename(src_path)
                # Break the loop and store the column index when the header ends
                if line[0] != '%':
                    start_line = n_line
                    break
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {filename}") from e

    # If we can't find the info in the header, try looking in the file name
    if 'SBE056' in filename:
        if meta_dict['instrument_model'] == 'N/A':
            meta_dict['instrument_model'] = 'SBE56'
        if meta_dict['SN'] == 'N/A':
            after_sbe = filename[filename.rfind('SBE056')+6:]
            match = re.search(r'\d+', after_sbe)
            sn = str(int(match.group(0)) if match else 'N/A')
            meta_dict['SN'] = sn

    # Load the columnar data
    df = pd.read_csv(filename, header=start_line)

    # Convert "Date" and "Time" strings to datetime
    # (Suppress UserWarning for date parsing)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        df['TIME'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])


    # Set datetime to index
    df.set_index('TIME', inplace=True)

    # Convert Temperature (string) to TEMP (float)
    df['TEMP'] = df['Temperature'].astype('float')

    # Convert pandas Dataframe to xarray Dataset
    ds = xr.Dataset.from_dataframe(df)

    # Remove unused variables
    ds = ds.drop_vars(['Date', 'Time', 'Temperature'])

    # Set TIME metadata
    ds['TIME'] = time.dt64_to_datenum(ds['TIME'], epoch = '1970-01-01')
    ds['TIME'].attrs['units'] = 'Days since 1970-01-01'

    # Set TEMP metadata
    ds['TEMP'].attrs['units'] = 'degree_Celsius'
    ds['TEMP'].attrs['sensor_calibration_date'] = meta_dict['cal_date']

    # Make a string for history
    history_lines = [
        f"{num2date(ds.TIME.min()).strftime('%Y-%m-%d')} - "
        f"{num2date(ds.TIME.max()).strftime('%Y-%m-%d')}: Data collection.\n"]

    # Add conversion date line only if it's not empty
    if meta_dict['conv_date']:
        history_lines.append(
            f"{meta_dict['conv_date']}: Data converted from .xml to .cnv "
            "using SeaBird software.\n"
        )

    # Add reading date line
    history_lines.append(
        f"{pd.Timestamp.now().strftime('%Y-%m-%d')}: Data read from .csv "
        "file to xarray Dataset using kval.\n"
    )

    # Assign global attributes
    ds.attrs.update({
        'instrument_model': meta_dict['instrument_model'],
        'instrument_serial_number': meta_dict['SN'].replace('0561', ''),
        'source_file': meta_dict['source_file'],
        'filename': os.path.basename(filename),
        'history': ''.join(history_lines)
    })

    return ds





def to_netcdf(
    ds: xr.Dataset,
    path: str = "./",
    file_name: bool = False,
    convention_check: bool = False,
    add_to_history: bool = False,
    verbose: bool = False
) -> None:
    """
    Export a dataset to a NetCDF file.

    Parameters:
    ----------
    ds : xr.Dataset
        The dataset produced by read_cnv().
    where : str, optional
        Path in which the NetCDF file will be located.
    filename : bool or str, optional
        The name of the NetCDF file. If False (default),
        the name of the cnv file will be used (no ".nc" suffix necessary).
    suffix : str, optional
        Option to add a suffix (e.g., "_prelim") to the file name.
    """

    if not file_name:
        file_name = (ds.filename.replace('.cnv', '.nc')
                                .replace('.btl', '.nc'))
    if not file_name.endswith('.nc'):
        file_name += '.nc'

    dataset.to_netcdf(ds, path=path, file_name=file_name,
                      convention_check=convention_check,
                      add_to_history=add_to_history, verbose=verbose)


# INTERNAL FUNCTIONS: PARSING


def _read_column_data_xr(source_file, header_info):
    """
    Reads columnar data from a single .cnv to an xarray Dataset.

    (By way of a pandas DataFrame)

    """
    df = pd.read_csv(
        source_file,
        header=header_info["hdr_end_line"] + 1,
        sep="\\s+",
        encoding="latin-1",
        names=header_info["col_names"],
    )

    # Convert to xarray DataFrame
    ds = xr.Dataset(df).rename({"dim_0": "scan_count"})

    # Will update "binning" later if we discover that it is binned
    ds.attrs["binned"] = "no"

    return ds


def _read_btl_column_data_xr(source_file, header_info, verbose=False):
    """
    Read the columnar data from a .btl file during information (previously)
    parsed from the header.

    A little clunky as the data are distributed over two rows (subrows), a la:

    Bottle        Date    Sal00     T090C     C0S/m
    Position      Time
      1       Jan 04 2021    34.6715   2.963803  1.5859    (avg)
               15:27:59                0.0001    0.000015  (sdev)
      2       Jan 04 2021    34.6733   2.530134  1.5663    (avg)
               15:31:33                0.0002    0.000012  (sdev)

    """

    # Read column data to dataframe
    df_all = pd.read_fwf(
        source_file,
        skiprows=header_info["start_line_btl_data"],
        sep="\\s+",
        names=header_info["col_names"] + ["avg_std"],
        encoding="latin-1",
    )

    # Read the primary (avg) and second (std rows)
    # + Reindexing and dropping the "index" and "avg_std" columns

    # Parse the first subrow
    df_first_subrow = (
        df_all.iloc[0::2]
        .reset_index(drop=False)
        .drop(["index", "avg_std"], axis=1)
    )

    # Parse the second subrow
    df_second_subrow = (
        df_all.iloc[1::2].reset_index().drop(["index", "avg_std"], axis=1)
    )

    # Build a datafra with all the information combined
    df_combined = pd.DataFrame()

    # Loop through all columns except date and bottle number:
    # - Add the variables from the first subrow if they are not empty
    # - Replace column names the nice names (if available)
    # - Add variables from the second subrow with _std as suffix
    # - Also adding units, sensors as variable attributes (.attr)

    for sbe_name_ in df_all.keys():
        # Skip "bottle number" and "time" rows, will deal with those separately
        if "Bottle" in sbe_name_ or "Date" in sbe_name_:
            pass
        else:
            # Read first subrow
            try:
                df_combined[sbe_name_.replace("/", "_")] = df_first_subrow[
                    sbe_name_
                ].astype(float)
            except:
                if verbose:
                    print(f"Could not read {sbe_name_} as float - > dropping")

            # Read second subrow
            try:
                df_combined[f'{sbe_name_.replace("/", "_")}_std'] = (
                    df_second_subrow[sbe_name_].astype(float)
                )
            except:
                if verbose:
                    print(
                        f"Could not read {sbe_name_}_std as float - > dropping"
                    )

    ## Add bottle number (assuming it is the first column)
    bottle_num_name = df_first_subrow.keys()[0]  # Name of the bottle column
    df_combined["NISKIN_NUMBER"] = df_first_subrow[bottle_num_name].astype(
        float
    )

    # Parse TIME_SAMPLE from first + second subrows (assuming it is the second
    # column) Using the nice pandas function to_datetime (for parsing) Then
    # converting to datenum (days since 1970-01-01) NOTE: Should put this
    # general functionality in util.time!)

    time_name = df_first_subrow.keys()[1]  # Name of the time column

    TIME_SAMPLE = []
    epoch = "1970-01-01"

    for datestr, timestr in zip(
        df_first_subrow[time_name], df_second_subrow[time_name]
    ):
        time_string = ", ".join([datestr, timestr])

        # Parse the time string to pandas Timestamp
        time_pd = pd.to_datetime(time_string)

        # Convert to days since epoch
        time_DSE = (time_pd - pd.Timestamp(epoch)) / pd.to_timedelta(
            1, unit="D"
        )

        TIME_SAMPLE += [time_DSE]

    df_combined["TIME_SAMPLE"] = TIME_SAMPLE
    df_combined["TIME_SAMPLE"].attrs = {
        "units": f"Days since {epoch}",
        "long_name": "Time stamp of bottle closing",
        "coverage_content_type": "coordinate",
        "SBE_source_variable": time_name,
    }

    # Convert DataFrame to xarray Dataset
    ds = xr.Dataset()

    variable_list = list(df_combined.keys())
    variable_list.remove("NISKIN_NUMBER")

    for varnm in variable_list:
        ds[varnm] = xr.DataArray(
            df_combined[varnm],
            dims=["NISKIN_NUMBER"],
            coords={"NISKIN_NUMBER": df_combined.NISKIN_NUMBER},
        )

        # Preserve metadata attributes for the 'Value' variable
        ds[varnm].attrs = df_combined[varnm].attrs

    ds["NISKIN_NUMBER"].attrs = {
        "long_name": "Niskin bottle number",
        "comment": "Designated number for each physical Niskin bottle "
        "on the CTD rosette (typically e.g. 1-24, 1-11)."
        " Bottles may be closed at different depths at different stations. ",
    }

    return ds


def _read_SBE_proc_steps(ds, header_info, _is_moored=False):
    """
    Parse the information about SBE processing steps from the cnv header into
    a more easily readable format and storing the information as the global
    variable *SBE_processing*.

    _is_moored is a boolean indicating whether this is a moored instrument
    (SBE37, SBE56 etc) - in this case we do not print a warning when we can't
    access the processing history (usually not appliccable.)

    NOTE: This is a long and clunky function. This is mainly because the input
    format is clunky.

    Also:
    - Adds a *SBE_processing_date* global variable (is this useful?)
    - Adds a *source_file* variable with the names of the .hex, .xmlcon,
      and .cnv files
    - Appends SBE processing history line to the *history* attribute.
    """
    SBElines = header_info["SBEproc_hist"]

    if SBElines is None:
        if not _is_moored:
            print(
                "NOTE: Unable to read SBE processing history, probably due to "
                "non-standard header format. "
            )
        return ds

    ct = 1  # Step counter, SBE steps
    dmy_fmt = "%Y-%m-%d"

    sbe_proc_str = [
        "SBE SOFTWARE PROCESSING STEPS (extracted"
        f' from .{header_info["source_file_type"]} file header):',
    ]

    for line in SBElines:
        # Get processing date
        if "datcnv_date" in line:
            proc_date_str = re.search(
                r"(\w{3} \d{2} \d{4} \d{2}:\d{2}:\d{2})", line
            ).group(1)
            proc_date = pd.to_datetime(proc_date_str)
            proc_date_ISO8601 = time.datetime_to_ISO8601(proc_date)
            proc_date_dmy = proc_date.strftime(dmy_fmt)
            history_str = (
                f"{proc_date_dmy}: Processed to "
                f'.{header_info["source_file_type"]} using'
                ' SBE software.'
            )

        # Get input file names (without paths)
        if "datcnv_in" in line.lower():
            match = re.search(r"\\([^\\]+)\.(HEX|DAT)", line, re.IGNORECASE)
            if match:
                hex_fn = match.group(0).split("\\")[-1]  # Just the filename
            else:
                hex_fn = None  # Or handle this case as needed

            try:
                xmlcon_fn = re.search(r"\\([^\\]+\.XMLCON)", line.upper()).group(1)
            except:
                xmlcon_fn = re.search(r"\\([^\\]+\.CON)", line.upper()).group(1)
        
        # thinf: try to include .dat files
        # Get input file names (without paths)
        #if "datcnv_in" in line:
        #        hex_fn = re.search(r"\\([^\\]+\.HEX)", line.upper()).group(1)
        #    try:
        #        xmlcon_fn = re.search(
        #            r"\\([^\\]+\.XMLCON)", line.upper()
        #        ).group(1)
        #    except:
        #        xmlcon_fn = re.search(r"\\([^\\]+\.CON)", line.upper()).group(
        #            1
        #        )

            src_files_raw = f"{hex_fn}, {xmlcon_fn}"
            sbe_proc_str += [
                f"{ct}. Raw data read from {hex_fn}, {xmlcon_fn}."
            ]
            ct += 1

        # SBE processing details
        # Get skipover scans
        if "datcnv_skipover" in line:
            skipover_scans = int(re.search(r"= (\d+)", line).group(1))
            if skipover_scans != 0:
                sbe_proc_str += [
                    f"{ct}. Skipped over {skipover_scans} initial scans."
                ]
                ct += 1

        # Get ox hysteresis correction
        if "datcnv_ox_hysteresis_correction" in line:
            ox_hyst_yn = re.search(r"= (\w+)", line).group(1)
            if ox_hyst_yn == "yes":
                sbe_proc_str += [
                    f"{ct}. Oxygen hysteresis correction applied."
                ]
                ct += 1

        # Get ox tau correction
        if "datcnv_ox_tau_correction" in line:
            ox_hyst_yn = re.search(r"= (\w+)", line).group(1)
            if ox_hyst_yn == "yes":
                sbe_proc_str += [f"{ct}. Oxygen tau correction applied."]
                ct += 1

        # Get low pass filter details
        if "filter_low_pass_tc_A" in line:
            lp_A = float(re.search(r" = (\d+\.\d+)", line).group(1))
        if "filter_low_pass_tc_B" in line:
            lp_B = float(re.search(r" = (\d+\.\d+)", line).group(1))
        if "filter_low_pass_A_vars" in line:
            try:
                lp_vars_A = re.search(r" = (.+)$", line).group(1).split()
                sbe_proc_str += [
                    f"{ct}. Low-pass filter with time constant {lp_A}"
                    + f' seconds applied to: {" ".join(lp_vars_A)}.'
                ]
                ct += 1
            except:
                print(
                    "FYI: Looks like filter A was not applied to any variables."
                )
        if "filter_low_pass_B_vars" in line:
            try:
                lp_vars_B = re.search(r" = (.+)$", line).group(1).split()
                sbe_proc_str += [
                    f"{ct}. Low-pass filter with time constant {lp_B}"
                    + f' seconds applied to: {" ".join(lp_vars_B)}.'
                ]
                ct += 1
            except:
                print(
                    "FYI: Looks like filter B was not applied to any variables."
                )

        # Get cell thermal mass correction details
        if "celltm_alpha" in line:
            celltm_alpha = re.search(r"= (.+)$", line).group(1)
        if "celltm_tau" in line:
            celltm_tau = re.search(r"= (.+)$", line).group(1)
        if "celltm_temp_sensor_use_for_cond" in line:
            celltm_sensors = re.search(r"= (.+)$", line).group(1)
            sbe_proc_str += [
                f"{ct}. Cell thermal mass correction applied to conductivity"
                f" from sensors: [{celltm_sensors}]. ",
                f"   > Parameters ALPHA: [{celltm_alpha}], TAU: [{celltm_tau}].",
            ]
            ct += 1

        # Get loop edit details
        if "loopedit_minVelocity" in line:
            loop_minvel = re.search(r"= (\d+(\.\d+)?)", line).group(1)
            _loop_minvel = (
                "fixed"  # Flag (fixed min speed vs percentage of mean speed)
            )

        if "loopedit_percentMeanSpeed" in line:
            loop_minpct_minvel = re.search(
                r"minV = (\d+(\.\d+)?)", line
            ).group(1)
            loop_minpct_ws = re.search(r"ws = (\d+(\.\d+)?)", line).group(1)
            loop_minpct_pct = re.search(
                r"percent = (\d+(\.\d+)?)", line
            ).group(1)
            _loop_minvel = (
                "pct"  # Flag (fixed min speed vs percentage of mean speed)
            )

        if "loopedit_surfaceSoak" in line:  # and float(loop_minvel)>0:
            if "do not remove" in line:
                _loop_ss_remove = False
            else:
                loop_ss_mindep = re.search(
                    r"minDepth = (\d+(\.\d+)?)", line
                ).group(1)
                loop_ss_maxdep = re.search(
                    r"maxDepth = (\d+(\.\d+)?)", line
                ).group(1)
                _loop_ss_deckpress = re.search(
                    r"useDeckPress = (\d+(\.\d+)?)", line
                ).group(1)
                if _loop_ss_deckpress == "0":
                    loop_ss_deckpress_str = "No"
                else:
                    loop_ss_deckpress_str = "Yes"
                _loop_ss_remove = True

        if "loopedit_excl_bad_scans" in line:  # and float(loop_minvel)>0:
            loop_excl_bad_scans = re.search(r"= (.+)", line).group(1)
            if loop_excl_bad_scans == "yes":
                loop_excl_str = "Bad scans excluded"
            else:
                loop_excl_str = "Bad scans not excluded"

            if _loop_minvel == "fixed":
                minvel_str = f"Minimum velocity (ms-1): {loop_minvel}"
            elif _loop_minvel == "pct":
                minvel_str = (
                    f"Minimum velocity: {loop_minpct_pct}% of mean "
                    f"velocity within a {loop_minpct_ws}-pt window\n"
                    f"     (>{loop_minpct_minvel} ms-1 for the first"
                    " window)"
                )

            if _loop_ss_remove:
                loop_ss_str = (
                    f"\n     Soak depth range (m): "
                    f"{loop_ss_mindep} to {loop_ss_maxdep} "
                    f"(Deck pressure offset: {loop_ss_deckpress_str})."
                )
            else:
                loop_ss_str = "\n     Surface soak not removed. "

            sbe_proc_str += [
                f"{ct}. Loop editing applied.",
                (
                    f"   > Parameters:\n     {minvel_str}. "
                    + f"{loop_ss_str}"
                    + f"\n   > {loop_excl_str}. "
                ),
            ]
            ct += 1

        # Get wild edit details
        if "wildedit_date" in line:
            sbe_proc_str += [f"{ct}. Wild editing applied."]
        if "wildedit_vars" in line:
            we_vars = re.search(r" = (.+)$", line).group(1).split()
            sbe_proc_str += [
                f'   > Applied to variables: {" ".join(we_vars)}.'
            ]
        if "wildedit_pass1_nstd" in line:
            we_pass1 = float(re.search(r" = (\d+\.\d+)", line).group(1))
        if "wildedit_pass2_nstd" in line:
            we_pass2 = float(re.search(r" = (\d+\.\d+)", line).group(1))
        if "wildedit_pass2_mindelta" in line:
            we_mindelta = float(re.search(r" = (\d+\.\d+)", line).group(1))
        if "wildedit_npoint" in line:
            we_npoint = float(re.search(r" = (\d+)", line).group(1))
            sbe_proc_str += [
                (
                    f"   > Parameters: n_std (first pass): {we_pass1}, "
                    f"n_std (second pass): {we_pass2}, min_delta: {we_mindelta},\n"
                    f"   > # points per test: {we_npoint}."
                )
            ]
            ct += 1

        if "Derive_in" in line or "derive_in" in line:
            sbe_proc_str += [
                f"{ct}. Derived EOS-8 salinity and other variables."
            ]
            ct += 1

        # Get window filter details
        if "wfilter_excl_bad_scans" in line:
            wf_bad_scans = re.search(r"= (.+)", line).group(1)
            if wf_bad_scans == "yes":
                wf_bad_scans = "Bad scans excluded"
            else:
                wf_bad_scans = "Bad scans not excluded"
        if "wfilter_action" in line:

            wf_variable = re.search(r"(.+) =", line).group(0).split()[-2]
            wf_filter_type = (
                re.search(r"= (.+)", line).group(0).split()[1].replace(",", "")
            )
            wf_filter_param = re.search(r"(\d+)", line).group(0)
            sbe_proc_str += [
                f"{ct}. Window filter ({wf_filter_type}, "
                f"{wf_filter_param}) applied to {wf_variable} ({wf_bad_scans})."
            ]
            ct += 1

        # Get align CTD details
        if "alignctd_adv" in line:
            # Find all matches in the string
            matches = re.findall(r"(\w+)\s+([0-9.]+)", line)
            # Rerutn a list of tuples with (variable, advance time in seconds)
            align_tuples = [(key, float(value)) for key, value in matches]
            sbe_proc_str += [f"{ct}. Misalignment correction applied."]
            sbe_proc_str += [
                "   > Parameters [variable (advance time, sec)]:"
            ]
            align_str = []
            for align_tuple in align_tuples:
                align_str += [f"{align_tuple[0]} ({align_tuple[1]})"]
            sbe_proc_str += [f'   > {", ".join(align_str)}']
            ct += 1

        # Get bin averaging details
        if "binavg_bintype" in line:
            bin_unit = re.search(r" = (.+)$", line).group(1)
        if "binavg_binsize" in line:
            bin_size = re.search(r" = (.+)$", line).group(1)
        if "binavg_excl_bad_scans" in line:
            binavg_excl_bad_scans = re.search(r"= (.+)", line)
            if binavg_excl_bad_scans == "yes":
                binavg_excl_str = "Bad scans excluded"
            else:
                binavg_excl_str = "Bad scans not excluded"
        if "binavg_skipover" in line:
            bin_skipover = re.search(r" = (.+)$", line).group(1)
            if bin_skipover != 0:
                bin_skipover_str = (
                    f", skipped over {bin_skipover} initial scans"
                )
            else:
                bin_skipover = ""
        if "binavg_surface_bin" in line:
            surfbin_yn = (
                re.search(r"surface_bin = (.+)$", line).group(1).split()
            )
            if surfbin_yn != "yes":
                surfbin_str = "(No surface bin)"
            else:
                surfbin_params = (
                    re.search(r"yes, (.+)$", line).group(1).split().upper()
                )
                surfbin_str = f"Surface bin parameters: {surfbin_params}"
            sbe_proc_str += [f"{ct}. Bin averaged ({bin_size} {bin_unit})."]
            sbe_proc_str += [f"   > {binavg_excl_str}{bin_skipover_str}."]
            sbe_proc_str += [f"   > {surfbin_str}."]
            SBE_binned = f"{bin_size} {bin_unit} (SBE software)"
            ct += 1
    try:
        ds.attrs["binned"] = SBE_binned
    except:
        pass
    ds.attrs["SBE_processing"] = "\n".join(sbe_proc_str)
    ds.attrs["SBE_processing_date"] = proc_date_ISO8601
    ds.attrs["history"] += f"\n{history_str}"
    ds.attrs["source_file"] = (
        f'{src_files_raw} -> {header_info["source_file"].upper()}'
    )

    return ds


def _read_sensor_info(source_file, verbose=False):
    """
    Look through the header for information about sensors:
        - Serial numbers
        - Calibration dates
    """

    sensor_dict = {}

    # Define stuff we want to remove from strings
    drop_str_patterns = ["<.*?>", "\n", " NPI"]
    drop_str_pattern_comb = "|".join(drop_str_patterns)

    with open(source_file, "r", encoding="latin-1") as f:
        look_sensor = False

        # Loop through header lines
        for n_line, line in enumerate(f.readlines()):

            # When encountering a <sensor> flag:
            # Begin looking for instrument info
            if "<sensor" in line.lower():
                # Set initial flags
                look_sensor = True
                sensor_header_line = n_line + 1
                store_sensor_info = False

            if look_sensor:
                # Look for an entry corresponding to the sensor in the
                # _sensor_info_dict (prescribed) dictionary
                # If found: read info. If not: Ignore.
                if n_line == sensor_header_line:

                    for (
                        sensor_str,
                        var_key,
                    ) in vardef.sensor_info_dict_SBE.items():

                        if sensor_str in line:
                            store_sensor_info = True
                            var_key_sensor = var_key

                    # Print if verbose
                    shline = line.replace("#     <!-- ", "").replace("\n", "")
                    (
                        print(f"\nRead from: {var_key_sensor} ({shline})")
                        if verbose
                        else None
                    )

                if store_sensor_info:
                    # Grab instrument serial number
                    if "<SerialNumber>" in line:
                        serial_number_index = (
                            line.rindex("<SerialNumber>") + 14
                        )

                        SN_instr = re.sub(
                            drop_str_pattern_comb,
                            "",
                            line[serial_number_index:],
                        )

                    # Grab calibration date
                    if "<CalibrationDate>" in line:
                        rind_cd = line.rindex("<CalibrationDate>") + 17
                        cal_date_instr = (
                            line[rind_cd:]
                            .replace("</CalibrationDate>", "")
                            .replace("\n", "")
                        )

            # When encountering a <sensor> flag:
            # Stop looking for instrument info and store
            # in dictionary
            if "</sensor>" in line.lower():

                # Store to dictionary
                if look_sensor and store_sensor_info:
                    sensor_dict[var_key_sensor] = {
                        "SN": SN_instr,
                        "cal_date": cal_date_instr,
                    }

                # Print if verbose
                (
                    print(
                        f"SN: {SN_instr}  // cal date: {cal_date_instr}+n",
                        f"Stop reading from {var_key_sensor}"
                        f" (save: {store_sensor_info})",
                    )
                    if verbose
                    else None
                )

                # Reset flags
                (
                    look_sensor,
                    var_key_sensor,
                ) = (
                    False,
                    None,
                )
                SN_instr, cal_date_instr = None, None

            # Stop reading after the END string (.cnv, .hdr)
            if "*END*" in line:
                return sensor_dict
    # For .btl files (no end string) - just read the whole file before
    # before returning

    return sensor_dict



def _add_meta_info_sbe56_cnv(ds, source_file, verbose=False):
    """

    Parsing sensor info from an alternative SBE56 format.
    Look through the header for information
        - Instrument type (redundant in most cases)
        - Serial numbers
        - Calibration dates
        - Source file
    """

    meta_dict = {}

    with open(source_file, "r", encoding="latin-1") as f:
        lines = f.readlines()
        for n_line, line in enumerate(lines):
            # Look for specific metadata
            if 'StatusData' in line and 'DeviceType' in line:
                # Regular expression to extract the device type
                match = re.search(r"DeviceType='([^']+)'", line)
                if match:
                        devicetype = match.group(1)
                        meta_dict['instrument_model'] = devicetype
            if 'StatusData' in line and 'SerialNumber' in line:
                # Regular expression to extract the serial number
                match = re.search(r"SerialNumber='([^']+)'", line)
                if match:
                    # Replacing redundant "056X" at the start
                    SN = (match.group(1)
                          .replace('0561', '').replace('0560', ''))
                    meta_dict['instrument_serial_number'] = SN
            if 'CalDate' in line:
                # Regular expression to extract the serial number
                match = re.search(r"<CalDate>([\d\-]+)</CalDate>", line)
                if match:
                    cal_date = match.group(1)
                    meta_dict['sensor_calibration_date'] = cal_date
            if 'Source file' in line:
                # Store only the file name (not the whole path)
                src_path = line.split()[-1].replace('\\', "/")
                meta_dict['source_file'] = os.path.basename(src_path)
            # Break the loop and store the column index when the header ends
            if '*END*' in line:
                break

    # Add parsed values as attributes
    for key, item in meta_dict.items():
        if key.startswith('instrument'):
            ds.attrs[key] = item
        if key == 'sensor_calibration_date':
            if item != 'N/A':
                ds.TEMP.attrs[key] = item

    # Add the global attribute `instrument_serial_number` as variable attribute
    # `sensor_serial_number` in `TEMP` for consistency.
    # (May cause some redundancy - not too worried)
    if 'instrument_serial_number' in ds.attrs:
        ds.TEMP.attrs['sensor_serial_number'] = ds.instrument_serial_number

    return ds


# INTERNAL FUNCTIONS: MODIFY THE DATASET

def _remove_duplicate_variables(ds):
    """
    Removes variables  ending with '_DUPLICATE'.

    Parameters:
    - ds (xarray.Dataset): The input xarray Dataset.
    """

    ds = ds.drop_vars(
        [var for var in ds.variables if "_DUPLICATE" in var]
    )

    return ds


def _assign_specified_lat_lon_station(ds, lat, lon, station):
    """
    Assign values to latitude, longitude, station attributes
    if specified by the user (not None).
    """

    if lat:
        ds.attrs["latitude"] = lat
    if lon:
        ds.attrs["longitude"] = lon
    if station:
        ds.attrs["station"] = station

    return ds


def _add_start_time(ds, header_info, start_time_NMEA=False):
    """
    Add a start_time attribute.

    Default behavior:
        - Use start_time header line
    If start_time_NMEA = True:
        - Use "NMEA UTC" line if present
        - If not, use start_time

    Compicated way of doing this because there are some occasional
    oddities where e.g.
    - The "start_time" line is some times incorrect
    - The "NMEA UTC" is not always present.

    Important to get right since this is used for assigning
    a time stamp to profiles.
    """

    if start_time_NMEA:
        try:
            ds.attrs["start_time"] = time.datetime_to_ISO8601(
                header_info["NMEA_time"]
            )
            ds.attrs["start_time_source"] = '"NMEA UTC" header line'

        except:
            try:
                ds.attrs["start_time"] = time.datetime_to_ISO8601(
                    header_info["start_time"]
                )
                ds.attrs["start_time_source"] = '"start_time" header line'

            except:
                raise Warning(
                    "Did not find a start time!"
                    ' (no "start_time" or NMEA UTC" header lines).'
                )
    else:
        ds.attrs["start_time"] = time.datetime_to_ISO8601(
            header_info["start_time"]
        )
        ds.attrs["start_time_source"] = '"start_time" header line'

    return ds


def _add_time_dim_profile(
    ds,
    epoch="1970-01-01",
    time_source="sample_time",
    suppress_latlon_warning=False,
):
    """
    Add a 0-dimensional TIME coordinate to a profile.
    Also adds the 0-d variables STATION, LATITUDE, and LONGITUDE.

    time_source:

    sample_time:  Average of TIME_SAMPLE varibale (which was calculated
                  from timeS or timeJ fields).

    start_time:   Use the start_time field (star of scans). This is extracted
                  from either the header; either the 'start_time' or
                  'NMEA UTC' lines (which of the twois specified in the
                  read_cnv() function usng the start_time_NMEA flag).
    """

    if "TIME_SAMPLE" in ds:
        ds = ds.assign_coords({"TIME": [ds.TIME_SAMPLE.mean()]})
        ds.TIME.attrs = {
            "units": ds.TIME_SAMPLE.units,
            "standard_name": "time",
            "long_name": "Average time of measurements",
            "SBE_source_variable": ds.TIME_SAMPLE.SBE_source_variable,
        }
    else:
        start_time_num = time.ISO8601_to_datenum(ds.attrs["start_time"])
        ds = ds.assign_coords({"TIME": [start_time_num]})
        ds.TIME.attrs = {
            "units": f"Days since {epoch} 00:00",
            "standard_name": "time",
            "long_name": "Start time of profile",
            "comment": f"Source: {ds.start_time_source}",
        }

    # Add STATION
    ds = _add_station_variable(ds)
    ds = _add_latlon_variables(ds, suppress_latlon_warning)

    return ds


def _add_station_variable(ds):
    """
    Adds a 0-d STATION variable to a profile.

    Grabs the value from the station attribute.

    (Requires that a 0-d TIME dimension exists, see
    _add_time_dim_profile).

    """

    station_array = xr.DataArray(
        [ds.station],
        dims="TIME",
        coords={"TIME": ds.TIME},
        attrs={"long_name": "CTD station ID", "cf_role": "profile_id"},
    )
    ds["STATION"] = station_array

    return ds


def _add_latlon_variables(ds, suppress_latlon_warning=False):
    """
    Adds a 0-d STATION variable to a profile.

    Grabs the value from the station attribute.

    (Requires that a 0-d TIME dimension exists, see
    _add_time_dim_profile).

    """

    missing = False

    if "latitude" in ds.attrs:
        if ds.latitude:  # If not "None"
            lat_value = ds.latitude
        else:
            lat_value = np.nan
            missing = "latitude"

    elif "LATITUDE_SAMPLE" in ds:
        lat_value = ds.LATITUDE_SAMPLE.mean()
    else:
        lat_value = np.nan
        missing = "latitude"

    lat_array = xr.DataArray(
        [lat_value],
        dims="TIME",
        coords={"TIME": ds.TIME},
        attrs={
            "standard_name": "latitude",
            "units": "degree_north",
            "long_name": "latitude",
        },
    )

    ds["LATITUDE"] = lat_array

    if "longitude" in ds.attrs:
        if ds.longitude:  # If not "None"
            lon_value = ds.longitude
        else:
            lon_value = np.nan
            if missing:
                missing += ", longitude"
            else:
                missing = "longitude"

    elif "LONGITUDE_SAMPLE" in ds:
        lon_value = ds.LONGITUDE_SAMPLE.mean()
    else:
        lon_value = np.nan
        if missing:
            missing += ", longitude"
        else:
            missing = "longitude"

    lon_array = xr.DataArray(
        [lon_value],
        dims="TIME",
        coords={"TIME": ds.TIME},
        attrs={
            "standard_name": "longitude",
            "units": "degree_east",
            "long_name": "longitude",
        },
    )
    ds["LONGITUDE"] = lon_array

    if missing and suppress_latlon_warning is False:
        warn_str = (
            f"{ds.STATION.values[0]}: Unable to find [{missing}] "
            "in .cnv file --> Assigning NaN values."
        )
        print(f"NOTE!: {warn_str}")

    return ds


def _add_header_attrs(
    ds, header_info, station_from_filename=False, decimals_latlon=4
):
    """
    Add the following as attributes if they are available from the header:

        ship, cruise, station, latitude, longitude, instrument_model

    If the attribute is already assigned, we don't change it

    If we don't have a station, we use the cnv file name base.
    (can be forced by setting station_from_filename = True)
    """
    for key in [
        "ship",
        "cruise_name",
        "station",
        "latitude",
        "longitude",
        "instrument_model",
    ]:

        if key in header_info and key not in ds.attrs:

            ds.attrs[key] = header_info[key]
            if key in ["latitude", "longitude"] and ds.attrs[key] is not None:
                ds.attrs[key] = np.round(ds.attrs[key], decimals_latlon)

    # Grab station from filename (stripping away .cnv and _bin)
    if "station" not in ds.attrs or station_from_filename:
        station_from_filename = (
            header_info["source_file"]
            .replace(".cnv", "")
            .replace(".CNV", "")
            .replace("_bin", "")
            .replace("_BIN", "")
            .replace(".btl", "")
            .replace(".BTL", "")
        )

        ds.attrs["station"] = station_from_filename

    return ds


def _apply_flag(ds):
    """
    Applies the *flag* value assigned by the SBE processing.

    -> Remove scans  where flag != 0

    """
    ds = ds.where(ds.SBE_FLAG == 0, drop=True)

    return ds


def _apply_flag_variables(ds: xr.Dataset, bad_flag_value: float) -> xr.Dataset:
    """
    Replaces values equal to `bad_flag_value` with NaN in all variables in the dataset.

    Parameters:
    - ds: xarray.Dataset
    - bad_flag_value: float, e.g. -9.990e-29

    Returns:
    - xarray.Dataset with bad values set to NaN and flagged entries removed
    """

    if bad_flag_value:
        # Replace bad_flag_value with NaN

        ds_clean = ds.copy(deep = True)

        for var in ds.data_vars:
            try:
                # Only apply .where to numeric variables that contain the bad_flag_value
                if np.issubdtype(ds[var].dtype, np.number) and (ds[var] == bad_flag_value).any():
                    ds_clean[var] = ds[var].where(ds[var] != bad_flag_value, np.nan)
            except Exception as e:
                pass
    else:
        return ds
    return ds_clean


def _remove_up_dncast(ds, keep="downcast"):
    """
    Takes a ctd Dataframe and returns a subset containing only either the
    upcast or downcast.

    Note:
    Very basic algorithm right now - just removing everything before/after the
    pressure max and relying on SBE flaggig for the rest.
    -> will likely have to replace with something more sophisticated.
    """
    # Index of max pressure, taken as "end of downcast"
    max_pres_index = int(np.argmax(ds.PRES.data))

    # If the max index is a single value following invalid values,
    # we interpret it as the start of the upcast and use the preceding
    # point as the "end of upcast index"
    if (ds.scan_count[max_pres_index] - ds.scan_count[max_pres_index - 1]) > 1:

        max_pres_index -= 1

    if keep == "upcast":
        # Remove everything *before* pressure max
        ds = ds.isel({"scan_count": slice(max_pres_index + 1, None)})
        ds.attrs["profile_extracted"] = "upcast"

    elif keep in ["dncast", "downcast"]:
        # Remove everything *after* pressure max
        ds = ds.isel({"scan_count": slice(None, max_pres_index + 1)})
        ds.attrs["profile_extracted"] = "downcast"

    else:
        raise Exception('"keep" must be either "upcast" or "dncast"')

    return ds


## UNUSED
def _remove_surface_soak(
    ds, scans_window=10, cutoff_speed=0, max_p=100, apply_to_flag=True
):
    """
    Removes surface soak period based on the pressure record.

    Looks for a profile starting point where
    1. Low-pass filtered speed [delta pressure / delta time] is below a certain threshold.
    2. Pressure is less than a threshold value.

    Should only be used for downcasts..


    NOTE: Starting to look good, but not using this for now.
    - May be able to solve my problems by fossling with SBE processing parameters..
    """

    # Calculate dPRES/dTIME
    dPRES = ds["PRES"].diff("scan_count")
    dTIME = ds["TIME_SAMPLE"].diff("scan_count") * 86400
    dPRES_dTIME = dPRES / dTIME

    # Apply LPF (rolling mean) to dPRES/dTIME
    dPRES_dTIME_LPF = dPRES_dTIME.rolling(
        scan_count=scans_window, center=True
    ).mean()

    # Evaluate speed threshold criterion
    is_above_speed_threshold = dPRES_dTIME_LPF > cutoff_speed
    # Evaluate pressure threshold criterion
    is_below_pressure_threshold = (
        ds.isel({"scan_count": slice(1, None)}).PRES < max_p
    )
    # Combine nto single boolean array
    both_criteria = is_above_speed_threshold * is_below_pressure_threshold

    # Take the profile start as the last index in the record where the combined criteria go
    # from False to True
    both_criteria_diff = np.diff(both_criteria * 1)
    profile_start = np.where(both_criteria_diff == 1)[0][-1]

    # Remove everything before the *profile_start* index.
    ds = ds.isel({"scan_count": slice(profile_start)})

    return ds


def _update_variables(ds, source_file, _is_moored=False):
    """
    Take a Dataset and
    - Update the header names from SBE names (e.g. 't090C')
      to more standardized name (e.g., 'TEMP1').
    - Assign the appropriate units and standard_name as attributes.
    - Assign sensor serial number(s) and/or calibration date(s)
      where available.

    `_is_moored` is a boolead indicating whether this is a moored sensor
    (SBE37, SBE56 etc).

    What to look for is specified in _variable_defs.py
    -> Will update dictionaries in there as we encounter differently
       formatted files.
    """

    # Try reading sensor infor using the conventional format first
    sensor_info = _read_sensor_info(source_file)

    for old_name in ds.keys():
        old_name_cap = old_name.upper()

        # For .btl-files we can have variables with a _std suffix
        if old_name_cap.endswith("_STD"):
            old_name_cap = old_name_cap.replace("_STD", "")
            std_suffix = True
        else:
            std_suffix = False

        if old_name_cap in vardef.SBE_name_map:

            var_dict = vardef.SBE_name_map[old_name_cap]

            if not std_suffix:
                new_name = var_dict["name"]
            else:
                new_name = var_dict["name"] + "_std"

            # If we have multiple instances of the same new_name:
            if new_name in ds.keys():
                new_name += "#2"

            unit = var_dict["units"]
            ds = ds.rename({old_name: new_name})
            ds[new_name].attrs["units"] = unit

            if "standard_name" in var_dict:
                ds[new_name].attrs["standard_name"] = var_dict["standard_name"]

            if "reference_scale" in var_dict:
                ds[new_name].attrs["reference_scale"] = var_dict[
                    "reference_scale"
                ]

            if "sensors" in var_dict:
                sensor_SNs = []
                sensor_caldates = []

                for sensor in var_dict["sensors"]:
                    try:
                        sensor_SNs += [sensor_info[sensor]["SN"]]
                        sensor_caldates += [sensor_info[sensor]["cal_date"]]

                    except:
                        if _is_moored:
                            pass
                        else:
                            print(
                                f"*NOTE*: Failed to find sensor {sensor}"
                                f" associated with variable {old_name_cap}.\n"
                                f"(file: {source_file})"
                            )
                            sensor_SNs += ["N/A"]
                            sensor_caldates += ["N/A"]


                ds[new_name].attrs["sensor_serial_number"] = ", ".join(
                    sensor_SNs
                )

                ds[new_name].attrs["sensor_calibration_date"] = ", ".join(
                    sensor_caldates
                )

                for key in "sensor_serial_number", "sensor_calibration_date":
                    if ds[new_name].attrs[key] == 'N/A':
                        del ds[new_name].attrs[key]

    ds.attrs['source_file'] = os.path.basename(source_file)


    return ds


def _remove_start_scan(ds, start_scan):
    """
    Remove all scans before *start_scan*
    """
    ds = ds.sel({"scan_count": slice(start_scan, None)})
    return ds


def _remove_end_scan(ds, end_scan):
    """
    Remove all scans after *end_scan*
    """
    ds = ds.sel({"scan_count": slice(None, end_scan + 1)})
    return ds


## INTERNAL FUNCTIONS: INSPECT


def _inspect_extracted(ds, ds0, start_scan=None, end_scan=None):
    """
    Plot the pressure tracks showing the portion extracted after
    and/or removing upcast.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ds0.scan_count, ds0.PRES, ".k", ms=3, label="All scans")
    ax.plot(ds.scan_count, ds.PRES, ".r", ms=4, label="Extracted for use")

    if start_scan:
        ax.axvline(start_scan, ls="--", zorder=0, label="start_scan")
    if end_scan:
        ax.axvline(end_scan, ls="--", zorder=0, label="end_scan")
    ax.set_ylabel("PRES [dbar]")
    ax.set_xlabel("SCAN COUNTS")
    ax.legend()
    ax.invert_yaxis()
    ax.grid(alpha=0.5)
    plt.show()


## INTERNAL FUNCTIONS: FORMAT CONVERSION


def _nmea_lon_to_decdeg(deg_str, min_str, EW_str):
    """
    Convert NMEA longitude to decimal degrees longitude.

    E.g.:

    ['006', '02.87', 'E'] (string) --> 6.04783333 (float)
    """

    if EW_str == "E":
        dec_sign = 1
    elif EW_str == "W":
        dec_sign = -1

    decdeg = int(deg_str) + float(min_str) / 60

    return decdeg * dec_sign


def _nmea_lat_to_decdeg(deg_str, min_str, NS_str):
    """
    Convert NMEA latitude to decimal degrees latitude.

    E.g.:

    ['69', '03.65', 'S'] (string) --> -69.060833 (float)
    """

    if NS_str == "N":
        dec_sign = 1
    elif NS_str == "S":
        dec_sign = -1

    decdeg = int(deg_str) + float(min_str) / 60

    return decdeg * dec_sign


def _nmea_time_to_datetime(mon, da, yr, hms):
    """
    Convert NMEA time to datetime timestamp.

    E.g.:

    ['Jan', '05', '2021', '15:58:23']  --> Timestamp('2021-01-05 15:58:23')
    """
    nmea_time_dt = pd.to_datetime(f"{mon} {da} {yr} {hms}")

    return nmea_time_dt


def _decdeg_from_line(line):
    """
    Parse a line containing latitude or longitude in degrees and minutes
    (e.g., '** Latitude: 68 50 s') and return decimal degrees.
    
    Handles various formats like:
      - '** Latitude: 081 16.1347'
      - '** Longitude: 003 00 e'
      - 'Latitude N 081 12.33'

    Returns:
        float or None: Decimal degrees, or None if parsing fails.
    """
    # Extract the part after the colon (if present)
    if ":" in line:
        deg_min_str = line[(line.rfind(":") + 1):].strip().split()
    else:
        deg_min_str = line.strip().split()

    if not deg_min_str:
        return None

    # Remove any non-numeric prefixes
    while deg_min_str and not deg_min_str[0].replace('.', '', 1).isdigit():
        deg_min_str = deg_min_str[1:]

    if len(deg_min_str) < 2:
        return None

    try:
        deg = float(deg_min_str[0])
        min = float(deg_min_str[1])
    except ValueError:
        return None

    # Default is positive (N or E)
    sign = 1

    # Look for directional indicators (in last or third element)
    direction = line.lower().strip().split()[-1]
    if direction in ['s', 'w']:
        sign = -1

    return sign * (deg + min / 60)

def _decdeg_from_line_old(line):
    """
    From a line line  '** Latitude: 081 16.1347'
    return decimal degrees, e.g. 81.26891166

    NOTE: This is to parse ad-hoc formatted files. May not work for everything.
    The default behaviour is to look for NMEA Lat/Lon - this is preferable and
    should be present in a well formatted header!
    """
    deg_min_str = line[(line.rfind(":") + 2):].replace("\n", "").split()
    if len(deg_min_str) == 0:  # If there is no actual lat/lon string..
        return None

    # Occasionally written without colon; as e.g. "Latitude N 081 12.33"
    # -> Need to remove the strings before parsing degrees

    if isinstance(deg_min_str[0], str):
        is_number = False
        while not is_number:
            try:
                # Test if we have a number
                float(deg_min_str[0])
                is_number = True
            except:
                # Flip the sign is S or W (convention is positive N/E)
                if deg_min_str[0] in ["S", "W"]:
                    deg_min_str[1] = str(-float(deg_min_str[1]))
                deg_min_str = deg_min_str[1:]

    deg = float(deg_min_str[0])
    min = float(deg_min_str[1])
    min_decdeg = min / 60
    sign_deg = np.sign(deg)
    decdeg = deg + sign_deg * min_decdeg

    return decdeg


def _convert_time(
    ds,
    header_info,
    epoch="1970-01-01",
    suppress_time_warning=False,
    start_time_NMEA=False,
):
    """
    Convert time either from julian days (TIME_JULD extracted from 'timeJ')
    or from time elapsed in seconds (TIME_ELAPSED extracted from timeS).

    suppress_time_warning: Don't show a warning if there are no
    timeJ or timeS fields (useful for loops etc).

    """

    if "TIME_ELAPSED" in ds.keys():
        ds = _convert_time_from_elapsed(
            ds, header_info, epoch=epoch, start_time_NMEA=start_time_NMEA
        )
        ds.TIME_SAMPLE.attrs["SBE_source_variable"] = "timeS"
    elif "TIME_JULD" in ds.keys():
        ds = _convert_time_from_juld(
            ds, header_info, epoch=epoch, start_time_NMEA=start_time_NMEA
        )
        ds.TIME_SAMPLE.attrs["SBE_source_variable"] = "timeJ"

    else:
        if not suppress_time_warning:
            print(
                "\nNOTE: Failed to extract sample time info "
                "(no timeS, timeJ or timeJv2 in .cnv file)."
                "\n(Not a big problem for profiles, the start_time "
                "can be used to assign a profile time)."
            )
    return ds


def _convert_time_from_elapsed(
    ds, header_info, epoch="1970-01-01", start_time_NMEA=False
):
    """
    Convert TIME_ELAPSED (sec)
    to TIME_SAMPLE (days since 1970-01-01)

    Only sensible reference I could find is here;
    https://search.r-project.org/CRAN/refmans/oce/html/read.ctd.sbe.html

    (_DSE: time since epoch)
    """

    if start_time_NMEA:
        ref_time = header_info["NMEA_time"]
    else:
        ref_time = header_info["start_time"]

    start_time_DSE = (ref_time - pd.Timestamp(epoch)) / pd.to_timedelta(
        1, unit="D"
    )

    elapsed_time_days = ds.TIME_ELAPSED / 86400

    time_stamp_DSE = start_time_DSE + elapsed_time_days

    ds["TIME_SAMPLE"] = time_stamp_DSE
    ds.TIME_SAMPLE.attrs["units"] = f"Days since {epoch} 00:00:00"
    ds = ds.drop_vars("TIME_ELAPSED")

    return ds


def _convert_time_from_juld(
    ds, header_info, epoch="1970-01-01", start_time_NMEA=False
):
    """
    Convert TIME_ELAPSED (sec)
    to TIME (days since 1970-01-01)

    Only sensible reference I could fnd is here;
    https://search.r-project.org/CRAN/refmans/oce/html/read.ctd.sbe.html

    (_DSE: time since epoch)
    """

    if start_time_NMEA:
        ref_time = header_info["NMEA_time"]
    else:
        ref_time = header_info["start_time"]

    year_start = ref_time.replace(month=1, day=1, hour=0, minute=0, second=0)
    time_stamp = pd.to_datetime(
        ds.TIME_JULD - 1,
        origin=year_start,
        unit="D",
        yearfirst=True,
    ).round("1s")
    time_stamp_DSE = (time_stamp - pd.Timestamp(epoch)) / pd.to_timedelta(
        1, unit="D"
    )

    ds["TIME_SAMPLE"] = (("scan_count"), time_stamp_DSE)
    ds.TIME_SAMPLE.attrs["units"] = f"Days since {epoch} 00:00:00"
    ds = ds.drop_vars("TIME_JULD")

    return ds


def _modify_moored(ds, hdict):
    """
    Some custom modifications for moored sensors
    """

    # Use TIME_SAMPLE as TIME
    ds = ds.rename_vars({"TIME_SAMPLE": "TIME"})

    # Remove som non-useful metadata attributes
    for remove_attr in [
        "binned",
        "latitude",
        "longitude",
        "station",
        "profile_extracted",
        "start_time",
        "start_time_source",
    ]:

        if remove_attr in ds.attrs:
            del ds.attrs[remove_attr]


    if 'instrument_serial_number' in hdict:
        ds.attrs["instrument_serial_number"] = (
            hdict["instrument_serial_number"]
        )
    else:
        # Add instrument serial number (should be same as TEMP)
        if "TEMP" in ds:
            if hasattr(ds.TEMP, "sensor_serial_number"):
                ds.attrs["instrument_serial_number"] = ds.TEMP.attrs[
                    "sensor_serial_number"
                ]

    # Add instrument serial number if we have it in the header dict
    # (otherwise get from TEMP)
    if False:


        # Add calibration date (get from TEMP)
        if "TEMP" in ds:
            if hasattr(ds.TEMP, "sensor_calibration_date"):
                ds.attrs["instrument_calibration_date"] = ds.TEMP.attrs[
                    "sensor_calibration_date"
                ]

        # Remove sensor_serial_number and from variable attributes
        for varnm in ds:
            if 'sensor_serial_number' in ds[varnm].attrs:
                del ds[varnm].attrs['sensor_serial_number']
            if 'sensor_calibration_date' in ds[varnm].attrs:
                del ds[varnm].attrs['sensor_calibration_date']


    # Set a sample interval
    if "sample_interval" in hdict:
        if " seconds" in hdict["sample_interval"]:
            sample_rate_s = int(hdict["sample_interval"].split()[0])
            ds.attrs["time_coverage_resolution"] = time.seconds_to_ISO8601(
                sample_rate_s
            )

    # Replace the first line of the history string ("Data collection")
    # To reflect a *range* rather than a time point..
    if "history" in ds.attrs:
        sample_dates = (
            num2date(ds.TIME[0]).strftime("%Y-%m-%d")
            + " - "
            + num2date(ds.TIME[-1]).strftime("%Y-%m-%d")
            + ":"
        )
        original_date = ds.history.split()[0]
        ds.attrs["history"] = ds.attrs["history"].replace(
            original_date, sample_dates
        )

    # Change coordinate from scan_count to TIME
    ds = xr_funcs.swap_var_coord(ds, "scan_count", "TIME", drop_original=True)

    return ds
