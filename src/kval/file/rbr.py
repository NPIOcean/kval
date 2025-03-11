'''
kval.file.rbr: Functions for parsing data from RBR instruments.

This module provides functions for parsing data from RBR instruments,
leveraging the pyRSKtools package developed by RBR
(https://pypi.org/project/pyRSKtools/).

The functions in this module call and extend the functionality of the
pyRSKtools library to integrate with the kval ecosystem.

Licensing Information:
----------------------
This module uses the pyRSKtools package, which is licensed under the
Apache License 2.0. A copy of the Apache License is included in the
root directory of this project (`LICENSE-APACHE`).

Note: All original code in this module is licensed under the MIT License.
For full licensing information, refer to the `LICENSE` file in the root
directory of this project.
"""
'''

import pyrsktools
import xarray as xr
import pandas as pd
from matplotlib.dates import date2num, num2date
from kval.file._variable_defs import RBR_name_map, RBR_units_map
from kval.util import time
from datetime import datetime
import os


def read_rsk(file: str, keep_total_pres: bool = False) -> xr.Dataset:
    """
    Parse an .rsk file with data from an RBR instrument into an xarray Dataset,
    preserving available metadata and converting units and variable names
    according to specified conventions.

    Parameters:
    ----------
    file : str
        Path to the .rsk file.
    keep_total_pres : bool
        Retain the full pressure if we calculate sea pressure (pressure-p_atm).
        Default is False.
    Returns:
    -------
    xr.Dataset
        An xarray Dataset containing the parsed data with updated units
        and variable names.
    """
    with pyrsktools.RSK(file) as rskdata:

        # Open file and read data
        rskdata.open()
        rskdata.readdata()

        if rskdata.channelexists("conductivity"):
            rskdata.derivesalinity()
        if rskdata.channelexists("pressure"):
            rskdata.deriveseapressure()
            p_atm = _extract_patm(rskdata)

        # Load data into a pandas DataFrame
        df_rsk = pd.DataFrame(rskdata.data)

        # Convert timestamp from datetime64[ms] to datetime64[ns]
        # (to squash a warning going from pd to xr; no practical implications)
        df_rsk["timestamp"] = df_rsk["timestamp"].astype("datetime64[ns]")

        # Set timestamp as the index (coordinate variable)
        df_rsk.set_index("timestamp", inplace=True)

        # Create an xarray Dataset from the pandas DataFrame
        ds_rsk = xr.Dataset.from_dataframe(df_rsk)

        # Retrieve channel names and units
        rsk_channel_names, rsk_channel_units = rskdata.getchannelnamesandunits(
            []
        )

        # Add instrument metadata
        ds_rsk.attrs["instrument_model"] = rskdata.instrument.model
        ds_rsk.attrs["instrument_serial_number"] = rskdata.instrument.serialID

        # Sampling scheme and time coverage resolution
        # (Stored a bit strangely in the rskdata object. Times often in ms.)
        if type(rskdata.scheduleInfo) == pyrsktools.datatypes.ContinuousInfo:
            time_res_seconds = rskdata.scheduleInfo.samplingPeriod / 1000
            sampling_str = (f'Continuous sampling - one measurement every'
                            f' {time_res_seconds/60} min')
        if type(rskdata.scheduleInfo) == pyrsktools.datatypes.AverageInfo:
            time_res_seconds = rskdata.scheduleInfo.repetitionPeriod / 1000
            sampling_str = (
                f'One average of {rskdata.scheduleInfo.samplingCount} samples '
                f'collected at {rskdata.scheduleInfo.samplingPeriod/1000} sec '
                f'intervals stored for every {time_res_seconds/60} min.')

        ds_rsk.attrs["time_coverage_resolution"] = time.seconds_to_ISO8601(
            time_res_seconds
        )
        ds_rsk.attrs["sampling_details"] = sampling_str


        # Add calibration dates:
        cal_dates = _build_cal_dates(rskdata)
        for varnm, cdate in cal_dates.items():
            if varnm in ds_rsk:
                ds_rsk[varnm].attrs["sensor_calibration_date"] = cdate

        # Add some variable-custom metadata
        if "sea_pressure" in ds_rsk:
            # Atmospheric pressure used in PRES calculation
            ds_rsk["sea_pressure"].attrs[
                "assumed_atmospheric_pressure_dbar"
            ] = p_atm
            # Calibration data of pressure sensor
            ds_rsk["sea_pressure"].attrs[
                "sensor_calibration_date"
            ] = ds_rsk.pressure.sensor_calibration_date

        if "salinity" in ds_rsk:
            # Calibration data of T/C sensors
            ds_rsk["salinity"].attrs["sensor_calibration_date"] = (
                f"{ds_rsk.temperature.sensor_calibration_date} (TEMP), "
                f"{ds_rsk.conductivity.sensor_calibration_date} (CNDC),"
            )

        # Drop the total pressure if we have sea pressure
        # (and have not set keep_total_pres=True)
        if keep_total_pres is False and "sea_pressure" in ds_rsk:
            ds_rsk = ds_rsk.drop_vars("pressure")

        # Modify units according to preferred formatting
        # (mS/cm -> mS cm-1, °C -> degC, etc..)
        updated_units = [
            RBR_units_map.get(unit, unit) for unit in rsk_channel_units
        ]

        # Map RBR names to updated units
        map_var_units = dict(zip(rsk_channel_names, updated_units))

        # Set units of each variable
        for key, unit in map_var_units.items():
            if key in ds_rsk:
                ds_rsk[key].attrs["units"] = unit

        # Rename the timestamp dimension to TIME
        ds_rsk = ds_rsk.rename_dims({"timestamp": "TIME"})

        # Update variable names according to conventions
        # (salinity -> PSAL, sea_pressure -> PRES, etc):
        # Filter out variables not present in the dataset
        filtered_RBR_name_map = {
            old_name: new_name
            for old_name, new_name in RBR_name_map.items()
            if old_name in ds_rsk.variables or old_name in ds_rsk.coords
        }
        # Change names
        ds_rsk = ds_rsk.rename_vars(filtered_RBR_name_map)

        # Convert TIME to Python epoch format and update attributes
        ds_rsk["TIME"] = ("TIME", date2num(ds_rsk["TIME"].values))
        ds_rsk["TIME"].attrs["units"] = "days since 1970-01-01"
        ds_rsk["TIME"].attrs["axis"] = "T"

        # Add a history attribute with some basic info
        first_date = num2date(ds_rsk.TIME.min()).strftime("%Y-%m-%d")
        last_date = num2date(ds_rsk.TIME.max()).strftime("%Y-%m-%d")
        now_date = pd.Timestamp.now().strftime("%Y-%m-%d")

        ds_rsk.attrs["history"] = (
            f"{first_date} - {last_date}: Data collection.\n"
        )
        ds_rsk.attrs["history"] += (
            f"{now_date}: Data read from .rsk file to xarray Dataset"
            " using pyRSKtools+kval."
        )

        ds_rsk.attrs["source_file"] = os.path.basename(
            rskdata.filename
        )

        return ds_rsk


def _build_cal_dates(rskdata):
    """
    Build a dictionary of calibration dates from a rsk object

    Args:
        rskdata (object): The data object containing calibration and channel
        information.

    Returns:
        dict: A dictionary where keys are channel long names and values are
              calibration dates in 'YYYY-MM-DD' format.
    """
    # Create a mapping from channelID to longName for channels
    # where isDerived=0
    channel_id_to_longname = {
        channel.channelID: channel.longName
        for channel in rskdata.channels
        if channel.isDerived == 0
    }

    # Initialize the dictionary to hold calibration dates
    cal_dates = {}

    # Populate the cal_dates dictionary with calibration dates
    for calibration in rskdata.calibrations:
        # Get the long name of the channel associated with the calibration
        long_name = channel_id_to_longname.get(calibration.channelOrder)

        if long_name:
            # Convert numpy.datetime64 to a Python datetime object
            cal_date = calibration.tstamp.astype(datetime)

            # Format the datetime object to 'YYYY-MM-DD' string
            cal_date_str = cal_date.strftime("%Y-%m-%d")

            # Add the long name and calibration date to the dictionary
            cal_dates[long_name] = cal_date_str

    return cal_dates


def _extract_patm(rskdata):
    """
    Extract the atmospheric pressure used for the most recent sea pressure
    calculation.

    (Want this information in metadata)

    Args:
        rskdata_log (dict): A dictionary where keys are timestamps
        (numpy.datetime64) and values are log messages.

    Returns:
        float: The atmospheric pressure used in the most recent sea pressure
        calculation, or None if not found.
    """
    # Initialize variables
    latest_timestamp = None
    p_atm = None

    # Iterate over log entries to find the most recent sea pressure calculation
    for timestamp, message in rskdata.logs.items():
        if (
            "Sea pressure calculated using an atmospheric pressure of"
            in message
        ):
            # Update latest timestamp and extract pressure
            latest_timestamp = timestamp
            # Extract pressure value from the message
            pressure_str = message.split("atmospheric pressure of ")[1].split(
                " dbar"
            )[0]
            p_atm = float(pressure_str)

    return p_atm
