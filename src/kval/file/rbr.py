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
from matplotlib.dates import date2num
from kval.file._variable_defs import RBR_name_map, RBR_units_map
from kval.util import time



def read(file: str) -> xr.Dataset:
    """
    Parse an .rsk file with data from an RBR instrument into an xarray Dataset,
    preserving available metadata and converting units and variable names 
    according to specified conventions.

    Parameters:
    ----------
    file : str
        Path to the .rsk file.

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

        # Load data into a pandas DataFrame
        df_rsk = pd.DataFrame(rskdata.data)

        # Convert timestamp from datetime64[ms] to datetime64[ns]
        # (to squash a warning going from pd to xr; no practical implications)
        df_rsk['timestamp'] = df_rsk['timestamp'].astype('datetime64[ns]')

        # Set timestamp as the index (coordinate variable)
        df_rsk.set_index('timestamp', inplace=True)

        # Create an xarray Dataset from the pandas DataFrame
        ds_rsk = xr.Dataset.from_dataframe(df_rsk)

        # Retrieve channel names and units
        rsk_channel_names, rsk_channel_units = (
            rskdata.getchannelnamesandunits([]))

        # Modify units according to preferred formatting
        # (mS/cm -> mS cm-1, °C -> degC, etc..)

        updated_units = [RBR_units_map.get(unit, unit) 
                         for unit in rsk_channel_units]

        # Map RBR names to updated units
        map_var_units = dict(zip(rsk_channel_names, updated_units))

        # Set units of each variable
        for key, unit in map_var_units.items():
            if key in ds_rsk:
                ds_rsk[key].attrs['units'] = unit

        # Rename the timestamp dimension to TIME
        ds_rsk = ds_rsk.rename_dims({'timestamp': 'TIME'})

        # Update variable names according to conventions
        # (salinity -> PSAL, sea_pressure -> PRES, etc):
        # Filter out variables not present in the dataset
        filtered_RBR_name_map = {
            old_name: new_name for old_name, new_name in RBR_name_map.items()
            if old_name in ds_rsk.variables or old_name in ds_rsk.coords
        }
        # Change names
        ds_rsk = ds_rsk.rename_vars(filtered_RBR_name_map)

        # Convert TIME to Python epoch format and update attributes
        ds_rsk['TIME'] = ('TIME', date2num(ds_rsk['TIME'].values))
        ds_rsk['TIME'].attrs['units'] = 'days since 1970-01-01'
        ds_rsk['TIME'].attrs['axis'] = 'T'

        # Add metadata
        ds_rsk.attrs['instrument'] = rskdata.instrument.model
        ds_rsk.attrs['instrument_serial_number'] = rskdata.instrument.serialID
        ds_rsk.attrs['time_coverage_resolution'] = time.seconds_to_ISO8601(
            rskdata.scheduleInfo.samplingperiod())

        # Optional metadata
        if False:  # Change to `True` if you want to include this metadata
            ds_rsk.attrs['calibrations'] = rskdata.calibrations  # Useful for calibration dates
            ds_rsk.attrs['filename'] = rskdata.filename
            ds_rsk.attrs['firmware_version'] = rskdata.instrument.firmwareVersion

        return ds_rsk
    