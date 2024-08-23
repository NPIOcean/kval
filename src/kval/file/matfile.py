"""
KVAL.IO.MATFILE
Writing to and from .mat files

NOTE:
This is a copy of a function used on a specific project.

The mat_to_xr functions are specifically designed to read time series data.

-> Rather hardcoded, want to look into a (wrapper for a) general .mat reader
  and perhaps give mat_to_xr more specific names.

"""

import xarray as xr
from scipy.io import matlab
import numpy as np
import datetime
from matplotlib.dates import date2num
from kval.util import time
from typing import Tuple, Dict, Any, Union
import os

def mat_to_xr_1D(
    matfile: str,
    time_name: str = "time",
    epoch: str = "1970-01-01",
    verbose: bool = True,
    field_name: Union[str, bool] = None,
) -> xr.Dataset:
    """
    Converts a MATLAB .mat file to an `xarray.Dataset` with 1D time data.

    Args:
        matfile (str): Path to the MATLAB .mat file.
        time_name (str): Name of the variable in the .mat file that contains
        time data. Must be in MATLAB datenum format (numeric array).
        epoch (str): Reference date for converting MATLAB datenum to numerical
        time. Defaults to '1970-01-01'. Not used in this function but
        provided for future expansion.
        verbose (bool): If True, prints warnings when parsing issues occur.
        Defaults to True.

    Returns:
        xr.Dataset: An `xarray.Dataset` with a 1D time coordinate and data
        variables from the .mat file.

    Notes:
        - 0-D variables are treated as metadata and added as global attributes.
        - Assumes time variable is in MATLAB datenum format.
        - Warnings printed if time variable cannot be parsed and `verbose` is
        True.
        - Data variables are added with a "TIME" coordinate.
        - Dataset is sorted chronologically based on the "TIME" coordinate.
    """
    # Read data/metadata from matfile
    data_dict, attr_dict, multiple_data_fields = _parse_matfile_to_dict(
        matfile, verbose=False)

    if multiple_data_fields:
        if field_name:
            data_dict = data_dict[field_name]
            attr_dict = attr_dict[field_name]
        else:
            raise Exception(
                f'Failed to read from {os.path.basename(matfile)} to xarray'
                ' Dataset beacuse the .mat file contains multiple data fields:'
                f'{data_dict.keys()}.\n'
                'Specify which field you want to load using the `field_name` '
                'parameter in matfile.mat_to_xr')

    # (Attempt to) parse time
    try:
        time_stamp = _parse_time(data_dict, time_name=time_name)
        time_num = date2num(time_stamp)

        # Remove time variable if we successfully parsed time
        data_dict.pop(time_name)
    except:
        raise Exception(
            f'NOTE: Unable to parse time from the "{time_name}" variable. '
            "\nTo try reading time from another variable, use the time_name flag.")

    # Collect in an xr Dataset
    ds = xr.Dataset(coords={"TIME": time_num})

    # Add data variables
    for varnm in data_dict:
        try:
            ds[varnm] = (("TIME"), data_dict[varnm])
        except:
            if verbose:
                print(
                    f'NOTE: Could not parse the variable "{varnm}" '
                    f" with shape: {data_dict[varnm].shape} as a TIME variable"
                    f' - expected shape ({ds.sizes["TIME"]}). '
                    " -> Skipping this variable."
                )
    # Add metadata
    for attrnm in attr_dict:
        ds.attrs[attrnm] = attr_dict[attrnm]

    # Sort chronologically
    ds = ds.sortby("TIME")

    return ds


def mat_to_xr_2D(
    matfile: str,
    time_name: str = "time",
    dim2_name_in: str = "PRES",
    dim2_name_out: str = "PRES",
    field_name: Union[str, bool] = None,
    epoch: str = "1970-01-01",
) -> xr.Dataset:
    """
    Convert a MATLAB .mat file with 2D variables to an xarray Dataset.

    Assumes two dimensions: time and a depth/pressure-type variable.

    Parameters:
        matfile (str): Path to the MATLAB file.
        time_name (str): Name of the time variable in the MATLAB file
            (default is 'time').
        dim2_name_in (str): Name of the second variable in the MATLAB file
            (default is 'PRES').
        dim2_name_out (str): Name for the second variable in the
            xarray Dataset (default is 'PRES').
        epoch (str): Reference date for time conversion (default is
            '1970-01-01').

    Returns:
        xr.Dataset: xarray Dataset with converted data.

    Notes:
        - 0-D variables are stored as metadata.
        - Time variable must be in MATLAB datenum format (e.g., [737510.3754]).
        - If time parsing fails, it prints a message and retains the variable.
        - Variables not fitting expected dimensions are skipped with a
          message.
        - Metadata from the MATLAB file is added as global attributes.
        - Dataset is sorted chronologically by time.
    """

    # Read data/metadata from matfile
    data_dict, attr_dict, multiple_data_fields = _parse_matfile_to_dict(
        matfile, verbose=False)

    if multiple_data_fields:
        if field_name:
            data_dict = data_dict[field_name]
            attr_dict = attr_dict[field_name]
        else:
            raise Exception(
                f'Failed to read from {os.path.basename(matfile)} to xarray'
                ' Dataset beacuse the .mat file contains multiple data fields:'
                f'{data_dict.keys()}.\n'
                'Specify which field you want to load using the `field_name` '
                'parameter in matfile.mat_to_xr')

    # (Attempt to) parse time
    try:
        time_stamp = _parse_time(data_dict, time_name=time_name)
        time_num = date2num(time_stamp)

        # Remove time variable if we successfully parsed time
        data_dict.pop(time_name)
    except:
        raise Exception(
            f'NOTE: Unable to parse time from the "{time_name}" variable. '
            "\nTo try reading time from another variable, use the time_name flag.")

    # Check whether dim2_name_in actually exists in the dataset
    if dim2_name_in not in data_dict.keys():
        raise Exception(
            f'NOTE: Could not find a "{dim2_name_in}" variable. '
            "\nTo use another variable as the second dimension, "
            "use the dim2_name_in flag.")

    # Collect in an xr Dataset
    ds = xr.Dataset(
        coords={"TIME": time_num, dim2_name_out: data_dict[dim2_name_in]}
    )

    # Add data variables
    # (Assigning the coordinates by looking at the dimensionality of the fields)
    for varnm, item in data_dict.items():
        dshape = data_dict[varnm].shape
        if dshape == (ds.sizes["TIME"],):
            ds[varnm] = (("TIME"), data_dict[varnm])
        elif dshape == (ds.sizes[dim2_name_out],):
            ds[varnm] = ((dim2_name_out), data_dict[varnm])
        elif dshape == (ds.sizes["TIME"], ds.sizes[dim2_name_out]):
            ds[varnm] = (("TIME", dim2_name_out), data_dict[varnm])
        elif dshape == (ds.sizes[dim2_name_out], ds.sizes["TIME"]):
            ds[varnm] = (("TIME", dim2_name_out), data_dict[varnm].T)
        else:
            print(
                f"NOTE: Trouble with variable {varnm} (shape: "
                f"{data_dict[varnm].shape})- does not seem "
                f'to fit into either TIME = ({ds.sizes["TIME"]}) '
                f" or {dim2_name_out} ({ds.sizes[dim2_name_out]})."
                "\n-> Skipping this variable"
            )

    # Add metadata
    for attrnm, item in attr_dict.items():
        ds.attrs[attrnm] = attr_dict[attrnm]

    # Sort chronologically
    ds = ds.sortby("TIME")

    return ds


def _parse_matfile_to_dict(
    matfile: str,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Parse a MATLAB .mat file into dictionaries of data and attributes.

    Skips variables with internal Matlab types and handles up to three levels
    of nesting. Assumes the file is in a format before MATLAB v7.3.

    Parameters:
        matfile (str): Path to the .mat file.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]:
            - data_dict: Dictionary of data variables.
            - attr_dict: Dictionary of global attributes (metadata).

    Raises:
        Exception: If multiple or no data keys are found in the .mat file.
    """

    # 1. LOAD DATA TO PYTHON
    matfile_dict = matlab.loadmat(matfile, squeeze_me=True)
    matfile_base = os.path.basename(matfile)

    # 2. IDENTIFY THE DICTIONARY CONTAINING THE DATA
    # (By default also loads a bunch of non-useful metadata)

    # Identify the name of the structure containing the data
    # (assuming this is the only field not on the format '__[]__')
    data_key_list = []

    for dict_key in matfile_dict.keys():
        if not dict_key.startswith("__"):
            data_key_list += [dict_key]

    # If there is more than one: Return an exception fo now (not sure whether
    # this is an issue)
    if len(data_key_list) > 1:
        if verbose:
            print(f'Found multiple data keys in {matfile_base}:\n{data_key_list}'
              '\n-> Will return nested dictionaries .')
        data_dicts, attr_dicts = {}, {}
        multiple_data_fields = True
    # If there is no apparent data field: Raise an exception
    elif len(data_key_list) == 0:
        raise Exception(
            f"No valid data key found in {matfile_dict}:\n"
            f"-> Inspect your matfile!"
        )
    else:
        multiple_data_fields = False
    for data_key in data_key_list:

        # Load the dictionary containing the data

        ddict = matfile_dict[data_key]

        # 3. SORT VARIABLES INTO DAtA AND ATTRIBUTE ARRAYS

        variables = ddict.dtype.names
        data_dict = {}
        attr_dict = {}

        for varnm in variables:
            parsed_variable = ddict[varnm].flatten()[0]

            # If the object is (float, int, str, etc): Interpret is as a global
            # attribute
            if isinstance(parsed_variable, (int, float, str)):
                attr_dict[varnm] = parsed_variable

            # This is none if ddict[varnm] is an array with data or similar
            # -> assign a variable
            elif parsed_variable.dtype.fields is None:

                parsed_data = parsed_variable

                # If the variable is of type MatlabOpaque, it cannot be read
                # outside MATLAB -> Print a message and skip the variable
                if isinstance(parsed_data, matlab._mio5_params.MatlabOpaque):
                    if verbose:
                        print(
                            f'NOTE: "{varnm}" is of an internal Matlab class'
                            " type (e.g. Datetime) "
                            "which cannot be read outside Matlab -> Skip"
                        )

                # If the object is iterable (list, array, etc): Interpret as
                # a data variable
                elif isinstance(parsed_data, (list, tuple, np.ndarray)):
                    data_dict[varnm] = parsed_data

            # If ddict[varnm] is an array containing other names variables:
            # parse them -> Loop through variables and add them individualy as
            # variables
            else:
                for varnm_internal in parsed_variable.dtype.names:
                    parsed_data = parsed_variable[varnm_internal].flatten()[0]

                    # If the variable is of type MatlabOpaque, it cannot be
                    # read outside MATLAB -> Print a message and skip the
                    # variable
                    if isinstance(parsed_data,
                                  matlab._mio5_params.MatlabOpaque):
                        if verbose:
                            print(
                                f'NOTE: "{varnm_internal}" is of an internal '
                                "Matlab class type (e.g. Datetime) which cannot "
                                "be +read outside Matlab -> Skip"
                            )

                    # If the object is iterable (list, array, etc): Interpret as
                    # a data variable
                    elif isinstance(parsed_data, (list, tuple, np.ndarray)):
                        data_dict[varnm_internal] = parsed_data

                    # Otherwise (float, int, str, etc): Interpret is as a
                    # global attribute
                    elif isinstance(parsed_data, (int, float, str)):
                        attr_dict[varnm_internal] = parsed_data

        # 4. RETURN OUTPUT
        if len(data_key_list) == 1:
            return data_dict, attr_dict, multiple_data_fields
        else:
            data_dicts[data_key] = data_dict
            attr_dicts[data_key] = attr_dict

    return data_dicts, attr_dicts, multiple_data_fields


def _parse_time(data_dict, time_name="time"):
    """
    Parse Matlab time read to a dictionary data_dict using
    parse_matfile_to_dict.

    Currently only works for time on the matlab datenum format. May want to
    expand to other formats like [yr, mo .. min, sec]. On the other hand, it
    might be cleanest to just require the fields
    """
    try:
        time_stamps = time.matlab_time_to_datetime(data_dict[time_name])
        return time_stamps
    except:  # May have to build other cases here, eventually.
        print(
            f'Unable to parse time from the "{time_name}" variable. '
            "(Expecting existing variable and Matlab datenum format)"
        )



def xr_to_mat(D, outfile, simplify=False):
    """
    Convert an xarray.Dataset to a MATLAB .mat file.

    A field 'TIME_mat' with Matlab datenums is added along with the data.

    Parameters:
    - D (xarray.Dataset): Input dataset to be converted.
    - outfile (str): Output file path for the MATLAB .mat file. If the path
      doesn't end with '.mat', it will be appended.
    - simplify (bool, optional): If True, simplify the dataset by extracting
      only coordinate and data variables (no metadata attributes). If False,
      the matfile will be a struct containing [attrs, data_vars, coords, dims].
      Defaults to False.

    Returns:
    None: The function saves the dataset as a MATLAB .mat file.

    Example:
    >>> xr_to_mat(D, 'output_matfile', simplify=True)
    """

    time_epoch = D.TIME.units.upper().replace("DAYS SINCE ", "")[:11]
    time_stamp = time.datenum_to_timestamp(D.TIME, D.TIME.units)
    time_mat = time.timestamp_to_matlab_time(time_stamp)

    data_dict = D.to_dict()

    if simplify:
        ds = {}
        for sub_dict_name in ["coords", "data_vars"]:
            sub_dict = data_dict[sub_dict_name]
            for varnm, item in sub_dict.items():
                ds[varnm] = sub_dict[varnm]["data"]

        ds["TIME_mat"] = time_mat
        data_dict = ds
        simple_str = " (simplified)"
    else:
        data_dict["coords"]["TIME_mat"] = time_mat
        simple_str = ""

    if not outfile.endswith(".mat"):
        outfile += ".mat"

    matlab.savemat(outfile, data_dict)
    print(f"Saved the{simple_str} Dataset to: {outfile}")
