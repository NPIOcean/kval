"""
Functions for making netcdfs cf-compliant.
(Working with xarray datasets)
"""

import cftime
from kval.metadata import _standard_attrs, _standard_attrs_org
from kval.calc import number
from kval.util import time, user_input
import numpy as np
import re
from collections import Counter
import pandas as pd
import xarray as xr
from typing import Optional


def nans_to_fill_value(ds, fill_value=-9999.0):
    '''
    Replace NaNs in any (non-coodinate) variables with a fill value,
    and update the _FillValue variable attributes accordingly.

    Only works on float values; leaves integer values alone.
    '''

    for key in list(ds.coords) + list(ds.data_vars):
        if np.issubdtype(ds[key].dtype, np.floating):  # Only for floats
            ds[key] = ds[key].fillna(fill_value)
            ds[key].attrs['_FillValue'] = fill_value

    return ds


def add_range_attrs(ds, vertical_var=None):
    """
    Add some global attributes based on the data.

    "vertical_var" specifies a coordinate from which to extract
    "geospatial_vertical_" parameters. If nothing is specified, we will
    look for a variable with attribute "axis":"Z".

    - LATITUDE, LONGITUDE variables are required in order to set
      geospatial range attributes
    - TIME variable is required in order to set
      time_coverage range attributes
    - Vertical coordinate variable (see above) isadd_standard_glob_attrs_ctd
      required in order to set geospatial_vertical range attributes
    """

    # Lateral
    try:
        ds.attrs["geospatial_lat_max"] = float(ds.LATITUDE.max().values)
        ds.attrs["geospatial_lon_max"] = float(ds.LONGITUDE.max().values)
        ds.attrs["geospatial_lat_min"] = float(ds.LATITUDE.min().values)
        ds.attrs["geospatial_lon_min"] = float(ds.LONGITUDE.min().values)
        ds.attrs["geospatial_bounds"] = _get_geospatial_bounds_wkt_str(ds)
        ds.attrs["geospatial_bounds_crs"] = "EPSG:4326"
    except (AttributeError, KeyError, ValueError) as e:
        print(
            "Did not find LATITUDE, LONGITUDE variables "
            '-> Could not set "geospatial" attributes.'
        )
        print(f"Error: {e}")

    # Vertical
    try:
        # If not specified: look for a variable with attribute *axis='Z'*
        if vertical_var is None:
            for varnm in list(ds.keys()) + list(ds.coords.keys()):
                if "axis" in ds[varnm].attrs:
                    if ds[varnm].attrs["axis"].upper() == "Z":
                        vertical_var = varnm

        # Make sure we dont include _FillValue (can have Nans for some aux
        # coordinates..)
        vertical_var_da = ds[vertical_var].copy(deep = True)
        if '_FillValue' in vertical_var_da.attrs:
            fill_value = vertical_var_da._FillValue
            vertical_var_da = vertical_var_da.where(
                vertical_var_da != fill_value)


        if vertical_var is not None:
            ds.attrs["geospatial_vertical_min"] = np.round(
                np.nanmin(vertical_var_da.values), 2)
            ds.attrs["geospatial_vertical_max"] = np.round(
                np.nanmax(vertical_var_da.values), 2)
            ds.attrs["geospatial_vertical_units"] = vertical_var_da.units
            ds.attrs["geospatial_vertical_positive"] = "down"
            ds.attrs["geospatial_bounds_vertical_crs"] = "EPSG:5831"

    except (AttributeError, KeyError, ValueError) as e:
        print(
            "Did not find vertical variable "
            '-> Could not set "geospatial_vertical" attributes.')
        print(f"Error: {e}")

    # Time
    try:
        if ds.TIME.dtype == np.dtype("datetime64[ns]"):
            start_time, end_time = ds.TIME.min(), ds.TIME.max()
        else:
            start_time = cftime.num2date(ds.TIME.min().values, ds.TIME.units)
            end_time = cftime.num2date(ds.TIME.max().values, ds.TIME.units)
        ds.attrs["time_coverage_start"] = time.datetime_to_ISO8601(start_time)
        ds.attrs["time_coverage_end"] = time.datetime_to_ISO8601(end_time)

        # Keep time_coverage_resolution if already present
        if "time_coverage_resolution" not in ds.attrs:
            tdiff_std = np.std(np.diff(ds.TIME))
            tdiff_median = np.median(np.diff(ds.TIME))
            if tdiff_std / tdiff_median < 1e-5:  # <- Looks like fixed interval
                ds.attrs["time_coverage_resolution"] = time.days_to_ISO8601(
                    tdiff_median
                )
            else:  # <- Looks like a variable interval
                ds.attrs["time_coverage_resolution"] = "variable"

        ds.attrs["time_coverage_duration"] = _get_time_coverage_duration_str(
            ds)
    except (AttributeError, KeyError, ValueError) as e:
        print(
            'Did not find TIME variable (or TIME variable lacks "units" '
            'attribute)\n-> Could not set "time_coverage" attributes.'
        )
        print(f"Error: {e}")

    return ds


def add_now_as_date_created(ds):
    """
    Add a global attribute "date_created" with todays date.
    """
    now_time = pd.Timestamp.now()
    now_str = time.datetime_to_ISO8601(now_time)

    ds.attrs["date_created"] = now_str

    return ds


def reorder_attrs(ds):
    """
    Reorder global and variable attributes of a dataset based on the
    specified order in _standard_attrs.

    Parameters:
        ds (xarray.Dataset): The dataset containing global attributes.
        ordered_list (list): The desired order of global attributes.

    Returns:
        xarray.Dataset: The dataset with reordered global attributes.
    """
    # GLOBAL
    reordered_list = _reorder_list(
        ds.attrs, _standard_attrs.global_attrs_ordered
    )
    attrs_dict = ds.attrs
    ds.attrs = {}
    for attr_name in reordered_list:
        ds.attrs[attr_name] = attrs_dict[attr_name]

    # VARIABLE
    for varnm in ds.data_vars:
        reordered_list_var = _reorder_list(
            ds[varnm].attrs, _standard_attrs.variable_attrs_ordered
        )
        var_attrs_dict = ds[varnm].attrs
        ds[varnm].attrs = {}
        for attr_name in reordered_list_var:
            ds[varnm].attrs[attr_name] = var_attrs_dict[attr_name]
    return ds


def remove_numbers_in_var_names(ds):
    """
    Remove numbers from variable names like "TEMP1", "PSAL2".

    If more than one exists (e.g. "TEMP1", "TEMP2") -> don't change anything.
    """
    # Get variable names
    all_varnms = [varnm for varnm in ds.data_vars]

    # Get number-stripped names
    varnms_stripped = [re.sub(r"\d", "", varnm) for varnm in all_varnms]

    # Identify duplicates
    counter = Counter(varnms_stripped)
    duplicates = [item for item, count in counter.items() if count > 1]

    # Strip names
    for varnm in all_varnms:
        if re.sub(r"\d", "", varnm) not in duplicates:
            varnm_stripped = re.sub(r"\d", "", varnm)
            ds = ds.rename_vars({varnm: varnm_stripped})

    return ds


def _reorder_list(input_list, ordered_list):
    """
    Reorder a list input_list according to the order specified in ordered_list
    """
    # Create a set of existing attributes for quick lookup
    existing_attributes = set(input_list)

    # Extract ordered attributes that exist in the dataset
    ordered_attributes = [
        attr for attr in ordered_list if attr in existing_attributes
    ]

    # Add any remaining attributes that are not in the ordered list
    remaining_attributes = [
        attr for attr in input_list if attr not in ordered_attributes
    ]

    # Concatenate the ordered and remaining attributes
    reordered_attributes = ordered_attributes + remaining_attributes

    return reordered_attributes


def _get_geospatial_bounds_wkt_str(ds, decimals=2):
    """
    Get the geospatial_bounds_crs value on the required format:

    'POLYGON((x1 y1, x2 y2, x3 y3, …, x1 y1))'

    Rounds the lat/lon to the specified number of digits, rounding
    the last digit in the direction such that all points are contained
    within the polygon

    NOTE: ds must already have geospatial_lat_max (etc) attributes.

    """

    lat_max = number.custom_round_ud(ds.geospatial_lat_max, decimals, "up")
    lat_min = number.custom_round_ud(ds.geospatial_lat_min, decimals, "dn")
    lon_max = number.custom_round_ud(ds.geospatial_lon_max, decimals, "up")
    lon_min = number.custom_round_ud(ds.geospatial_lon_min, decimals, "dn")

    corner_dict = (
        lon_min,
        lat_min,
        lon_min,
        lat_max,
        lon_max,
        lat_max,
        lon_max,
        lat_min,
        lon_min,
        lat_min,
    )
    wkt_str = "POLYGON ((%s %s, %s %s, %s %s, %s %s, %s %s))" % corner_dict

    return wkt_str


def _get_time_coverage_duration_str(ds):
    """
    Get the time duration based on first and last time stamp on
    the required ISO8601format (P[YYYY]-[MM]-[DD]T[hh]:[mm]:[ss])
    """
    start_dt, end_dt = cftime.num2date(
        [ds.TIME[0], ds.TIME[-1]], ds.TIME.units)
    duration_str = time.start_end_times_cftime_to_duration(start_dt, end_dt)
    return duration_str


def add_standard_var_attrs(
    ds: xr.Dataset, override: bool = False, data_type: Optional[str] = None
) -> xr.Dataset:
    """
    Add variable attributes to an xarray Dataset, as specified in
    kval.metadata._standard_attrs.

    If a variable has a "_std" suffix:
        - Do not use standard_name, valid_min, or valid_max.
        - Add "Standard deviation of" to the long_name attribute.

    If a variable has a number suffix (e.g., TEMP1, CNDC2):
        - Copy attributes from the base variable (e.g., TEMP).
        - Append "(primary)", "(secondary)", etc., to the long_name attribute.

    Parameters:
    -----------
    ds : xr.Dataset
        The xarray Dataset to modify.
    override : bool, optional
        If True, existing variable attributes will be overwritten. Defaults to
        False.
    data_type : str, optional
        The type of data (e.g., 'ctdprof', 'moored') used to infer additional
        attributes.

    Returns:
    --------
    ds : xr.Dataset
        The modified xarray Dataset with updated variable attributes.
    """

    for varnm in list(ds.data_vars) + list(ds.coords):

        # Handling number suffix (e.g. TEMP1, PSAL2)
        try:
            number = float(
                varnm[-1]
            )  # If not a number, will raise an exception
            core_name = varnm[:-1]
            if number == 1:
                long_name_add = " (primary sensor)"
            elif number == 2:
                long_name_add = " (secondary sensor)"
            elif number == 3:
                long_name_add = " (tertiary sensor)"
            else:
                long_name_add = ""
        except ValueError:
            long_name_add = ""
            core_name = varnm

        # Remove suffix for core name (e.g. "CNDC" from "CNDC_CTD")
        core_name = core_name.split("_")[0]

        # Add standard variable attributes if found in _standard_attrs
        if core_name in _standard_attrs.standard_var_attrs:
            var_attrs_dict = _standard_attrs.standard_var_attrs[
                core_name
            ].copy()
            for attr, item in var_attrs_dict.items():
                if override or attr not in ds[varnm].attrs:
                    ds[varnm].attrs[attr] = item

            # Append suffix (e.g., "(primary sensor)") to long_name
            if "long_name" in ds[varnm].attrs:
                ds[varnm].attrs["long_name"] += long_name_add

        # Handle specific case for .btl files
        if "source_files" in ds.attrs and "long_name" in ds[varnm].attrs:
            if ds.source_files[-3:].upper() == "BTL" and data_type == "ctdprof":
                long_name = ds[varnm].attrs["long_name"]
                long_name_nocap = long_name[0].lower() + long_name[1:]
                if (
                    varnm != "NISKIN_NUMBER"
                    and "NISKIN_NUMBER" in ds[varnm].dims
                ):
                    ds[varnm].attrs["long_name"] = (
                        f"Average {long_name_nocap} measured by CTD "
                        "during bottle closing")

        # Variables with _std suffix
        if varnm.endswith("_std") and data_type == "ctdprof":
            varnm_prefix = varnm.replace("_std", "")
            if varnm_prefix in _standard_attrs.standard_var_attrs:
                var_attrs_dict = _standard_attrs.standard_var_attrs[
                    varnm_prefix
                ].copy()

                # Do not copy specific attributes for _std variables
                for key in ["standard_name", "valid_min", "valid_max"]:
                    var_attrs_dict.pop(key, None)

                # Add remaining attributes
                for attr, item in var_attrs_dict.items():
                    if override or attr not in ds[varnm].attrs:
                        ds[varnm].attrs[attr] = item

            # Add "Standard deviation of" to long_name
            if "long_name" in ds[varnm].attrs:
                long_name = ds[varnm].attrs["long_name"]
                long_name_nocap = long_name[0].lower() + long_name[1:]
                ds[varnm].attrs[
                    "long_name"
                ] = f"Standard deviation of {long_name_nocap}"

            # Link ancillary variables
            ds[varnm].attrs["ancillary_variables"] = varnm_prefix
            ds[varnm_prefix].attrs["ancillary_variables"] = varnm

        # Guess coverage_content_type if not already present
        if not override and "coverage_content_type" not in ds[varnm].attrs:
            if varnm in ["NISKIN_NUMBER", "TIME", "TIME_SAMPLE"]:
                ds[varnm].attrs["coverage_content_type"] = "coordinate"
            elif varnm in ["LONGITUDE", "LATITUDE"]:
                ds[varnm].attrs["coverage_content_type"] = (
                    "referenceInformation")
            elif varnm in ["PROCESSING", "INSTRUMENT"]:
                ds[varnm].attrs["coverage_content_type"] = (
                    "auxiliaryInformation")
            else:
                ds[varnm].attrs["coverage_content_type"] = "physicalMeasurement"

        # Remove _FillValue from the PROCESSING variable
        if varnm == 'PROCESSING':
            if '_FillValue' in ds.PROCESSING.attrs:
                del ds.PROCESSING.attrs['_FillValue']

    return ds


def add_standard_glob_attrs_org(ds, override=False, org="npi"):
    """
    Adds standard organization, specific global variables for a CTD dataset as
    specified in kval.data.nc_format._standard_attrs_org.

    Includes  standard attribute values for things like "institution",
    "creator_name", etc.

    'org' is the organiztion (currently only 'npi' available)

    override: governs whether to override any global attributes that are
    already present (typically not advised..)
    """

    org_attrs = _standard_attrs_org.standard_globals_org[org.lower()]

    for attr, item in org_attrs.items():
        if attr not in ds.attrs:
            ds.attrs[attr] = item
        else:
            if override:
                ds.attrs[attr] = item

    return ds

def add_standard_glob_attrs_ctd(ds, override=False, org=False):
    """
    Adds standard global variables for a CTD dataset as specified in
    kval.metadata.standard_attrs_global_ctd.

    override: governs whether to override any global attributes that
    are already present (typically not advised..)
    """

    for attr, item in _standard_attrs.standard_attrs_global_ctd.items():
        if attr not in ds.attrs:
            ds.attrs[attr] = item
        else:
            if override:
                ds.attrs[attr] = item

    return ds

def add_standard_glob_attrs_moor(ds, override=False, org=None):
    """
    Adds standard global variables for a CTD dataset as specified in
    oceanograpy.data.nc_format.standard_attrs_global_moored.

    override: governs whether to override any global attributes that
    are already present (typically not advised..)
    """

    for attr, item in _standard_attrs.standard_attrs_global_moored.items():
        if attr not in ds.attrs:
            ds.attrs[attr] = item
        else:
            if override:
                ds.attrs[attr] = item

    return ds

def add_gmdc_keywords_ctd(ds, reset=True, moored=False):
    """
    Adds standard GMDC variables a la
    "OCEANS>OCEAN TEMPERATURE>WATER TEMPERATURE"
    reflecting the variables in the dictionary
    _standard_attrs.gmdc_keyword_dict_ctd
    """

    if moored:
        gmdc_dict = _standard_attrs.gmdc_keyword_dict_moored
    else:
        gmdc_dict = _standard_attrs.gmdc_keyword_dict_ctd
    keywords = []

    for varnm in ds.keys():
        for gmdc_kw in gmdc_dict:
            if gmdc_kw in varnm:
                keywords += [gmdc_dict[gmdc_kw]]

    unique_keywords = list(set(keywords))

    ds.attrs["keywords"] = ",".join(unique_keywords)

    return ds

def add_gmdc_keywords_moor(ds, reset=True):
    """
    Adds standard GMDC variables a la
    "OCEANS>OCEAN TEMPERATURE>WATER TEMPERATURE"
    reflecting the variables in the file
    """

    gmdc_dict = _standard_attrs.gmdc_keyword_dict_moored
    keywords = []

    for varnm in ds.keys():
        for gmdc_kw in gmdc_dict:
            if gmdc_kw in varnm:
                keywords += [gmdc_dict[gmdc_kw]]

    unique_keywords = list(set(keywords))

    ds.attrs["keywords"] = ",".join(unique_keywords)

    return ds

def set_var_attr(ds, variable, attr):
    """
    Set variable attributes for a dataset or data frame.

    Using interactive widgets.
    """

    # Check if the attribute already exists in ds
    if attr in ds[variable].attrs:
        initial_value = ds[variable].attrs[attr]
    else:
        initial_value = None

    option_dict = _standard_attrs.global_attrs_options
    if attr in option_dict:
        ds = user_input.set_var_attr_pulldown(
            ds,
            variable,
            attr,
            _standard_attrs.global_attrs_options[attr],
            initial_value=initial_value,
        )
    else:
        # Make a larger box for (usually) long attributes
        if attr in ["summary", "comment", "acknowledgment"]:
            rows = 10
        else:
            rows = 1
        ds = user_input.set_var_attr_textbox(
            ds, variable, attr, rows, initial_value=initial_value
        )

    return ds

def set_glob_attr(ds, attr):
    """
    Set global attributes for a  dataset or data frame.

    Using interactive widgets.

    """

    option_dict = _standard_attrs.global_attrs_options

    # Check if the attribute already exists in D
    if attr in ds.attrs:
        initial_value = ds.attrs[attr]
    else:
        initial_value = None

    if attr in option_dict:
        ds = user_input.set_attr_pulldown(
            ds,
            attr,
            _standard_attrs.global_attrs_options[attr],
            initial_value=initial_value,
        )
    else:
        # Make a larger box for (usually) long attributes
        if attr in ["summary", "comment", "acknowledgment"]:
            rows = 10
        else:
            rows = 1
        ds = user_input.set_attr_textbox(
            ds, attr, rows, initial_value=initial_value
        )

    return ds

def add_missing_glob(ds):
    """
    Prompts the user to fill in information for missing global attrubtes
    """

    var_attrs_dict_ref = _standard_attrs.variable_attrs_necessary.copy()

    for varnm in ds:
        if "PRES" in ds[varnm].dims or "NISKIN_NUMBER" in ds[varnm].dims:
            _attrs_dict_ref_var = var_attrs_dict_ref.copy()

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
                    set_var_attr(ds, varnm, var_attr)
                    any_missing = True

            if not any_missing:
                print(f"- {varnm}: OK")

    # Do this is if we define necessary global attributes:
    # glob_attrs_dict_ref = _standard_attrs.some_dict.copy()

    # for attr in glob_attrs_dict_ref:
    #    if attr not in ds.attrs:
    #        set_glob_attr(ds, attr)

    return ds