'''
Functions for making netcdfs cf-compliant.
(Working with xarray datasets)
'''
from kval.util import time
import cftime
from kval.metadata import _standard_attrs, _standard_attrs_org
from kval.calc import numbers
from kval.util import time, user_input
import numpy as np
import re
from collections import Counter


def add_range_attrs(D, vertical_var = None):
    '''
    Add some global attributes based on the data.

    "vertical_var" specifies a coordinate from which to extract
    "geospatial_vertical_" parameters. If nothing is specified, we will
    look for a variable with attribute "axis":"Z".

    - LATITUDE, LONGITUDE variables are required in order to set
      geospatial range attributes
    - TIME variable is required in order to set
      time_coverage range attributes
    - Vertical coordinate variable (see above) isadd_standard_glob_attrs_ctd required in order to set
      geospatial_vertical range attributes
    '''

    # Lateral
    try:
        D.attrs['geospatial_lat_max'] = D.LATITUDE.max().values
        D.attrs['geospatial_lon_max'] = D.LONGITUDE.max().values
        D.attrs['geospatial_lat_min'] = D.LATITUDE.min().values
        D.attrs['geospatial_lon_min'] = D.LONGITUDE.min().values
        D.attrs['geospatial_bounds'] = _get_geospatial_bounds_wkt_str(D)
        D.attrs['geospatial_bounds_crs'] = 'EPSG:4326'
    except:
        print('Did not find LATITUDE, LONGITUDE variables '
              '-> Could not set "geospatial" attributes.')
        

    # Vertical

    try:
        # If not specified: look for a variable with attribute *axis='Z'*
        if vertical_var == None:
            for varnm in list(D.keys()) + list(D.coords.keys()):
                if 'axis' in D[varnm].attrs:
                    if D[varnm].attrs['axis'].upper() == 'Z':
                        vertical_var = varnm

        if vertical_var != None:

            D.attrs['geospatial_vertical_min'] = D[vertical_var].min().values
            D.attrs['geospatial_vertical_max'] = D[vertical_var].max().values
            D.attrs['geospatial_vertical_units'] = D[vertical_var].units

            D.attrs['geospatial_vertical_positive'] = 'down'
            D.attrs['geospatial_bounds_vertical_crs'] = 'EPSG:5831'
    except:
        print('Did not find vertical variable '
        '-> Could not set "geospatial_vertical" attributes.')
    # Time
    try:
        if D.TIME.dtype==np.dtype('datetime64[ns]'):
            start_time, end_time = D.TIME.min(), D.TIME.max()
        else:
            start_time = cftime.num2date(D.TIME.min().values, D.TIME.units)
            end_time = cftime.num2date(D.TIME.max().values, D.TIME.units)
        D.attrs['time_coverage_start'] = time.datetime_to_ISO8601(start_time)
        D.attrs['time_coverage_end'] = time.datetime_to_ISO8601(end_time)
        D.attrs['time_coverage_resolution'] = 'variable' ## Note: Should not be variable for fixed sampling rate instruments!!! -> Fix this.
        D.attrs['time_coverage_duration'] = _get_time_coverage_duration_str(D)
    except:
        print('Did not find TIME variable (or TIME variable lacks "units" attribute)'
        '\n-> Could not set "time_coverage" attributes.')
    return D



def add_now_as_date_created(D):
    '''
    Add a global attribute "date_created" with todays date.
    '''
    now_time = pd.Timestamp.now()
    now_str = time.datetime_to_ISO8601(now_time)

    D.attrs['date_created'] = now_str

    return D




def reorder_attrs(D):
    """
    Reorder global and variable attributes of a dataset based on the 
    specified order in _standard_attrs.

    Parameters:
        ds (xarray.Dataset): The dataset containing global attributes.
        ordered_list (list): The desired order of global attributes.

    Returns:
        xarray.Dataset: The dataset with reordered global attributes.
    """
    ### GLOBAL
    reordered_list = _reorder_list(D.attrs, 
                                  _standard_attrs.global_attrs_ordered)
    attrs_dict = D.attrs
    D.attrs = {}
    for attr_name in reordered_list:
        D.attrs[attr_name] = attrs_dict[attr_name]

    ### VARIABLE
    for varnm in D.data_vars:
        reordered_list_var = _reorder_list(D[varnm].attrs, 
                      _standard_attrs.variable_attrs_ordered)
        var_attrs_dict = D[varnm].attrs
        D[varnm].attrs = {}
        for attr_name in reordered_list_var:
            D[varnm].attrs[attr_name] = var_attrs_dict[attr_name]
    return D



def remove_numbers_in_var_names(D):
    '''
    Remove numbers from variable names like "TEMP1", "PSAL2".

    If more than one exists (e.g. "TEMP1", "TEMP2") -> don't change anything.
    '''
    # Get variable names
    all_varnms = [varnm for varnm in D.data_vars]

    # Get number-stripped names 
    varnms_stripped = [re.sub(r'\d', '', varnm) for varnm in all_varnms]

    # Identify duplicates 
    counter = Counter(varnms_stripped)
    duplicates = [item for item, count in counter.items() if count > 1]

    # Strip names
    for varnm in all_varnms:
        if re.sub(r'\d', '', varnm) not in duplicates:
            varnm_stripped = re.sub(r'\d', '', varnm)
            D = D.rename_vars({varnm:varnm_stripped})

    return D


def _reorder_list(input_list, ordered_list):
    '''
    Reorder a list input_list according to the order specified in ordered_list
    '''
    # Create a set of existing attributes for quick lookup
    existing_attributes = set(input_list)

    # Extract ordered attributes that exist in the dataset
    ordered_attributes = [attr for attr in ordered_list if attr in existing_attributes]

    # Add any remaining attributes that are not in the ordered list
    remaining_attributes = [attr for attr in input_list if attr not in ordered_attributes]

    # Concatenate the ordered and remaining attributes
    reordered_attributes = ordered_attributes + remaining_attributes

    return reordered_attributes




def _get_geospatial_bounds_wkt_str(D, decimals = 2):
    '''
    Get the geospatial_bounds_crs value on the required format:

    'POLYGON((x1 y1, x2 y2, x3 y3, …, x1 y1))'

    Rounds the lat/lon to the specified number of digits, rounding
    the last digit in the direction such that all points are contained
    within the polygon

    NOTE: D must already have geospatial_lat_max (etc) attributes.
    
    '''

    lat_max = numbers.custom_round_ud(D.geospatial_lat_max, decimals, 'up')
    lat_min = numbers.custom_round_ud(D.geospatial_lat_min, decimals, 'dn')
    lon_max = numbers.custom_round_ud(D.geospatial_lon_max, decimals, 'up')
    lon_min = numbers.custom_round_ud(D.geospatial_lon_min, decimals, 'dn')

    corner_dict = (lon_min, lat_min, lon_min, lat_max, lon_max, lat_max, 
                    lon_max, lat_min, lon_min, lat_min)
    wkt_str = ('POLYGON ((%s %s, %s %s, %s %s, %s %s, %s %s))'%corner_dict)

    return wkt_str


def _get_time_coverage_duration_str(D):
    '''
    Get the time duration based on first and last time stamp on 
    the required ISO8601format (P[YYYY]-[MM]-[DD]T[hh]:[mm]:[ss])
    '''
    start_dt, end_dt = cftime.num2date([D.TIME[0], D.TIME[-1]], D.TIME.units)
    duration_str = time.start_end_times_cftime_to_duration(start_dt, end_dt)
    return duration_str


def add_standard_var_attrs(D, ctd_prof = False, override = False):
    '''
    Add variable attributes 
    as specified in oceanograpy.data.nc_format.standard_var_attrs

    if _std suffix: 
        - Don't use standard_name, valid_min/max
        - Add "Standard deviation of" to long_name 

    if number suffix (TEMP1, CNDC2..)
        - Add attributs from the TEMP variable
        - Add '(primary)', '(secondary) to long name
    
    Override governs whether to override any variable attributes that 
    are already present (typically not advised..)
    '''
    

    for varnm in list(D.data_vars) + list(D.coords):
    
        # Check if varnm ends with a number suffix (e.g. PSAL1, TEMP2)
        try:
            number = float(varnm[-1]) # Goes to except if not a number
            core_name = varnm[:-1]
            if number == 1:
                long_name_add = ' (primary sensor)'
            elif number == 2:
                long_name_add = ' (secondary sensor)'
            elif number == 3:
                long_name_add = ' (tertiary sensor)'
            else:
                long_name_add = ''
        except:
            long_name_add = ''
            core_name = varnm
        
        # Remove suffix to get the "core name", e.g. "CNDC" from "CNDC_CTD"
        # (this is what we look for when comparing with the attribute 
        # dictionary)
        core_name = core_name.split('_')[0]


        # Normal variables
        if core_name in _standard_attrs.standard_var_attrs:
            var_attrs_dict = (
                _standard_attrs.standard_var_attrs[core_name].copy())
            for attr, item in var_attrs_dict.items():
                if override:
                    D[varnm].attrs[attr] = item
                else:
                    if attr not in D[varnm].attrs:
                        D[varnm].attrs[attr] = item

                # Append suffix to long name
                if attr=='long_name':
                    D[varnm].attrs['long_name'] += long_name_add

        # For .btl files: Add "Average" to long_name attribute
        # Append "standard deviation of" to long_name
        if ('source_files' in D.attrs and 'long_name' in D[varnm].attrs 
            and D.source_files[-3:].upper == 'BTL'
            and ctd_prof):

            long_name = D[varnm].attrs["long_name"]
            long_name_nocap = long_name[0].lower() + long_name[1:]
            if varnm != 'NISKIN_NUMBER' and 'NISKIN_NUMBER' in D[varnm].dims: 
                D[varnm].attrs['long_name'] = ('Average '
                    f'{long_name_nocap} measured by CTD during bottle closing')


        # Variables with _std suffix    
        if varnm.endswith('_std') and ctd_prof:
            varnm_prefix = varnm.replace('_std', '')
            if varnm_prefix in _standard_attrs.standard_var_attrs:
                var_attrs_dict = _standard_attrs.standard_var_attrs[
                    varnm_prefix].copy()

                # Don't want to use standard_name for these
                for key in ['standard_name', 'valid_min', 'valid_max']:
                    if key in var_attrs_dict:
                        var_attrs_dict.pop(key)
                
                # Add the other attributes
                for attr, item in var_attrs_dict.items():
                    if override:
                        D[varnm].attrs[attr] = item
                    else:
                        if attr not in D[varnm].attrs:
                            D[varnm].attrs[attr] = item

            # Append "standard deviation of" to long_name
            if ('source_files' in D.attrs and 'long_name' in D[varnm].attrs 
                and D.source_files[-3:].upper == 'BTL'):
                long_name = D[varnm].attrs["long_name"]
                long_name_nocap = long_name[0].lower() + long_name[1:]
                D[varnm].attrs['long_name'] = ('Standard deviation of '
                        f'{long_name_nocap}')
                
            # Add ancillary_variable links between std and avg values
            D[varnm].attrs['ancillary_variables'] = varnm_prefix
            D[varnm_prefix].attrs['ancillary_variables'] = varnm


        # Add some variable attributes that should be standard for anything
        # measured on a CTD

        if ctd_prof:
            if varnm not in ['NISKIN_NUMBER', 'TIME', 'TIME_SAMPLE']: 

                if override==False and 'coverage_content_type' in D[varnm].attrs:
                    pass
                else:
                    D[varnm].attrs['coverage_content_type'] = 'physicalMeasurement'
                if override==False and 'sensor_mount' in D[varnm].attrs:
                    pass
                else:
                    D[varnm].attrs['sensor_mount'] = 'mounted_on_shipborne_profiler'
            else:
                D[varnm].attrs['coverage_content_type'] = 'coordinate'

            if varnm in ['SCAN', 'nbin']:
                D[varnm].attrs['coverage_content_type'] 
        
    return D


def add_standard_glob_attrs_org(D, override = False, org = 'npi'):
    '''
    Adds standard organization, specific global variables for a CTD dataset as specified in
    kval.data.nc_format._standard_attrs_org.


    Includes  standard attribute values for things like
    "institution", "creator_name", etc.

    'org' is the organiztion (currently only 'npi' available)


    override: governs whether to override any global attributes that 
    are already present (typically not advised..)
    '''
    
    org_attrs = _standard_attrs_org.standard_globals_org[org.lower()]

    for attr, item in org_attrs.items():
        if attr not in D.attrs:
            D.attrs[attr] = item
        else:
            if override:
                D.attrs[attr] = item

    return D


def add_standard_glob_attrs_ctd(D, override = False, org = False):
    '''
    Adds standard global variables for a CTD dataset as specified in
    oceanograpy.data.nc_format.standard_attrs_global_ctd.

    override: governs whether to override any global attributes that 
    are already present (typically not advised..)
    '''
    
    for attr, item in _standard_attrs.standard_attrs_global_ctd.items():
        if attr not in D.attrs:
            D.attrs[attr] = item
        else:
            if override:
                D.attrs[attr] = item

    return D


def add_standard_glob_attrs_moor(D, override = False, org = None):
    '''
    Adds standard global variables for a CTD dataset as specified in
    oceanograpy.data.nc_format.standard_attrs_global_moored.

    override: governs whether to override any global attributes that 
    are already present (typically not advised..)
    '''
    
    for attr, item in _standard_attrs.standard_attrs_global_moored.items():
        if attr not in D.attrs:
            D.attrs[attr] = item
        else:
            if override:
                D.attrs[attr] = item

    return D


def add_gmdc_keywords_ctd(D, reset = True):
    '''
    Adds standard GMDC variables a la 
    "OCEANS>OCEAN TEMPERATURE>WATER TEMPERATURE"
    reflecting the variables in the dictionary
    _standard_attrs.gmdc_keyword_dict_ctd
    '''

    gmdc_dict = _standard_attrs.gmdc_keyword_dict_ctd
    keywords = []

    for varnm in D.keys():
        for gmdc_kw in gmdc_dict:
            if gmdc_kw in varnm:
                keywords += [gmdc_dict[gmdc_kw]]
    
    unique_keywords = list(set(keywords))

    D.attrs['keywords'] = ','.join(unique_keywords)

    return D



def add_gmdc_keywords_moor(D, reset = True):
    '''
    Adds standard GMDC variables a la 
    "OCEANS>OCEAN TEMPERATURE>WATER TEMPERATURE"
    reflecting the variables in the file
    '''

    gmdc_dict = _standard_attrs.gmdc_keyword_dict_moored
    keywords = []

    for varnm in D.keys():
        for gmdc_kw in gmdc_dict:
            if gmdc_kw in varnm:
                keywords += [gmdc_dict[gmdc_kw]]
    
    unique_keywords = list(set(keywords))

    D.attrs['keywords'] = ','.join(unique_keywords)

    return D



def set_var_attr(D, variable, attr):
    '''
    Set variable attributes for a dataset or data frame.

    Using interactive widgets. 
    '''


    # Check if the attribute already exists in D
    if attr in D[variable].attrs:
        initial_value = D[variable].attrs[attr]
    else:
        initial_value = None

    option_dict = _standard_attrs.global_attrs_options
    if attr in option_dict:
        D = user_input.set_var_attr_pulldown(D, variable, attr, 
                    _standard_attrs.global_attrs_options[attr],
                    initial_value = initial_value)
    else:
        # Make a larger box for (usually) long attributes
        if attr in ['summary', 'comment', 'acknowledgment']:
            rows = 10
        else:
            rows = 1
        D = user_input.set_var_attr_textbox(D, variable, attr, rows,
                    initial_value = initial_value)

    return D




def set_glob_attr(D, attr):
    '''
    Set global attributes for a  dataset or data frame.

    Using interactive widgets. 

    '''

    option_dict = _standard_attrs.global_attrs_options

    # Check if the attribute already exists in D
    if attr in D.attrs:
        initial_value = D.attrs[attr]
    else:
        initial_value = None

    if attr in option_dict:
        D = user_input.set_attr_pulldown(D, attr, 
                    _standard_attrs.global_attrs_options[attr], 
                    initial_value=initial_value)
    else:
        # Make a larger box for (usually) long attributes
        if attr in ['summary', 'comment', 'acknowledgment']:
            rows = 10
        else:
            rows = 1
        D = user_input.set_attr_textbox(D, attr, rows, 
                            initial_value=initial_value)

    return D


def add_missing_glob(D):
    '''
    Prompts the user to fill in information for missing global attrubtes
    '''

    varattrs_dict_ref = _standard_attrs.variable_attrs_necessary.copy()

    for varnm in D:
        if 'PRES' in D[varnm].dims or 'NISKIN_NUMBER' in D[varnm].dims:
            _attrs_dict_ref_var = varattrs_dict_ref.copy()

            if varnm == 'CHLA':
                _attrs_dict_ref_var += [
                    'calibration_formula',
                    'coefficient_A',
                    'coefficient_B',]
                
            if varnm == 'PRES':
                _attrs_dict_ref_var += [
                    'axis',
                    'positive',]
                _attrs_dict_ref_var.remove('processing_level')
                _attrs_dict_ref_var.remove('QC_indicator')

            any_missing = False
            for var_attr in _attrs_dict_ref_var:
                if var_attr not in D[varnm].attrs:
                    set_var_attr(D, varnm, var_attr)
                    any_missing = True

            if not any_missing:
                    print(f'- {varnm}: OK')


    for attr in glob_attrs_dict_ref:
        if attr not in D.attrs:
            set_glob_attr(D, attr)

    return D