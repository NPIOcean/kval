'''
Functions for making netcdfs cf-compliant.
(Working with xarray datasets)
'''
from oceanograpy.util import time
import cftime
from oceanograpy.data.nc_format import _standard_attrs

def add_range_attrs_ctd(D):
    '''
    Add some global attributes based on the data
    '''
    # Lateral
    D.attrs['geospatial_lat_max'] = D.LATITUDE.max().values
    D.attrs['geospatial_lon_max'] = D.LONGITUDE.max().values
    D.attrs['geospatial_lat_min'] = D.LATITUDE.min().values
    D.attrs['geospatial_lon_min'] = D.LONGITUDE.min().values
    D.attrs['geospatial_bounds'] = _get_geospatial_bounds_wkt_str(D)
    D.attrs['geospatial_bounds_crs'] = 'EPSG:4326'
    
    # Vertical
    D.attrs['geospatial_vertical_min'] = D.PRES.min().values
    D.attrs['geospatial_vertical_max'] = D.PRES.max().values
    D.attrs['geospatial_vertical_positive'] = 'down'
    D.attrs['geospatial_vertical_units'] = 'dbar'
    D.attrs['geospatial_bounds_vertical_crs'] = 'EPSG:5831'

    # Time
    start_time = cftime.num2date(D.TIME[0], D.TIME.units)
    end_time = cftime.num2date(D.TIME[-1], D.TIME.units)
    D.attrs['time_coverage_start'] = time.datetime_to_ISO8601(start_time)
    D.attrs['time_coverage_end'] = time.datetime_to_ISO8601(end_time)
    D.attrs['time_coverage_resolution'] = 'variable'
    D.attrs['time_coverage_duration'] = _get_time_coverage_duration_str(D)

    return D

def _get_geospatial_bounds_wkt_str(D):
    '''
    Get the geospatial_bounds_crs value on the required format:

    'POLYGON((x1 y1, x2 y2, x3 y3, …, x1 y1))'

    Note: D must already have geospatial_lat_max (etc) attributes.
    
    
    '''
    lat_max = D.geospatial_lat_max
    lat_min = D.geospatial_lat_min
    lon_max = D.geospatial_lon_max
    lon_min = D.geospatial_lon_min

    corner_dict = (lon_min, lat_min, lon_min, lat_max, lon_max, lat_min, 
                  lon_max, lat_max, lon_min, lat_min)

    wkt_str = 'POLYGON ((%f %f, %f %f, %f %f, %f %f, %f %f))'%corner_dict

    return wkt_str

def _get_time_coverage_duration_str(D):
    '''
    Get the time duration based on first and last time stamp on 
    the required ISO8601format (P[YYYY]-[MM]-[DD]T[hh]:[mm]:[ss])
    '''
    start_dt, end_dt = cftime.num2date([D.TIME[0], D.TIME[-1]], D.TIME.units)
    duration_str = time.start_end_times_cftime_to_duration(start_dt, end_dt)
    return duration_str

def add_standard_var_attrs_ctd(D, override = False):
    '''
    Add variable attributes 
    as specified in oceanograpy.data.nc_format.standard_var_attrs_ctd

    Override governs whether to override any variable attributes that 
    are already present (typically not advised..)
    '''
    for varnm in list(D.data_vars) + list(D.coords):
        if varnm in _standard_attrs.standard_var_attrs_ctd:
            var_attrs_dict = _standard_attrs.standard_var_attrs_ctd[varnm]
            for attr, item in var_attrs_dict.items():
                if override:
                    D[varnm].attrs[attr] = item
                else:
                    if attr not in D[varnm].attrs:
                        D[varnm].attrs[attr] = item
    return D


def add_standard_glob_attrs_ctd(D, NPI = False, override = False):
    '''
    Adds standard global variables for a CTD dataset as specified in
    oceanograpy.data.nc_format.standard_attrs_global_ctd.

    If NPI = True, also add NPI standard attribute values for things like
    "institution", "creator_name", etc, as specified in 
    oceanograpy.data.nc_format.standard_globals_NPI_ctd

    override: governs whether to override any global attributes that 
    are already present (typically not advised..)
    '''
    
    for attr, item in _standard_attrs.standard_attrs_global_ctd.items():
        if attr not in D.attrs:
            D.attrs[attr] = item
        else:
            if override:
                D.attrs[attr] = item

    if NPI:
        for attr, item in _standard_attrs.standard_globals_NPI_ctd.items():
            if attr not in D.attrs:
                D.attrs[attr] = item
            else:
                if override:
                    D.attrs[attr] = item

    return D

def add_gmdc_keywords_ctd(D):
    '''
    Adds standard GMDC variables a la 
    "OCEANS>OCEAN TEMPERATURE>WATER TEMPERATURE"
    reflecting the variables in the file
    '''
    gmdc_dict = _standard_attrs.gmdc_keyword_dict_ctd
    keywords = ''

    for varnm in D.keys():
        if varnm in gmdc_dict:
            keywords += gmdc_dict[varnm]

    D.attrs['keywords'] = keywords

    return D


def add_global_attribute(D, attr):
    '''
    Prompts the user to enter the value of an attribute
    '''
