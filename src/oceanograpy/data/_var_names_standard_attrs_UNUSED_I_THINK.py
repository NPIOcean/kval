
var_attrs_dict = {
    'STATION': {
        'long_name': 'CTD station name',
        'cf_role': 'profile_id',
    },
    'CRUISE': {
        'long_name': 'Cruise ID',
        'cf_role': 'trajectory_id',
    },
    'TEMP': {
        'standard_name': 'sea_water_temperature',
        'units': 'degree_Celsius',
        'long_name': 'Sea water temperature',
        'valid_min' : -3,
        'valid_max' : 40,
        'reference_scale' : 'ITS-90',
    },
    'PSAL': {
        'standard_name': 'sea_water_practical_salinity',
        'units': '1',
        'long_name': 'Practical salinity',
        'valid_min' : 2,
        'valid_max' : 41,
        'reference_scale':'“PSS-78',
    },
    'CNDC': {
        'standard_name': 'sea_water_electrical_conductivity',
        'units': 'S m-1',
        'long_name': 'electrical conductivity',
        'valid_min' : 0,
        'valid_max' : 50,
    },
    'DOXY': {
        'standard_name': 'volume_fraction_of_oxygen_in_sea_water',# Check!
        'units': 'ml l-1',
        'long_name': 'Dissolved oxygen',
        'valid_min' : 0,
        'valid_max' : 1000,
    },
    'OXYV': {
        'units': 'volt',
        'long_name': 'Raw signal (voltage) of instrument output by oxygen sensor',
        'valid_min' : 0,
        'valid_max' : 1000,
    },
    'CHLA': {

        'standard_name': 'mass_concentration_chlorophyll_concentration_in_sea_water',
        'units': 'mg m-3',
        'long_name': 'Chlorophyll-A',
        'valid_min' : 0,
        'valid_max' : 100,
    },
    'PRES': {
        'standard_name': 'sea_water_pressure',
        'units': 'dbar',
        'long_name': 'pressure',
        'positive':'down',
        'valid_min' : -1,
        'valid_max' : 12000,
        'axis': 'Z',
    },
    'TIME': {
        'standard_name': 'time',
        'units': 'days since 1970-01-01',
        'long_name': 'Time stamp of profile',
        'axis':'T'
    },
    'LONGITUDE': {
        'standard_name': 'longitude',
        'units': 'degree_east',
        'long_name': 'longitude',
        'axis':'X',
    },
    'LATITUDE': {
        'standard_name': 'latitude',
        'units': 'degree_north',
        'long_name': 'latitude',
        'axis':'Y',
    }
}

def var_attrs():
    return var_attrs_dict


### GLOBAL ATTRIBUTE IMPORTANCE ###
# necessity codes:
# REQ: Required, HREC: Highly recommended, REC: Recommended, SUG: Suggested 

global_attrs_desc_importance = {
    'title':{
        'necessity':'REQ',
        'description':'''A short phrase or sentence describing the dataset. In
                      many discovery systems, the title will be displayed in the
                      results list from a search, and therefore should be human
                      readable and reasonable to display in a list of such
                      names. This attribute is also recommended by the NetCDF
                      Users Guide and the CF conventions. E.g.: "CF-1.10, ACDD-1.3"'''
    },
    'summary': {
        'necessity': 'REQ',
        'description': '''A paragraph describing the dataset, analogous to an
                       abstract for a paper.'''
    },
    'keywords': {
        'necessity': 'REQ',
        'description': '''A comma-separated list of key words and/or phrases.
                       Keywords may be common words or phrases, terms from a
                       controlled vocabulary (GCMD is often used), or URIs for
                       terms from a controlled vocabulary (see also
                       "keywords_vocabulary" attribute).'''
    },
    'id': {
        'necessity': 'REQ',
        'description': '''A unique identifier for the dataset. It is recommended
                      to provide a stable and unique identifier for better data
                      management and tracking.'''
    },
    'Conventions': {
        'necessity': 'REQ',
        'description': ''' 	A comma-separated list of the conventions that are
                       followed by the dataset. For files that follow this
                       version of ACDD, include the string 'ACDD-1.3'. (This
                       attribute is described in the NetCDF Users Guide.)'''

    }
}
