'''
VARIABLE_DEFS.PY

Contains information used to standardize variable namings:

- Mapping from SBE names (e.g. "prDM" -> "PRES", 't090C' -> 'TEMP1')
- CF-compliant attributes to append to the variables 
  ('standard_name', 'units', etc) 

'''

# Standard 
var_map = {
    'prDM':{'PRES', '', 'dbar'
    't090C':{'TEMP',
    't190C':{'TEMP2',
    ''

}





var_attrs_cf = {
    'PROFILE': {
        'long_name': 'Profile number in dataset (see PROFILE_ID for station/cast)',
    },
    
    'PROFILE_ID': {
        'long_name': 'profile ID ("station_cast")',
        'cf_role': 'profile_id',
    },
    'CRUISE': {
        'long_name': 'Cruise ID',
        'cf_role': 'trajectory_id',
    },
    'TEMP': {
        'standard_name': 'sea_water_temperature',
        'units': 'degree_Celsius',
        'long_name': 'sea water temperature',
    },
    'PSAL': {
        'standard_name': 'sea_water_practical_salinity',
        'units': '',
        'long_name': 'practical salinity',
    },
    'CNDC': {
        'standard_name': 'sea_water_electrical_conductivity',
        'units': 'S m-1',
        'long_name': 'electrical conductivity',
    },
    'PRES': {
        'standard_name': 'sea_water_pressure',
        'units': 'dbar',
        'long_name': 'pressure',
        'positive':'down',
    },
    'TIME': {
        'standard_name': 'time',
        'units': 'days since 1970-01-01',
        'long_name': 'time',
    },
    'LONGITUDE': {
        'standard_name': 'longitude',
        'units': 'degree_east',
        'long_name': 'longitude',
    },
    'LATITUDE': {
        'standard_name': 'latitude',
        'units': 'degree_north',
        'long_name': 'latitude',
    }
}
