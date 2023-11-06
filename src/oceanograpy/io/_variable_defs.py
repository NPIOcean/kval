'''
VARIABLE_DEFS.PY

!!INCOMPLETE!!

Contains information used to standardize variable namings:

- Mapping from SBE names (e.g. "prDM" -> "PRES", 't090C' -> 'TEMP1')
- CF-compliant attributes to append to the variables 
  ('standard_name', 'units', etc) 

'''

# NOte: Full list here:
#https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjJwIKg3q-CAxXSSvEDHeb0ATEQFnoECBcQAQ&url=https%3A%2F%2Fsensor.awi.de%2Frest%2Fsensors%2FonlineResources%2FgetOnlineResourcesFile%2F539%2FSeasave_7.26.4.pdf&usg=AOvVaw1-raALgk8d-3V_ciFeHcr7&opi=89978449
#(Seasave_7.26.4-1.pdf=

SBE_name_map = {
    'PRDM': {'name': 'PRES', 'units': 'dbar'},
    'T090C': {'name': 'TEMP1', 'units': 'degree_Celsius'},
    'T190C': {'name': 'TEMP2', 'units': 'degree_Celsius'},
    'C0S/M': {'name': 'CNDC1', 'units': 'S m-1'},
    'C1S/M': {'name': 'CNDC2', 'units': 'S m-1'},
    'SBOX0MM/KG': {'name': 'DOXY1', 'units': 'micromol kg-1'},
    'SBOX1MM/KG': {'name': 'DOXY2', 'units': 'micromol kg-1'},
    'SBOX0ML/L': {'name': 'DOXY1', 'units': 'mL l-1'},
    'SBOX1ML/L': {'name': 'DOXY2', 'units': 'mL l-1'},
    'SBEOX0MG/L': {'name': 'DOXY1', 'units': 'mg l-1'},
    'SBOX1ML/L': {'name': 'DOXY2', 'units': 'mL l-1'},
    'SBOX0V': {'name': 'OXYV1', 'units': 'volt'},
    'SBOX1V': {'name': 'OXYV2', 'units': 'volt'},
    'WETCDOM': {'name': 'CDOM1', 'units': 'mg m-3'},
    'WETCDOM1': {'name': 'CDOM2', 'units': 'mg m-3'},
    'CSTARAT0': {'name': 'ATTN1', 'units': 'm-1'},
    'WETSTAR': {'name': 'CHLA1', 'units': 'mg m-3'},
    'FLECO-AFL': {'name': 'CHLA1', 'units': 'mg m-3'},
    'FLECO-AFL1': {'name': 'CHLA2', 'units': 'mg m-3'},
    'AVGSVCM': {'name': 'SVEL_AVG', 'units': 'm s-1'},
    'SVCM': {'name': 'SVEL', 'units': 'm s-1'},
    'SAL00': {'name': 'PSAL1', 'units': '1'},
    'SAL11': {'name': 'PSAL2', 'units': '1'},
    'LATITUDE': {'name': 'LATITUDE', 'units': 'degree_north'},
    'LONGITUDE': {'name': 'LONGITUDE', 'units': 'degree_east'},
    'SCAN': {'name': 'SCAN', 'units': 'counts'},
    'TIMES': {'name': 'TIME_ELAPSED', 'units': 's'},
    'FLAG': {'name': 'SBE_FLAG', 'units': ''},

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
