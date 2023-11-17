'''
VARIABLE_DEFS.PY

!!INCOMPLETE!!

Contains information used to standardize variable namings:

- Mapping from SBE names (e.g. "prDM" -> "PRES", 't090C' -> 'TEMP1')
- CF-compliant attributes to append to the variables 
  ('standard_name', 'units', etc) 
- Mapping from SBE header instrument info to physical sensor
  (e.g. 'Temperature, 2 -->':'temp_sensor_2') 
'''

# NOte: Full list here:
#https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjJwIKg3q-CAxXSSvEDHeb0ATEQFnoECBcQAQ&url=https%3A%2F%2Fsensor.awi.de%2Frest%2Fsensors%2FonlineResources%2FgetOnlineResourcesFile%2F539%2FSeasave_7.26.4.pdf&usg=AOvVaw1-raALgk8d-3V_ciFeHcr7&opi=89978449
#(Seasave_7.26.4-1.pdf=

SBE_name_map = { # Note: replace / with _  !
    'PRDM': {'name': 'PRES', 'units': 'dbar', 
              'sensors':['pres_sensor']},
    'T090C': {'name': 'TEMP1', 'units': 'degree_Celsius', 
              'sensors':['temp_sensor_1']},
    'T190C': {'name': 'TEMP2', 'units': 'degree_Celsius', 
              'sensors':['temp_sensor_2']},
    'T4990C': {'name': 'TEMP1', 'units': 'degree_Celsius', 
              'sensors':['temp_sensor_1']},
    'POTEMP090C': {'name': 'PTEMP1', 'units': 'degree_Celsius', 
              'sensors':['temp_sensor_1']},
    'POTEMP190C': {'name': 'PTEMP2', 'units': 'degree_Celsius', 
              'sensors':['temp_sensor_2']},
    'C0S_M': {'name': 'CNDC1', 'units': 'S m-1',
              'sensors':['cndc_sensor_1']},
    'C1S_M': {'name': 'CNDC2', 'units': 'S m-1',
              'sensors':['cndc_sensor_2']},
    'C0MS_CM': {'name': 'CNDC1', 'units': 'mS cm-1',
              'sensors':['cndc_sensor_1']},
    'C1MS_CM': {'name': 'CNDC2', 'units': 'mS cm-1',
              'sensors':['cndc_sensor_2']},
    'SBEOX0MM_KG': {'name': 'DOXY1_instr', 'units': 'micromol kg-1',
              'sensors':['oxy_sensor_1']},
    'SBEOX1MM_KG': {'name': 'DOXY2_instr', 'units': 'micromol kg-1',
              'sensors':['oxy_sensor_2']},
    'SBEOX0ML_L': {'name': 'DOXY1_instr', 'units': 'mL l-1',
              'sensors':['oxy_sensor_1']},
    'SBEOX1ML_L': {'name': 'DOXY2_instr', 'units': 'mL l-1',
              'sensors':['oxy_sensor_2']},
    'SBEOX0MG/L': {'name': 'DOXY1_instr', 'units': 'mg l-1',
              'sensors':['oxy_sensor_1']},
    'SBOX1MG/L': {'name': 'DOXY2_instr', 'units': 'mg l-1',
              'sensors':['oxy_sensor_2']},
    'SBEOX0V': {'name': 'OXYV1', 'units': 'volt',
              'sensors':['oxy_sensor_1']},
    'SBEOX1V': {'name': 'OXYV2', 'units': 'volt',
              'sensors':['oxy_sensor_2']},
    'WETCDOM': {'name': 'CDOM1_instr', 'units': 'mg m-3',
              'sensors':['cdom_sensor_1']},
    'WETCDOM1': {'name': 'CDOM2_instr', 'units': 'mg m-3',
              'sensors':['cdom_sensor_2']},
    'CSTARAT0': {'name': 'ATTN1', 'units': 'm-1',
              'sensors':['attn_sensor_1']},
    'WETSTAR': {'name': 'CHLA1_instr', 'units': 'mg m-3',
              'sensors':['chla_sensor_1']},
    'FLECO-AFL': {'name': 'CHLA1_instr', 'units': 'mg m-3',
              'sensors':['chla_sensor_1']},
    'FLECO-AFL1': {'name': 'CHLA2_instr', 'units': 'mg m-3',
              'sensors':['chla_sensor_2']},
    'AVGSVCM': {'name': 'SVEL_AVG', 'units': 'm s-1'},
    'DEPSM': {'name': 'DEPTH', 'units': 'm'},
    'SIGMA-É00': {'name': 'SIGTH1', 'units': 'kg m-3',
                  'sensors':['temp_sensor_1', 'cndc_sensor_1']},
    'SIGMA-É11': {'name': 'SIGTH2', 'units': 'kg m-3',
                  'sensors':['temp_sensor_2', 'cndc_sensor_2']},
    'SIGMA-Ï¿½00': {'name': 'SIGTH1', 'units': 'kg m-3',
                  'sensors':['temp_sensor_1', 'cndc_sensor_1']},
    'SIGMA-Ï¿½11': {'name': 'SIGTH2', 'units': 'kg m-3',
                  'sensors':['temp_sensor_2', 'cndc_sensor_2']},

    'SVCM': {'name': 'SVEL', 'units': 'm s-1'},
    'SAL00': {'name': 'PSAL1', 'units': '1', 
              'sensors':['temp_sensor_1', 'cndc_sensor_1']},
    'SAL11': {'name': 'PSAL2', 'units': '1',
              'sensors':['temp_sensor_2', 'cndc_sensor_2']},

    'ALTM': {'name': 'ALTI', 'units': 'm',
              'sensors':['altimeter_sensor_1']},
    'LATITUDE': {'name': 'LATITUDE_SAMPLE', 'units': 'degree_north'},
    'LONGITUDE': {'name': 'LONGITUDE_SAMPLE', 'units': 'degree_east'},
    'SCAN': {'name': 'SCAN', 'units': 'counts'},
    'TIMES': {'name': 'TIME_ELAPSED', 'units': 's'},
    'TIMEJ': {'name': 'TIME_JULD', 'units': 'days'},

    'FLAG': {'name': 'SBE_FLAG', 'units': ''},

}


var_attrs = {
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
    'DEPTH': {
    'standard_name': 'depth',
    'units': 'm',
    'long_name': 'Depth in salt water',
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


### INSTRUMENT CONFIGURATION KEYS ###

sensor_info_dict = {
    'Temperature -->':'temp_sensor_1',
    'Temperature, 2 -->':'temp_sensor_2',
    'Conductivity -->':'cndc_sensor_1',
    'Conductivity, 2 -->':'cndc_sensor_2',
    'Pressure, Digiquartz with TC -->':'pres_sensor',
    'Count, Pressure, Strain Gauge -->':'pres_sensor',
    'Oxygen, SBE 43 -->':'oxy_sensor_1',
    'Oxygen, SBE 43, 2 -->':'oxy_sensor_2',
    'sbeox0V: Oxygen raw, SBE 43 -->':'oxy_sensor_1',
    'sbeox1V: Oxygen raw, SBE 43, 2 -->':'oxy_sensor_1',
    'Fluorometer, WET Labs ECO-AFL/FL -->':'chla_sensor_1',
    'Transmissometer, WET Labs C-Star -->':'attn_sensor_1',
    'Fluorometer, WET Labs ECO CDOM -->':'cdom_sensor_1',
    'Altimeter -->':'altimeter_sensor_1',
}
