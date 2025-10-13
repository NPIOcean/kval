'''
VARIABLE_DEFS.PY

!!INCOMPLETE!!

Contains information used to standardize variable namings:

SBE_name_map:
  Mapping from SBE names (e.g. "prDM" -> "PRES", 't090C' -> 'TEMP1')


var_attrs:
  List of CF-compliant attributes to append to the variables
  ('standard_name', 'units', etc). NOTE: This is done more
  properly in the data.ctd module. Should maybe remove from here?

sensor_info_dict_SBE:
  Mapping from SBE header instrument info to physical sensor
  (e.g. 'Temperature, 2 -->':'temp_sensor_2')
'''

# NOte: Full list here:
#https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjJwIKg3q-CAxXSSvEDHeb0ATEQFnoECBcQAQ&url=https%3A%2F%2Fsensor.awi.de%2Frest%2Fsensors%2FonlineResources%2FgetOnlineResourcesFile%2F539%2FSeasave_7.26.4.pdf&usg=AOvVaw1-raALgk8d-3V_ciFeHcr7&opi=89978449
#(Seasave_7.26.4-1.pdf=

SBE_name_map = { # Note: replace / with _  !
    'PRDM': {'name': 'PRES', 'units': 'dbar',
              'sensors':['pres_sensor'],},
    'T090C': {'name': 'TEMP1', 'units': 'degree_Celsius',
              'sensors':['temp_sensor_1']},
    'T190C': {'name': 'TEMP2', 'units': 'degree_Celsius',
              'sensors':['temp_sensor_2']},
    'TV290C': {'name': 'TEMP1', 'units': 'degree_Celsius',
              'sensors':['temp_sensor_1']},
    'T068C': {'name': 'TEMP1', 'units': 'degree_Celsius',
              'sensors':['temp_sensor_1'],
              'reference_scale':'IPTS-68'},
    'T168C': {'name': 'TEMP2', 'units': 'degree_Celsius',
              'sensors':['temp_sensor_2'],
              'reference_scale':'IPTS-68'},
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
    'COND0MS_CM': {'name': 'CNDC1', 'units': 'mS cm-1',
              'sensors':['cndc_sensor_1']},
    'COND1MS_CM': {'name': 'CNDC2', 'units': 'mS cm-1',
              'sensors':['cndc_sensor_2']},
    'COND0S_M': {'name': 'CNDC1', 'units': 'S m-1',
              'sensors':['cndc_sensor_1']},
    'COND1S_M': {'name': 'CNDC2', 'units': 'S m-1',
              'sensors':['cndc_sensor_2']},
    'SBEOX0MM_KG': {'name': 'DOXY1_instr', 'units': 'micromole kg-1',
              'sensors':['oxy_sensor_1'],
              'standard_name':
              'moles_of_oxygen_per_unit_mass_in_sea_water'},
    'SBEOX1MM_KG': {'name': 'DOXY2_instr', 'units': 'micromole kg-1',
              'sensors':['oxy_sensor_2'],
              'standard_name':
              'moles_of_oxygen_per_unit_mass_in_sea_water'},
    'SBOX0MM_KG': {'name': 'DOXY1_instr', 'units': 'micromole kg-1',
              'sensors':['oxy_sensor_1'],
              'standard_name':
              'moles_of_oxygen_per_unit_mass_in_sea_water'},
    'SBOX1MM_KG': {'name': 'DOXY2_instr', 'units': 'micromole kg-1',
              'sensors':['oxy_sensor_2'],
              'standard_name':
              'moles_of_oxygen_per_unit_mass_in_sea_water'},
    'SBEOX0ML_L': {'name': 'DOXY1_instr', 'units': 'mL l-1',
              'sensors':['oxy_sensor_1'],
              'standard_name':
              'volume_fraction_of_oxygen_in_sea_water'},
    'SBEOX1ML_L': {'name': 'DOXY2_instr', 'units': 'mL l-1',
              'sensors':['oxy_sensor_2'],
              'standard_name':
              'volume_fraction_of_oxygen_in_sea_water'},
    'SBEOX0MG/L': {'name': 'DOXY1_instr', 'units': 'mg l-1',
              'sensors':['oxy_sensor_1'],
              'standard_name':
              'mass_concentration_of_oxygen_in_sea_water'},
    'SBOX1MG/L': {'name': 'DOXY2_instr', 'units': 'mg l-1',
              'sensors':['oxy_sensor_2'],
              'standard_name':
              'mass_concentration_of_oxygen_in_sea_water'},
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
    'XMISS': {'name': 'TRANS1', 'units': '%',
              'sensors':['attn_sensor_1']},
    'BAT': {'name': 'ATTN1', 'units': 'm-1',
              'sensors':['attn_sensor_1']},
    'WETSTAR': {'name': 'CHLA1_fluorescence', 'units': 'mg m-3',
              'sensors':['chla_sensor_1']},
    'FLECO-AFL': {'name': 'CHLA1_fluorescence', 'units': 'mg m-3',
              'sensors':['chla_sensor_1']},
    'FLECO-AFL1': {'name': 'CHLA2_fluorescence', 'units': 'mg m-3',
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
    'SIGMA-Ã©00': {'name': 'SIGTH1', 'units': 'kg m-3',
                  'sensors':['temp_sensor_1', 'cndc_sensor_1']},
    'SIGMA-Ã©11': {'name': 'SIGTH1', 'units': 'kg m-3',
                  'sensors':['temp_sensor_2', 'cndc_sensor_2']},


    'SVCM': {'name': 'SVEL', 'units': 'm s-1'},
    'SAL00': {'name': 'PSAL1', 'units': '1',
              'sensors':['temp_sensor_1', 'cndc_sensor_1']},
    'SAL11': {'name': 'PSAL2', 'units': '1',
              'sensors':['temp_sensor_2', 'cndc_sensor_2']},
    'PAR': {'name': 'PAR', 'units': '?',
              'sensors':['par_sensor_1',]},
    'SPAR': {'name': 'SPAR', 'units': '?',
              'sensors':['spar_sensor_1',]},
    'ALTM': {'name': 'ALTI', 'units': 'm',
              'sensors':['altimeter_sensor_1']},

    'DZ_DTM' : {'name': 'DESCENT_RATE', 'units': 'm s-1',
              'sensors':['pres_sensor']},
    'LATITUDE': {'name': 'LATITUDE_SAMPLE', 'units': 'degree_north'},
    'LONGITUDE': {'name': 'LONGITUDE_SAMPLE', 'units': 'degree_east'},
    'SCAN': {'name': 'SCAN', 'units': 'counts'},
    'TIMES': {'name': 'TIME_ELAPSED', 'units': 's'},
    'TIMEJ': {'name': 'TIME_JULD', 'units': 'days'},
    'TIMEJV2': {'name': 'TIME_JULD', 'units': 'days'},
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
    },

}


### INSTRUMENT CONFIGURATION KEYS ###

sensor_info_dict_SBE = {
    'Temperature -->':'temp_sensor_1',
    'Temperature, 2 -->':'temp_sensor_2',
    'Water Temperature':'temp_sensor_1',

    'Conductivity -->':'cndc_sensor_1',
    'Conductivity, 2 -->':'cndc_sensor_2',

    'Pressure, Digiquartz with TC -->':'pres_sensor',
    'Count, Pressure, Strain Gauge -->':'pres_sensor',
    'Oxygen, SBE 43 -->':'oxy_sensor_1',
    'Oxygen, SBE 43, 2 -->':'oxy_sensor_2',
    'sbeox0V: Oxygen raw, SBE 43 -->':'oxy_sensor_1',
    'sbeox1V: Oxygen raw, SBE 43, 2 -->':'oxy_sensor_1',
    'Fluorometer, WET Labs WETstar -->':'chla_sensor_1',
    'Fluorometer, WET Labs WETstar, 2 -->':'chla_sensor_2',
    'Fluorometer, WET Labs ECO-AFL/FL -->':'chla_sensor_1',
    'Fluorometer, WET Labs ECO-AFL/FL, 2 -->':'chla_sensor_2',
    'Transmissometer, WET Labs C-Star -->':'attn_sensor_1',
    'Transmissometer, Chelsea/Seatech -->':'attn_sensor_1',
    'Fluorometer, WET Labs ECO CDOM -->':'cdom_sensor_1',
    'Fluorometer, WET Labs ECO CDOM, 2 -->':'cdom_sensor_2',
    'Altimeter -->':'altimeter_sensor_1',
    'PAR/Irradiance, Biospherical/Licor -->':'par_sensor_1',
    'SPAR voltage, SPAR, Biospherical/Licor -->':'spar_sensor_1',
}



#### RBR

# Dictionary mapping RBR variable names to NPI/OceanSITES
# ones (some of these are undefined or unknown to me)
RBR_name_map = {
    'conductivity':'CNDC',
    'temperature':'TEMP',
    'pressure':'TOTPRES',
    'sea_pressure':'PRES',
    'depth':'DEPTH',
    'salinity':'PSAL',
    'speed_of_sound':'SVEL',
    'specific_conductivity':'SPCNDC',
    'chlorophyll':'CHLA',
    'par':'PAR',
    'timestamp':'TIME'
}

# Dictionary mapping RBR units to NPI/UDUNITS ones

RBR_units_map = {
    'mS/cm':'mS cm-1',
    '°C':'degC',
    'PSU':'1',
    'm/s': 'm s-1',
    'µS/cm':'uS cm-1',
    'µg/l':'ug L-1',
    'µMol/m²/s':'umol m-2 s-1'}

