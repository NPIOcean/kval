
standard_var_attrs = {
    'STATION': {
        'long_name': 'CTD station name',
        'cf_role': 'profile_id',
    },
    'CRUISE': {
        'long_name': 'Cruise ID',
    },
    'TEMP': {
        'standard_name': 'sea_water_temperature',
        'units': 'degree_Celsius',
        'long_name': 'Sea water temperature',
        'valid_min' : -3.0,
        'valid_max' : 40.0,
        'reference_scale' : 'ITS-90',
    },
    'PSAL': {
        'standard_name': 'sea_water_practical_salinity',
        'units': '1',
        'long_name': 'Practical salinity',
        'valid_min' : 2.0,
        'valid_max' : 41.0,
        'reference_scale':'PSS-78',
    },
    'CNDC': {
        'standard_name': 'sea_water_electrical_conductivity',
        'units': 'S m-1',
        'long_name': 'Seawater electrical conductivity',
        'valid_min' : 0.0,
        'valid_max' : 50.0,
    },
    'ATTN': {
        'standard_name': 'volume_beam_attenuation_coefficient_of_radiative_flux_in_sea_water',
        'units': 'm-1',
        'long_name': 'Beam attenuation coefficient',
        'valid_min' : 0.0,
        'valid_max' : 100.0,
    },
    'DOXY': {
        'standard_name': 'volume_fraction_of_oxygen_in_sea_water',# Check!
        'units': 'ml l-1',
        'long_name': 'Dissolved oxygen',
        'valid_min' : 0.0,
        'valid_max' : 1000.0,
    },
    'DOXY_instr': {
        'standard_name': 'volume_fraction_of_oxygen_in_sea_water',# Check!
        'units': 'ml l-1',
        'long_name': 'Dissolved oxygen from sensor. Not calibrated against water samples.',
        'valid_min' : 0.0,
        'valid_max' : 1000.0,
        'processing_level':'Instrument data that has been converted to physical values',
        'QC_indicator':'unknown',
        'comment':'Not compared with water sample oxygen measurements. Quality unknown!'
    },
    'OXYV': {
        'units': 'volt',
        'long_name': 'Raw signal (voltage) of instrument output by oxygen sensor',
        'valid_min' : 0.0,
        'valid_max' : 1000.0,
    },
    'CHLA': {
        'standard_name': 'chlorophyll_concentration_in_sea_water',
        'units': 'mg m-3',
        'long_name': 'Chlorophyll-A',
        'valid_min' : -1.9,
        'valid_max' : 100.0,
    },
    'CHLA_instr': {
        'standard_name': 'chlorophyll_concentration_in_sea_water',
        'long_name': 'Chlorophyll-A from CTD fluorometer. Not calibrated with water sample chlorophyll measurements.',
        'valid_min' : -1.9,
        'valid_max' : 100.0,
        'processing_level':'Instrument data that has been converted to physical values',
        'QC_indicator':'unknown',
        'comment':('Nominal units are [mg m-3], but water sample calibrations are necessary in order to produce'
                   ' realistic absolute values. No correction for near-surface fluorescence quenching '
                   '(see e.g. https://doi.org/10.4319/lom.2012.10.483) has been applied.')
    },
    'CHLA_fluorescence': {
        'standard_name': 'chlorophyll_concentration_in_sea_water',
        'long_name': 'Chlorophyll-A from CTD fluorometer. Not calibrated with water sample chlorophyll measurements.',
        'valid_min' : -1.9,
        'valid_max' : 100.0,
        'processing_level':'Instrument data that has been converted to physical values',
        'QC_indicator':'unknown',
        'comment':('Nominal units are [mg m-3], but water sample calibrations are necessary in order to produce'
                   ' realistic absolute values. No correction for near-surface fluorescence quenching '
                   '(see e.g. https://doi.org/10.4319/lom.2012.10.483) has been applied.')
    },
    'CDOM_instr': {
        'long_name': ('Colored dissolved organic matter (CDOM) from CTD fluorometer. '
            'Not calibrated with water sample chlorophyll measurements.'),
        'processing_level':'Instrument data that has been converted to physical values',
        'QC_indicator':'unknown',
        'comment':('Nominal units are [mg m-3], but water sample calibrations are necessary in order to produce'
                   ' realistic absolute values.')
    },
    'PRES': {
        'standard_name': 'sea_water_pressure',
        'units': 'dbar',
        'long_name': 'Pressure due to seawater',
        'positive':'down',
        'valid_min' : -1.0,
        'valid_max' : 12000.0,
        'axis': 'Z',
    },
    'DEPTH': {
        'standard_name': 'depth',
        'units': 'm',
        'long_name': 'Water depth',
        'positive':'down',
        'valid_min' : -1.0,
        'valid_max' : 12000.0,
    },
    'SVEL': {
        'standard_name':'speed_of_sound_in_sea_water',
        'units': 'm s-1',
        'long_name': 'Sound velocity of seawater',
        'valid_min' : 1200.0,
        'valid_max' : 1800.0,
                },
   'SIGTH': {
        'units': 'kg m-3',
        'long_name': 'Sigma-theta density',
        'valid_min' : 0.0,
        'valid_max' : 50.0,
    },
    'UCUR': {
        'standard_name':'eastward_sea_water_velocity',
        'units':'m s-1', 
        'long_name': "Eastward sea water velocity",
        'valid_min' : -2.0,
        'valid_max' : 2.0, 
        },
    'VCUR': {
        'standard_name':'northward_sea_water_velocity',
        'units':'m s-1', 
        'long_name': "Northward sea water velocity",
        'valid_min' : -2.0,
        'valid_max' : 2.0, 
        },
    'WCUR': {
        'standard_name':'upward_sea_water_velocity',
        'units':'m s-1', 
        'long_name': "Upward sea water velocity",
        'valid_min' : -2.0,
        'valid_max' : 2.0, 
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



standard_attrs_global_ctd = {
    'Conventions':'ACDD-1.3, CF-1.8',
    'standard_name_vocabulary': 'CF Standard Name Table v74',
    'source': 'CTD profiles from SBE911+',
    'instrument':'In Situ/Laboratory Instruments>Profilers/Sounders>CTD',
    'data_set_language':'eng',
    'keywords_vocabulary':'NASA/GCMD Science Keywords 9.1.5',
    'standard_name_vocabulary':'CF Standard Name Table v83',
    'iso_topic_category':'oceans',
    'platform':'Water-based Platforms>Vessels'
}

standard_attrs_global_moored = {
    'Conventions':'ACDD-1.3, CF-1.8',
    'standard_name_vocabulary': 'CF Standard Name Table v74',
    'source': 'Subsurface mooring',
    'instrument':'In Situ/Laboratory Instruments>Profilers/Sounders>CTD',
    'data_set_language':'eng',
    'keywords_vocabulary':'NASA/GCMD Science Keywords 9.1.5',
    'standard_name_vocabulary':'CF Standard Name Table v83',
    'iso_topic_category':'oceans',
    'platform':'Water-based Platforms>Buoys>Moored>MOORINGS',
    'sensor_mount':'mounted_on_mooring_line',

}


gmdc_keyword_dict_ctd = {
    'PRES':'OCEANS>OCEAN PRESSURE>WATER PRESSURE',
    'TEMP':'OCEANS>OCEAN TEMPERATURE>OCEAN TEMPERATURE PROFILES',
    'CNDC':'OCEANS>SALINITY/DENSITY>CONDUCTIVITY>CONDUCTIVITY PROFILES',
    'PSAL':'OCEANS>SALINITY/DENSITY>OCEAN SALINITY>OCEAN SALINITY PROFILES',
    'CHLA':'OCEANS>OCEAN OPTICS>CHLOROPHYLL>CHLOROPHYLL CONCENTRATION',
    'DOXY':'OCEANS>OCEAN CHEMISTRY>OXYGEN',
    'CDOM':'OCEANS>OCEAN CHEMISTRY>ORGANIC MATTER>COLORED DISSOLVED ORGANIC MATTER',
    'ATTN':'OCEANS>OCEAN OPTICS>ATTENUATION/TRANSMISSION',
    'SVEL':'OCEANS>OCEAN ACOUSTICS>ACOUSTIC VELOCITY',


            }



gmdc_keyword_dict_moored = {
    'PRES':'OCEANS>OCEAN PRESSURE>WATER PRESSURE',
    'TEMP':'OCEANS>OCEAN TEMPERATURE>WATER TEMPERATURE',
    'CNDC':'OCEANS>SALINITY/DENSITY>CONDUCTIVITY',
    'PSAL':'OCEANS>SALINITY/DENSITY>SALINITY',
    'CHLA':'OCEANS>OCEAN OPTICS>CHLOROPHYLL',
    'DOXY':'OCEANS>OCEAN CHEMISTRY>OXYGEN',
    'CDOM':'OCEANS>OCEAN CHEMISTRY>ORGANIC MATTER>COLORED DISSOLVED ORGANIC MATTER',
    'ATTN':'OCEANS>OCEAN OPTICS>ATTENUATION/TRANSMISSION',
    'SVEL':'OCEANS>OCEAN ACOUSTICS>ACOUSTIC VELOCITY',
    'UCUR':'OCEANS>OCEAN CIRCULATION>OCEAN CURRENTS',
    'VCUR':'OCEANS>OCEAN CIRCULATION>OCEAN CURRENTS',
    'WCUR':'OCEANS>OCEAN CIRCULATION>OCEAN CURRENTS',
            }


missing_globals_ctd = {
    'title': '',
    'references': '',
    'comment': '',
    'QC_indicator':'',
    'id': '',
    'project': '',
    'program': '',
    'acknowledgement':'',
    'version':'',
    'cruise_name':'',
}

standard_globals_NPI = {
    'institution': 'Norwegian Polar Institute (NPI)',
    'license': 'CC-BY 4.0',
    'creator_name':'Norwegian Polar Institute (NPI)',
    'creator_email':'post@npolar.no',
    'creator_type':'institution',
    'creator_institution':'Norwegian Polar Institute (NPI)',
    'data_assembly_center':'NO/NPI',
    'naming_authority':'npolar.no',
    'instrument_vocabulary':'NASA/GCMD Instrument Keywords Version 17.2',
    'creator_url':'www.npolar.no',
    'publisher_name':'Norwegian Polar Institute (NPI)',
    'publisher_url':'www.npolar.no',
    'publisher_email':'post@npolar.no',
    'publisher_institution':'Norwegian Polar Institute (NPI)',
}




# Define standard options if we have them
global_attrs_options = {
    # Values and descriptions if we have them
    # From OceanSITES
    'processing_level':[
        "Raw instrument data",
        "Instrument data that has been converted to geophysical values",
        "Post-recovery calibrations have been applied",
        "Data has been scaled using contextual information",
        "Known bad data has been replaced with null values",
        "Known bad data has been replaced with values based on surrounding data",
        "Ranges applied, bad data flagged",
        "Data interpolated",
        "Data manually reviewed",
        "Data verified against model or other contextual information",
        "Other QC process applied"
    ],
    'QC_indicator': [
        ['unknown', 'No QC was performed.'],
        ['good data', 'All QC tests passed.'],
        ['probably good data', '(no description)'],
        ['potentially correctable bad data', 
         'These data are not to be used without scientific'
         ' correction or re-calibration.'],
        ['bad data', 'Data have failed one or more tests.'],
        ['nominal value', 
         'Data were not observed but reported. (e.g. instrument target depth.)'],
        ['interpolated value', 
         'Missing data may be interpolated from neighboring data in space or'
         'time'],
        ['missing value', 'This is a fill value']
    ]
}


# Preferred order for global and variable attrs

global_attrs_ordered = [
    'title',
    'summary',
    'history',
    'id',
    'version',
    'date_created',
    'comment',
    'cruise_name',
    'project',
    'project_number',

    'institution',

    'source',
    'platform',
    'instrument',
    'instrument_serial_number',
    'sensor_mount',
    
    'area',
    'location',
    'cruise',
    'processing_level',
    'QC_indicator',
    'license',



    'time_coverage_start',
    'time_coverage_end',
    'time_coverage_resolution',
    'time_coverage_duration',

    'geospatial_lat_max',
    'geospatial_lon_max',
    'geospatial_lat_min',
    'geospatial_lon_min',
    'geospatial_bounds',
    'geospatial_bounds_crs',

    'geospatial_vertical_min',
    'geospatial_vertical_max',
    'geospatial_vertical_positive',
    'geospatial_vertical_units',
    'geospatial_bounds_vertical_crs',

    'data_set_language',


    'data_assembly_center',

    'creator_name',
    'creator_email',
    'creator_type',
    'creator_institution',
    'creator_url',

    'publisher_name',
    'publisher_url',
    'publisher_email',
    'publisher_institution',

    'keywords',

    'binned',
    'latitude',
    'longitude',
    'SBE_processing',
    'source_files',

    'featureType',
    'standard_name_vocabulary',
    'instrument_vocabulary',
    'keywords_vocabulary',
    'iso_topic_category',
    'Conventions',
    'data_set_language',

    'naming_authority',

    'acknowledgment',
    ]

variable_attrs_ordered = [
    'units',
    'standard_name',
    'long_name',
    'comment',
    'calibration_formula',
    'coefficient_A',
    'coefficient_B',
    'sensor_serial_number',
    'sensor_calibration_date',
    'valid_min',
    'valid_max',
    'sensor_mount',
    'reference_scale',
    'coverage_content_type',
    'axis'
]

variable_attrs_necessary = [
    'units',
    'standard_name',
    'long_name',
    'sensor_serial_number',
    'sensor_calibration_date',
    'valid_min',
    'valid_max',
    'coverage_content_type',
    'processing_level',
    'QC_indicator',
]




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
