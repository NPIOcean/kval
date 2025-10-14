# Defines some standard metadata attributes for various

standard_var_attrs = {
    "STATION": {
        "long_name": "CTD station name",
        "cf_role": "profile_id",
    },
    "CRUISE": {
        "long_name": "Cruise ID",
    },
    "TEMP": {
        "standard_name": "sea_water_temperature",
        "units": "degree_Celsius",
        "long_name": "Sea water temperature",
        "valid_min": -3.0,
        "valid_max": 40.0,
        "reference_scale": "ITS-90",
    },
    "PSAL": {
        "standard_name": "sea_water_practical_salinity",
        "units": "1",
        "long_name": "Practical salinity",
        "valid_min": 2.0,
        "valid_max": 41.0,
        "reference_scale": "PSS-78",
    },
    "CNDC": {
        "standard_name": "sea_water_electrical_conductivity",
        "units": "S m-1",
        "long_name": "Seawater electrical conductivity",
        "valid_min": 0.0,
        "valid_max": 50.0,
    },
    "ATTN": {
        "standard_name": ("volume_beam_attenuation_coefficient_"
                          "of_radiative_flux_in_sea_water"),
        "units": "m-1",
        "long_name": "Beam attenuation coefficient",
        "valid_min": 0.0,
        "valid_max": 100.0,
    },
    "DOXY": {
        "standard_name": "volume_fraction_of_oxygen_in_sea_water",  # Check!
        "units": "ml l-1",
        "long_name": "Dissolved oxygen",
        "valid_min": 0.0,
        "valid_max": 1000.0,
    },
    "DOXY_instr": {
        "standard_name": "volume_fraction_of_oxygen_in_sea_water",  # Check!
        "units": "ml l-1",
        "long_name": ("Dissolved oxygen from sensor."
                      " Not calibrated against water samples."),
        "valid_min": 0.0,
        "valid_max": 1000.0,
        "processing_level": ("Instrument data that has been"
                             " converted to physical values"),
        "QC_indicator": "unknown",
        "comment": ("Not compared with water sample oxygen measurements. "
                        "Quality unknown!"),
    },
    "OXYV": {
        "units": "volt",
        "long_name": (
            "Raw signal (voltage) of instrument output by oxygen sensor"),
        "valid_min": 0.0,
        "valid_max": 1000.0,
    },
    "CHLA": {
        "standard_name": "chlorophyll_concentration_in_sea_water",
        "units": "mg m-3",
        "long_name": "Chlorophyll-A",
        "valid_min": -1.9,
        "valid_max": 100.0,
    },
    "CHLA_instr": {
        "standard_name": "chlorophyll_concentration_in_sea_water",
        "long_name": ("Chlorophyll-A from CTD fluorometer. Not calibrated "
                      "with water sample chlorophyll measurements."),
        "valid_min": -1.9,
        "valid_max": 100.0,
        "processing_level": (
            "Instrument data that has been converted to physical values"),
        "QC_indicator": "unknown",
        "comment": (
            "Nominal units are [mg m-3], but water sample calibrations are "
            "necessary in order to produce realistic absolute values. No "
            "correction for near-surface fluorescence quenching (see e.g. "
            "https://doi.org/10.4319/lom.2012.10.483) has been applied."
        ),
    },
    "CHLA_fluorescence": {
        "standard_name": "chlorophyll_concentration_in_sea_water",
        "long_name": ("Chlorophyll-A from CTD fluorometer. Not calibrated "
                      "with water sample chlorophyll measurements."),
        "valid_min": -1.9,
        "valid_max": 100.0,
        "processing_level": ("Instrument data that has been converted to "
                             "physical values"),
        "QC_indicator": "unknown",
        "comment": (
            "Nominal units are [mg m-3], but water sample calibrations are"
            " necessary in order to produce realistic absolute values. No "
            "correction for near-surface fluorescence quenching (see e.g. "
            "https://doi.org/10.4319/lom.2012.10.483) has been applied."
        ),
    },
    "CDOM_instr": {
        "long_name": (
            "Colored dissolved organic matter (CDOM) from CTD fluorometer. "
            "Not calibrated with water sample chlorophyll measurements."
        ),
        "processing_level": (
            "Instrument data that has been converted to physical values"),
        "QC_indicator": "unknown",
        "comment": (
            "Nominal units are [mg m-3], but water sample calibrations are "
            "necessary in order to produce realistic absolute values."),
    },
    "PRES": {
        "standard_name": "sea_water_pressure",
        "units": "decibar",
        "long_name": "Pressure due to seawater",
        "positive": "down",
        "valid_min": -1.0,
        "valid_max": 12000.0,
        "axis": "Z",
    },
    "DEPTH": {
        "standard_name": "depth",
        "units": "m",
        "long_name": "Water depth",
        "positive": "down",
        "valid_min": -1.0,
        "valid_max": 12000.0,
    },
    "SVEL": {
        "standard_name": "speed_of_sound_in_sea_water",
        "units": "m s-1",
        "long_name": "Sound velocity of seawater",
        "valid_min": 1200.0,
        "valid_max": 1800.0,
    },
    "SIGTH": {
        "units": "kg m-3",
        "long_name": "Sigma-theta density",
        "valid_min": 0.0,
        "valid_max": 50.0,
    },
    "NTRI": {
        "standard_name": "mole_concentration_of_nitrite_in_sea_water",
        "units": "mmol m-3",
        "long_name": "Nitrite (NO2-N)",
        "valid_min": -0.1,
        "valid_max": 50.0,
    },
    "NTRA": {
        "standard_name": "mole_concentration_of_nitrate_in_sea_water",
        "units": "mmol m-3",
        "long_name": "Nitrate (NO3-N)",
        "valid_min": -0.1,
        "valid_max": 50.0,
    },
    "PHOS": {
        "standard_name": "mole_concentration_of_phosphate_in_sea_water",
        "units": "mmol m-3",
        "long_name": "Phosphate (PO4-P)",
        "valid_min": -0.1,
        "valid_max": 50.0,
    },
    "DO18": {
        "standard_name": (
            "isotope_ratio_of_18O_to_16O_in_sea_water"
            "_excluding_solutes_and_solids"),
        "units": "1",
        "long_name": ("Per mille deviation in ratio of stable isotopes"
                      " oxygen-18 and oxygen-16 (excluding Solutes and "
                      "Solids) "),
        "valid_min": -10,
        "valid_max": 10,
    },
    "UCUR": {
        "standard_name": "eastward_sea_water_velocity",
        "units": "m s-1",
        "long_name": "Eastward sea water velocity",
        "valid_min": -2.0,
        "valid_max": 2.0,
    },
    "VCUR": {
        "standard_name": "northward_sea_water_velocity",
        "units": "m s-1",
        "long_name": "Northward sea water velocity",
        "valid_min": -2.0,
        "valid_max": 2.0,
    },
    "WCUR": {
        "standard_name": "upward_sea_water_velocity",
        "units": "m s-1",
        "long_name": "Upward sea water velocity",
        "valid_min": -2.0,
        "valid_max": 2.0,
    },

    "TIME": {
        "standard_name": "time",
        "units": "days since 1970-01-01",
        "long_name": "Time stamp of profile",
        "calendar": "gregorian",
        "axis": "T",
    },
    "LONGITUDE": {
        "standard_name": "longitude",
        "units": "degree_east",
        "long_name": "longitude",
        "axis": "X",
    },
    "LATITUDE": {
        "standard_name": "latitude",
        "units": "degree_north",
        "long_name": "latitude",
        "axis": "Y",
    },
    "SBE_FLAG": {
        "long_name": "Quality flag assigned in SBE processing.",
    },

}


standard_attrs_global_ctd = {
    "Conventions": "ACDD-1.3, CF-1.10",
    "source": "CTD profiles from SBE911+",
    "instrument": "In Situ/Laboratory Instruments>Profilers/Sounders>CTD",
    "data_set_language": "eng",
    "keywords_vocabulary": "NASA/GCMD Science Keywords 9.1.5",
    "standard_name_vocabulary": "CF Standard Name Table v83",
    "iso_topic_category": "oceans",
    "platform": "Water-based Platforms>Vessels",
    'platform_vocabulary': 'GCMD Platform Keywords Version 9.1.5',
    'sensor_mount': 'mounted_on_shipborne_profiler'
}

standard_attrs_global_moored = {
    "Conventions": "ACDD-1.3, CF-1.10",
    "source": "Subsurface mooring",
    "instrument": "In Situ/Laboratory Instruments>Profilers/Sounders>CTD",
    "data_set_language": "eng",
    "keywords_vocabulary": "NASA/GCMD Science Keywords 9.1.5",
    "standard_name_vocabulary": "CF Standard Name Table v83",
    "iso_topic_category": "oceans",
    "platform": "Water-based Platforms>Buoys>Moored>MOORINGS",
    'platform_vocabulary': 'GCMD Platform Keywords Version 9.1.5',
    "sensor_mount": "mounted_on_mooring_line",
    'featureType': 'timeSeries',
}


gmdc_keyword_dict_ctd = {
    "PRES": "OCEANS>OCEAN PRESSURE>WATER PRESSURE",
    "TEMP": "OCEANS>OCEAN TEMPERATURE>OCEAN TEMPERATURE PROFILES",
    "CNDC": "OCEANS>SALINITY/DENSITY>CONDUCTIVITY>CONDUCTIVITY PROFILES",
    "PSAL": "OCEANS>SALINITY/DENSITY>OCEAN SALINITY>OCEAN SALINITY PROFILES",
    "CHLA": "OCEANS>OCEAN OPTICS>CHLOROPHYLL>CHLOROPHYLL CONCENTRATION",
    "DOXY": "OCEANS>OCEAN CHEMISTRY>OXYGEN",
    "CDOM": ("OCEANS>OCEAN CHEMISTRY>ORGANIC MATTER>"
             "COLORED DISSOLVED ORGANIC MATTER"),
    "ATTN": "OCEANS>OCEAN OPTICS>ATTENUATION/TRANSMISSION",
    "SVEL": "OCEANS>OCEAN ACOUSTICS>ACOUSTIC VELOCITY",
}


gmdc_keyword_dict_moored = {
    "PRES": "OCEANS>OCEAN PRESSURE>WATER PRESSURE",
    "TEMP": "OCEANS>OCEAN TEMPERATURE>WATER TEMPERATURE",
    "CNDC": "OCEANS>SALINITY/DENSITY>CONDUCTIVITY",
    "PSAL": "OCEANS>SALINITY/DENSITY>SALINITY",
    "CHLA": "OCEANS>OCEAN OPTICS>CHLOROPHYLL",
    "DOXY": "OCEANS>OCEAN CHEMISTRY>OXYGEN",
    "CDOM": ("OCEANS>OCEAN CHEMISTRY>ORGANIC MATTER>"
             "COLORED DISSOLVED ORGANIC MATTER"),
    "ATTN": "OCEANS>OCEAN OPTICS>ATTENUATION/TRANSMISSION",
    "SVEL": "OCEANS>OCEAN ACOUSTICS>ACOUSTIC VELOCITY",
    "UCUR": "OCEANS>OCEAN CIRCULATION>OCEAN CURRENTS",
    "VCUR": "OCEANS>OCEAN CIRCULATION>OCEAN CURRENTS",
    "WCUR": "OCEANS>OCEAN CIRCULATION>OCEAN CURRENTS",
    "SEA_ICE_DRAFT": "CRYOSPHERE>SEA ICE>ICE DRAFT",
    "SEA_ICE_FRACTION": (
        "CRYOSPHERE>SEA ICE>SEA ICE CONCENTRATION>ICE FRACTION"),
    "UICE": "CRYOSPHERE>SEA ICE>SEA ICE MOTION",
    "VICE": "CRYOSPHERE>SEA ICE>SEA ICE MOTION",
}


missing_globals_ctd = {
    "title": "",
    "references": "",
    "comment": "",
    "QC_indicator": "",
    "id": "",
    "project": "",
    "program": "",
    "acknowledgement": "",
    "version": "",
    "cruise_name": "",
}


# Define standard options if we have them
global_attrs_options = {
    # Values and descriptions if we have them
    # From OceanSITES
    "processing_level": [
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
        "Other QC process applied",
    ],
    "QC_indicator": [
        ["unknown", "No QC was performed."],
        ["good data", "All QC tests passed."],
        ["probably good data", "(no description)"],
        [
            "potentially correctable bad data",
            "These data are not to be used without scientific"
            " correction or re-calibration.",
        ],
        ["bad data", "Data have failed one or more tests."],
        [
            "nominal value",
            "Data were not observed but reported. (e.g. instrument target depth.)",
        ],
        [
            "interpolated value",
            "Missing data may be interpolated from neighboring data in space or" "time",
        ],
        ["missing value", "This is a fill value"],
    ],
}


# Preferred order for global and variable attrs

global_attrs_ordered = [
    "title",
    "summary",
    "history",
    "id",
    "version",
    "product_version",
    "doi",
    "citation",
    "date_created",
    "date_issued",
    "date_modified",
    "date_metadata_modified",
    "processing_level",
    "QC_indicator",
    "data_set_progress",
    "comment",
    "source",
    "instrument",
    "instrument_model",
    "instrument_serial_number",
    "instrument_calibration_date",
    "source_file",
    "platform",
    "platform_name",
    "platform_call_sign",
    "platform_imo_code",
    "site_code",
    "site_doi",
    "sensor_mount",
    "institution",
    "project",
    "project_number",
    "program",
    "related_url",
    "area",
    "location",
    "cruise_name",
    "cruise",
    "ship",
    "processing_level",
    "QC_indicator",
    "license",
    "time_coverage_start",
    "time_coverage_end",
    "time_coverage_resolution",
    "time_coverage_duration",
    "geospatial_lat_max",
    "geospatial_lon_max",
    "geospatial_lat_min",
    "geospatial_lon_min",
    "geospatial_bounds",
    "geospatial_bounds_crs",
    "geospatial_vertical_min",
    "geospatial_vertical_max",
    "geospatial_vertical_positive",
    "geospatial_vertical_units",
    "geospatial_bounds_vertical_crs",
    "data_set_language",
    "contributor_name",
    "data_assembly_center",
    "creator_name",
    "creator_email",
    "creator_url",
    "creator_type",
    "creator_institution",
    "publisher_name",
    "publisher_email",
    "publisher_url",
    "publisher_type",
    "publisher_institution",
    "keywords",
    "binned",
    "latitude",
    "longitude",
    "SBE_processing",
    "source_files",
    "featureType",
    "standard_name_vocabulary",
    "instrument_vocabulary",
    "keywords_vocabulary",
    "platform_vocabulary",
    "iso_topic_category",
    "Conventions",
    "data_set_language",
    "naming_authority",
    "references",
    "metadata_link",
    "acknowledgment",
    "acknowledgement",
]

variable_attrs_ordered = [
    "units",
    "standard_name",
    "long_name",
    "comment",
    "processing_level",
    "QC_indicator",
    "calibration_formula",
    "coefficient_A",
    "coefficient_B",
    "sensor_serial_number",
    "sensor_calibration_date",
    "valid_min",
    "valid_max",
    "sensor_mount",
    "reference_scale",
    "coverage_content_type",
    "axis",
]

variable_attrs_necessary = [
    "units",
    "standard_name",
    "long_name",
    "sensor_serial_number",
    "sensor_calibration_date",
    "valid_min",
    "valid_max",
    "coverage_content_type",
    "processing_level",
    "QC_indicator",
]
