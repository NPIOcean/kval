"""
tests/unit_tests/file/test_sbe_xmlcon.py

Unit tests for kval.file.sbe_xmlcon — the SBE .xmlcon parser.

Test data lives in:
    tests/test_data/sbe_files/xmlcon/

Files (one per instrument / config type):
    SBE37SMP-ODO-RS232_03723180_2022_10_09.xmlcon
        SBE37 moored CTD — strain gauge pressure, A0/A1/A2/A3 temperature,
        SBE63 oxygen, 4 sensors total
    STA0243.XMLCON
        SBE911, complex modern config — dual T/C, dual SBE43 oxygen,
        ECO fluorometer, CDOM, CStar, altimeter, SPAR, NMEA, 3 NotInUse
    0015.XMLCON
        SBE911 — dual T/C, PAR, CStar, WETstar, SBE43, altimeter, SPAR,
        NMEA, 4 NotInUse
    205.XMLCON
        SBE911 minimal — single T/C, Seapoint fluorometer,
        VoltageWordsSuppressed=1, no NMEA, 8 NotInUse
    STNR52.XMLCON
        SBE911 minimal — identical sensor hardware to 205 but different
        pressure offset, VoltageWordsSuppressed=1, no NMEA
"""

import pytest
from pathlib import Path
from kval.file.sbe_xmlcon import (
    parse_xmlcon,
    get_sensors_by_type,
    get_active_sensors,
    summarise,
    _parse_cal_date,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TEST_DATA_DIR = Path("tests/test_data/sbe_files/xmlcon")

SBE37_FILE   = TEST_DATA_DIR / "SBE37SMP-ODO-RS232_03723180_2022_10_09.xmlcon"
STA0243_FILE = TEST_DATA_DIR / "STA0243.XMLCON"
FILE_0015    = TEST_DATA_DIR / "0015.XMLCON"
FILE_205     = TEST_DATA_DIR / "205.XMLCON"
STNR52_FILE  = TEST_DATA_DIR / "STNR52.XMLCON"

ALL_FILES = [SBE37_FILE, STA0243_FILE, FILE_0015, FILE_205, STNR52_FILE]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg_sbe37():
    return parse_xmlcon(SBE37_FILE)

@pytest.fixture(scope="module")
def cfg_sta0243():
    return parse_xmlcon(STA0243_FILE)

@pytest.fixture(scope="module")
def cfg_0015():
    return parse_xmlcon(FILE_0015)

@pytest.fixture(scope="module")
def cfg_205():
    return parse_xmlcon(FILE_205)

@pytest.fixture(scope="module")
def cfg_stnr52():
    return parse_xmlcon(STNR52_FILE)

@pytest.fixture(
    scope="module",
    params=ALL_FILES,
    ids=["sbe37", "sta0243", "0015", "205", "stnr52"],
)
def any_cfg(request):
    """Parametrised fixture: runs the test against all five xmlcon files."""
    return parse_xmlcon(request.param)


# ---------------------------------------------------------------------------
# parse_xmlcon — top-level output structure
# ---------------------------------------------------------------------------

class TestParseXmlcon:

    def test_returns_dict(self, any_cfg):
        assert isinstance(any_cfg, dict)

    def test_has_instrument_key(self, any_cfg):
        assert "instrument" in any_cfg

    def test_has_sensors_key(self, any_cfg):
        assert "sensors" in any_cfg

    def test_sensors_is_list(self, any_cfg):
        assert isinstance(any_cfg["sensors"], list)

    def test_sensors_not_empty(self, any_cfg):
        assert len(any_cfg["sensors"]) > 0

    def test_raises_on_missing_file(self):
        with pytest.raises(Exception):
            parse_xmlcon(TEST_DATA_DIR / "does_not_exist.xmlcon")


# ---------------------------------------------------------------------------
# Instrument-level fields
# ---------------------------------------------------------------------------

class TestInstrument:

    def test_name_sbe37(self, cfg_sbe37):
        assert cfg_sbe37["instrument"]["name"] == "SBE 37 Microcat"

    def test_name_sbe911(self, cfg_sta0243):
        assert cfg_sta0243["instrument"]["name"] == "SBE 911plus/917plus CTD"

    def test_device_type_sbe37(self, cfg_sbe37):
        assert cfg_sbe37["instrument"]["device_type"] == "SBE37SMP-ODO-RS232"

    def test_firmware_sbe37(self, cfg_sbe37):
        assert cfg_sbe37["instrument"]["firmware_version"] == "6.2.0"

    def test_voltage_words_suppressed_zero(self, cfg_sta0243):
        assert cfg_sta0243["instrument"]["voltage_words_suppressed"] == 0

    def test_voltage_words_suppressed_one_205(self, cfg_205):
        assert cfg_205["instrument"]["voltage_words_suppressed"] == 1

    def test_voltage_words_suppressed_one_stnr52(self, cfg_stnr52):
        assert cfg_stnr52["instrument"]["voltage_words_suppressed"] == 1

    def test_nmea_position_added_true(self, cfg_sta0243):
        assert cfg_sta0243["instrument"]["nmea_position_added"] is True

    def test_nmea_position_added_true_0015(self, cfg_0015):
        assert cfg_0015["instrument"]["nmea_position_added"] is True

    def test_nmea_position_added_false_205(self, cfg_205):
        assert cfg_205["instrument"]["nmea_position_added"] is False

    def test_nmea_position_added_false_stnr52(self, cfg_stnr52):
        assert cfg_stnr52["instrument"]["nmea_position_added"] is False

    def test_surface_par_added_true(self, cfg_sta0243):
        assert cfg_sta0243["instrument"]["surface_par_added"] is True

    def test_surface_par_added_false(self, cfg_205):
        assert cfg_205["instrument"]["surface_par_added"] is False

    def test_scan_time_added_true(self, cfg_0015):
        assert cfg_0015["instrument"]["scan_time_added"] is True

    def test_scan_time_added_false(self, cfg_stnr52):
        assert cfg_stnr52["instrument"]["scan_time_added"] is False

    def test_sample_interval_sbe37(self, cfg_sbe37):
        assert cfg_sbe37["instrument"]["sample_interval_seconds"] == pytest.approx(3600.0)

    def test_scans_to_average_default(self, any_cfg):
        assert any_cfg["instrument"]["scans_to_average"] == 1


# ---------------------------------------------------------------------------
# Sensor array — structure and ordering
# ---------------------------------------------------------------------------

class TestSensorArray:

    def test_sbe37_sensor_count(self, cfg_sbe37):
        assert len(cfg_sbe37["sensors"]) == 4

    def test_sensor_count_15_sta0243(self, cfg_sta0243):
        assert len(cfg_sta0243["sensors"]) == 15

    def test_sensor_count_15_0015(self, cfg_0015):
        assert len(cfg_0015["sensors"]) == 15

    def test_sensor_count_11_205(self, cfg_205):
        assert len(cfg_205["sensors"]) == 11

    def test_sensor_count_11_stnr52(self, cfg_stnr52):
        assert len(cfg_stnr52["sensors"]) == 11

    def test_sensor_indices_are_sequential(self, any_cfg):
        for i, sensor in enumerate(any_cfg["sensors"]):
            assert sensor["index"] == i

    def test_each_sensor_has_required_keys(self, any_cfg):
        required = {"index", "sensor_id", "type", "in_use"}
        for sensor in any_cfg["sensors"]:
            assert required.issubset(sensor.keys()), \
                f"Sensor at index {sensor.get('index')} missing keys"

    def test_each_active_sensor_has_coefficients(self, any_cfg):
        for sensor in any_cfg["sensors"]:
            if sensor["in_use"]:
                assert "coefficients" in sensor, \
                    f"Active sensor '{sensor['type']}' at index {sensor['index']} has no coefficients"

    def test_not_in_use_flagged_correctly(self, any_cfg):
        for sensor in any_cfg["sensors"]:
            if sensor["type"] == "NotInUse":
                assert sensor["in_use"] is False

    def test_not_in_use_count_sta0243(self, cfg_sta0243):
        assert len([s for s in cfg_sta0243["sensors"] if s["type"] == "NotInUse"]) == 3

    def test_not_in_use_count_205(self, cfg_205):
        assert len([s for s in cfg_205["sensors"] if s["type"] == "NotInUse"]) == 7

    def test_temperature_at_index_0(self, any_cfg):
        assert any_cfg["sensors"][0]["type"] == "TemperatureSensor"

    def test_conductivity_at_index_1(self, any_cfg):
        assert any_cfg["sensors"][1]["type"] == "ConductivitySensor"

    def test_sbe911_pressure_at_index_2(self, cfg_sta0243):
        assert cfg_sta0243["sensors"][2]["type"] == "PressureSensor"

    def test_sbe37_pressure_at_index_3(self, cfg_sbe37):
        """SBE37 has oxygen at index 2, pressure at index 3."""
        assert cfg_sbe37["sensors"][3]["type"] == "PressureSensor"


# ---------------------------------------------------------------------------
# Temperature sensor
# ---------------------------------------------------------------------------

class TestTemperatureSensor:

    def test_sbe37_uses_a0_coefficients(self, cfg_sbe37):
        temp = get_sensors_by_type(cfg_sbe37, "TemperatureSensor")[0]
        assert temp["use_GJ_equation"] is False
        for key in ["A0", "A1", "A2", "A3"]:
            assert key in temp["coefficients"]
            assert temp["coefficients"][key] is not None

    def test_sbe911_uses_ghij_coefficients(self, cfg_sta0243):
        temp = get_sensors_by_type(cfg_sta0243, "TemperatureSensor")[0]
        assert temp["use_GJ_equation"] is True
        for key in ["G", "H", "I", "J", "F0"]:
            assert key in temp["coefficients"]
            assert temp["coefficients"][key] is not None

    def test_sbe37_primary_temp_coefficients(self, cfg_sbe37):
        c = get_sensors_by_type(cfg_sbe37, "TemperatureSensor")[0]["coefficients"]
        assert c["A0"] == pytest.approx(-1.91141700e-4)
        assert c["A1"] == pytest.approx(3.16128100e-4)
        assert c["A2"] == pytest.approx(-4.74610900e-6)
        assert c["A3"] == pytest.approx(2.09572600e-7)

    def test_sta0243_primary_temp_coefficients(self, cfg_sta0243):
        c = get_sensors_by_type(cfg_sta0243, "TemperatureSensor")[0]["coefficients"]
        assert c["G"] == pytest.approx(4.36105290e-3)
        assert c["H"] == pytest.approx(6.40605936e-4)
        assert c["F0"] == pytest.approx(1000.0)

    def test_0015_primary_temp_g_coefficient(self, cfg_0015):
        c = get_sensors_by_type(cfg_0015, "TemperatureSensor")[0]["coefficients"]
        assert c["G"] == pytest.approx(4.41402855e-3)

    def test_205_primary_temp_g_coefficient(self, cfg_205):
        c = get_sensors_by_type(cfg_205, "TemperatureSensor")[0]["coefficients"]
        assert c["G"] == pytest.approx(4.39263237e-3)

    def test_dual_temperature_sensors_sta0243(self, cfg_sta0243):
        temps = get_sensors_by_type(cfg_sta0243, "TemperatureSensor")
        assert len(temps) == 2
        assert temps[0]["serial_number"] != temps[1]["serial_number"]

    def test_dual_temperature_sensors_0015(self, cfg_0015):
        assert len(get_sensors_by_type(cfg_0015, "TemperatureSensor")) == 2

    def test_single_temperature_sensor_205(self, cfg_205):
        assert len(get_sensors_by_type(cfg_205, "TemperatureSensor")) == 1

    def test_slope_and_offset_present(self, any_cfg):
        for sensor in get_sensors_by_type(any_cfg, "TemperatureSensor"):
            assert "Slope" in sensor["coefficients"]
            assert "Offset" in sensor["coefficients"]

    def test_205_and_stnr52_share_same_temp_sensor(self, cfg_205, cfg_stnr52):
        t205  = get_sensors_by_type(cfg_205,    "TemperatureSensor")[0]
        tstnr = get_sensors_by_type(cfg_stnr52, "TemperatureSensor")[0]
        assert t205["serial_number"] == tstnr["serial_number"]
        assert t205["coefficients"]["G"] == pytest.approx(tstnr["coefficients"]["G"])


# ---------------------------------------------------------------------------
# Conductivity sensor
# ---------------------------------------------------------------------------

class TestConductivitySensor:

    def test_all_use_gj_equation(self, any_cfg):
        for sensor in get_sensors_by_type(any_cfg, "ConductivitySensor"):
            assert sensor["use_GJ_equation"] is True

    def test_required_coefficients_present(self, any_cfg):
        for sensor in get_sensors_by_type(any_cfg, "ConductivitySensor"):
            for key in ["G", "H", "I", "J", "CPcor", "CTcor"]:
                assert key in sensor["coefficients"]
                assert sensor["coefficients"][key] is not None

    def test_sta0243_primary_cond_coefficients(self, cfg_sta0243):
        c = get_sensors_by_type(cfg_sta0243, "ConductivitySensor")[0]["coefficients"]
        assert c["G"] == pytest.approx(-1.03879335e1)
        assert c["CPcor"] == pytest.approx(-9.57e-8)
        assert c["CTcor"] == pytest.approx(3.25e-6)

    def test_dual_conductivity_sensors_sta0243(self, cfg_sta0243):
        conds = get_sensors_by_type(cfg_sta0243, "ConductivitySensor")
        assert len(conds) == 2
        assert conds[0]["serial_number"] != conds[1]["serial_number"]

    def test_single_conductivity_sensor_205(self, cfg_205):
        assert len(get_sensors_by_type(cfg_205, "ConductivitySensor")) == 1


# ---------------------------------------------------------------------------
# Pressure sensor
# ---------------------------------------------------------------------------

class TestPressureSensor:

    def test_sbe37_is_strain_gauge(self, cfg_sbe37):
        pres = get_sensors_by_type(cfg_sbe37, "PressureSensor")[0]
        assert pres["pressure_type"] == "strain_gauge"

    def test_sbe911_is_digiquartz(self, cfg_sta0243):
        pres = get_sensors_by_type(cfg_sta0243, "PressureSensor")[0]
        assert pres["pressure_type"] == "digiquartz"

    def test_digiquartz_required_coefficients(self, cfg_sta0243):
        c = get_sensors_by_type(cfg_sta0243, "PressureSensor")[0]["coefficients"]
        for key in ["C1", "C2", "C3", "D1", "D2", "T1", "T2", "T3", "T4", "AD590M", "AD590B"]:
            assert key in c

    def test_strain_gauge_required_coefficients(self, cfg_sbe37):
        c = get_sensors_by_type(cfg_sbe37, "PressureSensor")[0]["coefficients"]
        for key in ["PA0", "PA1", "PA2", "PTEMPA0", "PTCA0", "PTCB0"]:
            assert key in c

    def test_sta0243_pressure_coefficients(self, cfg_sta0243):
        c = get_sensors_by_type(cfg_sta0243, "PressureSensor")[0]["coefficients"]
        assert c["C1"] == pytest.approx(-4.517322e4)
        assert c["AD590M"] == pytest.approx(1.28164e-2)

    def test_0015_pressure_non_default_slope_offset(self, cfg_0015):
        c = get_sensors_by_type(cfg_0015, "PressureSensor")[0]["coefficients"]
        assert c["Slope"] == pytest.approx(1.0001357)
        assert c["Offset"] == pytest.approx(-0.2008)

    def test_205_and_stnr52_same_serial_different_offset(self, cfg_205, cfg_stnr52):
        """205 and STNR52 share the same pressure sensor but have different offsets."""
        p205  = get_sensors_by_type(cfg_205,    "PressureSensor")[0]
        pstnr = get_sensors_by_type(cfg_stnr52, "PressureSensor")[0]
        assert p205["serial_number"] == pstnr["serial_number"]
        assert p205["coefficients"]["Offset"] != pstnr["coefficients"]["Offset"]


# ---------------------------------------------------------------------------
# Oxygen sensor (SBE43)
# ---------------------------------------------------------------------------

class TestOxygenSensor:

    def test_sta0243_has_two_oxygen_sensors(self, cfg_sta0243):
        assert len(get_sensors_by_type(cfg_sta0243, "OxygenSensor")) == 2

    def test_0015_has_one_oxygen_sensor(self, cfg_0015):
        assert len(get_sensors_by_type(cfg_0015, "OxygenSensor")) == 1

    def test_205_has_no_oxygen_sensor(self, cfg_205):
        assert len(get_sensors_by_type(cfg_205, "OxygenSensor")) == 0

    def test_stnr52_has_no_oxygen_sensor(self, cfg_stnr52):
        assert len(get_sensors_by_type(cfg_stnr52, "OxygenSensor")) == 0

    def test_uses_2007_equation(self, cfg_sta0243):
        for sensor in get_sensors_by_type(cfg_sta0243, "OxygenSensor"):
            assert sensor["use_2007_equation"] is True

    def test_2007_coefficients_present(self, cfg_sta0243):
        for sensor in get_sensors_by_type(cfg_sta0243, "OxygenSensor"):
            for key in ["Soc", "offset", "A", "B", "C", "E", "Tau20", "H1", "H2", "H3"]:
                assert key in sensor["coefficients"]

    def test_sta0243_primary_oxygen_soc(self, cfg_sta0243):
        oxy = get_sensors_by_type(cfg_sta0243, "OxygenSensor")[0]
        assert oxy["coefficients"]["Soc"] == pytest.approx(5.0991e-1)

    def test_0015_oxygen_soc(self, cfg_0015):
        oxy = get_sensors_by_type(cfg_0015, "OxygenSensor")[0]
        assert oxy["coefficients"]["Soc"] == pytest.approx(4.5490e-1)

    def test_sbe37_has_sbe63_style_oxygen(self, cfg_sbe37):
        """SBE37 ODO uses SensorID 72 (SBE63 optical oxygen)."""
        oxy = get_sensors_by_type(cfg_sbe37, "OxygenSensor")
        assert len(oxy) == 1
        assert oxy[0]["sensor_id"] == 72


# ---------------------------------------------------------------------------
# Auxiliary sensors
# ---------------------------------------------------------------------------

class TestAuxiliarySensors:

    def test_eco_fluorometer_sta0243(self, cfg_sta0243):
        eco = get_sensors_by_type(cfg_sta0243, "FluoroWetlabECO_AFL_FL_Sensor")
        assert len(eco) == 1
        assert eco[0]["coefficients"]["ScaleFactor"] == pytest.approx(6.0)
        assert eco[0]["coefficients"]["Vblank"] == pytest.approx(0.056)

    def test_cdom_sta0243(self, cfg_sta0243):
        cdom = get_sensors_by_type(cfg_sta0243, "FluoroWetlabCDOM_Sensor")
        assert len(cdom) == 1
        assert cdom[0]["coefficients"]["ScaleFactor"] == pytest.approx(1.0)

    def test_cstar_sta0243(self, cfg_sta0243):
        cstar = get_sensors_by_type(cfg_sta0243, "WET_LabsCStar")
        assert len(cstar) == 1
        assert cstar[0]["coefficients"]["M"] == pytest.approx(21.3129)
        assert cstar[0]["coefficients"]["PathLength"] == pytest.approx(0.25)

    def test_cstar_0015(self, cfg_0015):
        cstar = get_sensors_by_type(cfg_0015, "WET_LabsCStar")
        assert len(cstar) == 1
        assert cstar[0]["coefficients"]["M"] == pytest.approx(21.86)

    def test_altimeter_sta0243(self, cfg_sta0243):
        alt = get_sensors_by_type(cfg_sta0243, "AltimeterSensor")
        assert len(alt) == 1
        assert alt[0]["coefficients"]["ScaleFactor"] == pytest.approx(15.0)

    def test_spar_sta0243_has_conversion_units(self, cfg_sta0243):
        spar = get_sensors_by_type(cfg_sta0243, "SPAR_Sensor")
        assert len(spar) == 1
        assert spar[0]["coefficients"]["ConversionFactor"] == pytest.approx(1.6466e3)
        assert spar[0]["coefficients"]["ConversionUnits"] == 1

    def test_spar_0015_missing_conversion_units(self, cfg_0015):
        """0015 SPAR lacks ConversionUnits — should parse without error, value is None."""
        spar = get_sensors_by_type(cfg_0015, "SPAR_Sensor")
        assert len(spar) == 1
        assert spar[0]["coefficients"]["ConversionUnits"] is None

    def test_par_biospherical_0015(self, cfg_0015):
        par = get_sensors_by_type(cfg_0015, "PAR_BiosphericalLicorChelseaSensor")
        assert len(par) == 1
        assert par[0]["coefficients"]["CalibrationConstant"] == pytest.approx(1.6502e10)

    def test_wetstar_0015(self, cfg_0015):
        wetstar = get_sensors_by_type(cfg_0015, "FluoroWetlabWetstarSensor")
        assert len(wetstar) == 1
        assert wetstar[0]["coefficients"]["ScaleFactor"] == pytest.approx(6.0)

    def test_seapoint_fluorometer_205(self, cfg_205):
        seapoint = get_sensors_by_type(cfg_205, "FluoroSeapointSensor")
        assert len(seapoint) == 1
        assert seapoint[0]["coefficients"]["GainSetting"] == 1

    def test_seapoint_fluorometer_stnr52(self, cfg_stnr52):
        assert len(get_sensors_by_type(cfg_stnr52, "FluoroSeapointSensor")) == 1


# ---------------------------------------------------------------------------
# Calibration date parsing
# ---------------------------------------------------------------------------

class TestCalibrationDateParsing:

    @pytest.mark.parametrize("raw, expected", [
        ("30-Mar-23",    "2023-03-30"),
        ("30-Mar-2023",  "2023-03-30"),
        ("28.03.23",     "2023-03-28"),
        ("28.11.2007",   "2007-11-28"),
        ("2019-10-01",   "2019-10-01"),
        ("24/12-2017",   "2017-12-24"),
        ("13-AUG-2024",  "2024-08-13"),
        ("01-04-2016",   "2016-04-01"),
        ("03-May-24",    "2024-05-03"),
        ("19-Dec-17",    "2017-12-19"),
        ("27-Apr-23",    "2023-04-27"),
        ("09-Aug-22",    "2022-08-09"),
    ])
    def test_known_date_formats(self, raw, expected):
        assert _parse_cal_date(raw) == expected

    def test_none_returns_none(self):
        assert _parse_cal_date(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_cal_date("") is None

    def test_unparseable_returns_original(self):
        assert _parse_cal_date("not-a-date") == "not-a-date"

    def test_all_active_sensor_dates_are_iso_format(self, any_cfg):
        """Every active sensor with a non-None date should be YYYY-MM-DD."""
        for sensor in get_active_sensors(any_cfg):
            date = sensor.get("calibration_date")
            if date:
                assert len(date) == 10, \
                    f"Date '{date}' is not YYYY-MM-DD in sensor {sensor['type']}"
                assert date[4] == "-" and date[7] == "-", \
                    f"Date '{date}' has wrong separators in sensor {sensor['type']}"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_get_sensors_by_type_correct_count(self, cfg_sta0243):
        assert len(get_sensors_by_type(cfg_sta0243, "TemperatureSensor")) == 2

    def test_get_sensors_by_type_unknown_returns_empty(self, cfg_sta0243):
        assert get_sensors_by_type(cfg_sta0243, "NonExistentSensor") == []

    def test_get_active_sensors_excludes_not_in_use(self, any_cfg):
        active = get_active_sensors(any_cfg)
        assert all(s["in_use"] for s in active)
        assert all(s["type"] != "NotInUse" for s in active)

    def test_get_active_sensors_count_sta0243(self, cfg_sta0243):
        # 15 total - 3 NotInUse = 12 active
        assert len(get_active_sensors(cfg_sta0243)) == 12

    def test_get_active_sensors_count_0015(self, cfg_0015):
        # 15 total - 4 NotInUse = 11 active
        assert len(get_active_sensors(cfg_0015)) == 11

    def test_get_active_sensors_count_205(self, cfg_205):
        # 11 total - 8 NotInUse = 3 active (T, C, P + Seapoint)
        assert len(get_active_sensors(cfg_205)) == 4

    def test_get_active_sensors_count_stnr52(self, cfg_stnr52):
        assert len(get_active_sensors(cfg_stnr52)) == 4

    def test_summarise_returns_string(self, any_cfg):
        assert isinstance(summarise(any_cfg), str)

    def test_summarise_contains_sbe37_name(self, cfg_sbe37):
        assert "SBE 37 Microcat" in summarise(cfg_sbe37)

    def test_summarise_contains_sbe911_name(self, cfg_sta0243):
        assert "SBE 911plus" in summarise(cfg_sta0243)

    def test_summarise_contains_not_in_use(self, cfg_sta0243):
        assert "not in use" in summarise(cfg_sta0243)

    def test_summarise_contains_serial_number(self, cfg_sta0243):
        assert "5127" in summarise(cfg_sta0243)