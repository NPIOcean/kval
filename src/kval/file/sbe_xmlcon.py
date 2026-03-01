"""
kval.file.sbe_xmlcon

Parser for Sea-Bird .xmlcon configuration files.

Extracts instrument configuration, sensor list, and calibration coefficients
from .xmlcon files produced by SBE Data Processing software.

Tested against:
  - SBE 911plus/917plus CTD (SensorID 8)
  - SBE 37 Microcat (SensorID 14)

Sensor types handled:
  - TemperatureSensor (SensorID 55, 58)
  - ConductivitySensor (SensorID 3)
  - PressureSensor / Digiquartz (SensorID 45, 46)
  - OxygenSensor / SBE43 (SensorID 38)
  - OxygenSensor / SBE63 (SensorID 63)  [coefficients parsed, untested]
  - FluoroWetlabECO_AFL_FL_Sensor (SensorID 20)  [WET Labs ECO-AFL/FL]
  - FluoroWetlabWetstarSensor (SensorID 21)
  - FluoroWetlabCDOM_Sensor (SensorID 19)
  - FluoroSeapointSensor (SensorID 11)
  - WET_LabsCStar (SensorID 71)  [transmissometer]
  - AltimeterSensor (SensorID 0)
  - PAR_BiosphericalLicorChelseaSensor (SensorID 42)
  - SPAR_Sensor (SensorID 51)
  - NotInUse (SensorID 27)
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_xmlcon(path: str | Path) -> dict:
    """
    Parse an SBE .xmlcon file and return a structured configuration dict.

    Parameters
    ----------
    path : str or Path
        Path to the .xmlcon file.

    Returns
    -------
    dict with keys:
        'instrument'  : dict of top-level instrument settings
        'sensors'     : list of sensor dicts, one per entry in SensorArray
                        (including NotInUse entries, preserving index order)

    Raises
    ------
    ValueError
        If the file cannot be parsed or lacks expected structure.

    Notes
    -----
    Sensor list index matches the physical channel order in the hex data
    stream, so NotInUse entries are retained and must not be dropped.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    instrument_el = root.find('Instrument')
    if instrument_el is None:
        raise ValueError(f"No <Instrument> element found in {path}")

    instrument = _parse_instrument(instrument_el)
    sensors = _parse_sensor_array(instrument_el)

    return {
        'instrument': instrument,
        'sensors': sensors,
    }


# ---------------------------------------------------------------------------
# Instrument-level parsing
# ---------------------------------------------------------------------------

def _parse_instrument(el: ET.Element) -> dict:
    """
    Parse top-level instrument configuration fields.

    Handles both SBE911 schema (FrequencyChannelsSuppressed) and
    SBE37 schema (FrequencyChannelsAdded / no NMEA fields).
    """

    def _int(tag, default=None):
        child = el.find(tag)
        return int(child.text) if child is not None else default

    def _float(tag, default=None):
        child = el.find(tag)
        return float(child.text) if child is not None else default

    def _str(tag, default=None):
        child = el.find(tag)
        return child.text.strip() if child is not None and child.text else default

    # SBE37 uses FrequencyChannelsAdded; SBE911 uses FrequencyChannelsSuppressed
    freq_suppressed = _int('FrequencyChannelsSuppressed')
    freq_added = _int('FrequencyChannelsAdded')

    return {
        'type_id':                       _int('Type'),
        'name':                          _str('Name'),      # XML parser normalises <n> -> <Name>
        'device_type':                   _str('DeviceType'),
        'firmware_version':              _str('FirmwareVersion'),
        'frequency_channels_suppressed': freq_suppressed,
        'frequency_channels_added':      freq_added,        # SBE37 only
        'voltage_words_suppressed':      _int('VoltageWordsSuppressed', 0),
        'scans_to_average':              _int('ScansToAverage', 1),
        'sample_interval_seconds':       _float('SampleIntervalSeconds'),
        'deck_unit_version':             _int('DeckUnitVersion'),
        'surface_par_added':             bool(_int('SurfaceParVoltageAdded', 0)),
        'scan_time_added':               bool(_int('ScanTimeAdded', 0)),
        'nmea_position_added':           bool(_int('NmeaPositionDataAdded', 0)),
        'nmea_depth_added':              bool(_int('NmeaDepthDataAdded', 0)),
        'nmea_time_added':               bool(_int('NmeaTimeAdded', 0)),
    }


# ---------------------------------------------------------------------------
# Sensor array parsing
# ---------------------------------------------------------------------------

def _parse_sensor_array(instrument_el: ET.Element) -> list[dict]:
    """Parse all sensors in SensorArray, preserving index order."""
    array_el = instrument_el.find('SensorArray')
    if array_el is None:
        raise ValueError("No <SensorArray> found")

    sensors = []
    for sensor_el in array_el.findall('Sensor'):
        index = int(sensor_el.get('index', -1))
        sensor_id = int(sensor_el.get('SensorID', -1))

        # The sensor type is the tag name of the single child element
        children = [c for c in sensor_el if not isinstance(c.tag, str) is False]
        if not children:
            sensors.append({'index': index, 'sensor_id': sensor_id,
                            'type': 'Unknown', 'in_use': False})
            continue

        type_el = children[0]
        sensor_type = type_el.tag

        parsed = _parse_sensor(sensor_type, type_el)
        parsed['index'] = index
        parsed['sensor_id'] = sensor_id
        parsed['type'] = sensor_type
        parsed['in_use'] = (sensor_type != 'NotInUse')
        sensors.append(parsed)

    return sensors


def _parse_sensor(sensor_type: str, el: ET.Element) -> dict:
    """Dispatch to the appropriate sensor parser."""
    parsers = {
        'TemperatureSensor':                _parse_temperature,
        'ConductivitySensor':               _parse_conductivity,
        'PressureSensor':                   _parse_pressure,
        'OxygenSensor':                     _parse_oxygen_sbe43,
        'FluoroWetlabECO_AFL_FL_Sensor':    _parse_eco_fluorometer,
        'FluoroWetlabWetstarSensor':        _parse_wetstar_fluorometer,
        'FluoroWetlabCDOM_Sensor':          _parse_cdom,
        'FluoroSeapointSensor':             _parse_seapoint_fluorometer,
        'WET_LabsCStar':                    _parse_cstar,
        'AltimeterSensor':                  _parse_altimeter,
        'PAR_BiosphericalLicorChelseaSensor': _parse_par,
        'SPAR_Sensor':                      _parse_spar,
        'NotInUse':                         _parse_not_in_use,
    }

    parser = parsers.get(sensor_type, _parse_generic)
    result = parser(el)
    return result


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _get_float(el: ET.Element, tag: str, default: float | None = None) -> float | None:
    child = el.find(tag)
    if child is not None and child.text and child.text.strip():
        return float(child.text)
    return default


def _get_int(el: ET.Element, tag: str, default: int | None = None) -> int | None:
    child = el.find(tag)
    if child is not None and child.text and child.text.strip():
        return int(child.text)
    return default


def _get_str(el: ET.Element, tag: str, default: str | None = None) -> str | None:
    child = el.find(tag)
    if child is not None and child.text and child.text.strip():
        return child.text.strip()
    return default


def _parse_cal_date(date_str: str | None) -> str | None:
    """
    Normalise calibration date strings to ISO 8601 (YYYY-MM-DD).

    Handles the wide variety of date formats seen across xmlcon files:
      '30-Mar-23', '28.03.23', '2019-10-01', '24/12-2017',
      '13-AUG-2024', '01-04-2016', '28.11.2007', '03-May-24'
    Returns the original string unchanged if parsing fails.
    """
    if not date_str:
        return None

    formats = [
        '%d-%b-%y',    # 30-Mar-23
        '%d-%b-%Y',    # 30-Mar-2023, 13-AUG-2024
        '%d.%m.%y',    # 28.03.23
        '%d.%m.%Y',    # 28.11.2007, 01.04.2016
        '%Y-%m-%d',    # 2019-10-01
        '%d/%m-%Y',    # 24/12-2017
        '%d/%m/%Y',    # 24/12/2017
        '%d-%m-%Y',    # 01-04-2016 (ambiguous - try after %d-%b-%Y)
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue

    # Could not parse - return as-is with a warning
    return date_str


def _common_fields(el: ET.Element) -> dict:
    """Fields present on almost every sensor."""
    return {
        'serial_number':    _get_str(el, 'SerialNumber'),
        'calibration_date': _parse_cal_date(_get_str(el, 'CalibrationDate')),
    }


# ---------------------------------------------------------------------------
# Individual sensor parsers
# ---------------------------------------------------------------------------

def _parse_temperature(el: ET.Element) -> dict:
    """
    Temperature sensor.

    SBE3 (SensorID 55, used on SBE911): G/H/I/J equation when UseG_J == 1.
    SBE4 (SensorID 58, used on SBE37):  A0/A1/A2/A3 equation (no UseG_J field).
    """
    result = _common_fields(el)

    use_gj_el = el.find('UseG_J')
    has_gj_field = use_gj_el is not None

    # SBE37-style: no UseG_J field, uses A0/A1/A2/A3
    if not has_gj_field and el.find('A0') is not None:
        result['use_GJ_equation'] = False
        result['coefficients'] = {
            'A0':     _get_float(el, 'A0'),
            'A1':     _get_float(el, 'A1'),
            'A2':     _get_float(el, 'A2'),
            'A3':     _get_float(el, 'A3'),
            'Slope':  _get_float(el, 'Slope', 1.0),
            'Offset': _get_float(el, 'Offset', 0.0),
        }
        return result

    # SBE911-style: UseG_J present
    use_gj = int(use_gj_el.text) if has_gj_field else 1
    result['use_GJ_equation'] = bool(use_gj)

    if use_gj:
        result['coefficients'] = {
            'G':      _get_float(el, 'G'),
            'H':      _get_float(el, 'H'),
            'I':      _get_float(el, 'I'),
            'J':      _get_float(el, 'J'),
            'F0':     _get_float(el, 'F0'),
            'Slope':  _get_float(el, 'Slope', 1.0),
            'Offset': _get_float(el, 'Offset', 0.0),
        }
    else:
        result['coefficients'] = {
            'A':      _get_float(el, 'A'),
            'B':      _get_float(el, 'B'),
            'C':      _get_float(el, 'C'),
            'D':      _get_float(el, 'D'),
            'Slope':  _get_float(el, 'Slope', 1.0),
            'Offset': _get_float(el, 'Offset', 0.0),
        }
    return result


def _parse_conductivity(el: ET.Element) -> dict:
    """
    SBE4 conductivity sensor.
    Uses G/H/I/J equation (equation="1") when UseG_J == 1.
    """
    use_gj = _get_int(el, 'UseG_J', 1)
    result = _common_fields(el)
    result['use_GJ_equation'] = bool(use_gj)
    result['conductivity_type'] = _get_int(el, 'ConductivityType')

    # Find the correct Coefficients block
    eq_target = '1' if use_gj else '0'
    coeffs = {}
    for coeff_el in el.findall('Coefficients'):
        if coeff_el.get('equation') == eq_target:
            if use_gj:
                coeffs = {
                    'G':     _get_float(coeff_el, 'G'),
                    'H':     _get_float(coeff_el, 'H'),
                    'I':     _get_float(coeff_el, 'I'),
                    'J':     _get_float(coeff_el, 'J'),
                    'CPcor': _get_float(coeff_el, 'CPcor'),
                    'CTcor': _get_float(coeff_el, 'CTcor'),
                    'WBOTC': _get_float(coeff_el, 'WBOTC'),
                }
            else:
                coeffs = {
                    'A':     _get_float(coeff_el, 'A'),
                    'B':     _get_float(coeff_el, 'B'),
                    'C':     _get_float(coeff_el, 'C'),
                    'D':     _get_float(coeff_el, 'D'),
                    'M':     _get_float(coeff_el, 'M'),
                    'CPcor': _get_float(coeff_el, 'CPcor'),
                }
            break

    coeffs['Slope'] = _get_float(el, 'Slope', 1.0)
    coeffs['Offset'] = _get_float(el, 'Offset', 0.0)
    result['coefficients'] = coeffs
    return result


def _parse_pressure(el: ET.Element) -> dict:
    """
    Digiquartz pressure sensor (SensorID 45) or strain gauge (SensorID 46).
    SensorID 45 = Digiquartz (SBE911) - uses C1/C2/C3/D1/D2/T1..T5/AD590
    SensorID 46 = strain gauge (SBE37)  - uses PA0/PA1/PA2/PTEMPA0..PTCB2
    Both are handled here; whichever coefficients are present are extracted.
    """
    result = _common_fields(el)

    # Digiquartz coefficients (SBE911)
    dq_coeffs = {k: _get_float(el, k) for k in
                 ['C1', 'C2', 'C3', 'D1', 'D2',
                  'T1', 'T2', 'T3', 'T4', 'T5',
                  'AD590M', 'AD590B']}
    dq_coeffs = {k: v for k, v in dq_coeffs.items() if v is not None}

    # Strain gauge coefficients (SBE37)
    sg_coeffs = {k: _get_float(el, k) for k in
                 ['PA0', 'PA1', 'PA2',
                  'PTEMPA0', 'PTEMPA1', 'PTEMPA2',
                  'PTCA0', 'PTCA1', 'PTCA2',
                  'PTCB0', 'PTCB1', 'PTCB2']}
    sg_coeffs = {k: v for k, v in sg_coeffs.items() if v is not None}

    if dq_coeffs:
        result['pressure_type'] = 'digiquartz'
        result['coefficients'] = dq_coeffs
    else:
        result['pressure_type'] = 'strain_gauge'
        result['coefficients'] = sg_coeffs

    result['coefficients']['Slope'] = _get_float(el, 'Slope', 1.0)
    result['coefficients']['Offset'] = _get_float(el, 'Offset', 0.0)
    return result


def _parse_oxygen_sbe43(el: ET.Element) -> dict:
    """
    SBE43 dissolved oxygen sensor.
    Uses 2007 equation (CalibrationCoefficients equation="1") when
    Use2007Equation == 1, which is standard for modern calibrations.
    """
    use_2007 = _get_int(el, 'Use2007Equation', 1)
    result = _common_fields(el)
    result['use_2007_equation'] = bool(use_2007)

    eq_target = '1' if use_2007 else '0'
    coeffs = {}
    for coeff_el in el.findall('CalibrationCoefficients'):
        if coeff_el.get('equation') == eq_target:
            if use_2007:
                coeffs = {k: _get_float(coeff_el, k) for k in
                          ['Soc', 'offset', 'A', 'B', 'C',
                           'D0', 'D1', 'D2', 'E',
                           'Tau20', 'H1', 'H2', 'H3']}
            else:
                coeffs = {k: _get_float(coeff_el, k) for k in
                          ['Boc', 'Soc', 'offset', 'Pcor', 'Tcor', 'Tau']}
            break

    coeffs['Slope'] = _get_float(el, 'Slope', 1.0)
    coeffs['Offset'] = _get_float(el, 'Offset', 0.0)
    result['coefficients'] = coeffs
    return result


def _parse_eco_fluorometer(el: ET.Element) -> dict:
    """WET Labs ECO-AFL/FL chlorophyll fluorometer."""
    result = _common_fields(el)
    result['coefficients'] = {
        'ScaleFactor': _get_float(el, 'ScaleFactor'),
        'Vblank':      _get_float(el, 'Vblank'),
    }
    return result


def _parse_wetstar_fluorometer(el: ET.Element) -> dict:
    """WET Labs WETstar fluorometer."""
    result = _common_fields(el)
    result['coefficients'] = {
        'ScaleFactor': _get_float(el, 'ScaleFactor'),
        'Vblank':      _get_float(el, 'Vblank'),
    }
    return result


def _parse_cdom(el: ET.Element) -> dict:
    """WET Labs ECO CDOM fluorometer."""
    result = _common_fields(el)
    result['coefficients'] = {
        'ScaleFactor': _get_float(el, 'ScaleFactor'),
        'Vblank':      _get_float(el, 'Vblank'),
    }
    return result


def _parse_seapoint_fluorometer(el: ET.Element) -> dict:
    """Seapoint fluorometer."""
    result = _common_fields(el)
    result['coefficients'] = {
        'GainSetting': _get_int(el, 'GainSetting'),
        'Offset':      _get_float(el, 'Offset', 0.0),
    }
    return result


def _parse_cstar(el: ET.Element) -> dict:
    """WET Labs C-Star transmissometer."""
    result = _common_fields(el)
    result['coefficients'] = {
        'M':          _get_float(el, 'M'),
        'B':          _get_float(el, 'B'),
        'PathLength': _get_float(el, 'PathLength'),
    }
    return result


def _parse_altimeter(el: ET.Element) -> dict:
    """Benthos / Teledyne altimeter."""
    result = _common_fields(el)
    result['coefficients'] = {
        'ScaleFactor': _get_float(el, 'ScaleFactor'),
        'Offset':      _get_float(el, 'Offset', 0.0),
    }
    return result


def _parse_par(el: ET.Element) -> dict:
    """Biospherical / Licor / Chelsea PAR sensor."""
    result = _common_fields(el)
    result['coefficients'] = {
        'M':                  _get_float(el, 'M'),
        'B':                  _get_float(el, 'B'),
        'CalibrationConstant': _get_float(el, 'CalibrationConstant'),
        'Multiplier':         _get_float(el, 'Multiplier'),
        'Offset':             _get_float(el, 'Offset', 0.0),
    }
    return result


def _parse_spar(el: ET.Element) -> dict:
    """Surface PAR sensor."""
    result = _common_fields(el)
    result['coefficients'] = {
        'ConversionFactor':  _get_float(el, 'ConversionFactor'),
        'RatioMultiplier':   _get_float(el, 'RatioMultiplier', 1.0),
        # ConversionUnits is optional (absent in some files)
        'ConversionUnits':   _get_int(el, 'ConversionUnits'),
    }
    return result


def _parse_not_in_use(el: ET.Element) -> dict:
    """Placeholder for an unused voltage channel."""
    return {
        'serial_number':    None,
        'calibration_date': None,
        'output_type':      _get_int(el, 'OutputType'),
        'free':             bool(_get_int(el, 'Free', 0)),
        'coefficients':     {},
    }


def _parse_generic(el: ET.Element) -> dict:
    """
    Fallback for unrecognised sensor types.
    Extracts serial number, calibration date, and all numeric leaf values.
    """
    result = _common_fields(el)
    coefficients = {}
    for child in el:
        if len(child) == 0 and child.text:  # leaf node
            try:
                coefficients[child.tag] = float(child.text)
            except (ValueError, TypeError):
                pass
    result['coefficients'] = coefficients
    return result


# ---------------------------------------------------------------------------
# Convenience helpers for downstream use
# ---------------------------------------------------------------------------

def get_sensors_by_type(config: dict, sensor_type: str) -> list[dict]:
    """Return all sensors of a given type, e.g. 'TemperatureSensor'."""
    return [s for s in config['sensors'] if s['type'] == sensor_type]


def get_active_sensors(config: dict) -> list[dict]:
    """Return only sensors that are in use (exclude NotInUse)."""
    return [s for s in config['sensors'] if s['in_use']]


def summarise(config: dict) -> str:
    """
    Return a human-readable summary of the parsed configuration.
    Useful for quick inspection in a notebook.
    """
    instr = config['instrument']
    lines = [
        f"Instrument : {instr.get('name', 'Unknown')}",
        f"Device type: {instr.get('device_type', 'N/A')}",
        f"Firmware   : {instr.get('firmware_version', 'N/A')}",
        f"Freq ch suppressed  : {instr['frequency_channels_suppressed']}",
        f"Volt words suppressed: {instr['voltage_words_suppressed']}",
        f"NMEA position : {instr['nmea_position_added']}",
        f"Surface PAR   : {instr['surface_par_added']}",
        f"Scan time     : {instr['scan_time_added']}",
        "",
        f"{'idx':>3}  {'SensorID':>8}  {'Type':<45}  {'Serial':<20}  {'Cal date'}",
        "-" * 100,
    ]
    for s in config['sensors']:
        status = "" if s['in_use'] else "  [not in use]"
        lines.append(
            f"{s['index']:>3}  {s['sensor_id']:>8}  {s['type']:<45}  "
            f"{str(s.get('serial_number') or ''):20}  "
            f"{str(s.get('calibration_date') or '')}{status}"
        )
    return "\n".join(lines)