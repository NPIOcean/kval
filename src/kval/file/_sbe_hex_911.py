"""
kval.file._sbe_hex_911

SBE 911plus/917plus hex scan parser.

Builds a scan layout from the hex file header + xmlcon configuration,
validates it against the stated bytes-per-scan, then unpacks all scans
into numpy arrays of raw counts/frequencies and decoded auxiliary fields.

The only public function is:

    parse_911_scans(hex_path, xmlcon_config, hex_header) -> dict[str, np.ndarray]

The returned dict is consumed by sbe_hex.py to build the xr.Dataset.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np

from kval.file._sbe_hex_common import (
    HexField,
    SBEHexLayoutError,
    SBEUnknownSensorWarning,
    freq_from_3bytes,
    voltages_from_3bytes,
    tempcomp_from_3chars,
    nmea_location_from_7bytes,
    system_time_from_8chars,
    par_from_3chars,
)


# ---------------------------------------------------------------------------
# Voltage channel to xmlcon sensor index mapping
# SBE911 always has 8 voltage channels (4 pairs), mapped to xmlcon indices 5-14
# The xmlcon sensor array has:
#   index 0: T1, 1: C1, 2: P, 3: T2, 4: C2, 5+: voltage channels
# ---------------------------------------------------------------------------
_VOLT_XMLCON_START = 5   # first voltage channel is always at xmlcon sensor index 5


# ---------------------------------------------------------------------------
# Scan layout builder
# ---------------------------------------------------------------------------

def build_911_layout(
    hex_header: dict,
    xmlcon_config: dict,
    hex_path: Path,
    xmlcon_path: Path,
) -> list[HexField]:
    """
    Build the ordered list of HexFields that describes one SBE911 scan.

    The layout is derived entirely from the hex file header flags and the
    xmlcon sensor configuration.  At the end we verify that the total
    number of hex characters matches bytes_per_scan × 2.  A mismatch
    raises SBEHexLayoutError with a detailed, human-readable explanation.

    Parameters
    ----------
    hex_header : dict
        Parsed hex file header (from parse_hex_header).
    xmlcon_config : dict
        Parsed xmlcon configuration (from parse_xmlcon).
    hex_path, xmlcon_path : Path
        Used only for error messages.

    Returns
    -------
    list[HexField]
        Ordered list of fields, one entry per logical field in the scan.
        Fields with store=False are consumed but not written to the Dataset.
    """
    layout: list[HexField] = []
    sensors = xmlcon_config["sensors"]
    freq_suppressed = hex_header["freq_suppressed"]

    # volt_suppressed: prefer the hex header value; fall back to xmlcon.
    # The hex header often omits this line (treats as 0), while the xmlcon
    # always carries the correct value.
    volt_suppressed = hex_header["volt_suppressed"]
    if volt_suppressed == 0:
        xmlcon_vws = xmlcon_config["instrument"].get("voltage_words_suppressed") or 0
        volt_suppressed = xmlcon_vws

    # ------------------------------------------------------------------
    # Frequency channels: T1, C1, P, [T2], [C2]
    # Each is 3 bytes = 6 hex chars.
    # Secondary channels are always physically present in the stream
    # unless FrequencyChannelsSuppressed > 0.
    # ------------------------------------------------------------------

    # Primary temperature (always present)
    layout.append(HexField(
        name="temperature_primary_raw",
        description="primary temperature (Hz)",
        n_hex_chars=6,
        convert=freq_from_3bytes,
        units="Hz",
    ))

    # Primary conductivity (always present)
    # Stored as freq_from_3bytes (= actual_freq / 2, due to SBE11 encoding).
    # _sbe_convert.py multiplies by 2 before calling convert_conductivity.
    layout.append(HexField(
        name="conductivity_primary_raw",
        description="primary conductivity (Hz, = actual freq / 2)",
        n_hex_chars=6,
        convert=freq_from_3bytes,
        units="Hz",
    ))

    # Pressure (always present)
    layout.append(HexField(
        name="pressure_raw",
        description="pressure (Hz)",
        n_hex_chars=6,
        convert=freq_from_3bytes,
        units="Hz",
    ))

    # Secondary temperature: present unless FreqSuppressed >= 2
    has_secondary_freq = (freq_suppressed <= 1)
    if has_secondary_freq:
        # Check if secondary T is actually a real sensor in xmlcon
        has_secondary_t = len([s for s in sensors
                                if s["type"] == "TemperatureSensor"]) >= 2
        layout.append(HexField(
            name="temperature_secondary_raw",
            description="secondary temperature (Hz)",
            n_hex_chars=6,
            convert=freq_from_3bytes if has_secondary_t else None,
            units="Hz",
            store=has_secondary_t,
        ))

    # Secondary conductivity: present unless FreqSuppressed >= 1
    has_secondary_c_freq = (freq_suppressed == 0)
    if has_secondary_c_freq:
        has_secondary_c = len([s for s in sensors
                                if s["type"] == "ConductivitySensor"]) >= 2
        layout.append(HexField(
            name="conductivity_secondary_raw",
            description="secondary conductivity (Hz, = actual freq / 2)",
            n_hex_chars=6,
            convert=freq_from_3bytes if has_secondary_c else None,
            units="Hz",
            store=has_secondary_c,
        ))

    # ------------------------------------------------------------------
    # Voltage channels: 0-7, stored as 4 pairs of 3 bytes each.
    # VoltageWordsSuppressed removes the LAST N pairs from the stream:
    #   volt_suppressed=0 -> all 4 pairs present  (pairs 0,1,2,3)
    #   volt_suppressed=1 -> 3 pairs present       (pairs 0,1,2)
    #   volt_suppressed=2 -> 2 pairs present       (pairs 0,1)
    #   volt_suppressed=3 -> 1 pair  present       (pair  0)
    #   volt_suppressed=4 -> no voltage pairs
    # ------------------------------------------------------------------
    n_pairs = max(0, min(4, 4 - volt_suppressed))

    for pair_idx in range(4):
        volt_a_idx = pair_idx * 2       # e.g. 0, 2, 4, 6
        volt_b_idx = volt_a_idx + 1     # e.g. 1, 3, 5, 7
        xmlcon_a = _VOLT_XMLCON_START + volt_a_idx
        xmlcon_b = _VOLT_XMLCON_START + volt_b_idx

        if pair_idx >= n_pairs:
            # This pair is suppressed - not in the stream at all
            continue

        # Get the sensor descriptors for both channels in this pair
        sensor_a = sensors[xmlcon_a] if xmlcon_a < len(sensors) else None
        sensor_b = sensors[xmlcon_b] if xmlcon_b < len(sensors) else None

        name_a, store_a, desc_a = _volt_field_info(sensor_a, volt_a_idx, xmlcon_a)
        name_b, store_b, desc_b = _volt_field_info(sensor_b, volt_b_idx, xmlcon_b)

        # The pair is encoded as a single 3-byte field using voltages_from_3bytes
        # We store it as two separate outputs using a split converter
        layout.append(_make_volt_pair_field(
            pair_idx, volt_a_idx, volt_b_idx,
            name_a, name_b, desc_a, desc_b,
            store_a, store_b,
        ))

    # ------------------------------------------------------------------
    # Surface PAR (if enabled): 3 unused chars + 3 PAR chars = 6 total
    # The A/D offset from the header is subtracted before dividing by 819.
    # ------------------------------------------------------------------
    if hex_header["surface_par_added"]:
        ad_offset = hex_header.get("ad_offset", 0)
        layout.append(HexField(
            name="_par_unused",
            description="surface PAR padding (unused)",
            n_hex_chars=3,
            convert=None,
            store=False,
        ))
        layout.append(HexField(
            name="surface_par_raw",
            description="surface PAR (raw counts - AD_offset) / 819",
            n_hex_chars=3,
            convert=lambda h, _off=ad_offset: (int(h, 16) - _off) / 819,
            units="counts/819",
        ))

    # ------------------------------------------------------------------
    # NMEA position (if enabled): 7 bytes = 14 hex chars
    # Contains lat + lon + sign byte, decoded together
    # ------------------------------------------------------------------
    if hex_header["nmea_position_added"]:
        layout.append(HexField(
            name="_nmea_loc",
            description="NMEA position (lat/lon packed, decoded separately)",
            n_hex_chars=14,
            convert=nmea_location_from_7bytes,
            units="degrees",
        ))

    # ------------------------------------------------------------------
    # NMEA time (some firmware versions): 4 bytes = 8 hex chars
    # Present when NMEA position is appended AND bytes_per_scan is 4
    # bytes larger than the layout without it. Detected by comparing
    # running hex char count to what the fixed tail fields will consume.
    # ------------------------------------------------------------------
    _tail_hex = 3 + 1 + 2 + (8 if hex_header["scan_time_added"] else 0)  # hex chars
    _current_hex = sum(f.n_hex_chars for f in layout)
    _remaining_hex = hex_header["bytes_per_scan"] * 2 - _current_hex - _tail_hex
    if _remaining_hex == 8 and hex_header["nmea_position_added"]:
        layout.append(HexField(
            name="_nmea_time",
            description="NMEA UTC time word (packed, not decoded)",
            n_hex_chars=8,
            convert=None,
            store=False,
        ))

    # ------------------------------------------------------------------
    # Temperature compensation: 3 hex chars (12-bit word)
    # Used in Digiquartz pressure conversion — always present in 911 scans.
    # ------------------------------------------------------------------
    layout.append(HexField(
        name="pressure_temp_comp_raw",
        description="pressure temperature compensation (12-bit count)",
        n_hex_chars=3,
        convert=tempcomp_from_3chars,
        units="counts",
    ))

    # ------------------------------------------------------------------
    # Status byte: 1 hex char (4-bit nibble)
    # Pump status, bottom contact, etc.  Stored as raw integer.
    # ------------------------------------------------------------------
    layout.append(HexField(
        name="status",
        description="status nibble (pump/contact flags)",
        n_hex_chars=1,
        convert=lambda h: int(h, 16),
        units="",
    ))

    # ------------------------------------------------------------------
    # Data integrity: 2 hex chars (1 byte)
    # Checksum / integrity flag.  Stored as raw integer.
    # ------------------------------------------------------------------
    layout.append(HexField(
        name="data_integrity",
        description="data integrity byte",
        n_hex_chars=2,
        convert=lambda h: int(h, 16),
        units="",
    ))

    # ------------------------------------------------------------------
    # System time: 4 bytes = 8 hex chars, little-endian Unix timestamp
    # Appended when 'Append System Time to Every Scan' is in header.
    # ------------------------------------------------------------------
    if hex_header["scan_time_added"]:
        layout.append(HexField(
            name="scan_time",
            description="per-scan UTC timestamp (little-endian Unix)",
            n_hex_chars=8,
            convert=system_time_from_8chars,
            units="datetime",
        ))

    # ------------------------------------------------------------------
    # Validate total against header
    # ------------------------------------------------------------------
    computed_hex_chars = sum(f.n_hex_chars for f in layout)
    expected_hex_chars = hex_header["bytes_per_scan"] * 2
    if computed_hex_chars != expected_hex_chars:
        raise SBEHexLayoutError(
            hex_path=hex_path,
            xmlcon_path=xmlcon_path,
            expected_bytes=hex_header["bytes_per_scan"],
            computed_bytes=computed_hex_chars // 2,
            layout=[f for f in layout if f.store or f.name.startswith("_")],
        )

    return layout


def _volt_field_info(
    sensor: Optional[dict],
    volt_idx: int,
    xmlcon_idx: int,
) -> tuple[str, bool, str]:
    """
    Return (field_name, store, description) for a single voltage channel.
    """
    if sensor is None or not sensor.get("in_use", False):
        return (
            f"_volt_{volt_idx}_unused",
            False,
            f"voltage channel {volt_idx} (not in use, xmlcon index {xmlcon_idx})",
        )
    sensor_type = sensor.get("type", "Unknown")
    serial = sensor.get("serial_number") or ""
    serial_str = f" SN:{serial}" if serial else ""
    return (
        f"volt_{volt_idx}_{sensor_type.lower()[:12]}",
        True,
        f"voltage channel {volt_idx}: {sensor_type}{serial_str}",
    )


class _VoltPairConverter:
    """
    Callable that extracts one voltage (A or B) from a 3-byte pair word.
    Needed because each pair encodes two channels but we store them separately.
    """
    def __init__(self, which: str):
        assert which in ("a", "b")
        self.which = which

    def __call__(self, hex6: str) -> float:
        va, vb = voltages_from_3bytes(hex6)
        return va if self.which == "a" else vb


def _make_volt_pair_field(
    pair_idx: int,
    volt_a_idx: int, volt_b_idx: int,
    name_a: str, name_b: str,
    desc_a: str, desc_b: str,
    store_a: bool, store_b: bool,
) -> HexField:
    """
    Create a HexField for one 3-byte voltage pair.

    Because a pair encodes two channels, we use a special wrapper that
    stores the pair as a single 6-char field but tags it with both
    channel names for the scan unpacker to handle.
    """
    return _VoltPairField(
        pair_idx=pair_idx,
        volt_a_idx=volt_a_idx, volt_b_idx=volt_b_idx,
        name_a=name_a, name_b=name_b,
        desc_a=desc_a, desc_b=desc_b,
        store_a=store_a, store_b=store_b,
    )


class _VoltPairField:
    """
    A HexField-like descriptor for a 3-byte voltage pair.
    Stores metadata for both channels; the scan unpacker handles splitting.
    """
    def __init__(
        self,
        pair_idx: int,
        volt_a_idx: int, volt_b_idx: int,
        name_a: str, name_b: str,
        desc_a: str, desc_b: str,
        store_a: bool, store_b: bool,
    ):
        self.pair_idx = pair_idx
        self.n_hex_chars = 6
        self.name = f"_volt_pair_{pair_idx}"       # internal name
        self.description = f"voltage pair {pair_idx} (channels {volt_a_idx}+{volt_b_idx})"
        self.units = "V"
        self.store = store_a or store_b            # True if either channel is stored
        # Sub-channel info
        self.name_a = name_a
        self.name_b = name_b
        self.desc_a = desc_a
        self.desc_b = desc_b
        self.store_a = store_a
        self.store_b = store_b

    @property
    def n_bytes(self) -> float:
        return self.n_hex_chars / 2

    def is_volt_pair(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Scan unpacker
# ---------------------------------------------------------------------------

def parse_911_scans(
    hex_path: Path,
    xmlcon_config: dict,
    hex_header: dict,
    xmlcon_path: Path,
) -> dict:
    """
    Parse all scan lines from an SBE911 hex file.

    Parameters
    ----------
    hex_path : Path
        Path to the .hex file.
    xmlcon_config : dict
        Parsed xmlcon (from parse_xmlcon).
    hex_header : dict
        Parsed hex header (from parse_hex_header).
    xmlcon_path : Path
        Path to the xmlcon file (used only in error messages).

    Returns
    -------
    dict
        Keys are field names (strings), values are numpy arrays of
        length n_scans.  The special key '_scan_times' contains a list
        of datetime objects (or None if no per-scan time was in the stream).
        The key '_nmea_lat' and '_nmea_lon' are included if NMEA was present.
    """
    layout = build_911_layout(hex_header, xmlcon_config, hex_path, xmlcon_path)

    n_scans = hex_header["n_scans"]
    expected_hex_len = hex_header["bytes_per_scan"] * 2

    # Pre-allocate output arrays
    arrays: dict[str, np.ndarray] = {}
    scan_times: list[Optional[datetime]] = [None] * n_scans
    nmea_lats = np.full(n_scans, np.nan)
    nmea_lons = np.full(n_scans, np.nan)
    has_scan_time = hex_header["scan_time_added"]
    has_nmea = hex_header["nmea_position_added"]

    for field in layout:
        if isinstance(field, _VoltPairField):
            if field.store_a and not field.name_a.startswith("_"):
                arrays[field.name_a] = np.full(n_scans, np.nan)
            if field.store_b and not field.name_b.startswith("_"):
                arrays[field.name_b] = np.full(n_scans, np.nan)
        elif field.store and not field.name.startswith("_"):
            if field.units == "datetime":
                pass  # scan_times handled separately
            else:
                arrays[field.name] = np.full(n_scans, np.nan)

    # Read and parse each scan line
    scan_idx = 0
    with open(hex_path, "r", errors="replace") as fh:
        for line_no, raw_line in enumerate(fh):
            if line_no < hex_header["data_start_line"]:
                continue
            line = raw_line.strip()
            if not line or line.startswith("*"):
                continue

            if len(line) != expected_hex_len:
                warnings.warn(
                    f"\n\nSkipping scan at line {line_no + 1} of '{hex_path.name}':\n"
                    f"  Expected {expected_hex_len} hex characters "
                    f"({hex_header['bytes_per_scan']} bytes), "
                    f"got {len(line)} characters.\n"
                    f"  This scan will be filled with NaN.\n",
                    stacklevel=2,
                )
                scan_idx += 1
                if scan_idx >= n_scans:
                    break
                continue

            # Walk the layout and extract each field
            pos = 0
            for field in layout:
                segment = line[pos: pos + field.n_hex_chars]
                pos += field.n_hex_chars

                if isinstance(field, _VoltPairField):
                    va, vb = voltages_from_3bytes(segment)
                    if field.store_a and not field.name_a.startswith("_"):
                        arrays[field.name_a][scan_idx] = va
                    if field.store_b and not field.name_b.startswith("_"):
                        arrays[field.name_b][scan_idx] = vb

                elif field.name == "_nmea_loc":
                    lat, lon = nmea_location_from_7bytes(segment)
                    nmea_lats[scan_idx] = lat
                    nmea_lons[scan_idx] = lon

                elif field.name == "scan_time":
                    scan_times[scan_idx] = system_time_from_8chars(segment)

                elif field.name == "_par_unused":
                    pass  # discard padding

                elif field.store and not field.name.startswith("_") and field.convert is not None:
                    arrays[field.name][scan_idx] = field.convert(segment)

            scan_idx += 1
            if scan_idx >= n_scans:
                break

    result = dict(arrays)
    if has_scan_time:
        result["_scan_times"] = scan_times
    if has_nmea:
        result["_nmea_lat"] = nmea_lats
        result["_nmea_lon"] = nmea_lons

    return result


# ---------------------------------------------------------------------------
# Sensor metadata extraction (for xarray variable attributes)
# ---------------------------------------------------------------------------

def get_911_sensor_attrs(xmlcon_config: dict) -> dict[str, dict]:
    """
    Build a dict mapping raw field names to their sensor metadata attributes.

    Returns
    -------
    dict
        Keys match the field names produced by parse_911_scans.
        Values are dicts suitable for use as xr.DataArray.attrs.
    """
    sensors = xmlcon_config["sensors"]
    attrs = {}

    def _base_attrs(sensor: dict, extra: dict = None) -> dict:
        a = {
            "sensor_type":      sensor.get("type", ""),
            "serial_number":    str(sensor.get("serial_number") or ""),
            "calibration_date": str(sensor.get("calibration_date") or ""),
            "sensor_index":     sensor.get("index", -1),
        }
        if extra:
            a.update(extra)
        return a

    # Primary T/C/P (always at indices 0, 1, 2)
    if len(sensors) > 0:
        attrs["temperature_primary_raw"] = _base_attrs(sensors[0], {"units": "Hz"})
    if len(sensors) > 1:
        attrs["conductivity_primary_raw"] = _base_attrs(sensors[1], {"units": "Hz"})
    if len(sensors) > 2:
        attrs["pressure_raw"] = _base_attrs(sensors[2], {"units": "Hz"})

    # Secondary T/C (indices 3, 4 if present)
    if len(sensors) > 3 and sensors[3].get("type") == "TemperatureSensor":
        attrs["temperature_secondary_raw"] = _base_attrs(sensors[3], {"units": "Hz"})
    if len(sensors) > 4 and sensors[4].get("type") == "ConductivitySensor":
        attrs["conductivity_secondary_raw"] = _base_attrs(sensors[4], {"units": "Hz"})

    # Voltage channels (indices 5+)
    for volt_idx in range(8):
        xmlcon_idx = _VOLT_XMLCON_START + volt_idx
        if xmlcon_idx >= len(sensors):
            break
        sensor = sensors[xmlcon_idx]
        if not sensor.get("in_use", False):
            continue
        field_name = f"volt_{volt_idx}_{sensor.get('type', 'unknown').lower()[:12]}"
        attrs[field_name] = _base_attrs(sensor, {"units": "V", "volt_channel": volt_idx})

    return attrs