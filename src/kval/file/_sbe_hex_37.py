"""
kval.file._sbe_hex_37

SBE 37 family hex scan parser (SBE37SM, SBE37SMP, SBE37SMP-ODO, etc.).

The SBE37 binary format is fixed-width and much simpler than the SBE911:
no secondary T/C, no voltage pairs, a fixed timestamp per scan.

Layout (all big-endian, fields always in this order):
    T_raw        6 hex chars  (24-bit integer count)
    C_raw        6 hex chars  (24-bit integer / 256)
    [SBE63_phase 6 hex chars] (if OxygenSensor SBE63 present)
    [SBE63_T     6 hex chars] (if OxygenSensor SBE63 present)
    [P_raw       6 hex chars] (if PressureSensor present)
    [P_Tcomp     4 hex chars] (if PressureSensor present)
    time         8 hex chars  (seconds since 2000-01-01 UTC)

SampleLength from header is the ground truth for layout validation.

Public function:
    parse_37_scans(hex_path, xmlcon_config, hex_header, xmlcon_path)
        -> dict[str, np.ndarray]
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
    sbe37_time_from_8chars,
)


# ---------------------------------------------------------------------------
# Scan layout builder
# ---------------------------------------------------------------------------

def build_37_layout(
    hex_header: dict,
    xmlcon_config: dict,
    hex_path: Path,
    xmlcon_path: Path,
) -> list[HexField]:
    """
    Build the ordered list of HexFields for one SBE37 scan.

    The layout is determined by which sensors are present in the xmlcon.
    Validated against bytes_per_scan from the hex header.

    Raises SBEHexLayoutError if computed byte count ≠ header value.
    """
    sensors = xmlcon_config["sensors"]
    sensor_types = [s.get("type") for s in sensors]

    layout: list[HexField] = []

    # Temperature — always present
    layout.append(HexField(
        name="temperature_raw",
        description="temperature (raw count)",
        n_hex_chars=6,
        convert=lambda h: int(h, 16),
        units="counts",
    ))

    # Conductivity — always present
    # Stored as raw A/D integer count (no /256 — that was wrong).
    # seabirdscientific convert_conductivity expects raw counts and
    # does the /1000 conversion internally.
    layout.append(HexField(
        name="conductivity_raw",
        description="conductivity (A/D counts)",
        n_hex_chars=6,
        convert=lambda h: int(h, 16),
        units="counts",
    ))

    # SBE63 oxygen (phase + temperature) — present if OxygenSensor in xmlcon
    # Note: SBE37 uses SBE63 optical oxygen, not SBE43.
    has_sbe63 = "OxygenSensor" in sensor_types
    if has_sbe63:
        layout.append(HexField(
            name="sbe63_phase_raw",
            description="SBE63 oxygen phase (raw count / 100000 − 10)",
            n_hex_chars=6,
            convert=lambda h: int(h, 16) / 100000 - 10,
            units="μsec",
        ))
        layout.append(HexField(
            name="sbe63_temperature_raw",
            description="SBE63 oxygen thermistor (raw count / 1000000 − 1)",
            n_hex_chars=6,
            convert=lambda h: int(h, 16) / 1000000 - 1,
            units="V",
        ))

    # Pressure + temperature compensation — present if PressureSensor in xmlcon
    has_pressure = "PressureSensor" in sensor_types
    if has_pressure:
        layout.append(HexField(
            name="pressure_raw",
            description="pressure (raw count)",
            n_hex_chars=6,
            convert=lambda h: int(h, 16),
            units="counts",
        ))
        layout.append(HexField(
            name="pressure_temp_comp_raw",
            description="pressure temperature compensation (raw count)",
            n_hex_chars=4,
            convert=lambda h: int(h, 16),
            units="counts",
        ))

    # Timestamp — always last, always present
    layout.append(HexField(
        name="scan_time",
        description="sample timestamp (seconds since 2000-01-01 UTC)",
        n_hex_chars=8,
        convert=sbe37_time_from_8chars,
        units="datetime",
    ))

    # Validate
    computed_hex_chars = sum(f.n_hex_chars for f in layout)
    expected_hex_chars = hex_header["bytes_per_scan"] * 2
    if computed_hex_chars != expected_hex_chars:
        raise SBEHexLayoutError(
            hex_path=hex_path,
            xmlcon_path=xmlcon_path,
            expected_bytes=hex_header["bytes_per_scan"],
            computed_bytes=computed_hex_chars // 2,
            layout=layout,
        )

    return layout


# ---------------------------------------------------------------------------
# Scan unpacker
# ---------------------------------------------------------------------------

def parse_37_scans(
    hex_path: Path,
    xmlcon_config: dict,
    hex_header: dict,
    xmlcon_path: Path,
) -> dict:
    """
    Parse all scan lines from an SBE37 hex file.

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
        Keys are field names, values are numpy arrays of length n_scans.
        The special key '_scan_times' contains a list of datetime objects.
    """
    layout = build_37_layout(hex_header, xmlcon_config, hex_path, xmlcon_path)

    n_scans = hex_header["n_scans"]
    expected_hex_len = hex_header["bytes_per_scan"] * 2

    # Pre-allocate
    arrays: dict[str, np.ndarray] = {}
    scan_times: list[Optional[datetime]] = [None] * n_scans

    for field in layout:
        if field.name == "scan_time":
            pass  # handled separately
        elif field.store:
            arrays[field.name] = np.full(n_scans, np.nan)

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

            pos = 0
            for field in layout:
                segment = line[pos: pos + field.n_hex_chars]
                pos += field.n_hex_chars

                if field.name == "scan_time":
                    scan_times[scan_idx] = sbe37_time_from_8chars(segment)
                elif field.store and field.convert is not None:
                    arrays[field.name][scan_idx] = field.convert(segment)

            scan_idx += 1
            if scan_idx >= n_scans:
                break

    result = dict(arrays)
    result["_scan_times"] = scan_times
    return result


# ---------------------------------------------------------------------------
# Sensor metadata extraction
# ---------------------------------------------------------------------------

def get_37_sensor_attrs(xmlcon_config: dict) -> dict[str, dict]:
    """
    Build a dict mapping raw field names to their sensor metadata attributes.
    """
    sensors = xmlcon_config["sensors"]
    attrs = {}

    def _find(sensor_type: str) -> Optional[dict]:
        for s in sensors:
            if s.get("type") == sensor_type:
                return s
        return None

    def _base(sensor: dict, extra: dict = None) -> dict:
        a = {
            "sensor_type":      sensor.get("type", ""),
            "serial_number":    str(sensor.get("serial_number") or ""),
            "calibration_date": str(sensor.get("calibration_date") or ""),
            "sensor_index":     sensor.get("index", -1),
        }
        if extra:
            a.update(extra)
        return a

    t = _find("TemperatureSensor")
    if t:
        attrs["temperature_raw"] = _base(t, {"units": "counts"})

    c = _find("ConductivitySensor")
    if c:
        attrs["conductivity_raw"] = _base(c, {"units": "counts/256"})

    p = _find("PressureSensor")
    if p:
        attrs["pressure_raw"] = _base(p, {"units": "counts"})
        attrs["pressure_temp_comp_raw"] = _base(p, {"units": "counts",
                                                     "note": "pressure temperature compensation"})

    o = _find("OxygenSensor")
    if o:
        attrs["sbe63_phase_raw"] = _base(o, {"units": "μsec",
                                              "note": "SBE63 phase delay"})
        attrs["sbe63_temperature_raw"] = _base(o, {"units": "V",
                                                    "note": "SBE63 thermistor voltage"})

    return attrs
