"""
kval.file._sbe_hex_common

Shared infrastructure for SBE hex file parsing:
  - Human-readable error classes
  - HexField descriptor (name, n_hex_chars, conversion function)
  - Hex file header parser
  - Low-level byte conversion utilities (matching seabirdscientific conventions)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional
import re


# ---------------------------------------------------------------------------
# Errors — written for humans, not just developers
# ---------------------------------------------------------------------------

class SBEHexError(Exception):
    """Base class for all SBE hex parsing errors."""
    pass


class SBEHexFileError(SBEHexError):
    """Raised when the hex file cannot be opened or has no data."""

    def __init__(self, path: Path, detail: str):
        self.path = path
        super().__init__(
            f"\n\nCould not read hex file: '{path.name}'\n"
            f"\n  Problem: {detail}\n"
            f"\n  Check that the file path is correct and the file is not corrupted.\n"
        )


class SBEHexLayoutError(SBEHexError):
    """
    Raised when the computed scan layout does not match the header's
    stated bytes-per-scan value.  This almost always means the .xmlcon
    and .hex files are mismatched (e.g. different casts or instruments).
    """

    def __init__(
        self,
        hex_path: Path,
        xmlcon_path: Path,
        expected_bytes: int,
        computed_bytes: int,
        layout: list[HexField],
    ):
        lines = [
            f"\n\nScan layout mismatch in '{hex_path.name}'",
            f"",
            f"  The header of '{hex_path.name}' says each scan is {expected_bytes} bytes,",
            f"  but the sensor layout computed from '{xmlcon_path.name}' adds up to",
            f"  {computed_bytes} bytes.  They don't match.",
            f"",
            f"  Most likely cause: the .xmlcon file does not belong to this cast.",
            f"  Make sure you are using the configuration file that was active when",
            f"  this data was collected.",
            f"",
            f"  Computed layout ({computed_bytes} bytes):",
        ]
        width_name = max(len(f.name) for f in layout)
        for field in layout:
            lines.append(
                f"    {field.name:<{width_name}}  "
                f"{field.n_hex_chars // 2} byte{'s' if field.n_hex_chars > 2 else ' '}"
                f"  ({field.description})"
            )
        lines += [
            f"    {'─' * (width_name + 20)}",
            f"    {'total':<{width_name}}  {computed_bytes} bytes  (expected {expected_bytes})",
            f"",
        ]
        super().__init__("\n".join(lines))


class SBEHexVersionWarning(UserWarning):
    """
    Raised (as a warning, not an error) when the hex file's software
    version string is one that has not been tested.  Parsing will
    continue, but results should be checked carefully.
    """
    pass


class SBEUnknownSensorWarning(UserWarning):
    """
    Raised when a sensor type found in the .xmlcon has no dedicated
    conversion function.  The raw voltage will still be stored, but
    it will not be converted to engineering units.
    """
    pass


# ---------------------------------------------------------------------------
# HexField — describes one field in a scan line
# ---------------------------------------------------------------------------

@dataclass
class HexField:
    """
    Describes one field in a binary hex scan.

    Attributes
    ----------
    name : str
        Variable name to use in the output Dataset (e.g. 'temperature_primary').
    description : str
        Human-readable description for error messages (e.g. 'primary temperature').
    n_hex_chars : int
        Number of hex characters this field occupies in the scan string.
        Must be even for whole-byte fields; 3-char fields (12-bit SBE911
        sub-byte fields) are also valid.
    convert : callable or None
        Function that takes the raw hex string segment and returns the
        raw count / frequency value.  Set to None for fields that are
        read but not stored (e.g. secondary channels when suppressed).
    units : str
        Units of the converted value, e.g. 'Hz', 'V', 'counts'.
    store : bool
        If False the field is consumed from the scan but not written to
        the output Dataset (e.g. unused secondary channels, padding).
    """
    name: str
    description: str
    n_hex_chars: int
    convert: Optional[Callable[[str], float]]
    units: str = ""
    store: bool = True

    @property
    def n_bytes(self) -> float:
        """Bytes consumed (may be fractional for 3-char nibble fields)."""
        return self.n_hex_chars / 2


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

# Software versions that have been tested and verified.
# A warning is issued for anything not in this set.
_TESTED_VERSIONS = {
    "Seasave V 7.26.7.107",
    "Seasave V 7.26.7.121",
    "SeatermV2 2.8.0.119",
}

# Supported instrument name strings as they appear in hex file headers.
_INSTRUMENT_911 = "SBE 9"         # matches "SBE 9 Data File" (SBE911/917)
_INSTRUMENT_37  = "SBE37"         # matches "SBE37SMP-ODO-RS232 Data File"


def parse_hex_header(path: Path) -> dict:
    """
    Parse the header block of an SBE hex file (all lines starting with '*').

    Returns a dict with keys:
        instrument_family : str   — '911' or '37'
        instrument_name   : str   — full name from first header line
        software_version  : str
        bytes_per_scan    : int   — from "Number of Bytes Per Scan"
        n_voltage_words   : int   — from "Number of Voltage Words" (911 only)
        freq_suppressed   : int   — 0 for all tested files
        volt_suppressed   : int   — 0 or 1 in tested files
        surface_par_added : bool
        scan_time_added   : bool  — 'Append System Time to Every Scan'
        nmea_position_added: bool
        nmea_depth_added  : bool
        nmea_time_added   : bool
        upload_time       : datetime or None
        latitude          : float or None  — from NMEA header line
        longitude         : float or None
        user_comments     : list[str]      — '**' lines
        raw_header_lines  : list[str]
        data_start_line   : int   — line index of first data line

    Raises
    ------
    SBEHexFileError
        If the file cannot be opened, has no *END* marker, or no data
        lines after the header.
    """
    import warnings

    path = Path(path)
    if not path.exists():
        raise SBEHexFileError(path, f"File not found: {path}")

    header_lines = []
    user_comments = []
    data_start_line = None

    with open(path, "r", errors="replace") as fh:
        for line_no, raw_line in enumerate(fh):
            line = raw_line.rstrip("\r\n")
            if line.startswith("*END*"):
                data_start_line = line_no + 1
                break
            header_lines.append(line)
            if line.startswith("**"):
                user_comments.append(line[2:].strip())

    if data_start_line is None:
        raise SBEHexFileError(
            path,
            "No '*END*' marker found.  The file may be truncated or not "
            "a valid SBE hex file."
        )

    # Count actual data lines
    n_data_lines = 0
    with open(path, "r", errors="replace") as fh:
        for i, raw_line in enumerate(fh):
            if i >= data_start_line:
                stripped = raw_line.strip()
                if stripped and not stripped.startswith("*"):
                    n_data_lines += 1

    if n_data_lines == 0:
        raise SBEHexFileError(
            path,
            "The file header was parsed successfully, but there are no "
            "data lines after the '*END*' marker.  The file may be empty."
        )

    def _find(pattern: str, default=None):
        """Return first match of regex pattern in header lines."""
        for line in header_lines:
            m = re.search(pattern, line, re.IGNORECASE)
            if m:
                return m
        return default

    # Instrument family from first header line
    first_line = header_lines[0] if header_lines else ""
    if _INSTRUMENT_911 in first_line:
        instrument_family = "911"
    elif _INSTRUMENT_37 in first_line:
        instrument_family = "37"
    else:
        instrument_family = "unknown"

    # Full instrument name
    m = re.match(r"\*\s*Sea-Bird\s+(.+?)\s+Data File", first_line, re.IGNORECASE)
    instrument_name = m.group(1).strip() if m else first_line.lstrip("* ").strip()

    # Software version
    m = _find(r"Software Version\s+(.+)")
    software_version = m.group(1).strip() if m else ""
    if software_version and software_version not in _TESTED_VERSIONS:
        warnings.warn(
            f"\n\nHex file software version '{software_version}' has not been "
            f"tested with this parser.  Parsing will continue, but please check "
            f"the output for obvious errors (e.g. wildly wrong temperatures or "
            f"pressures).\n"
            f"Tested versions: {sorted(_TESTED_VERSIONS)}\n",
            SBEHexVersionWarning,
            stacklevel=4,
        )

    # Bytes per scan — critical.
    # SBE911 uses: "* Number of Bytes Per Scan = 44"
    # SBE37  uses: "*    <SampleLength>21</SampleLength>"
    m = _find(r"Number of Bytes Per Scan\s*=\s*(\d+)")
    if m is None:
        m = _find(r"<SampleLength>\s*(\d+)\s*</SampleLength>")
    if m is None:
        raise SBEHexFileError(
            path,
            "Could not find 'Number of Bytes Per Scan' (SBE911) or "
            "'<SampleLength>' (SBE37) in the header.  "
            "This value is required to parse the scan data correctly."
        )
    bytes_per_scan = int(m.group(1))

    # Voltage words (SBE911 only)
    m = _find(r"Number of Voltage Words\s*=\s*(\d+)")
    n_voltage_words = int(m.group(1)) if m else 0

    # Suppression flags — from xmlcon but also in header for crosscheck
    m = _find(r"FrequencyChannelsSuppressed\s*=\s*(\d+)")
    freq_suppressed = int(m.group(1)) if m else 0
    m = _find(r"VoltageWordsSuppressed\s*=\s*(\d+)")
    volt_suppressed = int(m.group(1)) if m else 0

    # Optional appended data flags
    surface_par_added  = any("surface par voltage added" in l.lower() for l in header_lines)
    scan_time_added    = any("append system time" in l.lower() for l in header_lines)
    nmea_position_added = any("latitude/longitude added" in l.lower() for l in header_lines)
    nmea_depth_added   = any("nmea depth" in l.lower() for l in header_lines)
    nmea_time_added    = any("nmea time" in l.lower() for l in header_lines)

    # Upload time
    m = _find(r"System UpLoad Time\s*=\s*(.+)")
    upload_time = None
    if m:
        try:
            upload_time = datetime.strptime(m.group(1).strip(), "%b %d %Y %H:%M:%S")
        except ValueError:
            pass

    # NMEA position from header (single position, may not be present)
    latitude = longitude = None
    m_lat = _find(r"NMEA Latitude\s*=\s*(\d+)\s+(\d+\.\d+)\s+([NS])")
    if m_lat:
        lat = int(m_lat.group(1)) + float(m_lat.group(2)) / 60.0
        latitude = lat if m_lat.group(3).upper() == "N" else -lat
    m_lon = _find(r"NMEA Longitude\s*=\s*(\d+)\s+(\d+\.\d+)\s+([EW])")
    if m_lon:
        lon = int(m_lon.group(1)) + float(m_lon.group(2)) / 60.0
        longitude = lon if m_lon.group(3).upper() == "E" else -lon

    return {
        "instrument_family":  instrument_family,
        "instrument_name":    instrument_name,
        "software_version":   software_version,
        "bytes_per_scan":     bytes_per_scan,
        "n_voltage_words":    n_voltage_words,
        "freq_suppressed":    freq_suppressed,
        "volt_suppressed":    volt_suppressed,
        "surface_par_added":  surface_par_added,
        "scan_time_added":    scan_time_added,
        "nmea_position_added": nmea_position_added,
        "nmea_depth_added":   nmea_depth_added,
        "nmea_time_added":    nmea_time_added,
        "upload_time":        upload_time,
        "latitude":           latitude,
        "longitude":          longitude,
        "user_comments":      user_comments,
        "raw_header_lines":   header_lines,
        "data_start_line":    data_start_line,
        "n_scans":            n_data_lines,
    }


# ---------------------------------------------------------------------------
# Low-level byte conversion utilities
# Matches seabirdscientific.instrument_data conventions exactly.
# ---------------------------------------------------------------------------

def freq_from_3bytes(hex3: str) -> float:
    """
    Convert a 3-byte (6-char) SBE911 frequency word to Hz.

    Formula from SBE11plus V2 user manual:
        frequency = byte0 * 256 + byte1 + byte2 / 256

    Parameters
    ----------
    hex3 : str
        6-character hex string (e.g. '0D6F73').

    Returns
    -------
    float
        Frequency in Hz (typically 2000–7000 Hz for T/C,
        ~25000–80000 Hz for Digiquartz pressure).
    """
    b0 = int(hex3[0:2], 16)
    b1 = int(hex3[2:4], 16)
    b2 = int(hex3[4:6], 16)
    return b0 * 256 + b1 + b2 / 256


def voltages_from_3bytes(hex6: str) -> tuple[float, float]:
    """
    Convert a 3-byte (6-char) SBE911 voltage pair word to two voltages.

    Each voltage is 12 bits; adjacent channels share a byte.
    Voltage = 5 * (1 - raw / 4095)  [0–5 V range]

    Parameters
    ----------
    hex6 : str
        6-character hex string encoding two 12-bit voltage channels
        (e.g. '8F5039').

    Returns
    -------
    tuple[float, float]
        (voltage_a, voltage_b), each in the range 0–5 V.
    """
    raw_a = int(hex6[0:3], 16)
    raw_b = int(hex6[3:6], 16)
    return 5.0 * (1 - raw_a / 4095), 5.0 * (1 - raw_b / 4095)


def tempcomp_from_3chars(hex3: str) -> int:
    """
    Convert the SBE911 3-char (12-bit) temperature compensation word.

    Returns the raw integer count (used in Digiquartz pressure conversion).
    """
    return int(hex3, 16)


def nmea_location_from_7bytes(hex14: str) -> tuple[float, float]:
    """
    Decode a 7-byte (14-char) SBE911 NMEA location block.

    The block encodes latitude, longitude, and sign bits:
        bytes 0-2  : latitude  (unsigned) × 50000
        bytes 3-5  : longitude (unsigned) × 50000
        byte  6    : bit 0 = lat sign (0=N, 1=S)
                     bit 1 = lon sign (0=E, 1=W)

    Returns
    -------
    tuple[float, float]
        (latitude, longitude) in decimal degrees, with sign applied.
    """
    b = [int(hex14[i:i+2], 16) for i in range(0, 14, 2)]
    sign_byte = format(b[6], "08b")
    lat_sign = 1 if sign_byte[0] == "0" else -1
    lon_sign = 1 if sign_byte[1] == "0" else -1
    lat = lat_sign * (b[0] * 65536 + b[1] * 256 + b[2]) / 50000
    lon = lon_sign * (b[3] * 65536 + b[4] * 256 + b[5]) / 50000
    return lat, lon


def system_time_from_8chars(hex8: str) -> datetime:
    """
    Decode a 4-byte (8-char) little-endian Unix timestamp appended to
    SBE911 scans when 'Append System Time' is enabled.

    The bytes are stored in little-endian order in the stream, so the
    hex string must be byte-reversed before parsing.

    Returns
    -------
    datetime
        UTC datetime object.
    """
    # Reverse byte order: '986AF564' → '64F56A98'
    rev = hex8[6:8] + hex8[4:6] + hex8[2:4] + hex8[0:2]
    unix_ts = int(rev, 16)
    return datetime(1970, 1, 1) + timedelta(seconds=unix_ts)


def par_from_3chars(hex3: str) -> float:
    """
    Convert the SBE911 3-char (12-bit) surface PAR raw value to a voltage.

    The PAR word occupies 3 hex chars; the preceding 3 chars are unused
    padding (always '000' in tested files) — those are consumed by the
    caller before this function is called.

    Returns
    -------
    float
        Raw PAR count / 819 (matching seabirdscientific convention).
        Further conversion to physical units requires sensor calibration.
    """
    return int(hex3, 16) / 819


def sbe37_time_from_8chars(hex8: str) -> datetime:
    """
    Convert an 8-char SBE37 timestamp (seconds since 2000-01-01 UTC).

    The SBE37 stores time as a big-endian 4-byte integer counting seconds
    since 2000-01-01 00:00:00 UTC.

    Returns
    -------
    datetime
        UTC datetime object.
    """
    secs_since_2000 = int(hex8, 16)
    return datetime(2000, 1, 1) + timedelta(seconds=secs_since_2000)
