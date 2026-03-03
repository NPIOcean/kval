"""
kval.file.sbe_hex

Public API for reading Sea-Bird SBE hex (.hex) data files.

Supports:
  - SBE 911plus / 917plus  (profiling CTD)
  - SBE 37 family          (moored MicroCAT, all variants)

Usage
-----
    from kval.file.sbe_hex import parse_hex

    ds = parse_hex("Sta0243.hex", "STA0243.xmlcon")

The returned xr.Dataset contains:
  - Raw frequency / count / voltage arrays as data variables,
    with sensor metadata (serial number, calibration date, sensor type)
    stored as variable-level attributes.
  - Per-scan timestamps as the 'TIME' coordinate (if available in the
    hex stream), or a regularly-spaced array reconstructed from the
    upload time and sample rate.
  - NMEA latitude / longitude as data variables (SBE911 if NMEA enabled).
  - Dataset-level attributes: source file, instrument name, upload time,
    number of scans, bytes per scan, etc.

The raw data is suitable for passing directly to seabirdscientific
conversion functions, which expect the raw frequency / count values
produced here.

Notes
-----
- No engineering-unit conversion is performed here.  This module
  produces raw counts / frequencies; apply seabirdscientific.conversion
  (or kval.process.sbe_convert, once implemented) for physical units.
- The layout is validated against the 'Number of Bytes Per Scan' stated
  in the hex file header.  A mismatch raises SBEHexLayoutError with a
  detailed, human-readable explanation — this almost always means the
  .hex and .xmlcon files are from different casts.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Union
import numpy as np
import xarray as xr

from kval.file.sbe_xmlcon import parse_xmlcon
from kval.file._sbe_hex_common import (
    parse_hex_header,
    SBEHexError,
    SBEHexFileError,
)
from kval.file._sbe_hex_911 import (
    parse_911_scans,
    get_911_sensor_attrs,
)
from kval.file._sbe_hex_37 import (
    parse_37_scans,
    get_37_sensor_attrs,
)
from kval.file._sbe_convert import convert_raw_dataset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_hex(
    hex_path: Union[str, Path],
    xmlcon_path: Union[str, Path],
) -> xr.Dataset:
    """
    Parse an SBE hex file into an xarray Dataset.

    Parameters
    ----------
    hex_path : str or Path
        Path to the .hex data file.
    xmlcon_path : str or Path
        Path to the matching .xmlcon configuration file.  Must be the
        configuration that was active when the data was collected —
        a mismatch will be caught and reported clearly.

    Returns
    -------
    xr.Dataset
        Dataset with:
          - 'scan' as the integer index dimension
          - 'TIME' coordinate (datetime64) if per-scan timestamps are
            available, otherwise constructed from upload time + sample rate
          - One data variable per raw sensor channel
          - 'LATITUDE' and 'LONGITUDE' variables (SBE911 with NMEA only)
          - Variable-level attributes: sensor_type, serial_number,
            calibration_date, units
          - Dataset-level attributes: source_hex_file, source_xmlcon_file,
            instrument_name, upload_time, n_scans, bytes_per_scan,
            software_version, instrument_family

    Raises
    ------
    SBEHexFileError
        If the hex file cannot be read or contains no data.
    SBEHexLayoutError
        If the sensor layout derived from the xmlcon does not match the
        bytes-per-scan stated in the hex file header.
    ValueError
        If the instrument type is not supported (not 911 or 37 family).

    Examples
    --------
    >>> ds = parse_hex("Sta0243.hex", "STA0243.xmlcon")
    >>> print(ds)
    >>> ds["temperature_primary_raw"].attrs
    {'sensor_type': 'TemperatureSensor', 'serial_number': '4239',
     'calibration_date': '2023-03-15', 'units': 'Hz'}
    """
    hex_path   = Path(hex_path)
    xmlcon_path = Path(xmlcon_path)

    # ------------------------------------------------------------------
    # Step 1: Parse the hex file header and the xmlcon
    # ------------------------------------------------------------------
    hex_header     = parse_hex_header(hex_path)
    xmlcon_config  = parse_xmlcon(xmlcon_path)
    family         = hex_header["instrument_family"]

    # ------------------------------------------------------------------
    # Step 2: Dispatch to the correct instrument parser
    # ------------------------------------------------------------------
    if family == "911":
        raw_data    = parse_911_scans(hex_path, xmlcon_config, hex_header, xmlcon_path)
        sensor_attrs = get_911_sensor_attrs(xmlcon_config)
    elif family == "37":
        raw_data    = parse_37_scans(hex_path, xmlcon_config, hex_header, xmlcon_path)
        sensor_attrs = get_37_sensor_attrs(xmlcon_config)
    else:
        raise ValueError(
            f"\n\nUnsupported instrument type in '{hex_path.name}'.\n"
            f"\n  Detected family: '{family}'"
            f"\n  First header line: '{hex_header['raw_header_lines'][0] if hex_header['raw_header_lines'] else '(empty)'}'"
            f"\n\n  This parser supports SBE 911/917 plus and SBE 37 family instruments."
            f"\n  If you have a different instrument type, please open an issue or"
            f"\n  contact the kval maintainers.\n"
        )

    # ------------------------------------------------------------------
    # Step 3: Build TIME coordinate
    # ------------------------------------------------------------------
    n_scans     = hex_header["n_scans"]
    scan_times  = raw_data.pop("_scan_times", None)
    time_coord  = _build_time_coord(
        scan_times, n_scans, hex_header, xmlcon_config, family
    )

    # ------------------------------------------------------------------
    # Step 4: Extract NMEA arrays (SBE911 only)
    # ------------------------------------------------------------------
    nmea_lat = raw_data.pop("_nmea_lat", None)
    nmea_lon = raw_data.pop("_nmea_lon", None)

    # ------------------------------------------------------------------
    # Step 5: Assemble the xarray Dataset
    # ------------------------------------------------------------------
    scan_coord = np.arange(n_scans, dtype=np.int32)

    data_vars = {}
    coords    = {"scan": scan_coord}

    # TIME coordinate
    if time_coord is not None:
        coords["TIME"] = ("scan", time_coord)

    # Sensor data variables
    for field_name, arr in raw_data.items():
        if field_name.startswith("_"):
            continue
        attrs = sensor_attrs.get(field_name, {})
        data_vars[field_name] = xr.DataArray(arr, dims=["scan"], attrs=attrs)

    # NMEA position
    if nmea_lat is not None:
        data_vars["LATITUDE"] = xr.DataArray(
            nmea_lat, dims=["scan"],
            attrs={"units": "degrees_north", "long_name": "NMEA latitude",
                   "source": "NMEA appended to scan"},
        )
    if nmea_lon is not None:
        data_vars["LONGITUDE"] = xr.DataArray(
            nmea_lon, dims=["scan"],
            attrs={"units": "degrees_east", "long_name": "NMEA longitude",
                   "source": "NMEA appended to scan"},
        )

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # ------------------------------------------------------------------
    # Step 6: Dataset-level attributes
    # ------------------------------------------------------------------
    upload_time = hex_header.get("upload_time")
    ds.attrs = {
        "instrument_family":  family,
        "instrument_name":    hex_header.get("instrument_name", ""),
        "software_version":   hex_header.get("software_version", ""),
        "source_hex_file":    str(hex_path.resolve()),
        "source_xmlcon_file": str(xmlcon_path.resolve()),
        "upload_time":        upload_time.isoformat() if upload_time else "",
        "n_scans":            n_scans,
        "bytes_per_scan":     hex_header["bytes_per_scan"],
        "n_voltage_words":    hex_header.get("n_voltage_words", 0),
        "nmea_position_added": hex_header["nmea_position_added"],
        "scan_time_added":    hex_header["scan_time_added"],
        "surface_par_added":  hex_header["surface_par_added"],
        "user_comments":      " | ".join(hex_header.get("user_comments", [])),
        "xmlcon_instrument":  xmlcon_config["instrument"].get("name", ""),
        "xmlcon_device_type": xmlcon_config["instrument"].get("device_type", ""),
        "parsed_by":          "kval.file.sbe_hex",
    }

    # Header latitude/longitude (single position, not per-scan)
    if hex_header.get("latitude") is not None:
        ds.attrs["header_latitude"]  = hex_header["latitude"]
        ds.attrs["header_longitude"] = hex_header["longitude"]

    # ------------------------------------------------------------------
    # Step 7: Convert raw Hz / counts / volts to physical units
    # ------------------------------------------------------------------
    ds = convert_raw_dataset(ds, xmlcon_config)

    return ds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_time_coord(
    scan_times,
    n_scans: int,
    hex_header: dict,
    xmlcon_config: dict,
    family: str,
) -> np.ndarray | None:
    """
    Build a numpy datetime64 array for the TIME coordinate.

    Priority:
    1. Per-scan timestamps from the hex stream (most accurate).
    2. Reconstructed from upload_time and sample_interval_seconds.
    3. None — caller omits the TIME coordinate.
    """
    # Option 1: per-scan timestamps in the stream
    if scan_times and any(t is not None for t in scan_times):
        valid = [(i, t) for i, t in enumerate(scan_times) if t is not None]
        if len(valid) == n_scans:
            # All scans have timestamps — convert directly
            return np.array([np.datetime64(t, "ns") for t in scan_times])
        else:
            # Partial timestamps — use first valid as anchor + interpolate
            warnings.warn(
                f"Only {len(valid)} of {n_scans} scans have embedded timestamps. "
                f"Interpolating time from first valid timestamp.",
                stacklevel=4,
            )
            first_idx, first_time = valid[0]
            # Determine sample interval
            dt_s = _get_sample_interval(hex_header, xmlcon_config, family)
            if dt_s is None:
                # Can't interpolate without interval — fall through
                pass
            else:
                base = np.datetime64(first_time, "ns")
                offsets = np.array(
                    [(i - first_idx) * dt_s * 1e9 for i in range(n_scans)],
                    dtype="timedelta64[ns]",
                )
                return base + offsets

    # Option 2: reconstruct from upload_time and sample interval
    upload_time = hex_header.get("upload_time")
    dt_s = _get_sample_interval(hex_header, xmlcon_config, family)

    if upload_time is not None and dt_s is not None:
        # Upload time is the END of the file; reconstruct backwards.
        # For SBE911 (profiling), upload_time ≈ cast end time.
        # For SBE37 (moored), upload_time is when data was downloaded.
        end_time = np.datetime64(upload_time, "ns")
        offsets  = np.arange(n_scans, dtype="float64") * dt_s * 1e9
        start    = end_time - np.timedelta64(int((n_scans - 1) * dt_s * 1e9), "ns")
        times    = start + offsets.astype("timedelta64[ns]")
        warnings.warn(
            f"No per-scan timestamps found in hex stream. "
            f"TIME coordinate reconstructed from upload time "
            f"({upload_time.isoformat()}) and sample interval "
            f"({dt_s} s).  This is approximate.",
            stacklevel=4,
        )
        return times

    # Option 3: no time information available
    return None


def _get_sample_interval(
    hex_header: dict,
    xmlcon_config: dict,
    family: str,
) -> float | None:
    """
    Return sample interval in seconds, or None if unknown.

    For SBE911: 24 Hz (1/24 s between scans after deck unit averaging).
    For SBE37:  from xmlcon SampleIntervalSeconds.
    """
    if family == "37":
        si = xmlcon_config["instrument"].get("sample_interval_seconds")
        if si is not None:
            return float(si)
    elif family == "911":
        # SBE911 always samples at 24 Hz after deck unit averaging
        scans_avg = xmlcon_config["instrument"].get("scans_to_average", 1) or 1
        return scans_avg / 24.0
    return None
