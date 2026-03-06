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
from tqdm.notebook import tqdm

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
    raw_names: bool = False,
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
    raw_names : bool
        If True, skip variable renaming and keep internal raw names
        (e.g. 'temperature_primary_raw', 'conductivity_primary_raw').
        Useful for troubleshooting. Default False.

    Returns
    -------
    xr.Dataset
        Dataset with:
          - 'scan' as the integer index dimension
          - 'TIME_SCAN' coordinate (datetime64[ns], per-scan) if timestamps
            are available; assign a scalar TIME per cast later via
            assign_cast_time()
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

    # TIME_SCAN coordinate — per-scan timestamps along scan dimension.
    # TIME (scalar, per-cast) is assigned later via assign_cast_time().
    if time_coord is not None:
        coords["TIME_SCAN"] = ("scan", time_coord)

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

    # Cruise metadata from ** header lines — only set if present
    for key in ("station", "cruise_name", "ship", "operator", "bottom_depth"):
        val = hex_header.get(key)
        if val is not None:
            ds.attrs[key] = val
    if hex_header.get("moon_pool") is not None:
        ds.attrs["moon_pool"] = hex_header["moon_pool"]

    # Header latitude/longitude (single position, not per-scan)
    if hex_header.get("latitude") is not None:
        ds.attrs["header_latitude"]  = hex_header["latitude"]
        ds.attrs["header_longitude"] = hex_header["longitude"]

    # ------------------------------------------------------------------
    # Step 7: Convert raw Hz / counts / volts to physical units
    # ------------------------------------------------------------------
    ds = convert_raw_dataset(ds, xmlcon_config, raw_names=raw_names)

    return ds


def parse_hex_dir(
    hex_dir: Union[str, Path],
    xmlcon: Union[str, Path, None] = None,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Parse all SBE hex files in a directory into a single multi-cast Dataset.

    Each hex file is parsed individually via parse_hex(), then all casts are
    concatenated along a 'TIME' dimension (cast start time). The scan
    dimension is padded with NaN to the length of the longest cast.

    Parameters
    ----------
    hex_dir : str or Path
        Directory containing .hex files. All .hex files found are loaded.
    xmlcon : str, Path, or None
        If a path is given, this single xmlcon file is used for all casts.
        If None (default), each hex file is matched to an xmlcon with the
        same stem in the same directory (e.g. Sta0243.hex → STA0243.XMLCON).
        Matching is case-insensitive.
    verbose : bool
        Print progress. Default True.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions (TIME, scan_count). TIME is a scalar
        coordinate giving cast start time. STATION is a coordinate along
        TIME populated from the ** header if available. All per-cast
        variables are NaN-padded to the longest cast.

    Raises
    ------
    FileNotFoundError
        If no hex files are found, or if xmlcon auto-matching fails for
        any cast.
    """
    hex_dir = Path(hex_dir)
    hex_files = sorted(hex_dir.glob("*.hex")) + sorted(hex_dir.glob("*.HEX"))
    hex_files = sorted(set(hex_files))  # deduplicate if both extensions present

    if not hex_files:
        raise FileNotFoundError(
            f"No .hex files found in '{hex_dir}'."
        )

    if verbose:
        print(f"Found {len(hex_files)} hex file(s) in '{hex_dir}'.")

    # ------------------------------------------------------------------
    # Build xmlcon lookup: stem (lower) → Path
    # ------------------------------------------------------------------
    if xmlcon is not None:
        xmlcon_path = Path(xmlcon)
        xmlcon_lookup = None  # signal: use same file for all
    else:
        xmlcon_lookup = {}
        for f in hex_dir.iterdir():
            if f.suffix.lower() in (".xmlcon", ".xml"):
                xmlcon_lookup[f.stem.lower()] = f

    # ------------------------------------------------------------------
    # Parse each cast
    # ------------------------------------------------------------------
    datasets = []
    failed   = []

    for hex_path in tqdm(hex_files, desc="Parsing hex files", disable=not verbose):
        # Resolve xmlcon
        if xmlcon_lookup is None:
            xc = xmlcon_path
        else:
            xc = xmlcon_lookup.get(hex_path.stem.lower())
            if xc is None:
                matches = [v for k, v in xmlcon_lookup.items()
                           if k == hex_path.stem.lower()]
                xc = matches[0] if matches else None
            if xc is None:
                msg = (f"No matching xmlcon found for '{hex_path.name}'. "
                       f"Pass xmlcon=<path> to use a single xmlcon for all casts.")
                failed.append((hex_path, msg))
                continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds_cast = parse_hex(hex_path, xc)
            datasets.append(ds_cast)
        except Exception as e:
            failed.append((hex_path, str(e)))

    if failed and verbose:
        print(f"\n  {len(failed)} cast(s) failed:")
        for path, msg in failed:
            print(f"    {path.name}: {msg}")

    if not datasets:
        raise ValueError("No casts were parsed successfully.")
    # ------------------------------------------------------------------
    # Concatenate along TIME, padding scan dimension
    # ------------------------------------------------------------------
    max_scans = max(ds.sizes["scan"] for ds in datasets)

    padded   = []
    times    = []
    stations = []

    for ds_cast in datasets:
        n = ds_cast.sizes["scan"]
        pad = max_scans - n

        # Get cast start time from TIME_SCAN before padding
        if "TIME_SCAN" in ds_cast.coords:
            cast_time = ds_cast.TIME_SCAN.values[0]
        else:
            cast_time = np.datetime64("NaT")
        times.append(cast_time)
        stations.append(ds_cast.attrs.get("station", None))

        # Move TIME_SCAN from coord → data var so it survives concat+padding
        if "TIME_SCAN" in ds_cast.coords:
            ts = ds_cast.TIME_SCAN.values
            ds_cast = ds_cast.drop_vars("TIME_SCAN")
            ds_cast["TIME_SCAN"] = xr.DataArray(ts, dims=["scan"],
                                                 attrs={"long_name": "per-scan timestamp"})

        if pad > 0:
            new_scan = np.arange(max_scans)
            pad_vars = {}
            for var in list(ds_cast.data_vars):
                arr = ds_cast[var].values
                if np.issubdtype(arr.dtype, np.floating):
                    padded_arr = np.full(max_scans, np.nan, dtype=arr.dtype)
                    padded_arr[:n] = arr
                elif np.issubdtype(arr.dtype, "datetime64"):
                    padded_arr = np.full(max_scans, np.datetime64("NaT"), dtype=arr.dtype)
                    padded_arr[:n] = arr
                elif np.issubdtype(arr.dtype, np.integer):
                    padded_arr = np.zeros(max_scans, dtype=arr.dtype)
                    padded_arr[:n] = arr
                else:
                    padded_arr = arr
                pad_vars[var] = xr.DataArray(
                    padded_arr, dims=["scan"], attrs=ds_cast[var].attrs)
            ds_cast = xr.Dataset(pad_vars,
                                 coords={"scan": new_scan},
                                 attrs=ds_cast.attrs)
        padded.append(ds_cast)

    # Rename scan → scan_count
    padded = [ds.rename({"scan": "scan_count"}) for ds in padded]

    # Concatenate — use a dummy integer dimension, then assign TIME
    multi = xr.concat(padded, dim="TIME", join="outer")

    times_float = (np.array(times, dtype="datetime64[ns]").astype("int64")
                / 86_400e9)  # nanoseconds → days
    multi = multi.assign_coords(TIME=("TIME", times_float))
    multi["TIME"].attrs.update({
        "units":     "days since 1970-01-01",
        "long_name": "cast start time",
    })
    # Restore TIME_SCAN as a coordinate (now shaped TIME, scan_count)
    if "TIME_SCAN" in multi.data_vars:
        multi = multi.set_coords("TIME_SCAN")

    # Add STATION coordinate if any cast had one
    if any(s is not None for s in stations):
        multi = multi.assign_coords(
            STATION=("TIME", [s if s is not None else "" for s in stations])
        )

    # Carry over attrs from first cast (instrument info etc.)
    # Cast-specific attrs (station, n_scans etc.) are dropped
    shared_attrs = {
        k: v for k, v in datasets[0].attrs.items()
        if k in ("instrument_family", "instrument_name", "software_version",
                 "xmlcon_instrument", "xmlcon_device_type", "parsed_by",
                 "cruise_name", "ship", "operator")
    }
    shared_attrs["n_casts"] = len(datasets)
    shared_attrs["max_scan_count"] = max_scans
    multi.attrs = shared_attrs

    # Sort by cast time so casts are always in chronological order
    multi = multi.sortby("TIME")

    if verbose:
        size_bytes = sum(v.nbytes for v in multi.data_vars.values())
        if size_bytes >= 1e9:
            size_str = f"{size_bytes/1e9:.1f} GB"
        else:
            size_str = f"{size_bytes/1e6:.0f} MB"
        print(f"Loaded {len(datasets)} cast(s) → "
              f"TIME={len(datasets)}, scan_count={max_scans} ({size_str})")

    return multi


def assign_cast_time(
    ds: xr.Dataset,
    method: str = "start",
) -> xr.Dataset:
    """
    Assign a scalar TIME coordinate to each cast in a multi-cast Dataset.

    TIME_SCAN contains per-scan timestamps along the scan_count dimension.
    This function derives a single representative timestamp per cast and
    assigns it as the TIME coordinate along the TIME dimension.

    Call this after processing steps (loop edit, in-water mask) so that
    TIME reflects the actual in-water data rather than deck time.

    Parameters
    ----------
    ds : xr.Dataset
        Multi-cast Dataset with dims (TIME, scan_count) and TIME_SCAN
        coordinate. Typically produced by parse_hex_dir() followed by
        processing steps.
    method : str
        How to compute the per-cast timestamp:
        - 'start'  : first non-NaT TIME_SCAN value (default)
        - 'mean'   : mean of all non-NaT TIME_SCAN values
        - 'median' : median of all non-NaT TIME_SCAN values

    Returns
    -------
    xr.Dataset
        Dataset with TIME coordinate updated in-place along the TIME dim.

    Examples
    --------
    >>> ds = parse_hex_dir('raw/')
    >>> ds = in_water_mask(ds)
    >>> ds = assign_cast_time(ds, method='start')
    """
    if "TIME_SCAN" not in ds.coords:
        raise ValueError(
            "TIME_SCAN coordinate not found. "
            "parse_hex_dir() produces TIME_SCAN; has it been dropped?"
        )

    time_scan = ds.TIME_SCAN.values  # shape: (TIME, scan_count)
    n_casts = time_scan.shape[0]
    cast_times = np.full(n_casts, np.datetime64("NaT"), dtype="datetime64[ns]")

    for i in range(n_casts):
        row = time_scan[i]
        valid = row[~np.isnat(row)]
        if len(valid) == 0:
            continue
        if method == "start":
            cast_times[i] = valid[0]
        elif method == "mean":
            cast_times[i] = np.datetime64(
                int(valid.astype("int64").mean()), "ns")
        elif method == "median":
            cast_times[i] = np.datetime64(
                int(np.median(valid.astype("int64"))), "ns")
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'start', 'mean', or 'median'."
            )

    return ds.assign_coords(TIME=("TIME", cast_times))
# ---------------------------------------------------------------------------

def _build_time_coord(
    scan_times,
    n_scans: int,
    hex_header: dict,
    xmlcon_config: dict,
    family: str,
) -> np.ndarray | None:
    """
    Build a numpy datetime64 array for the TIME_SCAN coordinate.

    Priority:
    1. Per-scan timestamps from the hex stream (most accurate).
    2. Reconstructed from upload_time and sample_interval_seconds.
    3. None — caller omits the TIME_SCAN coordinate.
    """
    # Option 1: per-scan timestamps in the stream
    if scan_times and any(t is not None for t in scan_times):
        valid = [(i, t) for i, t in enumerate(scan_times) if t is not None]

        def _to_dt64(t):
            try:
                return np.datetime64(t, "ns")
            except Exception:
                return np.datetime64("NaT")

        if len(valid) == n_scans:
            arr = np.array([_to_dt64(t) for t in scan_times])
            if not np.all(np.isnat(arr)):
                return arr
        else:
            # Partial timestamps — interpolate from first valid
            warnings.warn(
                f"Only {len(valid)} of {n_scans} scans have embedded timestamps. "
                f"Interpolating time from first valid timestamp.",
                stacklevel=4,
            )
            first_idx, first_time = valid[0]
            dt_s = _get_sample_interval(hex_header, xmlcon_config, family)
            if dt_s is not None:
                try:
                    base = _to_dt64(first_time)
                    if not np.isnat(base):
                        offsets = (np.arange(n_scans, dtype="float64")
                                   - first_idx) * dt_s * 1e9
                        return base + offsets.astype("timedelta64[ns]")
                except Exception:
                    pass

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