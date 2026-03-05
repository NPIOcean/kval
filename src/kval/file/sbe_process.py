"""
kval.file.sbe_process

Processing steps for SBE CTD data in the multi-cast xarray Dataset format
produced by kval.file.sbe_hex.parse_hex_dir().

All functions wrap seabirdscientific.processing to guarantee SBE-compatible
algorithms, and operate on Datasets with dimensions (TIME, scan_count).

Typical workflow
----------------
    ds = parse_hex_dir('raw/')
    ds = celltm(ds)
    ds = align(ds, vars=['CNDC1', 'CNDC2'], offset=0.073)
    ds = low_pass_filter(ds)
    ds = loop_edit(ds)
    ds = wild_edit(ds)
    ds = bin_profiles(ds, bin_size=1)   # applies flags automatically

    # or inspect flags before binning:
    ds_clean = apply_flags(ds)

FLAG convention
---------------
A FLAG variable (TIME, scan_count) is created and updated by flagging
steps (loop_edit, wild_edit). Values are:
    0.0         — good scan
    -9.99e-29   — bad scan (SBE standard flag value)

Data variables are never modified by flagging steps — the raw converted
trace is always preserved. apply_flags() NaNs out flagged scans.

PROCESSING bookkeeping
----------------------
Every function appends an entry to ds.attrs['PROCESSING'], a list of dicts:
    {'step': <function name>, 'params': {...}, 'timestamp': <ISO8601>}
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Union, List, Optional

import numpy as np
import xarray as xr
from tqdm.notebook import tqdm

import seabirdscientific.processing as proc
from seabirdscientific.processing import MinVelocityType, WindowFilterType, CastType

# SBE standard bad-flag value
_FLAG_VALUE: float = proc.FLAG_VALUE

# Default physical variables to apply per-variable filters to
# (excludes position, time, and flag variables)
_DEFAULT_FILTER_VARS = (
    "TEMP1", "TEMP2", "PRES", "CNDC1", "CNDC2",
    "DOXY1_instr", "DOXY2_instr",
    "CHLA1_fluorescence", "CHLA2_fluorescence",
    "CDOM1_instr", "CDOM2_instr",
    "TRANS1", "ALTI", "PAR", "SPAR",
)

# NPI standard filter assignments from Filter.psa (SeaSoft 7.26.7.129)
# FilterType 0=none, 1=low-pass TC_A=0.030s, 2=low-pass TC_B=0.150s
# TEMP1/TEMP2/CNDC1/CNDC2 are not filtered (applied after celltm/align)
_NPI_FILTER_TC_A = 0.030   # oxygen sensors
_NPI_FILTER_TC_B = 0.150   # pressure and optical sensors

_NPI_FILTER_VARS_TC_A = ("DOXY1_instr", "DOXY2_instr")
_NPI_FILTER_VARS_TC_B = (
    "PRES",
    "CDOM1_instr", "CDOM2_instr",
    "CHLA1_fluorescence", "CHLA2_fluorescence",
    "TRANS1",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_interval(ds: xr.Dataset) -> float:
    """Return sample interval in seconds from dataset attrs."""
    si = ds.attrs.get("sample_interval_seconds")
    if si is not None:
        return float(si)
    # SBE911 default: 24 Hz after deck unit averaging
    scans_avg = ds.attrs.get("scans_to_average", 1) or 1
    return float(scans_avg) / 24.0


def _record(ds: xr.Dataset, step: str, params: dict) -> xr.Dataset:
    """Append a processing record to ds.attrs['PROCESSING']."""
    log = list(ds.attrs.get("PROCESSING", []))
    log.append({
        "step": step,
        "params": params,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    ds.attrs["PROCESSING"] = log
    return ds


def _resolve_vars(
    ds: xr.Dataset,
    vars: Optional[Union[str, List[str]]],
) -> List[str]:
    """Resolve a vars argument to a list of variable names present in ds."""
    if vars is None:
        candidates = _DEFAULT_FILTER_VARS
    elif isinstance(vars, str):
        candidates = [vars]
    else:
        candidates = list(vars)
    present = [v for v in candidates if v in ds.data_vars]
    missing = [v for v in candidates if v not in ds.data_vars]
    if missing and vars is not None:
        import warnings
        warnings.warn(
            f"sbe_process: variable(s) not found in dataset, skipping: {missing}",
            stacklevel=3,
        )
    return present


def _ensure_flag(ds: xr.Dataset) -> xr.Dataset:
    """Create FLAG variable if not present, initialised to 0 (good)."""
    if "FLAG" not in ds.data_vars:
        flag_data = np.zeros(
            (ds.sizes["TIME"], ds.sizes["scan_count"]), dtype=np.float64
        )
        ds["FLAG"] = xr.DataArray(
            flag_data, dims=["TIME", "scan_count"],
            attrs={"long_name": "scan quality flag",
                   "flag_values": "0.0 = good, -9.99e-29 = bad"},
        )
    return ds


def _iter_casts(ds: xr.Dataset, desc: str):
    """Yield (i, station_label) with a tqdm progress bar over TIME."""
    n = ds.sizes["TIME"]
    stations = ds.coords["STATION"].values if "STATION" in ds.coords else [str(i) for i in range(n)]
    for i in tqdm(range(n), desc=desc, unit="cast"):
        yield i, stations[i]


def _cast_slice(ds: xr.Dataset, i: int) -> tuple[np.ndarray, int]:
    """
    Return the valid (non-NaT) scan slice for cast i.

    TIME_SCAN NaT marks NaN-padding from shorter casts in the padded
    multi-cast array. We only want to process real data.

    Returns (valid_mask, n_valid) where valid_mask is a boolean array
    of length scan_count.
    """
    if "TIME_SCAN" in ds.coords:
        ts = ds.TIME_SCAN.values[i]
        valid = ~np.isnat(ts)
    else:
        # No TIME_SCAN — assume all scans are real
        valid = np.ones(ds.sizes["scan_count"], dtype=bool)
    return valid, int(valid.sum())


# ---------------------------------------------------------------------------
# Cell thermal mass correction
# ---------------------------------------------------------------------------

def celltm(
    ds: xr.Dataset,
    amplitude: float = 0.03,
    time_constant: float = 7.0,
    pairs: Optional[List[tuple[str, str]]] = None,
) -> xr.Dataset:
    """
    Remove conductivity cell thermal mass effects (SeaSoft CellTM).

    Wraps seabirdscientific.processing.cell_thermal_mass().

    Parameters
    ----------
    ds : xr.Dataset
        Multi-cast Dataset with dims (TIME, scan_count).
    amplitude : float
        Thermal anomaly amplitude (alpha). Default 0.03 (SBE standard).
    time_constant : float
        Thermal anomaly time constant 1/beta in seconds. Default 7.0.
    pairs : list of (temp_var, cndc_var) tuples, or None
        Sensor pairs to correct. Default: auto-detect from present
        variables — [(TEMP1, CNDC1), (TEMP2, CNDC2)] if both present.

    Returns
    -------
    xr.Dataset
        Dataset with corrected conductivity variables (in-place copy).
    """
    ds = ds.copy()
    dt = _sample_interval(ds)

    if pairs is None:
        pairs = []
        if "TEMP1" in ds and "CNDC1" in ds:
            pairs.append(("TEMP1", "CNDC1"))
        if "TEMP2" in ds and "CNDC2" in ds:
            pairs.append(("TEMP2", "CNDC2"))

    if not pairs:
        import warnings
        warnings.warn("celltm: no TEMP/CNDC pairs found in dataset.", stacklevel=2)
        return ds

    for i, _ in _iter_casts(ds, "Cell thermal mass"):
        valid, n = _cast_slice(ds, i)
        if n < 2:
            continue
        for temp_var, cndc_var in pairs:
            if temp_var not in ds or cndc_var not in ds:
                continue
            temp = ds[temp_var].values[i, valid]
            cndc = ds[cndc_var].values[i, valid]
            corrected = proc.cell_thermal_mass(temp, cndc, amplitude, time_constant, dt)
            vals = ds[cndc_var].values.copy()
            vals[i, valid] = corrected
            ds[cndc_var] = xr.DataArray(vals, dims=ds[cndc_var].dims,
                                        attrs=ds[cndc_var].attrs)

    return _record(ds, "celltm", {
        "amplitude": amplitude,
        "time_constant": time_constant,
        "pairs": [(t, c) for t, c in pairs],
    })


# ---------------------------------------------------------------------------
# CTD alignment (advance/delay a variable in time)
# ---------------------------------------------------------------------------

def align(
    ds: xr.Dataset,
    vars: Union[str, List[str], None] = None,
    offset: float = 0.073,
) -> xr.Dataset:
    """
    Apply a time offset to one or more variables (SeaSoft Align CTD).

    Wraps seabirdscientific.processing.align_ctd(). Typically used to
    advance conductivity relative to temperature to correct for the
    conductivity cell's response lag (~0.073 s for SBE 911).

    Note: not part of the NPI standard processing pipeline (no advance
    offset is set in DatCnv.psa). Available for users who need it.

    Parameters
    ----------
    ds : xr.Dataset
        Multi-cast Dataset with dims (TIME, scan_count).
    vars : str, list of str, or None
        Variable(s) to align. Default: ['CNDC1', 'CNDC2'] if present.
    offset : float
        Time offset in seconds. Positive = advance (shift earlier in time).
        Default 0.073 s (SBE 911 standard).

    Returns
    -------
    xr.Dataset
        Dataset with aligned variables (in-place copy).
    """
    ds = ds.copy()
    dt = _sample_interval(ds)

    if vars is None:
        vars = [v for v in ("CNDC1", "CNDC2") if v in ds.data_vars]
    var_list = _resolve_vars(ds, vars)

    for i, _ in _iter_casts(ds, "Align CTD"):
        valid, n = _cast_slice(ds, i)
        if n < 2:
            continue
        for var in var_list:
            arr = ds[var].values[i, valid]
            aligned = proc.align_ctd(arr, offset, dt)
            vals = ds[var].values.copy()
            vals[i, valid] = aligned
            ds[var] = xr.DataArray(vals, dims=ds[var].dims, attrs=ds[var].attrs)

    return _record(ds, "align", {"vars": var_list, "offset_s": offset})


# ---------------------------------------------------------------------------
# Low-pass filter
# ---------------------------------------------------------------------------

def low_pass_filter(
    ds: xr.Dataset,
    vars: Union[str, List[str], None] = None,
    time_constant: float = 0.15,
) -> xr.Dataset:
    """
    Apply SeaSoft low-pass filter to one or more variables.

    Wraps seabirdscientific.processing.low_pass_filter().

    NPI standard (Filter.psa, SeaSoft 7.26.7.129) applies this filter
    in two separate calls with different time constants and variable sets:

        ds = celltm(ds)
        ds = low_pass_filter(ds, vars=['DOXY1_instr', 'DOXY2_instr'],
                             time_constant=0.030)
        ds = low_pass_filter(ds, vars=['PRES', 'CHLA1_fluorescence',
                                       'CDOM1_instr', 'TRANS1'],
                             time_constant=0.150)
        ds = loop_edit(ds)
        ds = wild_edit(ds)
        binned = bin_profiles(ds)

    Parameters
    ----------
    ds : xr.Dataset
        Multi-cast Dataset with dims (TIME, scan_count).
    vars : str, list of str, or None
        Variable(s) to filter. Default: all standard physical variables
        present in the dataset.
    time_constant : float
        Filter time constant in seconds. Default 0.15 s.

    Returns
    -------
    xr.Dataset
        Dataset with filtered variables (in-place copy).
    """
    ds = ds.copy()
    dt = _sample_interval(ds)
    var_list = _resolve_vars(ds, vars)

    for i, _ in _iter_casts(ds, "Low-pass filter"):
        valid, n = _cast_slice(ds, i)
        if n < 2:
            continue
        for var in var_list:
            arr = ds[var].values[i, valid]
            if not np.issubdtype(arr.dtype, np.floating):
                continue
            filtered = proc.low_pass_filter(arr, time_constant, dt)
            vals = ds[var].values.copy()
            vals[i, valid] = filtered
            ds[var] = xr.DataArray(vals, dims=ds[var].dims, attrs=ds[var].attrs)

    return _record(ds, "low_pass_filter", {
        "vars": var_list, "time_constant_s": time_constant,
    })


# ---------------------------------------------------------------------------
# Butterworth filter
# ---------------------------------------------------------------------------

def butterworth_filter(
    ds: xr.Dataset,
    vars: Union[str, List[str], None] = None,
    time_constant: float = 0.15,
) -> xr.Dataset:
    """
    Apply a Butterworth low-pass filter to one or more variables.

    Wraps seabirdscientific.processing.butterworth_filter().

    Parameters
    ----------
    ds : xr.Dataset
    vars : str, list of str, or None
        Default: all standard physical variables present.
    time_constant : float
        Time constant in seconds (1 / (2π * cutoff_freq)). Default 0.15 s.
    """
    ds = ds.copy()
    dt = _sample_interval(ds)
    var_list = _resolve_vars(ds, vars)

    for i, _ in _iter_casts(ds, "Butterworth filter"):
        valid, n = _cast_slice(ds, i)
        if n < 2:
            continue
        for var in var_list:
            arr = ds[var].values[i, valid]
            if not np.issubdtype(arr.dtype, np.floating):
                continue
            filtered = proc.butterworth_filter(arr, time_constant, dt)
            vals = ds[var].values.copy()
            vals[i, valid] = filtered
            ds[var] = xr.DataArray(vals, dims=ds[var].dims, attrs=ds[var].attrs)

    return _record(ds, "butterworth_filter", {
        "vars": var_list, "time_constant_s": time_constant,
    })


# ---------------------------------------------------------------------------
# Window filter
# ---------------------------------------------------------------------------

def window_filter(
    ds: xr.Dataset,
    vars: Union[str, List[str], None] = None,
    window_type: WindowFilterType = WindowFilterType.GAUSSIAN,
    window_width: int = 3,
    half_width: float = 1.0,
    offset: float = 0.0,
    exclude_flags: bool = True,
) -> xr.Dataset:
    """
    Apply a window filter (boxcar, cosine, triangle, gaussian, median).

    Wraps seabirdscientific.processing.window_filter().

    Parameters
    ----------
    ds : xr.Dataset
    vars : str, list of str, or None
        Default: all standard physical variables present.
    window_type : WindowFilterType
        Filter shape. Default: GAUSSIAN.
    window_width : int
        Number of samples in the window (must be odd). Default 3.
    half_width : float
        Gaussian half-width parameter. Default 1.0.
    offset : float
        Gaussian centre offset. Default 0.0.
    exclude_flags : bool
        Exclude flagged scans from filter calculation. Default True.
    """
    ds = _ensure_flag(ds)
    ds = ds.copy()
    dt = _sample_interval(ds)
    var_list = _resolve_vars(ds, vars)

    for i, _ in _iter_casts(ds, "Window filter"):
        valid, n = _cast_slice(ds, i)
        if n < 2:
            continue
        flags = ds["FLAG"].values[i, valid]
        for var in var_list:
            arr = ds[var].values[i, valid]
            if not np.issubdtype(arr.dtype, np.floating):
                continue
            filtered = proc.window_filter(
                arr, flags, window_type, window_width, dt,
                half_width, offset, exclude_flags, _FLAG_VALUE,
            )
            vals = ds[var].values.copy()
            vals[i, valid] = filtered
            ds[var] = xr.DataArray(vals, dims=ds[var].dims, attrs=ds[var].attrs)

    return _record(ds, "window_filter", {
        "vars": var_list,
        "window_type": window_type.value,
        "window_width": window_width,
    })


# ---------------------------------------------------------------------------
# Loop edit
# ---------------------------------------------------------------------------

def loop_edit(
    ds: xr.Dataset,
    min_velocity: float = 0.25,
    min_velocity_type: MinVelocityType = MinVelocityType.FIXED,
    window_size: float = 300.0,
    mean_speed_percent: float = 20.0,
    remove_surface_soak: bool = True,
    min_soak_depth: float = 3.0,
    max_soak_depth: float = 20.0,
    use_deck_pressure_offset: bool = False,
    exclude_flags: bool = True,
) -> xr.Dataset:
    """
    Flag pressure loops and surface soak (SeaSoft Loop Edit).

    Wraps seabirdscientific.processing.loop_edit_pressure(). Operates
    per cast using PRES and LATITUDE. Updates the FLAG variable.

    Parameters
    ----------
    ds : xr.Dataset
        Multi-cast Dataset. Must contain PRES. LATITUDE used if present,
        otherwise defaults to 0° (equator — small error on pressure→depth).
    min_velocity : float
        Minimum descent velocity in m/s. Default 0.25.
    min_velocity_type : MinVelocityType
        FIXED (default) or PERCENT.
    window_size : float
        Window for mean speed calculation in seconds. Default 3.0.
    mean_speed_percent : float
        Percent of mean speed threshold (used with PERCENT type). Default 20.
    remove_surface_soak : bool
        Flag scans before min_soak_depth is reached. Default True.
    min_soak_depth : float
        Minimum depth (dbar) to consider start of downcast. Default 0.5.
    max_soak_depth : float
        Maximum depth (dbar) for soak region. Default 5.0.
    use_deck_pressure_offset : bool
        Offset soak depths by first pressure value. Default True.
    exclude_flags : bool
        Preserve existing flags. Default False.

    Returns
    -------
    xr.Dataset
        Dataset with updated FLAG variable.
    """
    ds = _ensure_flag(ds)
    ds = ds.copy()
    dt = _sample_interval(ds)

    if "PRES" not in ds:
        raise ValueError("loop_edit requires PRES variable.")

    import warnings
    failed = []

    for i, station in _iter_casts(ds, "Loop edit"):
        valid, n = _cast_slice(ds, i)
        if n < 4:
            continue

        pres = ds["PRES"].values[i, valid]
        flags = ds["FLAG"].values[i, valid].copy()

        if "LATITUDE" in ds:
            lat_arr = ds["LATITUDE"].values[i, valid]
            lat = float(np.nanmean(lat_arr))
        else:
            lat = 0.0

        try:
            _ = proc.loop_edit_pressure(
                pres, lat, flags, dt,
                min_velocity_type, min_velocity,
                window_size, mean_speed_percent,
                remove_surface_soak, min_soak_depth, max_soak_depth,
                use_deck_pressure_offset, exclude_flags, _FLAG_VALUE,
            )
        except Exception as e:
            failed.append((station, str(e)))
            continue

        full_flags = ds["FLAG"].values.copy()
        bad = flags == _FLAG_VALUE
        full_flags[i, valid] = np.where(bad, _FLAG_VALUE, full_flags[i, valid])
        ds["FLAG"] = xr.DataArray(full_flags, dims=ds["FLAG"].dims,
                                  attrs=ds["FLAG"].attrs)

    if failed:
        warnings.warn(
            f"loop_edit: {len(failed)} cast(s) skipped:\n" +
            "\n".join(f"  {s}: {e}" for s, e in failed),
            stacklevel=2,
        )

    return _record(ds, "loop_edit", {
        "min_velocity": min_velocity,
        "min_velocity_type": min_velocity_type.name,
        "remove_surface_soak": remove_surface_soak,
        "min_soak_depth": min_soak_depth,
        "max_soak_depth": max_soak_depth,
    })


# ---------------------------------------------------------------------------
# Wild edit
# ---------------------------------------------------------------------------

def wild_edit(
    ds: xr.Dataset,
    vars: Union[str, List[str], None] = None,
    std_pass_1: float = 2.0,
    std_pass_2: float = 20.0,
    scans_per_block: int = 100,
    distance_to_mean: float = 0.0,
    exclude_bad_flags: bool = True,
) -> xr.Dataset:
    """
    Flag outliers by standard deviation (SeaSoft Wild Edit).

    Wraps seabirdscientific.processing.wild_edit(). Updates FLAG.
    Run after loop_edit so that loop-flagged scans are excluded from
    the outlier statistics.

    Parameters
    ----------
    ds : xr.Dataset
    vars : str, list of str, or None
        Variables to check. Default: all standard physical variables.
    std_pass_1 : float
        Std dev threshold for first pass (temporary removal). Default 2.0.
    std_pass_2 : float
        Std dev threshold for second pass (final flagging). Default 20.0.
    scans_per_block : int
        Block size for statistics. Default 100.
    distance_to_mean : float
        Minimum distance from mean to flag. Default 0.0.
    exclude_bad_flags : bool
        Exclude loop-edit-flagged scans from statistics. Default True.

    Returns
    -------
    xr.Dataset
        Dataset with updated FLAG variable.
    """
    ds = _ensure_flag(ds)
    ds = ds.copy()
    var_list = _resolve_vars(ds, vars)

    for i, _ in _iter_casts(ds, "Wild edit"):
        valid, n = _cast_slice(ds, i)
        if n < 2:
            continue
        flags = ds["FLAG"].values[i, valid].copy()

        for var in var_list:
            arr = ds[var].values[i, valid].copy()
            if not np.issubdtype(arr.dtype, np.floating):
                continue
            # wild_edit writes flag_value into the data array for bad scans
            flagged = proc.wild_edit(
                arr, flags, std_pass_1, std_pass_2,
                scans_per_block, distance_to_mean,
                exclude_bad_flags, _FLAG_VALUE,
            )
            # Extract which scans were newly flagged and OR into FLAG
            newly_bad = flagged == _FLAG_VALUE
            full_flags = ds["FLAG"].values.copy()
            full_flags[i, valid] = np.where(
                newly_bad, _FLAG_VALUE, full_flags[i, valid]
            )
            ds["FLAG"] = xr.DataArray(full_flags, dims=ds["FLAG"].dims,
                                      attrs=ds["FLAG"].attrs)

    return _record(ds, "wild_edit", {
        "vars": var_list,
        "std_pass_1": std_pass_1,
        "std_pass_2": std_pass_2,
        "scans_per_block": scans_per_block,
    })


# ---------------------------------------------------------------------------
# Apply flags
# ---------------------------------------------------------------------------

def apply_flags(
    ds: xr.Dataset,
    vars: Union[str, List[str], None] = None,
) -> xr.Dataset:
    """
    Set flagged scans to NaN in data variables.

    Does not modify the FLAG variable itself. Safe to call multiple
    times — idempotent on already-NaN values.

    Parameters
    ----------
    ds : xr.Dataset
    vars : str, list of str, or None
        Variables to apply flags to. Default: all standard physical
        variables present.

    Returns
    -------
    xr.Dataset
        Copy with flagged scans set to NaN.
    """
    if "FLAG" not in ds.data_vars:
        return ds  # nothing to do

    ds = ds.copy()
    var_list = _resolve_vars(ds, vars)
    bad = ds["FLAG"].values == _FLAG_VALUE  # (TIME, scan_count) bool

    for var in var_list:
        arr = ds[var].values.copy()
        if not np.issubdtype(arr.dtype, np.floating):
            continue
        arr[bad] = np.nan
        ds[var] = xr.DataArray(arr, dims=ds[var].dims, attrs=ds[var].attrs)

    return _record(ds, "apply_flags", {"vars": var_list})


# ---------------------------------------------------------------------------
# Bin profiles
# ---------------------------------------------------------------------------

def bin_profiles(
    ds: xr.Dataset,
    bin_size: float = 1.0,
    bin_var: str = "PRES",
    cast_type: CastType = CastType.DOWNCAST,
    min_scans: int = 1,
    exclude_bad_scans: bool = True,
    interpolate: bool = False,
    include_surface_bin: bool = True,
) -> xr.Dataset:
    """
    Bin all casts to a regular pressure (or depth) grid.

    Applies flags before binning, then calls
    seabirdscientific.processing.bin_average() per cast and concatenates
    the results along TIME.

    Parameters
    ----------
    ds : xr.Dataset
        Multi-cast Dataset with dims (TIME, scan_count). Should have
        been through loop_edit (and optionally wild_edit) first.
    bin_size : float
        Bin width in dbar. Default 1.0.
    bin_var : str
        Variable to bin on. Default 'PRES'.
    cast_type : CastType
        Which part of the cast to bin: DOWNCAST (default), UPCAST, BOTH.
    min_scans : int
        Minimum scans per bin to include. Default 1.
    exclude_bad_scans : bool
        Exclude loop-edit-flagged scans from bins. Default True.
    interpolate : bool
        Interpolate to bin midpoints (SBE style). Default False.
    include_surface_bin : bool
        Include a surface bin at the start of the downcast. Default True
        (matches NPI standard BinAvg.psa).

    Returns
    -------
    xr.Dataset
        Dataset with dims (TIME, PRES). TIME and STATION coords preserved.
    """
    import pandas as pd

    if bin_var not in ds:
        raise ValueError(f"bin_profiles: '{bin_var}' not found in dataset.")

    # Apply flags before binning
    ds_flagged = apply_flags(ds)

    # Determine which variables to bin (all float data vars except FLAG)
    bin_vars = [
        v for v in ds_flagged.data_vars
        if v != "FLAG"
        and np.issubdtype(ds_flagged[v].dtype, np.floating)
    ]

    binned_casts = []

    for i, _ in _iter_casts(ds_flagged, "Bin profiles"):
        valid, n = _cast_slice(ds_flagged, i)
        if n < 2:
            binned_casts.append(None)
            continue

        # Build a DataFrame for this cast
        df_data = {}
        for var in bin_vars:
            df_data[var] = ds_flagged[var].values[i, valid]
        if "FLAG" in ds.data_vars:
            df_data["flag"] = ds["FLAG"].values[i, valid]
        df = pd.DataFrame(df_data)

        try:
            binned = proc.bin_average(
                df, bin_var, bin_size,
                include_scan_count=False,
                min_scans=min_scans,
                exclude_bad_scans=exclude_bad_scans,
                interpolate=interpolate,
                cast_type=cast_type,
                include_surface_bin=include_surface_bin,
                flag_value=_FLAG_VALUE,
            )
        except Exception as e:
            import warnings
            warnings.warn(
                f"bin_profiles: cast {i} failed ({e}), skipping.",
                stacklevel=2,
            )
            binned_casts.append(None)
            continue

        # Drop the flag column from the binned result
        if "flag" in binned.columns:
            binned = binned.drop(columns=["flag"])

        binned_casts.append(binned)

    # Find the union pressure grid
    all_pres = sorted(set(
        p for b in binned_casts if b is not None
        for p in b[bin_var].values
    ))

    # Build output Dataset
    n_time = ds.sizes["TIME"]
    n_pres = len(all_pres)
    pres_arr = np.array(all_pres)

    out_vars = {}
    for var in bin_vars:
        if var == bin_var:
            continue
        arr = np.full((n_time, n_pres), np.nan)
        for i, binned in enumerate(binned_casts):
            if binned is None or var not in binned.columns:
                continue
            for j, p in enumerate(pres_arr):
                row = binned[binned[bin_var] == p]
                if len(row) == 1:
                    arr[i, j] = row[var].values[0]
        out_vars[var] = xr.DataArray(
            arr, dims=["TIME", bin_var],
            attrs=ds[var].attrs if var in ds else {},
        )

    coords = {
        bin_var: (bin_var, pres_arr,
                  ds[bin_var].attrs if bin_var in ds else {}),
        "TIME": ("TIME", ds.TIME.values),
    }
    if "STATION" in ds.coords:
        coords["STATION"] = ("TIME", ds.STATION.values)

    out = xr.Dataset(out_vars, coords=coords)
    out.attrs = {k: v for k, v in ds.attrs.items()}

    return _record(out, "bin_profiles", {
        "bin_size": bin_size,
        "bin_var": bin_var,
        "cast_type": cast_type.name,
        "min_scans": min_scans,
        "include_surface_bin": include_surface_bin,
    })