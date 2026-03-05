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

# seabirdscientific.processing.bin_average uses np.concat which requires numpy>=2.0
if tuple(int(x) for x in np.__version__.split(".")[:2]) < (2, 0):
    raise ImportError(
        f"kval.file.sbe_process requires numpy>=2.0 "
        f"(found {np.__version__}). Please upgrade: pip install 'numpy>=2.0'"
    )

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
# Loop edit — numpy reimplementation of seabirdscientific internals
#
# The functions below (_np_find_depth_peaks, _np_flag_by_minima_maxima,
# _np_min_velocity_mask, _np_mean_speed_percent_mask, _np_loop_edit_pressure)
# are algorithmically identical to the corresponding private functions in
# seabirdscientific.processing (_find_depth_peaks, _flag_by_minima_maxima,
# _min_velocity_mask, _mean_speed_percent_mask, loop_edit_pressure).
#
# The only difference is that the pure-Python loops in the original have
# been replaced with numpy operations for performance. At 24 Hz with ~15k
# scans per cast the originals take O(minutes); these take O(milliseconds).
#
# If seabirdscientific updates its algorithm, these must be updated to match.
# seabirdscientific version at time of writing: checked against source in
# ~/miniforge3/envs/oyv/lib/python3.13/site-packages/seabirdscientific/processing.py
# ---------------------------------------------------------------------------

def _np_find_depth_peaks(
    depth: np.ndarray,
    flag: np.ndarray,
    remove_surface_soak: bool,
    flag_value: float,
    min_soak_depth: float,
    max_soak_depth: float,
) -> tuple[int, int]:
    """Numpy reimplementation of seabirdscientific.processing._find_depth_peaks."""
    good = flag != flag_value
    idx = np.arange(len(depth))

    if remove_surface_soak:
        in_soak = good & (depth > min_soak_depth) & (depth < max_soak_depth)
        if not in_soak.any():
            raise ValueError(
                f"No scans found between min_soak_depth={min_soak_depth} and "
                f"max_soak_depth={max_soak_depth} — cast may be too shallow or "
                f"never reached soak depth. Max depth: {np.nanmax(depth):.1f} m."
            )
        min_soak_depth_n = int(np.argmax(in_soak))
    else:
        min_soak_depth_n = 0

    past_soak = good & (depth > max_soak_depth)
    if not past_soak.any():
        raise ValueError(
            f"No scans found deeper than max_soak_depth={max_soak_depth} — "
            f"cast may be too shallow. Max depth: {np.nanmax(depth):.1f} m."
        )
    max_soak_depth_n = int(np.argmax(past_soak))

    # minimum depth index in [min_soak_depth_n, max_soak_depth_n)
    in_window = good & (idx >= min_soak_depth_n) & (idx < max_soak_depth_n)
    if not in_window.any():
        raise ValueError(
            f"No good scans in soak window [{min_soak_depth_n}, {max_soak_depth_n})."
        )
    window_depth = np.where(in_window, depth, np.inf)
    min_depth_n = int(np.argmin(window_depth))

    if min_depth_n == len(depth) - 1:
        max_depth_n = -1
    else:
        global_max = np.nanmax(depth)
        candidates = np.where((depth == global_max) & (idx > min_depth_n))[0]
        if len(candidates) == 0:
            raise ValueError(
                f"Could not find global depth maximum after scan {min_depth_n}."
            )
        max_depth_n = int(candidates[0])

    return min_depth_n, max_depth_n


def _np_min_velocity_mask(
    depth: np.ndarray,
    interval: float,
    min_velocity: float,
    domain_start: int,
    domain_end: int,
    is_upcast: bool,
) -> np.ndarray:
    """Numpy reimplementation of seabirdscientific.processing._min_velocity_mask."""
    sign = -1 if is_upcast else 1
    mask0 = sign * np.diff(depth, prepend=depth[0]) / interval >= min_velocity
    mask1 = sign * np.diff(depth, append=depth[-1]) / interval >= min_velocity
    mask = mask0 & mask1
    mask[:domain_start] = False
    mask[domain_end:] = False
    return mask


def _np_mean_speed_percent_mask(
    depth: np.ndarray,
    interval: float,
    min_velocity: float,
    mean_speed_percent: float,
    domain_start: int,
    domain_end: int,
    is_upcast: bool,
    diff_length: int,
) -> np.ndarray:
    """Numpy reimplementation of seabirdscientific.processing._mean_speed_percent_mask."""
    sign = -1 if is_upcast else 1
    mask0 = sign * np.diff(depth[0:diff_length]) / interval > min_velocity
    mask1 = sign * np.diff(depth[1:diff_length + 1]) / interval > min_velocity
    first_window_mask = np.concatenate([[False], (mask0 & mask1)])

    mean_speed = sign * (depth[diff_length:] - depth[:-diff_length]) / diff_length / interval
    speed = sign * np.diff(depth[diff_length:], prepend=depth[diff_length - 1]) / interval

    mask = np.concatenate((
        first_window_mask,
        speed > (mean_speed * mean_speed_percent / 100.0),
    ))
    mask[:domain_start] = False
    mask[domain_end:] = False
    return mask


def _np_flag_by_minima_maxima(
    depth: np.ndarray,
    flag: np.ndarray,
    min_depth_n: int,
    max_depth_n: int,
    flag_value: float,
) -> None:
    """Numpy reimplementation of seabirdscientific.processing._flag_by_minima_maxima.

    Flags data that does not exceed the most recent valid local minima/maxima.
    Modifies flag in-place, identical behaviour to the original.
    """
    # Downcast phase: track running maximum from min_depth_n to max_depth_n
    # Upcast phase: track running minimum from max_depth_n onwards
    # Any scan that doesn't exceed the running extremum gets flagged.
    # The original iterates forward once — we replicate that exactly.
    local_max = -10000.0
    local_min = 10000.0

    for n in range(len(depth)):
        d = depth[n]
        if flag[n] == flag_value:
            continue
        if n >= max_depth_n and d < local_min:
            local_min = d
        elif n >= min_depth_n and d > local_max:
            local_max = d
        else:
            flag[n] = flag_value


def _np_loop_edit_pressure(
    pressure: np.ndarray,
    latitude: float,
    flag: np.ndarray,
    sample_interval: float,
    min_velocity_type: MinVelocityType,
    min_velocity: float,
    window_size: float,
    mean_speed_percent: float,
    remove_surface_soak: bool,
    min_soak_depth: float,
    max_soak_depth: float,
    use_deck_pressure_offset: bool,
    exclude_flags: bool,
    flag_value: float = _FLAG_VALUE,
) -> np.ndarray:
    """Numpy reimplementation of seabirdscientific.processing.loop_edit_pressure.

    Algorithmically identical to the original. Converts pressure to depth
    using seabirdscientific.conversion.depth_from_pressure, then delegates
    to the numpy-vectorised internal helpers above.
    """
    from seabirdscientific.conversion import depth_from_pressure

    depth = depth_from_pressure(pressure, latitude)

    if not exclude_flags:
        flag[:] = 0.0

    if use_deck_pressure_offset:
        min_soak_depth -= depth[0]
        max_soak_depth -= depth[0]

    min_depth_n, max_depth_n = _np_find_depth_peaks(
        depth, flag, remove_surface_soak, flag_value, min_soak_depth, max_soak_depth,
    )

    if min_velocity_type == MinVelocityType.FIXED:
        downcast_mask = _np_min_velocity_mask(
            depth, sample_interval, min_velocity, min_depth_n, max_depth_n + 1, False,
        )
        upcast_mask = _np_min_velocity_mask(
            depth, sample_interval, min_velocity, max_depth_n, len(depth), True,
        )
    elif min_velocity_type == MinVelocityType.PERCENT:
        diff_length = int(window_size / sample_interval)
        downcast_mask = _np_mean_speed_percent_mask(
            depth, sample_interval, min_velocity, mean_speed_percent,
            min_depth_n, max_depth_n, False, diff_length,
        )
        upcast_mask = _np_mean_speed_percent_mask(
            depth, sample_interval, min_velocity, mean_speed_percent,
            max_depth_n, len(depth), True, diff_length,
        )
    else:
        raise ValueError(f"Unknown MinVelocityType: {min_velocity_type}")

    flag[~downcast_mask & ~upcast_mask] = flag_value
    _np_flag_by_minima_maxima(depth, flag, min_depth_n, max_depth_n, flag_value)

    cast = np.zeros(len(flag), dtype=np.int8)
    cast[downcast_mask] = -1
    cast[upcast_mask] = 1
    return cast


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

    Uses _np_loop_edit_pressure — a numpy reimplementation of
    seabirdscientific.processing.loop_edit_pressure with identical
    algorithm but vectorised for performance on high-frequency raw data.
    See the _np_* functions above for details.

    Parameters
    ----------
    ds : xr.Dataset
        Multi-cast Dataset. Must contain PRES. LATITUDE used if present,
        otherwise defaults to 0°.
    min_velocity : float
        Minimum descent velocity in m/s. Default 0.25.
    min_velocity_type : MinVelocityType
        FIXED (default) or PERCENT.
    window_size : float
        Window for mean speed calculation in seconds. Default 300.0.
    mean_speed_percent : float
        Percent of mean speed threshold (PERCENT type only). Default 20.
    remove_surface_soak : bool
        Flag scans before min_soak_depth is reached. Default True.
    min_soak_depth : float
        Minimum depth (m) before downcast starts. Default 3.0.
    max_soak_depth : float
        Maximum depth (m) of soak region. Default 20.0.
    use_deck_pressure_offset : bool
        Offset soak depths by first pressure value. Default False.
    exclude_flags : bool
        Preserve existing flags. Default True.

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
            lat = float(np.nanmean(ds["LATITUDE"].values[i, valid]))
        else:
            lat = 0.0

        try:
            if not np.any(np.isfinite(pres)):
                raise ValueError(
                    "pressure data is all NaN or Inf — instrument may have "
                    "malfunctioned or there is no in-water data in this cast."
                )
            _ = _np_loop_edit_pressure(
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
        full_flags[i, valid] = np.where(
            flags == _FLAG_VALUE, _FLAG_VALUE, full_flags[i, valid]
        )
        ds["FLAG"] = xr.DataArray(full_flags, dims=ds["FLAG"].dims,
                                  attrs=ds["FLAG"].attrs)

    if failed:
        msgs = []
        for s, e in failed:
            if "all NaN or Inf" in e:
                reason = (
                    f"pressure data is all NaN or Inf — possible instrument "
                    f"malfunction or no in-water data. Inspect the raw file "
                    f"before using this cast."
                )
            elif "too shallow" in e or "max_soak_depth" in e:
                reason = (
                    f"cast did not reach {max_soak_depth} m — may be a "
                    f"shallow test cast or aborted deployment."
                )
            elif "min_soak_depth" in e or "soak" in e.lower():
                reason = (
                    f"cast never passed through the soak depth window "
                    f"({min_soak_depth}–{max_soak_depth} m) — check that "
                    f"the cast reached water and was lowered normally."
                )
            else:
                reason = (
                    f"something looks wrong with the pressure data — possible "
                    f"instrument malfunction or no in-water data. "
                    f"Inspect the raw file. (Internal: {e})"
                )
            msgs.append(f"  {s}: {reason}")
        warnings.warn(
            f"loop_edit: {len(failed)} cast(s) skipped — no flags applied "
            f"for these casts:\n" + "\n".join(msgs),
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
    direction: str = "downcast",
    min_scans: int = 1,
    exclude_bad_scans: bool = True,
    interpolate: bool = False,
    include_surface_bin: bool = False,
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
    direction : str
        Which part of the cast to bin: 'downcast' (default), 'upcast',
        or 'both'.
    min_scans : int
        Minimum scans per bin to include. Default 1.
    exclude_bad_scans : bool
        Exclude loop-edit-flagged scans from bins. Default True.
    interpolate : bool
        Interpolate to bin midpoints (SBE style). Default False.
    include_surface_bin : bool
        Include a surface bin at the start of the downcast. Default False.

    Returns
    -------
    xr.Dataset
        Dataset with dims (TIME, PRES). TIME and STATION coords preserved.
    """
    _direction_map = {
        "downcast": CastType.DOWNCAST,
        "upcast": CastType.UPCAST,
        "both": CastType.BOTH,
    }
    if direction not in _direction_map:
        raise ValueError(
            f"bin_profiles: direction must be 'downcast', 'upcast', or 'both' "
            f"(got '{direction}')."
        )
    cast_type = _direction_map[direction]
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
    stations = list(ds.STATION.values) if "STATION" in ds.coords else \
               [str(i) for i in range(ds.sizes["TIME"])]

    for i, station in _iter_casts(ds_flagged, "Bin profiles"):
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
            import logging
            sbs_logger = logging.getLogger("seabirdscientific.processing")
            prev_level = sbs_logger.level
            sbs_logger.setLevel(logging.CRITICAL)
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
            sbs_logger.setLevel(prev_level)
        except Exception as e:
            import warnings
            warnings.warn(
                f"bin_profiles: skipping cast '{station}' — binning failed. "
                f"This may indicate a problem with the raw data (e.g. instrument "
                f"malfunction, aborted cast, or corrupt file). Inspect the cast "
                f"with ds.sel(TIME=...) before proceeding, or exclude it with "
                f"drop_cast(ds, '{station}'). (Internal error: {e})",
                stacklevel=2,
            )
            binned_casts.append(None)
            continue

        # Drop the flag column from the binned result
        if "flag" in binned.columns:
            binned = binned.drop(columns=["flag"])

        # Snap bin_var to clean bin-centre grid (0.5, 1.5, 2.5 ... for 1 dbar)
        # bin_average returns mean pressure per bin which varies slightly from
        # the centre. We preserve the true mean as PRES_mean and use the clean
        # grid for the coordinate so the output Dataset has a regular axis.
        half = bin_size / 2.0
        bin_centres = np.round(
            (binned[bin_var] - half) / bin_size
        ) * bin_size + half
        binned["PRES_mean"] = binned[bin_var].values
        binned[bin_var] = bin_centres

        # Drop duplicate bin centres (can occur at surface with irregular bins)
        binned = binned.drop_duplicates(subset=[bin_var], keep="first")

        binned_casts.append(binned)

    # Build output Dataset — reindex each cast to the union pressure grid
    # then stack, rather than querying per pressure level
    all_pres = sorted(set(
        p for b in binned_casts if b is not None
        for p in b[bin_var].values
    ))
    pres_arr = np.array(all_pres)
    n_time = ds.sizes["TIME"]
    n_pres = len(all_pres)

    all_out_vars = bin_vars + (
        ["PRES_mean"] if any(
            b is not None and "PRES_mean" in b.columns
            for b in binned_casts
        ) else []
    )

    out_vars = {}
    for var in all_out_vars:
        if var == bin_var:
            continue
        arr = np.full((n_time, n_pres), np.nan)
        for i, b in enumerate(binned_casts):
            if b is None or var not in b.columns:
                continue
            b_indexed = b.set_index(bin_var)[var].reindex(pres_arr)
            arr[i, :] = b_indexed.values
        attrs = ds[var].attrs if var in ds else {}
        if var == "PRES_mean":
            attrs = {
                "long_name": "mean pressure in bin",
                "units": ds[bin_var].attrs.get("units", "dbar"),
                "comment": (
                    "actual mean pressure of scans averaged into each bin; "
                    "the PRES coordinate is the bin centre"
                ),
            }
        out_vars[var] = xr.DataArray(arr, dims=["TIME", bin_var], attrs=attrs)

    # Position variables — flatten to per-cast mean and store as coords
    _pos_vars = [v for v in ("LATITUDE", "LONGITUDE") if v in out_vars]
    pos_coords = {}
    for var in _pos_vars:
        attrs = dict(ds[var].attrs) if var in ds else {}
        attrs["comment"] = "mean value over the downcast profile"
        pos_coords[var] = (
            "TIME",
            np.nanmean(out_vars.pop(var).values, axis=1),
            attrs,
        )

    coords = {
        bin_var: (bin_var, pres_arr,
                  ds[bin_var].attrs if bin_var in ds else {}),
        "TIME": ("TIME", ds.TIME.values),
        **pos_coords,
    }
    if "STATION" in ds.coords:
        coords["STATION"] = ("TIME", ds.STATION.values)

    out = xr.Dataset(out_vars, coords=coords)
    out.attrs = {k: v for k, v in ds.attrs.items()}

    # Drop casts where all data variables are NaN (e.g. bad casts that
    # failed binning) — these add nothing and clutter the output
    data_vars = [v for v in out.data_vars if v != "PRES_mean"]
    if data_vars:
        any_valid = np.zeros(out.sizes["TIME"], dtype=bool)
        for var in data_vars:
            any_valid |= np.any(np.isfinite(out[var].values), axis=1)
        if not any_valid.all():
            dropped = list(out.STATION.values[~any_valid]) \
                if "STATION" in out.coords else list(np.where(~any_valid)[0])
            import warnings
            warnings.warn(
                f"bin_profiles: dropping {(~any_valid).sum()} cast(s) with no "
                f"valid data after binning: {dropped}. Use drop_cast() before "
                f"bin_profiles() to suppress this warning.",
                stacklevel=2,
            )
            out = out.isel(TIME=any_valid)

    return _record(out, "bin_profiles", {
        "bin_size": bin_size,
        "bin_var": bin_var,
        "direction": direction,
        "min_scans": min_scans,
        "include_surface_bin": include_surface_bin,
    })

# ---------------------------------------------------------------------------
# Drop cast
# ---------------------------------------------------------------------------

def drop_cast(
    ds: xr.Dataset,
    station: Union[str, List[str]],
) -> xr.Dataset:
    """
    Mark one or more casts as entirely bad by setting all FLAG values to
    the bad flag value.

    The cast data is retained in the Dataset — use apply_flags() or
    bin_profiles() to exclude the flagged scans from output.

    Parameters
    ----------
    ds : xr.Dataset
        Multi-cast Dataset with dims (TIME, scan_count).
    station : str or list of str
        STATION label(s) to flag, e.g. '007' or ['007', '006'].

    Returns
    -------
    xr.Dataset
        Dataset with FLAG set to bad for the specified cast(s).
    """
    ds = _ensure_flag(ds)
    ds = ds.copy()

    if isinstance(station, str):
        station = [station]

    if "STATION" not in ds.coords:
        raise ValueError("drop_cast requires a STATION coordinate.")

    stations = list(ds.STATION.values)
    full_flags = ds["FLAG"].values.copy()

    dropped = []
    for s in station:
        if s not in stations:
            import warnings
            warnings.warn(f"drop_cast: station '{s}' not found.", stacklevel=2)
            continue
        i = stations.index(s)
        full_flags[i, :] = _FLAG_VALUE
        dropped.append(s)

    ds["FLAG"] = xr.DataArray(full_flags, dims=ds["FLAG"].dims,
                               attrs=ds["FLAG"].attrs)

    return _record(ds, "drop_cast", {"stations": dropped})