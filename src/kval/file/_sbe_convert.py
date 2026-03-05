"""
kval.file._sbe_convert

Converts raw SBE sensor outputs (Hz, counts, volts) to physical units
(°C, S/m, dbar, μmol/kg, mg/m³, etc.) using seabirdscientific.conversion
and seabirdscientific.cal_coefficients.

This module is called internally by sbe_hex.parse_hex() — it is not
part of the public API.

Conversion dependency order (must be respected):
    1. Temperature   — independent
    2. Pressure      — independent (uses internal temp-comp word)
    3. Conductivity  — needs temperature + pressure
    4. Voltage sensors (oxygen, fluorescence, etc.) — may need T, P, C

Supported sensors
-----------------
Core (always converted):
    TemperatureSensor (SBE3/SBE37)   → TEMP [°C ITS-90]
    ConductivitySensor               → CNDC [S/m]
    PressureSensor (digiquartz)      → PRES [dbar]
    PressureSensor (strain gauge)    → PRES [dbar]

Voltage sensors (converted when voltage data is present):
    OxygenSensor / SBE43             → DOXY [ml/l]
    OxygenSensor / SBE63             → DOXY [ml/l]  (SBE37 ODO)
    FluoroWetlabECO_AFL_FL_Sensor   → CHLA [mg/m³]
    FluoroWetlabWetstarSensor        → CHLA [mg/m³]
    FluoroWetlabCDOM_Sensor          → CDOM [ppb]
    WET_LabsCStar                    → TRANSMITTANCE [%]
    AltimeterSensor                  → ALT  [m]
    PAR_BiosphericalLicorChelseaSensor → PAR [μE/m²/s]
    SPAR_Sensor                      → SPAR [μE/m²/s]
    FluoroSeapointSensor             → SEAPOINT_FL [mg/m³]

Sensors without a known conversion (raw voltage stored as-is):
    Any other type — warning issued, variable kept with '_raw' suffix.
"""

from __future__ import annotations

import warnings
from typing import Optional
import numpy as np
import xarray as xr

import seabirdscientific.conversion as conv
import seabirdscientific.cal_coefficients as cc


# ---------------------------------------------------------------------------
# Variable name and unit definitions
# ---------------------------------------------------------------------------

# Maps the raw field name produced by the hex parser to the CF-style
# output variable name and its standard units.
_RAW_TO_OUTPUT: dict[str, tuple[str, str]] = {
    # SBE911 primary
    "temperature_primary_raw":     ("TEMP",      "degree_Celsius"),
    "conductivity_primary_raw":    ("CNDC",      "S m-1"),
    "pressure_raw":                ("PRES",      "dbar"),
    # SBE911 secondary
    "temperature_secondary_raw":   ("TEMP2",     "degree_Celsius"),
    "conductivity_secondary_raw":  ("CNDC2",     "S m-1"),
    # SBE37
    "temperature_raw":             ("TEMP",      "degree_Celsius"),
    "conductivity_raw":            ("CNDC",      "S m-1"),
    # Shared pressure temp-comp (consumed internally, not output)
    "pressure_temp_comp_raw":      ("_discard",  ""),
    # SBE37 SBE63 oxygen intermediates (consumed, replaced by DOXY)
    "sbe63_phase_raw":             ("_discard",  ""),
    "sbe63_temperature_raw":       ("_discard",  ""),
}

# Voltage sensor type → (output variable name, units)
_VOLT_SENSOR_OUTPUT: dict[str, tuple[str, str]] = {
    "OxygenSensor":                      ("DOXY",          "ml l-1"),
    "FluoroWetlabECO_AFL_FL_Sensor":    ("CHLA",          "mg m-3"),
    "FluoroWetlabWetstarSensor":         ("CHLA",          "mg m-3"),
    "FluoroWetlabCDOM_Sensor":           ("CDOM",          "ppb"),
    "WET_LabsCStar":                     ("TRANSMITTANCE", "%"),
    "AltimeterSensor":                   ("ALT",           "m"),
    "PAR_BiosphericalLicorChelseaSensor": ("PAR",          "microE m-2 s-1"),
    "SPAR_Sensor":                       ("SPAR",          "microE m-2 s-1"),
    "FluoroSeapointSensor":              ("SEAPOINT_FL",   "mg m-3"),
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def convert_raw_dataset(ds: xr.Dataset, xmlcon_config: dict) -> xr.Dataset:
    """
    Convert a raw xr.Dataset (Hz/counts/volts) to physical units in-place.

    Called by sbe_hex.parse_hex() after building the raw Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset from parse_hex() — variables have _raw suffix and
        contain Hz / integer counts / voltages.
    xmlcon_config : dict
        Parsed xmlcon configuration from parse_xmlcon().

    Returns
    -------
    xr.Dataset
        New dataset with physical-unit variables replacing the raw ones.
        Variables that could not be converted are kept with their original
        name and a warning is issued.
    """
    sensors    = xmlcon_config["sensors"]
    instrument = xmlcon_config["instrument"]
    family     = ds.attrs.get("instrument_family", "")

    # Work on a copy so we don't mutate the input
    ds_out = ds.copy()

    # ------------------------------------------------------------------
    # Step 1: Core T / P / C  (order matters)
    # ------------------------------------------------------------------
    if family == "911":
        ds_out = _convert_911_core(ds_out, sensors, instrument)
    elif family == "37":
        ds_out = _convert_37_core(ds_out, sensors)

    # ------------------------------------------------------------------
    # Step 2: Voltage sensors (use converted T/P/C where needed)
    # ------------------------------------------------------------------
    ds_out = _convert_voltage_sensors(ds_out, sensors)

    # ------------------------------------------------------------------
    # Step 3: SBE63 oxygen (SBE37 ODO) — needs converted T from step 1
    # ------------------------------------------------------------------
    if "sbe63_phase_raw" in ds_out and "sbe63_temperature_raw" in ds_out:
        ds_out = _convert_sbe63(ds_out, sensors)

    # ------------------------------------------------------------------
    # Step 4: Surface PAR (already in counts/819 from hex parser;
    #         convert using SPAR sensor coefficients if present)
    # ------------------------------------------------------------------
    if "surface_par_raw" in ds_out:
        ds_out = _convert_surface_par(ds_out, sensors)

    # ------------------------------------------------------------------
    # Step 5: Drop internal bookkeeping variables
    # ------------------------------------------------------------------
    to_drop = [v for v in ds_out.data_vars
               if v.endswith("_raw") or v in ("status", "data_integrity",
                                               "pressure_temp_comp_raw")]
    ds_out = ds_out.drop_vars([v for v in to_drop if v in ds_out])

    return ds_out


# ---------------------------------------------------------------------------
# SBE911 core conversion
# ---------------------------------------------------------------------------

def _convert_911_core(ds: xr.Dataset, sensors: list, instrument: dict) -> xr.Dataset:
    """Convert T1, C1, P (and T2, C2 if present) for SBE911."""

    sample_interval = _get_sample_interval_911(instrument)

    # ── Primary temperature ──────────────────────────────────────────
    t1_sensor = _find_sensor(sensors, "TemperatureSensor", nth=0)
    if t1_sensor and "temperature_primary_raw" in ds:
        freq = ds["temperature_primary_raw"].values
        coefs = _make_temp_freq_coefs(t1_sensor)
        temp1 = conv.convert_temperature_frequency(freq, coefs)
        temp1 = _apply_slope_offset(temp1, t1_sensor)
        ds = _replace_var(ds, "temperature_primary_raw", "TEMP",
                          temp1, "degree_Celsius", t1_sensor)

    # ── Primary pressure ─────────────────────────────────────────────
    p_sensor = _find_sensor(sensors, "PressureSensor", nth=0)
    if p_sensor and "pressure_raw" in ds:
        freq   = ds["pressure_raw"].values
        tcomp  = ds["pressure_temp_comp_raw"].values if "pressure_temp_comp_raw" in ds else np.zeros_like(freq)
        coefs  = _make_pressure_digiquartz_coefs(p_sensor)
        pres   = conv.convert_pressure_digiquartz(freq, tcomp, coefs,
                                                  units="dbar",
                                                  sample_interval=sample_interval)
        p_offset = float(p_sensor.get("coefficients", {}).get("Offset", 0.0))
        if p_offset != 0.0:
            pres = pres + p_offset
        ds = _replace_var(ds, "pressure_raw", "PRES", pres, "dbar", p_sensor)

    # ── Primary conductivity (needs TEMP + PRES) ──────────────────────
    c1_sensor = _find_sensor(sensors, "ConductivitySensor", nth=0)
    if c1_sensor and "conductivity_primary_raw" in ds and "TEMP" in ds and "PRES" in ds:
        # freq_from_3bytes gives Hz directly. convert_conductivity divides by
        # 1000 internally to get kHz. scalar=0.1 applies the /10 from the
        # SBE4 calibration equation (C = (g+hf²+if³+jf⁴)/10 * (1+dt+ep)).
        freq   = ds["conductivity_primary_raw"].values
        temp   = ds["TEMP"].values
        pres   = ds["PRES"].values
        coefs  = _make_conductivity_coefs(c1_sensor)
        cndc1  = conv.convert_conductivity(freq, temp, pres, coefs, scalar=0.1)
        cndc1  = _apply_slope_offset(cndc1, c1_sensor)
        ds = _replace_var(ds, "conductivity_primary_raw", "CNDC",
                          cndc1, "S m-1", c1_sensor)

    # ── Secondary temperature ─────────────────────────────────────────
    t2_sensor = _find_sensor(sensors, "TemperatureSensor", nth=1)
    if t2_sensor and "temperature_secondary_raw" in ds:
        freq  = ds["temperature_secondary_raw"].values
        coefs = _make_temp_freq_coefs(t2_sensor)
        temp2 = conv.convert_temperature_frequency(freq, coefs)
        temp2 = _apply_slope_offset(temp2, t2_sensor)
        ds = _replace_var(ds, "temperature_secondary_raw", "TEMP2",
                          temp2, "degree_Celsius", t2_sensor)

    # ── Secondary conductivity (needs TEMP2 + PRES) ───────────────────
    c2_sensor = _find_sensor(sensors, "ConductivitySensor", nth=1)
    if c2_sensor and "conductivity_secondary_raw" in ds and "TEMP2" in ds and "PRES" in ds:
        freq   = ds["conductivity_secondary_raw"].values
        temp   = ds["TEMP2"].values
        pres   = ds["PRES"].values
        coefs  = _make_conductivity_coefs(c2_sensor)
        cndc2  = conv.convert_conductivity(freq, temp, pres, coefs, scalar=0.1)
        cndc2  = _apply_slope_offset(cndc2, c2_sensor)
        ds = _replace_var(ds, "conductivity_secondary_raw", "CNDC2",
                          cndc2, "S m-1", c2_sensor)

    return ds


# ---------------------------------------------------------------------------
# SBE37 core conversion
# ---------------------------------------------------------------------------

def _convert_37_core(ds: xr.Dataset, sensors: list) -> xr.Dataset:
    """Convert T, C, P for SBE37 (uses TemperatureCoefficients, not frequency)."""

    # ── Temperature ───────────────────────────────────────────────────
    t_sensor = _find_sensor(sensors, "TemperatureSensor")
    if t_sensor and "temperature_raw" in ds:
        counts = ds["temperature_raw"].values
        coefs  = _make_temp_counts_coefs(t_sensor)
        temp   = conv.convert_temperature(counts, coefs)
        ds = _replace_var(ds, "temperature_raw", "TEMP",
                          temp, "degree_Celsius", t_sensor)

    # ── Pressure (strain gauge) ───────────────────────────────────────
    p_sensor = _find_sensor(sensors, "PressureSensor")
    if p_sensor and "pressure_raw" in ds:
        counts = ds["pressure_raw"].values
        tcomp  = ds["pressure_temp_comp_raw"].values if "pressure_temp_comp_raw" in ds else np.zeros_like(counts)
        coefs  = _make_pressure_strain_coefs(p_sensor)
        pres   = conv.convert_pressure(counts, tcomp, coefs, units="dbar")
        ds = _replace_var(ds, "pressure_raw", "PRES", pres, "dbar", p_sensor)

    # ── Conductivity (needs TEMP + PRES) ─────────────────────────────
    c_sensor = _find_sensor(sensors, "ConductivitySensor")
    if c_sensor and "conductivity_raw" in ds and "TEMP" in ds and "PRES" in ds:
        counts = ds["conductivity_raw"].values
        temp   = ds["TEMP"].values
        pres   = ds["PRES"].values
        coefs  = _make_conductivity_coefs(c_sensor)
        cndc   = conv.convert_conductivity(counts, temp, pres, coefs)
        ds = _replace_var(ds, "conductivity_raw", "CNDC", cndc, "S m-1", c_sensor)

    return ds


# ---------------------------------------------------------------------------
# SBE63 optical oxygen (SBE37 ODO)
# ---------------------------------------------------------------------------

def _convert_sbe63(ds: xr.Dataset, sensors: list) -> xr.Dataset:
    """Convert SBE63 phase + thermistor → dissolved oxygen [ml/l]."""

    o2_sensor = _find_sensor(sensors, "OxygenSensor")
    if o2_sensor is None:
        return ds

    c = o2_sensor.get("coefficients", {})

    try:
        o2_coefs = cc.Oxygen63Coefficients(
            a0=c["A0"], a1=c["A1"], a2=c["A2"],
            b0=c["B0"], b1=c["B1"],
            c0=c["C0"], c1=c["C1"], c2=c["C2"],
            e=c.get("pcor", 0.011),
        )
        therm_coefs = cc.Thermistor63Coefficients(
            ta0=c["TA0"], ta1=c["TA1"], ta2=c["TA2"], ta3=c["TA3"],
        )
    except KeyError as e:
        warnings.warn(
            f"SBE63 oxygen conversion skipped: missing coefficient {e}. "
            f"Raw phase/thermistor values retained.",
            stacklevel=3,
        )
        return ds

    phase   = ds["sbe63_phase_raw"].values
    therm_v = ds["sbe63_temperature_raw"].values
    pres    = ds["PRES"].values if "PRES" in ds else np.zeros(len(phase))
    # Salinity: compute from CNDC + TEMP + PRES if available, else 0
    sal = _compute_salinity(ds)

    doxy = conv.convert_sbe63_oxygen(
        raw_oxygen_phase=phase,
        thermistor=therm_v,
        pressure=pres,
        salinity=sal,
        coefs=o2_coefs,
        thermistor_coefs=therm_coefs,
        thermistor_units="volts",
    )

    ds = ds.drop_vars(["sbe63_phase_raw", "sbe63_temperature_raw"])
    ds["DOXY"] = xr.DataArray(
        doxy, dims=["scan"],
        attrs={**_base_attrs(o2_sensor),
               "units": "ml l-1",
               "long_name": "Dissolved oxygen (SBE63)",
               "comment": "Converted from SBE63 phase delay using seabirdscientific"},
    )
    return ds


# ---------------------------------------------------------------------------
# Voltage sensor conversion
# ---------------------------------------------------------------------------

def _convert_voltage_sensors(ds: xr.Dataset, sensors: list) -> xr.Dataset:
    """Convert all voltage channels that have a known sensor type."""

    # Build a map: volt field name → sensor dict
    volt_fields = {name: name for name in ds.data_vars
                   if name.startswith("volt_")}

    for field_name in list(volt_fields):
        if field_name not in ds:
            continue

        # Match this field back to a sensor by volt channel index
        # Field names look like: volt_0_oxygensensor, volt_2_fluorowetlab, etc.
        parts = field_name.split("_")
        if len(parts) < 2:
            continue
        try:
            volt_idx = int(parts[1])
        except ValueError:
            continue

        xmlcon_idx = 5 + volt_idx  # voltage sensors start at xmlcon index 5
        sensor = sensors[xmlcon_idx] if xmlcon_idx < len(sensors) else None
        if sensor is None or not sensor.get("in_use", False):
            continue

        sensor_type = sensor.get("type", "")
        volts = ds[field_name].values

        try:
            converted, out_name, units = _convert_one_voltage(
                sensor_type, volts, sensor, ds, volt_idx
            )
        except Exception as exc:
            warnings.warn(
                f"\nCould not convert voltage channel {volt_idx} "
                f"({sensor_type}):\n  {exc}\n"
                f"  Raw voltage kept as '{field_name}'.",
                stacklevel=3,
            )
            continue

        if converted is None:
            # Unknown sensor type — keep raw voltage with a warning
            warnings.warn(
                f"\nNo conversion available for sensor type '{sensor_type}' "
                f"on voltage channel {volt_idx}.\n"
                f"  Raw voltage kept as '{field_name}'.",
                stacklevel=3,
            )
            continue

        # Handle duplicate output names (e.g. two oxygen sensors → DOXY, DOXY2)
        if out_name in ds:
            out_name = out_name + "2"

        ds = ds.drop_vars([field_name])
        ds[out_name] = xr.DataArray(
            converted, dims=["scan"],
            attrs={**_base_attrs(sensor),
                   "units": units,
                   "volt_channel": volt_idx},
        )

    return ds


def _convert_one_voltage(
    sensor_type: str,
    volts: np.ndarray,
    sensor: dict,
    ds: xr.Dataset,
    volt_idx: int,
) -> tuple[Optional[np.ndarray], str, str]:
    """
    Convert a single voltage channel. Returns (array, output_name, units)
    or (None, '', '') if sensor type is unrecognised.
    """
    c = sensor.get("coefficients", {})

    if sensor_type == "OxygenSensor":
        # SBE43 dissolved oxygen
        temp = ds["TEMP"].values  if "TEMP"  in ds else np.zeros(len(volts))
        pres = ds["PRES"].values  if "PRES"  in ds else np.zeros(len(volts))
        sal  = _compute_salinity(ds)
        coefs = cc.Oxygen43Coefficients(
            soc=c["Soc"], v_offset=c["offset"], tau_20=c["Tau20"],
            a=c["A"], b=c["B"], c=c["C"], e=c["E"],
            d0=c["D0"], d1=c["D1"], d2=c["D2"],
            h1=c["H1"], h2=c["H2"], h3=c["H3"],
        )
        result = conv.convert_sbe43_oxygen(volts, temp, pres, sal, coefs)
        result = result * c.get("Slope", 1.0) + c.get("Offset", 0.0)
        return result, "DOXY", "ml l-1"

    elif sensor_type in ("FluoroWetlabECO_AFL_FL_Sensor",
                         "FluoroWetlabWetstarSensor"):
        coefs = cc.ECOCoefficients(
            slope=c["ScaleFactor"],
            offset=c["Vblank"],
        )
        result = conv.convert_eco(volts, coefs)
        return result, "CHLA", "mg m-3"

    elif sensor_type == "FluoroWetlabCDOM_Sensor":
        coefs = cc.ECOCoefficients(
            slope=c["ScaleFactor"],
            offset=c["Vblank"],
        )
        result = conv.convert_eco(volts, coefs)
        return result, "CDOM", "ppb"

    elif sensor_type == "AltimeterSensor":
        coefs = cc.AltimeterCoefficients(
            slope=c["ScaleFactor"],
            offset=c["Offset"],
        )
        result = conv.convert_altimeter(volts, coefs)
        return result, "ALT", "m"

    elif sensor_type == "PAR_BiosphericalLicorChelseaSensor":
        if c.get("B", 0) == 0:
            warnings.warn(
                f"\nPAR sensor (voltage channel {volt_idx}) has calibration "
                f"coefficient B=0 — sensor may not be calibrated. "
                f"Raw voltage kept as 'volt_{volt_idx}_par_raw'.",
                stacklevel=3,
            )
            return None, "", ""
        coefs = cc.PARCoefficients(
            im=c["CalibrationConstant"],
            a0=c["M"],
            a1=c["B"],
            multiplier=c.get("Multiplier", 1.0),
        )
        result = conv.convert_par_logarithmic(volts, coefs)
        return result, "PAR", "microE m-2 s-1"

    elif sensor_type == "WET_LabsCStar":
        # Seasoft linear equation: Transmission [%] = M * V + B
        # M and B are the linearization coefficients from the xmlcon.
        # (Not to be confused with beam attenuation; these are Seasoft's
        #  internal voltage-to-% mapping coefficients.)
        result = c["M"] * volts + c["B"]
        return result, "TRANSMITTANCE", "%"

    elif sensor_type == "FluoroSeapointSensor":
        # Simple linear: fluorescence = GainSetting * V + Offset
        gain   = float(c.get("GainSetting", 1.0))
        offset = float(c.get("Offset", 0.0))
        result = gain * volts + offset
        return result, "SEAPOINT_FL", "mg m-3"

    else:
        return None, "", ""


def _convert_surface_par(ds: xr.Dataset, sensors: list) -> xr.Dataset:
    """Convert the surface PAR word (already raw counts/819) if a SPAR sensor is configured.

    The xmlcon only carries ConversionFactor and RatioMultiplier for SPAR sensors,
    which is insufficient to use SPARCoefficients (which requires im/a0/a1).
    We use the direct linear equation instead:
        SPAR = ConversionFactor * (raw_counts / 819) * RatioMultiplier
    """
    spar_sensor = next(
        (s for s in sensors if s.get("type") == "SPAR_Sensor"), None
    )
    # surface_par_raw is already raw_counts/819 (done in hex parser)
    raw = ds["surface_par_raw"].values

    if spar_sensor is not None:
        c = spar_sensor.get("coefficients", {})
        cf = float(c.get("ConversionFactor", 1.0))
        rm = float(c.get("RatioMultiplier", 1.0))
        result = cf * raw * rm
        # If all values are identical the sensor was not connected — output NaN
        if np.all(result == result[0]):
            warnings.warn(
                "\nSPAR sensor output is constant — sensor may not be connected. "
                "Setting SPAR to NaN.",
                stacklevel=3,
            )
            result = np.full_like(result, np.nan)
        attrs = {**_base_attrs(spar_sensor), "units": "microE m-2 s-1",
                 "long_name": "Surface photosynthetically available radiation"}
    else:
        result = raw
        attrs  = {"units": "counts/819", "long_name": "Surface PAR (raw, no calibration)"}

    ds = ds.drop_vars(["surface_par_raw"])
    ds["SPAR"] = xr.DataArray(result, dims=["scan"], attrs=attrs)
    return ds


# ---------------------------------------------------------------------------
# Coefficient dataclass constructors
# ---------------------------------------------------------------------------

def _make_temp_freq_coefs(sensor: dict) -> cc.TemperatureFrequencyCoefficients:
    c = sensor["coefficients"]
    return cc.TemperatureFrequencyCoefficients(
        g=c["G"], h=c["H"], i=c["I"], j=c["J"], f0=c["F0"],
    )


def _make_temp_counts_coefs(sensor: dict) -> cc.TemperatureCoefficients:
    c = sensor["coefficients"]
    return cc.TemperatureCoefficients(
        a0=c["A0"], a1=c["A1"], a2=c["A2"], a3=c["A3"],
    )


def _make_conductivity_coefs(sensor: dict) -> cc.ConductivityCoefficients:
    c = sensor["coefficients"]
    return cc.ConductivityCoefficients(
        g=c["G"], h=c["H"], i=c["I"], j=c["J"],
        cpcor=c["CPcor"], ctcor=c["CTcor"],
        wbotc=c.get("WBOTC", 0.0),
    )


def _make_pressure_digiquartz_coefs(sensor: dict) -> cc.PressureDigiquartzCoefficients:
    c = sensor["coefficients"]
    return cc.PressureDigiquartzCoefficients(
        c1=c["C1"], c2=c["C2"], c3=c["C3"],
        d1=c["D1"], d2=c["D2"],
        t1=c["T1"], t2=c["T2"], t3=c["T3"], t4=c["T4"], t5=c["T5"],
        AD590M=c.get("AD590M"), AD590B=c.get("AD590B"),
    )


def _make_pressure_strain_coefs(sensor: dict) -> cc.PressureCoefficients:
    c = sensor["coefficients"]
    return cc.PressureCoefficients(
        pa0=c["PA0"], pa1=c["PA1"], pa2=c["PA2"],
        ptca0=c["PTCA0"], ptca1=c["PTCA1"], ptca2=c["PTCA2"],
        ptcb0=c["PTCB0"], ptcb1=c["PTCB1"], ptcb2=c["PTCB2"],
        ptempa0=c["PTEMPA0"], ptempa1=c["PTEMPA1"], ptempa2=c["PTEMPA2"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_slope_offset(values: np.ndarray, sensor: dict) -> np.ndarray:
    """Apply Seasoft Slope/Offset post-calibration adjustment.

    Seasoft convention: result = values * Slope + Offset
    Defaults are Slope=1.0, Offset=0.0 (no-op) if not present in xmlcon.
    """
    c = sensor.get("coefficients", {})
    slope  = float(c.get("Slope",  1.0))
    offset = float(c.get("Offset", 0.0))
    if slope == 1.0 and offset == 0.0:
        return values
    return values * slope + offset


# ---------------------------------------------------------------------------

def _find_sensor(
    sensors: list,
    sensor_type: str,
    nth: int = 0,
) -> Optional[dict]:
    """Return the nth sensor of a given type (0-indexed), or None."""
    matches = [s for s in sensors if s.get("type") == sensor_type and s.get("in_use", False)]
    return matches[nth] if nth < len(matches) else None


def _replace_var(
    ds: xr.Dataset,
    old_name: str,
    new_name: str,
    values: np.ndarray,
    units: str,
    sensor: dict,
) -> xr.Dataset:
    """Drop old_name, add new_name with converted values and metadata attrs."""
    old_attrs = dict(ds[old_name].attrs) if old_name in ds else {}
    ds = ds.drop_vars([old_name], errors="ignore")
    ds[new_name] = xr.DataArray(
        values, dims=["scan"],
        attrs={
            **old_attrs,
            "units": units,
            **_base_attrs(sensor),
        },
    )
    return ds


def _base_attrs(sensor: dict) -> dict:
    return {
        "sensor_type":      sensor.get("type", ""),
        "serial_number":    str(sensor.get("serial_number") or ""),
        "calibration_date": str(sensor.get("calibration_date") or ""),
    }


def _compute_salinity(ds: xr.Dataset) -> np.ndarray:
    """
    Compute practical salinity from CNDC, TEMP, PRES if all are present,
    otherwise return zeros (salinity=0 assumption).

    Used for oxygen conversion where salinity affects the solubility.
    """
    if all(v in ds for v in ("CNDC", "TEMP", "PRES")):
        import gsw
        # gsw.SP_from_C expects conductivity in mS/cm, not S/m
        cndc_mscm = ds["CNDC"].values * 10.0
        temp      = ds["TEMP"].values
        pres      = ds["PRES"].values
        return gsw.SP_from_C(cndc_mscm, temp, pres)
    else:
        n = len(next(iter(ds.data_vars.values())))
        return np.zeros(n)


def _get_sample_interval_911(instrument: dict) -> float:
    """Return the per-scan interval in seconds for a SBE911 (1/24 Hz default)."""
    scans_avg = instrument.get("scans_to_average", 1) or 1
    return float(scans_avg) / 24.0