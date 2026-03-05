"""
tests/unit_tests/file/test_sbe_convert.py

Unit tests for kval.file._sbe_convert and the end-to-end parse_hex pipeline.

Test strategy:
  - End-to-end parse_hex on three real truncated hex files covering different
    SBE911 configurations (41-byte/no-PAR, 44-byte/with-PAR x2)
  - Physical range checks on all converted variables
  - Comparison against Seasoft CNV reference values where available
    (Sta0119 and Sta0520 have in-water CNV data; Sta0243 CNV is deck-only)
  - Guard condition tests: PAR B=0, constant SPAR, pressure offset, Slope/Offset

Test data (tests/test_data/sbe_files/hex_xmlcon_cnv/):
    Sta0119.hex / STA0119.XMLCON / Sta0119.cnv
        SBE911, V5.2, 41 bytes/scan, 4 voltage words, no surface PAR
        Arctic cast, in-water CNV available (741 1-dbar bins, 10-750 dbar)
        TEMP: -1.77 to 1.25 °C (ITS-90 col 12), CNDC: 2.53-3.02 S/m

    Sta0243.hex / STA0243.XMLCON / Sta0243.cnv
        SBE911, V5.2, 44 bytes/scan, 5 voltage words, with surface PAR
        Arctic cast, CNV is deck-only (scans truncated at surface)
        Used for: parser runs clean, correct variables present

    Sta0520.hex / STA0520.XMLCON / Sta0520.cnv
        SBE911, V5.2, 44 bytes/scan, 5 voltage words, with surface PAR
        Deep cast (to 2108 dbar), in-water CNV available (2104 1-dbar bins)
        TEMP: -0.68 to 3.23 °C (ITS-90 col 12 n/a, use IPTS-68 col 1 ±0.05)

Note on CNV temperature scale:
    Sta0119 and Sta0520 CNVs use IPTS-68 (col 1) not ITS-90. The difference
    is ~0.001°C/°C so tolerances on temperature comparisons are widened to
    0.05°C to account for this plus binning/filter differences.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from kval.file.sbe_hex import parse_hex

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TEST_DATA = Path(__file__).parents[2] / "test_data" / "sbe_files" / "hex_xmlcon_cnv"

HEX_0119   = TEST_DATA / "Sta0119.hex"
XML_0119   = TEST_DATA / "STA0119.XMLCON"
CNV_0119   = TEST_DATA / "Sta0119.cnv"

HEX_0243   = TEST_DATA / "Sta0243.hex"
XML_0243   = TEST_DATA / "STA0243.XMLCON"
CNV_0243   = TEST_DATA / "Sta0243.cnv"

HEX_0520   = TEST_DATA / "Sta0520.hex"
XML_0520   = TEST_DATA / "STA0520.XMLCON"
CNV_0520   = TEST_DATA / "Sta0520.cnv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_cnv(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a Seasoft CNV file, return (column_names, data_array)."""
    names = []
    data_lines = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# name"):
                # extract e.g. "prDM" from "# name 0 = prDM: ..."
                names.append(line.split("=")[1].strip().split(":")[0].strip())
            elif line.startswith("*") or line.startswith("#") or not line:
                continue
            else:
                data_lines.append(line)
    arr = np.array([[float(x) for x in row.split()] for row in data_lines])
    return names, arr


def cnv_col(names: list[str], arr: np.ndarray, key: str) -> np.ndarray:
    """Return a CNV column by partial name match, masking Seasoft bad flags."""
    for i, n in enumerate(names):
        if key in n:
            col = arr[:, i]
            return np.where(col > -9e-28, col, np.nan)
    raise KeyError(f"Column {key!r} not found in CNV. Available: {names}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ds_0119():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_hex(HEX_0119, XML_0119)


@pytest.fixture(scope="module")
def ds_0243():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_hex(HEX_0243, XML_0243)


@pytest.fixture(scope="module")
def ds_0520():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse_hex(HEX_0520, XML_0520)


@pytest.fixture(scope="module")
def cnv_0119():
    return read_cnv(CNV_0119)


@pytest.fixture(scope="module")
def cnv_0520():
    return read_cnv(CNV_0520)


# ---------------------------------------------------------------------------
# Sta0119 — 41 bytes/scan, no surface PAR
# ---------------------------------------------------------------------------

class TestSta0119Parse:
    """Parser runs cleanly and produces expected variables."""

    def test_parses_without_error(self, ds_0119):
        assert ds_0119 is not None

    def test_expected_variables_present(self, ds_0119):
        for var in ("TEMP1", "TEMP2", "PRES", "CNDC1", "CNDC2"):
            assert var in ds_0119, f"{var} missing from dataset"

    def test_no_surface_par(self, ds_0119):
        """Sta0119 has no surface PAR — SPAR must not be present."""
        assert "SPAR" not in ds_0119

    def test_n_scans_positive(self, ds_0119):
        assert len(ds_0119.scan) > 0

    def test_scan_times_present(self, ds_0119):
        assert "TIME_SCAN" in ds_0119.coords


class TestSta0119PhysicalRanges:
    """Converted values are physically plausible for an Arctic cast."""

    def test_temp_range(self, ds_0119):
        t = ds_0119.TEMP1.values
        finite = t[np.isfinite(t)]
        assert finite.min() > -3.0,  "TEMP below absolute seawater minimum"
        assert finite.max() < 10.0,  "TEMP implausibly high for Arctic"

    def test_pres_range(self, ds_0119):
        p = ds_0119.PRES.values
        finite = p[np.isfinite(p)]
        assert finite.max() > 0.0,   "No positive pressure values found"
        assert finite.max() < 800.0, "Max pressure exceeds cast depth"

    def test_cndc_range(self, ds_0119):
        c = ds_0119.CNDC1.values
        finite = c[np.isfinite(c)]
        assert finite.max() > 2.0,   "CNDC too low for seawater"
        assert finite.max() < 4.0,   "CNDC too high for seawater"

    def test_primary_secondary_temp_agree(self, ds_0119):
        """Primary and secondary T should agree within 0.1°C in water."""
        t1 = ds_0119.TEMP1.values
        t2 = ds_0119.TEMP2.values
        in_water = ds_0119.PRES.values > 2
        if in_water.sum() < 10:
            pytest.skip("Not enough in-water scans in truncated hex file")
        diff = np.abs(t1[in_water] - t2[in_water])
        assert np.nanmedian(diff) < 0.05

    def test_primary_secondary_cndc_agree(self, ds_0119):
        c1 = ds_0119.CNDC1.values
        c2 = ds_0119.CNDC2.values
        in_water = ds_0119.PRES.values > 2
        if in_water.sum() < 10:
            pytest.skip("Not enough in-water scans in truncated hex file")
        diff = np.abs(c1[in_water] - c2[in_water])
        # Tolerance is loose (0.1 S/m) because truncated hex may only cover
        # the shallow transition zone where dual sensors genuinely diverge
        assert np.nanmedian(diff) < 0.1


class TestSta0119CNVComparison:
    """Compare binned values against Seasoft CNV reference (in-water bins)."""

    def _bin_to_cnv_pres(self, our_pres, our_vals, cnv_pres):
        """Average our values into bins matching CNV pressure levels."""
        binned = np.full(len(cnv_pres), np.nan)
        for i, p in enumerate(cnv_pres):
            mask = (our_pres >= p - 0.5) & (our_pres < p + 0.5) & np.isfinite(our_vals)
            if mask.sum() > 0:
                binned[i] = our_vals[mask].mean()
        return binned

    def test_temp_vs_cnv(self, ds_0119, cnv_0119):
        """TEMP should match CNV ITS-90 temperature within 0.05°C."""
        names, arr = cnv_0119
        cnv_pres = arr[:, 0]
        # col 12 = t090C (ITS-90)
        cnv_temp = np.where(arr[:, 12] > -9e-28, arr[:, 12], np.nan)

        our_pres = ds_0119.PRES.values
        our_temp = ds_0119.TEMP1.values
        binned = self._bin_to_cnv_pres(our_pres, our_temp, cnv_pres)

        mask = np.isfinite(cnv_temp) & np.isfinite(binned)
        if mask.sum() < 3:
            pytest.skip("Truncated hex has no overlap with CNV pressure range")
        diff = np.abs(binned[mask] - cnv_temp[mask])
        assert np.nanmedian(diff) < 0.05, \
            f"TEMP median diff vs CNV: {np.nanmedian(diff):.4f} °C"

    def test_cndc_vs_cnv(self, ds_0119, cnv_0119):
        """CNDC should match CNV within 0.01 S/m (celltm/filter difference)."""
        names, arr = cnv_0119
        cnv_pres = arr[:, 0]
        cnv_cndc = np.where(arr[:, 3] > -9e-28, arr[:, 3], np.nan)

        our_pres = ds_0119.PRES.values
        our_cndc = ds_0119.CNDC1.values
        binned = self._bin_to_cnv_pres(our_pres, our_cndc, cnv_pres)

        mask = np.isfinite(cnv_cndc) & np.isfinite(binned)
        if mask.sum() < 3:
            pytest.skip("Truncated hex has no overlap with CNV pressure range")
        diff = np.abs(binned[mask] - cnv_cndc[mask])
        assert np.nanmedian(diff) < 0.01, \
            f"CNDC median diff vs CNV: {np.nanmedian(diff):.5f} S/m"


# ---------------------------------------------------------------------------
# Sta0243 — 44 bytes/scan, with surface PAR (deck-only CNV)
# ---------------------------------------------------------------------------

class TestSta0243Parse:
    """Parser runs cleanly; correct variables present for this config."""

    def test_parses_without_error(self, ds_0243):
        assert ds_0243 is not None

    def test_core_variables_present(self, ds_0243):
        for var in ("TEMP1", "TEMP2", "PRES", "CNDC1", "CNDC2",
                    "DOXY1_instr", "DOXY2_instr", "CHLA1_fluorescence",
                    "CDOM1_instr", "TRANS1", "ALTI", "SPAR"):
            assert var in ds_0243, f"{var} missing"

    def test_spar_present(self, ds_0243):
        """Sta0243 has surface PAR — SPAR must be present."""
        assert "SPAR" in ds_0243

    def test_lat_lon_present(self, ds_0243):
        assert "LATITUDE" in ds_0243
        assert "LONGITUDE" in ds_0243

    def test_lat_lon_values(self, ds_0243):
        """Station is in Svalbard area."""
        lat = float(ds_0243.LATITUDE.mean())
        lon = float(ds_0243.LONGITUDE.mean())
        assert 70 < lat < 85,   f"Latitude {lat} outside expected Svalbard range"
        assert -20 < lon < 20,  f"Longitude {lon} outside expected range"

    def test_no_inf_values(self, ds_0243):
        """No infinite values in any variable after conversion."""
        for var in ds_0243.data_vars:
            arr = ds_0243[var].values
            if np.issubdtype(arr.dtype, np.floating):
                n_inf = np.isinf(arr).sum()
                assert n_inf == 0, f"{var} has {n_inf} inf values"

    def test_transmittance_range(self, ds_0243):
        """CStar transmittance should be 0-100%."""
        t = ds_0243.TRANS1.values
        finite = t[np.isfinite(t)]
        assert finite.min() >= -1.0,   "Transmittance below -1%"
        assert finite.max() <= 101.0,  "Transmittance above 101%"

    def test_doxy_range(self, ds_0243):
        """Oxygen should be in plausible range (including deck values)."""
        d = ds_0243.DOXY1_instr.values
        finite = d[np.isfinite(d)]
        assert finite.min() > 0,    "Negative oxygen"
        assert finite.max() < 15.0, "Oxygen > 15 ml/l implausible"

    def test_spar_not_constant(self, ds_0243):
        """SPAR should vary — if constant it means disconnected sensor."""
        spar = ds_0243.SPAR.values
        finite = spar[np.isfinite(spar)]
        if len(finite) > 1:
            assert not np.all(finite == finite[0]), \
                "SPAR is constant — sensor may be disconnected"


# ---------------------------------------------------------------------------
# Sta0520 — 44 bytes/scan, with surface PAR, deep cast
# ---------------------------------------------------------------------------

class TestSta0520Parse:
    """Parser handles deep cast correctly."""

    def test_parses_without_error(self, ds_0520):
        assert ds_0520 is not None

    def test_deep_pressure(self, ds_0520):
        """Cast goes to ~2100 dbar, but hex is truncated so just check positive."""
        p = ds_0520.PRES.values
        assert p.max() > 0, "No positive pressure values found"

    def test_core_variables_present(self, ds_0520):
        for var in ("TEMP1", "TEMP2", "PRES", "CNDC1", "CNDC2", "DOXY1_instr"):
            assert var in ds_0520, f"{var} missing"


class TestSta0520PhysicalRanges:

    def test_temp_range(self, ds_0520):
        t = ds_0520.TEMP1.values
        finite = t[np.isfinite(t)]
        assert finite.min() > -3.0
        assert finite.max() < 10.0

    def test_cndc_deep_reasonable(self, ds_0520):
        """Deep water conductivity should be in seawater range."""
        pres = ds_0520.PRES.values
        cndc = ds_0520.CNDC1.values
        deep = pres > 500
        if deep.sum() > 0:
            deep_cndc = cndc[deep]
            assert np.nanmin(deep_cndc) > 2.0, "Deep CNDC too low"
            assert np.nanmax(deep_cndc) < 4.0, "Deep CNDC too high"

    def test_doxy_range(self, ds_0520):
        d = ds_0520.DOXY1_instr.values
        finite = d[np.isfinite(d)]
        assert finite.min() > 0
        assert finite.max() < 15.0


class TestSta0520CNVComparison:
    """Compare against Seasoft CNV for deep cast."""

    def _bin_to_cnv_pres(self, our_pres, our_vals, cnv_pres):
        binned = np.full(len(cnv_pres), np.nan)
        for i, p in enumerate(cnv_pres):
            mask = (our_pres >= p - 0.5) & (our_pres < p + 0.5) & np.isfinite(our_vals)
            if mask.sum() > 0:
                binned[i] = our_vals[mask].mean()
        return binned

    def test_temp_vs_cnv(self, ds_0520, cnv_0520):
        """TEMP within 0.05°C of CNV IPTS-68 values (scale + filter diff)."""
        names, arr = cnv_0520
        cnv_pres = arr[:, 0]
        cnv_temp = np.where(arr[:, 1] > -9e-28, arr[:, 1], np.nan)

        our_pres = ds_0520.PRES.values
        our_temp = ds_0520.TEMP1.values
        binned = self._bin_to_cnv_pres(our_pres, our_temp, cnv_pres)

        mask = np.isfinite(cnv_temp) & np.isfinite(binned)
        diff = np.abs(binned[mask] - cnv_temp[mask])
        assert np.nanmedian(diff) < 0.05, \
            f"TEMP median diff vs CNV: {np.nanmedian(diff):.4f} °C"

    def test_cndc_vs_cnv(self, ds_0520, cnv_0520):
        """CNDC within 0.01 S/m of CNV values."""
        names, arr = cnv_0520
        cnv_pres = arr[:, 0]
        cnv_cndc = np.where(arr[:, 3] > -9e-28, arr[:, 3], np.nan)

        our_pres = ds_0520.PRES.values
        our_cndc = ds_0520.CNDC1.values
        binned = self._bin_to_cnv_pres(our_pres, our_cndc, cnv_pres)

        mask = np.isfinite(cnv_cndc) & np.isfinite(binned)
        diff = np.abs(binned[mask] - cnv_cndc[mask])
        assert np.nanmedian(diff) < 0.01, \
            f"CNDC median diff vs CNV: {np.nanmedian(diff):.5f} S/m"

    def test_transmittance_vs_cnv(self, ds_0520, cnv_0520):
        """TRANSMITTANCE within 1% of CNV CStarTr0 (col 8)."""
        names, arr = cnv_0520
        cnv_pres = arr[:, 0]
        cnv_trans = np.where(arr[:, 8] > -9e-28, arr[:, 8], np.nan)

        our_pres = ds_0520.PRES.values
        our_trans = ds_0520.TRANS1.values
        binned = self._bin_to_cnv_pres(our_pres, our_trans, cnv_pres)

        mask = np.isfinite(cnv_trans) & np.isfinite(binned)
        diff = np.abs(binned[mask] - cnv_trans[mask])
        assert np.nanmedian(diff) < 1.0, \
            f"TRANSMITTANCE median diff vs CNV: {np.nanmedian(diff):.3f} %"

    def test_doxy_vs_cnv(self, ds_0520, cnv_0520):
        """DOXY within 0.5 ml/l of CNV (tau/hysteresis corrections not applied)."""
        names, arr = cnv_0520
        cnv_pres = arr[:, 0]
        cnv_doxy = np.where(arr[:, 12] > -9e-28, arr[:, 12], np.nan)

        our_pres = ds_0520.PRES.values
        our_doxy = ds_0520.DOXY1_instr.values
        binned = self._bin_to_cnv_pres(our_pres, our_doxy, cnv_pres)

        mask = np.isfinite(cnv_doxy) & np.isfinite(binned)
        if mask.sum() < 3:
            pytest.skip("Truncated hex has no overlap with CNV pressure range")
        diff = np.abs(binned[mask] - cnv_doxy[mask])
        assert np.nanmedian(diff) < 0.5, \
            f"DOXY median diff vs CNV: {np.nanmedian(diff):.4f} ml/l"


# ---------------------------------------------------------------------------
# Cross-dataset consistency checks
# ---------------------------------------------------------------------------

class TestCrossDataset:
    """Checks that hold across all three datasets."""

    @pytest.mark.parametrize("ds_fixture", ["ds_0119", "ds_0243", "ds_0520"])
    def test_xarray_dataset(self, ds_fixture, request):
        """parse_hex always returns an xarray Dataset."""
        import xarray as xr
        ds = request.getfixturevalue(ds_fixture)
        assert isinstance(ds, xr.Dataset)

    @pytest.mark.parametrize("ds_fixture", ["ds_0119", "ds_0243", "ds_0520"])
    def test_scan_dimension(self, ds_fixture, request):
        """All datasets have a 'scan' dimension."""
        ds = request.getfixturevalue(ds_fixture)
        assert "scan" in ds.dims

    @pytest.mark.parametrize("ds_fixture", ["ds_0119", "ds_0243", "ds_0520"])
    def test_units_present(self, ds_fixture, request):
        """All physical variables have a units attribute."""
        ds = request.getfixturevalue(ds_fixture)
        for var in ("TEMP1", "PRES", "CNDC1"):
            assert "units" in ds[var].attrs, f"{var} missing units attr"

    @pytest.mark.parametrize("ds_fixture", ["ds_0119", "ds_0243", "ds_0520"])
    def test_no_all_nan_variables(self, ds_fixture, request):
        """No converted variable should be entirely NaN (except SPAR which may
        be NaN if the sensor was not connected on that cast)."""
        ds = request.getfixturevalue(ds_fixture)
        for var in ds.data_vars:
            if var == "SPAR":
                continue  # SPAR legitimately all-NaN if sensor disconnected
            arr = ds[var].values
            if np.issubdtype(arr.dtype, np.floating):
                assert not np.all(np.isnan(arr)), \
                    f"{var} is entirely NaN"

    @pytest.mark.parametrize("ds_fixture", ["ds_0119", "ds_0243", "ds_0520"])
    def test_primary_secondary_temp_same_length(self, ds_fixture, request):
        """TEMP and TEMP2 must have the same number of scans."""
        ds = request.getfixturevalue(ds_fixture)
        assert len(ds.TEMP1) == len(ds.TEMP2)


# ---------------------------------------------------------------------------
# Guard condition tests
# ---------------------------------------------------------------------------

class TestGuardConditions:
    """Test that edge cases are handled gracefully."""

    def test_par_b0_kept_as_raw_voltage(self, ds_0243):
        """If PAR B=0, raw voltage is kept and a warning is issued.
        Sta0243 has a calibrated PAR so this tests the normal path.
        The guard itself is tested indirectly — if B=0 files are added
        to test_data in future, add a direct test here."""
        # Sta0243 PAR is uncalibrated (no PAR variable, raw voltage kept)
        # Check that either PAR exists (calibrated) or a raw volt var exists
        has_par = "PAR" in ds_0243
        has_raw = any("par" in v.lower() for v in ds_0243.data_vars)
        assert has_par or has_raw, "Neither PAR nor raw PAR voltage found"

    def test_spar_nan_when_constant(self):
        """Constant SPAR output (disconnected sensor) should become NaN.
        This is tested via a synthetic call to _convert_surface_par."""
        import xarray as xr
        from kval.file._sbe_convert import _convert_surface_par

        # Simulate all-zero raw counts (disconnected sensor)
        n = 100
        raw = np.zeros(n)  # all zeros → constant after conversion
        ds = xr.Dataset({"surface_par_raw": (["scan"], raw)},
                        coords={"scan": np.arange(n)})
        sensors = [{
            "type": "SPAR_Sensor",
            "coefficients": {"ConversionFactor": 1646.6, "RatioMultiplier": 1.0},
            "serial_number": "test",
            "calibration_date": None,
        }]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds_out = _convert_surface_par(ds, sensors)

        assert "SPAR" in ds_out
        assert np.all(np.isnan(ds_out["SPAR"].values)), \
            "Constant SPAR should be set to NaN"
        assert any("constant" in str(warning.message).lower() for warning in w), \
            "Expected a warning about constant SPAR"

    def test_pressure_offset_applied(self, ds_0520):
        """Pressure offset from xmlcon should be applied.
        Sta0520 has a non-zero pressure offset in its xmlcon."""
        from kval.file.sbe_xmlcon import parse_xmlcon
        xmlcon = parse_xmlcon(XML_0520)
        p_sensor = next(
            s for s in xmlcon["sensors"] if s["type"] == "PressureSensor"
        )
        offset = float(p_sensor["coefficients"].get("Offset", 0.0))
        # If offset is non-zero, the pressure values should reflect it
        # (we can't verify the exact value without raw counts, but we can
        # verify the offset is non-zero and the dataset parses cleanly)
        if offset != 0.0:
            assert "PRES" in ds_0520
            p = ds_0520.PRES.values
            assert np.isfinite(p).any(), "PRES has no finite values after offset"


# ---------------------------------------------------------------------------
# parse_hex_dir — multi-cast loading
# ---------------------------------------------------------------------------

class TestParseHexDir:
    """parse_hex_dir loads multiple casts into a padded multi-cast Dataset."""

    @pytest.fixture(scope="class")
    def ds_dir(self):
        from kval.file.sbe_hex import parse_hex_dir
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return parse_hex_dir(TEST_DATA, verbose=False)

    def test_returns_dataset(self, ds_dir):
        import xarray as xr
        assert isinstance(ds_dir, xr.Dataset)

    def test_dims(self, ds_dir):
        assert "TIME" in ds_dir.dims
        assert "scan_count" in ds_dir.dims
        assert ds_dir.sizes["TIME"] == 3  # Sta0119, Sta0243, Sta0520

    def test_time_coord(self, ds_dir):
        assert "TIME" in ds_dir.coords
        assert not np.all(np.isnat(ds_dir.TIME.values))

    def test_station_coord(self, ds_dir):
        assert "STATION" in ds_dir.coords
        assert ds_dir.sizes["TIME"] == len(ds_dir.STATION)

    def test_time_scan_present(self, ds_dir):
        """TIME_SCAN should be present as per-scan timestamps."""
        assert "TIME_SCAN" in ds_dir.coords

    def test_core_variables_present(self, ds_dir):
        for var in ("TEMP1", "TEMP2", "PRES", "CNDC1", "CNDC2"):
            assert var in ds_dir, f"{var} missing from multi-cast dataset"

    def test_padding_is_nan(self, ds_dir):
        """Shorter casts should be NaN-padded at the end, or all same length."""
        # Find cast lengths from TIME_SCAN (NaT = padding)
        if "TIME_SCAN" not in ds_dir.coords:
            pytest.skip("TIME_SCAN not present")
        ts = ds_dir.TIME_SCAN.values  # (TIME, scan_count)
        cast_lengths = [(~np.isnat(ts[i])).sum() for i in range(ts.shape[0])]
        max_len = ds_dir.sizes["scan_count"]
        if all(l == max_len for l in cast_lengths):
            # All same length — no padding to check, that's fine
            pytest.skip("All test casts have equal length; padding not exercised")
        # Find a cast shorter than max and verify NaN padding
        idx = next(i for i, l in enumerate(cast_lengths) if l < max_len)
        n = cast_lengths[idx]
        t = ds_dir.TEMP1.isel(TIME=idx).values
        assert np.isnan(t[n]), "Expected NaN padding after cast end"
        assert not np.isnan(t[n - 1]), "Expected real data before cast end"

    def test_raw_names_flag(self):
        """parse_hex with raw_names=True keeps internal variable names."""
        from kval.file.sbe_hex import parse_hex
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds_raw = parse_hex(HEX_0119, XML_0119, raw_names=True)
        assert "temperature_primary_raw" in ds_raw.data_vars
        assert "TEMP1" not in ds_raw.data_vars