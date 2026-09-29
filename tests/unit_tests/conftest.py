"""Shared fixtures for the unit tests."""
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import pytest


@pytest.fixture
def make_ctd_ds():
    """Factory for a synthetic moored CTD dataset (numeric CF TIME, 10 min
    sampling by default). CNDC is consistent with PSAL/TEMP/PRES and is stored
    in the requested units ('mS cm-1' or 'S m-1')."""

    def _make(start="2021-11-09 11:20", n=3000, freq="10min", seed=0,
              cndc_units="mS cm-1"):
        rng = np.random.default_rng(seed)
        t = pd.date_range(start, periods=n, freq=freq)
        T = 1 + 0.5 * rng.standard_normal(n)
        S = 34.8 + 0.05 * rng.standard_normal(n)
        P = 49 + 0.2 * rng.standard_normal(n)
        C = gsw.C_from_SP(S, T, P)  # mS/cm
        if cndc_units in ("S m-1", "S/m"):
            C = C / 10
        days = (t - pd.Timestamp("1970-01-01")).total_seconds().values / 86400
        ds = xr.Dataset(
            {"TEMP": ("TIME", T, {"units": "degC"}),
             "PSAL": ("TIME", S, {"units": "1"}),
             "PRES": ("TIME", P, {"units": "dbar"}),
             "CNDC": ("TIME", C, {"units": cndc_units})},
            coords={"TIME": ("TIME", days, {
                "units": "days since 1970-01-01", "calendar": "standard"})})
        ds["LATITUDE"] = np.float32(81.4)
        ds["LONGITUDE"] = np.float32(31.2)
        return ds.set_coords(["LATITUDE", "LONGITUDE"])

    return _make
