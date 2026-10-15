"""Shared fixtures for the unit tests."""
import time
from pathlib import Path

import requests
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


# ---------------------------------------------------------------------
# RBR test files (too large for the repository; downloaded from Zenodo)
# ---------------------------------------------------------------------

RBR_FILE_URLS = {
    "conc_chl_par_example.rsk": "https://zenodo.org/records/13321317/files/conc_chl_par_example.rsk?download=1",
    "conc_example.rsk": "https://zenodo.org/records/13321317/files/conc_example.rsk?download=1",
    "solo_example.rsk": "https://zenodo.org/records/13321317/files/solo_example.rsk?download=1",
}
RBR_FILE_DIR = Path("tests/test_data/rbr_files")  # git-ignored

# Files that could not be downloaded in this test session (with the
# reason), so the next test that needs them is skipped at once instead of
# trying again
_RBR_DOWNLOAD_FAILED = {}


def _download_with_retry(url, retries=4, backoff=3.0):
    """
    GET a URL, retrying on any network problem (dropped connections,
    timeouts, server errors such as Zenodo's occasional 504). Raises the
    last error if every attempt fails.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=(15, 120))
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise last_exc


@pytest.fixture
def rbr_file(request):
    """
    Path to the RBR test file named by the test's parameter, e.g.
    @pytest.mark.parametrize("rbr_file", ["solo_example.rsk"], indirect=True).

    A file is downloaded only if it is not already in
    tests/test_data/rbr_files, and is then kept there (the folder is
    git-ignored), so each file is downloaded once rather than once per test.
    If Zenodo can't be reached (after a few attempts), the test is skipped
    with the reason, and so are later tests needing the same file.
    """
    file_name = request.param
    RBR_FILE_DIR.mkdir(parents=True, exist_ok=True)
    file_path = RBR_FILE_DIR / file_name

    if file_name in _RBR_DOWNLOAD_FAILED:
        pytest.skip(_RBR_DOWNLOAD_FAILED[file_name])
    if not file_path.exists():
        try:
            content = _download_with_retry(RBR_FILE_URLS[file_name])
        except requests.exceptions.RequestException as e:
            _RBR_DOWNLOAD_FAILED[file_name] = (
                f"Could not download {file_name} from Zenodo: {e}")
            pytest.skip(_RBR_DOWNLOAD_FAILED[file_name])
        # Write to a temporary name first, so an interrupted download
        # never leaves a broken file behind
        tmp_path = file_path.with_suffix(".part")
        tmp_path.write_bytes(content)
        tmp_path.replace(file_path)

    return file_path
