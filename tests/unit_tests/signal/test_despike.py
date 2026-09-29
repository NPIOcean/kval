import pytest
import numpy as np
import xarray as xr
from kval.signal import despike

@pytest.fixture
def sample_dataset():
    """Create a sample xarray Dataset for testing."""

    time = np.arange(0, 40, 0.1)
    temp = np.sin(time)  # Some synthetic data (a sine wave)
    #temp = np.ones(len(time))  # Some synthetic data (a sine wave)

    ds = xr.Dataset(
        {'temp': (['time'], temp)},
        coords={'time': time})

    # Inserting a NaN
    ds['temp'][21] = np.nan

    # Inserting 4 spikes
    ds['temp'][41] = 3
    ds['temp'][370] = 7
    ds['temp'][371] = 8
    ds['temp'][111] = 5
    return ds

def test_despike_default(sample_dataset):
    """Test despiking with default parameters."""
    ds = sample_dataset
    result = despike.despike_rolling(
        ds, variable="temp", window_size=11, n_std=1.5, dim="time"
    )

    assert isinstance(result, xr.Dataset), "Result should be a Dataset."
    assert "temp" in result, "The despiked variable should be in the Dataset."
    assert result["temp"].isnull().sum().item() == 5, "Should have 1 nan and 4 outliers."

def test_despike_return_index(sample_dataset):
    """Test despiking with return_index=True."""
    ds = sample_dataset
    result, outliers = despike.despike_rolling(
        ds, variable="temp", window_size=11, n_std=1.5, dim="time", return_index=True
    )

    assert isinstance(result, xr.Dataset), "Result should be a Dataset."
    assert isinstance(outliers, xr.DataArray), "Outliers should be a DataArray."
    assert outliers.sum().item() == 4, "Shoudl have 4 outliers"

def test_despike_plot(sample_dataset, monkeypatch):
    """Test despiking with plot=True (mock plt.show())."""
    import matplotlib.pyplot as plt
    # Mock plt.show() to prevent actual plotting during tests
    monkeypatch.setattr(plt, "show", lambda: None)

    ds = sample_dataset
    despike.despike_rolling(
        ds, variable="temp", window_size=11, n_std=1.5, dim="time", plot=True
    )

def test_despike_verbose(sample_dataset, capsys):
    """Test despiking with verbose=True."""
    ds = sample_dataset
    despike.despike_rolling(
        ds, variable="temp", window_size=11, n_std=1.5, dim="time", verbose=True
    )
    captured = capsys.readouterr()
    assert "Removed" in captured.out, "Verbose output should include the number of removed points."

def test_despike_min_periods(sample_dataset):
    """Test despiking with min_periods set."""
    ds = sample_dataset
    result = despike.despike_rolling(
        ds, variable="temp", window_size=11, n_std=1.5, dim="time", min_periods=3
    )
    assert isinstance(result, xr.Dataset), "Result should be a Dataset."


# ---------------------------------------------------------------------
# 0.5.1: NaNs in the window no longer blind the detection
# ---------------------------------------------------------------------
def _noisy_series(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    return xr.Dataset(
        {'x': ('time', 34.8 + 0.01 * rng.standard_normal(n))},
        coords={'time': np.arange(n)})


def test_despike_catches_spike_next_to_nan():
    ds = _noisy_series()
    ds['x'].values[500] += 0.5
    ds['x'].values[498] = np.nan
    out = despike.despike_rolling(ds, 'x', window_size=9, n_std=3, dim='time')
    assert np.isnan(out['x'].values[500])


def test_despike_strict_min_periods_restores_old_behaviour():
    ds = _noisy_series()
    ds['x'].values[500] += 0.5
    ds['x'].values[498] = np.nan
    out = despike.despike_rolling(
        ds, 'x', window_size=9, n_std=3, dim='time', min_periods=9)
    assert not np.isnan(out['x'].values[500])


def test_despike_few_false_positives_in_gappy_clean_data():
    ds = _noisy_series(n=20000, seed=4)
    rng = np.random.default_rng(5)
    ds['x'].values[rng.random(20000) < 0.05] = np.nan  # 5 % scattered gaps
    n_valid = int(np.isfinite(ds['x'].values).sum())
    out = despike.despike_rolling(ds, 'x', window_size=9, n_std=3, dim='time')
    removed = n_valid - int(np.isfinite(out['x'].values).sum())
    assert removed / n_valid < 0.005