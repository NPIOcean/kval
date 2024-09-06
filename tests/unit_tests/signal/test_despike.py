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
        ds, var_name="temp", window_size=11, n_std=1.5, dim="time"
    )

    assert isinstance(result, xr.Dataset), "Result should be a Dataset."
    assert "temp" in result, "The despiked variable should be in the Dataset."
    assert result["temp"].isnull().sum().item() == 5, "Should have 1 nan and 4 outliers."

def test_despike_return_index(sample_dataset):
    """Test despiking with return_index=True."""
    ds = sample_dataset
    result, outliers = despike.despike_rolling(
        ds, var_name="temp", window_size=11, n_std=1.5, dim="time", return_index=True
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
        ds, var_name="temp", window_size=11, n_std=1.5, dim="time", plot=True
    )

def test_despike_verbose(sample_dataset, capsys):
    """Test despiking with verbose=True."""
    ds = sample_dataset
    despike.despike_rolling(
        ds, var_name="temp", window_size=11, n_std=1.5, dim="time", verbose=True
    )
    captured = capsys.readouterr()
    assert "Removed" in captured.out, "Verbose output should include the number of removed points."

def test_despike_min_periods(sample_dataset):
    """Test despiking with min_periods set."""
    ds = sample_dataset
    result = despike.despike_rolling(
        ds, var_name="temp", window_size=11, n_std=1.5, dim="time", min_periods=3
    )
    assert isinstance(result, xr.Dataset), "Result should be a Dataset."
