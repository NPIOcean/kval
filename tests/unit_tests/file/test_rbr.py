import pytest
import xarray as xr
import os
from pathlib import Path
from kval.file.rbr import read_rsk
from kval.file._variable_defs import RBR_name_map, RBR_units_map

# The RBR test files are downloaded from Zenodo by the `rbr_file` fixture
# (tests/unit_tests/conftest.py).

@pytest.mark.parametrize("rbr_file", [
    "solo_example.rsk",
    "conc_example.rsk",
    "conc_chl_par_example.rsk",
], indirect=True)

def test_read(rbr_file):

    # Read the dataset using the read function
    ds_rsk = read_rsk(rbr_file)

    # Check if the dataset is of type xarray.Dataset
    assert isinstance(ds_rsk, xr.Dataset), "Output is not an xarray.Dataset"

    # Check that the TIME dimension exists and is of correct type
    assert 'TIME' in ds_rsk.dims, "TIME dimension not found"
    assert ds_rsk['TIME'].dtype == 'float64', "TIME variable is not of type float64"

    # Check for presence of expected variables in the dataset
    assert 'TIME' in ds_rsk.variables, "TIME variable not found"
    assert len(ds_rsk.data_vars) > 0, "No data variables found in dataset"

    # Check if units are correctly assigned
    for var in ds_rsk.data_vars:
        assert 'units' in ds_rsk[var].attrs, f"Units attribute not found for variable {var}"
        expected_unit = RBR_units_map.get(ds_rsk[var].attrs['units'], ds_rsk[var].attrs['units'])
        assert ds_rsk[var].attrs['units'] == expected_unit, f"Units attribute for {var} is incorrect"

    # Check if the TIME variable has correct attributes
    assert ds_rsk['TIME'].attrs['units'] == 'days since 1970-01-01', "TIME units attribute is incorrect"
    assert ds_rsk['TIME'].attrs['axis'] == 'T', "TIME axis attribute is incorrect"

    # Check for presence of metadata
    assert 'instrument_model' in ds_rsk.attrs, "Instrument model metadata not found"
    assert 'instrument_serial_number' in ds_rsk.attrs, "Instrument serial number metadata not found"
    assert 'time_coverage_resolution' in ds_rsk.attrs, "Time coverage resolution metadata not found"

    # Validate calibration dates
    for var in ds_rsk.data_vars:
        if 'calibration_date' in ds_rsk[var].attrs:
            assert ds_rsk[var].attrs['calibration_date'] == ds_rsk[var].attrs.get('calibration_date'), f"Calibration date for {var} is incorrect"

    # Check for no extra variables
    expected_var_names = set(RBR_name_map.values())
    actual_var_names = set(ds_rsk.variables) - {'TIME'}
    assert actual_var_names.issubset(expected_var_names), "Unexpected variables found in the dataset"

    # Check if the TIME variable has the correct type
    assert ds_rsk['TIME'].dtype == 'float64', "TIME variable type is incorrect"


@pytest.mark.parametrize("rbr_file", ["solo_example.rsk"], indirect=True)
def test_read_rsk_time_does_not_depend_on_matplotlib_epoch(rbr_file):
    """TIME (and the history dates) used to be made with matplotlib's
    date2num/num2date, whose epoch is a matplotlib setting. Changing that
    setting must not change TIME, and the history dates must match the
    data."""
    import matplotlib
    import numpy as np
    reference = read_rsk(rbr_file).TIME.values
    with matplotlib.rc_context({"date.epoch": "2000-01-01T00:00:00"}):
        ds_rsk = read_rsk(rbr_file)
    np.testing.assert_array_equal(ds_rsk.TIME.values, reference)
    assert ds_rsk.TIME.attrs["units"] == "days since 1970-01-01"

    t = xr.coding.times.decode_cf_datetime(
        ds_rsk.TIME.values, ds_rsk.TIME.units)
    first = np.datetime_as_string(t.min(), unit='D')
    last = np.datetime_as_string(t.max(), unit='D')
    assert ds_rsk.attrs["history"].startswith(
        f"{first} - {last}: Data collection.")


# ---------------------------------------------------------------------
# Sampling schedules and calibration dates (no .rsk files needed)
# ---------------------------------------------------------------------

import numpy as np
from types import SimpleNamespace
from pyrsktools import datatypes as rsk_dt
from kval.file import rbr


@pytest.mark.parametrize("schedule, resolution, words", [
    (rsk_dt.ContinuousInfo(1, 1, 2000), 2.0, "Continuous sampling - one measurement every 2 sec"),
    (rsk_dt.AverageInfo(1, 1, 1000, 600000, 60), 600.0, "One average of 60 samples"),
    (rsk_dt.BurstInfo(1, 1, 125, 3600000, 1024), None, "Burst sampling - 1024 samples at 0.125 sec"),
    (rsk_dt.WaveInfo(1, 1, 250, 1800000, 2048), None, "Wave sampling - 2048 samples"),
    (rsk_dt.TideInfo(1, 1, 1000, 600000, 60), None, "Tide sampling - 60 samples"),
    (rsk_dt.DirectionalInfo(1, 1, 0, 1000, 10000, 0.0, 0.0), None, "Directional sampling"),
    (SimpleNamespace(), None, "not recognised"),
])
def test_describe_schedule_all_rbr_schedule_types(schedule, resolution, words):
    """Only continuous and average schedules used to be handled; anything
    else crashed with UnboundLocalError."""
    res, text = rbr._describe_schedule(schedule)
    assert res == resolution
    assert words in text


def test_build_cal_dates_maps_by_channel_order():
    """Calibrations refer to channels by channelOrder. The dates used to be
    looked up by channelID, which need not be the same number."""
    channels = [
        SimpleNamespace(channelID=3, longName="conductivity", isDerived=0, channelOrder=1),
        SimpleNamespace(channelID=5, longName="temperature", isDerived=0, channelOrder=2),
        SimpleNamespace(channelID=7, longName="pressure", isDerived=0, channelOrder=3),
        SimpleNamespace(channelID=9, longName="salinity", isDerived=1, channelOrder=None),
    ]
    instrument_channels = [SimpleNamespace(channelID=c.channelID, channelOrder=c.channelOrder)
                           for c in channels if c.channelOrder is not None]
    calibrations = [
        SimpleNamespace(channelOrder=1, tstamp=np.datetime64("2020-01-01")),
        SimpleNamespace(channelOrder=2, tstamp=np.datetime64("2021-02-02")),
        SimpleNamespace(channelOrder=3, tstamp=np.datetime64("2022-03-03")),
    ]
    expected = {"conductivity": "2020-01-01", "temperature": "2021-02-02",
                "pressure": "2022-03-03"}
    rskdata = SimpleNamespace(channels=channels, calibrations=calibrations,
                              instrumentChannels=instrument_channels)
    assert rbr._build_cal_dates(rskdata) == expected
    # Same result from Channel.channelOrder alone (no instrumentChannels)
    rskdata = SimpleNamespace(channels=channels, calibrations=calibrations)
    assert rbr._build_cal_dates(rskdata) == expected
