import pytest
import xarray as xr
import pandas as pd
import numpy as np
from kval.util import xr_funcs  
from kval.util.xr_funcs import time_average

# Define a fixture for a mock dataset
@pytest.fixture
def mock_dataset() -> xr.Dataset:
    """
    Fixture to create a mock xarray.Dataset with metadata and variables for testing.
    """
    # Define dimensions
    Nt = 10
    time = pd.date_range('2024-01-01', periods=Nt, freq='D')
    pres = [100, 600, 2000, 6000, 11000]  # Pressure levels

    # Create data for TEMP(TIME, PRES)
    temp_data = 15 + 8 * np.random.randn(Nt, len(pres))  # Example temperature data

    # Create data for STATION(TIME) and OCEAN(TIME)
    station_data = [f'st{stnum:02.0f}' for stnum in np.arange(1, Nt + 1)]
    ocean_data = ['Atlantic', 'Arctic', 'Pacific', 'Mediterranean', 'Southern',
                 'Baltic', 'Indian', 'Caribbean', 'Weddell', 'Ross']

    # Create data for ZONE(PRES)
    zone_data = ['epipelagic', 'mesopelagic', 'bathypelagic', 'abyssopelagic', 'hadopelagic']

    # Create the Dataset
    ds = xr.Dataset(
        {
            'TEMP': (['TIME', 'PRES'], temp_data),
            'OCEAN': (['TIME'], ocean_data),
            'STATION': (['TIME'], station_data),
            'ZONE': (['PRES'], zone_data)
        },
        coords={
            'TIME': time,
            'PRES': pres
        },
        # Add some metadata
        attrs={
            'id': 'test_dataset',
            'description': 'This is a test dataset.',
            'author': 'Test Author'
        }
    )

    # Add variable attributes
    ds['TEMP'].attrs = {
        'units': 'degC',
        'long_name': 'Test Temperature'
    }

    return ds


### Testing the pick() function

def test_pick_single_TIME(mock_dataset):
    result = xr_funcs.pick(mock_dataset, STATION='st02')
    assert list(result.dims) == ['PRES']
    assert 'st02' in result.STATION.values
    assert 'Arctic' in result.OCEAN.values

def test_pick_multiple_TIME(mock_dataset):
    result = xr_funcs.pick(mock_dataset, STATION=['st02', 'st03'])
    assert result.sizes['TIME'] == 2
    assert 'st02' in result.STATION.values
    assert 'st03' in result.STATION.values

def test_pick_single_PRES(mock_dataset):
    result = xr_funcs.pick(mock_dataset, ZONE='epipelagic')
    assert list(result.dims) == ['TIME']
    assert 'epipelagic' in result.ZONE.values
    assert result.PRES.item() == 100

def test_pick_multiple_PRES(mock_dataset):
    result = xr_funcs.pick(mock_dataset, ZONE=['epipelagic', 'hadopelagic'])
    assert result.sizes['PRES'] == 2
    assert 'epipelagic' in result.ZONE.values
    assert 'hadopelagic' in result.ZONE.values

def test_pick_nonexistent_value(mock_dataset):
    result = xr_funcs.pick(mock_dataset, STATION='st11')
    assert list(result.dims) == ['TIME', 'PRES']
    assert result.sizes['TIME'] == 0

def test_pick_squeeze_false(mock_dataset):
    result = xr_funcs.pick(mock_dataset, STATION='st02', squeeze=False)
    assert result.sizes['TIME'] == 1
    assert result.sizes['PRES'] == 5  # PRES dimension should remain unchanged

def test_pick_with_squeeze(mock_dataset):
    result = xr_funcs.pick(mock_dataset, STATION='st02')
    assert 'TIME' not in result.dims

def test_pick_invalid_dimension(mock_dataset):
    with pytest.raises(ValueError):
        xr_funcs.pick(mock_dataset, TEMP=15)

def test_pick_nonexistent_variable(mock_dataset):
    with pytest.raises(ValueError, match="not found in the dataset"):
        xr_funcs.pick(mock_dataset, NOT_A_VAR='x')

def test_pick_multiple_conditions(mock_dataset):
    # Use multiple conditions: both STATION and ZONE
    result = xr_funcs.pick(mock_dataset,
                           STATION=['st02', 'st03', 'st05'],
                           ZONE=['epipelagic', 'bathypelagic'])

    # Check that only the entries that match both conditions are present
    assert result.sizes['TIME'] == 3  # Only one TIME index should match the condition
    assert result.sizes['PRES'] == 2  # Only one PRES index should match the condition

    # Verify the results
    assert 'epipelagic' in result.ZONE.values
    assert 'bathypelagic' in result.ZONE.values
    assert 'st02' in result.STATION.values
    assert 'st03' in result.STATION.values
    assert 'st05' in result.STATION.values
    assert 'st01' not in result.STATION.values
    assert 'abyssopelagic' not in result.ZONE.values

    # Check if `TIME` dimension is correctly filtered
    assert len(result.TIME) == 3
    assert pd.Timestamp('2024-01-02') in result.TIME
    assert pd.Timestamp('2024-01-03') in result.TIME
    assert pd.Timestamp('2024-01-05') in result.TIME
    assert pd.Timestamp('2024-01-04') not in result.TIME

    # Check if `PRES` dimension is correctly filtered
    assert (result.PRES.values == [100, 2000]).all()


### Testing the rename_attr() function

def test_rename_attr_global(mock_dataset):
    xr_funcs.rename_attr(mock_dataset, 'author', 'creator', verbose=False)
    assert mock_dataset.attrs['creator'] == 'Test Author'
    assert 'author' not in mock_dataset.attrs

def test_rename_attr_variable(mock_dataset):
    xr_funcs.rename_attr(mock_dataset['TEMP'], 'units', 'unit', verbose=False)
    assert mock_dataset['TEMP'].attrs['unit'] == 'degC'
    assert 'units' not in mock_dataset['TEMP'].attrs

def test_rename_attr_missing_prints_message(mock_dataset, capsys):
    xr_funcs.rename_attr(mock_dataset, 'nonexistent', 'new_name', verbose=True)
    captured = capsys.readouterr()
    assert 'Could not rename' in captured.out
    assert 'new_name' not in mock_dataset.attrs

def test_rename_attr_silent_when_verbose_false(mock_dataset, capsys):
    xr_funcs.rename_attr(mock_dataset, 'nonexistent', 'new_name', verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ''


### Testing the add_attrs_from_dict() function

def test_add_attrs_from_dict_basic(mock_dataset):
    xr_funcs.add_attrs_from_dict(mock_dataset, {'project': 'kval', 'institution': 'X'})
    assert mock_dataset.attrs['project'] == 'kval'
    assert mock_dataset.attrs['institution'] == 'X'

def test_add_attrs_from_dict_override_true_default(mock_dataset):
    xr_funcs.add_attrs_from_dict(mock_dataset, {'author': 'New Author'})
    assert mock_dataset.attrs['author'] == 'New Author'

def test_add_attrs_from_dict_override_false_preserves_existing(mock_dataset):
    xr_funcs.add_attrs_from_dict(
        mock_dataset, {'author': 'New Author', 'project': 'kval'}, override=False)
    assert mock_dataset.attrs['author'] == 'Test Author'  # unchanged
    assert mock_dataset.attrs['project'] == 'kval'  # new key still added

def test_add_attrs_from_dict_on_variable(mock_dataset):
    xr_funcs.add_attrs_from_dict(mock_dataset['TEMP'], {'valid_min': -2})
    assert mock_dataset['TEMP'].attrs['valid_min'] == -2


### Testing the swap_var_coord() function

def test_swap_var_coord_basic(mock_dataset):
    # Test basic swapping of TIME with STATION
    result = xr_funcs.swap_var_coord(mock_dataset, coordinate='TIME', variable='STATION')

    # Check that STATION is now a coordinate and TIME is a variable
    assert 'STATION' in result.coords
    assert 'TIME' in result.variables and 'TIME' not in result.coords
    assert list(result.dims) == ['STATION', 'PRES']

def test_swap_var_coord_with_drop(mock_dataset):
    # Test swapping with dropping the original coordinate
    result = xr_funcs.swap_var_coord(mock_dataset, coordinate='TIME', variable='STATION', drop_original=True)

    # Check that STATION is now a coordinate and TIME is completely removed
    assert 'STATION' in result.coords
    assert 'TIME' not in result.variables
    assert list(result.dims) == ['STATION', 'PRES']

def test_swap_var_coord_invalid_coordinate(mock_dataset):
    # Test with an invalid coordinate name
    with pytest.raises(ValueError, match="'INVALID' is not a coordinate in the Dataset."):
        xr_funcs.swap_var_coord(mock_dataset, coordinate='INVALID', variable='STATION')

def test_swap_var_coord_invalid_variable(mock_dataset):
    # Test with a variable that is already a coordinate
    with pytest.raises(ValueError, match="'PRES' is already a coordinate in the Dataset."):
        xr_funcs.swap_var_coord(mock_dataset, coordinate='TIME', variable='PRES')

def test_swap_var_coord_no_drop(mock_dataset):
    # Test swapping without dropping the original coordinate
    result = xr_funcs.swap_var_coord(mock_dataset, coordinate='PRES', variable='ZONE', drop_original=False)

    # Check that ZONE is now a coordinate and PRES is still in the dataset
    assert 'ZONE' in result.coords
    assert 'PRES' in result.variables and 'PRES' not in result.coords
    assert list(result.dims) == ['TIME', 'ZONE']

def test_swap_var_coord_preserve_data(mock_dataset):
    # Test that data is preserved correctly after swapping
    temp_before = mock_dataset['TEMP'].values
    result = xr_funcs.swap_var_coord(mock_dataset, coordinate='PRES', variable='ZONE')
    temp_after = result['TEMP'].values

    # Ensure data integrity is preserved after swapping
    np.testing.assert_array_equal(temp_before, temp_after)

def test_swap_var_coord_restore(mock_dataset):
    # Swap and then restore the original coordinate to test idempotence
    result = xr_funcs.swap_var_coord(mock_dataset, coordinate='TIME', variable='STATION')
    result = xr_funcs.swap_var_coord(result, coordinate='STATION', variable='TIME')

    # Check that the restored dataset matches the original dimensions and coordinates
    assert list(result.dims) == ['TIME', 'PRES']
    assert 'TIME' in result.coords
    assert 'STATION' in result.variables and 'STATION' not in result.coords
    assert mock_dataset.equals(result)


### Testing the promote_cf_coordinates() function

def test_promote_cf_coordinates_basic(mock_dataset):
    mock_dataset['TEMP'].attrs['coordinates'] = 'STATION OCEAN'
    result = xr_funcs.promote_cf_coordinates(mock_dataset)
    assert 'STATION' in result.coords
    assert 'OCEAN' in result.coords
    assert 'TEMP' not in result.coords

def test_promote_cf_coordinates_no_coordinates_attr_is_noop(mock_dataset):
    result = xr_funcs.promote_cf_coordinates(mock_dataset)
    assert 'STATION' not in result.coords
    assert 'OCEAN' not in result.coords

def test_promote_cf_coordinates_ignores_nonexistent_names(mock_dataset):
    """A 'coordinates' attribute referencing a variable that doesn't exist
    in the dataset should be silently ignored, not raise."""
    mock_dataset['TEMP'].attrs['coordinates'] = 'STATION NOT_A_REAL_VAR'
    result = xr_funcs.promote_cf_coordinates(mock_dataset)
    assert 'STATION' in result.coords
    assert 'NOT_A_REAL_VAR' not in result.coords
    assert 'NOT_A_REAL_VAR' not in result.variables


### Testing the append_processing_history() function

def test_append_processing_history_creates_when_missing(mock_dataset):
    result = xr_funcs.append_processing_history(
        mock_dataset, 'TEMP', 'Applied offset of 5.2 degC.')
    assert result['TEMP'].attrs['processing_history'] == (
        'Applied offset of 5.2 degC.')

def test_append_processing_history_appends_when_present(mock_dataset):
    mock_dataset['TEMP'].attrs['processing_history'] = 'Applied offset of 5.2 degC.'
    result = xr_funcs.append_processing_history(
        mock_dataset, 'TEMP', 'Rejected values above 30 degC.')
    assert result['TEMP'].attrs['processing_history'] == (
        'Applied offset of 5.2 degC. Rejected values above 30 degC.')

def test_append_processing_history_multiple_calls_accumulate_in_order(mock_dataset):
    ds = xr_funcs.append_processing_history(mock_dataset, 'TEMP', 'Step one.')
    ds = xr_funcs.append_processing_history(ds, 'TEMP', 'Step two.')
    ds = xr_funcs.append_processing_history(ds, 'TEMP', 'Step three.')
    assert ds['TEMP'].attrs['processing_history'] == (
        'Step one. Step two. Step three.')

def test_append_processing_history_custom_key(mock_dataset):
    result = xr_funcs.append_processing_history(
        mock_dataset, 'TEMP', 'Applied offset.', key='editing_log')
    assert result['TEMP'].attrs['editing_log'] == 'Applied offset.'
    assert 'processing_history' not in result['TEMP'].attrs

def test_append_processing_history_does_not_affect_other_variables(mock_dataset):
    result = xr_funcs.append_processing_history(mock_dataset, 'TEMP', 'Applied offset.')
    assert 'processing_history' not in result['OCEAN'].attrs

def test_append_processing_history_does_not_mutate_original(mock_dataset):
    """Since this uses deep=True, the dataset passed in should be
    completely untouched -- caller must use the returned dataset."""
    result = xr_funcs.append_processing_history(mock_dataset, 'TEMP', 'Applied offset.')
    assert 'processing_history' not in mock_dataset['TEMP'].attrs
    assert result is not mock_dataset

def test_append_processing_history_does_not_disturb_other_attrs(mock_dataset):
    """Existing unrelated attrs on the variable should survive untouched."""
    result = xr_funcs.append_processing_history(mock_dataset, 'TEMP', 'Applied offset.')
    assert result['TEMP'].attrs['units'] == 'degC'
    assert result['TEMP'].attrs['long_name'] == 'Test Temperature'

def test_append_processing_history_raises_if_variable_missing(mock_dataset):
    with pytest.raises(ValueError, match="not found in the Dataset"):
        xr_funcs.append_processing_history(mock_dataset, 'NOT_A_REAL_VAR', 'Note.')

def test_append_processing_history_deep_copy_false_mutates_in_place(mock_dataset):
    """With deep_copy=False, the caller's dataset is modified directly --
    intended for use inside functions that already made their own copy."""
    result = xr_funcs.append_processing_history(
        mock_dataset, 'TEMP', 'Applied offset.', deep_copy=False)
    assert result is mock_dataset
    assert mock_dataset['TEMP'].attrs['processing_history'] == 'Applied offset.'


## Testing time_average


# ---------------------------------------------------------------------
# Encoding-preservation behavior (the fix for the decode_cf / NameError
# issue: this function should not silently change whether TIME is
# CF-encoded or decoded -- it should return in the same state as the
# input, doing any necessary decoding/re-encoding internally).
# ---------------------------------------------------------------------
 
def test_time_average_preserves_encoded_input_as_encoded_output():
    ds = xr.Dataset({'TEMP': ('TIME', np.arange(10.0))}, coords={'TIME': np.arange(10.0)})
    ds['TIME'].attrs['units'] = 'days since 2021-01-01'
    ds['TIME'].attrs['calendar'] = 'standard'
    result = time_average(ds, '1D')
    assert np.issubdtype(result['TIME'].dtype, np.number)
    assert result['TIME'].attrs['units'] == 'days since 2021-01-01'
 
 
def test_time_average_preserves_decoded_input_as_decoded_output():
    ds = xr.Dataset({'TEMP': ('TIME', np.arange(10.0))},
                     coords={'TIME': pd.date_range('2021-01-01', periods=10)})
    result = time_average(ds, '1D')
    assert np.issubdtype(result['TIME'].dtype, np.datetime64)
 
 
def test_time_average_does_not_mutate_input():
    ds = xr.Dataset({'TEMP': ('TIME', np.arange(10.0))}, coords={'TIME': np.arange(10.0)})
    ds['TIME'].attrs['units'] = 'days since 2021-01-01'
    ds_original = ds.copy(deep=True)
    _ = time_average(ds)
    xr.testing.assert_identical(ds, ds_original)
 
 
def test_time_average_raises_if_encoded_time_has_no_units():
    """A numeric TIME with no 'units' attribute can't be decoded as CF
    time at all -- this should fail clearly rather than silently doing
    the wrong thing."""
    ds = xr.Dataset({'TEMP': ('TIME', np.arange(10.0))}, coords={'TIME': np.arange(10.0)})
    with pytest.raises(ValueError, match="units"):
        time_average(ds)
 
 
# ---------------------------------------------------------------------
# General behavior
# ---------------------------------------------------------------------
 
def test_time_average_computes_correct_mean():
    ds = xr.Dataset(
        {'TEMP': ('TIME', np.array([1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0]))},
        coords={'TIME': pd.date_range('2021-01-01', periods=8, freq='12h')},
    )
    result = time_average(ds, '1D', label='left')
    np.testing.assert_allclose(result['TEMP'].values, [1.0, 3.0, 5.0, 7.0])
 
 
@pytest.mark.parametrize(
    "label,expected_first_timestamp",
    [
        ("left", "2021-01-01T00:00:00"),
        ("right", "2021-01-02T00:00:00"),
        ("center", "2021-01-01T12:00:00"),
    ],
)
def test_time_average_label_placement(label, expected_first_timestamp):
    ds = xr.Dataset(
        {'TEMP': ('TIME', np.array([1.0, 1.0, 3.0, 3.0]))},
        coords={'TIME': pd.date_range('2021-01-01', periods=4, freq='12h')},
    )
    result = time_average(ds, '1D', label=label)
    assert result['TIME'].values[0] == np.datetime64(expected_first_timestamp)
 
 
def test_time_average_drops_non_numeric_time_dependent_variables():
    ds = xr.Dataset(
        {
            'TEMP': ('TIME', np.array([1.0, 1.0, 3.0, 3.0])),
            'STATION': ('TIME', np.array(['a', 'a', 'b', 'b'])),
        },
        coords={'TIME': pd.date_range('2021-01-01', periods=4, freq='12h')},
    )
    result = time_average(ds, '1D')
    assert 'STATION' not in result
    assert 'TEMP' in result
 
 
def test_time_average_preserves_time_independent_variables_unbroadcast():
    """A variable with no TIME dimension at all (e.g. ZONE(PRES)) should
    be passed through unchanged, not broadcast across the new time bins."""
    ds = xr.Dataset(
        {
            'TEMP': ('TIME', np.array([1.0, 1.0, 3.0, 3.0])),
            'ZONE': ('PRES', np.array([1, 2, 3])),
        },
        coords={'TIME': pd.date_range('2021-01-01', periods=4, freq='12h')},
    )
    result = time_average(ds, '1D')
    assert result['ZONE'].dims == ('PRES',)
    np.testing.assert_array_equal(result['ZONE'].values, [1, 2, 3])
 
 
def test_time_average_raises_for_missing_time_dim():
    ds = xr.Dataset({'TEMP': ('TIME', np.arange(4.0))},
                     coords={'TIME': pd.date_range('2021-01-01', periods=4)})
    with pytest.raises(ValueError, match="not a dimension"):
        time_average(ds, '1D', time_dim='NOTATIME')
 
 
def test_time_average_raises_for_invalid_label():
    ds = xr.Dataset({'TEMP': ('TIME', np.arange(4.0))},
                     coords={'TIME': pd.date_range('2021-01-01', periods=4)})
    with pytest.raises(ValueError, match="label must be"):
        time_average(ds, '1D', label='bogus')
 
 
def test_time_average_center_for_calendar_based_interval_is_bin_midpoint():
    """For calendar-based intervals (e.g. month-end 'ME') label='center'
    stamps each bin at the midpoint of its own bin edges."""
    ds = xr.Dataset({'TEMP': ('TIME', np.arange(60.0))},
                     coords={'TIME': pd.date_range('2021-01-01', periods=60)})
    out = time_average(ds, 'ME', label='center')
    # Data run 1 Jan - 1 Mar. January: 31 days -> 16 Jan 12:00;
    # February 2021: 28 days -> 15 Feb 00:00; March: 31 days -> 16 Mar 12:00
    np.testing.assert_array_equal(
        out.TIME.values,
        np.array(['2021-01-16T12:00', '2021-02-15T00:00', '2021-03-16T12:00'],
                 dtype='datetime64[ns]'))
 
@pytest.mark.filterwarnings("ignore:time_average:UserWarning")  # partial edge bins are expected here
def test_time_average_origin_shifts_bin_edges():
    ds = xr.Dataset(
        {'TEMP': ('TIME', np.arange(8.0))},
        coords={'TIME': pd.date_range('2021-01-01 00:00', periods=8, freq='3h')},
    )
    result_default = time_average(ds, '6h', label='left')
    result_shifted = time_average(ds, '6h', label='left', origin='2021-01-01 03:00')
    assert result_default['TIME'].values[0] != result_shifted['TIME'].values[0]
    assert result_shifted['TIME'].values[0] == np.datetime64('2020-12-31T21:00:00')
 


### Testing the reorder_coords_to_end() function

def test_reorder_coords_to_end_moves_specified_coords_last():
    ds = xr.Dataset(
        {'TEMP': ('TIME', np.array([1.0]))},
        coords={'LATITUDE': ((), 80.0), 'LONGITUDE': ((), 30.0), 'TIME': [0]},
    )
    assert list(ds.coords) == ['LATITUDE', 'LONGITUDE', 'TIME']  # problematic starting order
    result = xr_funcs.reorder_coords_to_end(ds, ['LATITUDE', 'LONGITUDE'])
    assert list(result.coords) == ['TIME', 'LATITUDE', 'LONGITUDE']


def test_reorder_coords_to_end_accepts_single_string():
    ds = xr.Dataset(
        {'TEMP': ('TIME', np.array([1.0]))},
        coords={'LATITUDE': ((), 80.0), 'TIME': [0]},
    )
    result = xr_funcs.reorder_coords_to_end(ds, 'LATITUDE')
    assert list(result.coords) == ['TIME', 'LATITUDE']


def test_reorder_coords_to_end_ignores_nonexistent_names():
    ds = xr.Dataset({'TEMP': ('TIME', np.array([1.0]))}, coords={'TIME': [0]})
    result = xr_funcs.reorder_coords_to_end(ds, ['NONEXISTENT'])
    assert list(result.coords) == ['TIME']


def test_promote_cf_coordinates_moves_newly_promoted_coords_to_end(mock_dataset):
    """promote_cf_coordinates should place newly-promoted coordinates
    after existing ones (e.g. after TIME/PRES dimension coordinates),
    not wherever they happened to sit in the file's own variable order."""
    ds = mock_dataset.copy()
    ds['TEMP'].attrs['coordinates'] = 'ZONE'
    result = xr_funcs.promote_cf_coordinates(ds)
    coord_order = list(result.coords)
    assert coord_order[-1] == 'ZONE'


# ---------------------------------------------------------------------
# 0.5.1 fixes: time_average partial bins, min_fraction, pandas 3 day bins,
# time_as_datetime only decoding the time coordinate
# ---------------------------------------------------------------------
import warnings


def test_time_average_warns_about_partial_edge_bins(make_ctd_ds):
    ds = make_ctd_ds()  # starts 11:20 -> partial first and last bin for '1D'
    with pytest.warns(UserWarning) as record:
        time_average(ds, "1D")
    messages = [str(w.message) for w in record]
    assert any("first" in m for m in messages)
    assert any("last" in m for m in messages)


def test_time_average_no_warning_for_complete_bins(make_ctd_ds):
    ds = make_ctd_ds(start="2021-11-09 00:00", n=144 * 5)  # 5 full days
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = time_average(ds, "1D")
    assert out.sizes["TIME"] == 5


def test_time_average_min_fraction_drops_incomplete_edge_bins(make_ctd_ds):
    ds = make_ctd_ds()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = xr_funcs.time_as_datetime(time_average(ds, "1D"))
    kept = xr_funcs.time_as_datetime(time_average(ds, "1D", min_fraction=0.9))
    assert kept.sizes["TIME"] == full.sizes["TIME"] - 2  # first and last bin
    np.testing.assert_allclose(
        kept.TEMP.values, full.TEMP.sel(TIME=kept.TIME.values).values)


def test_time_average_min_fraction_validation(make_ctd_ds):
    ds = make_ctd_ds()
    with pytest.raises(ValueError):
        time_average(ds, "1D", min_fraction=1.5)
    with pytest.raises(ValueError):  # calendar-based interval
        time_average(ds, "1ME", min_fraction=0.5)


def test_time_average_day_interval_honours_origin(make_ctd_ds):
    ds = make_ctd_ds()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = xr_funcs.time_as_datetime(time_average(
            ds, "1D", label="left", origin="2021-11-09 11:20"))
    assert out.TIME.values[0] == np.datetime64("2021-11-09T11:20:00")
    assert out.TIME.values[1] == np.datetime64("2021-11-10T11:20:00")


def test_time_average_default_origin_is_midnight(make_ctd_ds):
    ds = make_ctd_ds()  # starts 2021-11-09 11:20
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = xr_funcs.time_as_datetime(time_average(ds, "1D", label="left"))
    assert out.TIME.values[0] == np.datetime64("2021-11-09T00:00:00")
    assert out.TIME.values[1] == np.datetime64("2021-11-10T00:00:00")


def test_time_average_default_origin_aligns_multiday_bins():
    """Datasets starting on different days share 7-day bin edges (pandas'
    own default, origin='start_day', would not)."""
    def mk(start):
        t = pd.date_range(start, periods=30 * 24, freq="1h")
        return xr.Dataset({"TEMP": ("TIME", np.ones(len(t)))},
                          coords={"TIME": t})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a = time_average(mk("2021-11-09"), "7D", label="left")
        b = time_average(mk("2021-11-12"), "7D", label="left")
    shared = np.intersect1d(a.TIME.values, b.TIME.values)
    assert len(shared) >= 3
    assert a.TIME.values[0] == np.datetime64("2021-11-04")
    assert b.TIME.values[0] == np.datetime64("2021-11-11")


def test_time_average_calendar_interval_default_origin_does_not_warn():
    ds = xr.Dataset({"TEMP": ("TIME", np.arange(90.0))},
                    coords={"TIME": pd.date_range("2021-01-01", periods=90)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = time_average(ds, "MS", label="left")
    assert out.TIME.values[0] == np.datetime64("2021-01-01")


def test_time_average_1D_and_24h_agree(make_ctd_ds):
    ds = make_ctd_ds()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a = time_average(ds, "1D")
        b = time_average(ds, "24h")
    np.testing.assert_allclose(a.TEMP.values, b.TEMP.values)


def test_time_as_datetime_leaves_duration_variables_alone(make_ctd_ds):
    ds = make_ctd_ds()
    ds["DUR"] = ("TIME", np.arange(ds.sizes["TIME"], dtype=float),
                 {"units": "seconds"})
    decoded = xr_funcs.time_as_datetime(ds)
    assert decoded.DUR.dtype == np.float64
    assert decoded.DUR.attrs["units"] == "seconds"
    assert np.issubdtype(decoded.TIME.dtype, np.datetime64)
    back = xr_funcs.time_as_float(decoded)
    np.testing.assert_array_equal(back.TIME.values, ds.TIME.values)
    assert back.DUR.attrs["units"] == "seconds"

# ---------------------------------------------------------------------
# CF time bounds (TIME_bnds), cell_methods and the time_averaging attribute
# ---------------------------------------------------------------------

def _hourly_ds(start="2021-01-01", periods=24 * 70):
    t = pd.date_range(start, periods=periods, freq="1h")
    return xr.Dataset({"TEMP": ("TIME", np.arange(float(len(t))))},
                      coords={"TIME": t})


@pytest.mark.filterwarnings("ignore:time_average:UserWarning")
@pytest.mark.parametrize("label", ["left", "center", "right"])
def test_time_average_bounds_are_bin_edges_whatever_the_label(make_ctd_ds, label):
    ds = xr_funcs.time_as_datetime(make_ctd_ds())  # starts 2021-11-09 11:20
    out = time_average(ds, "1D", label=label)
    bnds = out.TIME_bnds.values
    assert out.TIME_bnds.dims == ("TIME", "bnds")
    assert "TIME_bnds" in out.coords and out.TIME.attrs["bounds"] == "TIME_bnds"
    assert bnds[0, 0] == np.datetime64("2021-11-09T00:00")
    np.testing.assert_array_equal(bnds[:, 1] - bnds[:, 0], np.timedelta64(1, "D"))
    expected = {"left": bnds[:, 0], "right": bnds[:, 1],
                "center": bnds[:, 0] + np.timedelta64(12, "h")}[label]
    np.testing.assert_array_equal(out.TIME.values, expected)


def test_time_average_bounds_rows_follow_min_fraction(make_ctd_ds):
    ds = xr_funcs.time_as_datetime(make_ctd_ds())
    out = time_average(ds, "1D", min_fraction=0.9)
    # first (partial) bin dropped: bounds start the next midnight, and each
    # remaining row still brackets its own timestamp
    assert out.TIME_bnds.values[0, 0] == np.datetime64("2021-11-10T00:00")
    assert out.sizes["TIME"] == out.TIME_bnds.shape[0]
    assert (out.TIME_bnds.values[:, 0] < out.TIME.values).all()
    assert (out.TIME.values < out.TIME_bnds.values[:, 1]).all()


@pytest.mark.filterwarnings("ignore:time_average:UserWarning")
def test_time_average_numeric_time_gives_numeric_bounds_in_same_units(make_ctd_ds):
    ds = make_ctd_ds()  # numeric TIME, days since 1970-01-01
    out = time_average(ds, "1D")
    assert out.TIME_bnds.dtype == np.float64
    assert out.TIME_bnds.attrs == {}  # CF: inherited from TIME, not repeated
    # days since 1970 at midnight are whole numbers; centre is +0.5
    np.testing.assert_array_equal(out.TIME_bnds.values % 1, 0)
    np.testing.assert_allclose(out.TIME.values, out.TIME_bnds.values[:, 0] + 0.5)
    decoded = xr.coding.times.decode_cf_datetime(
        out.TIME_bnds.values, out.TIME.attrs["units"])
    assert decoded[0, 0] == np.datetime64("2021-11-09T00:00")


@pytest.mark.filterwarnings("ignore:time_average:UserWarning")
def test_time_average_keeps_time_attributes(make_ctd_ds):
    ds = make_ctd_ds()
    ds["TIME"].attrs.update({"standard_name": "time", "axis": "T"})
    out = time_average(ds, "1D")
    assert out.TIME.attrs["standard_name"] == "time"
    assert out.TIME.attrs["axis"] == "T"
    assert out.TIME.attrs["units"] == "days since 1970-01-01"


@pytest.mark.parametrize("interval", ["MS", "ME"])
def test_time_average_calendar_month_bounds_are_whole_months(interval):
    out = time_average(_hourly_ds(), interval, label="left")
    np.testing.assert_array_equal(
        out.TIME_bnds.values,
        np.array([["2021-01-01", "2021-02-01"], ["2021-02-01", "2021-03-01"],
                  ["2021-03-01", "2021-04-01"]], dtype="datetime64[ns]"))
    np.testing.assert_array_equal(out.TIME.values, out.TIME_bnds.values[:, 0])
    # hourly values 0, 1, 2, ...: January holds hours 0..743
    assert out.TEMP.values[0] == np.mean(np.arange(744.0))


def test_time_average_sample_on_bin_edge_goes_to_later_bin():
    """[start, end): a sample exactly at 1 Feb 00:00 belongs to February."""
    t = pd.DatetimeIndex(["2021-01-15", "2021-01-31 23:00", "2021-02-01",
                          "2021-02-10"])
    ds = xr.Dataset({"TEMP": ("TIME", [1.0, 1.0, 5.0, 5.0])},
                    coords={"TIME": t})
    for interval in ["MS", "ME"]:
        out = time_average(ds, interval, label="left")
        np.testing.assert_array_equal(out.TEMP.values, [1.0, 5.0])


def test_time_average_weekly_bounds_run_monday_to_monday():
    out = time_average(_hourly_ds("2021-11-03", 24 * 21), "W", label="right")
    starts = pd.DatetimeIndex(out.TIME_bnds.values[:, 0])
    assert (starts.dayofweek == 0).all()  # Monday
    np.testing.assert_array_equal(out.TIME_bnds.values[:, 1] - out.TIME_bnds.values[:, 0],
                                  np.timedelta64(7, "D"))
    np.testing.assert_array_equal(out.TIME.values, out.TIME_bnds.values[:, 1])


@pytest.mark.filterwarnings("ignore:time_average:UserWarning")
def test_time_average_sets_cell_methods_and_time_averaging(make_ctd_ds):
    ds = make_ctd_ds()
    ds["ZONE"] = ("ZONE_DIM", np.array([1.0, 2.0]))  # no TIME dimension
    out = time_average(ds, "1D")
    for var in ["TEMP", "PSAL", "PRES", "CNDC"]:
        assert out[var].attrs["cell_methods"] == "TIME: mean"
    assert "cell_methods" not in out["ZONE"].attrs
    assert out.TIME.attrs["time_averaging"] == (
        "Time stamp at the CENTRE of each 1-day averaging bin. Each bin "
        "includes samples from its start time up to, but not including, its "
        "end time; bin start and end times are in TIME_bnds.")


@pytest.mark.filterwarnings("ignore:time_average:UserWarning")  # partial edge bins
@pytest.mark.parametrize("interval, label, words", [
    ("6h", "left", "START of each 6-hour"),
    ("90min", "right", "END of each 90-minute"),
    ("MS", "center", "CENTRE of each calendar-month"),
    ("YE", "left", "START of each calendar-year"),
])
def test_time_average_time_averaging_text(interval, label, words):
    out = time_average(_hourly_ds(periods=24 * 400), interval, label=label)
    assert out.TIME.attrs["time_averaging"].startswith(f"Time stamp at the {words} ")


@pytest.mark.filterwarnings("ignore:time_average:UserWarning")  # partial edge bins
def test_time_average_closed_right_is_described():
    out = time_average(_hourly_ds(periods=48), "1D", label="left", closed="right")
    assert "after its start time, up to and including its end time" in \
        out.TIME.attrs["time_averaging"]


@pytest.mark.filterwarnings("ignore:time_average:UserWarning")
def test_time_average_cell_methods_appended_and_bounds_replaced(make_ctd_ds):
    ds = make_ctd_ds()
    ds["TEMP"].attrs["cell_methods"] = "DEPTH: mean"
    daily = time_average(ds, "1D")
    assert daily.TEMP.attrs["cell_methods"] == "DEPTH: mean TIME: mean"

    weekly = time_average(daily, "7D", label="left")
    # averaging again does not repeat the entry, and the bounds describe
    # the new (7-day) bins rather than being averaged themselves
    assert weekly.TEMP.attrs["cell_methods"] == "DEPTH: mean TIME: mean"
    np.testing.assert_allclose(
        weekly.TIME_bnds.values[:, 1] - weekly.TIME_bnds.values[:, 0], 7.0)
    assert weekly.TIME.attrs["time_averaging"].startswith(
        "Time stamp at the START of each 7-day")


@pytest.mark.filterwarnings("ignore:time_average:UserWarning")
@pytest.mark.parametrize("label, name", [
    ("left", "Time (start of averaging bin)"),
    ("center", "Time (centre of averaging bin)"),
    ("right", "Time (end of averaging bin)")])
def test_time_average_sets_time_long_name(make_ctd_ds, label, name):
    ds = make_ctd_ds()
    ds["TIME"].attrs["long_name"] = "Time of measurement"
    out = time_average(ds, "1D", label=label)
    assert out.TIME.attrs["long_name"] == name


@pytest.mark.filterwarnings("ignore:time_average:UserWarning")
def test_standard_attrs_after_averaging_keep_averaged_long_name(make_ctd_ds):
    from kval.metadata.conventionalize import add_standard_var_attrs
    out = add_standard_var_attrs(time_average(make_ctd_ds(), "1D"))
    assert out.TIME.attrs["long_name"] == "Time (centre of averaging bin)"


@pytest.mark.filterwarnings("ignore:time_average:UserWarning")
def test_time_average_removes_input_time_coverage_resolution(make_ctd_ds):
    ds = make_ctd_ds()
    ds.attrs["time_coverage_resolution"] = "PT10M"
    ds.attrs["title"] = "keep me"
    out = time_average(ds, "1D")
    assert "time_coverage_resolution" not in out.attrs
    assert out.attrs["title"] == "keep me"


def test_time_as_float_encodes_bounds_with_time_units():
    t = pd.date_range("2021-01-01", periods=3)
    ds = xr.Dataset(coords={"TIME": t})
    ds = ds.assign_coords(TIME_bnds=(("TIME", "bnds"),
                                     np.stack([t, t + pd.Timedelta("1D")], 1)))
    ds["TIME"].attrs.update({"bounds": "TIME_bnds", "standard_name": "time"})
    out = xr_funcs.time_as_float(ds, units="days since 2021-01-01")
    np.testing.assert_array_equal(out.TIME.values, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(out.TIME_bnds.values,
                                  [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    assert "TIME_bnds" in out.coords
    assert out.TIME.attrs["standard_name"] == "time"
    assert out.TIME.attrs["bounds"] == "TIME_bnds"


def test_promote_cf_coordinates_promotes_bounds_and_global_coordinates():
    ds = xr.Dataset({"TEMP": ("TIME", [1.0, 2.0]),
                     "TIME_bnds": (("TIME", "bnds"), [[0.0, 1.0], [1.0, 2.0]])},
                    coords={"TIME": ("TIME", [0.5, 1.5], {"bounds": "TIME_bnds"})},
                    attrs={"coordinates": "TIME_bnds", "title": "x"})
    out = xr_funcs.promote_cf_coordinates(ds)
    assert "TIME_bnds" in out.coords
    assert "coordinates" not in out.attrs
    assert out.attrs["title"] == "x"


# ---------------------------------------------------------------------
# time_average: optional bin statistics (stats=...)

def _stats_ds():
    t = pd.date_range("2024-01-01", periods=48, freq="1h")
    ds = xr.Dataset({"TEMP": ("TIME", np.arange(48.0)),
                     "PSAL": ("TIME", np.full(48, 35.0))},
                    coords={"TIME": t})
    ds.TEMP.attrs = {"units": "degree_Celsius",
                     "standard_name": "sea_water_temperature",
                     "long_name": "Sea water temperature",
                     "units_metadata": "temperature: on_scale",
                     "valid_min": -3.0, "valid_max": 40.0}
    ds["TEMP"][5] = np.nan
    return ds


def test_time_average_default_is_mean_only():
    out = time_average(_stats_ds(), "1D")
    assert set(out.data_vars) == {"TEMP", "PSAL"}
    assert "ancillary_variables" not in out.TEMP.attrs


def test_time_average_n_samples():
    out = time_average(_stats_ds(), "1D", stats=["n_samples"])
    np.testing.assert_array_equal(out.TEMP_n_samples.values, [23, 24])
    np.testing.assert_array_equal(out.PSAL_n_samples.values, [24, 24])
    assert out.TEMP_n_samples.dtype == np.int32
    assert out.TEMP_n_samples.attrs["standard_name"] == "number_of_observations"
    assert out.TEMP_n_samples.attrs["units"] == "1"
    assert out.TEMP.attrs["ancillary_variables"] == "TEMP_n_samples"


def test_time_average_std_and_median():
    ds = _stats_ds()
    out = time_average(ds, "1D", stats=["std", "median"])
    day1 = ds.TEMP.values[:24]
    np.testing.assert_allclose(out.TEMP_std.values[0],
                               np.nanstd(day1, ddof=1))
    np.testing.assert_allclose(out.TEMP_median.values[0], np.nanmedian(day1))
    std_attrs = out.TEMP_std.attrs
    assert std_attrs["cell_methods"] == "TIME: standard_deviation"
    assert std_attrs["units"] == "degree_Celsius"
    assert std_attrs["units_metadata"] == "temperature: difference"
    assert "valid_min" not in std_attrs
    assert std_attrs["long_name"].startswith(
        "Standard deviation of sea water temperature")
    med_attrs = out.TEMP_median.attrs
    assert med_attrs["cell_methods"] == "TIME: median"
    assert med_attrs["valid_max"] == 40.0
    # Std and median are not ancillary variables; only the counts are
    assert "ancillary_variables" not in out.TEMP.attrs


def test_time_average_stats_order_and_unknown_raises():
    out = time_average(_stats_ds(), "1D", stats=["median", "n_samples"])
    assert list(out.data_vars) == ["TEMP", "TEMP_n_samples", "TEMP_median",
                                   "PSAL", "PSAL_n_samples", "PSAL_median"]
    assert "TIME_bnds" in out.coords
    with pytest.raises(ValueError, match="Unknown stats"):
        time_average(_stats_ds(), "1D", stats=["mean"])


def test_time_average_stats_of_averaged_data_are_replaced_not_nested():
    once = time_average(_stats_ds(), "6h", stats=["n_samples"])
    with pytest.warns(UserWarning, match="replacing"):
        twice = time_average(once, "1D", stats=["n_samples"])
    assert "TEMP_n_samples_n_samples" not in twice.data_vars
    # Now the number of 6-hour means in each day
    np.testing.assert_array_equal(twice.TEMP_n_samples.values, [4, 4])
    assert twice.TEMP.attrs["ancillary_variables"] == "TEMP_n_samples"


def test_time_average_stats_survive_min_fraction():
    out = time_average(_stats_ds().isel(TIME=slice(0, 40)), "1D",
                       min_fraction=0.9, stats=["n_samples"])
    assert out.sizes["TIME"] == 1
    assert out.TEMP_n_samples.sizes["TIME"] == 1


def test_standard_var_attrs_leave_bin_statistics_alone():
    from kval.metadata.conventionalize import add_standard_var_attrs
    out = add_standard_var_attrs(
        time_average(_stats_ds(), "1D", stats=["n_samples", "std"]))
    assert out.TEMP_n_samples.attrs["units"] == "1"
    assert "valid_min" not in out.TEMP_n_samples.attrs
    assert "valid_min" not in out.TEMP_std.attrs
    assert "reference_scale" in out.TEMP.attrs  # still added to TEMP itself
