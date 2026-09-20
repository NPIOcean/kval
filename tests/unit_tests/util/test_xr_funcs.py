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
 
 
def test_time_average_center_raises_informatively_for_calendar_based_interval():
    """label='center' isn't well-defined for calendar-based intervals
    (e.g. month-end 'ME') since they don't have a fixed duration -- this
    should raise a clear, actionable error rather than a cryptic one."""
    ds = xr.Dataset({'TEMP': ('TIME', np.arange(60.0))},
                     coords={'TIME': pd.date_range('2021-01-01', periods=60)})
    with pytest.raises(ValueError, match="label='left'"):
        time_average(ds, 'ME', label='center')
 
 
def test_time_average_origin_shifts_bin_edges():
    ds = xr.Dataset(
        {'TEMP': ('TIME', np.arange(8.0))},
        coords={'TIME': pd.date_range('2021-01-01 00:00', periods=8, freq='3h')},
    )
    result_default = time_average(ds, '6h', label='left')
    result_shifted = time_average(ds, '6h', label='left', origin='2021-01-01 03:00')
    assert result_default['TIME'].values[0] != result_shifted['TIME'].values[0]
    assert result_shifted['TIME'].values[0] == np.datetime64('2020-12-31T21:00:00')
 
