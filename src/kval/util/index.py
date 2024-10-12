'''
KVAL.UTIL.INDEX

Various functions related ti slicing/indexing
'''

import pandas as pd
import xarray as xr

def indices_to_slices(indices):

    if len(indices) == 0:
        return []

    slices = []
    start = indices[0]

    for i in range(1, len(indices)):
        # Check if the current index is not consecutive
        if indices[i] != indices[i-1] + 1:
            # If the slice has length 1, append the single index
            if start == indices[i-1]:
                slices.append(int(start))
            else:
                slices.append(slice(int(start), int(indices[i-1] + 1)))
            start = indices[i]

    # Handle the last slice
    if start == indices[-1]:
        slices.append(int(start))
    else:
        slices.append(slice(int(start), int(indices[-1] + 1)))

    return slices


def closest_index(arr, val):

    closest_index = abs(arr - val).argmin().item()

    return closest_index


def closest_index_time(ds, time_stamp, time_name='TIME'):

    ds_cp = xr.decode_cf(ds.copy())

    # Assuming you have a dataset `ds` with a coordinate 'time'
    # First, use sel to select the value closest to a point
    selected_data = ds_cp.sel({time_name: time_stamp}, method='nearest')

    # Get the index of the 'time' coordinate where the selection happened
    time_index = ds_cp.get_index(time_name).get_loc(selected_data[time_name].values)

    return time_index

