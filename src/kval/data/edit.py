'''
EDIT.PY

Functions for editing (generalized) datasets.
'''


import numpy as np

# Remove points 
# - Array of (PRES) points at a given TIME
#  


def remove_points_profile(ds, varnm, TIME_index, remove_inds):
    """
    Remove specified points from a profile in the dataset by setting them to NaN.
    
    Parameters:
    - ds: xarray.Dataset
      The dataset containing the variable to modify.
    - varnm: str
      The name of the variable to modify.
    - TIME_index: int
      The index along the TIME dimension to modify.
    - remove_inds: list or array-like
      Indices of points to remove (set to NaN) within the specified TIME profile.
    
    Returns:
    - ds: xarray.Dataset
      The dataset with specified points removed (set to NaN).
    """

    # Create a boolean array for removal
    remove_bool = np.zeros(len(ds[varnm].isel(TIME=TIME_index)), dtype=bool)
    remove_bool[remove_inds] = True

    # Use the `where` method to set the selected points to NaN
    ds[varnm].isel(TIME=TIME_index).values[:] = np.where(remove_bool, 
                                                         np.nan, 
                                                         ds[varnm].isel(TIME=TIME_index).values)

    return ds



# Apply fixed offset

# Apply drift correction

# Drop variables

# Threshold edit variable