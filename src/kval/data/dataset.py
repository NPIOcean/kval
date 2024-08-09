'''
DATASET.PY

Various functions to be applied to generalized datasets.
'''

from kval.util import time
import pandas as pd




#### MODIFY METADATA

def add_now_as_date_created(D):
    '''
    Add a global attribute "date_created" with todays date.
    '''
    now_time = pd.Timestamp.now()
    now_str = time.datetime_to_ISO8601(now_time)

    D.attrs['date_created'] = now_str

    return D




def reorder_attrs(D):
    """
    Reorder global and variable attributes of a dataset based on the 
    specified order in _standard_attrs.

    Parameters:
        ds (xarray.Dataset): The dataset containing global attributes.
        ordered_list (list): The desired order of global attributes.

    Returns:
        xarray.Dataset: The dataset with reordered global attributes.
    """
    ### GLOBAL
    reordered_list = _reorder_list(D.attrs, 
                                  _standard_attrs.global_attrs_ordered)
    attrs_dict = D.attrs
    D.attrs = {}
    for attr_name in reordered_list:
        D.attrs[attr_name] = attrs_dict[attr_name]

    ### VARIABLE
    for varnm in D.data_vars:
        reordered_list_var = _reorder_list(D[varnm].attrs, 
                      _standard_attrs.variable_attrs_ordered)
        var_attrs_dict = D[varnm].attrs
        D[varnm].attrs = {}
        for attr_name in reordered_list_var:
            D[varnm].attrs[attr_name] = var_attrs_dict[attr_name]
    return D


    #### HELPER FUNCTIONS   