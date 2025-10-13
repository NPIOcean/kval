'''
KVAL.METADATA.IO


Import/export metadata between NetCDF/xarray and human-readable yaml
'''

import xarray as xr
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString
import numpy as np

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)

# Optional: clean metadata (convert numpy arrays and scalars)
def clean_metadata(obj):
    if isinstance(obj, dict):
        return {k: clean_metadata(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_metadata(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

# Recursively wrap multiline strings as LiteralScalarString
def make_literals(obj):
    if isinstance(obj, dict):
        return {k: make_literals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_literals(v) for v in obj]
    elif isinstance(obj, str) and ('\n' in obj or len(obj) > 100):
        # Use literal block for multiline or very long strings
        return LiteralScalarString(obj)
    else:
        return obj

def export_metadata(ds, filename):
    """
    Export xarray Dataset metadata to YAML in a human-readable format.
    Preserves multiline strings with '|' block style.
    """
    meta = {
        "global_attributes": ds.attrs.copy(),
        "variables": {var: ds[var].attrs.copy() for var in list(ds.coords) + list(ds.data_vars)}
    }

    meta = clean_metadata(meta)
    meta = make_literals(meta)

    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(meta, f)



# Load metadata from YAML and update Dataset
def import_metadata(ds, filename, replace_existing=True):
    """
    Import metadata from a YAML file into an xarray Dataset.

    Will skip empty values.

    Parameters:
    - ds: xarray.Dataset — the dataset to update.
    - filename: str — path to YAML file.
    - replace_existing: bool — whether to override existing attributes.
    """
    with open(filename) as f:
        meta = yaml.load(f)

    def strip_trailing_newline(val):
        if isinstance(val, str):
            return val.rstrip('\n')
        return val

    def is_empty(val):
        if val is None:
            return True
        if isinstance(val, str) and val.strip() == "":
            return True
        if isinstance(val, (list, dict)) and len(val) == 0:
            return True
        return False


    # Update global attributes
    for key, val in meta.get("global_attributes", {}).items():
        val = strip_trailing_newline(val)
        if is_empty(val):
            continue  # skip empty attributes
        if replace_existing or key not in ds.attrs:
            ds.attrs[key] = val

    # Update variable attributes
    for var, attrs in meta.get("variables", {}).items():
        if var not in ds:
            continue  # skip variables not in dataset
        for key, val in attrs.items():
            val = strip_trailing_newline(val)
            if is_empty(val):
                continue  # skip empty attributes
            if replace_existing or key not in ds[var].attrs:
                ds[var].attrs[key] = val

    return ds