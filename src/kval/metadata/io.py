'''
KVAL.METADATA.IO


Import/export metadata between NetCDF/xarray and human-readable yaml
'''

import xarray as xr
import yaml
import numpy as np

# Optional: fallback to safe dump
def clean_metadata(obj):
    if isinstance(obj, dict):
        return {k: clean_metadata(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_metadata(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # numpy array → native list
    elif isinstance(obj, np.generic):
        return obj.item()  # numpy scalar → native Python type
    else:
        return obj

# Define LiteralStr first
class LiteralStr(str): pass

# Define str_presenter before registering it
def str_presenter(dumper, data):
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

# Define a custom Dumper
class LiteralDumper(yaml.SafeDumper):
    pass

# Register the presenter with the custom dumper
yaml.add_representer(LiteralStr, str_presenter, Dumper=LiteralDumper)

def make_multiline_literals(obj):
    """Wrap multiline or long strings with LiteralStr to trigger YAML | block style."""
    if isinstance(obj, dict):
        return {k: make_multiline_literals(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_multiline_literals(v) for v in obj]
    elif isinstance(obj, str):
        if '\n' in obj or len(obj) > 100:  # force wrap long strings too
            return LiteralStr(obj)
        else:
            return obj
    else:
        return obj




def export_metadata(ds, filename):
    meta = {
        "global_attributes": ds.attrs.copy(),
        "variables": {var: ds[var].attrs.copy() for
                      var in list(ds.coords) + list(ds.data_vars)}
    }
    # Clean and process for YAML output
    meta = clean_metadata(meta)
    meta = make_multiline_literals(meta)

    with open(filename, 'w') as f:
        yaml.dump(meta, f, sort_keys=False, Dumper=LiteralDumper,
                  allow_unicode=True)



# Load metadata from YAML and update Dataset
def import_metadata(ds, filename, replace_existing=True):
    """
    Import metadata from a YAML file into an xarray Dataset.

    Parameters:
    - ds: xarray.Dataset — the dataset to update.
    - filename: str — path to YAML file.
    - replace_existing: bool — whether to override existing attributes.
    """
    with open(filename) as f:
        meta = yaml.safe_load(f)

    # Update global attributes
    for key, val in meta.get("global_attributes", {}).items():
        if replace_existing or key not in ds.attrs:
            ds.attrs[key] = val

    # Update variable attributes
    for var, attrs in meta.get("variables", {}).items():
        if var not in ds:
            continue  # skip variables not in dataset
        for key, val in attrs.items():
            if replace_existing or key not in ds[var].attrs:
                ds[var].attrs[key] = val

    return ds