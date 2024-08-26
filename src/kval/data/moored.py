"""
KVAL.DATA.MOORED
"""

import xarray as xr
from typing import List, Optional, Union
import numpy as np
import functools
import inspect
import os
from kval.file import sbe, rbr
from kval.data import dataset, edit


# DECORATOR TO PRESERVE PROCESSING STEPS IN METADATA


def record_processing(description_template, py_comment=None):
    """
    A decorator to record processing steps and their input arguments in the
    dataset's metadata.

    Parameters:
    - description_template (str): A template for the description that includes
                                  placeholders for input arguments.

    Returns:
    - decorator function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(ds, *args, **kwargs):

            # Apply the function
            ds = func(ds, *args, **kwargs)

            # Check if the 'PROCESSING' field exists in the dataset
            if "PROCESSING" not in ds:
                # If 'PROCESSING' is not present, return the dataset without
                # any changes
                return ds

            # Prepare the description with input arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(ds, *args, **kwargs)
            bound_args.apply_defaults()

            # Format the description template with the actual arguments
            description = description_template.format(**bound_args.arguments)

            # Record the processing step
            ds["PROCESSING"].attrs["post_processing"] += description + "\n"

            # Prepare the function call code with arguments
            args_list = []
            for name, value in bound_args.arguments.items():
                # Skip the 'ds' argument as it's always present
                if name != "ds":
                    default_value = sig.parameters[name].default
                    if value != default_value:
                        if isinstance(value, str):
                            args_list.append(f"{name}='{value}'")
                        else:
                            args_list.append(f"{name}={value}")

            function_call = (
                f"ds = data.moored.{func.__name__}(ds, " f"{', '.join(args_list)})"
            )

            if py_comment:
                ds["PROCESSING"].attrs["python_script"] += (
                    f"\n\n# {py_comment.format(**bound_args.arguments)}"
                    f"\n{function_call}"
                )
            else:
                ds["PROCESSING"].attrs["python_script"] += f"\n\n{function_call}"
            return ds

        return wrapper

    return decorator


def load_moored(
    file: str,
    processing_variable=True,
) -> xr.Dataset:
    """
    Load moored instrument data from a file into an xarray Dataset.

    Should be able to read instruments from RBR (Concerto, Solo..) and
    SBE (SBE37, SBE16). Mlieage may vary for older file types.

    Parameters:
    - file (str):
        Path to the file.
    - processing_variable (bool):
        Whether to add processing history to the dataset.

    Returns:
    - xr.Dataset:
        The loaded dataset.
    """
    # Check file type and return an error if invalid
    if file.endswith(".rsk"):
        instr_type = "RBR"
    elif file.endswith(".cnv"):
        instr_type = "SBE"
    else:
        raise ValueError(
            f"Unable to load moored instrument {os.path.basename(file)}.\n"
            "Supported files are .cnv (SBE) and .rsk (RBR)."
        )

    # Load data
    if instr_type == "RBR":
        ds = rbr.read_rsk(file)
    elif instr_type == "SBE":
        ds = sbe.read_cnv(file)

    # Add PROCESSING variable with useful metadata
    # ( + remove some excessive global attributes)
    if processing_variable:
        ds = dataset.add_processing_history_var_ctd(
            ds,
        )
        # Add a source_file attribute (and remove from the global attrs)
        ds.PROCESSING.attrs["source_file"]= ds.source_files

        # Remove some unwanted global atributes
        for attr_name in ['source_files', 'filename',
                          'SBE_flags_applied', 'SBE_processing_date']:
            if attr_name in ds.attrs:
                del ds.attrs[attr_name]

        # Add python scipt snipped to reproduce this operation
        ds.PROCESSING.attrs["python_script"] += f"""from kval import data

data_dir = "./" # Directory containing `filename` (MUST BE SET BY THE USER!)
filename = "{os.path.basename(file)}"

# Load file into an xarray Dataset:
ds = data.moored.load_moored(
    data_dir + filename,
    processing_variable={processing_variable}
    )"""

    return ds
