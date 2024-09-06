"""
KVAL.DATA.MOORED
"""

import xarray as xr
from typing import Optional, Tuple, Union
import numpy as np
import functools
import inspect
import os
import matplotlib.pyplot as plt
from kval.file import sbe, rbr
from kval.data import dataset
from kval.util import internals
from kval.signal import despike

if internals.is_notebook():
    from IPython.display import display

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
                f"ds = data.moored.{func.__name__}(ds, "
                f"{', '.join(args_list)})"
            )

            if py_comment:
                ds["PROCESSING"].attrs["python_script"] += (
                    f"\n\n# {py_comment.format(**bound_args.arguments)}"
                    f"\n{function_call}"
                )
            else:
                ds["PROCESSING"].attrs["python_script"] += (
                    f"\n\n{function_call}")
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
    SBE (SBE37, SBE16). Mileage may vary for older file types.

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
        ds = dataset.add_processing_history_var_moored(
            ds,
        )
        # Add a source_file attribute (and remove from the global attrs)
        ds.PROCESSING.attrs["source_file"] = ds.source_files

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


# Chop record
# Note: We do the recording to PROCESSING inside the function, not in the
# decorator. (Too complex otherwise)
def chop_deck(
    ds: xr.Dataset,
    variable: str = 'PRES',
    sd_thr: float = 3.0,
    indices: Optional[Tuple[int, int]] = None,
    auto_accept: bool = False
) -> xr.Dataset:
    """
    Chop away start and end parts of a time series (xarray Dataset).
    Default behaviour is to look for the indices to remove. Looks for
    the indices where `variable` (e.g. temperature or pressure) is a
    specified number of standard deviations away from the mean (the user can
    also specify the indices to cut).

    Typical application: Remove data from a mooring record during which the
    instrument was on deck or being lowered/raised through the water column.

    The function can automatically suggest chopping based on the standard
    deviation of the specified variable or allow the user to manually input
    the desired range for chopping. A plot is displayed showing the proposed
    chop unless `auto_accept` is set to True.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the data to be chopped. The dataset
        should include the field corresponding to the specified `variable`.
    variable : str, optional
        The variable in the dataset used for determining chop boundaries.
        Defaults to 'PRES'.
    sd_thr : float, optional
        The standard deviation threshold for determining the chop boundaries
        when `indices` is not provided. Defaults to 3.0.
    indices : Optional[Tuple[int, int]], optional
        A tuple specifying the (start, stop) indices for manually chopping the
        dataset along the TIME dimension. If not provided, the function will
        use the standard deviation threshold to determine the range
        automatically. Defaults to None.
    auto_accept : bool, optional
        If `True`, automatically accepts the suggested chop based on the
        pressure record without prompting the user. Defaults to False.

    Returns
    -------
    xr.Dataset
        The chopped xarray Dataset, with the range outside the specified or
        computed boundaries removed.

    Raises
    ------
    ValueError
        If the specified `variable` is not present in the dataset or if the
        user input during the chop confirmation is invalid.
    """
    # Make sure we are working with a copy
    ds = ds.copy()

    # Confirm that `variable` exists in ds
    if variable not in ds:
        raise ValueError(
            f'Error: Cannot do chopping based on {variable} because it '
            'is not a variable in the datset.')

    if indices is None:
        # Calculate the mean and standard deviation
        chop_var = ds[variable].data
        chop_var_mean = np.ma.median(chop_var)
        chop_var_sd = np.ma.std(chop_var)

        indices = [None, None]

        # If we detect deck time at start of time series:
        # find a start index
        if chop_var[0] < chop_var_mean - sd_thr * chop_var_sd:
            indices[0] = np.where(
                np.diff(chop_var < chop_var_mean
                        - sd_thr * chop_var_sd))[0][0] + 1
        # If we detect deck time at end of time series:
        # find an end index
        if chop_var[-1] < chop_var_mean - sd_thr * chop_var_sd:
            indices[1] = np.where(
                np.diff(chop_var < chop_var_mean
                        - sd_thr * chop_var_sd))[0][-1]

        # A slice defining the suggested "good" range
        keep_slice = slice(*indices)

        if auto_accept:
            accept = "y"
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            index = np.arange(len(chop_var))

            ylab = variable
            if hasattr(ds[variable], 'units'):
                ylab = f'{ylab} [{ds[variable].units}]'

            ax.plot(index, chop_var, "k", label=variable)
            ax.plot(index[keep_slice], chop_var[keep_slice], "r",
                    label="Chopped Range")
            ax.set_xlabel("Index")
            ax.set_ylabel(ylab)
            ax.invert_yaxis()
            ax.set_title(f"Suggested chop: {indices} (to red curve).")
            ax.legend()

            # Ensure plot updates and displays (different within notebook with
            # widget backend..)
            if internals.is_notebook():
                display(fig)
            else:
                plt.show(block=False)

            print(f"Suggested chop: {indices} (to red curve)")
            accept = input("Accept (y/n)?: ")

            # Close the plot after input tochop_var avoid re-display
            plt.close(fig)

        if accept.lower() == "n":
            print("Not accepted -> Not chopping anything now.")
            print("NOTE: run chop(ds, indices =[A, B]) to manually set chop.")
            return ds

        elif accept.lower() == "y":
            pass
        else:
            raise ValueError(f'I do not understand your input "{accept}"'
                             '. Only "y" or "n" works. -> Exiting.')
    else:
        keep_slice = slice(indices[0], indices[1] + 1)

    L0 = ds.sizes["TIME"]
    print(f"Chopping to index: {indices}")
    ds = ds.isel(TIME=keep_slice)

    L1 = ds.sizes["TIME"]
    net_str = (f"Chopped {L0 - L1} samples using -> {indices} "
               f"(total samples {L0} -> {L1})")
    print(net_str)

    # Record to PROCESSING metadata variable
    if "PROCESSING" in ds:

        if keep_slice.start is None and keep_slice.stop is not None:
            start_end_str = 'end'
            indices_str = f'None, {keep_slice.stop-1}'
        elif keep_slice.start is not None and keep_slice.stop is None:
            start_end_str = 'start'
            indices_str = f'{keep_slice.start}, None'
        elif keep_slice.start is not None and keep_slice.stop is not None:
            start_end_str = 'start and end'
            indices_str = f'{keep_slice.start}, {keep_slice.stop-1}'

        ds["PROCESSING"].attrs["post_processing"] += (
            f"Chopped {L0 - L1} samples at the {start_end_str} "
            "of the time series.\n"
        )

        ds["PROCESSING"].attrs["python_script"] += (
            f'\n\n# Chopping away samples from the {start_end_str}'
            ' of the time series\n'
            f'ds = data.moored.chop_deck(ds, indices = [{indices_str}])'
        )

    return ds

# Hand edit outlier


# Programmatic edit points
def despike_rolling(
    ds: xr.Dataset,
    var_name: str,
    window_size: int,
    n_std: float,
    dim: str = 'TIME',
    filter_type: str = 'median',
    min_periods: Union[int, None] = None,
    plot: bool = False,
    verbose: bool = False,
) -> xr.Dataset:
    '''

    Despike a variable in a dataset by identifying and removing outliers
    based on a rolling mean/median and standard deviation.

    Outliers are data points where the variable deviates from the rolling
    mean/median by a number of standard deviations. Both the mean/median
    and standard deviation are calculated within a rolling window centered
    on each data point.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the variable to despike.
    var_name : str
        The name of the variable to despike.
    window_size : int
        The size of the rolling window for calculating the mean/median
        and standard deviation.
    n_std : float
        The number of standard deviations used as the threshold to identify
        outliers.
    dim : str
        The dimension along which to calculate the rolling statistics.
        Default: 'TIME'.
    filter_type : str, optional
        The type of filter to apply ('mean' or 'median'). Default is 'mean'.
    min_periods : int or None, optional
        The minimum number of observations in the window required to return
        a valid result. Default is None.
    plot : bool, optional
        If True, plots the original data and the despiking results.
        Default is False.
    verbose : bool, optional
        If True, print some basic info about the result of the despiking.
        Default is False.
    Returns
    -------

        - If `return_ds` is True and `return_index` is False: returns the
          updated dataset with the despiked variable.
        - If `return_ds` is False and `return_index` is False: returns the
          despiked variable as a DataArray.
        - If `return_ds` is True and `return_index` is True: returns a tuple
          of the updated dataset and a mask of outliers.
        - If `return_ds` is False and `return_index` is True: returns a tuple
          of the despiked variable and a mask of outliers.
    '''

    ds = despike.despike_rolling(
        ds, var_name, window_size, n_std, dim, filter_type, min_periods,
        True, False, plot, verbose)

    return ds

# Drift

# Standard metadata

# Threshold edit

# Filtering

# Recalculating sal

# Calculate gsw

# Interpolate onto new TIME

# Compare wth CTDs
