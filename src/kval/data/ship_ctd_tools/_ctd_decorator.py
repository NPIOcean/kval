'''
KVAL.DATA.SHIP_CTD_TOOLS._CTD_DECORATOR

Decorator function for recording processing steps to PROCESSING field.

For use in the kval.data.ctd module.
'''

import functools
import inspect


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
                f"ds = data.ctd.{func.__name__}(ds, "
                "{', '.join(args_list)})"
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
