
import matplotlib as mpl
import warnings

def is_notebook():
    """
    Check if the current environment is a Jupyter notebook.

    Returns:
        bool: True if the code is running in a Jupyter notebook or Jupyter QtConsole, 
              False otherwise.
    """

    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or Jupyter QtConsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (probably standard Python interpreter)
    except NameError:
        return False  # Probably standard Python interpreter

def check_plots_interactive():
    """
    Check whether the current Matplotlib backend supports interactive editing and widgets 
    within a Jupyter notebook environment.

    This function first checks if the code is running in a Jupyter notebook. If it is, it raises 
    a warning if the current Matplotlib backend does not support interactive features, such as 
    hand-editing in plots or using interactive widgets.

    Accepted interactive backends for Jupyter notebooks include:
    - 'nbAgg' (used by `%matplotlib notebook`)
    - 'module://ipympl.backend_nbagg' (used by `%matplotlib widget`)
    - 'module://matplotlib_inline.backend_inline' (used by `%matplotlib inline` but has limited interactivity)

    Raises:
        UserWarning: If the current Matplotlib backend does not support interactive plots 
        and widgets in a Jupyter notebook.

    Example:
        >>> check_plots_interactive()
        # If using a non-interactive or unsupported backend in a Jupyter notebook, a warning will be raised.
    """

    if is_notebook():
        current_backend = mpl.get_backend()
        notebook_interactive_backends = [
            'module://ipympl.backend_nbagg',        # `%matplotlib widget`
        ]
        print(current_backend)
        if current_backend not in notebook_interactive_backends:
            warnings.warn(
                'You are trying to use an interactive function from *kval*. '
                'However, it looks like you are running a Jupyter notebook '
                f'with the Matplotlib backend "{current_backend}", which may '
                'not fully support interactive plots or widgets. To ensure'
                ' full interactivity, switch to the recommended backend by'
                ' running the following in your Jupyter notebook:\n---\n'
                '> %matplotlib widget\n---',
                UserWarning
            )
    else:

        warnings.warn(
            'You are trying to use an interactive function from *kval*.'
            ' However, it looks like you are not running this script in a '
            'Jupyter notebook. Consider using a non-interactive function '
            'instead, or running the code in a Jupyter notebook (remember '
            'to execute "%matplotlib widget" at the top of your notebook).',
            UserWarning
        )

        