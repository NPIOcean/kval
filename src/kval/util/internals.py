import matplotlib as mpl
import warnings


def is_notebook():
    """
    Check if the current environment is a Jupyter notebook (or another
    kernel-backed IPython front-end, e.g. JupyterLab, VS Code, Colab).

    Returns:
        bool: True if running in a kernel-backed IPython environment,
              False otherwise (e.g. plain terminal IPython or script).
    """
    try:
        shell = get_ipython()
    except NameError:
        return False  # Standard Python interpreter, no IPython at all

    if shell is None:
        return False

    # Kernel-backed shells (Jupyter Notebook/Lab, VS Code, Colab,
    # qtconsole) expose a `.kernel` attribute; plain terminal IPython
    # does not.
    return hasattr(shell, 'kernel')



def check_interactive():
    """
    Check whether the current Matplotlib backend supports interactive editing
    and widgets within a Jupyter notebook environment.

    This function raises a warning if the current Matplotlib backend does not
    support interactive features in a Jupyter notebook.
    """

    # For consistency, ensure that warnings are always displayed
    warnings.simplefilter("always", UserWarning)

    if is_notebook():
        current_backend = mpl.get_backend()
        notebook_interactive_backends = [
            "module://ipympl.backend_nbagg",
            "widget",  # `%matplotlib widget`
        ]
        if current_backend not in notebook_interactive_backends:
            warnings.warn(
                "You are trying to use an interactive function from *kval*. "
                "However, it looks like you are running a Jupyter notebook "
                f'with the Matplotlib backend "{current_backend}", which may '
                "not fully support interactive plots or widgets. To ensure"
                " full interactivity, switch to the recommended backend by"
                " running the following in your Jupyter notebook:\n---\n"
                "> %matplotlib widget\n---",
            )
    else:
        warnings.warn(
            "You are trying to use an interactive function from *kval*. "
            "However, it looks like you are not running this script in a "
            "Jupyter notebook. Consider using a non-interactive function "
            "instead, or running the code in a Jupyter notebook (remember "
            'to execute "%matplotlib widget" at the top of your notebook).',
        )
