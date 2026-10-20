import matplotlib as mpl
import functools
from matplotlib import font_manager
import warnings
import contextlib
import traceback


def is_notebook():
    """
    Check if the current environment is a Jupyter notebook (or another
    kernel-backed IPython front-end, e.g. JupyterLab, VS Code, Colab).

    Returns:
        bool: True if running in a kernel-backed IPython environment,
              False otherwise (e.g. plain terminal IPython or script).
    """
    # Imported explicitly: the bare name get_ipython is only available as a
    # builtin while a cell is executing, not in widget callbacks (e.g. a
    # button click), where this used to wrongly return False.
    try:
        from IPython import get_ipython
    except ImportError:
        return False  # No IPython at all
    shell = get_ipython()

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


def make_figure(*args, **kwargs):
    """
    plt.subplots() with ipympl's auto-display suppressed.

    Under the widget backend, creating a figure while interactive mode
    is on schedules an immediate display of a still-empty canvas, which
    then races with the drawing that follows. Creating it under ioff()
    means nothing is shown until show_figure() is called explicitly.

    Always pair with show_figure().
    """
    import matplotlib.pyplot as plt
    with plt.ioff():
        return plt.subplots(*args, **kwargs)


def show_figure(fig):
    """
    Display a figure built with make_figure(). Outside a notebook,
    falls back to plt.show().

    Note the order: the canvas is displayed *before* drawing. Under
    ipympl the canvas has to be attached to the browser before anything
    can be painted into it -- drawing first can paint into a canvas the
    front end hasn't connected to yet, which shows up as a blank or
    half-rendered plot that only fills in once the user interacts with
    it. The figure is then drawn with fig.canvas.draw() (a synchronous
    draw), if the canvas has a figure manager.
    """
    import matplotlib.pyplot as plt
    from IPython.display import display

    if not is_notebook():
        plt.show()
        return

    display(fig.canvas)
    if getattr(fig.canvas, 'manager', None) is not None:
        fig.canvas.draw()


def _widget_bundle(widget):
    """
    The widget's display data (a dict with the widget-view mime type), or
    None if it can't be had -- e.g. ipywidgets 7, which has no
    _repr_mimebundle_. Callers then fall back to a plain display(widget).
    """
    repr_bundle = getattr(widget, '_repr_mimebundle_', None)
    if repr_bundle is None:
        return None
    try:
        bundle = repr_bundle()
    except Exception:
        return None
    if not bundle or 'application/vnd.jupyter.widget-view+json' not in bundle:
        return None
    return dict(bundle)


def display_with_snapshot(widget, fig=None):
    """
    Display an interactive widget, with a static picture of `fig` attached.

    In a live notebook only the interactive widget is shown. Viewers that
    cannot run widgets (e.g. GitHub's notebook preview) show the picture
    instead: `fig` as it looks when the widget is first displayed. This is
    the same fallback ipympl gives plain figures: one output holding both
    the widget and a PNG.

    If no picture can be made (no figure, or not in a notebook), this is
    just display(widget).
    """
    import base64
    import io
    from IPython.display import display

    if fig is None or not is_notebook():
        display(widget)
        return

    bundle = _widget_bundle(widget)
    if bundle is None:
        display(widget)
        return

    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi='figure')
    except Exception:
        display(bundle, raw=True)  # The widget alone, without a picture
        return

    png = base64.b64encode(buf.getvalue()).decode('ascii')
    # Display width in CSS pixels (ipympl doubles the dpi on HiDPI screens)
    width = int(fig.get_figwidth() * fig.dpi
                / getattr(fig.canvas, 'device_pixel_ratio', 1))
    bundle['image/png'] = png
    bundle['text/html'] = (
        f"<img src='data:image/png;base64,{png}' width={width}/>")
    display(bundle, raw=True)


def display_with_text(widget, text):
    """
    Display an interactive widget, with `text` attached as a static copy.

    Like display_with_snapshot(), but for widgets that show text (e.g. a
    report): live notebooks show the widget, viewers that cannot run
    widgets (e.g. GitHub's notebook preview) show `text` as preformatted
    text.
    """
    import html
    from IPython.display import display

    if not text or not is_notebook():
        display(widget)
        return

    bundle = _widget_bundle(widget)
    if bundle is None:
        display(widget)
        return

    bundle['text/plain'] = text
    bundle['text/html'] = f"<pre>{html.escape(text)}</pre>"
    display(bundle, raw=True)



@contextlib.contextmanager
def loud_output(output_widget):
    """
    Like `with output_widget:`, but any traceback is also printed to the
    real stderr so real errors dont drown and cause bad silent failures..

    ipywidgets' Output captures exceptions into the widget, so a failing
    callback looks like a dead button rather than an error -- especially
    if the widget isn't visible or gets cleared.
    """
    try:
        with output_widget:
            yield
    except Exception:
        traceback.print_exc()
        raise

@functools.lru_cache(maxsize=None)
def plot_font_family(preferred='Arial'):
    """
    Return `preferred` if matplotlib can find that font, else 'sans-serif'.

    Passing a missing font name to matplotlib works (it falls back to its
    default), but logs "findfont: Font family ... not found" every time a
    text element is drawn. Resolving the name once avoids that on systems
    without the font (common on Linux).
    """
    try:
        font_manager.findfont(preferred, fallback_to_default=False)
    except ValueError:
        return 'sans-serif'
    return preferred
