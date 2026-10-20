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


_WIDGET_VIEW = 'application/vnd.jupyter.widget-view+json'


def _ipywidgets_has_bundles():
    """False for ipywidgets 7, whose widgets have no _repr_mimebundle_."""
    try:
        import ipywidgets
    except ImportError:
        return False
    return hasattr(ipywidgets.DOMWidget, '_repr_mimebundle_')


def _figure_png(fig):
    """
    (base64 PNG, display width in CSS pixels) of `fig` as it looks now, or
    None if it can't be drawn.
    """
    import base64
    import io

    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi='figure')
    except Exception:
        return None
    png = base64.b64encode(buf.getvalue()).decode('ascii')
    # Display width in CSS pixels (ipympl doubles the dpi on HiDPI screens)
    width = int(fig.get_figwidth() * fig.dpi
                / getattr(fig.canvas, 'device_pixel_ratio', 1))
    return png, width


class FigureSnapshot:
    """
    A static picture of a figure, saved in the notebook next to an
    interactive widget, that follows the figure as it changes.

    Viewers that cannot run widgets (e.g. GitHub's notebook preview) show
    this picture in place of the widget. Live Jupyter does not show it.

    How:
    - The widget is displayed as one output, with no picture.
    - The picture is a second output: a PNG plus a widget reference with
      an empty model id. JupyterLab and Notebook 7 hide a widget output
      with an empty model id, live and after reopening, so only viewers
      without widget support show the PNG.
    - The picture output has a display id. Whenever the followed figure
      is drawn (a kval control changed it, or the user zoomed/panned),
      the picture is replaced, after the drawing has been quiet for
      `delay_ms`. Only the picture output is updated; the widget output
      is never touched (updating it would make JupyterLab add a second
      view of the widget).

    Tools whose figure is replaced on every change (moored.plot,
    threshold_pick, tsplot_pick) call follow() with each new figure.

    Falls back to a plain display(widget), with no picture, outside a
    notebook and with ipywidgets 7 (no _repr_mimebundle_).
    """

    def __init__(self, delay_ms=500):
        self.fig = None
        self.delay_ms = delay_ms
        self._cid = None
        self._timer = None
        self._handle = None
        self._busy = False  # True while rendering the picture ourselves

    # -- following a figure ------------------------------------------------

    def follow(self, fig):
        """Make the picture follow `fig` (replacing any earlier figure)."""
        if fig is self.fig:
            return
        self._disconnect()
        self.fig = fig
        if fig is None:
            return
        try:
            self._cid = fig.canvas.mpl_connect('draw_event', self._on_draw)
        except Exception:
            self._cid = None
        self._timer = None
        self._schedule()

    def _disconnect(self):
        if self.fig is not None and self._cid is not None:
            try:
                self.fig.canvas.mpl_disconnect(self._cid)
            except Exception:
                pass
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:
                pass
        self._cid = None
        self._timer = None

    def _on_draw(self, event):
        if not self._busy:
            self._schedule()

    def _schedule(self):
        """Update the picture once the figure has been quiet for delay_ms."""
        if self._handle is None or self.fig is None:
            return  # Not displayed yet: display() takes the first picture
        if self._timer is None:
            try:
                self._timer = self.fig.canvas.new_timer(
                    interval=self.delay_ms)
                self._timer.single_shot = True
                self._timer.add_callback(self.update)
            except Exception:
                self._timer = None
                return
        try:
            self._timer.stop()  # Restart: wait for the drawing to settle
            self._timer.start()
        except Exception:
            pass

    # -- the picture output ------------------------------------------------

    def _picture_bundle(self):
        """Display data for the picture output (None if no picture)."""
        if self.fig is None:
            return None
        self._busy = True  # savefig draws, which fires draw_event
        try:
            result = _figure_png(self.fig)
        finally:
            self._busy = False
        if result is None:
            return None
        png, width = result
        return {
            # Empty model id: hidden by JupyterLab/Notebook 7, live or not
            _WIDGET_VIEW: {'version_major': 2, 'version_minor': 0,
                           'model_id': ''},
            'image/png': png,
            'text/html': (
                f"<img src='data:image/png;base64,{png}' width={width}/>"),
            'text/plain': '',
        }

    def update(self):
        """Replace the saved picture with the figure as it looks now."""
        if self._handle is None:
            return
        bundle = self._picture_bundle()
        if bundle is None:
            return
        try:
            self._handle.update(bundle, raw=True)
        except Exception:
            pass

    # -- displaying --------------------------------------------------------

    def display(self, widget):
        """Display `widget`, followed by the picture of the current figure."""
        from IPython.display import display

        bundle = _widget_bundle(widget) if is_notebook() else None
        picture = self._picture_bundle() if bundle is not None else None
        if picture is None:  # No figure, no picture, or ipywidgets 7
            display(widget)
            return
        # No text fallback for the widget itself: viewers without widgets
        # show the picture below instead of "VBox(children=...".
        bundle.pop('image/png', None)
        bundle.pop('text/html', None)
        bundle['text/plain'] = ''
        display(bundle, raw=True)
        self._handle = display(picture, raw=True, display_id=True)

    def display_figure(self, fig):
        """
        Display an interactive (ipympl) figure on its own, as
        show_figure() does, with a following picture instead of ipympl's
        own picture (which is taken once, when the figure is first shown).
        """
        from IPython.display import display

        # ipympl's _repr_mimebundle_ also registers the figure as
        # displayed; keep everything but its picture.
        # ipywidgets 7: unchanged behaviour (as for the other tools)
        bundle = (_widget_bundle(fig.canvas)
                  if is_notebook() and _ipywidgets_has_bundles() else None)
        if bundle is None:
            show_figure(fig)
            return
        self.follow(fig)
        bundle.pop('image/png', None)
        bundle.pop('text/html', None)
        bundle['text/plain'] = ''
        display(bundle, raw=True)
        if getattr(fig.canvas, 'manager', None) is not None:
            fig.canvas.draw()
        picture = self._picture_bundle()
        if picture is not None:
            self._handle = display(picture, raw=True, display_id=True)


def display_with_snapshot(widget, fig=None):
    """
    Display an interactive widget, with a static picture of `fig` saved
    next to it for viewers that cannot run widgets (e.g. GitHub's notebook
    preview). The picture follows `fig` as it is redrawn; see
    FigureSnapshot. Returns the FigureSnapshot (call .follow(new_fig) if
    the tool replaces its figure).

    If no picture can be made (no figure, not in a notebook, ipywidgets 7),
    this is just display(widget).
    """
    snapshot = FigureSnapshot()
    snapshot.follow(fig)
    snapshot.display(widget)
    return snapshot


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
