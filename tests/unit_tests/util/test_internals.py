'''
Note: 

Turns out to be hard to do this testing - have to mock different backends/python environments.

A bit above my pay grade - skipping for now.
'''

import logging
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from kval.util import internals


@pytest.fixture(autouse=True)
def _clear_font_cache():
    internals.plot_font_family.cache_clear()
    yield
    internals.plot_font_family.cache_clear()


def test_plot_font_family_missing_font_falls_back(monkeypatch):
    def _raise(*args, **kwargs):
        raise ValueError('not found')
    monkeypatch.setattr(font_manager, 'findfont', _raise)
    assert internals.plot_font_family('Arial') == 'sans-serif'


def test_plot_font_family_available_font_kept(monkeypatch):
    monkeypatch.setattr(font_manager, 'findfont',
                        lambda *args, **kwargs: '/fake/arial.ttf')
    assert internals.plot_font_family('Arial') == 'Arial'


def test_plot_font_family_no_findfont_messages(caplog):
    # Whatever the resolved family is on this system, drawing with it
    # should not trigger matplotlib's "findfont: ... not found" log.
    fam = internals.plot_font_family('SurelyNotAnInstalledFont123')
    assert fam == 'sans-serif'
    with caplog.at_level(logging.WARNING, logger='matplotlib.font_manager'):
        fig, ax = plt.subplots()
        ax.set_title('x', fontfamily=fam)
        fig.canvas.draw()
        plt.close(fig)
    assert not [r for r in caplog.records if 'findfont' in r.getMessage()]


# is_notebook

class _KernelShell:
    kernel = object()


class _TerminalShell:
    pass


def test_is_notebook_without_get_ipython_builtin(monkeypatch):
    """In widget callbacks (button clicks etc.), get_ipython is not a
    builtin; it used to be looked up as one, so is_notebook() returned False
    in JupyterLab and interactive functions warned that they were not in a
    notebook."""
    import builtins
    import IPython
    monkeypatch.delattr(builtins, 'get_ipython', raising=False)
    monkeypatch.setattr(IPython, 'get_ipython', lambda: _KernelShell())
    assert internals.is_notebook()


def test_is_notebook_false_for_terminal_or_no_shell(monkeypatch):
    import IPython
    monkeypatch.setattr(IPython, 'get_ipython', lambda: _TerminalShell())
    assert not internals.is_notebook()
    monkeypatch.setattr(IPython, 'get_ipython', lambda: None)
    assert not internals.is_notebook()


# FigureSnapshot / display_with_snapshot: the widget, then a hidden picture
# output (for e.g. GitHub) that follows the figure

WIDGET_VIEW = "application/vnd.jupyter.widget-view+json"


class _FakeHandle:
    def __init__(self):
        self.updates = []

    def update(self, obj, **kw):
        self.updates.append((obj, kw))


class _FakeTimer:
    def __init__(self):
        self.callbacks = []
        self.starts = 0
        self.stops = 0
        self.single_shot = False

    def add_callback(self, f):
        self.callbacks.append(f)

    def start(self):
        self.starts += 1

    def stop(self):
        self.stops += 1

    def fire(self):
        for f in self.callbacks:
            f()


def _capture_display(monkeypatch, internals):
    """Record display() calls; display_id=True returns a fake handle."""
    import IPython.display
    shown = []

    def fake_display(obj, **kw):
        shown.append((obj, kw))
        if kw.get("display_id"):
            h = _FakeHandle()
            shown[-1] = (obj, kw, h)
            return h
    monkeypatch.setattr(internals, "is_notebook", lambda: True)
    monkeypatch.setattr(IPython.display, "display", fake_display)
    return shown


def _needs_ipywidgets8():
    import ipywidgets as widgets
    if not hasattr(widgets.VBox, "_repr_mimebundle_"):
        pytest.skip("ipywidgets < 8: no picture is attached (plain display)")


def test_display_with_snapshot_widget_then_hidden_picture(monkeypatch):
    """Live Jupyter shows the widget; static viewers (e.g. GitHub) the PNG.
    The picture is a separate output with an empty widget model id, which
    JupyterLab hides."""
    import base64
    _needs_ipywidgets8()
    import ipywidgets as widgets
    from kval.util import internals

    shown = _capture_display(monkeypatch, internals)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    box = widgets.VBox([widgets.Label("x")])
    snap = internals.display_with_snapshot(box, fig)
    plt.close(fig)

    assert isinstance(snap, internals.FigureSnapshot)
    assert len(shown) == 2
    widget_bundle, kw = shown[0]
    assert kw.get("raw") is True
    assert widget_bundle[WIDGET_VIEW]["model_id"] == box.model_id
    assert "image/png" not in widget_bundle
    assert widget_bundle["text/plain"] == ""  # no "VBox(children=..." text

    picture, kw, handle = shown[1]
    assert kw.get("raw") is True and kw.get("display_id") is True
    assert picture[WIDGET_VIEW]["model_id"] == ""  # hidden in JupyterLab
    assert base64.b64decode(picture["image/png"])[:4] == b"\x89PNG"
    assert picture["image/png"] in picture["text/html"]
    assert picture["text/plain"] == ""


def test_snapshot_follows_redraws(monkeypatch):
    """A draw of the figure (e.g. zoom/pan, or a kval control) schedules a
    picture update; when the timer fires, the saved picture is replaced
    with the figure as it looks now."""
    _needs_ipywidgets8()
    import ipywidgets as widgets
    from kval.util import internals

    shown = _capture_display(monkeypatch, internals)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    timer = _FakeTimer()
    monkeypatch.setattr(fig.canvas, "new_timer", lambda **kw: timer)
    snap = internals.display_with_snapshot(widgets.VBox([]), fig)
    first_png = shown[1][0]["image/png"]
    handle = shown[1][2]

    ax.set_xlim(0.2, 0.4)  # "zoom"
    fig.canvas.draw()      # fires draw_event
    assert timer.starts == 1 and timer.single_shot
    assert handle.updates == []  # waits for the drawing to settle
    fig.canvas.draw()
    assert timer.starts == 2 and timer.stops >= 1  # restarted

    timer.fire()
    assert len(handle.updates) == 1
    new_picture, kw = handle.updates[0]
    assert kw.get("raw") is True
    assert new_picture[WIDGET_VIEW]["model_id"] == ""
    assert new_picture["image/png"] != first_png
    # Rendering the picture draws the figure: that must not schedule again
    assert timer.starts == 2
    plt.close(fig)


def test_snapshot_follow_switches_figure(monkeypatch):
    """Tools that replace their figure (moored.plot, threshold_pick,
    tsplot_pick) call follow(new_fig): the picture then follows the new
    figure, and draws of the old one are ignored."""
    _needs_ipywidgets8()
    import ipywidgets as widgets
    from kval.util import internals

    shown = _capture_display(monkeypatch, internals)
    old, ax0 = plt.subplots()
    ax0.plot([0, 1])
    old_timer = _FakeTimer()
    monkeypatch.setattr(old.canvas, "new_timer", lambda **kw: old_timer)
    snap = internals.FigureSnapshot()
    snap.follow(old)
    snap.display(widgets.VBox([]))
    handle = shown[1][2]

    new, ax1 = plt.subplots()
    ax1.plot([1, 0], color="r")
    new_timer = _FakeTimer()
    monkeypatch.setattr(new.canvas, "new_timer", lambda **kw: new_timer)
    snap.follow(new)
    assert snap.fig is new
    assert new_timer.starts == 1  # new figure: update scheduled

    old.canvas.draw()
    assert old_timer.starts == 0  # old figure no longer followed

    new_timer.fire()
    assert len(handle.updates) == 1
    plt.close(old)
    plt.close(new)


def test_snapshot_not_displayed_does_nothing(monkeypatch):
    """Before display() there is no picture output to update."""
    from kval.util import internals
    shown = _capture_display(monkeypatch, internals)
    fig, ax = plt.subplots()
    timer = _FakeTimer()
    monkeypatch.setattr(fig.canvas, "new_timer", lambda **kw: timer)
    snap = internals.FigureSnapshot()
    snap.follow(fig)
    fig.canvas.draw()
    snap.update()
    assert timer.starts == 0 and shown == []
    plt.close(fig)


def test_display_figure_plain_canvas_falls_back_to_show_figure(monkeypatch):
    """Non-widget canvases (no _repr_mimebundle_, e.g. Agg) and ipywidgets
    7 are shown as before, through show_figure()."""
    from kval.util import internals
    called = []
    monkeypatch.setattr(internals, "is_notebook", lambda: True)
    monkeypatch.setattr(internals, "show_figure", lambda f: called.append(f))
    fig, ax = plt.subplots()
    internals.FigureSnapshot().display_figure(fig)
    assert called == [fig]
    plt.close(fig)


def test_display_figure_widget_canvas_gets_following_picture(monkeypatch):
    """A widget canvas (as ipympl's) is displayed without its own picture,
    followed by the hidden picture output."""
    _needs_ipywidgets8()
    from kval.util import internals

    shown = _capture_display(monkeypatch, internals)
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    bundle = {WIDGET_VIEW: {"version_major": 2, "version_minor": 0,
                            "model_id": "abc"},
              "image/png": "old", "text/html": "<img old>",
              "text/plain": "Canvas(...)"}
    monkeypatch.setattr(fig.canvas, "_repr_mimebundle_",
                        lambda **kw: dict(bundle), raising=False)
    monkeypatch.setattr(fig.canvas, "new_timer", lambda **kw: _FakeTimer())
    snap = internals.FigureSnapshot()
    snap.display_figure(fig)
    plt.close(fig)

    assert len(shown) == 2
    canvas_bundle = shown[0][0]
    assert canvas_bundle[WIDGET_VIEW]["model_id"] == "abc"
    assert "image/png" not in canvas_bundle
    assert "text/html" not in canvas_bundle
    picture = shown[1][0]
    assert picture[WIDGET_VIEW]["model_id"] == ""
    assert picture["image/png"] != "old"
    assert snap.fig is fig


def test_display_with_snapshot_without_figure_is_plain_display(monkeypatch):
    import ipywidgets as widgets
    import IPython.display
    from kval.util import internals

    shown = []
    monkeypatch.setattr(internals, "is_notebook", lambda: True)
    monkeypatch.setattr(IPython.display, "display",
                        lambda obj, **kw: shown.append(obj))
    box = widgets.VBox([])
    internals.display_with_snapshot(box, None)
    assert shown == [box]


def test_display_with_text_attaches_text(monkeypatch):
    """Static viewers show the text; live Jupyter the widget."""
    import ipywidgets as widgets
    if not hasattr(widgets.VBox, "_repr_mimebundle_"):
        pytest.skip("ipywidgets < 8: no text is attached (plain display)")
    import IPython.display
    from kval.util import internals

    shown = []
    monkeypatch.setattr(internals, "is_notebook", lambda: True)
    monkeypatch.setattr(IPython.display, "display",
                        lambda obj, **kw: shown.append((obj, kw)))
    box = widgets.VBox([])
    internals.display_with_text(box, "Report <ok> & done\n")

    assert len(shown) == 1
    bundle, kw = shown[0]
    assert kw.get("raw") is True
    assert bundle["application/vnd.jupyter.widget-view+json"]["model_id"] \
        == box.model_id
    assert bundle["text/plain"] == "Report <ok> & done\n"
    assert bundle["text/html"] == "<pre>Report &lt;ok&gt; &amp; done\n</pre>"


def test_display_helpers_fall_back_without_repr_mimebundle(monkeypatch):
    """ipywidgets 7 widgets have no _repr_mimebundle_: plain display, no
    crash."""
    import matplotlib.pyplot as plt
    import IPython.display
    from kval.util import internals

    class OldWidget:  # stands in for an ipywidgets 7 widget
        pass

    shown = []
    monkeypatch.setattr(internals, "is_notebook", lambda: True)
    monkeypatch.setattr(IPython.display, "display",
                        lambda obj, **kw: shown.append(obj))
    fig, ax = plt.subplots()
    w = OldWidget()
    internals.display_with_snapshot(w, fig)
    internals.display_with_text(w, "report")
    plt.close(fig)
    assert shown == [w, w]


def test_snapshot_picture_failure_is_plain_display(monkeypatch):
    """If the picture can't be drawn, the widget is shown as before (plain
    display, with its normal text fallback)."""
    _needs_ipywidgets8()
    import ipywidgets as widgets
    from kval.util import internals

    shown = _capture_display(monkeypatch, internals)
    monkeypatch.setattr(internals, "_figure_png", lambda fig: None)
    fig, ax = plt.subplots()
    box = widgets.VBox([])
    snap = internals.display_with_snapshot(box, fig)
    plt.close(fig)
    assert [s[0] for s in shown] == [box]
    snap.update()  # nothing to update; must not fail
