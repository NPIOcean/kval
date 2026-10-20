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


# display_with_snapshot: one output holding the widget and a PNG of the figure

def test_display_with_snapshot_attaches_png(monkeypatch):
    """Static viewers (e.g. GitHub) show the PNG; live Jupyter the widget."""
    import base64
    import ipywidgets as widgets
    if not hasattr(widgets.VBox, "_repr_mimebundle_"):
        pytest.skip("ipywidgets < 8: no picture is attached (plain display)")
    import matplotlib.pyplot as plt
    import IPython.display
    from kval.util import internals

    shown = []
    monkeypatch.setattr(internals, "is_notebook", lambda: True)
    monkeypatch.setattr(IPython.display, "display",
                        lambda obj, **kw: shown.append((obj, kw)))

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    box = widgets.VBox([widgets.Label("x")])
    internals.display_with_snapshot(box, fig)
    plt.close(fig)

    assert len(shown) == 1  # one output, not widget + separate picture
    bundle, kw = shown[0]
    assert kw.get("raw") is True
    assert bundle["application/vnd.jupyter.widget-view+json"]["model_id"] \
        == box.model_id
    assert base64.b64decode(bundle["image/png"])[:4] == b"\x89PNG"
    assert bundle["image/png"] in bundle["text/html"]


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
