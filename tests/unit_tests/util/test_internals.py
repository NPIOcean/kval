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
