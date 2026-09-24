"""
Tests for kval.plot.tsplot.ts_axes -- the public "give me a decorated but
empty T-S axis" entry point.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from kval.plot import tsplot


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _make_ds(sa_mean=34.9, ct_mean=1.5, n=200, seed=0):
    rng = np.random.default_rng(seed)
    return xr.Dataset(
        {"SA": ("TIME", rng.normal(sa_mean, 0.05, n)),
         "CT": ("TIME", rng.normal(ct_mean, 0.5, n))}
    )


# ---------------------------------------------------------------------
# Axis range
# ---------------------------------------------------------------------

def test_explicit_ranges_are_used_as_given():
    _, ax = tsplot.ts_axes(sa_range=(34.0, 35.2), ct_range=(-2.0, 4.0))
    assert ax.get_xlim() == (34.0, 35.2)
    assert ax.get_ylim() == (-2.0, 4.0)


def test_range_derived_from_a_single_dataset_covers_the_data():
    ds = _make_ds()
    _, ax = tsplot.ts_axes(datasets=ds)

    assert ax.get_xlim()[0] < float(ds.SA.min())
    assert ax.get_xlim()[1] > float(ds.SA.max())
    assert ax.get_ylim()[0] < float(ds.CT.min())
    assert ax.get_ylim()[1] > float(ds.CT.max())


def test_range_derived_from_several_datasets_covers_all_of_them():
    warm, cold = _make_ds(34.9, 1.5, seed=1), _make_ds(34.4, -1.2, seed=2)
    _, ax = tsplot.ts_axes(datasets=[warm, cold])

    sa_lo, sa_hi = ax.get_xlim()
    assert sa_lo < float(cold.SA.min())
    assert sa_hi > float(warm.SA.max())


def test_missing_range_and_datasets_raises_a_helpful_error():
    with pytest.raises(ValueError, match="pass sa_range and ct_range"):
        tsplot.ts_axes()


def test_all_nan_data_raises_rather_than_producing_a_nan_axis():
    ds = xr.Dataset({"SA": ("TIME", [np.nan] * 5),
                     "CT": ("TIME", [np.nan] * 5)})
    with pytest.raises(ValueError, match="no finite"):
        tsplot.ts_axes(datasets=ds)


def test_nans_are_ignored_when_deriving_the_range():
    ds = xr.Dataset({"SA": ("TIME", [34.5, np.nan, 35.0]),
                     "CT": ("TIME", [0.0, np.nan, 2.0])})
    _, ax = tsplot.ts_axes(datasets=ds)
    assert np.all(np.isfinite(ax.get_xlim()))
    assert np.all(np.isfinite(ax.get_ylim()))


def test_data_with_no_spread_still_gives_a_usable_axis():
    """A single point must not produce a zero-width -- or absurdly wide -- axis."""
    ds = xr.Dataset({"SA": ("TIME", [34.9]), "CT": ("TIME", [1.5])})
    _, ax = tsplot.ts_axes(datasets=ds)

    sa_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    assert 0 < sa_span < 1.0        # proportional padding would give ~3.5


# ---------------------------------------------------------------------
# The axis itself
# ---------------------------------------------------------------------

def test_axes_are_labelled_for_teos10():
    _, ax = tsplot.ts_axes(sa_range=(34, 35), ct_range=(-2, 4))
    assert "Absolute Salinity" in ax.get_xlabel()
    assert "Conservative Temperature" in ax.get_ylabel()


def test_an_existing_axis_is_decorated_in_place():
    fig_in, ax_in = plt.subplots()
    fig_out, ax_out = tsplot.ts_axes(sa_range=(34, 35), ct_range=(-2, 4),
                                     ax=ax_in)
    assert ax_out is ax_in
    assert fig_out is fig_in


def test_background_elements_are_drawn():
    _, ax = tsplot.ts_axes(sa_range=(34, 35), ct_range=(-2, 4),
                           density_contours=True, freezing_line=True)
    assert len(ax.collections) > 0     # contours
    assert len(ax.lines) > 0           # freezing line


def test_background_can_be_switched_off_entirely():
    _, ax = tsplot.ts_axes(sa_range=(34, 35), ct_range=(-2, 4),
                           density_contours=False, freezing_line=False)
    assert len(ax.collections) == 0
    assert len(ax.lines) == 0


# ---------------------------------------------------------------------
# The recursion trap
# ---------------------------------------------------------------------

def test_plotting_onto_the_axis_does_not_recurse():
    """
    The background redraws on limit changes, and contour() autoscales --
    so with autoscaling left on, adding data recurses until Python gives
    up. ts_axes fixes the limits to prevent that. This test fails with
    RecursionError if that ever regresses.
    """
    ds = _make_ds()
    fig, ax = tsplot.ts_axes(datasets=ds)

    ax.scatter(ds.SA, ds.CT, s=1, alpha=0.03, label="instrument A")
    ax.scatter(ds.SA + 0.1, ds.CT - 1, s=1, alpha=0.03, label="instrument B")
    ax.legend()
    fig.canvas.draw()


def test_zooming_redraws_the_background_without_recursing():
    fig, ax = tsplot.ts_axes(sa_range=(34.0, 35.2), ct_range=(-2, 4))
    n_lines_before = len(ax.lines)

    ax.set_xlim(34.6, 35.0)
    ax.set_ylim(0, 2)
    fig.canvas.draw()

    assert ax.get_xlim() == (34.6, 35.0)
    assert len(ax.lines) == n_lines_before   # redrawn, not accumulated