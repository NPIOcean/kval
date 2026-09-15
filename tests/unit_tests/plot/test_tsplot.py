"""
Tests for kval.plot.tsplot.

Covers the helper functions (_get_sa_ct, _resolve_color_values,
_draw_ts_background, _install_background_autoredraw), the static
tsplot() function, and the interactive tsplot_pick widget.
"""
import warnings

import numpy as np
import pytest
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmocean

from kval.plot import tsplot
from kval.util import internals


# --- Fixtures ---------------------------------------------------------

@pytest.fixture
def mock_dataset() -> xr.Dataset:
    """1D mooring-like dataset: TEMP/PSAL/PRES along TIME, with
    LATITUDE/LONGITUDE for computing SA/CT via gsw."""
    np.random.seed(0)
    n = 200
    return xr.Dataset(
        {
            'TEMP': ('TIME', 5 + 3 * np.random.randn(n)),
            'PSAL': ('TIME', 34 + 0.5 * np.random.randn(n)),
            'PRES': ('TIME', np.linspace(0, 500, n)),
            'LATITUDE': ((), 78.0),
            'LONGITUDE': ((), 15.0),
        },
        coords={'TIME': np.arange(n)},
    )


@pytest.fixture
def mock_dataset_2d() -> xr.Dataset:
    """2D CTD-section-like dataset (TIME, DEPTH) with a STATION
    coordinate varying only along TIME -- for testing categorical
    color_by and dimension broadcasting/mismatch."""
    np.random.seed(1)
    return xr.Dataset(
        {
            'TEMP': (('TIME', 'DEPTH'), 5 + 3 * np.random.randn(10, 5)),
            'PSAL': (('TIME', 'DEPTH'), 34 + 0.5 * np.random.randn(10, 5)),
            'PRES': (('TIME', 'DEPTH'),
                     np.tile(np.linspace(0, 100, 5), (10, 1))),
            'STATION': ('TIME', [f'st{i}' for i in range(10)]),
            'LATITUDE': ((), 78.0),
            'LONGITUDE': ((), 15.0),
        },
        coords={'TIME': np.arange(10), 'DEPTH': np.arange(5)},
    )


@pytest.fixture(autouse=True)
def close_figures_after_test():
    """Prevent figures leaking between tests (and matplotlib's
    >20-open-figures warning from firing during this test run itself)."""
    yield
    plt.close('all')


@pytest.fixture(autouse=True)
def _noop_plt_show(monkeypatch):
    """tsplot() correctly calls plt.show() when run outside a notebook
    (which is what pytest is) -- but the Agg backend used for testing
    can't actually show anything, so matplotlib warns about it on every
    such call. No-op it here; tests that specifically want to verify
    plt.show() gets called (see TestDisplayLogic) re-patch it themselves
    within the test, which overrides this."""
    monkeypatch.setattr(plt, 'show', lambda: None)


def _get_contour_y_extent(ax):
    """Helper: find the (min, max) y-extent actually reached by density
    contour lines drawn on ax, or None if there are none."""
    for child in ax.get_children():
        if hasattr(child, 'allsegs'):
            ys = [seg[:, 1] for segs in child.allsegs
                 for seg in segs if len(seg) > 0]
            if ys:
                return min(y.min() for y in ys), max(y.max() for y in ys)
    return None


# --- _get_sa_ct ---------------------------------------------------------

class TestGetSaCt:

    def test_computes_from_gsw_when_absent(self, mock_dataset):
        SA, CT = tsplot._get_sa_ct(mock_dataset)
        assert SA.shape == mock_dataset['TEMP'].shape
        assert CT.shape == mock_dataset['TEMP'].shape

    def test_uses_existing_sa_ct_when_present(self, mock_dataset):
        ds = mock_dataset.drop_vars(['LATITUDE', 'LONGITUDE'])
        ds['SA'] = ds['PSAL']
        ds['CT'] = ds['TEMP']
        SA, CT = tsplot._get_sa_ct(ds)
        assert np.array_equal(SA.values, ds['SA'].values)
        assert np.array_equal(CT.values, ds['CT'].values)

    def test_raises_when_missing_lat_lon(self, mock_dataset):
        ds = mock_dataset.drop_vars(['LATITUDE', 'LONGITUDE'])
        with pytest.raises(ValueError, match='LATITUDE'):
            tsplot._get_sa_ct(ds)


# --- _resolve_color_values ----------------------------------------------

class TestResolveColorValues:

    def test_broadcasts_subset_dims(self, mock_dataset_2d):
        SA, _ = tsplot._get_sa_ct(mock_dataset_2d)
        values = tsplot._resolve_color_values(
            mock_dataset_2d, 'STATION', SA)
        assert values.shape == SA.shape

    def test_raises_on_dimension_mismatch(self, mock_dataset_2d):
        ds = mock_dataset_2d.copy()
        ds['UNRELATED'] = ('OTHER_DIM', np.arange(3))
        ds = ds.assign_coords(OTHER_DIM=np.arange(3))
        SA, _ = tsplot._get_sa_ct(mock_dataset_2d)
        with pytest.raises(ValueError, match='OTHER_DIM'):
            tsplot._resolve_color_values(ds, 'UNRELATED', SA)

    def test_raises_on_missing_variable(self, mock_dataset):
        SA, _ = tsplot._get_sa_ct(mock_dataset)
        with pytest.raises(ValueError, match='not found'):
            tsplot._resolve_color_values(mock_dataset, 'NOT_A_VAR', SA)


# --- tsplot(): basic behavior --------------------------------------------

class TestTsplotBasic:

    def test_basic_scatter(self, mock_dataset):
        fig, ax = tsplot.tsplot(mock_dataset)
        assert ax.get_xlabel().startswith('Absolute Salinity')
        assert ax.get_ylabel().startswith('Conservative Temperature')
        assert fig is ax.figure

    def test_passed_in_ax_is_reused(self, mock_dataset):
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = tsplot.tsplot(mock_dataset, ax=ax_in)
        assert ax_out is ax_in
        assert fig_out is fig_in

    def test_invalid_mode_raises(self, mock_dataset):
        with pytest.raises(ValueError, match='mode'):
            tsplot.tsplot(mock_dataset, mode='not_a_mode')


# --- color_by -------------------------------------------------------------

class TestColorBy:

    def test_numeric_color_by_has_colorbar(self, mock_dataset):
        fig, ax = tsplot.tsplot(mock_dataset, color_by='PRES')
        assert len(fig.axes) == 2  # main axis + colorbar axis

    @pytest.mark.parametrize('units,expected_label', [
        ('dbar', 'PRES [dbar]'),
        (None, 'PRES'),
    ])
    def test_numeric_color_by_label_includes_units_when_present(
            self, mock_dataset, units, expected_label):
        ds = mock_dataset.copy(deep=True)
        if units:
            ds['PRES'].attrs['units'] = units
        fig, ax = tsplot.tsplot(ds, color_by='PRES')
        cbar_ax = fig.axes[-1]
        assert cbar_ax.yaxis.label.get_text() == expected_label

    def test_categorical_color_by_has_discrete_legend(self, mock_dataset_2d):
        fig, ax = tsplot.tsplot(mock_dataset_2d, color_by='STATION')
        leg = ax.get_legend()
        assert leg is not None
        assert leg.get_title().get_text() == 'STATION'
        labels = [t.get_text() for t in leg.get_texts()]
        assert 'st0' in labels

    def test_color_by_dimension_mismatch_raises(self, mock_dataset_2d):
        ds = mock_dataset_2d.copy()
        ds['UNRELATED'] = ('OTHER_DIM', np.arange(3))
        ds = ds.assign_coords(OTHER_DIM=np.arange(3))
        with pytest.raises(ValueError):
            tsplot.tsplot(ds, color_by='UNRELATED')

    def test_default_cmap_is_cividis(self, mock_dataset):
        fig, ax = tsplot.tsplot(
            mock_dataset, color_by='PRES', density_contours=False)
        sc = ax.collections[0]
        assert sc.get_cmap().name == 'cividis'

    def test_cmap_overridable(self, mock_dataset):
        fig, ax = tsplot.tsplot(
            mock_dataset, color_by='PRES', cmap='plasma',
            density_contours=False)
        sc = ax.collections[0]
        assert sc.get_cmap().name == 'plasma'


# --- hist2d mode ------------------------------------------------------

class TestHist2d:

    def test_color_by_raises_in_hist2d(self, mock_dataset):
        with pytest.raises(ValueError, match='color_by'):
            tsplot.tsplot(mock_dataset, mode='hist2d', color_by='PRES')

    def test_default_colormap_is_cmocean_amp(self, mock_dataset):
        fig, ax = tsplot.tsplot(mock_dataset, mode='hist2d')
        pcm = ax.collections[-1]
        assert pcm.get_cmap().name == cmocean.cm.amp.name

    def test_empty_bins_masked(self, mock_dataset):
        fig, ax = tsplot.tsplot(mock_dataset, mode='hist2d')
        pcm = ax.collections[-1]
        assert np.ma.count_masked(pcm.get_array()) > 0

    def test_facecolor_default_lightgrey(self, mock_dataset):
        fig, ax = tsplot.tsplot(mock_dataset, mode='hist2d')
        pcm = ax.collections[-1]
        assert np.allclose(pcm.get_cmap().get_bad(),
                           mcolors.to_rgba('lightgrey'))

    def test_custom_facecolor(self, mock_dataset):
        fig, ax = tsplot.tsplot(
            mock_dataset, mode='hist2d', hist_facecolor='red')
        pcm = ax.collections[-1]
        assert np.allclose(pcm.get_cmap().get_bad(), mcolors.to_rgba('red'))

    def test_bins_parameter_controls_resolution(self, mock_dataset):
        fig, ax = tsplot.tsplot(mock_dataset, mode='hist2d', bins=10)
        shape10 = ax.collections[-1].get_array().shape
        fig, ax = tsplot.tsplot(mock_dataset, mode='hist2d', bins=50)
        shape50 = ax.collections[-1].get_array().shape
        assert shape50[0] > shape10[0]
        assert shape50[1] > shape10[1]

    def test_custom_hist_cmap_string(self, mock_dataset):
        fig, ax = tsplot.tsplot(
            mock_dataset, mode='hist2d', hist_cmap='viridis')
        pcm = ax.collections[-1]
        assert pcm.get_cmap().name == 'viridis'

    def test_extent_matches_axis(self, mock_dataset):
        """The histogram rectangle should exactly fill the visible axis
        area, not sit inside it as a separate, smaller box (regression
        test for the 'grey box artifact' bug)."""
        fig, ax = tsplot.tsplot(mock_dataset, mode='hist2d')
        pcm = ax.collections[-1]
        bbox = pcm.get_datalim(ax.transData)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        assert bbox.x0 == pytest.approx(xlim[0])
        assert bbox.x1 == pytest.approx(xlim[1])
        assert bbox.y0 == pytest.approx(ylim[0])
        assert bbox.y1 == pytest.approx(ylim[1])


# --- freezing line & density contours ------------------------------------

class TestFreezingLineAndContours:

    def test_freezing_line_off_by_default(self, mock_dataset):
        fig, ax = tsplot.tsplot(mock_dataset)
        assert not [l for l in ax.get_lines()
                   if l.get_label() == 'Freezing point']

    def test_freezing_line_style(self, mock_dataset):
        """Black and dashed -- distinct from both the default scatter
        color (matplotlib default is the same blue the freezing line
        used to be, before this was fixed) and the grey density
        contours."""
        fig, ax = tsplot.tsplot(mock_dataset, freezing_line=True)
        fl = [l for l in ax.get_lines() if l.get_label() == 'Freezing point']
        assert len(fl) == 1
        assert fl[0].get_color() == 'k'
        assert fl[0].get_linestyle() == '--'

    def test_contours_extend_to_cover_freezing_line(self, mock_dataset):
        """Regression test: density contours used to be truncated
        relative to the visible axis extent when the freezing line
        pulled the axis wider than the data-only range."""
        fig, ax = tsplot.tsplot(
            mock_dataset, freezing_line=True, density_contours=True)
        ylim = ax.get_ylim()
        extent = _get_contour_y_extent(ax)
        assert extent is not None
        assert extent[0] == pytest.approx(ylim[0], abs=1e-6)

    @pytest.mark.parametrize('new_ylim,check', [
        ((-5, 20), lambda ext: ext[0] <= -4.5 and ext[1] >= 19.5),
        ((7, 9), lambda ext: ext[0] >= 6.9 and ext[1] <= 9.1),
    ], ids=['wider', 'narrower'])
    def test_resize_redraws_to_match_new_view(
            self, mock_dataset, new_ylim, check):
        fig, ax = tsplot.tsplot(mock_dataset, freezing_line=True)
        ax.set_ylim(*new_ylim)
        extent = _get_contour_y_extent(ax)
        assert check(extent)

    def test_gade_line_not_implemented(self, mock_dataset):
        with pytest.raises(NotImplementedError):
            tsplot._draw_ts_background(
                plt.subplots()[1], (34, 35), (0, 5), gade_line=True)

    def test_no_background_means_no_autoredraw_callbacks(self, mock_dataset):
        """If neither contours nor freezing line are requested, no
        callback should be installed at all (nothing to redraw)."""
        fig, ax = tsplot.tsplot(
            mock_dataset, density_contours=False, freezing_line=False)
        assert len(ax.callbacks.callbacks.get('xlim_changed', {})) == 0


# --- legends ------------------------------------------------------------

class TestLegends:

    def test_categorical_and_freezing_line_combined_legend(
            self, mock_dataset_2d):
        fig, ax = tsplot.tsplot(
            mock_dataset_2d, color_by='STATION', freezing_line=True)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert 'Freezing point' in labels
        assert 'st0' in labels

    def test_numeric_colorbar_and_freezing_line_legend_coexist(
            self, mock_dataset):
        fig, ax = tsplot.tsplot(
            mock_dataset, color_by='PRES', freezing_line=True)
        assert len(fig.axes) == 2  # colorbar still present
        assert ax.get_legend() is not None  # freezing line legend too


# --- grid -----------------------------------------------------------------

class TestGrid:

    @pytest.mark.parametrize('grid_on', [False, True])
    def test_grid_toggle(self, mock_dataset, grid_on):
        fig, ax = tsplot.tsplot(mock_dataset, grid=grid_on)
        visible = any(l.get_visible() for l in ax.get_xgridlines())
        assert visible == grid_on


# --- styling ----------------------------------------------------------

class TestStyling:

    def test_all_text_elements_styled_consistently(self, mock_dataset):
        """One plot with every optional feature enabled (color_by,
        contours, freezing line) exercises every styled element at
        once: axis labels, tick labels, colorbar label, contour
        labels, and legend text should all match the same color/font
        constants."""
        fig, ax = tsplot.tsplot(
            mock_dataset, color_by='PRES', density_contours=True,
            freezing_line=True)

        assert ax.xaxis.label.get_color() == tsplot._TEXT_COLOR
        assert ax.xaxis.label.get_fontfamily() == [tsplot._FONT_FAMILY]
        assert ax.yaxis.label.get_color() == tsplot._TEXT_COLOR

        assert ax.get_xticklabels()[0].get_color() == tsplot._TEXT_COLOR

        cbar_ax = fig.axes[-1]
        assert cbar_ax.yaxis.label.get_color() == tsplot._TEXT_COLOR

        assert len(ax.texts) > 0
        assert ax.texts[0].get_color() == tsplot._TEXT_COLOR

        leg = ax.get_legend()
        assert leg.get_texts()[0].get_color() == tsplot._TEXT_COLOR

    def test_no_global_leakage(self, mock_dataset):
        """Styling must be scoped to tsplot's own figure, not mutate
        matplotlib's global rcParams."""
        tsplot.tsplot(mock_dataset)
        fig2, ax2 = plt.subplots()
        ax2.set_xlabel('unrelated plot')
        assert ax2.xaxis.label.get_color() == 'black'


# --- display logic ------------------------------------------------------

class TestDisplayLogic:
    """tsplot() shows the figure it creates itself (needed for the
    ipympl/%matplotlib widget backend, which doesn't reliably
    auto-display a figure just because it exists) -- but only the
    figure it created itself, and via the right mechanism for the
    context it's running in."""

    def test_notebook_uses_display(self, mock_dataset, monkeypatch):
        monkeypatch.setattr(internals, 'is_notebook', lambda: True)
        calls = []
        monkeypatch.setattr(tsplot, 'display', lambda obj: calls.append(obj))
        show_calls = []
        monkeypatch.setattr(plt, 'show', lambda: show_calls.append(True))
        tsplot.tsplot(mock_dataset)
        assert len(calls) == 1
        assert len(show_calls) == 0

    def test_script_uses_show(self, mock_dataset, monkeypatch):
        monkeypatch.setattr(internals, 'is_notebook', lambda: False)
        calls = []
        monkeypatch.setattr(tsplot, 'display', lambda obj: calls.append(obj))
        show_calls = []
        monkeypatch.setattr(plt, 'show', lambda: show_calls.append(True))
        tsplot.tsplot(mock_dataset)
        assert len(calls) == 0
        assert len(show_calls) == 1

    def test_passed_in_ax_does_not_trigger_display(
            self, mock_dataset, monkeypatch):
        """Composing tsplot into a larger figure shouldn't force a
        premature render before the caller has finished building it."""
        monkeypatch.setattr(internals, 'is_notebook', lambda: True)
        calls = []
        monkeypatch.setattr(tsplot, 'display', lambda obj: calls.append(obj))
        fig, ax = plt.subplots()
        tsplot.tsplot(mock_dataset, ax=ax)
        assert len(calls) == 0


# --- tsplot_pick widget -------------------------------------------------

@pytest.fixture
def picker(mock_dataset):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = tsplot.tsplot_pick(mock_dataset)
    yield p
    if p.fig is not None:
        plt.close(p.fig)


class TestTsplotPick:

    def test_constructs_without_error(self, picker):
        assert picker.fig is not None

    def test_marker_options(self, picker):
        assert picker.marker_dropdown.options == ('o', '.', '+')

    def test_freezing_checkbox_default_false(self, picker):
        assert picker.freezing_checkbox.value is False

    def test_mode_toggle_labels_capitalized(self, picker):
        assert picker.mode_toggle.options == (
            ('Scatter', 'scatter'), ('Hist2d', 'hist2d'))
        assert picker.mode_toggle.value == 'scatter'

    def test_bins_slider_visibility_toggles_with_mode(self, picker):
        assert picker.bins_slider.layout.display == 'none'
        picker.mode_toggle.value = 'hist2d'
        assert picker.bins_slider.layout.display == ''
        picker.mode_toggle.value = 'scatter'
        assert picker.bins_slider.layout.display == 'none'

    def test_scatter_only_controls_visibility_toggles_with_mode(
            self, picker):
        for control in (picker.marker_dropdown, picker.alpha_slider,
                        picker.color_dropdown):
            assert control.layout.display == ''
        picker.mode_toggle.value = 'hist2d'
        for control in (picker.marker_dropdown, picker.alpha_slider,
                        picker.color_dropdown):
            assert control.layout.display == 'none'

    def test_figure_does_not_accumulate(self, picker):
        for i in range(15):
            picker.mode_toggle.value = ['scatter', 'hist2d'][i % 2]
            picker.alpha_slider.value = 0.1 + (i % 9) * 0.1
        assert len(plt.get_fignums()) == 1

    def test_close_button_closes_figure(self, picker):
        fig_number = picker.fig.number
        picker.close_button.click()
        assert fig_number not in plt.get_fignums()