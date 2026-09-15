"""
KVAL.PLOT.TSPLOT

Temperature-Salinity (T-S) diagrams, shared between CTD and moored data.

Always plots on TEOS-10 axes (Absolute Salinity, Conservative
Temperature), computing them via gsw if not already present in the
dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import gsw
import cmocean
import ipywidgets as widgets
from IPython.display import display, clear_output

from kval.util import internals


# Applied explicitly to each text element as it's created (not via
# plt.rc_context/rcParams) -- rcParams like axes.labelcolor turn out to
# be baked in at *axes creation* time, not when set_xlabel() etc. is
# actually called, so relying on them is timing-fragile, especially
# since tsplot can be handed an already-existing ax. Explicit .set_color()
# / fontfamily= calls work regardless of when/how the axes was made, and
# never touch global matplotlib state.
_TEXT_COLOR = '#404040'
_FONT_FAMILY = 'Arial'  # matplotlib falls back gracefully if unavailable


def _get_sa_ct(ds: xr.Dataset,
               temp_var: str = 'TEMP', psal_var: str = 'PSAL',
               pres_var: str = 'PRES',
               lat_var: str = 'LATITUDE', lon_var: str = 'LONGITUDE',
               sa_var: str = 'SA', ct_var: str = 'CT'):
    """
    Get Absolute Salinity and Conservative Temperature for a dataset.

    Uses ds[sa_var]/ds[ct_var] directly if both are already present.
    Otherwise, computes them via gsw (not written back to ds) -- this
    requires lat_var/lon_var to be present.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    temp_var, psal_var, pres_var : str
        Names of the in-situ temperature, practical salinity, and
        pressure variables (used only if sa_var/ct_var aren't present).
    lat_var, lon_var : str
        Names of the latitude/longitude variables (used only if
        sa_var/ct_var aren't present).
    sa_var, ct_var : str
        Names to look for existing Absolute Salinity / Conservative
        Temperature variables under.

    Returns
    -------
    (xr.DataArray, xr.DataArray)
        SA, CT.

    Raises
    ------
    ValueError
        If sa_var/ct_var aren't present and lat_var/lon_var (needed to
        compute them) aren't either.
    """
    if sa_var in ds and ct_var in ds:
        return ds[sa_var], ds[ct_var]

    if lat_var not in ds or lon_var not in ds:
        raise ValueError(
            f"Can't get Absolute Salinity / Conservative Temperature: "
            f"'{sa_var}'/'{ct_var}' not in the dataset, and computing "
            f"them requires '{lat_var}'/'{lon_var}', which aren't "
            "present either.")

    SA = gsw.SA_from_SP(ds[psal_var], ds[pres_var], ds[lon_var], ds[lat_var])
    CT = gsw.CT_from_t(SA, ds[temp_var], ds[pres_var])
    return SA, CT


def _resolve_color_values(ds: xr.Dataset, color_by: str,
                          reference: xr.DataArray):
    """
    Get color_by values broadcast to match `reference`'s shape (typically
    the SA or CT DataArray).

    Uses xarray's own broadcasting rather than reimplementing it. If
    color_by has a dimension `reference` doesn't, that's a genuine
    mismatch (e.g. trying to color 2D moored TEMP/PSAL by a variable
    that only exists per-station on a CTD section) -- raises a clear
    error rather than guessing.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing color_by.
    color_by : str
        Name of the variable to color by.
    reference : xr.DataArray
        DataArray whose shape/dims color_by's values should be
        broadcast to match (typically SA or CT from _get_sa_ct).

    Returns
    -------
    np.ndarray
        color_by's values, broadcast to reference's shape.

    Raises
    ------
    ValueError
        If color_by isn't in ds, or has a dimension reference doesn't.
    """
    if color_by not in ds:
        raise ValueError(f"'{color_by}' not found in the dataset.")

    color_da = ds[color_by]
    extra_dims = set(color_da.dims) - set(reference.dims)
    if extra_dims:
        raise ValueError(
            f"Can't color by '{color_by}': it has dimension(s) "
            f"{sorted(extra_dims)} that the T-S data doesn't have. "
            "color_by must vary only along dimensions shared with "
            "temperature/salinity.")

    return color_da.broadcast_like(reference).values


def _draw_ts_background(ax, sa_range, ct_range, pres: float = 0,
                        density_contours: bool = True,
                        freezing_line: bool = True,
                        gade_line: bool = False, gade_params: dict = None,
                        n_grid: int = 100):
    """
    Draw density contours and/or a freezing line onto an existing axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw onto.
    sa_range, ct_range : (float, float)
        (min, max) Absolute Salinity / Conservative Temperature range to
        cover.
    pres : float, default=0
        Pressure [dbar] used for the freezing line (freezing point is
        pressure-dependent). Does not affect density contours, which are
        always potential density referenced to 0 dbar (sigma0), the
        standard T-S diagram convention.
    density_contours : bool, default=True
        Whether to draw sigma0 (potential density) contours.
    freezing_line : bool, default=True
        Whether to draw the seawater freezing point line.
    gade_line : bool, default=False
        Reserved for a future Gade-line implementation (meltwater mixing
        line). Not yet implemented -- raises NotImplementedError if True.
    gade_params : dict, optional
        Reserved for future use (source water point, ice type, etc.).
    n_grid : int, default=100
        Grid resolution used for computing density contours.

    Returns
    -------
    list
        The matplotlib artists created (contour set, contour labels,
        freezing line), so a caller can remove() them later to redraw
        at a different range (see _install_background_autoredraw).
    """
    if gade_line:
        raise NotImplementedError(
            "Gade line plotting is not yet implemented. The gade_line "
            "parameter is reserved for a future version.")

    artists = []

    if density_contours:
        sa_grid = np.linspace(sa_range[0], sa_range[1], n_grid)
        ct_grid = np.linspace(ct_range[0], ct_range[1], n_grid)
        SA_mesh, CT_mesh = np.meshgrid(sa_grid, ct_grid)
        sigma0 = gsw.sigma0(SA_mesh, CT_mesh)
        cs = ax.contour(SA_mesh, CT_mesh, sigma0, colors='grey',
                        linestyles='--', linewidths=0.7)
        labels = ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
        for label in labels:
            label.set_color(_TEXT_COLOR)
            label.set_fontfamily(_FONT_FAMILY)
        artists.append(cs)
        artists.extend(labels)

    if freezing_line:
        sa_line = np.linspace(sa_range[0], sa_range[1], n_grid)
        ct_freezing = gsw.CT_freezing(sa_line, pres, 0)
        line, = ax.plot(sa_line, ct_freezing, color='k',
                        linestyle='--', linewidth=1, label='Freezing point')
        artists.append(line)

    return artists


def _install_background_autoredraw(ax, pres: float = 0,
                                   density_contours: bool = True,
                                   freezing_line: bool = False):
    """
    Make the density contours / freezing line redraw automatically to
    match the axis's current view, whenever that view changes (zoom,
    pan, or any other resize) -- not just at initial plot creation.

    Without this, contours/freezing line are computed once over
    whatever range happened to be visible at creation time; zooming or
    panning to a different area afterward would just show blank space
    there instead of the background extending to cover it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to install the auto-redraw behavior on.
    pres : float, default=0
        Pressure [dbar] used for the freezing line, as in
        _draw_ts_background.
    density_contours, freezing_line : bool
        Which background elements to keep redrawn. If both are False,
        no callback is installed at all (nothing to redraw).
    """
    if not (density_contours or freezing_line):
        return

    state = {'artists': []}

    def _redraw(_ax):
        for artist in state['artists']:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass  # already gone somehow -- fine, nothing to clean up
        state['artists'] = _draw_ts_background(
            ax, ax.get_xlim(), ax.get_ylim(), pres=pres,
            density_contours=density_contours, freezing_line=freezing_line)

    ax.callbacks.connect('xlim_changed', _redraw)
    ax.callbacks.connect('ylim_changed', _redraw)
    _redraw(ax)  # initial draw, at whatever the axis limits are right now


def tsplot(ds: xr.Dataset,
          temp_var: str = 'TEMP', psal_var: str = 'PSAL',
          pres_var: str = 'PRES',
          lat_var: str = 'LATITUDE', lon_var: str = 'LONGITUDE',
          sa_var: str = 'SA', ct_var: str = 'CT',
          color_by: str | None = None, mode: str = 'scatter',
          density_contours: bool = True, freezing_line: bool = False,
          freezing_line_pres: float = 0,
          marker: str = 'o', alpha: float = 0.7, cmap: str = 'cividis',
          bins: int = 30, hist_cmap=None, hist_facecolor: str = 'lightgrey',
          grid: bool = False,
          ax=None, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a Temperature-Salinity (T-S) diagram on TEOS-10 axes.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    temp_var, psal_var, pres_var : str
        Names of in-situ temperature, practical salinity, and pressure
        variables. Only used if sa_var/ct_var aren't already present.
    lat_var, lon_var : str
        Names of latitude/longitude variables. Only used (and then
        required) if sa_var/ct_var aren't already present.
    sa_var, ct_var : str
        Names to look for existing Absolute Salinity / Conservative
        Temperature under, computed via gsw if not found.
    color_by : str or None, optional
        Name of a variable to color points by (scatter mode only -- see
        mode). Must vary only along dimensions shared with the T-S data.
    mode : {'scatter', 'hist2d'}, default='scatter'
        'scatter' plots individual points, optionally colored by
        color_by. 'hist2d' plots a 2D histogram of point density instead
        -- color_by is not supported in this mode (raises ValueError if
        given), since a 2D histogram bin already mixes many points
        together.
    density_contours : bool, default=True
        Whether to draw sigma0 (potential density) contours.
    freezing_line : bool, default=False
        Whether to draw the seawater freezing point line.
    freezing_line_pres : float, default=0
        Pressure [dbar] used for the freezing line calculation.
    marker : str, default='o'
        Marker style (scatter mode only).
    alpha : float, default=0.7
        Point transparency (scatter mode only).
    cmap : str, default='cividis'
        Colormap used when color_by is given (scatter mode).
    bins : int, default=30
        Number of bins per axis for mode='hist2d'.
    hist_cmap : str or matplotlib.colors.Colormap, optional
        Colormap for mode='hist2d'. Defaults to cmocean's 'amp' if not
        given.
    hist_facecolor : str, default='lightgrey'
        Color used for empty (zero-count) bins in mode='hist2d', so they
        read clearly as "no data" rather than blending into the low end
        of the colormap.
    grid : bool, default=False
        Whether to show gridlines.
    ax : matplotlib.axes.Axes, optional
        Axis to plot onto. If None, a new figure/axis is created.
    **kwargs
        Passed through to ax.scatter() or ax.hist2d().

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)

    Examples
    --------
    >>> fig, ax = tsplot(ds, color_by='PRES')
    >>> fig, ax = tsplot(ds, mode='hist2d')
    """
    if mode not in ('scatter', 'hist2d'):
        raise ValueError(f"mode must be 'scatter' or 'hist2d', got '{mode}'.")
    if mode == 'hist2d' and color_by is not None:
        raise ValueError(
            "color_by is not supported with mode='hist2d' -- a 2D "
            "histogram bin already mixes many points together, so "
            "there's no single value to color it by. Use mode='scatter' "
            "instead, or omit color_by.")

    SA, CT = _get_sa_ct(ds, temp_var=temp_var, psal_var=psal_var,
                        pres_var=pres_var, lat_var=lat_var, lon_var=lon_var,
                        sa_var=sa_var, ct_var=ct_var)

    sa_vals = np.asarray(SA.values if hasattr(SA, 'values') else SA).flatten()
    ct_vals = np.asarray(CT.values if hasattr(CT, 'values') else CT).flatten()
    finite = np.isfinite(sa_vals) & np.isfinite(ct_vals)
    sa_vals, ct_vals = sa_vals[finite], ct_vals[finite]

    with plt.ioff():  # suppress premature auto-display under ion()
                      # (widget backend) while the plot is still
                      # being built -- see tsplot's display logic
                      # right after this block for why
        created_own_fig = ax is None
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        sa_pad = 0.05 * (np.nanmax(sa_vals) - np.nanmin(sa_vals) or 1)
        ct_pad = 0.05 * (np.nanmax(ct_vals) - np.nanmin(ct_vals) or 1)
        sa_range = (np.nanmin(sa_vals) - sa_pad, np.nanmax(sa_vals) + sa_pad)
        ct_range = (np.nanmin(ct_vals) - ct_pad, np.nanmax(ct_vals) + ct_pad)

        if freezing_line:
            # Nicer initial view: make sure the starting range comfortably
            # includes the freezing line across the full SA span. Not load
            # bearing for correctness anymore (see _install_background_
            # autoredraw below), just avoids the freezing line being cut off
            # in the very first view before anyone's zoomed/panned at all.
            freezing_ct_at_range = gsw.CT_freezing(
                np.array(sa_range), freezing_line_pres, 0)
            ct_range = (min(ct_range[0], np.nanmin(freezing_ct_at_range)),
                       ct_range[1])

        ax.set_xlim(sa_range)
        ax.set_ylim(ct_range)

        # Density contours / freezing line redraw automatically to match
        # whatever the axis's current view is, including after zooming or
        # panning later -- not just at this initial range. See
        # _install_background_autoredraw's docstring.
        _install_background_autoredraw(
            ax, pres=freezing_line_pres, density_contours=density_contours,
            freezing_line=freezing_line)

        legend_handles, legend_labels = [], []
        legend_title = None

        if mode == 'scatter':
            if color_by is not None:
                color_vals = _resolve_color_values(ds, color_by, SA)
                color_vals = np.asarray(color_vals).flatten()[finite]
                if np.issubdtype(color_vals.dtype, np.number):
                    sc = ax.scatter(sa_vals, ct_vals, c=color_vals, cmap=cmap,
                                   marker=marker, alpha=alpha, **kwargs)
                    cbar = fig.colorbar(sc, ax=ax)
                    cbar_units = ds[color_by].attrs.get('units')
                    cbar_label = (f'{color_by} [{cbar_units}]' if cbar_units
                                 else color_by)
                    cbar.set_label(cbar_label, color=_TEXT_COLOR,
                                  fontfamily=_FONT_FAMILY)
                    cbar.ax.tick_params(colors=_TEXT_COLOR)
                else:
                    # Categorical (e.g. station names): factorize to integer
                    # codes for plotting, but show a discrete legend with the
                    # real category labels rather than a numeric colorbar.
                    categories, codes = np.unique(color_vals, return_inverse=True)
                    sc = ax.scatter(sa_vals, ct_vals, c=codes, cmap=cmap,
                                   marker=marker, alpha=alpha, **kwargs)
                    cat_handles, _ = sc.legend_elements(num=len(categories))
                    legend_handles += list(cat_handles)
                    legend_labels += list(categories)
                    legend_title = color_by
            else:
                ax.scatter(sa_vals, ct_vals, marker=marker, alpha=alpha, **kwargs)
        else:  # mode == 'hist2d'
            counts, sa_edges, ct_edges = np.histogram2d(
                sa_vals, ct_vals, bins=bins, range=[sa_range, ct_range])
            # histogram2d returns shape (n_sa_bins, n_ct_bins); pcolormesh
            # expects the first axis to match the second coordinate array,
            # so transpose to (n_ct_bins, n_sa_bins).
            counts = counts.T
            counts_masked = np.ma.masked_where(counts == 0, counts)

            resolved_cmap = cmocean.cm.amp if hist_cmap is None else hist_cmap
            if isinstance(resolved_cmap, str):
                resolved_cmap = plt.get_cmap(resolved_cmap)
            resolved_cmap = resolved_cmap.copy()
            resolved_cmap.set_bad(color=hist_facecolor)

            pcm = ax.pcolormesh(sa_edges, ct_edges, counts_masked,
                                cmap=resolved_cmap, **kwargs)
            cbar = fig.colorbar(pcm, ax=ax)
            cbar.set_label('Count', color=_TEXT_COLOR,
                          fontfamily=_FONT_FAMILY)
            cbar.ax.tick_params(colors=_TEXT_COLOR)

        ax.set_xlabel('Absolute Salinity [g kg$^{-1}$]',
                     color=_TEXT_COLOR, fontfamily=_FONT_FAMILY)
        ax.set_ylabel('Conservative Temperature [$\\degree$C]',
                     color=_TEXT_COLOR, fontfamily=_FONT_FAMILY)
        ax.grid(grid)

        ax.tick_params(axis='both', colors=_TEXT_COLOR,
                       labelfontfamily=_FONT_FAMILY)
        if freezing_line:
            freezing_handles, freezing_labels = ax.get_legend_handles_labels()
            legend_handles += freezing_handles
            legend_labels += freezing_labels

        if legend_handles:
            leg = ax.legend(legend_handles, legend_labels, fontsize=8,
                            loc='best', title=legend_title)
            for text in leg.get_texts():
                text.set_color(_TEXT_COLOR)
                text.set_fontfamily(_FONT_FAMILY)
            if leg.get_title() is not None:
                leg.get_title().set_color(_TEXT_COLOR)
                leg.get_title().set_fontfamily(_FONT_FAMILY)

    if created_own_fig:
        # The whole plot was just built with auto-display suppressed
        # (plt.ioff() above), specifically to avoid ipympl/widget's
        # known issue where interactive mode auto-displays a figure the
        # moment it's created -- i.e. blank, before any of this
        # function's drawing has happened -- which can then race with
        # the later draw calls and intermittently leave the blank
        # version showing. Now that it's fully built, show it once,
        # cleanly.
        if internals.is_notebook():
            display(fig)
        else:
            plt.show()

    return fig, ax


class tsplot_pick:
    """
    Interactive T-S diagram with live controls for color variable, plot
    mode, density contours, freezing line, marker, and alpha.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to plot from.
    **tsplot_kwargs
        Passed through to tsplot() on every redraw (e.g. temp_var,
        psal_var, if your dataset uses non-default variable names).

    Notes
    -----
    Gade line support is planned but not yet implemented -- no control
    for it is shown yet.
    """

    def __init__(self, ds: xr.Dataset, **tsplot_kwargs):
        internals.check_interactive()

        self.ds = ds
        self.tsplot_kwargs = tsplot_kwargs
        self.fig = None  # tracked so _redraw can close the previous one

        color_options = [None] + list(ds.data_vars)

        self.color_dropdown = widgets.Dropdown(
            options=color_options, value=None, description='Color by:',
            layout=widgets.Layout(width='200px'))
        self.mode_toggle = widgets.ToggleButtons(
            options=[('Scatter', 'scatter'), ('Hist2d', 'hist2d')],
            value='scatter', description='Mode:',
            style={'button_width': '70px'})
        self.density_checkbox = widgets.Checkbox(
            value=True, description='Density contours',
            indent=False, layout=widgets.Layout(width='160px'))
        self.freezing_checkbox = widgets.Checkbox(
            value=False, description='Freezing line',
            indent=False, layout=widgets.Layout(width='140px'))
        self.grid_checkbox = widgets.Checkbox(
            value=False, description='Grid',
            indent=False, layout=widgets.Layout(width='80px'))
        self.marker_dropdown = widgets.Dropdown(
            options=['o', '.', '+'], value='o', description='Marker:',
            layout=widgets.Layout(width='140px'))
        self.alpha_slider = widgets.FloatSlider(
            value=0.7, min=0.05, max=1.0, step=0.05, description='Alpha:',
            layout=widgets.Layout(width='260px'))
        self.bins_slider = widgets.IntSlider(
            value=30, min=5, max=100, step=5, description='Bins:',
            layout=widgets.Layout(width='260px'))
        self.close_button = widgets.Button(
            description='Close', button_style='danger',
            layout=widgets.Layout(width='70px'))
        self.close_button.on_click(self._on_close)
        self.output = widgets.Output()

        controls = [self.color_dropdown, self.mode_toggle,
                   self.density_checkbox, self.freezing_checkbox,
                   self.marker_dropdown, self.alpha_slider,
                   self.grid_checkbox, self.bins_slider]
        for control in controls:
            control.observe(self._redraw, names='value')

        # marker/alpha (scatter-only) and bins (hist2d-only) share one row
        # and swap places depending on mode, rather than each getting its
        # own row that leaves an empty gap when hidden.
        self.widget_box = widgets.VBox(
            [widgets.HBox([self.mode_toggle, self.color_dropdown,
                          self.close_button]),
             widgets.HBox([self.density_checkbox, self.freezing_checkbox,
                          self.grid_checkbox]),
             widgets.HBox([self.marker_dropdown, self.alpha_slider,
                          self.bins_slider]),
             self.output],
            layout=widgets.Layout(width='620px'))

        display(self.widget_box)
        self._redraw(None)

    def _on_close(self, _):
        """Close the figure and the widget controls, to avoid leaving a
        hanging figure open once you're done with a plot."""
        if self.fig is not None:
            plt.close(self.fig)
        self.widget_box.close()

    def _redraw(self, change):
        mode = self.mode_toggle.value
        # marker/alpha/color_by only do anything in scatter mode, bins
        # only in hist2d -- show/hide each set accordingly rather than
        # leaving controls visible that silently do nothing.
        scatter_display = '' if mode == 'scatter' else 'none'
        hist_display = '' if mode == 'hist2d' else 'none'
        self.marker_dropdown.layout.display = scatter_display
        self.alpha_slider.layout.display = scatter_display
        self.color_dropdown.layout.display = scatter_display
        self.bins_slider.layout.display = hist_display

        with self.output:
            clear_output(wait=True)
            if self.fig is not None:
                plt.close(self.fig)  # don't let figures accumulate on
                                      # every control change (matplotlib
                                      # keeps pyplot-created figures alive
                                      # until explicitly closed)
            color_by = self.color_dropdown.value
            if mode == 'hist2d':
                color_by = None  # not supported together; drop silently
                                  # in the widget rather than erroring on
                                  # a live control change
            self.fig, ax = tsplot(
                self.ds, color_by=color_by, mode=mode,
                density_contours=self.density_checkbox.value,
                freezing_line=self.freezing_checkbox.value,
                marker=self.marker_dropdown.value,
                alpha=self.alpha_slider.value,
                grid=self.grid_checkbox.value,
                bins=self.bins_slider.value,
                **self.tsplot_kwargs)
            # tsplot() now handles displaying the figure it creates
            # internally (see its own ioff()/display() logic) -- calling
            # plt.show() again here would duplicate it