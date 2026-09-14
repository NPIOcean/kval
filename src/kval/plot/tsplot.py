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
import ipywidgets as widgets
from IPython.display import display, clear_output

from kval.util import internals


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
    """
    if gade_line:
        raise NotImplementedError(
            "Gade line plotting is not yet implemented. The gade_line "
            "parameter is reserved for a future version.")

    if density_contours:
        sa_grid = np.linspace(sa_range[0], sa_range[1], n_grid)
        ct_grid = np.linspace(ct_range[0], ct_range[1], n_grid)
        SA_mesh, CT_mesh = np.meshgrid(sa_grid, ct_grid)
        sigma0 = gsw.sigma0(SA_mesh, CT_mesh)
        cs = ax.contour(SA_mesh, CT_mesh, sigma0, colors='grey',
                        linestyles='--', linewidths=0.7)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

    if freezing_line:
        sa_line = np.linspace(sa_range[0], sa_range[1], n_grid)
        ct_freezing = gsw.CT_freezing(sa_line, pres, 0)
        ax.plot(sa_line, ct_freezing, color='tab:blue', linestyle='-',
               linewidth=1, label='Freezing point')


def tsplot(ds: xr.Dataset,
          temp_var: str = 'TEMP', psal_var: str = 'PSAL',
          pres_var: str = 'PRES',
          lat_var: str = 'LATITUDE', lon_var: str = 'LONGITUDE',
          sa_var: str = 'SA', ct_var: str = 'CT',
          color_by: str | None = None, mode: str = 'scatter',
          density_contours: bool = True, freezing_line: bool = True,
          freezing_line_pres: float = 0,
          marker: str = 'o', alpha: float = 0.7, cmap: str = 'viridis',
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
    freezing_line : bool, default=True
        Whether to draw the seawater freezing point line.
    freezing_line_pres : float, default=0
        Pressure [dbar] used for the freezing line calculation.
    marker : str, default='o'
        Marker style (scatter mode only).
    alpha : float, default=0.7
        Point transparency (scatter mode only).
    cmap : str, default='viridis'
        Colormap used when color_by is given.
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

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    sa_pad = 0.05 * (np.nanmax(sa_vals) - np.nanmin(sa_vals) or 1)
    ct_pad = 0.05 * (np.nanmax(ct_vals) - np.nanmin(ct_vals) or 1)
    sa_range = (np.nanmin(sa_vals) - sa_pad, np.nanmax(sa_vals) + sa_pad)
    ct_range = (np.nanmin(ct_vals) - ct_pad, np.nanmax(ct_vals) + ct_pad)

    _draw_ts_background(ax, sa_range, ct_range, pres=freezing_line_pres,
                        density_contours=density_contours,
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
                cbar.set_label(color_by)
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
        ax.hist2d(sa_vals, ct_vals, **kwargs)

    ax.set_xlabel('Absolute Salinity [g kg$^{-1}$]')
    ax.set_ylabel('Conservative Temperature [$\\degree$C]')

    if freezing_line:
        freezing_handles, freezing_labels = ax.get_legend_handles_labels()
        legend_handles += freezing_handles
        legend_labels += freezing_labels

    if legend_handles:
        ax.legend(legend_handles, legend_labels, fontsize=8, loc='best',
                 title=legend_title)

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

        color_options = [None] + list(ds.data_vars)

        self.color_dropdown = widgets.Dropdown(
            options=color_options, value=None, description='Color by:')
        self.mode_toggle = widgets.ToggleButtons(
            options=['scatter', 'hist2d'], value='scatter',
            description='Mode:')
        self.density_checkbox = widgets.Checkbox(
            value=True, description='Density contours')
        self.freezing_checkbox = widgets.Checkbox(
            value=True, description='Freezing line')
        self.marker_dropdown = widgets.Dropdown(
            options=['o', '.', 'x', '+', '^', 's'], value='o',
            description='Marker:')
        self.alpha_slider = widgets.FloatSlider(
            value=0.7, min=0.05, max=1.0, step=0.05, description='Alpha:')
        self.output = widgets.Output()

        controls = [self.color_dropdown, self.mode_toggle,
                   self.density_checkbox, self.freezing_checkbox,
                   self.marker_dropdown, self.alpha_slider]
        for control in controls:
            control.observe(self._redraw, names='value')

        self.widget_box = widgets.VBox(
            [widgets.HBox([self.mode_toggle, self.color_dropdown]),
             widgets.HBox([self.density_checkbox, self.freezing_checkbox]),
             widgets.HBox([self.marker_dropdown, self.alpha_slider]),
             self.output])

        display(self.widget_box)
        self._redraw(None)

    def _redraw(self, change):
        with self.output:
            clear_output(wait=True)
            color_by = self.color_dropdown.value
            mode = self.mode_toggle.value
            if mode == 'hist2d':
                color_by = None  # not supported together; drop silently
                                  # in the widget rather than erroring on
                                  # a live control change
            tsplot(
                self.ds, color_by=color_by, mode=mode,
                density_contours=self.density_checkbox.value,
                freezing_line=self.freezing_checkbox.value,
                marker=self.marker_dropdown.value,
                alpha=self.alpha_slider.value,
                **self.tsplot_kwargs)
            plt.show()