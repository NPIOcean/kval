import numpy as np
import pytest
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.quiver import Quiver

from kval.maps import maptools  # adjust import according to your package structure

def test_quiver_basic():
    # Create a figure and Cartopy GeoAxes with PlateCarree projection
    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
    
    # Generate a small lat/lon grid
    lon = np.linspace(-10, 10, 5)
    lat = np.linspace(40, 50, 5)
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    # Create simple uniform vector field (east=1, north=0.5)
    u = np.ones_like(lon2d)
    v = 0.5 * np.ones_like(lat2d)
    
    q = maptools.quiver_proj(ax, lon2d, lat2d, u, v)
    
    # Assert that a Quiver object is returned
    assert isinstance(q, Quiver), "Expected a matplotlib.quiver.Quiver instance"
    
    plt.close(fig)  # close figure to avoid display in tests

def test_quiver_empty_after_mask():
    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
    
    # Data with latitudes too close to poles (masked out)
    lon = np.array([0, 10])
    lat = np.array([89.99, -89.99])
    u = np.array([1, 1])
    v = np.array([1, 1])
    
    result = maptools.quiver_proj(ax, lon, lat, u, v, pole_buffer=0.02)
    
    # Should return None due to no valid vectors
    assert result is None, "Expected None when no valid vectors to plot"
    
    plt.close(fig)
    
def test_quiver_stride():
    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
    
    lon = np.linspace(-5, 5, 10)
    lat = np.linspace(45, 55, 10)
    lon2d, lat2d = np.meshgrid(lon, lat)
    u = np.ones_like(lon2d)
    v = np.ones_like(lat2d)
    
    q1 = maptools.quiver_proj(ax, lon2d, lat2d, u, v, stride=1)
    q2 = maptools.quiver_proj(ax, lon2d, lat2d, u, v, stride=2)
    
    # With stride=2, fewer vectors plotted
    assert len(q2.U) < len(q1.U)
    
    plt.close(fig)

def test_quiver_1d_vs_2d_input():
    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))

    # 1D arrays
    lon_1d = np.array([-5, 0, 5])
    lat_1d = np.array([45, 50, 55])
    u_1d = np.array([1, 1, 1])
    v_1d = np.array([0, 0.5, 1])

    q1 = maptools.quiver_proj(ax, lon_1d, lat_1d, u_1d, v_1d)

    # 2D arrays (meshgrid)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    u_2d = np.ones_like(lon_2d)
    v_2d = 0.5 * np.ones_like(lat_2d)

    q2 = maptools.quiver_proj(ax, lon_2d, lat_2d, u_2d, v_2d)

    assert isinstance(q1, Quiver)
    assert isinstance(q2, Quiver)

    plt.close(fig)


def test_quiver_scale_length_by_basis_false():
    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))

    lon = np.linspace(-5, 5, 5)
    lat = np.linspace(45, 50, 5)
    lon2d, lat2d = np.meshgrid(lon, lat)
    u = np.ones_like(lon2d)
    v = 0.5 * np.ones_like(lat2d)

    q = maptools.quiver_proj(ax, lon2d, lat2d, u, v, scale_length_by_basis=False)

    assert isinstance(q, Quiver)
    plt.close(fig)


def test_quiver_scale_auto_explicit():
    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))

    lon = np.linspace(-10, 10, 6)
    lat = np.linspace(40, 50, 6)
    lon2d, lat2d = np.meshgrid(lon, lat)
    u = np.ones_like(lon2d)
    v = 0.5 * np.ones_like(lat2d)

    q = maptools.quiver_proj(ax, lon2d, lat2d, u, v, scale='auto')

    assert isinstance(q, Quiver)
    plt.close(fig)


def test_quiver_with_nans():
    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))

    lon = np.array([0, 10, 20, 30])
    lat = np.array([40, 45, 50, 55])
    u = np.array([1, np.nan, 1, 1])
    v = np.array([0.5, 0.5, np.inf, 0.5])

    q = maptools.quiver_proj(ax, lon, lat, u, v)

    # Since there are some invalid vectors, quiver_proj should still return a Quiver object
    # but ignoring invalid points
    assert isinstance(q, Quiver)
    plt.close(fig)
