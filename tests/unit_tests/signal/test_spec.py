"""
Tests for kval.signal.spec (windowed FFT and power spectral density).
"""

import numpy as np
import pytest

from kval.signal import spec


def _sine(n=4096, freq=0.1, noise=0.01, seed=0):
    t = np.arange(n) * 1.0  # 1 s sampling
    rng = np.random.default_rng(seed)
    return t, np.sin(2 * np.pi * freq * t) + noise * rng.standard_normal(n)


@pytest.mark.parametrize("window", ["bman", "han", "none"])
def test_psd_1d_series_gives_1d_spectrum_with_correct_peak(window):
    """A 1-D series used to give an (n_freq, 2 x n_freq) array, because the
    window was multiplied as an outer product."""
    t, y = _sine()
    s, f = spec.psd(y, t, n=4, window=window)
    assert s.shape == f.shape
    assert s.ndim == 1
    assert f[np.argmax(s)] == pytest.approx(0.1, abs=0.002)
    assert (f > 0).all()


@pytest.mark.parametrize("window", ["bman", "han", "none"])
def test_psd_integrates_to_variance_of_white_noise(window):
    """Parseval: the one-sided spectrum integrates to the variance."""
    rng = np.random.default_rng(1)
    y = 2.0 * rng.standard_normal(2**15)  # variance 4
    t = np.arange(y.size) * 0.5
    s, f = spec.psd(y, t, n=8, window=window)
    assert np.sum(s) * (f[1] - f[0]) == pytest.approx(4.0, rel=0.05)


def test_psd_columns_are_independent_series():
    t, y = _sine()
    s1, _ = spec.psd(y, t, n=4)
    s2, _ = spec.psd(np.stack([y, 2 * y], axis=1), t, n=4)
    assert s2.shape == (s1.size, 2)
    np.testing.assert_allclose(s2[:, 0], s1)
    np.testing.assert_allclose(s2[:, 1], 4 * s1)


def test_psd_axis_1_matches_axis_0():
    """wfft used to cut the segments along axis 0 whatever `axis` was."""
    t, y = _sine()
    yy = np.stack([y, 2 * y], axis=1)
    s0, f0 = spec.psd(yy, t, n=4, axis=0)
    s1, f1 = spec.psd(yy.T, t, n=4, axis=1)
    np.testing.assert_allclose(s1, s0.T)
    np.testing.assert_allclose(f1, f0)


def test_psd_complex_series_is_two_sided():
    """Complex input used to raise (UFuncTypeError / undefined rhslice)."""
    t = np.arange(4096) * 1.0
    s, f = spec.psd(np.exp(-2j * np.pi * 0.1 * t), t, n=4)
    assert s.shape == f.shape
    assert f.min() < 0 < f.max()
    assert f[np.argmax(s)] == pytest.approx(-0.1, abs=0.002)


def test_apply_window_keeps_the_shape():
    y = np.ones(100)
    assert spec.apply_window(y, window="han").shape == (100,)
    yy = np.ones((100, 3))
    assert spec.apply_window(yy, window="han", axis=0).shape == (100, 3)
    assert spec.apply_window(yy.T, window="han", axis=1).shape == (3, 100)


def test_wfft_warns_for_uneven_time_grid():
    t, y = _sine(n=1024)
    t = t.copy()
    t[500:] += 0.5
    with pytest.warns(UserWarning, match="uneven"):
        spec.wfft(y, t, n=2)


def test_wfft_regular_float_grid_does_not_warn():
    """The check used exact float comparison, so e.g. a 0.1 s grid warned."""
    import warnings
    t = np.arange(1024) * 0.1
    y = np.sin(t)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        spec.wfft(y, t, n=2)
