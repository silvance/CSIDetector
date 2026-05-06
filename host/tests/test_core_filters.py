"""Smoke tests for csidetector.core.detector filters."""

from __future__ import annotations

import numpy as np
import pytest

from csidetector.core import detector


def test_hampel_filter_replaces_outlier():
    # Realistic noise — when MAD is zero (constant data) the filter
    # intentionally leaves spikes alone to avoid degenerate thresholds.
    rng = np.random.default_rng(0)
    x = 1.0 + rng.normal(0, 0.02, size=40)
    x[20] = 100.0     # obvious spike
    out = detector.hampel_filter(x, k=3.0, window=7)
    assert abs(out[20] - 1.0) < 0.1, f"spike not suppressed: {out[20]}"
    # Non-spike values are preserved (within reasonable tolerance).
    assert np.allclose(out[:10], x[:10], atol=0.05)


def test_hampel_filter_window_must_be_odd():
    with pytest.raises(ValueError):
        detector.hampel_filter(np.zeros(10), window=4)


def test_hampel_filter_2d_per_column():
    rng = np.random.default_rng(1)
    x = 1.0 + rng.normal(0, 0.02, size=(40, 4))
    x[20, 1] = 50.0
    out = detector.hampel_filter(x, k=3.0, window=7)
    assert abs(out[20, 1] - 1.0) < 0.2
    # Other columns at the same row are essentially unchanged.
    assert np.allclose(out[20, [0, 2, 3]], x[20, [0, 2, 3]], atol=0.05)
