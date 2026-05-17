"""Smoke tests for csidetector.core.detector.MotionDetector."""

from __future__ import annotations

import math

import numpy as np
import pytest

from csidetector.core import detector


def test_compute_baseline_handles_minimum_input():
    rng = np.random.default_rng(0)
    # Need at least 3*window samples; use 200 with window=50.
    amps = 1.0 + rng.normal(0, 0.01, size=(200, 16))
    b = detector.compute_baseline(amps, window=50)
    assert 0.005 < b < 0.05, f"unrealistic baseline: {b}"


def test_motion_detector_transitions(synthetic_amplitudes):
    cfg = detector.DetectorConfig(window=50, enter_ratio=3.0, exit_ratio=1.5)
    still = synthetic_amplitudes(n=300, n_sub=64, noise=0.01)
    motion = synthetic_amplitudes(n=300, n_sub=64, noise=0.10, seed=1)
    baseline = detector.compute_baseline(still[:200], window=50)

    # Detector must see baseline-like data first to fill its window.
    det = detector.MotionDetector(np.arange(64), baseline, cfg)
    # Feed all-still: expect no motion.
    for amp in still:
        _, motion_state = det.update(amp)
    assert motion_state is False, "phantom motion on still data"

    # Now feed motion: detector should flip True.
    saw_motion = False
    for amp in motion:
        _, motion_state = det.update(amp)
        if motion_state:
            saw_motion = True
            break
    assert saw_motion, "missed obvious motion"

    # Feed still again: detector should flip back to False.
    saw_clear = False
    for amp in still:
        _, motion_state = det.update(amp)
        if motion_state is False:
            saw_clear = True
            break
    assert saw_clear, "stuck in MOTION after data went quiet"


# --------------------------------------------------------------------------
# Constructor validation.
# --------------------------------------------------------------------------

def test_motion_detector_rejects_nan_baseline():
    """The old code did max(nan, x) which returns nan, then ratio = nan,
    then neither enter/exit branch ever fires — silently no alerts.
    Reject explicitly so the operator gets a clear error."""
    with pytest.raises(ValueError, match="baseline"):
        detector.MotionDetector(np.arange(8), float("nan"))


def test_motion_detector_rejects_zero_baseline():
    with pytest.raises(ValueError, match="baseline"):
        detector.MotionDetector(np.arange(8), 0.0)


def test_motion_detector_rejects_negative_baseline():
    with pytest.raises(ValueError, match="baseline"):
        detector.MotionDetector(np.arange(8), -0.1)


def test_motion_detector_rejects_inf_baseline():
    with pytest.raises(ValueError, match="baseline"):
        detector.MotionDetector(np.arange(8), float("inf"))


def test_motion_detector_rejects_zero_window():
    cfg = detector.DetectorConfig(window=0)
    with pytest.raises(ValueError, match="window"):
        detector.MotionDetector(np.arange(8), 0.1, cfg)


# --------------------------------------------------------------------------
# Mid-stream shape mismatch (bandwidth / MCS change).
# --------------------------------------------------------------------------

def test_motion_detector_drops_too_short_samples_without_crashing(synthetic_amplitudes):
    """If a later sample has fewer subcarriers than the idx mask covers
    (e.g. a mid-stream bandwidth change), the old code raised IndexError
    and killed alert-mode detect. New behavior: silently drop the
    runt sample and keep going."""
    cfg = detector.DetectorConfig(window=50)
    still = synthetic_amplitudes(n=100, n_sub=64, noise=0.01)
    baseline = detector.compute_baseline(still[:75], window=25)
    det = detector.MotionDetector(np.arange(64), baseline, cfg)
    # Normal-width sample: accepted.
    score, _ = det.update(still[0])
    assert math.isfinite(score)
    # Runt sample (32 subcarriers, but idx covers 0..63): must not crash.
    runt = np.ones(32, dtype=np.float32)
    score, motion = det.update(runt)
    assert score == 0.0
    assert motion is False
    # Subsequent normal-width sample is still processed correctly.
    score, _ = det.update(still[1])
    assert math.isfinite(score)
