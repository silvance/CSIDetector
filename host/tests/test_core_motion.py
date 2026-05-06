"""Smoke tests for csidetector.core.detector.MotionDetector."""

from __future__ import annotations

import numpy as np

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
