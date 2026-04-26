"""Sliding-window motion detector over CSI amplitude.

The signal: when the medium between transmitter and receiver is static, the
CSI amplitudes per subcarrier are nearly constant frame-to-frame (only thermal
noise + small AGC jitter). When something moves through the path, multipath
phases shift, which changes the per-subcarrier amplitudes. Summing the recent
per-subcarrier standard deviation gives a scalar motion score.

The detector compares that score to a threshold derived from a baseline that
the user captures while the room is still. Hysteresis prevents flicker around
the threshold.
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class DetectorConfig:
    window: int = 50           # samples per sliding window (~0.5s at 100Hz)
    enter_ratio: float = 3.0   # score / baseline to trigger motion
    exit_ratio: float = 1.5    # score / baseline to clear motion
    min_baseline: float = 1e-3  # floor to avoid divide-by-zero in silent rooms


class MotionDetector:
    def __init__(self, subcarrier_idx: np.ndarray, baseline: float,
                 config: DetectorConfig = DetectorConfig()):
        self.idx = subcarrier_idx
        self.baseline = max(float(baseline), config.min_baseline)
        self.cfg = config
        self._buf: collections.deque[np.ndarray] = collections.deque(maxlen=config.window)
        self._in_motion = False

    def update(self, amplitude: np.ndarray) -> tuple[float, bool]:
        """Push one CSI amplitude vector and return (score, in_motion)."""
        self._buf.append(amplitude[self.idx])
        if len(self._buf) < self._buf.maxlen:
            return 0.0, self._in_motion
        stack = np.stack(self._buf)
        score = float(np.mean(np.std(stack, axis=0)))
        ratio = score / self.baseline
        if not self._in_motion and ratio >= self.cfg.enter_ratio:
            self._in_motion = True
        elif self._in_motion and ratio <= self.cfg.exit_ratio:
            self._in_motion = False
        return score, self._in_motion


def compute_baseline(amplitudes: np.ndarray, window: int) -> float:
    """Mean per-subcarrier std over rolling windows of a still-room capture.

    `amplitudes` has shape (n_samples, n_subcarriers). We drop the head and
    tail of the capture to avoid edge effects from the user starting/stopping
    the recording."""
    if amplitudes.shape[0] < window * 2:
        raise ValueError(
            f"need at least {window * 2} samples for a stable baseline, got {amplitudes.shape[0]}"
        )
    trim = window
    body = amplitudes[trim:-trim]
    n_windows = body.shape[0] - window + 1
    scores = np.empty(n_windows, dtype=np.float64)
    for i in range(n_windows):
        scores[i] = np.mean(np.std(body[i : i + window], axis=0))
    return float(np.median(scores))
