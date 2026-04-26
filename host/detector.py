"""Sliding-window motion detector over CSI amplitude.

When the medium between transmitter and receiver is static, per-subcarrier
CSI amplitudes are nearly constant frame-to-frame (only thermal noise +
AGC jitter). Motion through the path shifts multipath phases, which
changes those amplitudes. Summing the recent per-subcarrier standard
deviation gives a scalar motion score.

Two refinements borrowed from francescopace/espectre's MVS algorithm:

* AGC settle wait — the radio's auto-gain takes ~10 s to lock after boot;
  baselines computed before then are dominated by gain transients.
* Hampel outlier filter — replaces points more than k * MAD from the
  rolling median with the median, dropping single-frame spikes that would
  otherwise inflate the variance score.
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Optional

import numpy as np


AGC_SETTLE_SECONDS_DEFAULT = 10.0


@dataclasses.dataclass
class DetectorConfig:
    window: int = 50           # samples per sliding window (~0.5 s at 100 Hz)
    enter_ratio: float = 3.0   # score / baseline to trigger motion
    exit_ratio: float = 1.5    # score / baseline to clear motion
    min_baseline: float = 1e-3
    hampel_k: float = 3.0      # outlier threshold in MAD units
    hampel_window: int = 7     # odd window length for the running median


def hampel_filter(x: np.ndarray, k: float = 3.0, window: int = 7) -> np.ndarray:
    """Replace points more than k MADs from a rolling median with the median.

    Operates per-column when given a 2D array. Window must be odd; we pad
    with edge values so output shape matches input shape.
    """
    if window % 2 == 0:
        raise ValueError("hampel window must be odd")
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False

    half = window // 2
    padded = np.pad(x, ((half, half), (0, 0)), mode="edge")
    out = x.copy()
    # Vectorize over the window dimension.
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    # windows shape: (n, n_cols, window)
    med = np.median(windows, axis=-1)
    mad = np.median(np.abs(windows - med[..., None]), axis=-1)
    # 1.4826 makes MAD a consistent estimator of std for Gaussian noise.
    threshold = k * 1.4826 * mad
    deviations = np.abs(x - med)
    mask = (threshold > 0) & (deviations > threshold)
    out[mask] = med[mask]

    return out.squeeze(axis=1) if squeeze else out


class MotionDetector:
    def __init__(self, subcarrier_idx: np.ndarray, baseline: float,
                 config: DetectorConfig = DetectorConfig()):
        self.idx = subcarrier_idx
        self.baseline = max(float(baseline), config.min_baseline)
        self.cfg = config
        self._buf: collections.deque[np.ndarray] = collections.deque(maxlen=config.window)
        self._in_motion = False

    def update(self, amplitude: np.ndarray) -> tuple[float, bool]:
        self._buf.append(amplitude[self.idx])
        if len(self._buf) < self._buf.maxlen:
            return 0.0, self._in_motion
        stack = np.stack(self._buf)
        filtered = hampel_filter(stack, k=self.cfg.hampel_k, window=self.cfg.hampel_window)
        score = float(np.mean(np.std(filtered, axis=0)))
        ratio = score / self.baseline
        if not self._in_motion and ratio >= self.cfg.enter_ratio:
            self._in_motion = True
        elif self._in_motion and ratio <= self.cfg.exit_ratio:
            self._in_motion = False
        return score, self._in_motion


def compute_baseline(amplitudes: np.ndarray, window: int) -> float:
    """Median per-subcarrier sliding-window std across a still-room capture."""
    if amplitudes.shape[0] < window * 2:
        raise ValueError(
            f"need at least {window * 2} samples for a stable baseline, got {amplitudes.shape[0]}"
        )
    trim = window
    body = amplitudes[trim:-trim]
    filtered = hampel_filter(body)
    n_windows = filtered.shape[0] - window + 1
    scores = np.empty(n_windows, dtype=np.float64)
    for i in range(n_windows):
        scores[i] = np.mean(np.std(filtered[i : i + window], axis=0))
    return float(np.median(scores))
