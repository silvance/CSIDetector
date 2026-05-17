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
import math
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
        # Reject NaN / non-positive / non-finite baselines explicitly.
        # max(nan, x) returns nan in Python, which then produces nan
        # ratios — neither enter nor exit branch fires, the detector
        # silently never alerts, and the operator has no signal that
        # anything is wrong. Better to crash now with a clear error.
        if not math.isfinite(baseline) or baseline <= 0:
            raise ValueError(
                f"MotionDetector: baseline must be a positive finite number, "
                f"got {baseline!r}. Re-run calibration.")
        if config.window <= 0:
            raise ValueError(
                f"MotionDetector: window must be > 0, got {config.window!r}")
        self.idx = subcarrier_idx
        self.baseline = max(float(baseline), config.min_baseline)
        self.cfg = config
        self._buf: collections.deque[np.ndarray] = collections.deque(maxlen=config.window)
        self._in_motion = False

    def update(self, amplitude: np.ndarray) -> tuple[float, bool]:
        # Defensively drop samples whose subcarrier count is too small
        # for the index — happens on mid-stream bandwidth/MCS changes
        # the same way it does for the multi-link _LinkBuffer. Without
        # this, alert-mode `detect` crashes with IndexError and stops
        # delivering notifications.
        if self.idx.size and self.idx.max() >= amplitude.size:
            return 0.0, self._in_motion
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
    """Median per-subcarrier sliding-window std across a still-room capture.

    Trims `window` samples from each end, then takes a sliding window of
    size `window` over the remainder — so the minimum input is
    `3*window` samples to produce a single sliding window. The previous
    `2*window` check was off by one and let the body shrink to zero
    rows, which crashed numpy on `np.empty(-window+1)`.
    """
    min_required = 3 * window
    if amplitudes.shape[0] < min_required:
        raise ValueError(
            f"need at least {min_required} samples for a stable baseline, "
            f"got {amplitudes.shape[0]}"
        )
    trim = window
    body = amplitudes[trim:-trim]
    filtered = hampel_filter(body)
    n_windows = filtered.shape[0] - window + 1
    scores = np.empty(n_windows, dtype=np.float64)
    for i in range(n_windows):
        scores[i] = np.mean(np.std(filtered[i : i + window], axis=0))
    return float(np.median(scores))


def compute_link_baselines(samples_by_link: dict[tuple[str, str], list[np.ndarray]],
                            window: int) -> dict[tuple[str, str], float]:
    """Per-(tx_mac, rx_mac) baseline from a multi-link still-room capture.

    Keyed by link tuple, not RX-only: TX1↔RX_n and TX2↔RX_n have
    genuinely different still-room σ because their multipath paths
    differ, and applying a single per-RX baseline mis-normalizes one
    of them. Each link needs at least 2*window samples; links with
    fewer are skipped (likely never received during the capture).

    Active subcarriers are derived from the union of "non-zero in any
    sample" rather than locked from the first sample, so a flaky
    first frame can't permanently exclude a subcarrier.
    """
    out: dict[tuple[str, str], float] = {}
    for key, rows in samples_by_link.items():
        if len(rows) < window * 3:  # matches compute_baseline's minimum
            continue
        amps = np.stack(rows)
        idx = np.flatnonzero(np.any(amps > 0, axis=0))
        if idx.size == 0:
            continue
        out[key] = compute_baseline(amps[:, idx], window)
    return out
