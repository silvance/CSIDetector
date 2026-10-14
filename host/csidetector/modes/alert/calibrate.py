"""Alert-mode still-room calibration.

Single-stream baseline: drops the first ``settle_seconds`` after the
radio's AGC settles, then collects ``seconds`` (or ``max_samples``,
whichever first) of CSI and computes the σ baseline used by
``alert detect``.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional

import numpy as np

from csidetector.core import collector as csi_collector
from csidetector.core import detector


def collect_amplitudes(source: str,
                       seconds: Optional[float],
                       max_samples: Optional[int],
                       settle_seconds: float = 0.0
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Stream samples from ``source`` until ``seconds`` or ``max_samples``."""
    if seconds is None and max_samples is None:
        raise SystemExit(
            "calibrate: at least one of --seconds or --max-samples is "
            "required (otherwise it would run forever).")
    start = time.monotonic()            # see detect.py
    settle_until = start + settle_seconds
    deadline = (settle_until + seconds) if seconds else None
    stop = threading.Event()

    def idle() -> None:
        # Called while no data arrives (serial): without it the deadline
        # is only checked per sample, and no CSI would mean waiting forever.
        if deadline and time.monotonic() >= deadline:
            stop.set()

    src = csi_collector.open_source(source, stop=stop, on_idle=idle)
    rows: list[np.ndarray] = []
    idx = None
    skipped = 0
    for sample in src:
        if time.monotonic() < settle_until:
            skipped += 1
            continue
        if idx is None:
            idx = np.flatnonzero(sample.amplitude > 0)
            if idx.size == 0:
                continue
        rows.append(sample.amplitude)
        if max_samples and len(rows) >= max_samples:
            break
        if deadline and time.monotonic() >= deadline:
            break
    if not rows or idx is None:
        raise SystemExit(
            f"no CSI samples received from {source}. Check that the transmitter "
            f"is powered and was flashed for the same channel as the receiver, "
            f"and that this is the receiver's port.")
    if settle_seconds > 0:
        print(f"dropped {skipped} samples during AGC settle "
              f"({settle_seconds:.1f}s)", file=sys.stderr)
    amps = np.stack(rows)[:, idx]
    return amps, idx


def run_calibrate(source: str,
                  seconds: Optional[float],
                  settle: float,
                  max_samples: Optional[int],
                  window: int) -> int:
    """Compute and print the still-room σ baseline for a single stream."""
    amps, idx = collect_amplitudes(source, seconds, max_samples,
                                   settle_seconds=settle)
    baseline = detector.compute_baseline(amps, window)
    print(f"baseline={baseline:.6f}  samples={amps.shape[0]}  "
          f"subcarriers={idx.size}", file=sys.stderr)
    print(f"{baseline:.6f}")
    return 0
