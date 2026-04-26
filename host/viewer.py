"""Live CSI heatmap viewer.

Renders a rolling subcarrier-vs-time waterfall: x-axis is time (newest on
the right), y-axis is active subcarrier, color is amplitude in dB. Motion
in front of the radio shows up as vertical color streaks across the
heatmap. Optionally overlays a running motion-intensity score.

Run as:

    python host/run.py view /dev/ttyUSB0
"""

from __future__ import annotations

import collections
import threading
from typing import Optional

import numpy as np

import csi_collector


class _SampleBuffer:
    """Thread-safe ring buffer of recent CSI amplitude vectors."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: collections.deque[np.ndarray] = collections.deque(maxlen=capacity)
        self._idx: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def push(self, sample: csi_collector.CSISample) -> None:
        amp = sample.amplitude
        with self._lock:
            if self._idx is None:
                idx = np.flatnonzero(amp > 0)
                if idx.size == 0:
                    return
                self._idx = idx
            self._buf.append(amp[self._idx])

    def snapshot(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            if not self._buf or self._idx is None:
                return None, None
            stack = np.stack(self._buf)
            return stack, self._idx.copy()


def _reader_thread(source: str, buf: _SampleBuffer, stop: threading.Event) -> None:
    for sample in csi_collector.open_source(source):
        if stop.is_set():
            break
        buf.push(sample)


def run_viewer(source: str, history: int = 500, motion_window: int = 50) -> int:
    """Open a matplotlib window with a live CSI waterfall + motion score.

    `history` is the number of samples shown horizontally (~5 s at 100 Hz).
    `motion_window` is the trailing window used for the motion-score line.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    buf = _SampleBuffer(history)
    stop = threading.Event()
    reader = threading.Thread(target=_reader_thread, args=(source, buf, stop), daemon=True)
    reader.start()

    fig, (ax_heat, ax_score) = plt.subplots(
        2, 1, figsize=(10, 6),
        gridspec_kw={"height_ratios": [4, 1]}, sharex=False,
    )
    fig.suptitle(f"CSI live view — {source}")

    # Placeholder image until we know subcarrier count.
    img = ax_heat.imshow(
        np.zeros((1, history)), aspect="auto", origin="lower",
        cmap="viridis", interpolation="nearest",
    )
    ax_heat.set_ylabel("active subcarrier")
    ax_heat.set_xlabel("samples (newest →)")
    cbar = fig.colorbar(img, ax=ax_heat, label="|H| (dB)")
    del cbar  # keep reference alive without lint warning

    score_line, = ax_score.plot([], [], color="tab:red")
    ax_score.set_xlim(0, history)
    ax_score.set_ylabel("motion σ")
    ax_score.set_xlabel("samples")
    ax_score.grid(True, alpha=0.3)

    def update(_frame):
        stack, idx = buf.snapshot()
        if stack is None or idx is None:
            return img, score_line
        amps_db = 20 * np.log10(stack + 1e-3)
        # imshow expects (rows, cols) = (subcarriers, time). stack is (time, subcarriers).
        img.set_data(amps_db.T)
        img.set_extent([0, stack.shape[0], 0, idx.size])
        vmin, vmax = np.percentile(amps_db, [5, 99])
        if vmax > vmin:
            img.set_clim(vmin, vmax)

        if stack.shape[0] >= motion_window:
            scores = np.empty(stack.shape[0] - motion_window + 1)
            for i in range(scores.size):
                scores[i] = np.mean(np.std(stack[i : i + motion_window], axis=0))
            xs = np.arange(motion_window - 1, stack.shape[0])
            score_line.set_data(xs, scores)
            ax_score.set_ylim(0, max(scores.max() * 1.2, 1e-3))
        return img, score_line

    anim = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        stop.set()
    del anim
    return 0
