"""Multi-RX heatmap viewer.

Reads CSI samples from N receivers over UDP, computes per-link motion-σ
on a sliding window, and draws a 2D floor-plan overlay where each
TX-RX line is tinted by its current motion intensity. Crossings of
high-intensity links indicate roughly where motion is happening.

Configuration is a JSON file describing the room, TX position, and
each RX's MAC + position. See `links.example.json`.

Run as:

    python run.py heatmap udp:5566 --links links.json
"""

from __future__ import annotations

import collections
import json
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

import csi_collector


@dataclass
class _RXConfig:
    mac: str
    x: float
    y: float
    label: str


def _load_links(path: str) -> tuple[dict, list[_RXConfig]]:
    with open(path) as f:
        cfg = json.load(f)
    rxs = [_RXConfig(mac=r["mac"].lower(), x=r["x"], y=r["y"], label=r.get("label", r["mac"][-5:]))
           for r in cfg["rxs"]]
    return cfg, rxs


class _LinkBuffer:
    """Per-RX rolling amplitude buffer + active-subcarrier mask."""

    def __init__(self, capacity: int):
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

    def motion_score(self, window: int) -> float:
        with self._lock:
            if len(self._buf) < window or self._idx is None:
                return 0.0
            recent = np.stack(list(self._buf)[-window:])
        return float(np.mean(np.std(recent, axis=0)))


def _reader_thread(source: str, buffers: dict[str, _LinkBuffer],
                   unknown_macs: set[str], stop: threading.Event) -> None:
    for sample in csi_collector.open_source(source):
        if stop.is_set():
            break
        if sample.rx_id is None:
            continue
        rx = sample.rx_id.lower()
        buf = buffers.get(rx)
        if buf is None:
            unknown_macs.add(rx)
            continue
        buf.push(sample)


def run_heatmap(source: str, links_path: str,
                history: int = 500, motion_window: int = 50) -> int:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.cm import get_cmap

    cfg, rxs = _load_links(links_path)
    room = cfg["room"]
    tx = cfg["tx"]
    buffers = {rx.mac: _LinkBuffer(history) for rx in rxs}
    unknown = set()
    stop = threading.Event()
    reader = threading.Thread(target=_reader_thread,
                              args=(source, buffers, unknown, stop), daemon=True)
    reader.start()

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(f"CSI link heatmap — {source}")
    ax.set_xlim(-0.2, room["width_m"] + 0.2)
    ax.set_ylim(-0.2, room["height_m"] + 0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.add_patch(plt.Rectangle((0, 0), room["width_m"], room["height_m"],
                               fill=False, edgecolor="black", linewidth=1.5))

    cmap = get_cmap("magma")
    line_artists = []
    label_artists = []
    for rx in rxs:
        line, = ax.plot([tx["x"], rx.x], [tx["y"], rx.y],
                        color=cmap(0.0), linewidth=4, solid_capstyle="round")
        line_artists.append(line)
        ax.plot(rx.x, rx.y, "o", color="tab:blue", markersize=12)
        label_artists.append(ax.text(rx.x, rx.y + 0.15, f"{rx.label}\n0.000",
                                     ha="center", va="bottom", fontsize=9))
    ax.plot(tx["x"], tx["y"], "*", color="tab:orange", markersize=18)
    ax.text(tx["x"], tx["y"] + 0.15, tx.get("label", "TX"),
            ha="center", va="bottom", fontsize=10, fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, label="motion σ (normalized)")
    del cbar

    # Auto-scale: the motion-σ floor depends on noise; track running max
    # and normalize lines against it so weak/strong motion both render.
    running_max = [1e-3]

    def update(_frame):
        scores = [buffers[rx.mac].motion_score(motion_window) for rx in rxs]
        if scores:
            running_max[0] = max(running_max[0] * 0.99, max(scores), 1e-3)
        denom = running_max[0]
        for line, lbl, rx, score in zip(line_artists, label_artists, rxs, scores):
            tint = min(score / denom, 1.0)
            line.set_color(cmap(tint))
            lbl.set_text(f"{rx.label}\n{score:.4f}")
        if unknown:
            ax.set_title(f"unknown MACs streaming (add to links.json): {', '.join(sorted(unknown))}",
                         fontsize=8, color="tab:red")
        return [*line_artists, *label_artists]

    anim = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        stop.set()
    del anim
    return 0
