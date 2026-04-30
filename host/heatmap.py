"""Multi-RX, multi-TX heatmap viewer.

Reads CSI samples over UDP, computes per-link motion-σ on a sliding
window, and draws a 2D floor-plan overlay where every TX-RX line is
tinted by its current motion intensity. With multiple TXs, each TX's
fan of links is drawn from a different orange-shaded star marker, so
the two fans are visually distinguishable.

Configuration is a JSON file describing the room polygon, TX
positions, and each RX's MAC + position. See `links.example.json`.

Run as:

    python run.py heatmap udp:5566 --links links.json [--baselines b.json]
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
class _Node:
    mac: str
    x: float
    y: float
    label: str


def _load_links(path: str) -> tuple[dict, list[_Node], list[_Node]]:
    with open(path) as f:
        cfg = json.load(f)
    txs_cfg = cfg["txs"] if "txs" in cfg else [cfg["tx"]]
    txs = []
    for i, t in enumerate(txs_cfg):
        if "mac" not in t:
            raise SystemExit(
                f"links config: TX entry {i} is missing 'mac'. "
                f"Add e.g. \"mac\": \"ac:a7:04:2c:42:54\" to that entry.")
        txs.append(_Node(mac=t["mac"].lower(), x=float(t["x"]), y=float(t["y"]),
                         label=t.get("label", f"TX{i+1}")))
    rxs = [_Node(mac=r["mac"].lower(), x=float(r["x"]), y=float(r["y"]),
                 label=r.get("label", r["mac"][-5:]))
           for r in cfg["rxs"]]
    return cfg, txs, rxs


class _LinkBuffer:
    """Per-(TX, RX) rolling amplitude buffer."""

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


def _reader_thread(source: str,
                   buffers: dict[tuple[str, str], _LinkBuffer],
                   tx_macs: set[str],
                   unknown: set[str],
                   stop: threading.Event) -> None:
    for sample in csi_collector.open_source(source):
        if stop.is_set():
            break
        if sample.rx_id is None:
            continue
        rx = sample.rx_id.lower()
        tx = sample.mac.lower()
        if tx not in tx_macs:
            continue
        key = (tx, rx)
        buf = buffers.get(key)
        if buf is None:
            unknown.add(rx)
            continue
        buf.push(sample)


def run_heatmap(source: str, links_path: str,
                history: int = 500, motion_window: int = 50,
                baselines_path: Optional[str] = None) -> int:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.cm import get_cmap

    cfg, txs, rxs = _load_links(links_path)
    room = cfg["room"]
    tx_macs = {t.mac for t in txs}

    if "polygon" in room:
        polygon = np.array(room["polygon"], dtype=float)
        bbox_min, bbox_max = polygon.min(axis=0), polygon.max(axis=0)
    else:
        w, h = float(room["width_m"]), float(room["height_m"])
        polygon = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
        bbox_min, bbox_max = np.array([0.0, 0.0]), np.array([w, h])

    buffers: dict[tuple[str, str], _LinkBuffer] = {
        (t.mac, r.mac): _LinkBuffer(history) for t in txs for r in rxs
    }
    baselines: dict[str, float] = {}
    if baselines_path:
        with open(baselines_path) as f:
            baselines = {k.lower(): float(v) for k, v in json.load(f).items()}
    unknown: set[str] = set()
    stop = threading.Event()
    threading.Thread(target=_reader_thread,
                     args=(source, buffers, tx_macs, unknown, stop),
                     daemon=True).start()

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle(f"CSI link heatmap — {source}")
    ax.set_xlim(bbox_min[0] - 0.3, bbox_max[0] + 0.3)
    ax.set_ylim(bbox_min[1] - 0.3, bbox_max[1] + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.add_patch(plt.Polygon(polygon, fill=False, edgecolor="black", linewidth=1.5))

    cmap = get_cmap("magma")
    # One line + value-label per (TX, RX) pair.
    line_artists: list = []
    label_artists: list = []
    pair_keys: list[tuple[str, str]] = []
    for tx in txs:
        for rx in rxs:
            line, = ax.plot([tx.x, rx.x], [tx.y, rx.y],
                            color=cmap(0.0), linewidth=3, solid_capstyle="round",
                            alpha=0.85)
            line_artists.append(line)
            pair_keys.append((tx.mac, rx.mac))
            # Tiny value label at the midpoint, so multiple lines through
            # an RX don't pile their text on top of each other.
            mx, my = (tx.x + rx.x) / 2.0, (tx.y + rx.y) / 2.0
            label_artists.append(ax.text(mx, my, "", fontsize=7,
                                         color="white",
                                         ha="center", va="center",
                                         bbox=dict(facecolor="black", alpha=0.5,
                                                   edgecolor="none", pad=1.5)))

    # RX dots + labels (drawn after lines so they sit on top).
    for rx in rxs:
        ax.plot(rx.x, rx.y, "o", color="tab:blue", markersize=12, zorder=5)
        ax.text(rx.x, rx.y + 0.18, rx.label, ha="center", va="bottom",
                fontsize=9, color="tab:blue", zorder=6)
    # TX stars; each TX gets a slightly different shade so the two fans
    # are visually distinguishable. With one TX, this still draws a star.
    tx_shades = ["tab:orange", "tab:red", "darkorange", "firebrick"]
    for i, tx in enumerate(txs):
        ax.plot(tx.x, tx.y, "*", color=tx_shades[i % len(tx_shades)],
                markersize=20, zorder=5)
        ax.text(tx.x, tx.y + 0.18, tx.label, ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color=tx_shades[i % len(tx_shades)], zorder=6)

    # Coloring modes match the 3D viewer:
    # - With baselines: color by ratio to per-link still-room baseline
    #   (1× = idle, RATIO_FULL_BRIGHT× = saturated). Stable across runs.
    # - Without: fall back to running-max normalization.
    RATIO_FULL_BRIGHT = 5.0
    use_ratio = bool(baselines)
    label = "motion ratio (× still-room)" if use_ratio else "motion σ (normalized)"
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, label=label)
    del cbar

    running_max = [1e-3]

    def update(_frame):
        # Compute per-link motion score, then per-link metric (ratio or
        # raw σ) for tint and label.
        sigmas = [buffers[k].motion_score(motion_window) for k in pair_keys]
        if use_ratio:
            # Same baseline applies to every TX from a given RX (still-
            # room noise is RX-side, not link-specific).
            metrics = []
            for (tx_mac, rx_mac), sigma in zip(pair_keys, sigmas):
                base = max(baselines.get(rx_mac, 1e-3), 1e-6)
                metrics.append(sigma / base)
            tints = [min(m / RATIO_FULL_BRIGHT, 1.0) for m in metrics]
            text_fmt = lambda m, s: f"{m:.2f}×"
        else:
            if sigmas:
                running_max[0] = max(running_max[0] * 0.99, max(sigmas), 1e-3)
            tints = [min(s / running_max[0], 1.0) for s in sigmas]
            metrics = sigmas
            text_fmt = lambda m, s: f"{s:.3f}"
        for line, lbl, tint, m, s in zip(
                line_artists, label_artists, tints, metrics, sigmas):
            line.set_color(cmap(tint))
            lbl.set_text(text_fmt(m, s))
        if unknown:
            ax.set_title(f"unknown RX MACs (add to links.json): "
                         f"{', '.join(sorted(unknown))}",
                         fontsize=8, color="tab:red")
        return [*line_artists, *label_artists]

    anim = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        stop.set()
    del anim
    return 0
