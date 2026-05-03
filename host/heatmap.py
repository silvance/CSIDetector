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


# Ratios at and below this anchor mean "no motion" (output equals or
# falls below the still-room baseline) and render as black. Anything
# above is the dynamic range we color across.
RATIO_FLOOR = 1.0
# Default ratio at which links saturate to the brightest cmap value.
# Real CSI motion ratios rarely exceed ~2-3×; saturating at 5× as the
# old default did pushed every realistic motion into magma's near-
# black lower third.
DEFAULT_RATIO_FULL_BRIGHT = 3.0


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
    """Per-(TX, RX) rolling amplitude buffer.

    Active-subcarrier mask is derived from the first MASK_PROBE samples
    (union of nonzero-anywhere) instead of being locked from the very
    first sample. A flaky first frame would otherwise permanently drop
    a subcarrier from this link's view.
    """

    MASK_PROBE = 32

    def __init__(self, capacity: int):
        self._buf: collections.deque[np.ndarray] = collections.deque(maxlen=capacity)
        self._idx: Optional[np.ndarray] = None
        # Pre-mask buffer of raw amplitudes for the probe phase.
        self._probe: list[np.ndarray] = []
        self._lock = threading.Lock()

    def push(self, sample: csi_collector.CSISample) -> None:
        amp = sample.amplitude
        with self._lock:
            if self._idx is None:
                self._probe.append(amp)
                if len(self._probe) < self.MASK_PROBE:
                    return
                # Take the union of nonzero subcarriers across the probe
                # window — guards against a single all-zero frame.
                stacked = np.stack(self._probe)
                idx = np.flatnonzero(np.any(stacked > 0, axis=0))
                if idx.size == 0:
                    # Probe came back all-zero (link is dead). Drop the
                    # probe and try again — eventually we either get
                    # data or stay stuck (which is what we'd want).
                    self._probe.clear()
                    return
                self._idx = idx
                # Backfill the probe samples with the chosen mask.
                for a in self._probe:
                    self._buf.append(a[idx])
                self._probe = []  # release the references
            else:
                # Drop samples whose subcarrier count differs from the
                # mask we locked in. Happens on a mid-stream MCS or
                # bandwidth shift; without this, indexing IndexErrors
                # and silently kills the reader thread.
                if self._idx[-1] >= amp.size:
                    return
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
                   unknown_rx: set[str],
                   unknown_tx: set[str],
                   stop: threading.Event) -> None:
    for sample in csi_collector.open_source(source):
        if stop.is_set():
            break
        if sample.rx_id is None:
            continue
        rx = sample.rx_id.lower()
        tx = sample.mac.lower()
        if tx not in tx_macs:
            unknown_tx.add(tx)
            continue
        key = (tx, rx)
        buf = buffers.get(key)
        if buf is None:
            unknown_rx.add(rx)
            continue
        buf.push(sample)


def _load_baselines(path: Optional[str], txs, rxs) -> dict[tuple[str, str], float]:
    """Read baselines.json and return per-(tx_mac, rx_mac) values.

    Two formats accepted:
      - New: keys are "tx_mac|rx_mac" → float (one entry per link).
      - Legacy: keys are "rx_mac" → float (per-RX). Replicated across
        every TX from that RX so old files still work; logged as
        "applying same baseline to multiple TXs from <RX>" so users
        know to recalibrate when accuracy matters.
    """
    if not path:
        return {}
    with open(path) as f:
        raw = json.load(f)
    out: dict[tuple[str, str], float] = {}
    legacy_rx_macs: set[str] = set()
    tx_macs = [t.mac for t in txs]
    for k, v in raw.items():
        k = k.lower()
        if "|" in k:
            tx, rx = k.split("|", 1)
            out[(tx, rx)] = float(v)
        else:
            # Legacy per-RX entry — fan out to every TX.
            legacy_rx_macs.add(k)
            for tx in tx_macs:
                out[(tx, k)] = float(v)
    if legacy_rx_macs:
        print(f"heatmap: baselines.json uses legacy per-RX schema for "
              f"{len(legacy_rx_macs)} entries; same baseline applied to all "
              f"TX→RX links from each. Re-run `calibrate-links` for a "
              f"per-link baseline.")
    return out


def run_heatmap(source: str, links_path: str,
                history: int = 500, motion_window: int = 50,
                baselines_path: Optional[str] = None,
                full_bright: float = DEFAULT_RATIO_FULL_BRIGHT) -> int:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib as mpl

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
    baselines = _load_baselines(baselines_path, txs, rxs)
    unknown_rx: set[str] = set()
    unknown_tx: set[str] = set()
    stop = threading.Event()
    threading.Thread(target=_reader_thread,
                     args=(source, buffers, tx_macs,
                           unknown_rx, unknown_tx, stop),
                     daemon=True).start()

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle(f"CSI link heatmap — {source}")
    ax.set_xlim(bbox_min[0] - 0.3, bbox_max[0] + 0.3)
    ax.set_ylim(bbox_min[1] - 0.3, bbox_max[1] + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.add_patch(plt.Polygon(polygon, fill=False, edgecolor="black", linewidth=1.5))

    cmap = mpl.colormaps["magma"]
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

    # Coloring: with baselines, ratio = current_σ / per-link baseline,
    # tint anchored so RATIO_FLOOR (=1×, still-room) maps to black and
    # full_bright (e.g. 3×) saturates. Without baselines, fall back to
    # running-max normalization.
    use_ratio = bool(baselines)
    if use_ratio:
        if full_bright <= RATIO_FLOOR:
            raise SystemExit(
                f"--full-bright must exceed {RATIO_FLOOR} (got {full_bright}); "
                f"otherwise the dynamic range collapses.")
        cbar_label = (f"motion ratio (× still-room) — "
                      f"floor {RATIO_FLOOR:g}×, full {full_bright:g}×")
        norm = plt.Normalize(vmin=RATIO_FLOOR, vmax=full_bright)
    else:
        cbar_label = "motion σ (normalized)"
        norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label=cbar_label)

    running_max = [1e-3]
    span = full_bright - RATIO_FLOOR

    def update(_frame):
        sigmas = [buffers[k].motion_score(motion_window) for k in pair_keys]
        if use_ratio:
            metrics = []
            has_baseline = []
            for (tx_mac, rx_mac), sigma in zip(pair_keys, sigmas):
                base = baselines.get((tx_mac, rx_mac))
                if base is None or base <= 0:
                    # No (or zero) baseline for this link: don't fabricate
                    # a 1e-3 divisor — that produced phantom huge ratios
                    # and a saturated colormap. Render dim with a "—"
                    # label so the missing baseline is visible.
                    metrics.append(0.0)
                    has_baseline.append(False)
                else:
                    metrics.append(sigma / base)
                    has_baseline.append(True)
            tints = [
                0.0 if not ok else min(max(m - RATIO_FLOOR, 0.0) / span, 1.0)
                for m, ok in zip(metrics, has_baseline)
            ]
            text_fmt = lambda m, s, ok: f"{m:.2f}×" if ok else "—"
        else:
            if sigmas:
                running_max[0] = max(running_max[0] * 0.99, max(sigmas), 1e-3)
            tints = [min(s / running_max[0], 1.0) for s in sigmas]
            metrics = sigmas
            has_baseline = [True] * len(sigmas)
            text_fmt = lambda m, s, ok: f"{s:.3f}"
        for line, lbl, tint, m, s, ok in zip(
                line_artists, label_artists, tints, metrics, sigmas, has_baseline):
            line.set_color(cmap(tint))
            lbl.set_text(text_fmt(m, s, ok))
        notes = []
        if unknown_rx:
            notes.append(f"unknown RX: {', '.join(sorted(unknown_rx))}")
        if unknown_tx:
            notes.append(f"unknown TX: {', '.join(sorted(unknown_tx))}")
        if notes:
            ax.set_title("  |  ".join(notes), fontsize=8, color="tab:red")
        return [*line_artists, *label_artists]

    anim = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    # `anim` is intentionally bound for the duration of plt.show(); without
    # a live reference, matplotlib garbage-collects FuncAnimation and the
    # animation freezes silently.
    try:
        plt.show()
    finally:
        stop.set()
    return 0
