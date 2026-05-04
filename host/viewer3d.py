"""2.5D room viewer.

Floor as a likelihood heatmap (from `localize.Localizer`), walls
extruded vertically, person estimate as a vertical pin at the
likelihood argmax. Updates at the matplotlib animation rate (~10 Hz);
math is fast enough that this isn't a bottleneck.

The viewer accepts the same `links.json` and `baselines.json` as the
flat heatmap. Multi-TX schemas are supported (txs is a list); a config
with the legacy single `tx` key is normalized to a 1-element list.
"""

from __future__ import annotations

import collections
import json
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

import csi_collector
import localize


@dataclass
class _Node:
    mac: str
    x: float
    y: float
    label: str


def _load_links(path: str) -> tuple[np.ndarray, list[_Node], list[_Node]]:
    with open(path) as f:
        cfg = json.load(f)
    room = cfg["room"]
    if "polygon" in room:
        polygon = np.array(room["polygon"], dtype=float)
    else:
        # Backward-compat with the rectangle schema.
        w, h = float(room["width_m"]), float(room["height_m"])
        polygon = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    txs_cfg = cfg["txs"] if "txs" in cfg else [cfg["tx"]]
    txs = []
    for i, t in enumerate(txs_cfg):
        if "mac" not in t:
            raise SystemExit(
                f"links config: TX entry {i} is missing 'mac' — every "
                f"transmitter needs its factory MAC so source-tagged samples "
                f"can be routed to the right link. Add e.g. "
                f'"mac": "ac:a7:04:2c:42:54" to that entry.'
            )
        txs.append(_Node(mac=t["mac"].lower(), x=float(t["x"]), y=float(t["y"]),
                         label=t.get("label", f"TX{i+1}")))
    rxs = [_Node(mac=r["mac"].lower(), x=float(r["x"]), y=float(r["y"]),
                 label=r.get("label", r["mac"][-5:]))
           for r in cfg["rxs"]]
    return polygon, txs, rxs


class _LinkBuffer:
    """Per-(TX, RX) rolling amplitude buffer.

    Active-subcarrier mask from the first MASK_PROBE samples (union of
    nonzero-anywhere) so a flaky first frame can't permanently drop a
    subcarrier. Same pattern as heatmap._LinkBuffer.
    """

    MASK_PROBE = 32

    def __init__(self, capacity: int):
        self._buf: collections.deque[np.ndarray] = collections.deque(maxlen=capacity)
        self._idx: Optional[np.ndarray] = None
        self._probe: list[np.ndarray] = []
        self._lock = threading.Lock()

    def push(self, sample: csi_collector.CSISample) -> None:
        amp = sample.amplitude
        with self._lock:
            if self._idx is None:
                self._probe.append(amp)
                if len(self._probe) < self.MASK_PROBE:
                    return
                stacked = np.stack(self._probe)
                idx = np.flatnonzero(np.any(stacked > 0, axis=0))
                if idx.size == 0:
                    self._probe.clear()
                    return
                self._idx = idx
                for a in self._probe:
                    self._buf.append(a[idx])
                self._probe = []
            else:
                # Drop samples whose subcarrier count differs from the
                # locked-in mask — guards against IndexError when a
                # mid-stream MCS / bandwidth shift shrinks the array.
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
                   stop: threading.Event,
                   status: dict) -> None:
    # See heatmap._reader_thread — same try/except pattern. One bad
    # packet shouldn't kill the thread; a dead source should be visible.
    try:
        for sample in csi_collector.open_source(source):
            if stop.is_set():
                break
            try:
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
                status["last_packet_ts"] = time.monotonic()
                status["pkt_count"] = status.get("pkt_count", 0) + 1
            except Exception as exc:
                status["bad_samples"] = status.get("bad_samples", 0) + 1
                status["last_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        status["fatal_error"] = f"{type(exc).__name__}: {exc}"


def _load_baselines(path: Optional[str], txs, rxs) -> dict[tuple[str, str], float]:
    """Read baselines.json. Accepts the wrapped {_meta, links} envelope,
    flat keys "tx_mac|rx_mac" (per-link), or legacy "rx_mac" keys (per-RX,
    fanned out to every TX).
    """
    if not path:
        return {}
    import os
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "links" in raw and "_meta" in raw:
        link_map = raw["links"]
        try:
            age_s = time.time() - os.path.getmtime(path)
            if age_s > 3600:
                age_h = age_s / 3600.0
                print(f"view3d: WARNING — baselines.json is {age_h:.1f}h old; "
                      f"re-run `calibrate-links` if motion looks off "
                      f"(RF drift on this timescale is common).")
        except OSError:
            pass
    else:
        link_map = raw
    out: dict[tuple[str, str], float] = {}
    legacy = 0
    tx_macs = [t.mac for t in txs]
    for k, v in link_map.items():
        k = k.lower()
        if "|" in k:
            tx, rx = k.split("|", 1)
            out[(tx, rx)] = float(v)
        else:
            legacy += 1
            for tx in tx_macs:
                out[(tx, k)] = float(v)
    if legacy:
        print(f"view3d: baselines.json has {legacy} legacy per-RX entries; "
              f"each replicated across all TXs from that RX. Re-run "
              f"`calibrate-links` for per-link baselines.")
    return out


def run_viewer3d(source: str, links_path: str,
                 history: int = 500, motion_window: int = 50,
                 baselines_path: Optional[str] = None,
                 grid_step: float = 0.1, link_sigma_m: float = 0.3,
                 node_exclusion_m: float = 0.5,
                 wall_height_m: float = 2.5,
                 max_pins: int = 3,
                 pin_separation_m: float = 1.0,
                 pin_smoothing: float = 0.4) -> int:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib as mpl
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    polygon, txs, rxs = _load_links(links_path)
    tx_macs = {t.mac for t in txs}
    tx_pos = {t.mac: np.array([t.x, t.y]) for t in txs}
    rx_pos = {r.mac: np.array([r.x, r.y]) for r in rxs}

    loc = localize.Localizer(polygon, tx_pos, rx_pos,
                             grid_step=grid_step, link_sigma_m=link_sigma_m,
                             node_exclusion_m=node_exclusion_m)

    # baselines.json supports two key formats; see _load_baselines below.
    baselines = _load_baselines(baselines_path, txs, rxs)
    use_ratio = bool(baselines)

    buffers: dict[tuple[str, str], _LinkBuffer] = {
        (t.mac, r.mac): _LinkBuffer(history) for t in txs for r in rxs
    }
    unknown_rx: set[str] = set()
    unknown_tx: set[str] = set()
    status: dict = {"pkt_count": 0, "bad_samples": 0, "last_packet_ts": 0.0}
    stop = threading.Event()
    threading.Thread(target=_reader_thread,
                     args=(source, buffers, tx_macs,
                           unknown_rx, unknown_tx, stop, status),
                     daemon=True).start()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle(f"CSI 2.5D — {source}")
    cmap = mpl.colormaps["magma"]

    # Walls: extrude each polygon edge into a vertical quad.
    wall_polys = []
    n = len(polygon)
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        wall_polys.append([
            (p0[0], p0[1], 0.0),
            (p1[0], p1[1], 0.0),
            (p1[0], p1[1], wall_height_m),
            (p0[0], p0[1], wall_height_m),
        ])
    walls = Poly3DCollection(wall_polys, facecolor=(0.85, 0.85, 0.9, 0.15),
                             edgecolor="black", linewidths=1.0)
    ax.add_collection3d(walls)

    # Floor surface — pcolormesh-equivalent in 3D via plot_surface with z=0.
    Z0 = np.zeros_like(loc.X)
    surf = ax.plot_surface(loc.X, loc.Y, Z0, facecolors=cmap(np.zeros_like(loc.X)),
                           rstride=1, cstride=1, shade=False, antialiased=False,
                           edgecolor="none")

    # TX/RX markers.
    for t in txs:
        ax.scatter([t.x], [t.y], [0], s=120, marker="*", color="tab:orange",
                   depthshade=False)
        ax.text(t.x, t.y, 0.05, t.label, color="tab:orange",
                fontsize=9, fontweight="bold")
    for r in rxs:
        ax.scatter([r.x], [r.y], [0], s=60, marker="o", color="tab:blue",
                   depthshade=False)
        ax.text(r.x, r.y, 0.05, r.label, color="tab:blue", fontsize=8)

    # Up to `max_pins` person pins (one per detected local maximum
    # after non-max suppression). Each pin keeps its own EMA-smoothed
    # position so it doesn't chatter cell-to-cell on a noisy grid.
    pin_lines = []
    pin_dots = []
    pin_state: list[Optional[tuple[float, float]]] = [None] * max_pins
    PIN_PALETTE = ["tab:red", "tab:cyan", "tab:green", "tab:purple", "tab:olive"]
    for i in range(max_pins):
        c = PIN_PALETTE[i % len(PIN_PALETTE)]
        line, = ax.plot([0, 0], [0, 0], [0, 1.7], color=c,
                        linewidth=3, alpha=0.0)
        dot, = ax.plot([0], [0], [1.7], "o", color=c, markersize=10,
                       alpha=0.0)
        pin_lines.append(line)
        pin_dots.append(dot)

    bbox_min = polygon.min(axis=0)
    bbox_max = polygon.max(axis=0)
    ax.set_xlim(bbox_min[0] - 0.2, bbox_max[0] + 0.2)
    ax.set_ylim(bbox_min[1] - 0.2, bbox_max[1] + 0.2)
    ax.set_zlim(0, wall_height_m + 0.2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=35, azim=-60)
    ax.set_box_aspect((bbox_max[0] - bbox_min[0],
                       bbox_max[1] - bbox_min[1],
                       wall_height_m))

    # Auto-scale: track running max of grid value so a quiet room renders
    # uniformly dark and motion stands out.
    running_max = [1e-3]
    PIN_THRESHOLD = 0.5  # show person pin when normalized argmax > this

    def update(_frame):
        scores: dict[tuple[str, str], float] = {}
        for (tx_mac, rx_mac), buf in buffers.items():
            sigma = buf.motion_score(motion_window)
            if use_ratio:
                base = baselines.get((tx_mac, rx_mac))
                if base is None or base <= 0:
                    # No baseline for this link — don't fabricate a 1e-3
                    # divisor (would saturate this link's kernel and
                    # produce a phantom person pin). Drop to 0.
                    metric = 0.0
                else:
                    # Subtract baseline so still-room links contribute ~0.
                    metric = max(sigma / base - 1.0, 0.0)
            else:
                metric = sigma
            scores[(tx_mac, rx_mac)] = metric

        grid = loc.update(scores)
        running_max[0] = max(running_max[0] * 0.99, float(grid.max()), 1e-3)
        norm = grid / running_max[0]
        # `plot_surface` doesn't update facecolors cleanly; redrawing every
        # frame is the documented workaround. Cheap at this grid size.
        surf.set_facecolors(cmap(np.clip(norm, 0, 1)).reshape(-1, 4))

        peaks = loc.topk_local_maxima(grid, max_pins,
                                      min_separation_m=pin_separation_m)
        # Render up to max_pins peaks above PIN_THRESHOLD with EMA-smoothed
        # positions so pins don't chatter cell-to-cell.
        alpha = pin_smoothing
        for i, (line, dot) in enumerate(zip(pin_lines, pin_dots)):
            if i < len(peaks):
                px, py, pv = peaks[i]
                normv = pv / running_max[0]
                if normv > PIN_THRESHOLD:
                    prev = pin_state[i]
                    if prev is None:
                        sx, sy = px, py
                    else:
                        sx = alpha * px + (1.0 - alpha) * prev[0]
                        sy = alpha * py + (1.0 - alpha) * prev[1]
                    pin_state[i] = (sx, sy)
                    line.set_data_3d([sx, sx], [sy, sy], [0, 1.7])
                    dot.set_data_3d([sx], [sy], [1.7])
                    line.set_alpha(min(normv, 1.0))
                    dot.set_alpha(min(normv, 1.0))
                    continue
            # No peak (or below threshold) for this pin slot.
            pin_state[i] = None
            line.set_alpha(0.0)
            dot.set_alpha(0.0)

        notes = []
        if status.get("fatal_error"):
            notes.append(f"READER DIED: {status['fatal_error']}")
        else:
            now = time.monotonic()
            since = now - status.get("last_packet_ts", 0.0)
            if status.get("last_packet_ts", 0.0) == 0.0:
                notes.append("waiting for first packet…")
            elif since > 2.0:
                notes.append(f"no packets for {since:.1f}s")
        if status.get("bad_samples", 0):
            notes.append(f"{status['bad_samples']} bad samples")
        if unknown_rx:
            notes.append(f"unknown RX: {', '.join(sorted(unknown_rx))}")
        if unknown_tx:
            notes.append(f"unknown TX: {', '.join(sorted(unknown_tx))}")
        if notes:
            ax.set_title("  |  ".join(notes), fontsize=8, color="tab:red")
        else:
            ax.set_title("")
        return [surf, *pin_lines, *pin_dots]

    # `anim` is intentionally bound for the duration of plt.show(); without
    # a live reference, matplotlib garbage-collects FuncAnimation and the
    # animation freezes silently.
    anim = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        stop.set()
    return 0
