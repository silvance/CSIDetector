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
    """Read baselines.json. Accepts either keys "tx_mac|rx_mac" (new,
    per-link) or "rx_mac" (legacy, per-RX, fanned out to every TX).
    """
    if not path:
        return {}
    with open(path) as f:
        raw = json.load(f)
    out: dict[tuple[str, str], float] = {}
    legacy = 0
    tx_macs = [t.mac for t in txs]
    for k, v in raw.items():
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
                 wall_height_m: float = 2.5) -> int:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.cm import get_cmap
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    polygon, txs, rxs = _load_links(links_path)
    tx_macs = {t.mac for t in txs}
    tx_pos = {t.mac: np.array([t.x, t.y]) for t in txs}
    rx_pos = {r.mac: np.array([r.x, r.y]) for r in rxs}

    loc = localize.Localizer(polygon, tx_pos, rx_pos,
                             grid_step=grid_step, link_sigma_m=link_sigma_m)

    # baselines.json supports two key formats; see _load_baselines below.
    baselines = _load_baselines(baselines_path, txs, rxs)
    use_ratio = bool(baselines)

    buffers: dict[tuple[str, str], _LinkBuffer] = {
        (t.mac, r.mac): _LinkBuffer(history) for t in txs for r in rxs
    }
    unknown_rx: set[str] = set()
    unknown_tx: set[str] = set()
    stop = threading.Event()
    threading.Thread(target=_reader_thread,
                     args=(source, buffers, tx_macs,
                           unknown_rx, unknown_tx, stop),
                     daemon=True).start()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle(f"CSI 2.5D — {source}")
    cmap = get_cmap("magma")

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

    # Person pin: vertical line from floor to ~head height at the argmax.
    person_line, = ax.plot([0, 0], [0, 0], [0, 1.7], color="tab:red",
                            linewidth=3, alpha=0.0)
    person_dot, = ax.plot([0], [0], [1.7], "o", color="tab:red", markersize=10,
                           alpha=0.0)

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
                base = baselines.get((tx_mac, rx_mac), 1e-3)
                # Subtract baseline so still-room links contribute ~0.
                metric = max(sigma / max(base, 1e-6) - 1.0, 0.0)
            else:
                metric = sigma
            scores[(tx_mac, rx_mac)] = metric

        grid = loc.update(scores)
        running_max[0] = max(running_max[0] * 0.99, float(grid.max()), 1e-3)
        norm = grid / running_max[0]
        # `plot_surface` doesn't update facecolors cleanly; redrawing every
        # frame is the documented workaround. Cheap at this grid size.
        surf.set_facecolors(cmap(np.clip(norm, 0, 1)).reshape(-1, 4))

        argx, argy, argv = loc.argmax_xy(grid)
        normv = argv / running_max[0]
        if normv > PIN_THRESHOLD:
            person_line.set_data_3d([argx, argx], [argy, argy], [0, 1.7])
            person_dot.set_data_3d([argx], [argy], [1.7])
            person_line.set_alpha(min(normv, 1.0))
            person_dot.set_alpha(min(normv, 1.0))
        else:
            person_line.set_alpha(0.0)
            person_dot.set_alpha(0.0)

        notes = []
        if unknown_rx:
            notes.append(f"unknown RX: {', '.join(sorted(unknown_rx))}")
        if unknown_tx:
            notes.append(f"unknown TX: {', '.join(sorted(unknown_tx))}")
        if notes:
            ax.set_title("  |  ".join(notes), fontsize=8, color="tab:red")
        return [surf, person_line, person_dot]

    anim = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        stop.set()
    del anim
    return 0
