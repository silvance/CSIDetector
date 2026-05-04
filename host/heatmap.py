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
import time
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

# How many recent metric samples each sparkline shows. At the 100ms
# frame interval below, 200 samples ≈ 20s of history.
SPARKLINE_HISTORY_LEN = 200
# Window over which per-link packet rate is averaged.
PKT_RATE_WINDOW_S = 3.0
# Seconds the badge stays in "flash" styling after a state transition.
BADGE_FLASH_DURATION_S = 0.6
# Per-TX line styles so the two fans are distinguishable when they
# physically overlap. Cycles for >4 TXs.
TX_LINESTYLES = ["-", "--", "-.", ":"]


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
        # Wallclock of the most recent successfully-pushed sample, used
        # by the viewer to flag dead links in the title bar.
        self.last_push_ts: float = 0.0
        # Recent push timestamps, used to compute per-link pkt/s for the
        # health strip. Cap is generous (covers >50 Hz × PKT_RATE_WINDOW_S).
        self._push_ts: collections.deque[float] = collections.deque(maxlen=512)

    def push(self, sample: csi_collector.CSISample) -> None:
        amp = sample.amplitude
        with self._lock:
            now = time.monotonic()
            self.last_push_ts = now
            self._push_ts.append(now)
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

    def packet_rate(self, window_s: float = PKT_RATE_WINDOW_S) -> float:
        cutoff = time.monotonic() - window_s
        with self._lock:
            ts = list(self._push_ts)
        n = sum(1 for t in ts if t >= cutoff)
        return n / window_s


def _reader_thread(source: str,
                   buffers: dict[tuple[str, str], _LinkBuffer],
                   tx_macs: set[str],
                   unknown_rx: set[str],
                   unknown_tx: set[str],
                   stop: threading.Event,
                   status: dict) -> None:
    # Whole loop is wrapped in try/except so a malformed packet, a
    # transient socket error, or a numpy edge case can't silently kill
    # the thread (and freeze the viewer). The exception is recorded
    # in `status` so the viewer can surface it in the title bar.
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
                # One bad sample shouldn't kill the stream. Track the
                # count so a steady stream of garbage is visible.
                status["bad_samples"] = status.get("bad_samples", 0) + 1
                status["last_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        # Source itself died (socket closed, file ended, etc).
        status["fatal_error"] = f"{type(exc).__name__}: {exc}"


def _load_baselines(path: Optional[str], txs, rxs) -> dict[tuple[str, str], float]:
    """Read baselines.json and return per-(tx_mac, rx_mac) values.

    Three formats accepted (in order of preference):
      - Wrapped: {"_meta": {...}, "links": {"tx|rx": float, ...}}.
        Metadata enables stale-file warnings.
      - Flat link-keyed: {"tx_mac|rx_mac": float, ...}.
      - Legacy per-RX: {"rx_mac": float, ...}. Replicated across every
        TX from that RX so old files still work; logged so users know
        to recalibrate when accuracy matters.
    """
    if not path:
        return {}
    import os
    with open(path) as f:
        raw = json.load(f)
    meta = raw.get("_meta") if isinstance(raw, dict) else None
    if meta is not None and "links" in raw:
        link_map = raw["links"]
        # Stale-file warning: file mtime > 1 hour suggests environment
        # has likely drifted enough that the baselines aren't valid.
        try:
            age_s = time.time() - os.path.getmtime(path)
            if age_s > 3600:
                age_h = age_s / 3600.0
                print(f"heatmap: WARNING — baselines.json is {age_h:.1f}h old; "
                      f"consider re-running `calibrate-links` (RF environments "
                      f"drift on this timescale).")
        except OSError:
            pass
    else:
        link_map = raw
    out: dict[tuple[str, str], float] = {}
    legacy_rx_macs: set[str] = set()
    tx_macs = [t.mac for t in txs]
    for k, v in link_map.items():
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
                full_bright: float = DEFAULT_RATIO_FULL_BRIGHT,
                motion_enter: float = 2.0,
                motion_exit: float = 1.5) -> int:
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
    status: dict = {"pkt_count": 0, "bad_samples": 0, "last_packet_ts": 0.0}
    stop = threading.Event()
    threading.Thread(target=_reader_thread,
                     args=(source, buffers, tx_macs,
                           unknown_rx, unknown_tx, stop, status),
                     daemon=True).start()

    # Dark theme palette (used by main, sparklines, and pkt-rate strip).
    BG = "#0a0a0a"
    PANEL = "#181818"
    GRID = "#333333"
    AXIS_TXT = "#bbbbbb"
    pair_keys: list[tuple[str, str]] = [(t.mac, r.mac) for t in txs for r in rxs]
    link_label = {(t.mac, r.mac): f"{t.label}↔{r.label}" for t in txs for r in rxs}
    n_links = len(pair_keys)

    fig = plt.figure(figsize=(13, 8.5), facecolor=BG)
    # Source string lives in the bottom-right corner so the badge owns
    # the top of the figure.
    fig.text(0.99, 0.01, f"source: {source}", color="#666",
             fontsize=8, ha="right", va="bottom", family="monospace")
    # Outer layout: main heatmap upper-left, sparkline column right,
    # packet-rate strip across the bottom.
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    outer = GridSpec(2, 2, figure=fig,
                     width_ratios=[2.4, 1.0], height_ratios=[10, 1.4],
                     left=0.06, right=0.97, top=0.88, bottom=0.10,
                     wspace=0.18, hspace=0.22)
    ax = fig.add_subplot(outer[0, 0])
    ax.set_facecolor(PANEL)
    ax.set_xlim(bbox_min[0] - 0.3, bbox_max[0] + 0.3)
    ax.set_ylim(bbox_min[1] - 0.3, bbox_max[1] + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color=AXIS_TXT, fontsize=10)
    ax.set_ylabel("y (m)", color=AXIS_TXT, fontsize=10)
    ax.tick_params(colors=AXIS_TXT, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.add_patch(plt.Polygon(polygon, fill=False,
                             edgecolor=AXIS_TXT, linewidth=1.5))

    cmap = mpl.colormaps["magma"]
    # One line + value-label per (TX, RX) pair. Per-TX linestyle so two
    # fans crossing the same point are still distinguishable.
    line_artists: list = []
    label_artists: list = []
    for tx_idx, tx in enumerate(txs):
        style = TX_LINESTYLES[tx_idx % len(TX_LINESTYLES)]
        for rx in rxs:
            kwargs = dict(color=cmap(0.0), linewidth=3.5,
                          alpha=0.92, linestyle=style)
            if style == "-":
                kwargs["solid_capstyle"] = "round"
            else:
                kwargs["dash_capstyle"] = "round"
            line, = ax.plot([tx.x, rx.x], [tx.y, rx.y], **kwargs)
            line_artists.append(line)
            # Tiny value label at the midpoint, so multiple lines through
            # an RX don't pile their text on top of each other.
            mx, my = (tx.x + rx.x) / 2.0, (tx.y + rx.y) / 2.0
            label_artists.append(ax.text(mx, my, "", fontsize=7,
                                         color="white",
                                         ha="center", va="center",
                                         bbox=dict(facecolor="#000000", alpha=0.55,
                                                   edgecolor="none", pad=1.5)))

    # RX dots + labels (drawn after lines so they sit on top).
    for rx in rxs:
        ax.plot(rx.x, rx.y, "o", color="#4ea3ff", markersize=13, zorder=5,
                markeredgecolor=BG, markeredgewidth=1.0)
        ax.text(rx.x, rx.y + 0.20, rx.label, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#4ea3ff", zorder=6)
    # TX stars; each TX gets a slightly different shade so the two fans
    # are visually distinguishable. With one TX, this still draws a star.
    tx_shades = ["#ffb347", "#ff6363", "#ffa64d", "#e25c5c"]
    for i, tx in enumerate(txs):
        c = tx_shades[i % len(tx_shades)]
        ax.plot(tx.x, tx.y, "*", color=c, markersize=22, zorder=5,
                markeredgecolor=BG, markeredgewidth=1.0)
        ax.text(tx.x, tx.y + 0.20, tx.label, ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=c, zorder=6)

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
    cbar = fig.colorbar(sm, ax=ax, label=cbar_label, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(colors=AXIS_TXT, labelsize=8)
    cbar.ax.yaxis.label.set_color(AXIS_TXT)
    cbar.outline.set_edgecolor(GRID)

    # Per-link sparkline column. Each link gets a small inset axes
    # showing the last SPARKLINE_HISTORY_LEN metric samples. Threshold
    # lines (motion_enter / motion_exit) are drawn in-axes when ratio
    # mode is active so you can read the state at a glance.
    spark_grid = GridSpecFromSubplotSpec(n_links, 1, subplot_spec=outer[0, 1],
                                         hspace=0.30)
    spark_axes: list = []
    spark_lines: list = []
    spark_value_texts: list = []
    metric_history: dict[tuple[str, str], collections.deque] = {
        k: collections.deque(maxlen=SPARKLINE_HISTORY_LEN) for k in pair_keys
    }
    spark_y_top = full_bright * 1.1 if use_ratio else 1.0
    spark_y_bot = 0.0 if use_ratio else 0.0
    for i, k in enumerate(pair_keys):
        sax = fig.add_subplot(spark_grid[i, 0])
        sax.set_facecolor(PANEL)
        sax.set_xlim(0, SPARKLINE_HISTORY_LEN - 1)
        sax.set_ylim(spark_y_bot, spark_y_top)
        sax.set_xticks([])
        sax.set_yticks([])
        for s in sax.spines.values():
            s.set_color(GRID)
        if use_ratio:
            sax.axhline(motion_enter, color="#ff6363",
                        linestyle=":", linewidth=0.7, alpha=0.7)
            sax.axhline(motion_exit, color="#5fcf6f",
                        linestyle=":", linewidth=0.7, alpha=0.7)
            sax.axhline(RATIO_FLOOR, color=GRID,
                        linestyle="-", linewidth=0.5, alpha=0.7)
        line, = sax.plot([], [], color="#dddddd", linewidth=1.2)
        spark_axes.append(sax)
        spark_lines.append(line)
        # Labels live inside the axes to avoid overflowing into the main
        # plot or clipping the right edge: link name pinned upper-left,
        # current value upper-right (recolored each frame by tint).
        sax.text(0.02, 0.92, link_label[k], transform=sax.transAxes,
                 ha="left", va="top", fontsize=8, color=AXIS_TXT,
                 family="monospace")
        val = sax.text(0.98, 0.92, "—", transform=sax.transAxes,
                       ha="right", va="top", fontsize=9, color="#888",
                       family="monospace", fontweight="bold")
        spark_value_texts.append(val)

    # Packet-rate / health strip across the bottom: one bar per link
    # showing pkt/s averaged over PKT_RATE_WINDOW_S. Bars turn red when
    # rate drops to zero (link presumed dead).
    rate_ax = fig.add_subplot(outer[1, :])
    rate_ax.set_facecolor(PANEL)
    bar_x = np.arange(n_links)
    rate_bars = rate_ax.bar(bar_x, np.zeros(n_links),
                            color="#3a8fb7", width=0.78,
                            edgecolor=BG, linewidth=0.5)
    rate_ax.set_xlim(-0.6, n_links - 0.4)
    rate_ax.set_ylim(0, 10)  # autoscaled in update()
    rate_ax.set_xticks(bar_x)
    rate_ax.set_xticklabels([link_label[k] for k in pair_keys],
                            rotation=20, ha="right", fontsize=8,
                            color=AXIS_TXT)
    rate_ax.tick_params(axis="y", colors=AXIS_TXT, labelsize=8)
    rate_ax.tick_params(axis="x", colors=AXIS_TXT)
    rate_ax.set_ylabel("pkt/s", fontsize=9, color=AXIS_TXT)
    rate_ax.grid(axis="y", color=GRID, linewidth=0.4, alpha=0.6)
    rate_ax.set_axisbelow(True)
    for s in rate_ax.spines.values():
        s.set_color(GRID)
    rate_ymax = [10.0]  # autoscaling cap, mutable so closure can update

    # Big presence/motion badge across the top of the figure. Driven
    # by the median per-link motion ratio with hysteresis so the
    # state doesn't flicker on noisy frames.
    badge_text = fig.text(0.5, 0.945, "INITIALIZING", ha="center", va="center",
                          fontsize=24, fontweight="bold",
                          color="white",
                          bbox=dict(facecolor="#666666", edgecolor="none",
                                    boxstyle="round,pad=0.7"))
    presence_state = ["INIT"]
    state_change_ts = [0.0]
    BADGE_STYLE = {
        "INIT":   ("INITIALIZING",    "#666666"),
        "EMPTY":  ("EMPTY",           "#2faa55"),
        "MOTION": ("MOTION DETECTED", "#d94545"),
    }

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
        for k, line, lbl, tint, m, s, ok in zip(
                pair_keys, line_artists, label_artists,
                tints, metrics, sigmas, has_baseline):
            color = cmap(tint)
            line.set_color(color)
            lbl.set_text(text_fmt(m, s, ok))
            # Append to history; the sparkline reads it below.
            metric_history[k].append(m if ok else float("nan"))

        # Update sparklines + their right-side value badge.
        for k, sline, vtxt, tint, m, ok in zip(
                pair_keys, spark_lines, spark_value_texts,
                tints, metrics, has_baseline):
            hist = metric_history[k]
            if hist:
                arr = np.array(hist, dtype=float)
                xs = np.arange(len(arr)) + (SPARKLINE_HISTORY_LEN - len(arr))
                sline.set_data(xs, arr)
                sline.set_color(cmap(0.25 + 0.75 * tint))
            vtxt.set_text(text_fmt(m, 0.0, ok))
            vtxt.set_color(cmap(0.25 + 0.75 * tint) if ok else "#666")

        # Update packet-rate strip. Bars turn dim-red when the link is
        # silent so dead links pop visually even before the title-bar
        # warning kicks in.
        rates = [buffers[k].packet_rate() for k in pair_keys]
        max_rate = max(rates) if rates else 0.0
        if max_rate * 1.15 > rate_ymax[0]:
            rate_ymax[0] = max_rate * 1.3 + 1.0
            rate_ax.set_ylim(0, rate_ymax[0])
        for bar, r in zip(rate_bars, rates):
            bar.set_height(r)
            bar.set_color("#d94545" if r < 0.5 else "#3a8fb7")

        # Update presence/motion badge using hysteresis on the median
        # link metric. Median (not max) so a single noisy link can't
        # drive the demo state; (not mean) so a few zeroed-out
        # missing-baseline links don't drag the signal down.
        if use_ratio:
            valid = [m for m, ok in zip(metrics, has_baseline) if ok]
        else:
            valid = list(metrics)
        if valid:
            valid.sort()
            mid = valid[len(valid) // 2]
            prev = presence_state[0]
            cur = prev
            if cur == "INIT" and status.get("pkt_count", 0) > motion_window:
                cur = "EMPTY" if mid < motion_enter else "MOTION"
            elif cur == "EMPTY" and mid >= motion_enter:
                cur = "MOTION"
            elif cur == "MOTION" and mid <= motion_exit:
                cur = "EMPTY"
            if cur != prev:
                state_change_ts[0] = time.monotonic()
            presence_state[0] = cur
            label, color = BADGE_STYLE[cur]
            if cur != "INIT":
                label = f"{label}   (median {mid:.2f}×)"
            badge_text.set_text(label)
            patch = badge_text.get_bbox_patch()
            patch.set_facecolor(color)
            # Brief flash on transition: white border + extra padding.
            since_change = time.monotonic() - state_change_ts[0]
            if since_change < BADGE_FLASH_DURATION_S:
                patch.set_edgecolor("white")
                patch.set_linewidth(3.0)
            else:
                patch.set_edgecolor("none")
                patch.set_linewidth(0.0)
        notes = []
        # Reader-thread health on the title so a dead stream is obvious.
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
        # Per-link staleness: any link whose last push is > 3s old is
        # almost certainly dead. Group by TX so a fully-dead TX shows
        # up as a single cluster instead of N entries scattered through
        # the title bar.
        now = time.monotonic()
        rx_by_mac = {r.mac: r.label for r in rxs}
        tx_by_mac = {t.mac: t.label for t in txs}
        tx_order = [t.label for t in txs]
        dead_by_tx: dict[str, list[str]] = {}
        for k, buf in buffers.items():
            ts = buf.last_push_ts
            if ts == 0.0 or (now - ts) > 3.0:
                tx_lbl = tx_by_mac.get(k[0], "?")
                rx_lbl = rx_by_mac.get(k[1], "?")
                dead_by_tx.setdefault(tx_lbl, []).append(rx_lbl)
        if dead_by_tx and status.get("pkt_count", 0) > 0:
            # Only flag dead links once at least some packets have arrived
            # — pre-startup, every link looks dead.
            parts = []
            for tx_lbl in tx_order:
                if tx_lbl not in dead_by_tx:
                    continue
                rxs_dead = sorted(dead_by_tx[tx_lbl])
                parts.append(f"{tx_lbl}→{{{','.join(rxs_dead)}}}")
            notes.append(f"dead: {' | '.join(parts)}")
        if unknown_rx:
            notes.append(f"unknown RX: {', '.join(sorted(unknown_rx))}")
        if unknown_tx:
            notes.append(f"unknown TX: {', '.join(sorted(unknown_tx))}")
        if notes:
            ax.set_title("  |  ".join(notes), fontsize=8, color="tab:red")
        else:
            ax.set_title("")
        return [*line_artists, *label_artists, badge_text]

    anim = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    # `anim` is intentionally bound for the duration of plt.show(); without
    # a live reference, matplotlib garbage-collects FuncAnimation and the
    # animation freezes silently.
    try:
        plt.show()
    finally:
        stop.set()
    return 0
