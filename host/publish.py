"""Host-side presence/motion publisher.

Listens to the same CSI UDP feed as `heatmap` / `view3d`, runs the
median-ratio + hysteresis logic from heatmap.py, and broadcasts a small
state packet to a target IP:port (e.g. an ESP32-C5 with a screen
running on the same hotspot).

Decoupled from the matplotlib viewers so the C5 can show state whether
or not anyone has the host viewer open. Runs as a long-lived service.

Wire format of outgoing state packet (little-endian, packed; 24 bytes):
    uint8   version          (1)
    uint8   state             (0=INIT, 1=EMPTY, 2=MOTION)
    uint8   reserved[2]
    float32 median_ratio      (current median per-link motion ratio)
    float32 max_ratio         (current max per-link motion ratio)
    uint32  state_changes     (cumulative count since process start)
    uint64  ts_ms             (monotonic milliseconds since process start)

C5 firmware just decodes that and updates the screen.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from typing import Optional

import numpy as np

import csi_collector
import heatmap as _hm  # reuse _LinkBuffer + _load_links + _load_baselines


_STATE_PKT = struct.Struct("<BBHffIQ")
assert _STATE_PKT.size == 24

STATE_INIT = 0
STATE_EMPTY = 1
STATE_MOTION = 2
STATE_NAMES = {STATE_INIT: "INIT", STATE_EMPTY: "EMPTY", STATE_MOTION: "MOTION"}


def run_publisher(source: str, links_path: str,
                  c5_addr: str, c5_port: int,
                  baselines_path: Optional[str] = None,
                  history: int = 500, motion_window: int = 50,
                  motion_enter: float = 2.0, motion_exit: float = 1.5,
                  publish_hz: float = 5.0,
                  verbose: bool = False) -> int:
    """Listen for CSI samples, compute presence state, broadcast UDP to C5."""
    cfg, txs, rxs = _hm._load_links(links_path)
    tx_macs = {t.mac for t in txs}
    pair_keys = [(t.mac, r.mac) for t in txs for r in rxs]
    buffers = {k: _hm._LinkBuffer(history) for k in pair_keys}
    baselines = _hm._load_baselines(baselines_path, txs, rxs)
    use_ratio = bool(baselines)

    unknown_rx: set[str] = set()
    unknown_tx: set[str] = set()
    status: dict = {"pkt_count": 0, "bad_samples": 0, "last_packet_ts": 0.0}
    stop = threading.Event()
    threading.Thread(target=_hm._reader_thread,
                     args=(source, buffers, tx_macs,
                           unknown_rx, unknown_tx, stop, status),
                     daemon=True).start()

    out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Allow broadcast in case c5_addr is e.g. 10.42.0.255.
    out_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    target = (c5_addr, c5_port)

    state = STATE_INIT
    state_changes = 0
    started = time.monotonic()
    period = 1.0 / publish_hz
    next_emit = time.monotonic()

    print(f"presence-publisher: listening on {source}, publishing to "
          f"{c5_addr}:{c5_port} at {publish_hz:g} Hz "
          f"(enter={motion_enter}, exit={motion_exit})", flush=True)

    try:
        while True:
            now = time.monotonic()
            if now < next_emit:
                time.sleep(min(0.01, next_emit - now))
                continue
            next_emit = now + period

            # Compute per-link motion ratios.
            sigmas = [buffers[k].motion_score(motion_window) for k in pair_keys]
            metrics = []
            for (tx_mac, rx_mac), sigma in zip(pair_keys, sigmas):
                if use_ratio:
                    base = baselines.get((tx_mac, rx_mac))
                    if base is None or base <= 0:
                        continue   # no baseline → exclude from aggregation
                    metrics.append(sigma / base)
                else:
                    metrics.append(sigma)

            if metrics:
                metrics.sort()
                median = metrics[len(metrics) // 2]
                mx = metrics[-1]
            else:
                median = 0.0
                mx = 0.0

            # Hysteresis state machine. Stays in INIT until enough packets
            # have arrived for the motion-σ window to be meaningful.
            new_state = state
            if state == STATE_INIT and status.get("pkt_count", 0) > motion_window:
                new_state = STATE_EMPTY if median < motion_enter else STATE_MOTION
            elif state == STATE_EMPTY and median >= motion_enter:
                new_state = STATE_MOTION
            elif state == STATE_MOTION and median <= motion_exit:
                new_state = STATE_EMPTY
            if new_state != state:
                state_changes += 1
                if verbose or True:   # always log transitions; they're rare
                    print(f"{csi_collector.now_iso()} {STATE_NAMES[state]} -> "
                          f"{STATE_NAMES[new_state]} (median {median:.2f}×)",
                          flush=True)
                state = new_state

            ts_ms = int((now - started) * 1000.0)
            pkt = _STATE_PKT.pack(1, state, 0, float(median), float(mx),
                                  state_changes, ts_ms)
            try:
                out_sock.sendto(pkt, target)
            except OSError as exc:
                if verbose:
                    print(f"sendto failed: {exc}", flush=True)
    except KeyboardInterrupt:
        print("\npublisher stopped.")
    finally:
        stop.set()
        out_sock.close()
    return 0
