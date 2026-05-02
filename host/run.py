"""Command-line entry point for the CSI motion detector.

`<source>` accepts a serial port (`/dev/ttyUSB0`, `COM5`), a saved log
file, `udp:<port>` for the multi-RX hotspot setup, or `-` for stdin.
Subcommands that key off rx_id (heatmap, view3d, calibrate-links) only
make sense with a `udp:<port>` source.

Subcommands:

    capture <source> <out.log> [--seconds N] [--baud B]
        Dump raw CSI_DATA lines to disk. Use to record a still-room
        baseline or a labeled session for offline analysis.

    calibrate <source> [--seconds N] [--settle S] [--window W] [--max-samples M]
        Single-stream still-room baseline (one RX). Drops the first
        --settle seconds to let the radio's AGC lock. For multi-RX
        UDP setups use `calibrate-links` instead.

    detect <source> --baseline B [--window W] [--enter R] [--exit R]
        Live binary detection on a single stream. Prints an event line
        on every transition between STILL and MOTION.

    view <source> [--history N] [--window W]
        Single-stream waterfall: subcarrier × time + motion-σ trace.
        With a UDP source containing multiple RXs, pins to the first
        rx_id seen and drops the rest.

    heatmap <source> --links links.json [--baselines b.json] [--full-bright R]
        Multi-RX, multi-TX 2D floor-plan view. Each TX-RX line is
        tinted by its motion-σ ratio against the per-RX baseline.
        Source must carry rx_id (i.e. `udp:<port>`). --full-bright
        sets the ratio that saturates the colormap (default 3.0×).

    view3d <source> --links links.json [--baselines b.json] [--grid-step G]
        2.5D room view: walls extruded, floor as a likelihood heatmap
        derived from per-link motion-σ, person pin at the brightest
        cell.

    calibrate-links <source> [--out baselines.json] [--settle S] [--seconds N]
        Multi-RX still-room calibration. Writes a {rx_mac: baseline}
        JSON. --settle doubles as a walk-out timer; the script counts
        down before recording starts.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import csi_collector
import detector


def _raw_lines(source: str, baud: int):
    if source == "-":
        for line in sys.stdin:
            yield line
        return
    if source.startswith("/dev/") or source.lower().startswith("com"):
        import serial
        with serial.Serial(source, baud, timeout=1) as ser:
            while True:
                raw = ser.readline()
                if not raw:
                    continue
                yield raw.decode("ascii", errors="replace")
        return
    with open(source, "r") as f:
        for line in f:
            yield line


def cmd_capture(args: argparse.Namespace) -> int:
    deadline = time.time() + args.seconds if args.seconds else None
    n = 0
    with open(args.out, "w") as f:
        # Preserve the header line so downstream tools that key off it work.
        for line in _raw_lines(args.source, args.baud):
            stripped = line.strip()
            if not (stripped.startswith("CSI_DATA,") or stripped.startswith("type,")):
                continue
            f.write(line if line.endswith("\n") else line + "\n")
            if stripped.startswith("CSI_DATA,"):
                n += 1
            if deadline and time.time() >= deadline:
                break
    print(f"captured {n} samples to {args.out}", file=sys.stderr)
    return 0


def _collect_amplitudes(source: str, seconds: float | None,
                        max_samples: int | None,
                        settle_seconds: float = 0.0
                        ) -> tuple[np.ndarray, np.ndarray]:
    src = csi_collector.open_source(source)
    rows: list[np.ndarray] = []
    idx = None
    start = time.time()
    settle_until = start + settle_seconds
    deadline = (settle_until + seconds) if seconds else None
    skipped = 0
    for sample in src:
        if time.time() < settle_until:
            skipped += 1
            continue
        if idx is None:
            idx = np.flatnonzero(sample.amplitude > 0)
            if idx.size == 0:
                continue
        rows.append(sample.amplitude)
        if max_samples and len(rows) >= max_samples:
            break
        if deadline and time.time() >= deadline:
            break
    if not rows or idx is None:
        raise SystemExit("no CSI samples received")
    if settle_seconds > 0:
        print(f"dropped {skipped} samples during AGC settle ({settle_seconds:.1f}s)",
              file=sys.stderr)
    amps = np.stack(rows)[:, idx]
    return amps, idx


def cmd_calibrate(args: argparse.Namespace) -> int:
    amps, idx = _collect_amplitudes(
        args.source, args.seconds, args.max_samples,
        settle_seconds=args.settle,
    )
    baseline = detector.compute_baseline(amps, args.window)
    print(f"baseline={baseline:.6f}  samples={amps.shape[0]}  subcarriers={idx.size}",
          file=sys.stderr)
    print(f"{baseline:.6f}")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    cfg = detector.DetectorConfig(
        window=args.window,
        enter_ratio=args.enter,
        exit_ratio=args.exit,
    )
    src = csi_collector.open_source(args.source)
    det: detector.MotionDetector | None = None
    last_state = False
    settle_until = time.time() + args.settle
    for sample in src:
        if time.time() < settle_until:
            continue
        if det is None:
            idx = np.flatnonzero(sample.amplitude > 0)
            if idx.size == 0:
                continue
            det = detector.MotionDetector(idx, args.baseline, cfg)
        score, motion = det.update(sample.amplitude)
        if motion != last_state:
            event = "MOTION" if motion else "STILL"
            print(f"{csi_collector.now_iso()} {event} score={score:.4f} "
                  f"baseline={det.baseline:.4f} ratio={score/det.baseline:.2f}",
                  flush=True)
            last_state = motion
        elif args.verbose:
            print(f"{csi_collector.now_iso()} score={score:.4f} "
                  f"ratio={score/det.baseline:.2f}",
                  flush=True)
    return 0


def cmd_view(args: argparse.Namespace) -> int:
    import viewer
    return viewer.run_viewer(args.source, history=args.history, motion_window=args.window)


def cmd_heatmap(args: argparse.Namespace) -> int:
    import heatmap
    return heatmap.run_heatmap(args.source, args.links,
                               history=args.history, motion_window=args.window,
                               baselines_path=args.baselines,
                               full_bright=args.full_bright)


def cmd_view3d(args: argparse.Namespace) -> int:
    import viewer3d
    return viewer3d.run_viewer3d(
        args.source, args.links,
        history=args.history, motion_window=args.window,
        baselines_path=args.baselines,
        grid_step=args.grid_step, link_sigma_m=args.link_sigma,
        wall_height_m=args.wall_height,
    )


def cmd_calibrate_links(args: argparse.Namespace) -> int:
    """Multi-RX still-room calibration. Writes {rx_mac: baseline} as JSON."""
    import json
    import queue
    import threading

    samples_q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def reader():
        try:
            for sample in csi_collector.open_source(args.source):
                if stop.is_set():
                    return
                samples_q.put(sample)
        except Exception as exc:
            samples_q.put(exc)

    threading.Thread(target=reader, daemon=True).start()

    per_rx: dict[str, list[np.ndarray]] = {}
    start = time.time()
    settle_until = start + args.settle
    deadline = settle_until + args.seconds

    print(f"\n>>> LEAVE THE ROOM NOW <<<  starting capture in {args.settle:.0f}s "
          f"(then recording for {args.seconds:.0f}s)\n", file=sys.stderr, flush=True)

    # Settle phase: tick every second, drop any samples that arrive.
    received_during_settle = 0
    next_tick = start + 1.0
    while time.time() < settle_until:
        try:
            item = samples_q.get(timeout=0.2)
            if isinstance(item, Exception):
                raise item
            received_during_settle += 1
        except queue.Empty:
            pass
        now = time.time()
        if now >= next_tick:
            remaining = max(0, int(settle_until - now + 0.5))
            note = "" if received_during_settle else "  [WARNING: no packets yet]"
            print(f"  ...{remaining}s until recording starts{note}",
                  file=sys.stderr, flush=True)
            next_tick = now + 1.0

    if received_during_settle == 0:
        stop.set()
        raise SystemExit(
            "no packets received during settle — receivers are not streaming. "
            "check `sudo tcpdump -ni <hotspot_iface> udp port 5566` and the "
            "firewall zone for the hotspot interface.")

    print(f"\n>>> RECORDING <<<  hold still for {args.seconds:.0f}s\n",
          file=sys.stderr, flush=True)

    next_tick = time.time() + 5.0
    while time.time() < deadline:
        try:
            item = samples_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if isinstance(item, Exception):
            raise item
        if item.rx_id is not None:
            per_rx.setdefault(item.rx_id, []).append(item.amplitude)
        if time.time() >= next_tick:
            counts = ", ".join(f"{k[-5:]}={len(v)}" for k, v in sorted(per_rx.items()))
            print(f"  +{int(time.time() - settle_until)}s  {counts}",
                  file=sys.stderr, flush=True)
            next_tick = time.time() + 5.0

    stop.set()
    baselines = detector.compute_link_baselines(per_rx, window=args.window)
    if not baselines:
        raise SystemExit("no usable baselines — did any RX deliver enough samples?")
    # Flag RXs that streamed but didn't hit the threshold; without this,
    # a flaky RX silently vanishes from baselines.json and its links
    # later render at 0× in the heatmap with no obvious cause.
    min_required = 2 * args.window
    short = [(mac, len(rows)) for mac, rows in per_rx.items()
             if mac not in baselines]
    print(f"\nper-RX:", file=sys.stderr)
    for mac, b in sorted(baselines.items()):
        print(f"  {mac}  baseline={b:.6f}  ({len(per_rx[mac])} samples)",
              file=sys.stderr)
    for mac, n in sorted(short):
        print(f"  {mac}  SKIPPED — only {n} samples, need >= {min_required}; "
              f"this RX's links will render at 0× in the heatmap",
              file=sys.stderr)
    with open(args.out, "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="csi-detector")
    sub = p.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="dump raw CSI lines to a log file")
    cap.add_argument("source", help="serial port, log path, or '-' for stdin")
    cap.add_argument("out", help="output log path")
    cap.add_argument("--seconds", type=float, default=None)
    cap.add_argument("--baud", type=int, default=921600)
    cap.set_defaults(func=cmd_capture)

    cal = sub.add_parser("calibrate", help="estimate still-room baseline score")
    cal.add_argument("source")
    cal.add_argument("--seconds", type=float, default=15.0)
    cal.add_argument("--settle", type=float, default=detector.AGC_SETTLE_SECONDS_DEFAULT,
                     help="seconds to drop after start while AGC locks (default 10)")
    cal.add_argument("--max-samples", type=int, default=None)
    cal.add_argument("--window", type=int, default=50)
    cal.set_defaults(func=cmd_calibrate)

    det_p = sub.add_parser("detect", help="live motion detection")
    det_p.add_argument("source")
    det_p.add_argument("--baseline", type=float, required=True)
    det_p.add_argument("--settle", type=float, default=detector.AGC_SETTLE_SECONDS_DEFAULT)
    det_p.add_argument("--window", type=int, default=50)
    det_p.add_argument("--enter", type=float, default=3.0)
    det_p.add_argument("--exit", type=float, default=1.5)
    det_p.add_argument("--verbose", action="store_true")
    det_p.set_defaults(func=cmd_detect)

    view_p = sub.add_parser("view", help="live matplotlib CSI heatmap")
    view_p.add_argument("source")
    view_p.add_argument("--history", type=int, default=500,
                        help="samples shown horizontally (~5 s at 100 Hz)")
    view_p.add_argument("--window", type=int, default=50,
                        help="motion-score sliding window (samples)")
    view_p.set_defaults(func=cmd_view)

    hm = sub.add_parser("heatmap", help="multi-RX floor-plan motion overlay (UDP)")
    hm.add_argument("source", help="udp:<port> typically")
    hm.add_argument("--links", required=True,
                    help="JSON config with room, TXs, and RX positions (see links.example.json)")
    hm.add_argument("--history", type=int, default=500)
    hm.add_argument("--window", type=int, default=50)
    hm.add_argument("--baselines", default=None,
                    help="JSON map of RX MAC -> still-room σ. When given, "
                         "links are colored by ratio (× baseline) instead of "
                         "auto-scaled raw σ.")
    hm.add_argument("--full-bright", type=float, default=3.0,
                    help="ratio at which links saturate to the brightest "
                         "cmap value (default 3.0; only used with --baselines). "
                         "Lower this if motion looks washed-out as 'all dark'.")
    hm.set_defaults(func=cmd_heatmap)

    v3 = sub.add_parser("view3d", help="2.5D room view: floor heatmap + person pin")
    v3.add_argument("source", help="udp:<port> typically")
    v3.add_argument("--links", required=True)
    v3.add_argument("--baselines", default=None)
    v3.add_argument("--history", type=int, default=500)
    v3.add_argument("--window", type=int, default=50)
    v3.add_argument("--grid-step", type=float, default=0.1,
                    help="grid resolution in meters (default 0.1 = 10 cm)")
    v3.add_argument("--link-sigma", type=float, default=0.3,
                    help="kernel σ for per-link influence on cells (default 0.3 m)")
    v3.add_argument("--wall-height", type=float, default=2.5)
    v3.set_defaults(func=cmd_view3d)

    cl = sub.add_parser("calibrate-links",
                        help="per-RX still-room baseline (writes JSON for `heatmap --baselines`)")
    cl.add_argument("source", help="udp:<port> typically")
    cl.add_argument("--out", default="baselines.json")
    cl.add_argument("--seconds", type=float, default=30.0,
                    help="recording duration after the settle delay (default 30s)")
    cl.add_argument("--settle", type=float, default=detector.AGC_SETTLE_SECONDS_DEFAULT,
                    help="seconds to wait before recording starts. "
                         "Doubles as your walk-out timer; bump it (e.g. --settle 30) "
                         "to give yourself time to leave the room.")
    cl.add_argument("--window", type=int, default=50)
    cl.set_defaults(func=cmd_calibrate_links)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
