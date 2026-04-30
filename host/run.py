"""Command-line entry point for the CSI motion detector.

Subcommands:

    capture <source> <out.log> [--seconds N]
        Dump raw CSI_DATA lines to disk. Use this to record a still-room
        baseline or a labeled session for offline analysis.

    calibrate <log_or_source> [--seconds N] [--window W] [--settle S]
        Read still-room samples and emit a baseline motion score. Drops
        the first --settle seconds to let the radio's AGC lock.

    detect <source> --baseline B [--window W] [--enter R] [--exit R]
        Live detection. Prints an event line on every transition between
        STILL and MOTION.

    view <source> [--history N] [--window W]
        Open a live matplotlib heatmap (subcarrier x time, color = |H| in
        dB) with a motion-score line below it.

    heatmap <source> --links links.json [--history N] [--window W]
        Multi-RX floor-plan view: per TX-RX line tinted by current
        motion-σ. Source is typically `udp:<port>`; the host listens
        for binary CSI packets from receivers running with
        CSI_RX_WIFI_SSID configured.
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
                               baselines_path=args.baselines)


def cmd_calibrate_links(args: argparse.Namespace) -> int:
    """Multi-RX still-room calibration. Writes {rx_mac: baseline} as JSON."""
    import json
    src = csi_collector.open_source(args.source)
    per_rx: dict[str, list[np.ndarray]] = {}
    start = time.time()
    settle_until = start + args.settle
    deadline = settle_until + args.seconds
    skipped = 0
    print(f"\n>>> LEAVE THE ROOM NOW <<<  starting capture in {args.settle:.0f}s "
          f"(then recording for {args.seconds:.0f}s)\n", file=sys.stderr, flush=True)
    next_tick = start + 1.0
    capture_announced = False
    for sample in src:
        now = time.time()
        if now < settle_until:
            skipped += 1
            if now >= next_tick:
                remaining = int(settle_until - now + 0.5)
                print(f"  ...{remaining}s until recording starts",
                      file=sys.stderr, flush=True)
                next_tick = now + 1.0
            continue
        if not capture_announced:
            print(f"\n>>> RECORDING <<<  hold still for {args.seconds:.0f}s\n",
                  file=sys.stderr, flush=True)
            capture_announced = True
            next_tick = now + 5.0
        if now >= deadline:
            break
        rx = sample.rx_id
        if rx is None:
            continue
        per_rx.setdefault(rx, []).append(sample.amplitude)
        if now >= next_tick:
            counts = ", ".join(f"{k[-5:]}={len(v)}" for k, v in sorted(per_rx.items()))
            print(f"  +{int(now - settle_until)}s  {counts}",
                  file=sys.stderr, flush=True)
            next_tick = now + 5.0
    baselines = detector.compute_link_baselines(per_rx, window=args.window)
    if not baselines:
        raise SystemExit("no usable baselines — did any RX deliver enough samples?")
    print(f"\ndropped {skipped} settle samples", file=sys.stderr)
    for mac, b in sorted(baselines.items()):
        print(f"  {mac}  baseline={b:.6f}  ({len(per_rx[mac])} samples)",
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
                    help="JSON config with room, TX, and RX positions (see links.example.json)")
    hm.add_argument("--baselines", default=None,
                    help="JSON file from `calibrate-links`; enables ratio-based coloring")
    hm.add_argument("--history", type=int, default=500)
    hm.add_argument("--window", type=int, default=50)
    hm.set_defaults(func=cmd_heatmap)

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
