"""Command-line entry point for the CSI motion detector.

Three subcommands:

    capture  <source> <out.log>
        Just dump raw CSI lines to disk. Use this to record a still-room
        baseline or a labeled test session for offline analysis.

    calibrate <log_or_source> [--seconds N] [--window W]
        Read still-room samples and emit a baseline score. Pipe the result
        into `detect` via --baseline.

    detect   <source> --baseline B [--window W] [--enter R] [--exit R]
        Live detection. Prints a status line per sample and an event line
        on every transition between still and motion.
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
        for line in _raw_lines(args.source, args.baud):
            if not line.startswith("CSI "):
                continue
            f.write(line if line.endswith("\n") else line + "\n")
            n += 1
            if deadline and time.time() >= deadline:
                break
    print(f"captured {n} samples to {args.out}", file=sys.stderr)
    return 0


def _collect_amplitudes(source: str, seconds: float | None,
                        max_samples: int | None) -> tuple[np.ndarray, np.ndarray]:
    src = csi_collector.open_source(source)
    deadline = time.time() + seconds if seconds else None
    rows: list[np.ndarray] = []
    idx = None
    for sample in src:
        if idx is None:
            # Discover active subcarriers from the first sample's nonzero bins.
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
    amps = np.stack(rows)[:, idx]
    return amps, idx


def cmd_calibrate(args: argparse.Namespace) -> int:
    amps, idx = _collect_amplitudes(args.source, args.seconds, args.max_samples)
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
    for sample in src:
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
    cal.add_argument("--max-samples", type=int, default=None)
    cal.add_argument("--window", type=int, default=50)
    cal.set_defaults(func=cmd_calibrate)

    det_p = sub.add_parser("detect", help="live motion detection")
    det_p.add_argument("source")
    det_p.add_argument("--baseline", type=float, required=True)
    det_p.add_argument("--window", type=int, default=50)
    det_p.add_argument("--enter", type=float, default=3.0)
    det_p.add_argument("--exit", type=float, default=1.5)
    det_p.add_argument("--verbose", action="store_true")
    det_p.set_defaults(func=cmd_detect)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
