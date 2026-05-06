"""CLI dispatcher for the csidetector package.

Subcommand tree:

    csi-detector capture <source> <out>
    csi-detector localize  heatmap | view3d | calibrate-links | publish
    csi-detector alert     calibrate | detect | view-waterfall

The pre-refactor flat names (``heatmap``, ``view3d``, ``calibrate``,
``detect``, …) are also registered at the top level as back-compat
aliases so existing scripts keep working unchanged.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from csidetector.core import collector as csi_collector
from csidetector.core import detector


# --------------------------------------------------------------------------
# Helpers shared between top-level and mode-scoped commands.
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Command bodies. Alert-mode bodies live in csidetector.modes.alert.{calibrate,
# detect}. Localize-mode bodies still inline here as thin wrappers around
# csidetector.modes.localize.* — moving them out is a Phase 5 polish task.
# --------------------------------------------------------------------------

def cmd_capture(args: argparse.Namespace) -> int:
    deadline = time.time() + args.seconds if args.seconds else None
    n = 0
    with open(args.out, "w") as f:
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


def cmd_calibrate(args: argparse.Namespace) -> int:
    from csidetector.modes.alert.calibrate import run_calibrate
    return run_calibrate(
        source=args.source,
        seconds=args.seconds,
        settle=args.settle,
        max_samples=args.max_samples,
        window=args.window,
    )


def cmd_detect(args: argparse.Namespace) -> int:
    from csidetector.modes.alert.detect import run_detect
    from csidetector.modes.alert.config import load_config, build_notifier

    cfg = load_config(getattr(args, "alert_config", None))
    notifier = build_notifier(cfg)

    # CLI flags override config file values for the runtime knobs.
    alert_cfg = cfg.get("alert", {}) if isinstance(cfg, dict) else {}
    cooldown_s = (args.cooldown_s
                  if args.cooldown_s is not None
                  else float(alert_cfg.get("cooldown_s", 60.0)))
    clear_on_exit = (args.clear_on_exit
                     if args.clear_on_exit is not None
                     else bool(alert_cfg.get("clear_on_exit", False)))
    location_label = (args.location
                      if args.location is not None
                      else str(alert_cfg.get("location_label", "")))

    return run_detect(
        source=args.source,
        baseline=args.baseline,
        window=args.window,
        enter_ratio=args.enter,
        exit_ratio=args.exit,
        settle=args.settle,
        verbose=args.verbose,
        notifier=notifier,
        cooldown_s=cooldown_s,
        clear_on_exit=clear_on_exit,
        location_label=location_label,
    )


def cmd_view(args: argparse.Namespace) -> int:
    from csidetector.modes.alert import view_waterfall
    return view_waterfall.run_viewer(
        args.source, history=args.history, motion_window=args.window)


def cmd_heatmap(args: argparse.Namespace) -> int:
    from csidetector.modes.localize import heatmap
    return heatmap.run_heatmap(args.source, args.links,
                               history=args.history, motion_window=args.window,
                               baselines_path=args.baselines,
                               full_bright=args.full_bright,
                               motion_enter=args.motion_enter,
                               motion_exit=args.motion_exit,
                               motion_quantile=args.motion_quantile,
                               motion_max_enter=args.motion_max_enter,
                               motion_max_exit=args.motion_max_exit,
                               calibrate_settle_s=args.calibrate_settle,
                               calibrate_record_s=args.calibrate_record)


def cmd_publish(args: argparse.Namespace) -> int:
    from csidetector.modes.localize import publisher
    return publisher.run_publisher(
        args.source, args.links,
        c5_addr=args.c5_addr, c5_port=args.c5_port,
        baselines_path=args.baselines,
        history=args.history, motion_window=args.window,
        motion_enter=args.motion_enter, motion_exit=args.motion_exit,
        motion_quantile=args.motion_quantile,
        motion_max_enter=args.motion_max_enter,
        motion_max_exit=args.motion_max_exit,
        publish_hz=args.hz, verbose=args.verbose,
    )


def cmd_view3d(args: argparse.Namespace) -> int:
    from csidetector.modes.localize import view3d
    return view3d.run_viewer3d(
        args.source, args.links,
        history=args.history, motion_window=args.window,
        baselines_path=args.baselines,
        grid_step=args.grid_step, link_sigma_m=args.link_sigma,
        node_exclusion_m=args.node_exclusion,
        wall_height_m=args.wall_height,
        max_pins=args.max_pins,
        pin_separation_m=args.pin_separation,
        pin_smoothing=args.pin_smoothing,
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

    per_link: dict[tuple[str, str], list[np.ndarray]] = {}
    start = time.time()
    settle_until = start + args.settle
    deadline = settle_until + args.seconds

    print(f"\n>>> LEAVE THE ROOM NOW <<<  starting capture in {args.settle:.0f}s "
          f"(then recording for {args.seconds:.0f}s)\n", file=sys.stderr, flush=True)

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
        if item.rx_id is not None and item.mac is not None:
            key = (item.mac.lower(), item.rx_id.lower())
            per_link.setdefault(key, []).append(item.amplitude)
        if time.time() >= next_tick:
            by_rx: dict[str, int] = {}
            for (tx, rx), rows in per_link.items():
                by_rx[rx] = by_rx.get(rx, 0) + len(rows)
            counts = ", ".join(f"{k[-5:]}={v}" for k, v in sorted(by_rx.items()))
            print(f"  +{int(time.time() - settle_until)}s  {counts}",
                  file=sys.stderr, flush=True)
            next_tick = time.time() + 5.0

    stop.set()
    baselines = detector.compute_link_baselines(per_link, window=args.window)
    if not baselines:
        raise SystemExit("no usable baselines — did any link deliver enough samples?")
    min_required = 3 * args.window
    short = [(key, len(rows)) for key, rows in per_link.items()
             if key not in baselines]
    print(f"\nper-link:", file=sys.stderr)
    for (tx, rx), b in sorted(baselines.items()):
        print(f"  TX={tx}  RX={rx}  baseline={b:.6f}  "
              f"({len(per_link[(tx, rx)])} samples)", file=sys.stderr)
    for (tx, rx), n in sorted(short):
        print(f"  TX={tx}  RX={rx}  SKIPPED — only {n} samples, "
              f"need >= {min_required}; this link will render at 0× in the heatmap",
              file=sys.stderr)
    links_obj = {f"{tx}|{rx}": b for (tx, rx), b in baselines.items()}
    samples_obj = {f"{tx}|{rx}": len(rows) for (tx, rx), rows in per_link.items()}
    out_obj = {
        "_meta": {
            "format": "csidetector-baselines/1",
            "created": csi_collector.now_iso(),
            "settle_seconds": args.settle,
            "record_seconds": args.seconds,
            "window": args.window,
            "samples_per_link": samples_obj,
        },
        "links": links_obj,
    }
    with open(args.out, "w") as f:
        json.dump(out_obj, f, indent=2)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------
# Argparse-builder helpers. Each ``_attach_<cmd>`` function fully configures
# one parser; we call them once for the back-compat top-level form and once
# for the new mode-scoped form so both routes share an identical arg set.
# --------------------------------------------------------------------------

def _attach_capture(p: argparse.ArgumentParser) -> None:
    p.add_argument("source", help="serial port, log path, or '-' for stdin")
    p.add_argument("out", help="output log path")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--baud", type=int, default=921600)
    p.set_defaults(func=cmd_capture)


def _attach_calibrate(p: argparse.ArgumentParser) -> None:
    p.add_argument("source")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--settle", type=float,
                   default=detector.AGC_SETTLE_SECONDS_DEFAULT,
                   help="seconds to drop after start while AGC locks (default 10)")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--window", type=int, default=50)
    p.set_defaults(func=cmd_calibrate)


def _attach_detect(p: argparse.ArgumentParser) -> None:
    p.add_argument("source")
    p.add_argument("--baseline", type=float, required=True)
    p.add_argument("--settle", type=float,
                   default=detector.AGC_SETTLE_SECONDS_DEFAULT)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--enter", type=float, default=3.0)
    p.add_argument("--exit", type=float, default=1.5)
    p.add_argument("--verbose", action="store_true")
    # Alert-mode notifier knobs. Without --alert-config the detector
    # behaves exactly like before (prints transitions, no notifications).
    p.add_argument("--alert-config", default=None,
                   help="path to alert.toml with notifier credentials + "
                        "alert preferences. Without it, no notifications "
                        "fire (legacy stdout-only behavior preserved).")
    p.add_argument("--cooldown-s", type=float, default=None,
                   help="minimum seconds between repeated MOTION alerts "
                        "(default 60, or whatever alert.toml [alert] "
                        "cooldown_s sets).")
    # Tri-state on a bool flag — None means 'use config'. argparse
    # supports this via dest + const tricks; simpler: store_true and
    # let cmd_detect treat None as 'inherit from config'.
    p.add_argument("--clear-on-exit", action="store_const", const=True,
                   default=None,
                   help="also send a notification on MOTION → STILL "
                        "(off by default).")
    p.add_argument("--location", default=None,
                   help="short label prepended to alert messages "
                        "(e.g. 'Office'). Overrides location_label "
                        "in alert.toml.")
    p.set_defaults(func=cmd_detect)


def _attach_view(p: argparse.ArgumentParser) -> None:
    p.add_argument("source")
    p.add_argument("--history", type=int, default=500,
                   help="samples shown horizontally (~5 s at 100 Hz)")
    p.add_argument("--window", type=int, default=50,
                   help="motion-score sliding window (samples)")
    p.set_defaults(func=cmd_view)


def _attach_heatmap(p: argparse.ArgumentParser) -> None:
    p.add_argument("source", help="udp:<port> typically")
    p.add_argument("--links", required=True,
                   help="JSON config with room, TXs, and RX positions "
                        "(see links.example.json)")
    p.add_argument("--history", type=int, default=500)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--baselines", default=None,
                   help="JSON map of RX MAC -> still-room σ. When given, "
                        "links are colored by ratio (× baseline) instead of "
                        "auto-scaled raw σ.")
    p.add_argument("--full-bright", type=float, default=3.0,
                   help="ratio at which links saturate to the brightest "
                        "cmap value (default 3.0; only used with --baselines).")
    p.add_argument("--motion-enter", type=float, default=2.0,
                   help="median per-link ratio above which the room is "
                        "declared MOTION DETECTED (default 2.0×).")
    p.add_argument("--motion-exit", type=float, default=1.5,
                   help="median per-link ratio below which the room is "
                        "declared EMPTY (default 1.5×).")
    p.add_argument("--motion-quantile", type=float, default=0.75,
                   help="quantile of per-link ratios used to decide "
                        "MOTION/EMPTY (default 0.75).")
    p.add_argument("--motion-max-enter", type=float, default=3.5,
                   help="single-link ratio above which MOTION fires "
                        "regardless of the quantile (default 3.5×).")
    p.add_argument("--motion-max-exit", type=float, default=2.5,
                   help="single-link ratio below which the max-trigger "
                        "ladder stops voting MOTION (default 2.5×).")
    p.add_argument("--calibrate-settle", type=float, default=20.0,
                   help="seconds you have to leave the room after pressing "
                        "[C] before live recalibration starts recording "
                        "(default 20).")
    p.add_argument("--calibrate-record", type=float, default=15.0,
                   help="seconds of still-room σ recorded after the settle "
                        "window during live recalibration (default 15).")
    p.set_defaults(func=cmd_heatmap)


def _attach_view3d(p: argparse.ArgumentParser) -> None:
    p.add_argument("source", help="udp:<port> typically")
    p.add_argument("--links", required=True)
    p.add_argument("--baselines", default=None)
    p.add_argument("--history", type=int, default=500)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--grid-step", type=float, default=0.1,
                   help="grid resolution in meters (default 0.1 = 10 cm)")
    p.add_argument("--link-sigma", type=float, default=0.3,
                   help="kernel σ for per-link influence on cells "
                        "(default 0.3 m)")
    p.add_argument("--node-exclusion", type=float, default=0.5,
                   help="zero out cells within this radius (m) of any "
                        "TX/RX position. 0 disables. Default 0.5 m.")
    p.add_argument("--wall-height", type=float, default=2.5)
    p.add_argument("--max-pins", type=int, default=3)
    p.add_argument("--pin-separation", type=float, default=1.0)
    p.add_argument("--pin-smoothing", type=float, default=0.4)
    p.set_defaults(func=cmd_view3d)


def _attach_calibrate_links(p: argparse.ArgumentParser) -> None:
    p.add_argument("source", help="udp:<port> typically")
    p.add_argument("--out", default="baselines.json")
    p.add_argument("--seconds", type=float, default=30.0,
                   help="recording duration after the settle delay (default 30s)")
    p.add_argument("--settle", type=float,
                   default=detector.AGC_SETTLE_SECONDS_DEFAULT,
                   help="seconds to wait before recording starts. Doubles "
                        "as your walk-out timer; bump it (e.g. --settle 30) "
                        "to give yourself time to leave the room.")
    p.add_argument("--window", type=int, default=50)
    p.set_defaults(func=cmd_calibrate_links)


def _attach_publish(p: argparse.ArgumentParser) -> None:
    p.add_argument("source", help="udp:<port> typically")
    p.add_argument("--links", required=True)
    p.add_argument("--baselines", default=None)
    p.add_argument("--c5-addr", required=True,
                   help="IP of the receiving display (e.g. 10.42.0.42), or "
                        "10.42.0.255 for hotspot subnet broadcast.")
    p.add_argument("--c5-port", type=int, default=5567,
                   help="UDP port the C5 listens on (default 5567).")
    p.add_argument("--hz", type=float, default=5.0,
                   help="state-update emit rate (default 5 Hz).")
    p.add_argument("--history", type=int, default=500)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--motion-enter", type=float, default=2.0)
    p.add_argument("--motion-exit", type=float, default=1.5)
    p.add_argument("--motion-quantile", type=float, default=0.75)
    p.add_argument("--motion-max-enter", type=float, default=3.5)
    p.add_argument("--motion-max-exit", type=float, default=2.5)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_publish)


# --------------------------------------------------------------------------
# Top-level parser builder.
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="csi-detector")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- Top-level back-compat aliases (existing scripts keep working) ----
    _attach_capture(sub.add_parser("capture",
                                   help="dump raw CSI lines to a log file"))
    _attach_calibrate(sub.add_parser("calibrate",
                                     help="alert-mode: estimate still-room baseline"))
    _attach_detect(sub.add_parser("detect",
                                  help="alert-mode: live binary motion detection"))
    _attach_view(sub.add_parser("view",
                                help="single-stream waterfall viewer"))
    _attach_heatmap(sub.add_parser("heatmap",
                                   help="localize-mode: 2D floor-plan motion overlay (UDP)"))
    _attach_view3d(sub.add_parser("view3d",
                                  help="localize-mode: 2.5D room view"))
    _attach_calibrate_links(sub.add_parser("calibrate-links",
                                           help="localize-mode: per-link still-room baselines"))
    _attach_publish(sub.add_parser("publish",
                                   help="localize-mode: broadcast presence state to a remote display"))

    # ---- New canonical form: `localize <subcommand>` ----
    loc = sub.add_parser("localize",
                         help="multi-RX/TX room-scale localization mode")
    loc_sub = loc.add_subparsers(dest="localize_cmd", required=True)
    _attach_heatmap(loc_sub.add_parser("heatmap",
                                       help="2D floor-plan motion overlay"))
    _attach_view3d(loc_sub.add_parser("view3d",
                                      help="2.5D room view + person pins"))
    _attach_calibrate_links(loc_sub.add_parser("calibrate-links",
                                               help="per-link still-room baselines"))
    _attach_publish(loc_sub.add_parser("publish",
                                       help="broadcast presence state to a remote display"))

    # ---- New canonical form: `alert <subcommand>` ----
    al = sub.add_parser("alert",
                        help="single-TX/single-RX binary motion detection mode")
    al_sub = al.add_subparsers(dest="alert_cmd", required=True)
    _attach_calibrate(al_sub.add_parser("calibrate",
                                        help="estimate still-room baseline"))
    _attach_detect(al_sub.add_parser("detect",
                                     help="live binary motion detection"))
    _attach_view(al_sub.add_parser("view-waterfall",
                                   help="single-stream waterfall viewer "
                                        "(useful for install-time QA)"))

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
