"""Alert-mode live detection: single-stream MotionDetector + Notifier.

On every STILL ↔ MOTION transition, prints an event line to stdout
(legacy behavior) and dispatches a notification if a notifier is
configured. A configurable cooldown suppresses repeated MOTION alerts
during continuous activity. The optional ``clear_on_exit`` flag fires
a "clear" notification on MOTION → STILL.

Phase 3 will wrap the notifier in a durable queue so transient outages
don't drop alerts; for now sends are direct and best-effort (failures
log to stderr and are otherwise ignored).
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Optional

import numpy as np

from csidetector.core import collector as csi_collector
from csidetector.core import detector
from csidetector.modes.alert.notifier import (
    Event, Notifier, NullNotifier, PermanentNotifierError,
)


logger = logging.getLogger("csidetector.alert.detect")


def run_detect(source: str,
               baseline: float,
               *,
               window: int = 50,
               enter_ratio: float = 3.0,
               exit_ratio: float = 1.5,
               settle: float = detector.AGC_SETTLE_SECONDS_DEFAULT,
               verbose: bool = False,
               notifier: Optional[Notifier] = None,
               cooldown_s: float = 60.0,
               clear_on_exit: bool = False,
               location_label: str = "") -> int:
    """Run the live detection loop. Returns the process exit code."""
    if notifier is None:
        notifier = NullNotifier()

    cfg = detector.DetectorConfig(
        window=window,
        enter_ratio=enter_ratio,
        exit_ratio=exit_ratio,
    )
    src = csi_collector.open_source(source)
    det: detector.MotionDetector | None = None
    last_state = False
    last_alert_ts = 0.0
    settle_until = time.time() + settle

    for sample in src:
        if time.time() < settle_until:
            continue
        if det is None:
            idx = np.flatnonzero(sample.amplitude > 0)
            if idx.size == 0:
                continue
            det = detector.MotionDetector(idx, baseline, cfg)
        score, motion = det.update(sample.amplitude)
        if motion != last_state:
            kind = "MOTION" if motion else "STILL"
            ratio = score / det.baseline
            print(f"{csi_collector.now_iso()} {kind} score={score:.4f} "
                  f"baseline={det.baseline:.4f} ratio={ratio:.2f}",
                  flush=True)

            now = time.time()
            should_notify = False
            if motion:
                # STILL → MOTION: notify unless we recently did.
                if now - last_alert_ts >= cooldown_s:
                    should_notify = True
                    last_alert_ts = now
            elif clear_on_exit:
                # MOTION → STILL: notify only if explicitly enabled.
                should_notify = True

            if should_notify:
                _dispatch(notifier, kind, ratio, location_label)

            last_state = motion
        elif verbose:
            print(f"{csi_collector.now_iso()} score={score:.4f} "
                  f"ratio={score/det.baseline:.2f}",
                  flush=True)
    return 0


def _dispatch(notifier: Notifier, kind: str, ratio: float,
              location_label: str) -> None:
    """Build an Event and hand it to the notifier; log + drop on failure.

    Phase 3 replaces this best-effort path with a durable enqueue so
    transient failures don't lose events.
    """
    parts = []
    if location_label:
        parts.append(location_label)
    parts.append(kind)
    parts.append(f"ratio={ratio:.2f}×")
    message = " | ".join(parts)
    event = Event.now(kind=kind, message=message)
    try:
        notifier.send(event)
    except PermanentNotifierError as exc:
        logger.error("notifier permanent failure (dropping event %s): %s",
                     event.id, exc)
    except Exception as exc:
        # Transient. Phase 2 has no queue, so we log + drop. Phase 3
        # will retry via the durable queue.
        logger.warning("notifier transient failure (dropping event %s): %s",
                       event.id, exc)
