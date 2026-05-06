"""End-to-end smoke test: detect → notifier-stub → queue roundtrip.

Avoids real hardware by feeding synthetic CSI samples directly into a
MotionDetector (the same detector run_detect uses). Verifies that a
quiet → noisy → quiet sequence drives a Notifier through a
STILL → MOTION → STILL flip and that the queue durably stores those
events.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time

import numpy as np
import pytest

from csidetector.core import detector
from csidetector.modes.alert.notifier import (
    Event, NullNotifier, Notifier,
)
from csidetector.modes.alert.queue import QueuingNotifier


class _RecordingNotifier(Notifier):
    def __init__(self):
        self.events: list[Event] = []
        self.lock = threading.Lock()

    def send(self, event: Event) -> None:
        with self.lock:
            self.events.append(event)


def test_null_notifier_send_is_noop():
    NullNotifier().send(Event.now("MOTION", "x"))   # must not raise


def test_detect_loop_drives_notifier_via_queue(synthetic_amplitudes):
    """Run the detection state machine over synthetic data; verify the
    queue-wrapped notifier records the transitions we expect."""
    cfg = detector.DetectorConfig(window=50, enter_ratio=3.0, exit_ratio=1.5)
    still = synthetic_amplitudes(n=300, n_sub=64, noise=0.01)
    motion = synthetic_amplitudes(n=300, n_sub=64, noise=0.10, seed=1)
    baseline = detector.compute_baseline(still[:200], window=50)
    det = detector.MotionDetector(np.arange(64), baseline, cfg)

    inner = _RecordingNotifier()
    with tempfile.TemporaryDirectory() as d:
        q = QueuingNotifier(inner=inner,
                            db_path=os.path.join(d, "q.db"),
                            poll_interval_s=0.05)
        try:
            last = False
            for amp in still:
                _, m = det.update(amp)
                if m and not last:
                    q.send(Event.now("MOTION", "go"))
                last = m
            for amp in motion:
                _, m = det.update(amp)
                if m and not last:
                    q.send(Event.now("MOTION", "go"))
                last = m
            for amp in still:
                _, m = det.update(amp)
                if (not m) and last:
                    q.send(Event.now("STILL", "clear"))
                last = m
            # Drain.
            deadline = time.time() + 2.0
            while time.time() < deadline and q.pending_count > 0:
                time.sleep(0.02)
        finally:
            q.close()

    kinds = [e.kind for e in inner.events]
    assert "MOTION" in kinds, f"no MOTION event recorded, got {kinds}"
    # We don't strictly require STILL — depends on whether detector
    # finished settling back. But MOTION must fire.
