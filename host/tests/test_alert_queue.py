"""Smoke tests for the alert-mode durable outbound queue."""

from __future__ import annotations

import os
import tempfile
import threading
import time

import pytest

from csidetector.modes.alert.notifier import (
    Event, Notifier, PermanentNotifierError,
)
from csidetector.modes.alert.queue import QueuingNotifier


class _StubInner(Notifier):
    """Notifier that records every send and can be made to fail on demand."""

    def __init__(self):
        self.calls: list[str] = []
        self.fail_first_n = 0
        self.permanent_on: set[str] = set()
        self.lock = threading.Lock()

    def send(self, event: Event) -> None:
        with self.lock:
            self.calls.append(event.id)
            if event.id in self.permanent_on:
                raise PermanentNotifierError("simulated 4xx")
            if self.fail_first_n > 0:
                self.fail_first_n -= 1
                raise RuntimeError("simulated network down")


@pytest.fixture
def db_path():
    d = tempfile.mkdtemp()
    yield os.path.join(d, "queue.db")


def _wait_for(predicate, timeout: float = 2.0, step: float = 0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return False


def test_queue_happy_path(db_path):
    inner = _StubInner()
    q = QueuingNotifier(inner=inner, db_path=db_path, poll_interval_s=0.1)
    try:
        ev = Event.now("MOTION", "first")
        q.send(ev)
        assert _wait_for(lambda: ev.id in inner.calls and q.pending_count == 0)
    finally:
        q.close()


def test_queue_idempotent_on_duplicate_ids(db_path):
    inner = _StubInner()
    q = QueuingNotifier(inner=inner, db_path=db_path, poll_interval_s=0.1)
    try:
        ev = Event.now("MOTION", "once")
        q.send(ev)
        assert _wait_for(lambda: ev.id in inner.calls)
        n0 = len(inner.calls)
        q.send(ev)   # re-enqueue same id — must be a no-op
        time.sleep(0.4)
        assert len(inner.calls) == n0
    finally:
        q.close()


def test_queue_defers_transient_failure(db_path):
    inner = _StubInner()
    inner.fail_first_n = 5
    q = QueuingNotifier(inner=inner, db_path=db_path, poll_interval_s=0.1)
    try:
        ev = Event.now("MOTION", "flaky")
        q.send(ev)
        # First attempt fires immediately and fails; the row stays
        # pending until the next backoff window expires.
        assert _wait_for(lambda: ev.id in inner.calls)
        time.sleep(0.4)
        assert q.pending_count >= 1, "transient failure was not deferred"
    finally:
        q.close()


def test_queue_marks_permanent_failure_dead(db_path):
    inner = _StubInner()
    ev = Event.now("MOTION", "bad")
    inner.permanent_on.add(ev.id)
    q = QueuingNotifier(inner=inner, db_path=db_path, poll_interval_s=0.1)
    try:
        q.send(ev)
        assert _wait_for(lambda: q.dead_count == 1)
        assert q.pending_count == 0, "permanently-failed row must not be pending"
    finally:
        q.close()


def test_queue_survives_restart(db_path):
    """A pending event from a previous session is preserved on restart."""
    inner1 = _StubInner()
    inner1.fail_first_n = 1   # first attempt fails → row deferred
    q1 = QueuingNotifier(inner=inner1, db_path=db_path, poll_interval_s=0.1)
    ev = Event.now("MOTION", "queued")
    q1.send(ev)
    assert _wait_for(lambda: ev.id in inner1.calls)
    q1.close()

    # Re-open the same DB; the deferred row should still be there.
    inner2 = _StubInner()
    q2 = QueuingNotifier(inner=inner2, db_path=db_path, poll_interval_s=0.1)
    try:
        # Backoff means the worker won't immediately retry, but the row
        # survives. pending_count > 0 proves persistence.
        assert q2.pending_count >= 1
    finally:
        q2.close()


def test_queue_close_drains_pending_events(db_path):
    """Per docstring, close() drains pending events as a best-effort
    final attempt. The bug: _drain_once short-circuited inside its for
    loop on _stopping.is_set(), so the final shutdown drain fetched
    rows but never sent any of them. Fixed by passing force=True to
    the final drain.
    """
    inner = _StubInner()
    q = QueuingNotifier(inner=inner, db_path=db_path,
                        poll_interval_s=10.0,   # ~never wakes naturally
                        backoff_base_s=0.05)
    # Enqueue several events without giving the natural poll a chance.
    events = [Event.now("MOTION", f"e{i}") for i in range(5)]
    # Pause the worker by holding inner.lock briefly so the first
    # _drain_once on natural wake hasn't caught up. Actually simpler:
    # enqueue and immediately close. With the old bug, close()'s final
    # drain would skip all events. With the fix (force=True), close()
    # delivers them all before returning.
    for ev in events:
        q.send(ev)
    q.close()

    delivered = {e.id for e in events} & set(inner.calls)
    assert delivered == {e.id for e in events}, (
        f"close() should drain all pending — only delivered {len(delivered)}/5")


def test_queue_retry_then_success_marks_sent(db_path):
    """Transient failure → backoff window expires → next attempt succeeds.

    Uses ``backoff_base_s=0.05`` so the deterministic retry sequence
    completes in well under a second. With production defaults
    (``BACKOFF_BASE_S=60``) this test would block for >1 minute.
    """
    inner = _StubInner()
    inner.fail_first_n = 2     # fail twice, then succeed
    q = QueuingNotifier(inner=inner, db_path=db_path,
                        poll_interval_s=0.05,
                        backoff_base_s=0.05)
    try:
        ev = Event.now("MOTION", "flaky-but-eventual")
        q.send(ev)

        # Three attempts total: two failures + one success.
        assert _wait_for(lambda: inner.calls.count(ev.id) >= 3, timeout=3.0), \
            f"only {inner.calls.count(ev.id)} attempts made"
        # And the row is no longer pending.
        assert _wait_for(lambda: q.pending_count == 0, timeout=3.0), \
            f"pending {q.pending_count} after success"
        # Not marked dead either.
        assert q.dead_count == 0
    finally:
        q.close()
