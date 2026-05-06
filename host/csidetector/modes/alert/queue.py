"""Durable outbound queue for alert notifications.

Wraps a ``Notifier`` so that ``send()`` persists the event in SQLite
and returns immediately. A background worker thread drains the queue
with exponential backoff: transient failures get rescheduled, permanent
failures (4xx from the upstream service) are marked dead and not
retried. Idempotency is the responsibility of the event id (UUIDv4 in
``Event.now()``): re-enqueueing the same id is a SQL no-op.

Database location defaults to ``~/.csidetector/alert-queue.db`` and is
created on first use. Schema migrates additively — existing rows survive
restarts.

Schema::

    CREATE TABLE events (
        event_id TEXT PRIMARY KEY,
        created_ts REAL NOT NULL,
        event_type TEXT NOT NULL,
        message TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        next_retry_ts REAL NOT NULL DEFAULT 0,
        sent_ts REAL
    )

``attempts = -1`` is the permanent-fail sentinel (Telegram 4xx etc).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from csidetector.modes.alert.notifier import (
    Event, Notifier, PermanentNotifierError,
)


logger = logging.getLogger("csidetector.alert.queue")

# Default queue path. Configurable via [queue] path in alert.toml.
DEFAULT_DB_PATH = "~/.csidetector/alert-queue.db"
# Backoff schedule for transient failures.
BACKOFF_BASE_S = 60.0
BACKOFF_MAX_S = 3600.0
# Worker poll interval (when no event-arrival signal kicks it sooner).
WORKER_POLL_S = 30.0
# Sentinel used in the attempts column for events we'll never retry.
PERMANENT_FAIL = -1


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id      TEXT PRIMARY KEY,
    created_ts    REAL NOT NULL,
    event_type    TEXT NOT NULL,
    message       TEXT NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT,
    next_retry_ts REAL NOT NULL DEFAULT 0,
    sent_ts       REAL
);
CREATE INDEX IF NOT EXISTS idx_pending
    ON events(sent_ts, next_retry_ts)
    WHERE sent_ts IS NULL AND attempts >= 0;
"""


def _backoff_seconds(attempts: int) -> float:
    """min(base * 2^(attempts-1), cap). attempts=1 → 60s, 2 → 120s, …"""
    if attempts <= 0:
        return 0.0
    return min(BACKOFF_BASE_S * (2 ** (attempts - 1)), BACKOFF_MAX_S)


class _Store:
    """Thin SQLite wrapper, single connection guarded by a Lock.

    Single connection (rather than connection-per-thread) keeps the
    schema migration sequence trivial and avoids WAL-mode setup. The
    write rate is tiny (≤ 1 row per state transition), so the lock is
    not a contention concern.
    """

    def __init__(self, db_path: str):
        path = Path(os.path.expanduser(db_path)).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(path), check_same_thread=False, isolation_level=None,
        )
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)

    def enqueue(self, event: Event) -> bool:
        """Insert a new row. Returns False if event_id already exists."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO events "
                "(event_id, created_ts, event_type, message, "
                " attempts, last_error, next_retry_ts, sent_ts) "
                "VALUES (?, ?, ?, ?, 0, NULL, 0, NULL)",
                (event.id, event.ts, event.kind, event.message),
            )
            return cur.rowcount > 0

    def fetch_due(self, now: float, limit: int = 32) -> list[Event]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT event_id, created_ts, event_type, message "
                "FROM events "
                "WHERE sent_ts IS NULL AND attempts >= 0 "
                "  AND next_retry_ts <= ? "
                "ORDER BY created_ts ASC LIMIT ?",
                (now, limit),
            )
            return [Event(id=r[0], ts=r[1], kind=r[2], message=r[3])
                    for r in cur.fetchall()]

    def mark_sent(self, event_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE events SET sent_ts = ?, last_error = NULL "
                "WHERE event_id = ?", (time.time(), event_id),
            )

    def mark_failed(self, event_id: str, error: str, *,
                    permanent: bool) -> None:
        with self._lock:
            if permanent:
                self._conn.execute(
                    "UPDATE events SET attempts = ?, last_error = ? "
                    "WHERE event_id = ?",
                    (PERMANENT_FAIL, error, event_id),
                )
                return
            cur = self._conn.execute(
                "SELECT attempts FROM events WHERE event_id = ?",
                (event_id,))
            row = cur.fetchone()
            if row is None:
                return  # vanished — nothing to do
            attempts = int(row[0]) + 1
            next_ts = time.time() + _backoff_seconds(attempts)
            self._conn.execute(
                "UPDATE events SET attempts = ?, last_error = ?, "
                " next_retry_ts = ? WHERE event_id = ?",
                (attempts, error, next_ts, event_id),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # Inspection helpers — handy for tests and the CLI status command.
    def pending_count(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM events "
                "WHERE sent_ts IS NULL AND attempts >= 0")
            return int(cur.fetchone()[0])

    def dead_count(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE attempts < 0")
            return int(cur.fetchone()[0])


class QueuingNotifier(Notifier):
    """Notifier wrapper that durably queues events and retries on failure."""

    def __init__(self, inner: Notifier,
                 db_path: str = DEFAULT_DB_PATH,
                 poll_interval_s: float = WORKER_POLL_S):
        self._inner = inner
        self._store = _Store(db_path)
        self._poll = float(poll_interval_s)
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread = threading.Thread(
            target=self._worker_loop, name="alert-queue-worker", daemon=True)
        self._thread.start()

    # --- Notifier interface ---------------------------------------------

    def send(self, event: Event) -> None:
        self._store.enqueue(event)
        self._wake.set()  # nudge the worker so a healthy connection delivers fast

    def close(self) -> None:
        """Signal the worker to drain + exit. Returns within ~poll_interval."""
        self._stopping.set()
        self._wake.set()
        self._thread.join(timeout=self._poll + 5.0)
        try:
            self._inner.close()
        except Exception:  # noqa: BLE001
            pass
        self._store.close()

    # --- Inspection (used by tests + future status command) -------------

    @property
    def pending_count(self) -> int:
        return self._store.pending_count()

    @property
    def dead_count(self) -> int:
        return self._store.dead_count()

    # --- Worker loop ----------------------------------------------------

    def _worker_loop(self) -> None:
        # Drain on start so events queued while the process was offline
        # are retried as soon as we come back up.
        while not self._stopping.is_set():
            self._drain_once()
            # Either wait for a wake signal (new enqueue) or for the
            # poll interval (so a long-failed event eventually retries).
            self._wake.wait(timeout=self._poll)
            self._wake.clear()
        # Final drain attempt on shutdown — best-effort.
        self._drain_once()

    def _drain_once(self) -> None:
        try:
            due = self._store.fetch_due(now=time.time())
        except sqlite3.Error as exc:
            logger.error("queue: fetch_due failed: %s", exc)
            return
        for event in due:
            if self._stopping.is_set():
                return
            try:
                self._inner.send(event)
            except PermanentNotifierError as exc:
                logger.error("queue: dropping event %s permanently: %s",
                             event.id, exc)
                self._store.mark_failed(event.id, str(exc), permanent=True)
            except Exception as exc:  # transient: retry later
                logger.warning("queue: deferring event %s: %s",
                               event.id, exc)
                self._store.mark_failed(event.id, str(exc), permanent=False)
            else:
                self._store.mark_sent(event.id)


def from_config(cfg: dict, inner: Notifier) -> Notifier:
    """Wrap ``inner`` in a QueuingNotifier when [queue] is enabled (default)."""
    queue_cfg = cfg.get("queue") or {}
    enabled = queue_cfg.get("enabled", True)
    if not enabled:
        return inner
    db_path = queue_cfg.get("path", DEFAULT_DB_PATH)
    poll = float(queue_cfg.get("poll_interval_s", WORKER_POLL_S))
    return QueuingNotifier(inner=inner, db_path=db_path, poll_interval_s=poll)
