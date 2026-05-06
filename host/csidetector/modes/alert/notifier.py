"""Notifier abstraction for alert mode.

A Notifier consumes ``Event`` objects emitted on STILL/MOTION
transitions. Phase 2 ships ``NullNotifier`` (no-op default) and
``TelegramNotifier`` (HTTPS to api.telegram.org). Phase 3 adds a
``QueuingNotifier`` wrapper that durably stores events in SQLite and
hands them to an inner sender on a retry loop.
"""

from __future__ import annotations

import abc
import dataclasses
import time
import uuid


@dataclasses.dataclass(frozen=True)
class Event:
    """An alert-worthy state transition.

    The ``id`` field is a UUIDv4 — it makes Phase 3's queue idempotent
    (re-enqueueing the same event is a no-op).
    """
    id: str
    ts: float           # unix epoch seconds
    kind: str           # "MOTION" | "STILL"
    message: str

    @classmethod
    def now(cls, kind: str, message: str) -> "Event":
        return cls(id=str(uuid.uuid4()), ts=time.time(),
                   kind=kind, message=message)


class Notifier(abc.ABC):
    """Send a single Event somewhere.

    Implementations must be safe to call from a single thread; they do
    not need to be thread-safe (the detect loop is single-threaded, and
    Phase 3's queue worker also calls send() serially).
    """

    @abc.abstractmethod
    def send(self, event: Event) -> None:
        """Send the event. Raise on transient failure (caller may retry);
        raise ``PermanentNotifierError`` on irrecoverable failure (caller
        should drop the event). Plain success returns None."""

    def close(self) -> None:
        """Optional cleanup hook. Default: no-op."""


class PermanentNotifierError(Exception):
    """Raised by a Notifier when an event must not be retried.

    Examples: HTTP 4xx response from the upstream service (bad token,
    bad chat id, malformed message). Distinct from transient errors
    (network timeout, 5xx) which the caller is free to retry.
    """


class NullNotifier(Notifier):
    """Default no-op notifier — used when no config is provided.

    Detect mode still prints transition lines to stdout, so the operator
    sees state changes even without a configured notifier.
    """

    def send(self, event: Event) -> None:
        return None
