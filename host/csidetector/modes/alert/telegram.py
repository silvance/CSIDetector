"""Telegram Bot API notifier.

Sends a one-line text message via ``sendMessage`` for each Event.
Token + chat id come from the alert-mode config TOML (see
``host/alert.example.toml``).

Uses ``urllib.request`` (stdlib) to keep the alert-mode dependency
footprint minimal — no extra requirements.txt entry needed.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from csidetector.modes.alert.notifier import (
    Event, Notifier, PermanentNotifierError,
)


_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"
_TIMEOUT_S = 10.0


class TelegramNotifier(Notifier):
    def __init__(self, bot_token: str, chat_id: str,
                 timeout_s: float = _TIMEOUT_S):
        if not bot_token:
            raise ValueError("TelegramNotifier: bot_token is empty")
        if not chat_id:
            raise ValueError("TelegramNotifier: chat_id is empty")
        self._url = _API_BASE.format(token=bot_token)
        self._chat_id = str(chat_id)
        self._timeout_s = float(timeout_s)

    def send(self, event: Event) -> None:
        # Default-format the message if the event didn't pre-render one.
        text = event.message or self._default_text(event)
        body = urllib.parse.urlencode({
            "chat_id": self._chat_id,
            "text": text,
        }).encode("utf-8")
        req = urllib.request.Request(
            self._url, data=body, method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # 4xx from Telegram = bad token / bad chat id / malformed
            # message — never going to succeed on retry. Surface as
            # permanent so the queue (Phase 3) doesn't loop forever.
            if 400 <= exc.code < 500:
                raise PermanentNotifierError(
                    f"telegram returned HTTP {exc.code}: {exc.reason}") from exc
            raise   # 5xx / network: transient, caller may retry
        except urllib.error.URLError:
            raise   # transient network issue
        if not payload.get("ok", False):
            # Body says ok=false even though HTTP was 2xx — Telegram does
            # this on some malformed-input cases. Treat as permanent.
            desc = payload.get("description", "unknown error")
            raise PermanentNotifierError(f"telegram reply not ok: {desc}")

    @staticmethod
    def _default_text(event: Event) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.ts))
        return f"[{event.kind}] {ts}"


def from_config(cfg: dict) -> Optional[TelegramNotifier]:
    """Build a TelegramNotifier from the [notifier] table of alert.toml.

    Returns None when the config has no notifier configured (or its
    type is not ``telegram``). Callers should fall back to NullNotifier.
    """
    notifier_cfg = cfg.get("notifier") or {}
    if notifier_cfg.get("type") != "telegram":
        return None
    return TelegramNotifier(
        bot_token=notifier_cfg.get("bot_token", ""),
        chat_id=notifier_cfg.get("chat_id", ""),
    )
