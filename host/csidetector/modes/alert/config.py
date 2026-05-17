"""Alert-mode config loader (TOML).

Schema (see ``host/alert.example.toml``):

    [notifier]
    type = "telegram"
    bot_token = "..."
    chat_id = "..."

    [alert]
    cooldown_s = 60          # min seconds between repeated MOTION alerts
    clear_on_exit = false    # also notify on MOTION -> STILL
    location_label = "Office"  # prepended to the message text

Returns a dict with the parsed sections. Missing sections become empty
dicts so callers can ``cfg.get("alert", {}).get("cooldown_s", default)``
unconditionally.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Optional


def load_config(path: Optional[str]) -> dict:
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"alert: config file not found: {path}")
    with p.open("rb") as f:
        try:
            return tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            raise SystemExit(f"alert: failed to parse {path}: {exc}") from exc


def build_notifier(cfg: dict):
    """Construct a Notifier from the [notifier] table.

    When a sender is configured, it is wrapped in a QueuingNotifier so
    transient failures don't drop events. Disable the queue with
    ``[queue] enabled = false`` in alert.toml if you want unbuffered
    sends (rarely useful — recommended only for tests).

    Misconfigured creds (empty bot_token / chat_id) raise as a friendly
    ``SystemExit`` rather than letting the underlying ``ValueError``
    bubble up as an unhandled traceback from the CLI.
    """
    from csidetector.modes.alert.notifier import NullNotifier
    from csidetector.modes.alert.queue import from_config as queue_from_config
    from csidetector.modes.alert.telegram import from_config as telegram_from_config

    try:
        inner = telegram_from_config(cfg)
    except ValueError as exc:
        raise SystemExit(f"alert: notifier config invalid: {exc}") from exc
    if inner is None:
        return NullNotifier()
    return queue_from_config(cfg, inner=inner)
