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
    """Construct a Notifier from the [notifier] table, or NullNotifier."""
    from csidetector.modes.alert.notifier import NullNotifier
    from csidetector.modes.alert.telegram import from_config as telegram_from_config

    notifier = telegram_from_config(cfg)
    if notifier is not None:
        return notifier
    return NullNotifier()
