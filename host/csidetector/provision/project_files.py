"""Writers for the host-side config files the setup wizard produces.

  alert.toml  — Telegram credentials + alert preferences (owner-only).
  links.json  — localize-mode node list. Merged into an existing file so a
                hand-measured room polygon and node positions are never
                lost; new boards get provisional perimeter positions marked
                ``"placed": false`` until they're placed on the floorplan.

Both writers back up an existing file before replacing it and write
atomically (temp file + rename).
"""

from __future__ import annotations

import json
import os
import re
import time
import tomllib
from pathlib import Path
from typing import Optional

from csidetector.provision.devices import normalize_mac, try_normalize_mac

_TOKEN_RE = re.compile(r"^\d{5,}:[A-Za-z0-9_-]{30,}$")
_CHAT_ID_RE = re.compile(r"^(-?\d{3,}|@[A-Za-z][A-Za-z0-9_]{4,})$")

DEFAULT_ROOM = {"polygon": [[0.0, 0.0], [5.0, 0.0], [5.0, 4.0], [0.0, 4.0]]}


def _atomic_write(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _backup(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    backup.write_bytes(path.read_bytes())
    os.chmod(backup, path.stat().st_mode & 0o777)
    return backup


# --------------------------------------------------------------------------
# alert.toml
# --------------------------------------------------------------------------

def validate_telegram(bot_token: str, chat_id: str) -> list[str]:
    errors = []
    if not _TOKEN_RE.match(bot_token.strip()):
        errors.append("Bot token should look like 123456789:AAH… "
                      "(from @BotFather).")
    if not _CHAT_ID_RE.match(chat_id.strip()):
        errors.append("Chat ID should be a number (e.g. 8722080863, or "
                      "negative for groups) or an @channel name.")
    return errors


def _toml_str(value: str) -> str:
    out = ['"']
    for ch in value:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def render_alert_toml(bot_token: str, chat_id: str, location_label: str = "",
                      cooldown_s: float = 60, clear_on_exit: bool = False,
                      queue: Optional[dict] = None) -> str:
    q = {"enabled": True}
    q.update(queue or {})
    lines = [
        "# Written by `python run.py setup`. Contains a credential —",
        "# keep it chmod 600 and out of git.",
        "",
        "[notifier]",
        'type = "telegram"',
        f"bot_token = {_toml_str(bot_token.strip())}",
        f"chat_id = {_toml_str(chat_id.strip())}",
        "",
        "[alert]",
        f"cooldown_s = {float(cooldown_s):g}",
        f"clear_on_exit = {'true' if clear_on_exit else 'false'}",
    ]
    if location_label:
        lines.append(f"location_label = {_toml_str(location_label)}")
    lines += ["", "[queue]"]
    for key, value in q.items():
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value}")
        elif isinstance(value, str):
            lines.append(f"{key} = {_toml_str(value)}")
    return "\n".join(lines) + "\n"


def write_alert_toml(path: Path, bot_token: str, chat_id: str,
                     location_label: str = "") -> Optional[Path]:
    """Write alert.toml (0600). Carries cooldown / clear_on_exit / queue
    settings over from an existing file. Returns the backup path, if any."""
    path = Path(path)
    alert, queue = {}, {}
    if path.exists():
        try:
            existing = tomllib.loads(path.read_text())
            alert = existing.get("alert", {}) or {}
            queue = existing.get("queue", {}) or {}
        except (OSError, tomllib.TOMLDecodeError):
            pass
    text = render_alert_toml(
        bot_token, chat_id, location_label,
        cooldown_s=alert.get("cooldown_s", 60),
        clear_on_exit=bool(alert.get("clear_on_exit", False)),
        queue={k: v for k, v in queue.items()
               if isinstance(v, (bool, int, float, str))},
    )
    tomllib.loads(text)   # never write something the loader can't read
    backup = _backup(path)
    _atomic_write(path, text, mode=0o600)
    return backup


# --------------------------------------------------------------------------
# links.json
# --------------------------------------------------------------------------

def _room_bbox(room: dict) -> tuple[float, float, float, float]:
    if isinstance(room.get("polygon"), list) and room["polygon"]:
        xs = [float(p[0]) for p in room["polygon"]]
        ys = [float(p[1]) for p in room["polygon"]]
        return min(xs), min(ys), max(xs), max(ys)
    w = float(room.get("width_m", 5.0))
    h = float(room.get("height_m", 4.0))
    return 0.0, 0.0, w, h


def _perimeter_points(bbox, n: int) -> list[tuple[float, float]]:
    """n points evenly spaced around the bbox perimeter, starting at the
    bottom-left corner — provisional spots that keep the viewer usable."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    per = 2 * (w + h) or 1.0
    pts = []
    for i in range(n):
        d = per * i / max(n, 1)
        if d < w:
            pts.append((x0 + d, y0))
        elif d < w + h:
            pts.append((x1, y0 + d - w))
        elif d < 2 * w + h:
            pts.append((x1 - (d - w - h), y1))
        else:
            pts.append((x0, y1 - (d - 2 * w - h)))
    return [(round(x, 2), round(y, 2)) for x, y in pts]


def merge_links(existing: Optional[dict],
                txs: list[tuple[str, str]],
                rxs: list[tuple[str, str]]) -> dict:
    """Add (mac, label) nodes to a links.json dict without disturbing
    nodes that are already there.

    Existing nodes keep their position and label. New nodes are spread
    around the room perimeter and flagged ``"placed": false``. A legacy
    single ``"tx"`` entry is converted to a ``"txs"`` list.
    """
    out = dict(existing or {})
    out.setdefault("room", dict(DEFAULT_ROOM))
    legacy_tx = out.pop("tx", None)
    old_txs = list(out.get("txs") or [])
    if isinstance(legacy_tx, dict) and not old_txs:
        old_txs = [legacy_tx]
    old_rxs = list(out.get("rxs") or [])

    def merged(old: list, new: list[tuple[str, str]]):
        nodes = [dict(n) for n in old if isinstance(n, dict)]
        known = {try_normalize_mac(n.get("mac", "")) for n in nodes}
        fresh = []
        for mac, label in new:
            norm = normalize_mac(mac)
            if norm in known:
                continue
            known.add(norm)
            fresh.append((norm, label))
        return nodes, fresh

    tx_nodes, tx_new = merged(old_txs, txs)
    rx_nodes, rx_new = merged(old_rxs, rxs)
    spots = iter(_perimeter_points(_room_bbox(out["room"]),
                                   len(tx_new) + len(rx_new)))
    for i, (mac, label) in enumerate(tx_new, start=len(tx_nodes) + 1):
        x, y = next(spots)
        tx_nodes.append({"mac": mac, "x": x, "y": y,
                         "label": label or f"TX{i}", "placed": False})
    for i, (mac, label) in enumerate(rx_new, start=len(rx_nodes) + 1):
        x, y = next(spots)
        rx_nodes.append({"mac": mac, "x": x, "y": y,
                         "label": label or f"RX{i}", "placed": False})
    out["txs"] = tx_nodes
    out["rxs"] = rx_nodes
    return out


def write_links(path: Path, txs: list[tuple[str, str]],
                rxs: list[tuple[str, str]]) -> tuple[dict, Optional[Path]]:
    """Merge nodes into links.json at ``path``. Returns (data, backup)."""
    path = Path(path)
    existing = None
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"{path} exists but isn't valid JSON ({exc}); "
                             f"fix or move it first") from exc
        if not isinstance(existing, dict):
            raise ValueError(f"{path} is not a JSON object")
    data = merge_links(existing, txs, rxs)
    backup = _backup(path)
    _atomic_write(path, json.dumps(data, indent=2) + "\n")
    return data, backup
