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
import math
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


def _write_new(path: Path, data: bytes, mode: int, exclusive: bool) -> None:
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    fd = os.open(path, flags, mode)
    try:
        # fchmod before writing: O_CREAT's mode is masked by the umask and
        # ignored for a pre-existing file, and the content may be a secret.
        os.fchmod(fd, mode)
        os.write(fd, data)
    finally:
        os.close(fd)


def _atomic_write(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    _write_new(tmp, text.encode(), mode, exclusive=False)
    os.replace(tmp, path)


def _backup(path: Path, mode: int) -> Optional[Path]:
    """Copy ``path`` aside with ``mode`` permissions under a unique name.

    The mode is set explicitly rather than copied: an alert.toml that was
    0644 before chmod 600 must not leave a world-readable token copy.
    """
    if not path.exists():
        return None
    data = path.read_bytes()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for n in range(1000):
        suffix = f".bak-{stamp}" + (f"-{n}" if n else "")
        backup = path.with_name(path.name + suffix)
        try:
            _write_new(backup, data, mode, exclusive=True)
            return backup
        except FileExistsError:
            continue
    raise OSError(f"could not create a backup name for {path}")


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
    backup = _backup(path, mode=0o600)
    _atomic_write(path, text, mode=0o600)
    return backup


# --------------------------------------------------------------------------
# links.json
# --------------------------------------------------------------------------

def _num(v) -> Optional[float]:
    """``v`` as a finite float, or None. Accepts what the viewers accept
    (they call float() on these, so "0.3" works too); booleans are
    refused as an obvious mistake."""
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _is_num(v) -> bool:
    return _num(v) is not None


def validate_links_shape(data) -> None:
    """Raise ValueError if ``data`` isn't a links.json shape the viewers
    (and merge_links) can use. Catches hand-edits like a polygon point
    missing its y, ``"room": null`` or a non-string MAC before they turn
    into a crash."""
    if not isinstance(data, dict):
        raise ValueError("links.json must be a JSON object")
    room = data.get("room", DEFAULT_ROOM)
    if not isinstance(room, dict):
        raise ValueError("'room' must be an object")
    if "polygon" in room:
        poly = room["polygon"]
        if (not isinstance(poly, list) or len(poly) < 3 or not all(
                isinstance(pt, (list, tuple)) and len(pt) == 2
                and all(_is_num(c) for c in pt) for pt in poly)):
            raise ValueError("'room.polygon' must be a list of at least three "
                             "[x, y] number pairs")
    else:
        for key in ("width_m", "height_m"):
            # The viewers need both when there's no polygon.
            if not (_num(room.get(key)) or 0) > 0:
                raise ValueError(f"'room' needs a 'polygon' or a positive '{key}'")
    lists = [("txs", data.get("txs")), ("rxs", data.get("rxs"))]
    if "tx" in data:
        lists.append(("tx", [data["tx"]]))
    for key, nodes in lists:
        if nodes is None:
            continue
        if not isinstance(nodes, list):
            raise ValueError(f"'{key}' must be a list")
        for i, node in enumerate(nodes):
            where = f"'{key}'[{i}]" if key != "tx" else "'tx'"
            if not isinstance(node, dict):
                raise ValueError(f"{where} must be an object")
            if not isinstance(node.get("mac"), str) or not try_normalize_mac(node["mac"]):
                raise ValueError(f"{where} needs a 'mac' like aa:bb:cc:dd:ee:ff")
            if not (_is_num(node.get("x")) and _is_num(node.get("y"))):
                raise ValueError(f"{where} needs numeric 'x' and 'y'")


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
    bottom-left corner."""
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


def _placeholder_spots(bbox, taken: list[tuple[float, float]],
                       n: int) -> list[tuple[float, float]]:
    """Pick n perimeter spots, each as far as possible from existing nodes
    and from each other (greedy farthest-point), so provisional nodes
    don't land on top of placed ones."""
    candidates = _perimeter_points(bbox, max(16, 8 * (n + len(taken))))
    chosen: list[tuple[float, float]] = []
    occupied = list(taken)
    for _ in range(n):
        if not occupied:
            best = candidates[0]
        else:
            best = max(candidates, key=lambda c: min(
                (c[0] - o[0]) ** 2 + (c[1] - o[1]) ** 2 for o in occupied))
        chosen.append(best)
        occupied.append(best)
    return chosen


def merge_links(existing: Optional[dict],
                txs: list[tuple[str, str]],
                rxs: list[tuple[str, str]]) -> dict:
    """Add (mac, label) nodes to a links.json dict without disturbing
    nodes that are already there.

    Existing nodes keep their position and label. A board that changed
    role (reflashed TX → RX or back) moves to the other list, keeping its
    position. New nodes get provisional perimeter spots away from existing
    nodes and are flagged ``"placed": false``. A legacy single ``"tx"``
    entry is converted to a ``"txs"`` list. Raises ValueError for a
    malformed ``existing`` or a MAC given in both roles.
    """
    if existing is not None:
        validate_links_shape(existing)
    tx_norm = [(normalize_mac(m), l) for m, l in txs]
    rx_norm = [(normalize_mac(m), l) for m, l in rxs]
    both = {m for m, _ in tx_norm} & {m for m, _ in rx_norm}
    if both:
        raise ValueError(f"MAC(s) listed as both TX and RX: {', '.join(sorted(both))}")

    out = dict(existing or {})
    out.setdefault("room", dict(DEFAULT_ROOM))
    legacy_tx = out.pop("tx", None)
    tx_nodes = [dict(n) for n in (out.get("txs") or []) if isinstance(n, dict)]
    if isinstance(legacy_tx, dict) and not tx_nodes:
        tx_nodes = [dict(legacy_tx)]
    rx_nodes = [dict(n) for n in (out.get("rxs") or []) if isinstance(n, dict)]

    def take(nodes: list, mac: str) -> Optional[dict]:
        for i, n in enumerate(nodes):
            if try_normalize_mac(n.get("mac", "")) == mac:
                return nodes.pop(i)
        return None

    def present(nodes: list, mac: str) -> bool:
        return any(try_normalize_mac(n.get("mac", "")) == mac for n in nodes)

    fresh: list[tuple[list, str, str, str]] = []    # (dest, mac, label, prefix)
    for dest, other, new, prefix in ((tx_nodes, rx_nodes, tx_norm, "TX"),
                                     (rx_nodes, tx_nodes, rx_norm, "RX")):
        for mac, label in new:
            if present(dest, mac):
                continue
            moved = take(other, mac)
            if moved is not None:
                moved["mac"] = mac
                if label:
                    moved["label"] = label
                dest.append(moved)
            else:
                fresh.append((dest, mac, label, prefix))

    taken = [(_num(n["x"]), _num(n["y"])) for n in tx_nodes + rx_nodes
             if _is_num(n.get("x")) and _is_num(n.get("y"))]
    spots = iter(_placeholder_spots(_room_bbox(out["room"]), taken, len(fresh)))
    for dest, mac, label, prefix in fresh:
        x, y = next(spots)
        dest.append({"mac": mac, "x": x, "y": y,
                     "label": label or f"{prefix}{len(dest) + 1}", "placed": False})
    out["txs"] = tx_nodes
    out["rxs"] = rx_nodes
    return out


def write_links(path: Path, txs: list[tuple[str, str]],
                rxs: list[tuple[str, str]]) -> tuple[dict, Optional[Path]]:
    """Merge nodes into links.json at ``path``. Returns (data, backup).

    Raises ValueError (file untouched) when the existing file isn't valid
    JSON or isn't a usable links.json shape.
    """
    path = Path(path).resolve()           # through a symlink, not over it
    existing = None
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"{path} exists but isn't valid JSON ({exc}); "
                             f"fix or move it first") from exc
        try:
            validate_links_shape(existing)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}. Fix or move it first.") from exc
    data = merge_links(existing, txs, rxs)
    backup = _backup(path, mode=0o644)
    _atomic_write(path, json.dumps(data, indent=2) + "\n")
    return data, backup


def save_links(path: Path, data: dict) -> Optional[Path]:
    """Write a whole links.json (the floorplan editor's save). Returns the
    backup of the previous file, if there was one.

    Validated first: a shape the viewers can't load raises ValueError and
    nothing is written.
    """
    validate_links_shape(data)
    # Through a symlink, not over it: the image path stored relative to
    # links.json is relative to the real file's folder.
    path = Path(path).resolve()
    backup = _backup(path, mode=0o644)
    _atomic_write(path, json.dumps(data, indent=2) + "\n")
    return backup
