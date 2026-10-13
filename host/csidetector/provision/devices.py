"""Attached-board discovery and the known-device registry.

ESP32-S3 / C3 / C5 boards that expose the chip's native USB-Serial/JTAG
controller (Espressif VID 0x303A) report the chip's factory MAC as their
USB serial number, so we can identify them without opening the port.
Boards behind a USB-UART bridge (CP210x, CH34x, FTDI) report the
bridge's serial instead; those need an explicit esptool ``read_mac``.

The registry maps MAC → label / role / last-flash metadata and lives at
``~/.csidetector/devices.json``. It never stores Wi-Fi passwords.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger("csidetector.provision.devices")

ESPRESSIF_VID = 0x303A
# USB-UART bridges commonly found on ESP32 dev boards.
_BRIDGE_VIDS = {
    0x10C4: "CP210x bridge",
    0x1A86: "CH34x bridge",
    0x0403: "FTDI bridge",
}

ROLE_TX = "tx"
ROLE_RX = "rx"
ROLES = ("", ROLE_TX, ROLE_RX)

_MAC_HEX_RE = re.compile(r"^[0-9a-f]{12}$")


def normalize_mac(value: str) -> str:
    """Return ``aa:bb:cc:dd:ee:ff`` for any common MAC spelling.

    Accepts colon-, dash-, or un-separated hex in either case. Raises
    ValueError for anything else.
    """
    s = (value or "").strip().lower().replace(":", "").replace("-", "")
    if not _MAC_HEX_RE.match(s):
        raise ValueError(f"not a MAC address: {value!r}")
    return ":".join(s[i:i + 2] for i in range(0, 12, 2))


def try_normalize_mac(value: str) -> str:
    try:
        return normalize_mac(value)
    except ValueError:
        return ""


@dataclasses.dataclass
class AttachedBoard:
    """A USB serial device that might be an ESP32."""
    port: str
    vid: Optional[int]
    pid: Optional[int]
    usb_serial: str
    description: str
    mac: str = ""        # lowercase colon form, "" when unknown
    chip: str = ""       # e.g. "ESP32-S3"; "" until identified

    @property
    def interface(self) -> str:
        if self.vid == ESPRESSIF_VID:
            return "native USB"
        return _BRIDGE_VIDS.get(self.vid or -1, "USB serial")


def list_boards(comports: Optional[Iterable] = None) -> list[AttachedBoard]:
    """Enumerate USB serial ports, newest naming order first.

    ``comports`` is injectable for tests; by default pyserial's
    ``list_ports.comports()`` is used. Non-USB ports (no VID, e.g.
    built-in /dev/ttyS*) are skipped — no ESP32 dev board lives there.
    """
    if comports is None:
        from serial.tools import list_ports
        comports = list_ports.comports()
    boards: list[AttachedBoard] = []
    for p in comports:
        vid = getattr(p, "vid", None)
        if vid is None:
            continue
        serial = getattr(p, "serial_number", None) or ""
        mac = try_normalize_mac(serial) if vid == ESPRESSIF_VID else ""
        boards.append(AttachedBoard(
            port=p.device,
            vid=vid,
            pid=getattr(p, "pid", None),
            usb_serial=serial,
            description=getattr(p, "description", "") or "",
            mac=mac,
        ))
    boards.sort(key=lambda b: b.port)
    return boards


# esptool v4 prints "Chip is ESP32-S3 (QFN56) ..." and "MAC: aa:..";
# v5 prints "Chip type:   ESP32-S3 (QFN56) ..." and "MAC:   aa:..",
# and on some chips "BASE MAC:". Accept all of them.
_CHIP_RE = re.compile(r"(?:Chip type:\s*|Chip is\s+)(ESP32[-A-Za-z0-9]*)")
_MAC_LINE_RE = re.compile(
    r"^\s*(?:BASE\s+)?MAC:\s*((?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2})\b",
    re.MULTILINE)


def parse_esptool_identify(output: str) -> tuple[str, str]:
    """Extract (chip, mac) from esptool ``read_mac`` / ``chip_id`` output.

    Either value is "" when not found.
    """
    chip_m = _CHIP_RE.search(output or "")
    mac_m = _MAC_LINE_RE.search(output or "")
    chip = chip_m.group(1).upper() if chip_m else ""
    mac = mac_m.group(1).lower() if mac_m else ""
    return chip, mac


def chip_to_target(chip: str) -> str:
    """Map esptool's chip name to an ESP-IDF target: ESP32-S3 → esp32s3."""
    return chip.lower().replace("-", "") if chip else ""


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

@dataclasses.dataclass
class DeviceRecord:
    mac: str
    label: str = ""
    role: str = ""              # "" | "tx" | "rx"
    chip: str = ""
    firmware: str = ""          # firmware project last flashed
    last_flashed: float = 0.0   # unix seconds; 0 = never via setup
    config: dict = dataclasses.field(default_factory=dict)  # no secrets
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "DeviceRecord":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


def default_registry_path() -> Path:
    return Path(os.path.expanduser("~/.csidetector/devices.json"))


class Registry:
    """MAC-keyed store of known boards, persisted as JSON.

    A corrupt file is moved aside (``devices.json.corrupt-<ts>``) rather
    than silently overwritten, so hand-entered labels aren't lost.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else default_registry_path()
        self._records: dict[str, DeviceRecord] = {}
        self.load()

    def load(self) -> None:
        self._records = {}
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            devices = raw.get("devices", {}) if isinstance(raw, dict) else {}
            for mac, rec in devices.items():
                norm = try_normalize_mac(mac)
                if norm and isinstance(rec, dict):
                    rec = dict(rec, mac=norm)
                    self._records[norm] = DeviceRecord.from_dict(rec)
        except (OSError, ValueError, TypeError) as exc:
            backup = self.path.with_name(
                f"{self.path.name}.corrupt-{int(time.time())}")
            logger.warning("device registry %s unreadable (%s); moved to %s",
                           self.path, exc, backup)
            try:
                os.replace(self.path, backup)
            except OSError:
                pass
            self._records = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "csidetector-devices/1",
            "devices": {mac: dataclasses.asdict(r)
                        for mac, r in sorted(self._records.items())},
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.path)

    def get(self, mac: str) -> Optional[DeviceRecord]:
        return self._records.get(try_normalize_mac(mac))

    def upsert(self, mac: str, **changes) -> DeviceRecord:
        norm = normalize_mac(mac)
        rec = self._records.get(norm) or DeviceRecord(mac=norm)
        for key, value in changes.items():
            if not hasattr(rec, key) or key == "mac":
                raise AttributeError(f"unknown device field: {key}")
            setattr(rec, key, value)
        self._records[norm] = rec
        self.save()
        return rec

    def all(self) -> list[DeviceRecord]:
        return sorted(self._records.values(),
                      key=lambda r: (r.role, r.label or "~", r.mac))

    def tx_macs(self) -> list[str]:
        return [r.mac for r in self.all() if r.role == ROLE_TX]
