"""The settings block written to a board's csi_cfg partition at flash time.

The firmware is prebuilt per chip; the per-board values (channel,
transmitter MAC filter, hotspot SSID/password, label…) go into a 4 KB
partition the firmware reads at boot. Format: see
firmware/components/csi_settings/include/csi_settings.h — the C parser
there is compiled and run against this encoder by the tests.
"""

from __future__ import annotations

import struct
import zlib
from typing import Union

MAGIC = b"CSIS"
VERSION = 1
HEADER = struct.Struct("<4sBBHI")      # magic, version, reserved, length, crc32
SIZE = 0x1000
LABEL_MAX_BYTES = 63                   # char label[64] in csi_settings_t

# Longest value the firmware accepts per key (its buffer size - 1).
_MAX_BYTES = {"role": 3, "label": LABEL_MAX_BYTES, "tx_macs": 127, "ssid": 32,
              "password": 64, "host_ip": 15}


def fit_label(label: str) -> str:
    """``label`` cut to what the board stores, on a character boundary."""
    data = (label or "").strip().encode("utf-8")[:LABEL_MAX_BYTES]
    return data.decode("utf-8", errors="ignore")


def encode(values: dict[str, Union[str, int]], size: int = SIZE) -> bytes:
    """The partition image for ``values``, padded with 0xFF (erased flash)
    to ``size``. Raises ValueError for a value the firmware would reject."""
    lines = []
    for key, value in values.items():
        text = str(int(value)) if isinstance(value, int) else str(value)
        if not key.isidentifier() or "=" in key:
            raise ValueError(f"bad settings key {key!r}")
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in text):
            raise ValueError(f"{key}: control characters aren't allowed")
        if len(text.encode("utf-8")) > _MAX_BYTES.get(key, 11):
            raise ValueError(f"{key}: too long for the board")
        lines.append(f"{key}={text}\n")
    payload = "".join(lines).encode("utf-8")
    blob = HEADER.pack(MAGIC, VERSION, 0, len(payload), zlib.crc32(payload)) + payload
    if len(blob) > size:
        raise ValueError("settings don't fit in the settings partition")
    return blob + b"\xff" * (size - len(blob))


def decode(blob: bytes) -> dict[str, str]:
    """The key/values in a settings block; ValueError if there's none or
    it's damaged."""
    if len(blob) < HEADER.size:
        raise ValueError("too short")
    magic, version, _, length, crc = HEADER.unpack_from(blob)
    if magic != MAGIC:
        raise ValueError("no settings block")
    if version != VERSION:
        raise ValueError(f"settings version {version} isn't supported")
    payload = blob[HEADER.size:HEADER.size + length]
    if len(payload) != length or zlib.crc32(payload) != crc:
        raise ValueError("settings block is damaged")
    out = {}
    for line in payload.decode("utf-8").split("\n"):
        if line:
            key, _, value = line.partition("=")
            out[key] = value
    return out
