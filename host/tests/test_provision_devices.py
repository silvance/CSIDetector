"""Device discovery + registry — no hardware, no Qt."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from csidetector.provision import devices as dv


def _port(device, vid=None, pid=None, serial=None, desc=""):
    return SimpleNamespace(device=device, vid=vid, pid=pid,
                           serial_number=serial, description=desc)


@pytest.mark.parametrize("raw", [
    "A0:F2:62:F0:39:D8", "a0-f2-62-f0-39-d8", "a0f262f039d8", " a0:f2:62:f0:39:d8 ",
])
def test_normalize_mac_accepts_common_spellings(raw):
    assert dv.normalize_mac(raw) == "a0:f2:62:f0:39:d8"


@pytest.mark.parametrize("raw", ["", "a0:f2:62", "zz:f2:62:f0:39:d8", "a0f262f039d8ff"])
def test_normalize_mac_rejects_garbage(raw):
    with pytest.raises(ValueError):
        dv.normalize_mac(raw)


def test_list_boards_reads_mac_from_espressif_usb_serial():
    boards = dv.list_boards([
        _port("/dev/ttyACM1", vid=0x303A, pid=0x1001, serial="A0:F2:62:F0:39:D8"),
        _port("/dev/ttyUSB0", vid=0x10C4, pid=0xEA60, serial="0001"),
        _port("/dev/ttyS0"),                       # no VID → skipped
        _port("/dev/ttyACM0", vid=0x303A, pid=0x1001, serial="garbage"),
    ])
    assert [b.port for b in boards] == ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0"]
    by_port = {b.port: b for b in boards}
    assert by_port["/dev/ttyACM1"].mac == "a0:f2:62:f0:39:d8"
    assert by_port["/dev/ttyACM1"].interface == "native USB"
    # Bridge serial is the bridge's, not the chip's — must not be used as MAC.
    assert by_port["/dev/ttyUSB0"].mac == ""
    assert by_port["/dev/ttyUSB0"].interface == "CP210x bridge"
    # Unparseable Espressif serial → unknown MAC, not a crash.
    assert by_port["/dev/ttyACM0"].mac == ""


ESPTOOL_V5 = """esptool v5.2.0
Connected to ESP32-S3 on /dev/ttyACM0:
Chip type:          ESP32-S3 (QFN56) (revision v0.2)
Features:           Wi-Fi, BT 5 (LE), Dual Core + LP Core, 240MHz
MAC:                a0:f2:62:f0:39:d8

Stub flasher running.
MAC:                a0:f2:62:f0:39:d8
Hard resetting via RTS pin...
"""

ESPTOOL_V4 = """esptool.py v4.8.1
Serial port /dev/ttyUSB0
Connecting....
Chip is ESP32-S3 (QFN56) (revision v0.2)
Features: WiFi, BLE, Embedded PSRAM 2MB (AP_3v3)
MAC: 34:85:18:aa:bb:cc
Uploading stub...
"""

ESPTOOL_C5 = """Chip type:          ESP32-C5 (QFN32) (revision v1.0)
BASE MAC:           60:55:f9:00:11:22
MAC_EXT:            ff:fe
"""


@pytest.mark.parametrize("output,chip,mac", [
    (ESPTOOL_V5, "ESP32-S3", "a0:f2:62:f0:39:d8"),
    (ESPTOOL_V4, "ESP32-S3", "34:85:18:aa:bb:cc"),
    (ESPTOOL_C5, "ESP32-C5", "60:55:f9:00:11:22"),
    ("Failed to connect to ESP32-S3: No serial data received.", "", ""),
])
def test_parse_esptool_identify(output, chip, mac):
    assert dv.parse_esptool_identify(output) == (chip, mac)


def test_chip_to_target():
    assert dv.chip_to_target("ESP32-S3") == "esp32s3"
    assert dv.chip_to_target("ESP32") == "esp32"
    assert dv.chip_to_target("") == ""


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

def test_registry_round_trip(tmp_path):
    path = tmp_path / "devices.json"
    reg = dv.Registry(path)
    reg.upsert("A0F262F039D8", label="Desk TX", role=dv.ROLE_TX)
    reg.upsert("a0:f2:62:f0:39:28", label="South RX", role=dv.ROLE_RX)

    again = dv.Registry(path)
    rec = again.get("a0:f2:62:f0:39:d8")
    assert rec is not None and rec.label == "Desk TX" and rec.role == dv.ROLE_TX
    assert again.tx_macs() == ["a0:f2:62:f0:39:d8"]
    assert json.loads(path.read_text())["format"] == "csidetector-devices/1"


def test_registry_rejects_unknown_fields(tmp_path):
    reg = dv.Registry(tmp_path / "devices.json")
    with pytest.raises(AttributeError):
        reg.upsert("a0:f2:62:f0:39:d8", colour="red")


def test_registry_moves_corrupt_file_aside(tmp_path):
    path = tmp_path / "devices.json"
    path.write_text("{not json")
    reg = dv.Registry(path)
    assert reg.all() == []
    # Original content preserved next to it, not silently overwritten.
    backups = list(tmp_path.glob("devices.json.corrupt-*"))
    assert len(backups) == 1 and backups[0].read_text() == "{not json"


def test_registry_ignores_unknown_keys_and_bad_macs_on_load(tmp_path):
    path = tmp_path / "devices.json"
    path.write_text(json.dumps({"devices": {
        "A0:F2:62:F0:39:D8": {"label": "x", "future_field": 1},
        "not-a-mac": {"label": "y"},
    }}))
    reg = dv.Registry(path)
    assert [r.mac for r in reg.all()] == ["a0:f2:62:f0:39:d8"]
