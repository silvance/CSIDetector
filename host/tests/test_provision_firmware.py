"""Firmware config, command construction, output parsing — no hardware."""

from __future__ import annotations

import json
import os
import re
import shlex
import stat
import sys
from pathlib import Path

import pytest

from csidetector.provision import firmware as fw

POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX-only (bash, file modes, symlinks or /proc)")

REPO_FIRMWARE = fw.repo_firmware_root()


# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------

def test_settings_keys_are_ones_the_firmware_reads():
    """A key the firmware's parser doesn't know is silently ignored there,
    so a typo here would flash the default value instead."""
    parser = (REPO_FIRMWARE / "components" / "csi_settings" / "csi_settings_parse.c").read_text()
    known = set(re.findall(r'KEY\("(\w+)"\)', parser))
    assert set(fw.TxConfig().settings()) <= known
    assert set(fw.RxConfig(ssid="CSIDetector").settings()) <= known


def test_tx_config_validation():
    assert fw.TxConfig().validate() == []
    assert fw.TxConfig(channel=14).validate()
    assert fw.TxConfig(rate_hz=0).validate()


def test_rx_uart_only_leaves_hotspot_settings_unset():
    cfg = fw.RxConfig(tx_macs=["A0F262F039D8"])
    assert cfg.validate() == []
    k = cfg.settings()
    assert k["ssid"] == "" and k["password"] == ""
    assert k["tx_macs"] == "a0:f2:62:f0:39:d8"
    assert "host_ip" not in k
    assert cfg.summary()["mode"] == "uart"


def test_rx_hotspot_config_and_summary_excludes_password():
    cfg = fw.RxConfig(tx_macs=["a0:f2:62:f0:39:d8", "ac:a7:04:2c:24:cc"],
                      ssid="CSIDetector", password="hunter22",
                      host_ip="10.42.1.1", host_port=5566)
    assert cfg.validate() == []
    k = cfg.settings()
    assert k["tx_macs"] == "a0:f2:62:f0:39:d8, ac:a7:04:2c:24:cc"
    assert k["host_ip"] == "10.42.1.1" and k["password"] == "hunter22"
    assert "hunter22" not in repr(cfg.summary())


def test_labels_are_cut_to_what_the_board_holds():
    cfg = fw.TxConfig(label="  " + "Küche " * 20)
    assert len(cfg.settings()["label"].encode()) <= 63
    assert fw.TxConfig(label="bad\nlabel").validate()


@pytest.mark.parametrize("kwargs,needle", [
    ({"tx_macs": ["a0:f2:62:f0:39:d8"] * 5}, "at most"),
    ({"tx_macs": ["nope"]}, "Not a MAC"),
    ({"ssid": "x", "host_ip": "10.42.0"}, "IPv4"),
    ({"ssid": "x", "password": "short"}, "8–64"),
    ({"ssid": "a" * 33}, "32 bytes"),
    ({"ssid": "x", "host_port": 0}, "port"),
    ({"channel": 0}, "Channel"),
])
def test_rx_config_validation_errors(kwargs, needle):
    errors = fw.RxConfig(**kwargs).validate()
    assert any(needle in e for e in errors), errors


# --------------------------------------------------------------------------
# Prebuilt firmware
# --------------------------------------------------------------------------

def test_the_shipped_firmware_is_complete():
    """firmware/prebuilt (built by CI from the sources) has both projects
    for every supported chip, intact, with the settings partition."""
    for target in fw.TARGETS:
        for project in fw.PROJECTS.values():
            image = fw.find_image(project, target, REPO_FIRMWARE / "prebuilt")
            fw.check_image(image)
            assert image.chip == target
            assert (image.settings_offset, image.settings_size) == (0xF000, 0x1000)
            assert (image.nvs_offset, image.nvs_size) == (0x9000, 0x5000)
            app = max(image.files)
            assert app[0] == 0x10000 and app[1].name == f"{project}.bin"


def test_flash_job_finds_its_image(prebuilt):
    rx = fw.FlashJob(port="/dev/ttyACM0", config=fw.RxConfig())
    tx = fw.FlashJob(port="/dev/ttyACM0", config=fw.TxConfig(), target="esp32c5")
    assert rx.validate() == [] and tx.validate() == []
    assert rx.image().project == "csi_receiver" and tx.image().target == "esp32c5"
    assert tx.image().files[0][0] == 0x2000           # the C5's bootloader offset


def test_unsupported_or_missing_firmware_is_explained(prebuilt, tmp_path, monkeypatch):
    job = fw.FlashJob(port="/dev/ttyACM0", config=fw.TxConfig(), target="esp32h2")
    assert any("ESP32-H2 boards aren't supported" in e for e in job.validate())
    (prebuilt / "manifest.json").write_text(json.dumps(
        {"format": fw.MANIFEST_FORMAT, "images": {}}))
    assert any("no prebuilt csi_transmitter firmware for the ESP32-S3" in e
               for e in fw.FlashJob(port="x", config=fw.TxConfig()).validate())
    monkeypatch.setenv("CSIDETECTOR_FIRMWARE_DIR", str(tmp_path / "nowhere"))
    assert any("firmware is missing" in e
               for e in fw.FlashJob(port="x", config=fw.TxConfig()).validate())
    (prebuilt / "manifest.json").write_text('{"format": "csidetector-firmware/9"}')
    monkeypatch.setenv("CSIDETECTOR_FIRMWARE_DIR", str(prebuilt))
    with pytest.raises(fw.FirmwareError, match="Update CSIDetector"):
        fw.load_manifest()


def test_flash_command_names_the_chip_and_every_part(prebuilt, tmp_path):
    job = fw.FlashJob(port="COM7", config=fw.TxConfig(), target="esp32")
    image = job.image()
    argv = fw.flash_command(job, image, tmp_path / "s.bin")
    assert argv[:3] == [sys.executable, "-m", "esptool"]
    assert argv[argv.index("--chip") + 1] == "esp32"     # esptool refuses another chip
    assert argv[argv.index("--port") + 1] == "COM7"
    assert "write-flash" in argv and argv[argv.index("--flash-size") + 1] == "2MB"
    tail = argv[argv.index("2MB") + 1:]
    assert tail[0::2] == ["0x1000", "0x8000", "0x9000", "0xf000", "0x10000"]
    assert tail[7] == str(tmp_path / "s.bin")
    assert Path(tail[5]).read_bytes() == b"\xff" * 0x5000     # NVS blanked
    assert fw.identify_command("COM7")[-3:] == ["--port", "COM7", "read-mac"]


@POSIX_ONLY
def test_settings_files_are_private_and_stale_ones_are_cleared(prebuilt, tmp_path):
    job = fw.FlashJob(port="x", config=fw.RxConfig(ssid="Lab", password="hunter22"))
    old = fw.write_settings_file(job, job.image())
    os.utime(old, (0, 0))                      # from a crash long ago
    new = fw.write_settings_file(job, job.image())
    assert not old.exists() and new.exists()
    assert stat.S_IMODE(new.stat().st_mode) == 0o600
    assert stat.S_IMODE(new.parent.stat().st_mode) == 0o700
    fw.remove_settings_file(new)
    assert not new.exists()


# --------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("line,expected", [
    ("[512/1024] Building C object esp-idf/main/CMakeFiles/main.c.obj", None),
    ("Writing at 0x00010000... (25 %)", ("flash", 25)),
    ("Writing at 0x00010000 [=====>         ]  32.4% 123/456 bytes", ("flash", 32)),
    ("Hash of data verified.", None),
    ("Leaving...", None),
])
def test_parse_progress(line, expected):
    assert fw.parse_progress(line) == expected


def test_failure_hints():
    assert any("busy" in h for h in fw.failure_hints(
        "serial.serialutil.SerialException: could not open port /dev/ttyACM0"))
    assert any("BOOT" in h for h in fw.failure_hints(
        "A fatal error occurred: Failed to connect to ESP32-S3: No serial data received."))
    assert any("different chip" in h for h in fw.failure_hints(
        "A fatal error occurred: This chip is ESP32-C3, not ESP32-S3. "
        "Wrong --chip argument?"))
    assert fw.failure_hints("all good") == []


# --------------------------------------------------------------------------
# Hotspot detection
# --------------------------------------------------------------------------

IP_ADDR = """1: lo    inet 127.0.0.1/8 scope host lo\\       valid_lft forever preferred_lft forever
2: enp3s0    inet 192.168.1.20/24 brd 192.168.1.255 scope global dynamic enp3s0\\       valid_lft 80000sec
5: wlp2s0u2    inet 10.42.1.1/24 brd 10.42.1.255 scope global noprefixroute wlp2s0u2\\       valid_lft forever
"""


def _fake_runner(responses):
    def run(argv):
        key = " ".join(argv)
        for prefix, out in responses.items():
            if key.startswith(prefix):
                return out
        return ""
    return run


def test_parse_ip_addr():
    assert fw.parse_ip_addr(IP_ADDR) == {
        "lo": "127.0.0.1", "enp3s0": "192.168.1.20", "wlp2s0u2": "10.42.1.1"}


def test_detect_hotspots_finds_ap_mode_connection():
    run = _fake_runner({
        "nmcli -t -f NAME,TYPE,DEVICE connection show --active":
            "Home\\:5G:802-11-wireless:wlp1s0\n"
            "Hotspot:802-11-wireless:wlp2s0u2\n"
            "Wired:802-3-ethernet:enp3s0\n",
        "ip -4 -o addr show": IP_ADDR,
        "nmcli -g 802-11-wireless.mode,802-11-wireless.ssid,802-11-wireless.channel connection show Home:5G":
            "infrastructure\nHomeNet\n0\n",
        "nmcli -g 802-11-wireless.mode,802-11-wireless.ssid,802-11-wireless.channel connection show Hotspot":
            "ap\nCSIDetector\n11\n",
    })
    hs = fw.detect_hotspots(run)
    assert len(hs) == 1
    assert (hs[0].connection, hs[0].iface, hs[0].ssid, hs[0].channel, hs[0].ip) == \
        ("Hotspot", "wlp2s0u2", "CSIDetector", 11, "10.42.1.1")


def test_detect_hotspots_auto_channel_falls_back_to_iw():
    run = _fake_runner({
        "nmcli -t": "Hotspot:802-11-wireless:wlp2s0u2\n",
        "ip -4": IP_ADDR,
        "nmcli -g": "ap\nCSIDetector\n0\n",
        "iw dev wlp2s0u2 info": "Interface wlp2s0u2\n\ttype AP\n\tchannel 6 (2437 MHz), width: 20 MHz\n",
    })
    assert fw.detect_hotspots(run)[0].channel == 6


def test_detect_hotspots_survives_missing_nmcli():
    def run(argv):
        raise FileNotFoundError("nmcli")
    assert fw.detect_hotspots(run) == []


def test_esptool_missing_hint():
    assert any("esptool isn't installed" in h for h in fw.failure_hints(
        "C:\\py\\python.exe: No module named esptool"))