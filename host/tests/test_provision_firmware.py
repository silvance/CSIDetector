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


def _kconfig_symbols(project: str) -> set[str]:
    text = (REPO_FIRMWARE / project / "main" / "Kconfig.projbuild").read_text()
    return set(re.findall(r"^config\s+(\w+)", text, re.MULTILINE))


# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------

def test_kconfig_symbols_exist_in_firmware():
    """Generated keys must match the firmware's Kconfig exactly —
    a typo here silently flashes the default value instead."""
    tx_keys = set(fw.TxConfig().kconfig())
    rx_keys = set(fw.RxConfig(ssid="CSIDetector").kconfig())
    assert tx_keys <= _kconfig_symbols("csi_transmitter")
    assert rx_keys <= _kconfig_symbols("csi_receiver")


def test_tx_config_validation():
    assert fw.TxConfig().validate() == []
    assert fw.TxConfig(channel=14).validate()
    assert fw.TxConfig(rate_hz=0).validate()


def test_rx_uart_only_leaves_hotspot_settings_unset():
    cfg = fw.RxConfig(tx_macs=["A0F262F039D8"])
    assert cfg.validate() == []
    k = cfg.kconfig()
    assert k["CSI_RX_WIFI_SSID"] == ""
    assert k["CSI_RX_FILTER_TX_MAC"] == "a0:f2:62:f0:39:d8"
    assert "CSI_RX_HOST_IP" not in k
    assert cfg.summary()["mode"] == "uart"


def test_rx_hotspot_config_and_summary_excludes_password():
    cfg = fw.RxConfig(tx_macs=["a0:f2:62:f0:39:d8", "ac:a7:04:2c:24:cc"],
                      ssid="CSIDetector", password="hunter22",
                      host_ip="10.42.1.1", host_port=5566)
    assert cfg.validate() == []
    k = cfg.kconfig()
    assert k["CSI_RX_FILTER_TX_MAC"] == "a0:f2:62:f0:39:d8, ac:a7:04:2c:24:cc"
    assert k["CSI_RX_HOST_IP"] == "10.42.1.1"
    assert "hunter22" not in repr(cfg.summary())


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
# sdkconfig rendering
# --------------------------------------------------------------------------

def test_render_sdkconfig_escapes_quotes_and_backslashes():
    text = fw.render_sdkconfig_defaults({"CSI_RX_WIFI_PASS": 'a"b\\c', "N": 5})
    assert 'CONFIG_CSI_RX_WIFI_PASS="a\\"b\\\\c"' in text
    assert "CONFIG_N=5" in text


def test_render_sdkconfig_rejects_newline_injection():
    with pytest.raises(ValueError):
        fw.render_sdkconfig_defaults(
            {"CSI_RX_WIFI_SSID": 'x"\nCONFIG_CSI_RX_HOST_IP="6.6.6.6'})


# --------------------------------------------------------------------------
# Flash job + build dir
# --------------------------------------------------------------------------

def _job(tmp_path, config, target="esp32s3", port="/dev/ttyACM0"):
    return fw.FlashJob(port=port, config=config, target=target,
                       firmware_root=REPO_FIRMWARE, build_root=tmp_path)


def test_flash_job_paths_are_per_project_and_target(tmp_path):
    rx = _job(tmp_path, fw.RxConfig())
    tx = _job(tmp_path, fw.TxConfig(), target="esp32c3")
    assert rx.build_dir == tmp_path / "csi_receiver-esp32s3"
    assert tx.build_dir == tmp_path / "csi_transmitter-esp32c3"
    assert rx.validate() == [] and tx.validate() == []


def test_flash_job_validate_catches_missing_project(tmp_path):
    job = fw.FlashJob(port="/dev/ttyACM0", config=fw.TxConfig(),
                      firmware_root=tmp_path / "nowhere", build_root=tmp_path)
    assert any("not found" in e for e in job.validate())


@POSIX_ONLY
def test_prepare_build_dir_reuses_build_when_config_unchanged(tmp_path):
    job = _job(tmp_path, fw.RxConfig(ssid="CSIDetector", password="hunter22"))
    assert fw.prepare_build_dir(job) is True
    mode = stat.S_IMODE(os.stat(job.overlay_path).st_mode)
    assert mode == 0o600, oct(mode)          # overlay may hold the password

    # Simulate idf.py having generated an sdkconfig from the overlay.
    job.sdkconfig_path.write_text("CONFIG_FROM_BUILD=y\n")
    assert fw.prepare_build_dir(job) is False
    assert job.sdkconfig_path.exists()       # reused → fast reflash

    job.config.channel = 6
    assert fw.prepare_build_dir(job) is True
    assert not job.sdkconfig_path.exists()   # forced regeneration
    assert "CONFIG_CSI_RX_CHANNEL=6" in job.overlay_path.read_text()


def test_prepare_build_dir_rebuilds_when_project_defaults_change(tmp_path):
    """A `git pull` that edits the project's sdkconfig.defaults must reach
    the cached build — kconfgen would otherwise keep the stale value."""
    root = tmp_path / "firmware"
    proj = root / "csi_transmitter"
    proj.mkdir(parents=True)
    (proj / "CMakeLists.txt").write_text("")
    (proj / "sdkconfig.defaults").write_text("CONFIG_A=y\n")
    job = fw.FlashJob(port="/dev/ttyACM0", config=fw.TxConfig(),
                      firmware_root=root, build_root=tmp_path / "cache")
    assert fw.prepare_build_dir(job) is True
    job.sdkconfig_path.write_text("generated\n")
    assert fw.prepare_build_dir(job) is False

    (proj / "sdkconfig.defaults").write_text("CONFIG_A=n\n")
    assert fw.prepare_build_dir(job) is True
    assert not job.sdkconfig_path.exists()

    # A target-specific defaults file appearing counts too.
    job.sdkconfig_path.write_text("generated\n")
    (proj / "sdkconfig.defaults.esp32s3").write_text("CONFIG_B=y\n")
    assert fw.prepare_build_dir(job) is True


@POSIX_ONLY
def test_build_dirs_are_private(tmp_path):
    """Generated sdkconfig / sdkconfig.h hold the hotspot password and are
    written 0644 by kconfgen, so the directories must block other users."""
    job = _job(tmp_path / "cache", fw.RxConfig(ssid="X", password="hunter22"))
    fw.prepare_build_dir(job)
    assert stat.S_IMODE(os.stat(job.build_dir).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(job.build_root).st_mode) == 0o700


@POSIX_ONLY
def test_flash_command_quotes_paths_and_keeps_secrets_out(tmp_path):
    export = tmp_path / "esp idf" / "export.sh"
    job = _job(tmp_path, fw.RxConfig(ssid="My Hotspot", password="hunter22"))
    argv = fw.flash_command(job, export)
    assert argv[:2] == ["bash", "-c"]
    script = argv[2]
    tokens = shlex.split(script)
    assert str(export) in tokens                       # space-safe
    assert "-p" in tokens and "/dev/ttyACM0" in tokens
    assert tokens[-1] == "flash"
    defaults = next(t for t in tokens if t.startswith("SDKCONFIG_DEFAULTS="))
    assert defaults.endswith(f";{job.overlay_path}")
    # Wi-Fi credentials go through the 0600 overlay, never argv.
    assert "hunter22" not in script and "My Hotspot" not in script


@POSIX_ONLY
def test_identify_command_uses_idf_python_esptool(tmp_path):
    argv = fw.identify_command("/dev/ttyACM0", tmp_path / "export.sh")
    tokens = shlex.split(argv[2])
    assert tokens[-6:] == ["python", "-m", "esptool", "--port", "/dev/ttyACM0",
                           "read_mac"]


@POSIX_ONLY
def test_idf_env_strips_active_venv():
    base = {
        "VIRTUAL_ENV": "/home/u/proj/.venv",
        "PATH": "/home/u/proj/.venv/bin:/usr/local/bin:/usr/bin",
        "PYTHONHOME": "/x",
        "HOME": "/home/u",
    }
    env = fw.idf_env(base)
    assert "VIRTUAL_ENV" not in env and "PYTHONHOME" not in env
    assert env["PATH"] == "/usr/local/bin:/usr/bin"
    assert env["HOME"] == "/home/u"


def test_find_idf_export_priority(tmp_path):
    explicit = tmp_path / "a" / "export.sh"
    idf_path = tmp_path / "b"
    explicit.parent.mkdir()
    explicit.write_text("")
    idf_path.mkdir()
    (idf_path / "export.sh").write_text("")
    assert fw.find_idf_export({"IDF_EXPORT": str(explicit),
                               "IDF_PATH": str(idf_path)}) == explicit
    assert fw.find_idf_export({"IDF_PATH": str(idf_path)}) == idf_path / "export.sh"


# --------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("line,expected", [
    ("[512/1024] Building C object esp-idf/main/CMakeFiles/main.c.obj", ("build", 50)),
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
    assert any("export.sh" in h for h in fw.failure_hints(
        "ESP-IDF export.sh failed to load"))
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


def test_find_idf_export_windows_installer_layout(tmp_path):
    root = tmp_path / "Espressif" / "frameworks"
    for v in ("esp-idf-v5.1.2", "esp-idf-v5.2.1", "esp-idf-v5.10.0"):
        (root / v).mkdir(parents=True)
        (root / v / "export.bat").write_text("")
    env = {"IDF_TOOLS_PATH": str(tmp_path / "Espressif"), "SystemDrive": str(tmp_path / "nowhere")}
    assert fw.find_idf_export(env, windows=True) == root / "esp-idf-v5.10.0" / "export.bat"


def test_paths_with_spaces_are_refused(tmp_path):
    job = _job(tmp_path / "build dir", fw.TxConfig())
    assert any("path with spaces" in e for e in job.validate())


def test_env_dump_parsing():
    env = {"IDF_PATH": "/opt/idf", "Path": "/bin", "TEMP": "C:\\Users\\José\\Temp"}
    dump = json.dumps({"env": env, "python": "/opt/idf-env/python.exe"})   # ASCII-only
    out = f"Activating ESP-IDF 5.5\nnoise=ignored\n{fw.ENV_MARKER}{dump}\n"
    idf = fw.parse_env_dump(out)
    assert idf.idf_py == Path("/opt/idf/tools/idf.py") and idf.python == "/opt/idf-env/python.exe"
    assert idf.env["TEMP"].endswith("José\\Temp") and "noise" not in idf.env
    assert fw.parse_env_dump("export.bat failed") is None
    # export.bat "succeeded" but IDF's python wasn't first on PATH: the dump
    # (which imports esptool) never ran, so there's no marker.
    assert fw.parse_env_dump("Activating ESP-IDF\nNo module named esptool\n") is None
    no_idf = json.dumps({"env": {"Path": "/bin"}, "python": "x"})
    assert fw.parse_env_dump(f"{fw.ENV_MARKER}{no_idf}\n") is None


def test_windows_export_env_for_the_installer_layout(tmp_path):
    root = tmp_path / "Espressif"
    export = root / "frameworks" / "esp-idf-v5.4.1" / "export.bat"
    export.parent.mkdir(parents=True)
    (root / "tools" / "idf-python" / "3.11.2").mkdir(parents=True)
    (root / "tools" / "idf-git" / "2.44.0" / "cmd").mkdir(parents=True)
    env = fw.windows_export_env(export, {"Path": "/usr/bin", "MSYSTEM": "MINGW64"})
    assert "MSYSTEM" not in env                      # export.bat refuses to run with it
    assert env["IDF_TOOLS_PATH"] == str(root)
    parts = env["Path"].split(os.pathsep)
    assert parts[0].endswith("3.11.2") and parts[1].endswith("cmd") and parts[-1] == "/usr/bin"
    other = fw.windows_export_env(tmp_path / "esp-idf" / "export.bat", {"PATH": "/bin"})
    assert other == {"PATH": "/bin"}                 # not the installer layout: as is


def test_esptool_missing_hint():
    assert any("environment didn't load" in h for h in fw.failure_hints(
        "C:\\py\\python.exe: No module named esptool"))