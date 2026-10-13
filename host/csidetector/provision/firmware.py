"""Firmware images, per-board settings, esptool commands and output parsing.

The firmware is prebuilt once per chip (firmware/prebuilt/, built by
scripts/build_firmware.py in CI) and flashed with esptool — a pip
package installed with the app — so setting up boards needs no ESP-IDF
or compiler. What differs per board (channel, transmitter MAC filter,
hotspot SSID/password, label) goes into the board's csi_cfg partition as
a settings block (board_settings.py), written in the same esptool run as
the firmware.

esptool runs as ``python -m esptool`` with this app's own Python.
"""

from __future__ import annotations

import dataclasses
import hashlib
import ipaddress
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional, Union

from csidetector.provision import board_settings
from csidetector.provision.devices import ROLE_RX, ROLE_TX, normalize_mac

# Firmware MAC filter capacity — see MAX_FILTER_MACS in
# firmware/csi_receiver/main/main.c.
MAX_FILTER_MACS = 4
DEFAULT_TARGET = "esp32s3"
# Chips the prebuilt firmware is built for (scripts/build_firmware.py).
TARGETS = ("esp32s3", "esp32c3", "esp32c5", "esp32c6", "esp32")
# Chips with a USB-Serial/JTAG controller. The receiver's
# sdkconfig.defaults routes its console (and the CSI stream) there, so on
# these chips UART-mode CSI arrives on the native USB port only.
USB_JTAG_TARGETS = frozenset({"esp32s3", "esp32c3", "esp32c5", "esp32c6"})
FLASH_BAUD = 460800
MANIFEST_FORMAT = "csidetector-firmware/1"

PROJECTS = {
    ROLE_TX: "csi_transmitter",
    ROLE_RX: "csi_receiver",
}

IS_WINDOWS = os.name == "nt"
# The default Windows Mobile hotspot address.
WINDOWS_HOTSPOT_IP = "192.168.137.1"


def local_ipv4_addresses() -> list[str]:
    """This computer's IPv4 addresses (not loopback), for picking the host
    IP when the hotspot can't be read automatically (Windows, macOS)."""
    import socket
    found: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    except OSError:
        pass
    try:   # the address of the default route (no packet is sent)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))
            found.append(sock.getsockname()[0])
    except OSError:
        pass
    out = []
    for ip in found:
        if not ip.startswith("127.") and ip not in out:
            out.append(ip)
    return out


def repo_firmware_root() -> Path:
    # host/csidetector/provision/firmware.py → repo root is parents[3].
    return Path(__file__).resolve().parents[3] / "firmware"


def prebuilt_root() -> Path:
    """The prebuilt firmware: firmware/prebuilt, or $CSIDETECTOR_FIRMWARE_DIR
    (a developer's own scripts/build_firmware.py --out DIR)."""
    if os.environ.get("CSIDETECTOR_FIRMWARE_DIR"):
        return Path(os.environ["CSIDETECTOR_FIRMWARE_DIR"])
    return repo_firmware_root() / "prebuilt"


def private_dir() -> Path:
    """Owner-only folder for the settings files esptool writes to boards
    (they can hold the hotspot password). On Windows %LOCALAPPDATA%, which
    is private to the user."""
    if os.environ.get("XDG_CACHE_HOME"):
        base = Path(os.environ["XDG_CACHE_HOME"]) / "csidetector"
    elif IS_WINDOWS and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"]) / "csidetector"
    else:
        base = Path(os.path.expanduser("~/.cache")) / "csidetector"
    path = base / "board-settings"
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------

def _has_control_chars(s: str) -> bool:
    return any(ord(c) < 0x20 or ord(c) == 0x7F for c in s)


def _validate_label(label: str) -> list[str]:
    if _has_control_chars(label or ""):
        return ["The label contains control characters."]
    return []


@dataclasses.dataclass
class TxConfig:
    channel: int = 11
    rate_hz: int = 100
    label: str = ""              # stored on the board (see fit_label)

    role = ROLE_TX

    def validate(self) -> list[str]:
        errors = _validate_label(self.label)
        if not 1 <= int(self.channel) <= 13:
            errors.append("Channel must be between 1 and 13.")
        if not 1 <= int(self.rate_hz) <= 500:
            errors.append("Broadcast rate must be between 1 and 500 Hz.")
        return errors

    def settings(self) -> dict[str, Union[int, str]]:
        """What the board's settings block holds."""
        return {"role": ROLE_TX, "label": board_settings.fit_label(self.label),
                "channel": int(self.channel), "rate_hz": int(self.rate_hz)}

    def summary(self) -> dict:
        return {"channel": int(self.channel), "rate_hz": int(self.rate_hz)}


@dataclasses.dataclass
class RxConfig:
    channel: int = 11
    tx_macs: list[str] = dataclasses.field(default_factory=list)
    ssid: str = ""               # "" → UART-only (alert mode)
    password: str = ""
    host_ip: str = "10.42.0.1"
    host_port: int = 5566
    label: str = ""

    role = ROLE_RX

    @property
    def uart_only(self) -> bool:
        return not self.ssid

    def validate(self) -> list[str]:
        errors = _validate_label(self.label)
        if not 1 <= int(self.channel) <= 13:
            errors.append("Channel must be between 1 and 13.")
        if len(self.tx_macs) > MAX_FILTER_MACS:
            errors.append(f"The receiver accepts at most {MAX_FILTER_MACS} "
                          f"transmitter MACs (got {len(self.tx_macs)}).")
        for mac in self.tx_macs:
            try:
                normalize_mac(mac)
            except ValueError:
                errors.append(f"Not a MAC address: {mac!r}")
        if self.ssid:
            if _has_control_chars(self.ssid) or len(self.ssid.encode()) > 32:
                errors.append("SSID must be at most 32 bytes with no "
                              "control characters.")
            if self.password and not 8 <= len(self.password.encode()) <= 64:
                errors.append("Wi-Fi password must be empty (open hotspot) "
                              "or 8–64 characters.")
            if _has_control_chars(self.password):
                errors.append("Wi-Fi password contains control characters.")
            elif len(self.password) == 64 and not re.fullmatch(r"[0-9A-Fa-f]{64}",
                                                               self.password):
                errors.append("A 64-character Wi-Fi key must be hexadecimal (0-9, "
                              "a-f); a passphrase is 8–63 characters.")
            try:
                ipaddress.IPv4Address(self.host_ip)
            except (ipaddress.AddressValueError, ValueError):
                errors.append(f"Host IP is not an IPv4 address: "
                              f"{self.host_ip!r}")
            if not 1 <= int(self.host_port) <= 65535:
                errors.append("Host UDP port must be between 1 and 65535.")
        return errors

    def settings(self) -> dict[str, Union[int, str]]:
        """What the board's settings block holds (the password included:
        it never leaves the settings file esptool writes)."""
        values: dict[str, Union[int, str]] = {
            "role": ROLE_RX, "label": board_settings.fit_label(self.label),
            "channel": int(self.channel),
            "tx_macs": ", ".join(normalize_mac(m) for m in self.tx_macs),
            "ssid": self.ssid,
            "password": self.password if self.ssid else "",
        }
        if self.ssid:
            values["host_ip"] = self.host_ip
            values["host_port"] = int(self.host_port)
        return values

    def summary(self) -> dict:
        """Recordable config — deliberately excludes the password."""
        out = {
            "channel": int(self.channel),
            "tx_macs": [normalize_mac(m) for m in self.tx_macs],
            "mode": "uart" if self.uart_only else "hotspot",
        }
        if self.ssid:
            out.update(ssid=self.ssid, host_ip=self.host_ip,
                       host_port=int(self.host_port))
        return out


Config = Union[TxConfig, RxConfig]


# --------------------------------------------------------------------------
# Prebuilt firmware
# --------------------------------------------------------------------------

class FirmwareError(ValueError):
    """The prebuilt firmware is missing, damaged, or not built for a chip.
    The message is for the user."""


@dataclasses.dataclass
class FirmwareImage:
    project: str
    target: str
    chip: str                                  # esptool --chip
    flash_mode: str
    flash_freq: str
    flash_size: str
    files: list[tuple[int, Path, str]]         # (offset, path, sha256)
    settings_offset: int
    settings_size: int
    nvs_offset: int = 0                        # blanked on every flash (0: not)
    nvs_size: int = 0


def chip_name(target: str) -> str:
    """esp32c5 → ESP32-C5."""
    return "ESP32" + ("-" + target[5:].upper() if len(target) > 5 else "")


def supported_chips() -> str:
    return ", ".join(chip_name(t) for t in TARGETS)


def load_manifest(root: Optional[Path] = None) -> dict:
    root = Path(root) if root else prebuilt_root()
    path = root / "manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError:
        raise FirmwareError(
            f"The prebuilt firmware is missing ({path}). Download CSIDetector "
            f"again, or get the firmware/prebuilt folder from the repository.") from None
    except (OSError, ValueError) as exc:
        raise FirmwareError(f"The prebuilt firmware's manifest ({path}) can't be "
                            f"read: {exc}") from None
    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        raise FirmwareError(f"{path} isn't a firmware manifest this version of the "
                            f"setup app understands. Update CSIDetector.")
    return manifest


def available_targets(project: str, root: Optional[Path] = None) -> list[str]:
    try:
        images = load_manifest(root).get("images", {})
    except FirmwareError:
        return []
    return [t for t in TARGETS if f"{project}-{t}" in images]


def find_image(project: str, target: str, root: Optional[Path] = None) -> FirmwareImage:
    """The prebuilt image for (project, target). Raises FirmwareError."""
    root = Path(root) if root else prebuilt_root()
    manifest = load_manifest(root)
    entry = manifest.get("images", {}).get(f"{project}-{target}")
    if not isinstance(entry, dict):
        built = sorted({chip_name(v.get("target", "")) for v in
                        manifest.get("images", {}).values() if isinstance(v, dict)})
        raise FirmwareError(f"There's no prebuilt {project} firmware for the "
                            f"{chip_name(target)} (built for: {', '.join(built) or 'none'}).")
    try:
        return FirmwareImage(
            project=project, target=target, chip=str(entry["chip"]),
            flash_mode=str(entry["flash_mode"]), flash_freq=str(entry["flash_freq"]),
            flash_size=str(entry["flash_size"]),
            files=[(int(f["offset"], 0), root / f["path"], str(f["sha256"]))
                   for f in entry["files"]],
            settings_offset=int(entry["settings"]["offset"], 0),
            settings_size=int(entry["settings"]["size"]),
            nvs_offset=int(entry.get("nvs", {}).get("offset", "0"), 0),
            nvs_size=int(entry.get("nvs", {}).get("size", 0)))
    except (KeyError, TypeError, ValueError) as exc:
        raise FirmwareError(f"The firmware manifest's entry for {project}-{target} is "
                            f"incomplete ({exc}). Update CSIDetector.") from None


def check_image(image: FirmwareImage) -> None:
    """Raise FirmwareError unless every file is there and intact (a partial
    download, or a git checkout that mangled the binaries)."""
    for _, path, digest in image.files:
        try:
            data = path.read_bytes()
        except OSError:
            raise FirmwareError(f"Firmware file missing: {path}. Download "
                                f"CSIDetector again.") from None
        if hashlib.sha256(data).hexdigest() != digest:
            raise FirmwareError(f"Firmware file damaged: {path} (checksum mismatch). "
                                f"Download CSIDetector again.")


# --------------------------------------------------------------------------
# Flash job
# --------------------------------------------------------------------------

@dataclasses.dataclass
class FlashJob:
    port: str
    config: Config
    target: str = DEFAULT_TARGET
    firmware_root: Path = dataclasses.field(default_factory=prebuilt_root)

    @property
    def role(self) -> str:
        return self.config.role

    @property
    def project(self) -> str:
        return PROJECTS[self.role]

    def image(self) -> FirmwareImage:
        return find_image(self.project, self.target, self.firmware_root)

    def validate(self) -> list[str]:
        errors = list(self.config.validate())
        if self.target not in TARGETS:
            errors.append(f"{chip_name(self.target)} boards aren't supported "
                          f"(supported: {supported_chips()}).")
        else:
            try:
                self.image()
            except FirmwareError as exc:
                errors.append(str(exc))
        if not self.port:
            errors.append("No serial port selected.")
        return errors


_STALE_SETTINGS_S = 600


def write_settings_file(job: FlashJob, image: FirmwareImage) -> Path:
    """The board's settings block, in a new owner-only file for esptool to
    write. The caller deletes it once esptool is done; leftovers from a
    crash are removed here, next time."""
    blob = board_settings.encode(job.config.settings(), image.settings_size)
    folder = private_dir()
    clear_stale_settings_files(folder)
    fd, name = tempfile.mkstemp(prefix="settings-", suffix=".bin", dir=folder)
    with os.fdopen(fd, "wb") as f:
        f.write(blob)
    if not IS_WINDOWS:
        os.chmod(name, 0o600)
    return Path(name)


def blank_file(size: int) -> Path:
    """``size`` bytes of erased flash (0xFF), for blanking a region."""
    path = private_dir() / f"blank-{size:x}.bin"
    if not path.is_file() or path.stat().st_size != size:
        path.write_bytes(b"\xff" * size)
    return path


def clear_stale_settings_files(folder: Optional[Path] = None) -> None:
    """Remove settings files a crash left behind (older than a flash takes)."""
    folder = folder or private_dir()
    now = time.time()
    for old in folder.glob("settings-*.bin"):
        try:
            if now - old.stat().st_mtime > _STALE_SETTINGS_S:
                old.unlink()
        except OSError:
            pass


def remove_settings_file(path: Optional[Path]) -> None:
    if path is not None:
        try:
            Path(path).unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------
# esptool
# --------------------------------------------------------------------------

def esptool_env() -> dict:
    """The environment for esptool: this app's, with output unbuffered (so
    progress streams) and UTF-8 (a Windows pipe defaults to the ANSI code
    page)."""
    return dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")


def _esptool() -> list[str]:
    return [sys.executable, "-m", "esptool"]


def flash_command(job: FlashJob, image: FirmwareImage, settings_file: Path) -> list[str]:
    """esptool write-flash: the bootloader, partition table, app and the
    board's settings block in one go, with NVS blanked. ``--chip`` makes
    esptool refuse a board that isn't the chip the image is for."""
    pairs = [(offset, path) for offset, path, _ in image.files]
    pairs.append((image.settings_offset, settings_file))
    if image.nvs_size:
        pairs.append((image.nvs_offset, blank_file(image.nvs_size)))
    argv = [*_esptool(), "--chip", image.chip, "--port", job.port,
            "--baud", str(FLASH_BAUD), "--before", "default-reset",
            "--after", "hard-reset", "write-flash",
            "--flash-mode", image.flash_mode, "--flash-freq", image.flash_freq,
            "--flash-size", image.flash_size]
    for offset, path in sorted(pairs):
        argv += [hex(offset), str(path)]
    return argv


def identify_command(port: str) -> list[str]:
    """esptool read-mac: prints the chip type and the factory MAC."""
    return [*_esptool(), "--port", port, "read-mac"]


def descendant_pids(root: int, proc_dir: str = "/proc") -> list[int]:
    """All transitive children of ``root`` (Linux /proc walk).

    Cancelling a flash must reach esptool, which runs as a grandchild of
    idf.py (idf.py → ninja → esptool). Killing only the top process
    leaves esptool holding the serial port. Returns [] where /proc is
    unavailable (macOS), in which case only the top process is killed.
    """
    children: dict[int, list[int]] = {}
    try:
        entries = os.listdir(proc_dir)
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(os.path.join(proc_dir, entry, "stat")) as f:
                data = f.read()
        except OSError:
            continue
        # "pid (comm) state ppid ..." — comm may contain spaces and ')'.
        fields = data[data.rfind(")") + 2:].split()
        if len(fields) < 2:
            continue
        try:
            children.setdefault(int(fields[1]), []).append(int(entry))
        except ValueError:
            continue
    out: list[int] = []
    stack = [root]
    while stack:
        for child in children.get(stack.pop(), []):
            out.append(child)
            stack.append(child)
    return out


def terminate_tree(root: int, sig: int = 15) -> None:
    """Send ``sig`` to every descendant of ``root``, deepest first. On
    Windows: taskkill /T /F, which ends the whole tree including ``root``."""
    if IS_WINDOWS:
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(root)],
                           capture_output=True, timeout=10,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            pass
        return
    for pid in reversed(descendant_pids(root)):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


# --------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------

_FLASH_PCT_RE = re.compile(
    r"Writing at 0x[0-9a-fA-F]+.*?(\d{1,3}(?:\.\d+)?)\s*%")


def parse_progress(line: str) -> Optional[tuple[str, int]]:
    """("flash", pct) for esptool's write progress (per file written)."""
    m = _FLASH_PCT_RE.search(line)
    if m:
        return "flash", min(100, int(float(m.group(1))))
    return None


_HINTS = [
    (re.compile(r"could not open port|Permission denied|Device or resource busy"
                r"|multiple access on port|Access is denied", re.I),
     "The serial port is busy, unplugged, or not permitted. Close anything "
     "else using it (the alert monitor, a serial monitor such as PuTTY or "
     "the Arduino IDE, the heatmap) and check the cable."
     + ("" if IS_WINDOWS else " On Linux your user also needs to be in the "
        "`dialout` group.")),
    (re.compile(r"Failed to connect to (?:ESP32|Espressif)|No serial data received|"
                r"Wrong boot mode detected", re.I),
     "The chip didn't enter download mode. Hold BOOT, tap RESET (or "
     "unplug and replug while holding BOOT), release BOOT, then retry."),
    (re.compile(r"No module named '?esptool'?", re.I),
     "esptool isn't installed for this Python. "
     + ("Run setup-windows.bat again (it installs it)." if IS_WINDOWS else
        "Run: python -m pip install -r requirements.txt (in the host folder).")),
    (re.compile(r"This chip is ESP32[-\w]*,? not ESP32|Wrong (?:--)?chip argument", re.I),
     "The board is a different chip than the one selected (was it "
     "swapped?). Click Identify, or just Flash again: an unidentified "
     "board is identified first."),
]


WRONG_CHIP_RE = re.compile(r"This chip is ESP32[-\w]*,? not ESP32", re.I)


def failure_hints(output: str) -> list[str]:
    return [hint for rx, hint in _HINTS if rx.search(output or "")]


# --------------------------------------------------------------------------
# Hotspot auto-detection (NetworkManager)
# --------------------------------------------------------------------------

@dataclasses.dataclass
class HotspotInfo:
    connection: str
    iface: str
    ssid: str = ""
    channel: int = 0
    ip: str = ""


_IP_ADDR_RE = re.compile(r"^\d+:\s+(\S+)\s+inet\s+(\d+\.\d+\.\d+\.\d+)/\d+",
                         re.MULTILINE)


def parse_ip_addr(output: str) -> dict[str, str]:
    """``ip -4 -o addr show`` → {iface: first IPv4}."""
    out: dict[str, str] = {}
    for iface, ip in _IP_ADDR_RE.findall(output or ""):
        out.setdefault(iface, ip)
    return out


def _split_nmcli(line: str) -> list[str]:
    # nmcli -t escapes ':' and '\' inside fields with a backslash.
    parts, cur, esc = [], [], False
    for ch in line:
        if esc:
            cur.append(ch)
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == ":":
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def _nmcli_unescape(value: str) -> str:
    return re.sub(r"\\(.)", r"\1", value)


_IW_CHANNEL_RE = re.compile(r"^\s*channel\s+(\d+)\b", re.MULTILINE)


def parse_iw_channel(output: str) -> int:
    """``iw dev <iface> info`` → operating channel, 0 if absent."""
    m = _IW_CHANNEL_RE.search(output or "")
    return int(m.group(1)) if m else 0


Runner = Callable[[list[str]], str]


def _default_runner(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True,
                          timeout=5, check=False).stdout


def detect_hotspots(run: Runner = _default_runner) -> list[HotspotInfo]:
    """Active NetworkManager Wi-Fi connections in AP mode, with SSID,
    channel, and the host's IP on that interface — exactly the values a
    localize-mode receiver must be flashed with."""
    try:
        active = run(["nmcli", "-t", "-f", "NAME,TYPE,DEVICE",
                      "connection", "show", "--active"])
        addrs = parse_ip_addr(run(["ip", "-4", "-o", "addr", "show"]))
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in active.splitlines():
        fields = _split_nmcli(line)
        if len(fields) < 3 or fields[1] != "802-11-wireless":
            continue
        name, iface = fields[0], fields[2]
        try:
            props = run(["nmcli", "-g",
                         "802-11-wireless.mode,802-11-wireless.ssid,"
                         "802-11-wireless.channel",
                         "connection", "show", name]).splitlines()
        except (OSError, subprocess.SubprocessError):
            continue
        if len(props) < 3 or props[0].strip() != "ap":
            continue
        try:
            channel = int(props[2].strip() or 0)
        except ValueError:
            channel = 0
        if channel == 0:
            # Channel "auto" in the profile: ask the radio what it chose.
            try:
                channel = parse_iw_channel(run(["iw", "dev", iface, "info"]))
            except (OSError, subprocess.SubprocessError):
                channel = 0
        found.append(HotspotInfo(connection=name, iface=iface,
                                 ssid=_nmcli_unescape(props[1].strip()),
                                 channel=channel,
                                 ip=addrs.get(iface, "")))
    return found
