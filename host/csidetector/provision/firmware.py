"""Firmware configuration, flash-command construction, and output parsing.

Flashing never touches the developer's hand-edited ``sdkconfig`` in the
firmware project. Each (project, target) pair gets its own build
directory under ``~/.cache/csidetector/firmware/`` with a generated
``sdkconfig.setup`` defaults overlay; ``idf.py`` is pointed at that
directory with ``-B`` / ``-D SDKCONFIG`` / ``-D SDKCONFIG_DEFAULTS``.
Reflashing a second board with an identical config reuses the build, so
only the first flash pays the full compile.

On Linux/macOS, ESP-IDF commands run through ``bash -c '. export.sh &&
…'``. On Windows there's no bash: ``export.bat`` is run once through
``cmd.exe`` to capture the environment it sets up (cached per export.bat),
then ``python idf.py …`` / ``python -m esptool …`` run directly in it — no
cmd quoting rules in the way. Either way the host Python venv is stripped
first — otherwise the venv's python shadows IDF's and ``idf.py`` dies
with ``No module named 'click'``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import ipaddress
import os
import glob
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Union

from csidetector.provision.devices import ROLE_RX, ROLE_TX, normalize_mac

# Firmware MAC filter capacity — see MAX_FILTER_MACS in
# firmware/csi_receiver/main/main.c.
MAX_FILTER_MACS = 4
DEFAULT_TARGET = "esp32s3"
TARGETS = ("esp32s3", "esp32c3", "esp32c5", "esp32s2", "esp32")
# Chips with a USB-Serial/JTAG controller. The receiver's
# sdkconfig.defaults routes its console (and the CSI stream) there, so on
# these chips UART-mode CSI arrives on the native USB port only.
USB_JTAG_TARGETS = frozenset({"esp32s3", "esp32c3", "esp32c5"})

PROJECTS = {
    ROLE_TX: "csi_transmitter",
    ROLE_RX: "csi_receiver",
}

IS_WINDOWS = os.name == "nt"
EXPORT_NAME = "export.bat" if IS_WINDOWS else "export.sh"


def repo_firmware_root() -> Path:
    # host/csidetector/provision/firmware.py → repo root is parents[3].
    return Path(__file__).resolve().parents[3] / "firmware"


def cache_root() -> Path:
    """Where builds go. On Windows: %LOCALAPPDATA% (private to the user —
    generated sdkconfigs hold the hotspot password), short names, and no
    spaces (ESP-IDF can't build in a path with spaces; a user name can
    have them)."""
    if os.environ.get("CSIDETECTOR_BUILD_DIR"):
        return Path(os.environ["CSIDETECTOR_BUILD_DIR"])
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]) / "csidetector" / "firmware"
    if IS_WINDOWS and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"])
        if " " in str(base):
            base = Path(_short_path(str(base)))
        return base / "csid" / "fw"
    return Path(os.path.expanduser("~/.cache")) / "csidetector" / "firmware"


def _short_path(path: str) -> str:
    """Windows 8.3 short form of an existing path (no spaces), else as is."""
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf)):
            return buf.value
    except (AttributeError, OSError):
        pass
    return path


# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------

def _has_control_chars(s: str) -> bool:
    return any(ord(c) < 0x20 or ord(c) == 0x7F for c in s)


@dataclasses.dataclass
class TxConfig:
    channel: int = 11
    rate_hz: int = 100

    role = ROLE_TX

    def validate(self) -> list[str]:
        errors = []
        if not 1 <= int(self.channel) <= 13:
            errors.append("Channel must be between 1 and 13.")
        if not 1 <= int(self.rate_hz) <= 500:
            errors.append("Broadcast rate must be between 1 and 500 Hz.")
        return errors

    def kconfig(self) -> dict[str, Union[int, str]]:
        return {
            "CSI_TX_CHANNEL": int(self.channel),
            "CSI_TX_RATE_HZ": int(self.rate_hz),
        }

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

    role = ROLE_RX

    @property
    def uart_only(self) -> bool:
        return not self.ssid

    def validate(self) -> list[str]:
        errors = []
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
            if self.password and not 8 <= len(self.password) <= 64:
                errors.append("Wi-Fi password must be empty (open hotspot) "
                              "or 8–64 characters.")
            if _has_control_chars(self.password):
                errors.append("Wi-Fi password contains control characters.")
            try:
                ipaddress.IPv4Address(self.host_ip)
            except (ipaddress.AddressValueError, ValueError):
                errors.append(f"Host IP is not an IPv4 address: "
                              f"{self.host_ip!r}")
            if not 1 <= int(self.host_port) <= 65535:
                errors.append("Host UDP port must be between 1 and 65535.")
        return errors

    def kconfig(self) -> dict[str, Union[int, str]]:
        macs = ", ".join(normalize_mac(m) for m in self.tx_macs)
        values: dict[str, Union[int, str]] = {
            "CSI_RX_CHANNEL": int(self.channel),
            "CSI_RX_FILTER_TX_MAC": macs,
            "CSI_RX_WIFI_SSID": self.ssid,
            "CSI_RX_WIFI_PASS": self.password if self.ssid else "",
        }
        if self.ssid:
            values["CSI_RX_HOST_IP"] = self.host_ip
            values["CSI_RX_HOST_PORT"] = int(self.host_port)
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


def kconfig_escape(value: str) -> str:
    """Escape a string for a ``CONFIG_X="..."`` line in sdkconfig."""
    if _has_control_chars(value):
        raise ValueError("config strings may not contain control characters")
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_sdkconfig_defaults(values: dict[str, Union[int, str]]) -> str:
    lines = ["# Generated by `python run.py setup` — do not edit by hand."]
    for key, value in values.items():
        if isinstance(value, bool):
            raise TypeError(f"{key}: bool values are not supported")
        if isinstance(value, int):
            lines.append(f"CONFIG_{key}={value}")
        else:
            lines.append(f'CONFIG_{key}="{kconfig_escape(value)}"')
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Flash job
# --------------------------------------------------------------------------

@dataclasses.dataclass
class FlashJob:
    port: str
    config: Config
    target: str = DEFAULT_TARGET
    firmware_root: Path = dataclasses.field(default_factory=repo_firmware_root)
    build_root: Path = dataclasses.field(default_factory=cache_root)

    @property
    def role(self) -> str:
        return self.config.role

    @property
    def project(self) -> str:
        return PROJECTS[self.role]

    @property
    def project_dir(self) -> Path:
        return self.firmware_root / self.project

    @property
    def build_dir(self) -> Path:
        # Per-target build dir: idf.py refuses to reuse a build dir that
        # was configured for a different chip.
        return self.build_root / f"{self.project}-{self.target}"

    @property
    def sdkconfig_path(self) -> Path:
        return self.build_dir / "sdkconfig"

    @property
    def overlay_path(self) -> Path:
        return self.build_dir / "sdkconfig.setup"

    def validate(self) -> list[str]:
        errors = list(self.config.validate())
        if self.target not in TARGETS:
            errors.append(f"Unsupported target: {self.target}")
        if not self.port:
            errors.append("No serial port selected.")
        if not (self.project_dir / "CMakeLists.txt").is_file():
            errors.append(f"Firmware project not found: {self.project_dir}")
        for what, path in (("The CSIDetector folder", self.firmware_root.parent),
                           ("The build folder", self.build_root)):
            if " " in str(path):
                errors.append(
                    f"{what} is in a path with spaces ({path}), which ESP-IDF "
                    f"can't build in. Move it to a path without spaces, e.g. "
                    + (r"C:\csidetector" if IS_WINDOWS else "~/csidetector")
                    + (" (or set CSIDETECTOR_BUILD_DIR)." if what.startswith("The build")
                       else "."))
        return errors


def _base_defaults_fingerprint(job: FlashJob) -> str:
    """Hash of the project's own defaults files that idf.py will load.

    kconfgen lets an existing sdkconfig win over defaults, so a `git pull`
    that changes the project's sdkconfig.defaults would otherwise never
    reach a cached build.
    """
    h = hashlib.sha256()
    for name in ("sdkconfig.defaults", f"sdkconfig.defaults.{job.target}"):
        path = job.project_dir / name
        h.update(name.encode() + b"\0")
        try:
            h.update(path.read_bytes())
        except OSError:
            h.update(b"<absent>")
    return h.hexdigest()


def _private_dir(path: Path) -> None:
    """mkdir -p, then 0700: generated sdkconfig / sdkconfig.h files under
    the build dir carry the hotspot password and are written 0644."""
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def prepare_build_dir(job: FlashJob) -> bool:
    """Write the sdkconfig overlay; return True when it changed.

    SDKCONFIG_DEFAULTS only seeds values missing from an existing
    sdkconfig, so when the overlay or the project's own defaults change we
    delete the generated sdkconfig to force the new values in. When
    neither changed the previous build is reused and only the flash step
    runs.
    """
    content = (f"# base-defaults-sha256: {_base_defaults_fingerprint(job)}\n"
               + render_sdkconfig_defaults(job.config.kconfig()))
    _private_dir(job.build_root)
    _private_dir(job.build_dir)
    try:
        previous = job.overlay_path.read_text()
    except OSError:
        previous = None
    if previous == content and job.sdkconfig_path.exists():
        return False
    # The overlay may hold the hotspot password: owner-only.
    fd = os.open(job.overlay_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    os.chmod(job.overlay_path, 0o600)
    try:
        job.sdkconfig_path.unlink()
    except FileNotFoundError:
        pass
    return True


# --------------------------------------------------------------------------
# ESP-IDF shell
# --------------------------------------------------------------------------

def find_idf_export(env: Optional[dict] = None,
                    windows: bool = IS_WINDOWS) -> Optional[Path]:
    """Locate ESP-IDF's export script ($IDF_EXPORT, $IDF_PATH,
    ~/esp/esp-idf; on Windows also the ESP-IDF installer's
    C:\\Espressif\\frameworks\\esp-idf-v*, newest first)."""
    env = os.environ if env is None else env
    name = "export.bat" if windows else "export.sh"
    candidates = []
    if env.get("IDF_EXPORT"):
        candidates.append(Path(env["IDF_EXPORT"]).expanduser())
    if env.get("IDF_PATH"):
        candidates.append(Path(env["IDF_PATH"]).expanduser() / name)
    candidates.append(Path(os.path.expanduser("~")) / "esp" / "esp-idf" / name)
    if windows:
        drive = env.get("SystemDrive", "C:")
        for root in (env.get("IDF_TOOLS_PATH"), f"{drive}\\Espressif"):
            if root:
                found = sorted(glob.glob(os.path.join(root, "frameworks",
                                                      "esp-idf-v*", name)),
                               key=_version_key, reverse=True)
                candidates += [Path(f) for f in found]
        candidates.append(Path(f"{drive}\\esp\\esp-idf") / name)
    for c in candidates:
        if c.is_file():
            return c
    return None


def _version_key(path: str) -> tuple:
    m = re.search(r"esp-idf-v(\d+)\.(\d+)(?:\.(\d+))?", path)
    return tuple(int(g or 0) for g in m.groups()) if m else (0,)


def idf_env(base: Optional[dict] = None) -> dict:
    """Copy of the environment with any Python venv removed from it."""
    env = dict(os.environ if base is None else base)
    strip = set()
    scripts = "Scripts" if IS_WINDOWS else "bin"
    venv = env.pop("VIRTUAL_ENV", None)
    if venv:
        strip.add(os.path.join(venv, scripts))
    if base is None and sys.prefix != sys.base_prefix:
        strip.add(os.path.join(sys.prefix, scripts))
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV_PROMPT"):
        env.pop(key, None)
    if "PATH" in env and strip:
        norm = {os.path.normpath(s) for s in strip}
        env["PATH"] = os.pathsep.join(
            p for p in env["PATH"].split(os.pathsep)
            if os.path.normpath(p) not in norm)
    return env


def _idf_shell(export_sh: Path, command: list[str]) -> list[str]:
    script = (f". {shlex.quote(str(export_sh))} >/dev/null 2>&1 || "
              f"{{ echo 'ESP-IDF export.sh failed to load' >&2; exit 97; }}; "
              f"exec {shlex.join(command)}")
    return ["bash", "-c", script]


# -- Windows: capture export.bat's environment once ---------------------------

ENV_MARKER = "__CSIDETECTOR_IDF_ENV__"


@dataclasses.dataclass
class IdfEnv:
    """ESP-IDF's environment as export.bat sets it up, and what to run."""
    env: dict
    python: str            # IDF's own python.exe
    idf_py: Path           # %IDF_PATH%\\tools\\idf.py


_IDF_ENV_CACHE: dict[str, tuple[float, IdfEnv]] = {}


def env_capture_command(export_bat: Path) -> tuple[list[str], str]:
    """(argv, native command line) for cmd.exe: run export.bat (its output
    shows in the log), then print a marker and the environment.

    The native command line is passed verbatim (QProcess.setNativeArguments):
    cmd.exe doesn't follow the quoting rules other programs use, so a path
    with spaces would otherwise break."""
    native = (f'/d /s /c "call "{export_bat}" || (echo ESP-IDF export.bat failed '
              f'to load & exit /b 97) & echo {ENV_MARKER} & set"')
    return ["cmd.exe"], native


def _env_get(env: dict, key: str) -> str:
    """Windows environment names are case-insensitive (``set`` prints Path)."""
    for k, v in env.items():
        if k.upper() == key.upper():
            return v
    return ""


def parse_env_dump(output: str) -> Optional[IdfEnv]:
    """The IdfEnv from env_capture_command's output, or None if export.bat
    didn't get as far as the marker or the result has no usable IDF."""
    if ENV_MARKER not in output:
        return None
    env: dict[str, str] = {}
    for line in output.split(ENV_MARKER, 1)[1].splitlines():
        key, eq, value = line.partition("=")
        if eq and key and not key.startswith("="):     # skip =C:=C:\ entries
            env[key] = value
    idf_path = _env_get(env, "IDF_PATH")
    if not idf_path:
        return None
    python = ""
    venv = _env_get(env, "IDF_PYTHON_ENV_PATH")
    if venv and Path(venv, "Scripts", "python.exe").is_file():
        python = str(Path(venv, "Scripts", "python.exe"))
    if not python:
        python = shutil.which("python", path=_env_get(env, "PATH")) or ""
    if not python:
        return None
    return IdfEnv(env, python, Path(idf_path) / "tools" / "idf.py")


def cached_idf_env(export_bat: Path) -> Optional[IdfEnv]:
    """The captured environment for this export.bat, unless it changed."""
    key = str(Path(export_bat))
    try:
        mtime = Path(export_bat).stat().st_mtime
    except OSError:
        return None
    hit = _IDF_ENV_CACHE.get(key)
    return hit[1] if hit and hit[0] == mtime else None


def remember_idf_env(export_bat: Path, idf: IdfEnv) -> None:
    try:
        _IDF_ENV_CACHE[str(Path(export_bat))] = (Path(export_bat).stat().st_mtime, idf)
    except OSError:
        pass


def _flash_args(job: FlashJob) -> list[str]:
    defaults = f"{job.project_dir / 'sdkconfig.defaults'};{job.overlay_path}"
    return [
        "-C", str(job.project_dir),
        "-B", str(job.build_dir),
        "-D", f"SDKCONFIG={job.sdkconfig_path}",
        "-D", f"SDKCONFIG_DEFAULTS={defaults}",
        "-D", f"IDF_TARGET={job.target}",
        "-p", job.port,
        "flash",
    ]


def flash_command(job: FlashJob, export_sh: Path,
                  idf: Optional[IdfEnv] = None) -> list[str]:
    """``idf: IdfEnv`` on Windows (run directly in the captured environment);
    otherwise through bash and export.sh."""
    if idf is not None:
        return [idf.python, str(idf.idf_py), *_flash_args(job)]
    return _idf_shell(export_sh, ["idf.py", *_flash_args(job)])


def identify_command(port: str, export_sh: Path, esptool_cmd: str = "read_mac",
                     idf: Optional[IdfEnv] = None) -> list[str]:
    """esptool read_mac through IDF's Python (which ships esptool).

    ``read_mac`` works on esptool v4 and v5 (v5 also accepts
    ``read-mac``); callers retry with the dashed form if a future
    esptool drops the underscore alias ("invalid choice").
    """
    if idf is not None:
        return [idf.python, "-m", "esptool", "--port", port, esptool_cmd]
    return _idf_shell(export_sh, [
        "python", "-m", "esptool", "--port", port, esptool_cmd,
    ])


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

_BUILD_STEP_RE = re.compile(r"^\[(\d+)/(\d+)\]")
_FLASH_PCT_RE = re.compile(
    r"Writing at 0x[0-9a-fA-F]+.*?(\d{1,3}(?:\.\d+)?)\s*%")


def parse_progress(line: str) -> Optional[tuple[str, int]]:
    """("build", pct) for ninja steps, ("flash", pct) for esptool writes."""
    m = _BUILD_STEP_RE.match(line)
    if m:
        done, total = int(m.group(1)), int(m.group(2))
        if total > 0:
            return "build", min(100, done * 100 // total)
    m = _FLASH_PCT_RE.search(line)
    if m:
        return "flash", min(100, int(float(m.group(1))))
    return None


_HINTS = [
    (re.compile(r"could not open port|Permission denied|Device or resource busy"
                r"|multiple access on port|Access is denied", re.I),
     "The serial port is busy, unplugged, or not permitted. Close anything "
     "else using it (the alert monitor, a serial monitor such as PuTTY or "
     "the Arduino IDE, `idf.py monitor`, the heatmap) and check the cable."
     + ("" if IS_WINDOWS else " On Linux your user also needs to be in the "
        "`dialout` group.")),
    (re.compile(r"Failed to connect to ESP32|No serial data received|"
                r"Wrong boot mode detected", re.I),
     "The chip didn't enter download mode. Hold BOOT, tap RESET (or "
     "unplug and replug while holding BOOT), release BOOT, then retry."),
    (re.compile(r"No module named 'click'|idf\.py: command not found|"
                r"ESP-IDF export\.(?:sh|bat) failed|is not recognized as an internal"
                r"|No module named 'esptool'", re.I),
     f"ESP-IDF's environment didn't load. Check the {EXPORT_NAME} path in "
     f"the setup window, or run ESP-IDF's "
     f"{'install.bat' if IS_WINDOWS else 'install.sh'} once"
     + (" (or use the ESP-IDF Tools Installer)." if IS_WINDOWS else ".")),
    (re.compile(r"This chip is ESP32[-\w]*,? not ESP32|Wrong --chip argument", re.I),
     "The board is a different chip than the firmware was built for (was "
     "it swapped?). Click Identify, or just Flash again: an unidentified "
     "board is identified before building."),
    (re.compile(r"Target settings are not consistent", re.I),
     f"The cached build was configured for another chip. Delete "
     f"{cache_root()} and retry."),
]


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
