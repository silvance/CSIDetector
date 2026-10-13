"""Scriptable driver for the setup GUI, for exploratory GUI testing.

Runs the real SetupWindow / SetupWizard against a simulated bench:

* Boards are plugged and unplugged in a model: native USB-Serial/JTAG or
  a CP2102 / CH340 / FTDI bridge, any chip, with scripted behaviour
  ("ok", "no_connect", "busy", "hang", "build_fail"). The app enumerates
  them through the real ``list_boards()`` parsing.
* Fake ``idf.py`` and ``python -m esptool`` executables read that model,
  so the app's real QProcess path runs end to end: output streaming,
  progress parsing, failure hints, esptool's wrong-chip error, and
  Cancel's process-tree kill. Each successful flash records the
  firmware config the board now carries (``Session.firmware(mac)``).
* A receiver can stream simulated CSI (``csi(port)``; still or motion),
  read through the collector's real serial loop via a fake
  ``serial.Serial``, so the alert monitor can be calibrated, armed and
  starved of data.
* Native-USB boards drop off the bus briefly after esptool resets them,
  as real ones do.
* Message boxes and file dialogs are recorded (and rendered to PNG)
  instead of blocking. Questions are answered from a queue (default Yes).
* Unhandled exceptions and Qt warnings are recorded, never fatal.
* Every step saves a screenshot and a text dump of the visible widgets
  to ``out_dir``; ``report.md`` indexes them.

Everything the app writes (registry, settings, monitor state, alert.toml,
links.json, build dirs) goes to a temp dir, never the real home or repo.

Script usage (from host/)::

    from tests.gui_driver import Session
    with Session("/tmp/run1") as s:
        s.plug("/dev/ttyACM0", chip="ESP32-S3")
        s.click("Setup wizard")
        s.next()
        ...

CLI (from host/)::

    python -m tests.gui_driver --list
    python -m tests.gui_driver alert_happy_path --out /tmp/run1
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import random
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import types
from pathlib import Path
from typing import Any, Callable, Optional

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt, QtMsgType, qInstallMessageHandler  # noqa: E402
from PyQt6.QtGui import QMouseEvent  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QAbstractButton, QAbstractItemView, QAbstractScrollArea, QAbstractSpinBox,
    QApplication, QBoxLayout, QCheckBox, QDoubleSpinBox, QGraphicsView,
    QComboBox, QFileDialog, QFormLayout, QLabel, QLayout, QLineEdit, QListWidget,
    QMessageBox, QPlainTextEdit, QProgressBar, QRadioButton, QSpinBox, QTableWidget,
    QTabWidget, QWidget, QWizard,
)

HOST_DIR = Path(__file__).resolve().parents[1]
if str(HOST_DIR) not in sys.path:
    sys.path.insert(0, str(HOST_DIR))

import csidetector.modes.alert.gui as alert_gui  # noqa: E402
import csidetector.modes.alert.telegram as telegram  # noqa: E402
import serial as pyserial  # noqa: E402
from csidetector.provision import devices as dv  # noqa: E402
from csidetector.provision import firmware as fw  # noqa: E402
from csidetector.provision import gui  # noqa: E402
from csidetector.provision import wizard as wz  # noqa: E402

_REAL_LIST_BOARDS = dv.list_boards

# USB identities per connection type: (vid, pid, description).
VIA = {
    "native": (dv.ESPRESSIF_VID, 0x1001, "USB JTAG/serial debug unit"),
    "cp2102": (0x10C4, 0xEA60, "CP2102 USB to UART Bridge Controller"),
    "ch340": (0x1A86, 0x7523, "USB Serial"),
    "ftdi": (0x0403, 0x6001, "FT232R USB UART"),
}
BEHAVIORS = ("ok", "no_connect", "busy", "hang", "build_fail")
QT_NOISE = ("propagateSizeHints",)


@dataclasses.dataclass
class FakeBoard:
    port: str
    chip: str          # as esptool names it: ESP32-S3, ESP32-C3, ESP32-D0WD-V3, …
    mac: str
    via: str = "native"
    behavior: str = "ok"

    def comport(self) -> types.SimpleNamespace:
        vid, pid, desc = VIA[self.via]
        serial = self.mac.upper() if self.via == "native" else "0001"
        return types.SimpleNamespace(device=self.port, vid=vid, pid=pid,
                                     serial_number=serial, description=desc)


@dataclasses.dataclass
class Dialog:
    kind: str          # warning | information | critical | question | file
    title: str
    text: str
    answer: str = ""
    image: str = ""


@dataclasses.dataclass
class Step:
    n: int
    action: str
    outcome: str
    screen: str
    dump: str
    dialogs: list
    image: str = ""
    problems: list = dataclasses.field(default_factory=list)


# --------------------------------------------------------------------------
# Fake ESP-IDF tools
# --------------------------------------------------------------------------

_COMMON = r'''
import json, os, subprocess, sys, time
BENCH = {bench!r}

def load(name, default):
    try:
        with open(os.path.join(BENCH, name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default

CTL = load("control.json", {{}})
DELAY = float(CTL.get("line_delay", 0.01))

def out(line):
    print(line, flush=True)
    time.sleep(DELAY)

PREFIXES = ("C61", "S3", "S2", "C2", "C3", "C5", "C6", "H2", "P4")

def chip_target(chip):
    chip = chip.upper()
    for p in PREFIXES:
        if chip.startswith("ESP32-" + p):
            return "esp32" + p.lower()
    return "esp32" if chip.startswith("ESP32") else ""

def target_name(target):
    return "ESP32" + ("-" + target[5:].upper() if len(target) > 5 else "")

def hard_reset(port):
    """The chip resets: a native-USB board leaves the bus for a moment."""
    out("Hard resetting via RTS pin...")
    os.makedirs(os.path.join(BENCH, "reset"), exist_ok=True)
    with open(os.path.join(BENCH, "reset", port.replace("/", "_")), "w") as f:
        f.write(str(time.time()))
'''

_FAKE_IDF = _COMMON + r'''
args = sys.argv[1:]

def opt(flag):
    return args[args.index(flag) + 1] if flag in args else ""

defs = {{}}
for i, a in enumerate(args):
    if a == "-D" and i + 1 < len(args) and "=" in args[i + 1]:
        k, v = args[i + 1].split("=", 1)
        defs[k] = v
port = opt("-p")
target = defs.get("IDF_TARGET", "esp32")
project = os.path.basename(opt("-C").rstrip("/"))
board = load("boards.json", {{}}).get(port)
behavior = board["behavior"] if board else "absent"

out("Executing action: all (aliases: build)")
out("Running ninja in directory " + opt("-B"))
for i in range(1, 41):
    out("[%d/40] Building C object esp-idf/main/CMakeFiles/__idf_main.dir/main.c.obj" % i)
if behavior == "build_fail":
    out("FAILED: esp-idf/main/CMakeFiles/__idf_main.dir/main.c.obj")
    out("main.c:42:5: error: 'CONFIG_CSI_RX_FILTER_TX_MAC' undeclared (first use in this function)")
    out("ninja: build stopped: subcommand failed.")
    sys.exit(2)
out("Project build complete.")
out("Executing action: flash")
out("esptool.py --chip %s -p %s -b 460800 write_flash ..." % (target, port))
out("Serial port " + port)
if behavior == "busy":
    out("serial.serialutil.SerialException: [Errno 16] could not open port %s: "
        "[Errno 16] Device or resource busy: '%s'" % (port, port))
    sys.exit(1)
out("Connecting....")
if behavior in ("absent", "no_connect"):
    out("A fatal error occurred: Failed to connect to %s: No serial data received."
        % target_name(target))
    sys.exit(2)
chip = board["chip"]
if chip_target(chip) != target:
    out("A fatal error occurred: This chip is %s not %s. Wrong --chip argument?"
        % (target_name(chip_target(chip)), target_name(target)))
    sys.exit(2)
out("Chip is %s (revision v0.2)" % chip)
out("Features: WiFi, BLE")
out("MAC: " + board["mac"])
out("Uploading stub...")
out("Running stub...")
if behavior == "hang":
    # esptool, as a grandchild (python, so it exists on Windows too)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    with open(os.path.join(BENCH, "children.txt"), "a") as f:
        f.write("%d\n" % child.pid)
    out("Writing at 0x00010000... (10 %)")
    child.wait()
    sys.exit(1)
for pct in range(0, 101, 10):
    out("Writing at 0x%08x... (%d %%)" % (0x10000 + pct * 0x800, pct))
out("Wrote 812345 bytes (512345 compressed) at 0x00010000 in 6.1 seconds")
out("Hash of data verified.")
hard_reset(port)

cfg = {{}}
overlay = defs.get("SDKCONFIG_DEFAULTS", "").split(";")[-1]
try:
    with open(overlay) as f:
        for line in f:
            line = line.strip()
            if line.startswith("CONFIG_") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k] = "<redacted>" if "PASS" in k else v.strip('"')
except OSError:
    pass
os.makedirs(os.path.join(BENCH, "firmware"), exist_ok=True)
with open(os.path.join(BENCH, "firmware", board["mac"].replace(":", "") + ".json"), "w") as f:
    json.dump({{"project": project, "target": target, "port": port, "config": cfg}}, f, indent=1)
out("Done")
'''

_FAKE_ESPTOOL = _COMMON + r'''
args = sys.argv[1:]
port = args[args.index("--port") + 1] if "--port" in args else ""
cmd = args[-1] if args else ""
ok_cmds = ("read-mac",) if CTL.get("esptool_dash_only") else ("read_mac", "read-mac")
if cmd not in ok_cmds:
    out("esptool: error: argument operation: invalid choice: '%s'" % cmd)
    sys.exit(2)
board = load("boards.json", {{}}).get(port)
behavior = board["behavior"] if board else "absent"
out("esptool.py v4.7.0")
out("Serial port " + port)
if behavior == "busy":
    out("serial.serialutil.SerialException: [Errno 16] could not open port %s: "
        "[Errno 16] Device or resource busy: '%s'" % (port, port))
    sys.exit(1)
out("Connecting....")
if behavior in ("absent", "no_connect"):
    out("A fatal error occurred: Failed to connect to Espressif device: No serial data received.")
    sys.exit(2)
out("Detecting chip type... " + target_name(chip_target(board["chip"])))
out("Chip is %s (revision v0.2)" % board["chip"])
out("MAC: " + board["mac"])
hard_reset(port)
'''


def _alive(pid: int) -> bool:
    """Is the process running (not a zombie)? os.kill(pid, 0) can't be
    used on Windows: signal 0 there is CTRL_C_EVENT."""
    if fw.IS_WINDOWS:
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259                                       # STILL_ACTIVE
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
        return state != "Z"
    except OSError:
        pass
    try:
        os.kill(pid, 0)                                   # no /proc (macOS)
        return True
    except OSError:
        return False


def _kill(pid: int) -> None:
    if fw.IS_WINDOWS:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _write_exe(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


# --------------------------------------------------------------------------
# Widget helpers
# --------------------------------------------------------------------------

_HTML_TAG = re.compile(r"</?(?:a|b|i|u|s|p|br|em|strong|span|div|font|small|big|tt|code|pre|"
                       r"h[1-6]|ul|ol|li|table|tr|td|th|hr|img|nobr|qt)\b[^>]*>", re.I)


def _norm(text: str) -> str:
    text = _HTML_TAG.sub(" ", text or "").replace("&", "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text.strip(" :<>….")


def _plain(text: str, limit: int = 240) -> str:
    text = re.sub(r"&(?!&)", "", text or "").replace("&&", "&")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = _HTML_TAG.sub("", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = text.replace("\n", " ⏎ ")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _layout_widgets(item) -> list[QWidget]:
    """Widgets inside a layout item, recursively."""
    if item is None:
        return []
    w = item.widget()
    if w is not None:
        return [w]
    lay = item.layout() if not isinstance(item, QLayout) else item
    out: list[QWidget] = []
    if lay is not None:
        for i in range(lay.count()):
            out += _layout_widgets(lay.itemAt(i))
    return out


def _internal(w: QWidget) -> bool:
    """Sub-widgets of composite widgets (a spin box's line edit, a
    table's viewport) aren't things a user addresses on their own."""
    p = w.parentWidget()
    while p is not None:
        if isinstance(p, (QAbstractSpinBox, QComboBox, QAbstractItemView)):
            return True
        p = p.parentWidget()
    return False


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------

class DriverError(AssertionError):
    """The script asked for something that isn't on screen."""


class Session:
    def __init__(self, out_dir: Optional[str | Path] = None, screenshots: bool = True,
                 line_delay: float = 0.01, name: str = "session", fast_timers: bool = False,
                 calibrate_s: tuple[float, float] = (2.0, 2.0),
                 reset_drop_s: Optional[float] = None):
        """``calibrate_s``: the monitor's (leave-the-room, record) seconds
        (the app uses 30 + 30). ``reset_drop_s``: how long a native board is
        off the bus after a reset (0 = never). Default: as on real hardware,
        shorter than the app's post-flash re-scan delay."""
        self.name = name
        self.screenshots = screenshots and out_dir is not None
        self.out = Path(out_dir) if out_dir else None
        self._tmp = Path(tempfile.mkdtemp(prefix="csid-gui-"))
        self.bench = self._tmp / "bench"
        self.home = self._tmp / "home"
        self.host_dir = self._tmp / "host"
        self.boards: dict[str, FakeBoard] = {}
        self.hotspots: list[fw.HotspotInfo] = []
        self.telegram_ok = True
        self.telegram_sent: list[str] = []
        self.dialogs: list[Dialog] = []
        self.steps: list[Step] = []
        self.exceptions: list[str] = []
        self.qt_warnings: list[str] = []
        self.failed_expectations: list[str] = []
        self.notes: list[str] = []
        self._answers: list[str] = []
        self._files: list[str] = []
        self._dialogs_seen = 0
        self._mac_seq = 0
        self._line_delay = line_delay
        self._control: dict[str, Any] = {"line_delay": line_delay}
        self._restore: list[Callable[[], None]] = []
        self._focus = ""
        self._fast_timers = fast_timers
        self._calibrate_s = calibrate_s
        self.reset_drop_s = (reset_drop_s if reset_drop_s is not None
                             else 0.1 if fast_timers else 0.8)
        self._csi: dict[str, dict] = {}       # port → {"rate", "motion", "seq"}
        self.window: Optional[gui.SetupWindow] = None

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> "Session":
        try:
            self.start()
        except BaseException:
            self.close()                    # undo the patches start() made
            raise
        return self

    def __exit__(self, *exc) -> None:
        if exc[0] is not None and not isinstance(exc[1], (DriverError, KeyboardInterrupt)):
            self.exceptions.append("script error: " + "".join(
                traceback.format_exception(*exc)))
        self.close()

    def start(self) -> None:
        self.app = QApplication.instance() or QApplication([])
        for d in (self.bench, self.home, self.host_dir, self._tmp / "fakebin"):
            d.mkdir(parents=True, exist_ok=True)
        if self.out:
            self.out.mkdir(parents=True, exist_ok=True)
        self._install_fake_tools()
        self._patch_environment()
        self._sync_bench()
        self.window = gui.SetupWindow(
            dv.Registry(self.home / ".csidetector" / "devices.json"),
            idf_export=str(self.export_sh),
            list_boards=self._list_boards, detect_hotspots=lambda: list(self.hotspots),
            settings_path=self.home / ".csidetector" / "setup.json")
        self.window.show()
        self._step("start the setup app", "shown")

    def close(self) -> None:
        try:
            self._close_app()
        finally:
            for undo in reversed(self._restore):
                undo()
            self._restore.clear()
            if self.out:
                self.write_report()
            shutil.rmtree(self._tmp, ignore_errors=True)

    def _close_app(self) -> None:
        self._csi.clear()
        if self.window is not None:
            for runner in self._runners():
                if runner.running:
                    runner.cancel()
                    runner.wait(3000)
            for w in (self.window.wizard, self.window._monitor, *self.window._retired_monitors):
                if w is not None:
                    w.close()
            self.window.devices.shutdown()
            self.window.close()
            self.window.deleteLater()
            QTest.qWait(50)
            self.window = None
        leftovers = self.leftover_processes()
        if leftovers:
            self.failed_expectations.append(
                f"processes left running after the session: {leftovers}")
            for pid in leftovers:
                _kill(pid)

    def _patch(self, obj: Any, attr: str, value: Any) -> None:
        old = getattr(obj, attr)
        setattr(obj, attr, value)
        self._restore.append(lambda: setattr(obj, attr, old))

    def _patch_environment(self) -> None:
        for key, value in (("XDG_CACHE_HOME", str(self._tmp / "cache")),
                           ("HOME", str(self.home)),
                           ("USERPROFILE", str(self.home))):      # ~ on Windows
            old = os.environ.get(key)
            os.environ[key] = value
            self._restore.append(lambda k=key, o=old: os.environ.__setitem__(k, o)
                                 if o is not None else os.environ.pop(k, None))
        state_dir = self.home / ".csidetector"
        self._patch(alert_gui, "_STATE_DIR", state_dir)
        self._patch(alert_gui, "_STATE_PATH", state_dir / "state.json")
        self._patch(wz, "default_host_dir", lambda: self.host_dir)
        if self._fast_timers:
            self._patch(gui.TIMING, "auto_refresh_ms", 250)
            self._patch(gui.TIMING, "post_process_refresh_ms", 200)
            self._patch(alert_gui, "NO_DATA_WARN_S", 1.0)
        self._patch(alert_gui, "CALIBRATE_SETTLE_S", self._calibrate_s[0])
        self._patch(alert_gui, "CALIBRATE_RECORD_S", self._calibrate_s[1])
        # The monitor finds its receiver again via list_boards().
        self._patch(dv, "list_boards", lambda comports=None: self._list_boards())
        # Never the host's real /dev/serial/by-id (a plugged-in board would
        # change what the wizard saves).
        self._patch(dv, "stable_port", lambda port, mac="", by_id_dir=None: port)
        self._patch(pyserial, "Serial", self._fake_serial_class())
        for kind in ("warning", "information", "critical", "question"):
            self._patch(QMessageBox, kind, self._message_box(kind))
        self._patch(QFileDialog, "getOpenFileName", self._file_dialog("open"))
        self._patch(QFileDialog, "getSaveFileName", self._file_dialog("save"))

        session = self

        class FakeTelegram:
            def __init__(self, bot_token: str = "", chat_id: str = "", *a, **k):
                self.chat_id = chat_id

            def send(self, event) -> None:
                if not session.telegram_ok:
                    raise RuntimeError("Unauthorized (401): check the bot token")
                session.telegram_sent.append(str(getattr(event, "message", event)))

        self._patch(telegram, "TelegramNotifier", FakeTelegram)

        old_hook = sys.excepthook
        sys.excepthook = self._excepthook
        self._restore.append(lambda: setattr(sys, "excepthook", old_hook))
        qInstallMessageHandler(self._qt_message)
        self._restore.append(lambda: qInstallMessageHandler(None))

    def _install_fake_tools(self) -> None:
        if fw.IS_WINDOWS:
            self._install_fake_tools_windows()
            return
        fakebin = self._tmp / "fakebin"
        py = sys.executable
        _write_exe(fakebin / "idf.py",
                   f"#!{py}\n" + _FAKE_IDF.format(bench=str(self.bench)))
        _write_exe(fakebin / "fake_esptool.py",
                   f"#!{py}\n" + _FAKE_ESPTOOL.format(bench=str(self.bench)))
        _write_exe(fakebin / "python", textwrap.dedent(f"""\
            #!/bin/sh
            if [ "$1" = "-m" ] && [ "$2" = "esptool" ]; then
                shift 2
                exec {py} {fakebin / 'fake_esptool.py'} "$@"
            fi
            exec {py} "$@"
            """))
        self.export_sh = self._tmp / "esp-idf" / "export.sh"
        self.export_sh.parent.mkdir()
        self.export_sh.write_text(f'export PATH="{fakebin}:$PATH"\n')

    def _install_fake_tools_windows(self) -> None:
        """Windows: an export.bat (in a folder with a space, so cmd.exe's
        quoting is exercised) that sets IDF_PATH to a fake IDF whose
        tools\\idf.py is the fake, and PYTHONPATH so ``python -m esptool``
        finds a fake esptool package. The app runs them with the python on
        the captured PATH, as with real ESP-IDF."""
        fakeidf = self._tmp / "fake idf"
        (fakeidf / "tools").mkdir(parents=True)
        (fakeidf / "tools" / "idf.py").write_text(_FAKE_IDF.format(bench=str(self.bench)))
        pkg = self._tmp / "fakebin" / "esptool"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(_FAKE_ESPTOOL.format(bench=str(self.bench)))
        self.export_sh = self._tmp / "esp idf" / "export.bat"
        self.export_sh.parent.mkdir()
        self.export_sh.write_text(
            "@echo off\r\n"
            "echo Activating ESP-IDF (fake)\r\n"
            f'set "IDF_PATH={fakeidf}"\r\n'
            f'set "PYTHONPATH={self._tmp / "fakebin"}"\r\n'
            f'set "PATH={os.path.dirname(sys.executable)};%PATH%"\r\n')
    # -- the bench -------------------------------------------------------

    def plug(self, port: str, chip: str = "ESP32-S3", mac: str = "",
             via: str = "native", behavior: str = "ok") -> FakeBoard:
        """Connect a board. Same MAC on two ports = one board with both
        its UART and native USB connectors plugged in."""
        if via not in VIA:
            raise DriverError(f"via must be one of {sorted(VIA)}")
        if behavior not in BEHAVIORS:
            raise DriverError(f"behavior must be one of {BEHAVIORS}")
        if not mac:
            self._mac_seq += 1
            mac = f"30:30:f9:00:00:{self._mac_seq:02x}"
        board = FakeBoard(port, chip, dv.normalize_mac(mac), via, behavior)
        self.boards[port] = board
        self._sync_bench()
        self.notes.append(f"plugged {port}: {chip} {board.mac} via {via} ({behavior})")
        return board

    def unplug(self, port: str) -> None:
        self.boards.pop(port, None)
        self._sync_bench()
        self.notes.append(f"unplugged {port}")

    def set_behavior(self, port: str, behavior: str) -> None:
        self.boards[port].behavior = behavior
        self._sync_bench()

    def hotspot(self, ssid: str = "CSIDetector", channel: int = 6, ip: str = "10.42.0.1",
                connection: str = "", iface: str = "wlan0") -> None:
        """Pretend a NetworkManager hotspot is running."""
        self.hotspots = [fw.HotspotInfo(connection or ssid, iface, ssid, channel, ip)]

    def csi(self, port: str, rate: float = 100.0, motion: bool = False,
            on: bool = True) -> None:
        """The receiver on ``port`` hears its transmitter: it streams CSI
        lines at ``rate`` Hz (``motion`` = someone moving). ``on=False``
        stops the stream (transmitter unplugged)."""
        if on:
            self._csi[port] = {"rate": rate, "motion": motion, "seq": 0}
        else:
            self._csi.pop(port, None)
        self.notes.append(f"CSI on {port}: " + (
            f"{rate:g} Hz, {'motion' if motion else 'still'}" if on else "off"))

    def motion(self, port: str, on: bool = True) -> None:
        """Someone starts / stops moving between the boards."""
        self._csi[port]["motion"] = on
        self.notes.append(f"motion on {port}: {on}")

    def _fake_serial_class(self):
        session = self

        class FakeSerial:
            """serial.Serial for the collector: CSI lines from the bench,
            b"" after ``timeout`` when nothing streams (as pyserial does)."""

            def __init__(self, port: str, baudrate: int = 9600, timeout=None, **kw):
                if port not in session.boards:
                    raise pyserial.SerialException(
                        f"[Errno 2] could not open port {port}: [Errno 2] No such "
                        f"file or directory: '{port}'")
                self.port, self.timeout = port, timeout or 1.0

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def close(self) -> None:
                pass

            def readline(self) -> bytes:
                if self.port not in session.boards:
                    raise pyserial.SerialException(
                        "device reports readiness to read but returned no data "
                        "(device disconnected or multiple access on port?)")
                stream = session._csi.get(self.port)
                if not stream:
                    time.sleep(self.timeout)
                    return b""
                time.sleep(1.0 / stream["rate"])
                stream["seq"] += 1
                return session._csi_line(self.port, stream).encode()

        return FakeSerial

    def _csi_line(self, port: str, stream: dict) -> str:
        """A CSI_DATA line as the receiver firmware prints it: 64
        subcarriers (guard / DC zeroed), amplitude noisier with motion."""
        fwr = self.firmware(self.boards[port].mac) or {}
        cfg = fwr.get("config", {})
        tx = (cfg.get("CONFIG_CSI_RX_FILTER_TX_MAC", "").split(",")[0].strip()
              or "aa:bb:cc:dd:ee:ff")
        channel = cfg.get("CONFIG_CSI_RX_CHANNEL", "11")
        sigma = 4.0 if stream["motion"] else 0.6
        iq = []
        for k in range(64):
            amp = 0 if (k < 6 or k > 58 or k == 32) else \
                max(1, min(120, round(20 + 6 * math.sin(k / 5) + random.gauss(0, sigma))))
            iq += [0, amp]                 # (Im, Re) pairs
        fields = ["CSI_DATA", str(stream["seq"]), tx, "-52", "11", "1", "0", "0", "0", "0",
                  "0", "0", "0", "0", "-92", "0", str(channel), "0",
                  str(int(time.time() * 1e6) & 0x7FFFFFFF), "0", "0", "0", str(len(iq)),
                  "0", '"' + json.dumps(iq).replace(" ", "") + '"']
        return ",".join(fields) + "\n"

    def esptool_dash_only(self, on: bool = True) -> None:
        """Simulate an esptool that only accepts ``read-mac``."""
        self._control["esptool_dash_only"] = on
        self._sync_bench()

    def firmware(self, mac: str) -> Optional[dict]:
        """What the fake flasher last wrote to this board (password redacted)."""
        path = self.bench / "firmware" / (dv.normalize_mac(mac).replace(":", "") + ".json")
        return json.loads(path.read_text()) if path.exists() else None

    def leftover_processes(self) -> list[int]:
        """Fake esptool grandchildren (from behavior="hang") still alive."""
        path = self.bench / "children.txt"
        return [int(pid) for pid in (path.read_text().split() if path.exists() else [])
                if _alive(int(pid))]

    def _sync_bench(self) -> None:
        (self.bench / "boards.json").write_text(json.dumps(
            {p: dataclasses.asdict(b) for p, b in self.boards.items()}, indent=1))
        (self.bench / "control.json").write_text(json.dumps(self._control))

    def _list_boards(self) -> list[dv.AttachedBoard]:
        return _REAL_LIST_BOARDS([b.comport() for b in self.boards.values()
                                  if not self._resetting(b)])

    def _resetting(self, board: FakeBoard) -> bool:
        """A native-USB board re-enumerates after a reset: briefly gone."""
        if board.via != "native" or self.reset_drop_s <= 0:
            return False
        marker = self.bench / "reset" / board.port.replace("/", "_")
        try:
            return time.time() - float(marker.read_text()) < self.reset_drop_s
        except (OSError, ValueError):
            return False

    def _runners(self) -> list:
        if self.window is None:
            return []
        out = [self.window.devices.flasher._runner]
        if self.window.wizard is not None:
            out.append(self.window.wizard.flasher._runner)
        return out

    # -- recorders -------------------------------------------------------

    def _excepthook(self, exc_type, exc, tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        self.exceptions.append(text)
        sys.stderr.write(text)

    def _qt_message(self, mode, context, message) -> None:
        if mode == QtMsgType.QtDebugMsg or any(n in message for n in QT_NOISE):
            return
        self.qt_warnings.append(f"{mode.name}: {message}")

    def _message_box(self, kind: str):
        session = self

        def box(parent, title, text, *args, **kwargs):
            answer = ""
            if kind == "question":
                answer = session._answers.pop(0) if session._answers else "Yes"
                result = (QMessageBox.StandardButton.Yes if answer.lower() == "yes"
                          else QMessageBox.StandardButton.No)
            else:
                result = QMessageBox.StandardButton.Ok
            d = Dialog(kind, title, text, answer)
            session.dialogs.append(d)
            d.image = session._render_box(kind, title, text)
            return result
        return box

    def _file_dialog(self, kind: str):
        session = self

        def dialog(*args, **kwargs):
            path = session._files.pop(0) if session._files else ""
            session.dialogs.append(Dialog("file", f"{kind} file", str(args[1:3]),
                                          path or "(cancelled)"))
            return path, ""
        return dialog

    def _render_box(self, kind: str, title: str, text: str) -> str:
        if not self.screenshots:
            return ""
        icon = {"warning": QMessageBox.Icon.Warning, "critical": QMessageBox.Icon.Critical,
                "question": QMessageBox.Icon.Question}.get(kind, QMessageBox.Icon.Information)
        buttons = (QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                   if kind == "question" else QMessageBox.StandardButton.Ok)
        box = QMessageBox(icon, title, text, buttons)
        box.show()                          # laid out (and wrapped) as on screen
        QApplication.processEvents()
        name = f"{len(self.steps):02d}-dialog-{len(self.dialogs)}.png"
        box.grab().save(str(self.out / name))
        box.close()
        box.deleteLater()
        return name

    # -- what's on screen --------------------------------------------------

    def screen(self) -> QWidget:
        """The window a user would be looking at."""
        w = self.window
        if self._focus == "monitor" and w._monitor is not None:
            return w._monitor
        if self._focus != "main" and w.wizard is not None and w.wizard.isVisible():
            return w.wizard
        return w

    def focus(self, which: str) -> None:
        """'auto' (wizard if open, else main), 'main', or 'monitor'."""
        self._focus = "" if which == "auto" else which

    @property
    def wizard(self) -> Optional[wz.SetupWizard]:
        return self.window.wizard if self.window else None

    def page(self):
        wiz = self.wizard
        return wiz.currentPage() if wiz is not None and wiz.isVisible() else None

    def _visible(self, cls=QWidget) -> list[QWidget]:
        root = self.screen()
        found = [w for w in root.findChildren(cls)
                 if w.isVisible() and w.window() is root and not _internal(w)]
        return sorted(found, key=self._pos)

    def _pos(self, w: QWidget) -> tuple[int, int]:
        p = w.mapTo(self.screen(), QPoint(0, 0))
        return (p.y(), p.x())

    def _label_map(self) -> dict[int, list[str]]:
        """id(widget) → user-visible names (form labels, the label just
        before it in a row, accessible name, placeholder)."""
        names: dict[int, list[str]] = {}

        def add(w, text):
            if text and _norm(text):
                names.setdefault(id(w), []).append(_plain(text, 80))

        for host in [self.screen(), *self.screen().findChildren(QWidget)]:
            lay = host.layout()
            for layout in ([lay] + lay.findChildren(QLayout)) if lay else []:
                if isinstance(layout, QFormLayout):
                    for row in range(layout.rowCount()):
                        label = layout.itemAt(row, QFormLayout.ItemRole.LabelRole)
                        field = layout.itemAt(row, QFormLayout.ItemRole.FieldRole)
                        lw = label.widget() if label else None
                        if isinstance(lw, QLabel):
                            for w in _layout_widgets(field):
                                add(w, lw.text())
                elif (isinstance(layout, QBoxLayout)
                      and layout.direction() == QBoxLayout.Direction.LeftToRight):
                    # "Label: [field]" rows.
                    prev = None
                    for i in range(layout.count()):
                        w = layout.itemAt(i).widget()
                        if isinstance(prev, QLabel) and w is not None \
                                and not isinstance(w, QLabel):
                            add(w, prev.text())
                        prev = w
        for w in self._visible():
            if w.accessibleName():
                names.setdefault(id(w), []).insert(0, _plain(w.accessibleName(), 80))
            if isinstance(w, QLineEdit):
                add(w, w.placeholderText())
        return names

    def dump(self) -> str:
        """Visible widgets in reading order, as text."""
        names = self._label_map()
        lines = []
        for w in self._visible():
            n = names.get(id(w), [])
            tag = f" [{n[0]}]" if n else ""
            off = "" if w.isEnabled() else " (disabled)"
            if isinstance(w, QWizard):
                continue
            if isinstance(w, QRadioButton):
                lines.append(f"radio ({'•' if w.isChecked() else ' '}) {_plain(w.text())!r}{off}")
            elif isinstance(w, QCheckBox):
                lines.append(f"checkbox [{'x' if w.isChecked() else ' '}] {_plain(w.text())!r}{off}")
            elif isinstance(w, QAbstractButton):
                if w.text():
                    lines.append(f"button {_plain(w.text())!r}{off}")
            elif isinstance(w, QLabel):
                if w.text().strip() and not w.buddy():
                    lines.append(f"text {_plain(w.text(), 2000)!r}")
            elif isinstance(w, QLineEdit):
                value = ("●" * len(w.text()) if w.echoMode() != QLineEdit.EchoMode.Normal
                         else w.text())
                ro = " (read-only)" if w.isReadOnly() else ""
                lines.append(f"field{tag} = {value!r}{off}{ro}")
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                lines.append(f"spin{tag} = {w.value()}{w.suffix()}{off}")
            elif isinstance(w, QComboBox):
                lines.append(f"dropdown{tag} = {w.currentText()!r} "
                             f"({w.count()} options){off}")
            elif isinstance(w, QListWidget):
                lines.append(f"list{tag}: {w.count()} item(s){off}")
                for i in range(w.count()):
                    it = w.item(i)
                    mark = {Qt.CheckState.Checked: "[x] ", Qt.CheckState.Unchecked: "[ ] "}.get(
                        it.checkState(), "") if it.flags() & Qt.ItemFlag.ItemIsUserCheckable else ""
                    lines.append(f"    {mark}{_plain(it.text(), 120)}")
            elif isinstance(w, QTableWidget):
                cols = [w.horizontalHeaderItem(c).text() if w.horizontalHeaderItem(c) else ""
                        for c in range(w.columnCount())]
                sel = {i.row() for i in w.selectedIndexes()}
                lines.append(f"table{tag}: {' | '.join(cols)}")
                for r in range(w.rowCount()):
                    cells = [w.item(r, c).text() if w.item(r, c) else ""
                             for c in range(w.columnCount())]
                    lines.append(f"  {'>' if r in sel else ' '} {' | '.join(cells)}")
            elif isinstance(w, QPlainTextEdit):
                tail = w.toPlainText().splitlines()[-6:]
                lines.append(f"textbox{tag}: {len(w.toPlainText().splitlines())} line(s)")
                lines += [f"    | {_plain(t, 140)}" for t in tail]
            elif isinstance(w, QProgressBar):
                value = f"{w.value()}%" if w.value() >= w.minimum() else "empty"
                lines.append(f"progress{tag} = {value}"
                             f"{' (text hidden)' if not w.isTextVisible() else ''}")
            elif isinstance(w, QGraphicsView):
                canvas = getattr(w, "node_items", None)
                extra = (f": {len(canvas)} board marker(s), "
                         f"{len(getattr(w, 'corner_items', []))} corner(s)"
                         if canvas is not None else "")
                lines.append(f"canvas{tag}{extra}")
            elif isinstance(w, QTabWidget):
                tabs = [("*" if i == w.currentIndex() else "") + w.tabText(i).replace("&&", "&")
                        for i in range(w.count())]
                lines.append(f"tabs: {', '.join(tabs)}")
        return "\n".join(lines)

    def texts(self) -> str:
        """All visible text on screen, lower-cased — for quick ``in`` checks."""
        return self.dump().lower()

    def page_title(self) -> str:
        p = self.page()
        return p.title() if p is not None else ""

    # -- finding widgets ---------------------------------------------------

    def _pick(self, kind: str, cands: list, query: str, nth: Optional[int]):
        q = _norm(query)
        tiers =([c for c in cands if q == c[0]], [c for c in cands if c[0].startswith(q)],
                 [c for c in cands if q in c[0]])
        for tier in tiers:
            # Several names can point at one widget; dedupe.
            seen, uniq = set(), []
            for name, w in tier:
                if id(w) not in seen:
                    seen.add(id(w))
                    uniq.append((name, w))
            if len(uniq) == 1 and nth is None:
                return uniq[0][1]
            if uniq and nth is not None:
                if nth >= len(uniq):
                    break
                return uniq[nth][1]
            if len(uniq) > 1:
                listing = "\n".join(f"  nth={i}: {n!r} ({type(w).__name__}"
                                    f"{', disabled' if not w.isEnabled() else ''})"
                                    for i, (n, w) in enumerate(uniq))
                raise DriverError(f"{kind} {query!r} is ambiguous; pass nth=:\n{listing}")
        raise DriverError(f"no visible {kind} matching {query!r} on screen.\n"
                          f"On screen now:\n{textwrap.indent(self.dump(), '  ')}")

    def button(self, text: str, nth: Optional[int] = None) -> QAbstractButton:
        cands = [(_norm(b.text()), b) for b in self._visible(QAbstractButton) if b.text()]
        return self._pick("button", cands, text, nth)

    def field(self, name: str, nth: Optional[int] = None) -> QWidget:
        names = self._label_map()
        kinds = (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QTableWidget,
                 QPlainTextEdit, QCheckBox, QRadioButton)
        cands = []
        for w in self._visible():
            if isinstance(w, kinds):
                for n in names.get(id(w), []):
                    cands.append((_norm(n), w))
                if isinstance(w, (QCheckBox, QRadioButton)):
                    cands.append((_norm(w.text()), w))
        return self._pick("field", cands, name, nth)

    # -- actions ---------------------------------------------------------

    def click(self, text: str, nth: Optional[int] = None) -> bool:
        """Click a visible button by its text. A disabled button does
        nothing (as for a user) and the step says so."""
        b = self.button(text, nth)
        label = _plain(b.text())
        if not b.isEnabled():
            self._step(f"click {label!r}", "nothing happened: the button is disabled")
            return False
        QTest.mouseClick(b, Qt.MouseButton.LeftButton)
        self._step(f"click {label!r}", "clicked")
        return True

    def fill(self, name: str, value: Any, nth: Optional[int] = None) -> bool:
        """Set a field found by its label / placeholder: text boxes are
        typed into, spin boxes set, dropdowns chosen by visible text,
        checkboxes and radio buttons set to bool(value)."""
        w = self.field(name, nth)
        if not w.isEnabled():
            self._step(f"set {name!r}", "nothing happened: the field is disabled")
            return False
        if isinstance(w, QLineEdit):
            w.setFocus()
            w.selectAll()
            QTest.keyClick(w, Qt.Key.Key_Backspace)
            if str(value):
                QTest.keyClicks(w, str(value))
            w.editingFinished.emit()
            shown = "●" * len(str(value)) if w.echoMode() != QLineEdit.EchoMode.Normal else value
            self._step(f"type {shown!r} into {name!r}", "typed")
        elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
            w.setValue(int(value) if isinstance(w, QSpinBox) else float(value))
            self._step(f"set {name!r} to {w.value()}", "set")
        elif isinstance(w, QComboBox):
            return self._choose(w, name, str(value))
        elif isinstance(w, (QCheckBox, QRadioButton)):
            if w.isChecked() != bool(value):
                # On the indicator: the middle of a full-width row is empty
                # space that (as for a real click) doesn't toggle it.
                QTest.mouseClick(w, Qt.MouseButton.LeftButton, pos=QPoint(8, w.height() // 2))
            self._step(f"{'check' if value else 'uncheck'} {name!r}",
                       "checked" if w.isChecked() else "unchecked")
        else:
            raise DriverError(f"don't know how to fill a {type(w).__name__}")
        return True

    def choose(self, name: str, option: str, nth: Optional[int] = None) -> bool:
        """Pick a dropdown option by (part of) its visible text."""
        w = self.field(name, nth)
        if not isinstance(w, QComboBox):
            raise DriverError(f"{name!r} is a {type(w).__name__}, not a dropdown")
        if not w.isEnabled():
            self._step(f"choose {option!r} in {name!r}",
                       "nothing happened: the dropdown is disabled")
            return False
        return self._choose(w, name, option)

    def _choose(self, w: QComboBox, name: str, option: str) -> bool:
        q = _norm(option)
        options = [w.itemText(i) for i in range(w.count())]
        idx = next((i for i, t in enumerate(options) if _norm(t) == q), None)
        if idx is None:
            hits = [i for i, t in enumerate(options) if q in _norm(t)]
            if len(hits) != 1:
                raise DriverError(f"option {option!r} in {name!r}: "
                                  f"{'ambiguous' if hits else 'not found'}; options: {options}")
            idx = hits[0]
        w.setCurrentIndex(idx)
        w.activated.emit(idx)
        self._step(f"choose {options[idx]!r} in {name!r}", "chosen")
        return True

    def tick(self, text: str, on: bool = True) -> None:
        """Tick / untick a checkable list item by (part of) its text."""
        hits = []
        for lw in self._visible(QListWidget):
            for i in range(lw.count()):
                it = lw.item(i)
                if _norm(text) in _norm(it.text()) and it.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    hits.append(it)
        if len(hits) != 1:
            raise DriverError(f"list item {text!r}: {'ambiguous' if hits else 'not found'}\n"
                              f"{textwrap.indent(self.dump(), '  ')}")
        hits[0].setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self._step(f"{'tick' if on else 'untick'} {text!r}", "done")

    def select_row(self, text: str) -> None:
        """Select the table row containing ``text`` (port, MAC or label)."""
        for t in self._visible(QTableWidget):
            for r in range(t.rowCount()):
                cells = [t.item(r, c).text() if t.item(r, c) else "" for c in range(t.columnCount())]
                if any(_norm(text) in _norm(c) for c in cells):
                    t.selectRow(r)
                    self._step(f"select row {text!r}", " | ".join(cells))
                    return
        raise DriverError(f"no table row contains {text!r}")

    def widget(self, name: str, nth: Optional[int] = None) -> QWidget:
        """Any visible widget by its label / accessible name (e.g. a canvas)."""
        names = self._label_map()
        cands = [(_norm(n), w) for w in self._visible() for n in names.get(id(w), [])]
        return self._pick("widget", cands, name, nth)

    def _mouse(self, w: QWidget, kind: QEvent.Type, pos: QPoint,
               button: Qt.MouseButton, buttons: Qt.MouseButton) -> None:
        target = w.viewport() if isinstance(w, QAbstractScrollArea) else w
        ev = QMouseEvent(kind, QPointF(pos), QPointF(target.mapToGlobal(pos)),
                         button, buttons, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(target, ev)

    def click_at(self, name: str, pos: QPoint, double: bool = False) -> None:
        """Click at a point in a widget (viewport pixels for a canvas)."""
        w = self.widget(name)
        left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
        self._mouse(w, QEvent.Type.MouseButtonPress, pos, left, left)
        self._mouse(w, QEvent.Type.MouseButtonRelease, pos, left, none)
        if double:
            self._mouse(w, QEvent.Type.MouseButtonDblClick, pos, left, left)
            self._mouse(w, QEvent.Type.MouseButtonRelease, pos, left, none)
        self._step(f"{'double-' if double else ''}click {name!r} at ({pos.x()}, {pos.y()})",
                   "clicked")

    def drag(self, name: str, start: QPoint, end: QPoint, steps: int = 8) -> None:
        """Press at ``start``, move to ``end`` with the button held, release.
        (QTest.mouseMove can't hold a button, which item drags need.)"""
        w = self.widget(name)
        left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
        self._mouse(w, QEvent.Type.MouseButtonPress, start, left, left)
        for i in range(1, steps + 1):
            p = QPoint(start.x() + (end.x() - start.x()) * i // steps,
                       start.y() + (end.y() - start.y()) * i // steps)
            self._mouse(w, QEvent.Type.MouseMove, p, none, left)
        self._mouse(w, QEvent.Type.MouseButtonRelease, end, left, none)
        self._step(f"drag in {name!r} from ({start.x()}, {start.y()}) to "
                   f"({end.x()}, {end.y()})", "dragged")

    def edit_cell(self, table: str, row_text: str, column: str, value: Any) -> None:
        """Type into a table cell: the row containing ``row_text``, the
        column whose header starts with ``column``."""
        t = self.widget(table)
        cols = [_norm(t.horizontalHeaderItem(c).text()) for c in range(t.columnCount())]
        col = next((c for c, h in enumerate(cols) if h.startswith(_norm(column))), None)
        rows = [r for r in range(t.rowCount())
                if any(_norm(row_text) == _norm(t.item(r, c).text())
                       for c in range(t.columnCount()) if t.item(r, c))]
        if col is None or len(rows) != 1:
            raise DriverError(f"cell {row_text!r}/{column!r} in {table!r}: "
                              f"{'no column' if col is None else f'{len(rows)} rows'}")
        t.item(rows[0], col).setText(str(value))
        self._step(f"set {table!r} [{row_text} / {column}] to {value!r}", "edited")

    def tab(self, name: str) -> None:
        for tw in self._visible(QTabWidget):
            for i in range(tw.count()):
                if _norm(name) in _norm(tw.tabText(i)):
                    tw.setCurrentIndex(i)
                    self._step(f"open tab {tw.tabText(i)!r}", "opened")
                    return
        raise DriverError(f"no tab {name!r}")

    def _nav(self, which: QWizard.WizardButton, label: str) -> bool:
        wiz = self.wizard
        if wiz is None or not wiz.isVisible():
            raise DriverError("the wizard isn't open")
        b = wiz.button(which)
        if not b.isVisible():
            self._step(f"click {label!r}", "nothing happened: the button isn't shown")
            return False
        if not b.isEnabled():
            self._step(f"click {label!r}", "nothing happened: the button is disabled")
            return False
        QTest.mouseClick(b, Qt.MouseButton.LeftButton)
        self._step(f"click {label!r}", "clicked")
        return True

    _NAV = {"back": QWizard.WizardButton.BackButton, "next": QWizard.WizardButton.NextButton,
            "finish": QWizard.WizardButton.FinishButton,
            "cancel": QWizard.WizardButton.CancelButton}

    def nav(self, which: str) -> QAbstractButton:
        """A wizard navigation button: back, next, finish or cancel."""
        if self.wizard is None:
            raise DriverError("the wizard isn't open")
        return self.wizard.button(self._NAV[which.lower()])

    def nav_usable(self, which: str) -> bool:
        b = self.nav(which)
        return b.isVisible() and b.isEnabled()

    def next(self) -> bool:
        return self._nav(QWizard.WizardButton.NextButton, "Next")

    def back(self) -> bool:
        return self._nav(QWizard.WizardButton.BackButton, "Back")

    def finish(self) -> bool:
        return self._nav(QWizard.WizardButton.FinishButton, "Finish")

    def cancel_wizard(self) -> bool:
        return self._nav(QWizard.WizardButton.CancelButton, "Cancel (wizard)")

    def press(self, key: str, field: Optional[str] = None) -> None:
        """Press a key ("Return", "Escape", "Tab", …) in a field (by its
        label, as for fill) or on the window."""
        w = self.field(field) if field else self.screen()
        if field:
            w.setFocus()
        QTest.keyClick(w, getattr(Qt.Key, f"Key_{key}"))
        self._step(f"press {key}" + (f" in {field!r}" if field else ""), "pressed")

    def answer(self, *answers: str) -> None:
        """Queue answers ("Yes"/"No") for the next question dialogs."""
        self._answers += list(answers)

    def choose_file(self, path: str) -> None:
        """Queue a path for the next file dialog ("" = cancel)."""
        self._files.append(path)

    # -- waiting ---------------------------------------------------------

    def busy(self) -> bool:
        return any(r.running for r in self._runners())

    def wait(self, seconds: float, note: str = "") -> None:
        """Let real time pass (timers fire, processes run)."""
        QTest.qWait(int(seconds * 1000))
        self._step(f"wait {seconds:g}s" + (f" ({note})" if note else ""), "waited")

    def wait_refresh(self, note: str = "board list refresh") -> None:
        """Wait long enough for the app to re-scan the boards (auto
        refresh, and the post-flash re-scan)."""
        t = gui.TIMING
        self.wait((max(t.auto_refresh_ms, t.post_process_refresh_ms) + 300) / 1000, note)

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Wait for any running flash / identify to finish."""
        deadline = time.monotonic() + timeout
        start = time.monotonic()
        while self.busy() and time.monotonic() < deadline:
            QTest.qWait(20)
        QTest.qWait(80)
        ok = not self.busy()
        self._step("wait for the flash / identify to finish",
                   f"finished after {time.monotonic() - start:.1f}s" if ok
                   else f"STILL RUNNING after {timeout:g}s")
        if not ok:
            self.failed_expectations.append(f"process still running after {timeout:g}s")
        return ok

    def wait_for(self, predicate: Callable[[], bool], timeout: float = 5.0,
                 what: str = "condition", required: bool = True) -> bool:
        """Wait until ``predicate()``; a timeout is a failed expectation
        unless ``required=False``."""
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            QTest.qWait(20)
        ok = bool(predicate())
        self._step(f"wait for {what}", "true" if ok else f"still false after {timeout:g}s")
        if not ok and required:
            self.expect(False, f"{what} (timed out after {timeout:g}s)")
        return ok

    # -- checks & records ------------------------------------------------

    def expect(self, condition: Any, message: str) -> bool:
        """Record a check. A failure is reported, not raised, so the
        scenario keeps going and shows what happens next."""
        ok = bool(condition)
        if not ok:
            self.failed_expectations.append(f"step {len(self.steps)}: {message}")
            if self.steps:
                self.steps[-1].problems.append(f"EXPECTED: {message}")
        return ok

    def snap(self, note: str) -> None:
        """Record the current screen without doing anything."""
        self._step(f"look: {note}", "")

    def _step(self, action: str, outcome: str) -> None:
        QTest.qWait(15 if self._fast_timers else 60)
        n = len(self.steps)
        new = self.dialogs[self._dialogs_seen:]
        self._dialogs_seen = len(self.dialogs)
        root = self.screen()
        title = root.windowTitle()
        if isinstance(root, QWizard) and root.currentPage() is not None:
            title += f" — page {root.currentPage().title()!r}"
        step = Step(n, action, outcome, title, self.dump(), new)
        if self.screenshots:
            slug = re.sub(r"[^a-z0-9]+", "-", action.lower()).strip("-")[:40]
            step.image = f"{n:02d}-{slug}.png"
            root.grab().save(str(self.out / step.image))
        self.steps.append(step)

    def write_report(self) -> Path:
        path = self.out / "report.md"
        lines = [f"# GUI session: {self.name}", ""]
        lines += ["## Summary", "",
                  f"- steps: {len(self.steps)}",
                  f"- unhandled exceptions: {len(self.exceptions)}",
                  f"- failed expectations: {len(self.failed_expectations)}",
                  f"- Qt warnings: {len(self.qt_warnings)}",
                  f"- dialogs: {len(self.dialogs)}", ""]
        for title, items in (("Unhandled exceptions", self.exceptions),
                             ("Failed expectations", self.failed_expectations),
                             ("Qt warnings", self.qt_warnings),
                             ("Bench", self.notes)):
            if items:
                lines += [f"## {title}", ""] + [f"- {i}" if "\n" not in i else
                                                f"```\n{i}\n```" for i in items] + [""]
        lines += ["## Steps", ""]
        for s in self.steps:
            lines += [f"### {s.n:02d}. {s.action}", "",
                      f"- outcome: {s.outcome}" if s.outcome else "",
                      f"- window: {s.screen}"]
            lines += [f"- ‼ {p}" for p in s.problems]
            for d in s.dialogs:
                ans = f" → answered {d.answer}" if d.answer else ""
                img = f" ([image]({d.image}))" if d.image else ""
                lines.append(f"- dialog ({d.kind}) {d.title!r}: {_plain(d.text, 600)!r}{ans}{img}")
            if s.image:
                lines += ["", f"![{s.action}]({s.image})"]
            lines += ["", "```", s.dump, "```", ""]
        path.write_text("\n".join(lines))
        return path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    from tests.gui_scenarios import SCENARIOS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenario", nargs="*", help="scenario name(s); default: all")
    ap.add_argument("--list", action="store_true", help="list scenarios")
    ap.add_argument("--out", default="gui-run", help="output directory")
    ap.add_argument("--no-screenshots", action="store_true")
    ap.add_argument("--line-delay", type=float, default=0.01,
                    help="seconds per line of fake build/flash output")
    args = ap.parse_args(argv)
    if args.list:
        for name, fn in SCENARIOS.items():
            print(f"{name:32s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0
    names = args.scenario or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        ap.error(f"unknown scenario(s): {unknown}; see --list")
    bad = 0
    for name in names:
        out = Path(args.out) / name
        with Session(out, screenshots=not args.no_screenshots,
                     line_delay=args.line_delay, name=name) as s:
            try:
                SCENARIOS[name](s)
            except DriverError as exc:
                s.failed_expectations.append(f"driver: {exc}")
        problems = len(s.exceptions) + len(s.failed_expectations)
        bad += bool(problems)
        print(f"{name}: {len(s.steps)} steps, {len(s.exceptions)} exceptions, "
              f"{len(s.failed_expectations)} failed expectations, "
              f"{len(s.qt_warnings)} Qt warnings → {out / 'report.md'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
