"""Alert-mode desktop GUI.

A single Qt window for arming, calibrating, and monitoring an
alert-mode sensor without touching the terminal. Reuses the
existing detect pipeline:

  csidetector.core.detector.MotionDetector  — same scoring + hysteresis
  csidetector.modes.alert.notifier          — Notifier ABC + Event
  csidetector.modes.alert.queue             — durable Telegram delivery
  csidetector.modes.alert.config            — alert.toml loader

The GUI runs the detect loop on a background QThread so the UI stays
responsive. Per-sample state crosses thread boundaries via Qt signals,
not shared mutables.

State (last baseline, last source, last thresholds) persists at
~/.csidetector/state.json so the window launches with sensible
defaults across sessions.
"""

from __future__ import annotations

import json
import os
import platform
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import (
    QObject, QStandardPaths, QThread, Qt, QTimer, QUrl, pyqtSignal, pyqtSlot,
)
from PyQt6.QtGui import (
    QColor, QDesktopServices, QFont, QFontDatabase, QPainter, QPaintEvent, QPen,
)
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

import csidetector
from csidetector.core import collector as csi_collector
from csidetector.core import detector
from csidetector.core import recording
from csidetector.modes.alert.calibrate import collect_amplitudes
from csidetector.modes.alert.config import build_notifier, load_config
from csidetector.modes.alert.notifier import Event, NullNotifier


_STATE_DIR = Path(os.path.expanduser("~/.csidetector"))
_STATE_PATH = _STATE_DIR / "state.json"

# How many recent ratios the sparkline keeps. At ~14 Hz sample rate
# that's ~14 s of history; at 100 Hz, ~2 s. Tradeoff: bigger = more
# context, smaller = noisier paintEvent.
_SPARKLINE_LEN = 200
# How often we recompute the pkt-rate readout (s).
_PKT_RATE_WINDOW_S = 3.0
# Calibration: time to leave the room, then still-room recording time (s).
CALIBRATE_SETTLE_S = 30.0
CALIBRATE_RECORD_S = 30.0
# Say so after this long without a single CSI sample (s).
NO_DATA_WARN_S = 3.0
# The detector window the monitor arms with (recorded with each recording).
DETECT_WINDOW = 50


def recordings_dir() -> Path:
    """Where Record saves: Documents/CSIDetector recordings, easy to find
    and attach to an email or a chat."""
    docs = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
    return Path(docs or Path.home()) / "CSIDetector recordings"


def _is_serial(source: str) -> bool:
    """A serial port the collector opens (/dev/tty*, /dev/cu.*, by-id, COMn)."""
    return source.startswith("/dev/") or source.upper().startswith("COM")


def _friendly_source_error(source: str, exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if "could not open port" in str(exc):
        return (f"Couldn't open the receiver's port {source}. Is the receiver "
                f"plugged into this computer?\n\n({text})")
    return text


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

@dataclass
class _PersistedState:
    source: str = "COM3" if os.name == "nt" else "/dev/ttyACM0"
    baseline: float = 0.0
    baseline_ts: float = 0.0
    enter_ratio: float = 1.7
    exit_ratio: float = 1.3
    location: str = ""
    alert_config_path: str = ""
    # Which boards the baseline was calibrated with (set by the setup
    # wizard), and how to find the receiver again if its port number
    # changes.
    receiver_mac: str = ""
    transmitter_mac: str = ""
    channel: int = 0             # Wi-Fi channel the baseline was taken on

    @classmethod
    def load(cls) -> "_PersistedState":
        if not _STATE_PATH.exists():
            return cls()
        try:
            raw = json.loads(_STATE_PATH.read_text())
            return cls(**{k: raw.get(k, getattr(cls(), k)) for k in cls.__dataclass_fields__})
        except (OSError, json.JSONDecodeError, TypeError):
            return cls()

    def save(self) -> None:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(asdict(self), indent=2))


# --------------------------------------------------------------------------
# Background workers. QObject subclasses moved to QThread; communicate
# with the UI exclusively via Qt signals (no shared dict mutations).
# --------------------------------------------------------------------------

class _DetectWorker(QObject):
    """Runs the detect loop and emits per-sample signals."""

    ratio_updated = pyqtSignal(float, float)   # score, ratio
    state_changed = pyqtSignal(str)             # 'INIT' | 'STILL' | 'MOTION'
    pkt_rate_updated = pyqtSignal(float)
    alert_sent = pyqtSignal(float)              # ts of the alert
    no_data = pyqtSignal(float)                 # seconds without CSI; 0 = resumed
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, source: str, baseline: float,
                 enter_ratio: float, exit_ratio: float,
                 notifier, location: str, cooldown_s: float = 60.0,
                 window: int = 50, settle_s: float = 0.0):
        super().__init__()
        self._source = source
        self._baseline = baseline
        self._enter = enter_ratio
        self._exit = exit_ratio
        self._notifier = notifier
        self._location = location
        self._cooldown_s = cooldown_s
        self._window = window
        self._settle_s = settle_s
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            self._run_loop()
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(_friendly_source_error(self._source, exc))
        finally:
            try:
                self._notifier.close()
            except Exception:  # noqa: BLE001
                pass
            self.finished.emit()

    def _run_loop(self) -> None:
        cfg = detector.DetectorConfig(
            window=self._window,
            enter_ratio=self._enter,
            exit_ratio=self._exit,
        )
        det: Optional[detector.MotionDetector] = None
        # None until the detector is warmed up, so its first verdict (STILL
        # included) is reported; the badge says INITIALIZING until then.
        last_motion: Optional[bool] = None
        last_alert_ts = 0.0
        ts_history: list[float] = []   # recent push timestamps for pkt rate
        last_rate_emit = 0.0
        self.state_changed.emit("INIT")
        settle_until = time.time() + self._settle_s
        quiet = {"since": time.time(), "warned": False}

        def idle() -> None:
            gap = time.time() - quiet["since"]
            if gap >= NO_DATA_WARN_S:
                quiet["warned"] = True
                self.no_data.emit(gap)
                self.pkt_rate_updated.emit(0.0)

        for sample in csi_collector.open_source(self._source, stop=self._stop,
                                                on_idle=idle):
            if self._stop.is_set():
                return
            now = time.time()
            quiet["since"] = now
            if quiet["warned"]:
                quiet["warned"] = False
                self.no_data.emit(0.0)
            ts_history.append(now)
            cutoff = now - _PKT_RATE_WINDOW_S
            while ts_history and ts_history[0] < cutoff:
                ts_history.pop(0)
            # Throttle the rate emit to ~3 Hz so the UI label isn't
            # repainted on every CSI sample (saves a lot of work at
            # 100 Hz streams).
            if now - last_rate_emit > 0.33:
                self.pkt_rate_updated.emit(len(ts_history) / _PKT_RATE_WINDOW_S)
                last_rate_emit = now

            if now < settle_until:
                continue

            if det is None:
                idx = np.flatnonzero(sample.amplitude > 0)
                if idx.size == 0:
                    continue
                det = detector.MotionDetector(idx, self._baseline, cfg)

            score, motion = det.update(sample.amplitude)
            if not det.ready:
                continue                  # no score until the window is full
            ratio = score / det.baseline if det.baseline else 0.0
            self.ratio_updated.emit(score, ratio)

            if motion != last_motion:
                kind = "MOTION" if motion else "STILL"
                self.state_changed.emit(kind)
                should_notify = (
                    motion and (now - last_alert_ts >= self._cooldown_s)
                )
                if should_notify:
                    last_alert_ts = now
                    self._dispatch(kind, ratio)
                    self.alert_sent.emit(now)
                last_motion = motion

    def _dispatch(self, kind: str, ratio: float) -> None:
        parts = []
        if self._location:
            parts.append(self._location)
        parts.append(kind)
        parts.append(f"ratio={ratio:.2f}×")
        ev = Event.now(kind=kind, message=" | ".join(parts))
        try:
            self._notifier.send(ev)
        except Exception as exc:  # noqa: BLE001
            # The notifier may be the QueuingNotifier wrapper; transient
            # / permanent failures are logged by the queue worker
            # itself. Surface here just for visibility — don't propagate.
            self.error_occurred.emit(f"notifier: {exc}")


class _CalibrateWorker(QObject):
    """Runs the calibration loop with progress signals."""

    phase_changed = pyqtSignal(str, float)   # phase name, remaining seconds
    done = pyqtSignal(float, int)             # baseline, n_samples
    no_data = pyqtSignal(float)               # seconds without CSI; 0 = resumed
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, source: str, settle_s: Optional[float] = None,
                 record_s: Optional[float] = None, window: int = 50):
        super().__init__()
        self._source = source
        # Looked up at call time (not as default args) so they can be tuned.
        self._settle_s = CALIBRATE_SETTLE_S if settle_s is None else settle_s
        self._record_s = CALIBRATE_RECORD_S if record_s is None else record_s
        self._window = window
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            self._run_inner()
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(_friendly_source_error(self._source, exc))
        finally:
            self.finished.emit()

    def _run_inner(self) -> None:
        rows: list[np.ndarray] = []
        idx: Optional[np.ndarray] = None
        start = time.time()
        settle_until = start + self._settle_s
        deadline = settle_until + self._record_s
        st = {"emit": 0.0, "sample": start, "warned": False, "timed_out": False}

        def phase(now: float) -> None:
            # The countdown keeps ticking with or without data.
            if now - st["emit"] > 0.2:
                if now < settle_until:
                    self.phase_changed.emit("LEAVE THE ROOM",
                                            max(0.0, settle_until - now))
                else:
                    self.phase_changed.emit("RECORDING",
                                            max(0.0, deadline - now))
                st["emit"] = now

        def idle() -> None:
            now = time.time()
            phase(now)
            if now - st["sample"] >= NO_DATA_WARN_S:
                st["warned"] = True
                self.no_data.emit(now - st["sample"])
            if now >= deadline:
                st["timed_out"] = True
                self._stop.set()         # ends the source iterator

        for sample in csi_collector.open_source(self._source, stop=self._stop,
                                                on_idle=idle):
            if self._stop.is_set():
                break
            now = time.time()
            st["sample"] = now
            if st["warned"]:
                st["warned"] = False
                self.no_data.emit(0.0)
            phase(now)
            if now < settle_until:
                continue
            if idx is None:
                idx = np.flatnonzero(sample.amplitude > 0)
                if idx.size == 0:
                    continue
            rows.append(sample.amplitude)
            if now >= deadline:
                break

        if self._stop.is_set() and not st["timed_out"]:
            return                       # cancelled
        if not rows or idx is None:
            raise RuntimeError(
                f"No CSI data arrived from {self._source}. Check that the "
                f"transmitter is powered and was flashed for the same channel "
                f"as the receiver, and that this is the receiver's port.")
        if len(rows) < 3 * self._window:
            raise RuntimeError(
                f"Only {len(rows)} CSI samples arrived in {self._record_s:.0f} s "
                f"(need {3 * self._window}). Is the transmitter powered and "
                f"close enough?")
        amps = np.stack(rows)[:, idx]
        baseline = detector.compute_baseline(amps, self._window)
        self.done.emit(float(baseline), amps.shape[0])


class _RecordWorker(QObject):
    """Saves the receiver's raw CSI lines to a recording file: waits
    ``delay_s`` (time to get into position), then records ``length_s``."""

    progress = pyqtSignal(str, float, int)    # 'WAIT' | 'REC', seconds left, samples
    no_data = pyqtSignal(float)               # seconds without CSI; 0 = resumed
    saved = pyqtSignal(str, object)           # path, summary
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, source: str, folder: Path, meta: dict,
                 delay_s: float, length_s: float):
        super().__init__()
        self._source = source
        self._folder = Path(folder)
        self._meta = dict(meta)
        self._delay_s = delay_s
        self._length_s = length_s
        self._stop = threading.Event()
        self._stopped_by = "stopped"
        self._marks: "queue.SimpleQueue[None]" = queue.SimpleQueue()
        self._writer: Optional[recording.RecordingWriter] = None
        self._end_at = 0.0

    def request_stop(self, reason: str = "stopped") -> None:
        if not self._stop.is_set():
            self._stopped_by = reason
        self._stop.set()

    def add_mark(self) -> None:
        """From the GUI thread: written by the worker with its time."""
        self._marks.put(None)

    @pyqtSlot()
    def run(self) -> None:
        try:
            self._run_inner()
        except Exception as exc:  # noqa: BLE001
            self._stopped_by = "error"
            self.error_occurred.emit(_friendly_source_error(self._source, exc))
        finally:
            try:
                self._finish_file()
            except Exception as exc:  # noqa: BLE001
                self.error_occurred.emit(f"Couldn't finish the recording: {exc}")
            self.finished.emit()

    def _run_inner(self) -> None:
        self._meta.setdefault("git_commit", _git_commit())
        start_at = time.time() + self._delay_s
        st = {"emit": 0.0, "data": time.time(), "warned": False}

        def tick(now: float) -> None:
            if self._writer is None and now >= start_at and not self._stop.is_set():
                self._open(now)
            w = self._writer
            if w is not None:
                while True:
                    try:
                        self._marks.get_nowait()
                    except queue.Empty:
                        break
                    w.mark()
                if now >= self._end_at:
                    self.request_stop("timer")
            if now - st["emit"] >= 0.25:
                st["emit"] = now
                if w is None:
                    self.progress.emit("WAIT", max(0.0, start_at - now), 0)
                else:
                    self.progress.emit("REC", max(0.0, self._end_at - now), w.samples)

        def idle() -> None:
            now = time.time()
            tick(now)
            if now - st["data"] >= NO_DATA_WARN_S:
                st["warned"] = True
                self.no_data.emit(now - st["data"])

        for line in csi_collector.iter_serial_lines(self._source, stop=self._stop,
                                                    on_idle=idle):
            now = time.time()
            tick(now)
            if self._stop.is_set():
                break
            if line.startswith("CSI_DATA,"):
                st["data"] = now
                if st["warned"]:
                    st["warned"] = False
                    self.no_data.emit(0.0)
            if self._writer is not None:
                self._writer.write(line)

    def _open(self, now: float) -> None:
        meta = dict(self._meta)
        meta["started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now))
        meta["started_unix"] = round(now, 3)
        self._writer = recording.new_recording(self._folder, meta.get("condition", ""),
                                               meta, now)
        self._end_at = now + self._length_s

    def _finish_file(self) -> None:
        w, self._writer = self._writer, None
        if w is None:
            return                         # stopped before recording began
        while True:                        # marks pressed just before the end
            try:
                self._marks.get_nowait()
            except queue.Empty:
                break
            w.mark()
        if w.samples == 0:
            w.abandon()
            if self._stopped_by != "error":
                self.error_occurred.emit(
                    f"No CSI arrived from {self._source} while recording, so "
                    f"nothing was saved. Check that the transmitter is powered "
                    f"and was flashed for the same channel as the receiver.")
            return
        summary = w.close(self._stopped_by)
        self.saved.emit(str(w.path), summary)


def _git_commit() -> str:
    """The checked-out commit (``+dirty`` with local edits), so a recording
    says which code made it."""
    here = Path(__file__).resolve().parent
    kw = dict(cwd=here, capture_output=True, text=True, timeout=3,
              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], **kw)
    except (OSError, subprocess.SubprocessError):
        return ""
    if head.returncode != 0:
        return ""
    commit = head.stdout.strip()
    try:
        # --no-optional-locks: never take index.lock in the user's checkout.
        dirty = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain",
                                "--untracked-files=no"], **kw)
    except (OSError, subprocess.SubprocessError):
        return commit
    return commit + ("+dirty" if dirty.stdout.strip() else "")


# Board settings a recording may carry: what shapes the CSI, nothing that
# names the network or could be secret.
_BOARD_CONFIG_KEYS = ("channel", "rate_hz", "tx_macs", "mode")


def _board_info(mac: str) -> dict:
    """What the setup app knows about a board (label, chip, firmware,
    channel). Read-only, and never anything secret."""
    if not mac:
        return {}
    try:
        raw = json.loads((_STATE_DIR / "devices.json").read_text())
        rec = raw.get("devices", {}).get(mac, {})
    except (OSError, ValueError, AttributeError):
        return {}
    if not isinstance(rec, dict):
        return {}
    out = {k: rec[k] for k in ("label", "role", "chip", "firmware") if k in rec}
    config = rec.get("config")
    if isinstance(config, dict):
        out["config"] = {k: config[k] for k in _BOARD_CONFIG_KEYS if k in config}
    return out


def _writable_recordings_dir() -> tuple[Path, str]:
    """(folder, note): Documents/CSIDetector recordings, or the home
    folder's if Documents can't be written to (e.g. Windows' Controlled
    folder access). Checked before recording starts, not after the user
    has left the room. Raises OSError if neither works."""
    first = recordings_dir()
    error: Optional[OSError] = None
    for folder in (first, Path.home() / "CSIDetector recordings"):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / f".write-test-{os.getpid()}"
            probe.write_bytes(b"")
            probe.unlink()
        except OSError as exc:
            error = error or exc
            continue
        note = "" if folder == first else f" ({first} can't be written to: {error})"
        return folder, note
    raise error or OSError(f"can't write to {first}")


# --------------------------------------------------------------------------
# Custom widgets
# --------------------------------------------------------------------------

class _Sparkline(QWidget):
    """Simple ratio-vs-time line graph drawn with QPainter."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._values: list[float] = []
        self._threshold = 1.7
        self.setMinimumHeight(60)

    def push(self, value: float) -> None:
        self._values.append(value)
        if len(self._values) > _SPARKLINE_LEN:
            self._values = self._values[-_SPARKLINE_LEN:]
        self.update()

    def set_threshold(self, value: float) -> None:
        self._threshold = value
        self.update()

    def paintEvent(self, ev: QPaintEvent) -> None:  # noqa: N802
        p = QPainter(self)
        try:
            r = self.rect()
            p.fillRect(r, QColor("#1a1a1a"))
            if not self._values:
                return
            max_y = max(max(self._values), self._threshold * 1.2, 1.0)
            min_y = 0.0
            scale_y = (r.height() - 8) / (max_y - min_y) if max_y > min_y else 0.0
            # Threshold line
            ty = r.bottom() - int((self._threshold - min_y) * scale_y) - 4
            p.setPen(QPen(QColor("#d94545"), 1, Qt.PenStyle.DashLine))
            p.drawLine(r.left() + 2, ty, r.right() - 2, ty)
            # Trace
            p.setPen(QPen(QColor("#9eb1c6"), 1))
            n = len(self._values)
            step_x = (r.width() - 4) / max(n - 1, 1)
            prev_x = r.left() + 2
            prev_y = r.bottom() - int((self._values[0] - min_y) * scale_y) - 4
            for i in range(1, n):
                x = int(r.left() + 2 + i * step_x)
                y = r.bottom() - int((self._values[i] - min_y) * scale_y) - 4
                p.drawLine(prev_x, prev_y, x, y)
                prev_x, prev_y = x, y
        finally:
            p.end()


class _StatusBadge(QLabel):
    """Big color-coded state badge."""

    _STYLES = {
        "DISARMED": ("DISARMED",         "#666666"),
        "INIT":     ("INITIALIZING…",    "#dca035"),
        "STILL":    ("ARMED — STILL",    "#2faa55"),
        "MOTION":   ("MOTION DETECTED",  "#d94545"),
        "ERROR":    ("ERROR",            "#b54a4a"),
        "CALIB":    ("CALIBRATING",      "#1c7eb6"),
        "REC":      ("RECORDING",        "#7a3fb0"),
    }

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        f = QFont()
        f.setPointSize(20)
        f.setBold(True)
        self.setFont(f)
        self.set_state("DISARMED")

    def set_state(self, name: str, extra: str = "") -> None:
        text, color = self._STYLES.get(name, ("?", "#444"))
        if extra:
            text = f"{text}   {extra}"
        self.setText(text)
        self.setStyleSheet(
            f"background-color: {color}; color: white; "
            f"padding: 18px; border-radius: 8px;")


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class CSIDetectorWindow(QMainWindow):
    test_result = pyqtSignal(bool, str)      # Test alert, from its thread

    def __init__(self, alert_config_path: str = ""):
        super().__init__()
        self.setWindowTitle("CSIDetector — Alert Mode")
        self.setMinimumSize(560, 520)

        self._state = _PersistedState.load()
        if alert_config_path:
            self._state.alert_config_path = alert_config_path

        # --- widgets -----------------------------------------------------
        self._badge = _StatusBadge()
        self._ratio_label = QLabel("Ratio: — (threshold —)")
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self._ratio_label.setFont(mono)
        self._ratio_bar = QProgressBar()
        self._ratio_bar.setRange(0, 300)   # 0.00× to 3.00×
        self._ratio_bar.setTextVisible(False)
        self._ratio_bar.setVisible(False)  # empty, it reads as a text box
        self._no_data = QLabel("")
        self._no_data.setWordWrap(True)
        self._no_data.setStyleSheet("color: #b54a4a; font-weight: bold;")
        self._no_data.setVisible(False)
        self._sparkline = _Sparkline()
        self._sparkline.set_threshold(self._state.enter_ratio)

        self._stats = QLabel("")
        self._stats.setFont(mono)
        self._stats.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self._calibrate_btn = QPushButton("Calibrate")
        self._arm_btn = QPushButton("Arm")
        self._disarm_btn = QPushButton("Disarm")
        self._test_btn = QPushButton("Test alert")
        self._edit_btn = QPushButton("Edit config")

        self._calibrate_btn.clicked.connect(self._on_calibrate)
        self._arm_btn.clicked.connect(self._on_arm)
        self._disarm_btn.clicked.connect(self._on_disarm)
        self._test_btn.clicked.connect(self._on_test)
        self._edit_btn.clicked.connect(self._on_edit)
        self._disarm_btn.setEnabled(False)
        self.test_result.connect(self._on_test_result)

        # Record: raw CSI to a file for working on detection offline.
        self._rec_condition = QComboBox()
        for key, label in recording.CONDITIONS:
            self._rec_condition.addItem(label, key)
        self._rec_note = QLineEdit()
        self._rec_note.setPlaceholderText("optional: e.g. door open, two people, TX on the shelf")
        self._rec_delay = QSpinBox()
        self._rec_delay.setRange(0, 300)
        self._rec_delay.setValue(15)
        self._rec_delay.setSuffix(" s")
        self._rec_delay.setToolTip("Time to get into position first (leave the "
                                   "room for an empty-room recording).")
        self._rec_length = QSpinBox()
        self._rec_length.setRange(1, 60)
        self._rec_length.setValue(2)
        self._rec_length.setSuffix(" min")
        self._record_btn = QPushButton("Record")
        self._mark_btn = QPushButton("Mark")
        self._mark_btn.setToolTip("Press when something happens (someone walks "
                                  "in, sits down, leaves); saved with the time.")
        self._mark_btn.setEnabled(False)
        self._open_rec_btn = QPushButton("Open folder")
        self._rec_status = QLabel(
            f"Saves the receiver's raw CSI to a file in {recordings_dir()}, "
            f"one file per recording, to send to the developers.")
        self._rec_status.setWordWrap(True)
        self._rec_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self._record_btn.clicked.connect(self._on_record)
        self._mark_btn.clicked.connect(self._on_mark)
        self._open_rec_btn.clicked.connect(self._on_open_recordings)

        # --- layout ------------------------------------------------------
        central = QWidget()
        v = QVBoxLayout(central)
        v.addWidget(self._badge)
        v.addWidget(self._no_data)
        v.addWidget(self._ratio_label)
        v.addWidget(self._ratio_bar)
        v.addWidget(self._sparkline, stretch=1)
        v.addWidget(self._stats)
        h = QHBoxLayout()
        for b in (self._calibrate_btn, self._arm_btn, self._disarm_btn,
                  self._test_btn, self._edit_btn):
            h.addWidget(b)
        v.addLayout(h)
        rec_box = QGroupBox("Record data for the developers")
        rv = QVBoxLayout(rec_box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Condition:"))
        row.addWidget(self._rec_condition, 1)
        rv.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Note:"))
        row.addWidget(self._rec_note, 1)
        rv.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Start in:"))
        row.addWidget(self._rec_delay)
        row.addWidget(QLabel("Length:"))
        row.addWidget(self._rec_length)
        row.addStretch(1)
        for b in (self._record_btn, self._mark_btn, self._open_rec_btn):
            row.addWidget(b)
        rv.addLayout(row)
        rv.addWidget(self._rec_status)
        v.addWidget(rec_box)
        self.setCentralWidget(central)

        # --- worker state ------------------------------------------------
        self._detect_thread: Optional[QThread] = None
        self._detect_worker: Optional[_DetectWorker] = None
        self._calib_thread: Optional[QThread] = None
        self._calib_worker: Optional[_CalibrateWorker] = None
        self._rec_thread: Optional[QThread] = None
        self._rec_worker: Optional[_RecordWorker] = None
        self._rec_saved = False           # this recording produced a file
        self._rec_rate_ref: Optional[tuple[float, int]] = None   # (time, samples)
        self._rec_folder: Optional[Path] = None   # where the last recording went
        # Set on close: a calibration result still queued from the worker
        # must not be saved over state written after the window closed.
        self._closed = False
        self._notifier_active = None      # the queuing notifier currently in use
        self._last_alert_ts: float = 0.0
        self._pkt_rate: float = 0.0

        self._refresh_stats()
        # Periodically refresh queue size + "last alert" relative time.
        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._refresh_stats)
        self._stats_timer.start(1000)

    # ----------------------------------------------------------------------
    # Calibrate
    # ----------------------------------------------------------------------

    def _on_calibrate(self) -> None:
        if self._calib_worker is not None:         # the button reads "Cancel"
            self._badge.set_state("CALIB", "stopping…")
            self._calib_worker.request_stop()
            return
        if self._detect_thread is not None or self._rec_thread is not None:
            return
        source = self._resolve_source()
        self._badge.set_state("CALIB", "starting…")
        self._calibrate_btn.setText("Cancel calibration")
        self._arm_btn.setEnabled(False)
        self._record_btn.setEnabled(False)

        thread = QThread(self)
        worker = _CalibrateWorker(source=source)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.phase_changed.connect(self._on_calib_phase)
        worker.done.connect(self._on_calib_done)
        worker.no_data.connect(self._on_no_data)
        worker.error_occurred.connect(self._on_calib_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_calib_finished)
        thread.start()
        self._calib_thread = thread
        self._calib_worker = worker

    def _on_calib_phase(self, phase: str, remaining: float) -> None:
        self._badge.set_state("CALIB", f"{phase} — {remaining:.0f}s")

    def _on_calib_done(self, baseline: float, n_samples: int) -> None:
        if self._closed:
            return
        self._state.baseline = baseline
        self._state.baseline_ts = time.time()
        self._state.save()
        QMessageBox.information(
            self, "Calibration complete",
            f"Baseline: {baseline:.4f}\n"
            f"Samples: {n_samples}\n\n"
            "Typical still-room σ is 0.05–1.5. Values above 2 mean the "
            "room wasn't actually still during recording — consider "
            "re-running.")

    def _on_calib_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Calibration failed", msg)

    def _on_calib_finished(self) -> None:
        self._calib_thread = None
        self._calib_worker = None
        self._calibrate_btn.setText("Calibrate")
        self._calibrate_btn.setEnabled(True)
        self._arm_btn.setEnabled(True)
        self._record_btn.setEnabled(True)
        self._on_no_data(0.0)
        self._badge.set_state("DISARMED")
        self._refresh_stats()

    # ----------------------------------------------------------------------
    # Arm / Disarm
    # ----------------------------------------------------------------------

    def _on_arm(self) -> None:
        if self._state.baseline <= 0:
            QMessageBox.warning(
                self, "No baseline",
                "Run Calibrate first — there's no still-room baseline to "
                "compare current σ against.")
            return
        if self._detect_thread is not None or self._rec_thread is not None:
            return

        # Construct notifier from alert.toml (or NullNotifier).
        try:
            cfg = load_config(self._state.alert_config_path) \
                  if self._state.alert_config_path else {}
            notifier = build_notifier(cfg)
        except SystemExit as exc:
            QMessageBox.warning(self, "Config error", str(exc))
            return
        self._notifier_active = notifier
        location = (
            cfg.get("alert", {}).get("location_label", "")
            if isinstance(cfg, dict) else ""
        )

        worker = _DetectWorker(
            source=self._resolve_source(),
            baseline=self._state.baseline,
            enter_ratio=self._state.enter_ratio,
            exit_ratio=self._state.exit_ratio,
            notifier=notifier,
            location=location,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.ratio_updated.connect(self._on_ratio)
        worker.state_changed.connect(self._on_state)
        worker.pkt_rate_updated.connect(self._on_pkt_rate)
        worker.alert_sent.connect(self._on_alert)
        worker.no_data.connect(self._on_no_data)
        worker.error_occurred.connect(self._on_detect_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_detect_finished)
        thread.start()
        self._detect_thread = thread
        self._detect_worker = worker

        self._arm_btn.setEnabled(False)
        self._calibrate_btn.setEnabled(False)
        self._record_btn.setEnabled(False)
        self._disarm_btn.setEnabled(True)
        self._badge.set_state("INIT")

    def _on_disarm(self) -> None:
        if self._detect_worker is None:
            return
        self._badge.set_state("DISARMED", "stopping…")
        self._detect_worker.request_stop()
        # iter_serial checks the stop flag on every 1 s read timeout too,
        # so this takes effect within ~1 s even with no CSI arriving. UI
        # re-enables in _on_detect_finished.

    def _on_ratio(self, score: float, ratio: float) -> None:
        self._ratio_label.setText(
            f"Ratio: {ratio:5.2f}×  (threshold {self._state.enter_ratio:.2f}×)  "
            f"σ={score:.3f}")
        self._ratio_bar.setVisible(True)
        self._ratio_bar.setValue(int(min(ratio, 3.0) * 100))
        self._sparkline.push(ratio)

    def _on_state(self, state: str) -> None:
        if state == "MOTION":
            self._badge.set_state("MOTION")
        elif state == "STILL":
            self._badge.set_state("STILL")
        elif state == "INIT":
            self._badge.set_state("INIT")

    def _on_pkt_rate(self, hz: float) -> None:
        self._pkt_rate = hz

    def _on_alert(self, ts: float) -> None:
        self._last_alert_ts = ts
        self._refresh_stats()

    def _on_detect_error(self, msg: str) -> None:
        # Show but don't tear down — the worker may have already cleaned up.
        QMessageBox.warning(self, "Detect error", msg)

    def _on_detect_finished(self) -> None:
        self._detect_thread = None
        self._detect_worker = None
        self._notifier_active = None
        self._arm_btn.setEnabled(True)
        self._calibrate_btn.setEnabled(True)
        self._record_btn.setEnabled(True)
        self._disarm_btn.setEnabled(False)
        self._badge.set_state("DISARMED")
        self._ratio_bar.setVisible(False)
        self._on_no_data(0.0)
        self._pkt_rate = 0.0
        self._refresh_stats()

    def _on_no_data(self, seconds: float) -> None:
        if seconds <= 0:
            self._no_data.setVisible(False)
            return
        self._no_data.setText(
            f"⚠ No CSI from {self._state.source} for {seconds:.0f} s. Is the "
            f"transmitter powered, and flashed for the same channel as the "
            f"receiver?")
        self._no_data.setVisible(True)

    def _resolve_source(self) -> str:
        """The receiver's port now. /dev/ttyACM* numbers change when boards
        are replugged; a native-USB receiver is found again by its MAC
        (its USB serial number). Bridge boards can't be, so their saved
        port (ideally a /dev/serial/by-id link) is used as is."""
        source, mac = self._state.source, self._state.receiver_mac
        # Not a /dev/serial/by-id link, which already names the board.
        if mac and _is_serial(source) and not source.startswith("/dev/serial/by-id/"):
            from csidetector.provision import devices as dv
            try:
                boards = dv.list_boards()
            except Exception:  # noqa: BLE001 — no pyserial / no ports
                boards = []
            port = next((b.port for b in boards if b.mac == mac), "")
            if port and port != source:
                self._state.source = port
                self._state.save()
                self._refresh_stats()
        return self._state.source

    # ----------------------------------------------------------------------
    # Record
    # ----------------------------------------------------------------------

    def _on_record(self) -> None:
        if self._rec_worker is not None:          # the button reads "Stop"
            self._rec_worker.request_stop("stop button")
            self._record_btn.setEnabled(False)    # until the file is closed
            self._mark_btn.setEnabled(False)
            return
        if self._detect_thread is not None or self._calib_thread is not None:
            return
        source = self._resolve_source()
        if not _is_serial(source):
            QMessageBox.information(
                self, "Can't record from here",
                f"Recording reads the receiver's serial port, but this monitor "
                f"reads {source}. Run the setup wizard to point it at the "
                f"receiver.")
            return
        try:
            folder, folder_note = _writable_recordings_dir()
        except OSError as exc:
            QMessageBox.warning(self, "Can't save recordings",
                                f"Recordings go in {recordings_dir()}, but it can't be "
                                f"written to:\n\n{exc}")
            return
        self._rec_folder = folder
        st = self._state
        key = self._rec_condition.currentData()
        meta = {
            "format": recording.FORMAT,
            "condition": key,
            "condition_label": self._rec_condition.currentText(),
            "note": self._rec_note.text().strip(),
            "source": source,
            "receiver_mac": st.receiver_mac,
            "transmitter_mac": st.transmitter_mac,
            "receiver": _board_info(st.receiver_mac),
            "transmitter": _board_info(st.transmitter_mac),
            "channel": st.channel,
            "baseline": st.baseline,
            "baseline_age_s": round(time.time() - st.baseline_ts) if st.baseline_ts else None,
            "enter_ratio": st.enter_ratio,
            "exit_ratio": st.exit_ratio,
            "detector_window": DETECT_WINDOW,
            "start_delay_s": self._rec_delay.value(),
            "planned_length_s": self._rec_length.value() * 60,
            "app_version": csidetector.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
        }
        worker = _RecordWorker(source, folder, meta,
                               delay_s=float(self._rec_delay.value()),
                               length_s=float(self._rec_length.value() * 60))
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_rec_progress)
        worker.no_data.connect(self._on_no_data)
        worker.saved.connect(self._on_rec_saved)
        worker.error_occurred.connect(self._on_rec_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_rec_finished)
        thread.start()
        self._rec_thread = thread
        self._rec_worker = worker
        self._rec_saved = False
        self._rec_rate_ref = None

        self._record_btn.setText("Stop recording")
        self._arm_btn.setEnabled(False)
        self._calibrate_btn.setEnabled(False)
        for w in (self._rec_condition, self._rec_note, self._rec_delay, self._rec_length):
            w.setEnabled(False)
        self._badge.set_state("REC", "starting…")
        self._rec_status.setText(f"Recording: {self._rec_condition.currentText()}, "
                                 f"into {folder}.{folder_note}")

    def _on_rec_progress(self, phase: str, seconds_left: float, samples: int) -> None:
        left = f"{int(seconds_left) // 60}:{int(seconds_left) % 60:02d}"
        if phase == "WAIT":
            self._badge.set_state("REC", f"starts in {left}")
            return
        self._mark_btn.setEnabled(True)
        self._badge.set_state("REC", f"{left} left · {samples:,} samples")
        # The packet rate while recording, so a thin stream shows up now
        # rather than in the file.
        now = time.monotonic()
        prev = self._rec_rate_ref
        if prev is not None and now - prev[0] >= 1.0:
            self._pkt_rate = max(0.0, (samples - prev[1]) / (now - prev[0]))
            self._rec_rate_ref = (now, samples)
            self._refresh_stats()
        elif prev is None:
            self._rec_rate_ref = (now, samples)

    def _on_mark(self) -> None:
        if self._rec_worker is not None:
            self._rec_worker.add_mark()

    def _on_rec_saved(self, path: str, summary: dict) -> None:
        self._rec_saved = True
        if self._closed:
            return
        p = Path(path)
        try:
            n = p.stat().st_size
            size = f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{max(1, round(n / 1e3))} KB"
        except OSError:
            size = "?"
        secs = int(summary.get("duration_s", 0))
        text = (f"✓ Saved {p.name} ({size}, {secs // 60}:{secs % 60:02d}, "
                f"{summary.get('samples', 0):,} samples at "
                f"{summary.get('rate_hz', 0):.0f} Hz) in {p.parent}. "
                f"Send this file to the developers.")
        if summary.get("stopped_by") == "error":
            text += " ⚠ It stopped early (see the message); what came before is kept."
        if summary.get("gaps_over_1s"):
            text += (f" ⚠ The stream paused {summary['gaps_over_1s']} time(s) "
                     f"for over a second (longest {summary.get('max_gap_s', 0):.0f} s).")
        self._rec_status.setText(text)

    def _on_rec_error(self, msg: str) -> None:
        if self._closed:
            return
        if not self._rec_saved:
            self._rec_status.setText("✗ " + msg.split("\n")[0])
        QMessageBox.warning(self, "Recording problem", msg)

    def _on_rec_finished(self) -> None:
        self._rec_thread = None
        self._rec_worker = None
        if self._closed:
            return
        if not self._rec_saved and self._rec_status.text().startswith("Recording:"):
            self._rec_status.setText("Stopped before recording began: nothing saved.")
        self._record_btn.setText("Record")
        self._record_btn.setEnabled(True)
        self._mark_btn.setEnabled(False)
        self._arm_btn.setEnabled(True)
        self._calibrate_btn.setEnabled(True)
        for w in (self._rec_condition, self._rec_note, self._rec_delay, self._rec_length):
            w.setEnabled(True)
        self._on_no_data(0.0)
        self._pkt_rate = 0.0
        self._refresh_stats()
        self._badge.set_state("DISARMED")

    def _on_open_recordings(self) -> None:
        """The folder recordings actually went to (the home folder's when
        Documents couldn't be written to)."""
        folder = self._rec_folder
        if folder is None or not folder.is_dir():
            try:
                folder, _note = _writable_recordings_dir()
            except OSError as exc:
                QMessageBox.warning(self, "Can't open the folder", str(exc))
                return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # ----------------------------------------------------------------------
    # Test alert + edit config
    # ----------------------------------------------------------------------

    _NO_TELEGRAM = ("Telegram isn't set up, so alerts only show in this window. "
                    "To get messages on your phone, run the setup wizard "
                    "(\"Setup wizard…\" in the setup app, or "
                    "`python run.py setup --wizard`) and fill in the Telegram "
                    "page.")

    def _on_test(self) -> None:
        """Send one message straight to Telegram (not via the retry queue)
        and report what actually happened."""
        if not self._state.alert_config_path:
            QMessageBox.information(self, "Telegram not set up", self._NO_TELEGRAM)
            return
        from csidetector.modes.alert import telegram
        try:
            cfg = load_config(self._state.alert_config_path)
            sender = telegram.from_config(cfg)
        except (SystemExit, ValueError) as exc:
            QMessageBox.warning(self, "alert.toml problem", str(exc))
            return
        if sender is None:
            QMessageBox.information(self, "Telegram not set up", self._NO_TELEGRAM)
            return
        self._test_btn.setEnabled(False)
        self._test_btn.setText("Sending…")

        def run() -> None:
            try:
                sender.send(Event.now(kind="TEST", message="CSIDetector test alert ✓"))
                ok, msg = True, "Sent. Check Telegram on your phone."
            except Exception as exc:  # noqa: BLE001 — report whatever went wrong
                ok, msg = False, f"{type(exc).__name__}: {exc}"
            try:
                self.test_result.emit(ok, msg)
            except RuntimeError:
                pass                      # window closed meanwhile

        threading.Thread(target=run, daemon=True).start()

    def _on_test_result(self, ok: bool, message: str) -> None:
        self._test_btn.setEnabled(True)
        self._test_btn.setText("Test alert")
        if ok:
            QMessageBox.information(self, "Test alert sent", message)
        else:
            QMessageBox.warning(self, "Test alert failed",
                                f"{message}\n\nCheck the bot token and chat ID "
                                f"(re-run the setup wizard to change them).")

    def _on_edit(self) -> None:
        if not self._state.alert_config_path:
            QMessageBox.information(self, "Telegram not set up", self._NO_TELEGRAM)
            return
        # Open in the system default editor for .toml. xdg-open on Linux,
        # 'open' on macOS, default association on Windows.
        try:
            if platform.system() == "Linux":
                subprocess.Popen(["xdg-open", self._state.alert_config_path])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", self._state.alert_config_path])
            else:
                os.startfile(self._state.alert_config_path)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Editor open failed", str(exc))

    # ----------------------------------------------------------------------
    # Stats footer
    # ----------------------------------------------------------------------

    def _refresh_stats(self) -> None:
        lines = []
        if self._state.baseline > 0:
            ts = self._state.baseline_ts
            ago = time.time() - ts
            when = (f"{int(ago/60)}m ago" if ago > 90
                    else f"{int(ago)}s ago")
            lines.append(
                f"Baseline:   {self._state.baseline:.4f}  ({when})"
            )
        else:
            lines.append("Baseline:   — (run Calibrate)")
        lines.append(f"Pkt rate:   {self._pkt_rate:5.1f} Hz")
        # Queue stats from the active notifier (if it's a QueuingNotifier).
        try:
            pending = getattr(self._notifier_active, "pending_count", None)
            dead = getattr(self._notifier_active, "dead_count", None)
            if pending is not None:
                lines.append(f"Queue:      {pending} pending, {dead} dead")
        except Exception:  # noqa: BLE001
            pass
        if self._last_alert_ts > 0:
            ago = time.time() - self._last_alert_ts
            when = (f"{int(ago/60)}m ago" if ago > 90
                    else f"{int(ago)}s ago")
            lines.append(f"Last alert: {when}")
        else:
            lines.append("Last alert: never")
        lines.append(f"Source:     {self._state.source}")
        if self._state.alert_config_path:
            lines.append(f"Config:     {self._state.alert_config_path}")
        self._stats.setText("\n".join(lines))

    def stop_workers(self, wait_ms: int = 3000) -> None:
        """Stop detect / calibration / recording and wait for their threads
        (they check the stop flag at least every ~1 s, even with no CSI
        arriving). A recording in progress is closed and kept."""
        self._closed = True
        for worker, thread in ((self._detect_worker, self._detect_thread),
                               (self._calib_worker, self._calib_thread),
                               (self._rec_worker, self._rec_thread)):
            try:
                if worker is not None:
                    worker.request_stop()
                if thread is not None:
                    # quit() here, not only via worker.finished → thread.quit:
                    # that connection is queued to this (GUI) thread, which
                    # is blocked in wait() — the thread would never stop.
                    thread.quit()
                    thread.wait(wait_ms)
            except RuntimeError:
                pass                      # already finished and deleted

    def closeEvent(self, ev) -> None:  # noqa: N802
        self.stop_workers()
        super().closeEvent(ev)


# --------------------------------------------------------------------------
# Entry point invoked by csidetector.cli.cmd_alert_gui
# --------------------------------------------------------------------------

def run_gui(source: Optional[str] = None,
            alert_config: Optional[str] = None,
            baseline: Optional[float] = None,
            enter_ratio: Optional[float] = None,
            exit_ratio: Optional[float] = None,
            location: Optional[str] = None) -> int:
    """Launch the alert-mode GUI. CLI args override the persisted state."""
    import sys
    from csidetector.provision.gui import _install_excepthook, _quit_on_signals
    app = QApplication(sys.argv)
    _install_excepthook()
    win = CSIDetectorWindow(alert_config_path=alert_config or "")
    # Ctrl+C / closing the terminal: stop the workers so a recording's file
    # is finished, not cut off.
    app.aboutToQuit.connect(win.stop_workers)
    _quit_on_signals(app)
    # CLI overrides land in the persisted state so they survive a relaunch.
    if source:
        win._state.source = source
        win._state.receiver_mac = ""      # an explicit source isn't looked up
    if baseline is not None and baseline > 0:
        win._state.baseline = baseline
        win._state.baseline_ts = time.time()
    if enter_ratio is not None:
        win._state.enter_ratio = enter_ratio
        win._sparkline.set_threshold(enter_ratio)
    if exit_ratio is not None:
        win._state.exit_ratio = exit_ratio
    if location is not None:
        win._state.location = location
    win._state.save()
    win._refresh_stats()
    win.show()
    return app.exec()
