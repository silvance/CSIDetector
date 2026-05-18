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
import subprocess
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import (
    QObject, QThread, Qt, QTimer, pyqtSignal, pyqtSlot,
)
from PyQt6.QtGui import (
    QColor, QFont, QPainter, QPaintEvent, QPen,
)
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from csidetector.core import collector as csi_collector
from csidetector.core import detector
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


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

@dataclass
class _PersistedState:
    source: str = "/dev/ttyACM0"
    baseline: float = 0.0
    baseline_ts: float = 0.0
    enter_ratio: float = 1.7
    exit_ratio: float = 1.3
    location: str = ""
    alert_config_path: str = ""

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
            self.error_occurred.emit(f"{type(exc).__name__}: {exc}")
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
        last_motion = False
        last_alert_ts = 0.0
        ts_history: list[float] = []   # recent push timestamps for pkt rate
        last_rate_emit = 0.0
        self.state_changed.emit("INIT")
        settle_until = time.time() + self._settle_s

        for sample in csi_collector.open_source(self._source):
            if self._stop.is_set():
                return
            now = time.time()
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
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, source: str, settle_s: float = 30.0,
                 record_s: float = 30.0, window: int = 50):
        super().__init__()
        self._source = source
        self._settle_s = settle_s
        self._record_s = record_s
        self._window = window
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            self._run_inner()
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()

    def _run_inner(self) -> None:
        src = csi_collector.open_source(self._source)
        rows: list[np.ndarray] = []
        idx: Optional[np.ndarray] = None
        start = time.time()
        settle_until = start + self._settle_s
        deadline = settle_until + self._record_s
        last_emit = 0.0

        for sample in src:
            if self._stop.is_set():
                return
            now = time.time()
            if now - last_emit > 0.2:
                if now < settle_until:
                    self.phase_changed.emit("LEAVE THE ROOM",
                                            max(0.0, settle_until - now))
                else:
                    self.phase_changed.emit("RECORDING",
                                            max(0.0, deadline - now))
                last_emit = now
            if now < settle_until:
                continue
            if idx is None:
                idx = np.flatnonzero(sample.amplitude > 0)
                if idx.size == 0:
                    continue
            rows.append(sample.amplitude)
            if now >= deadline:
                break

        if not rows or idx is None:
            raise RuntimeError("no CSI samples received during calibration")
        amps = np.stack(rows)[:, idx]
        baseline = detector.compute_baseline(amps, self._window)
        self.done.emit(float(baseline), amps.shape[0])


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
        self._ratio_label.setFont(QFont("monospace"))
        self._ratio_bar = QProgressBar()
        self._ratio_bar.setRange(0, 300)   # 0.00× to 3.00×
        self._ratio_bar.setTextVisible(False)
        self._sparkline = _Sparkline()
        self._sparkline.set_threshold(self._state.enter_ratio)

        self._stats = QLabel("")
        self._stats.setFont(QFont("monospace"))
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

        # --- layout ------------------------------------------------------
        central = QWidget()
        v = QVBoxLayout(central)
        v.addWidget(self._badge)
        v.addWidget(self._ratio_label)
        v.addWidget(self._ratio_bar)
        v.addWidget(self._sparkline, stretch=1)
        v.addWidget(self._stats)
        h = QHBoxLayout()
        for b in (self._calibrate_btn, self._arm_btn, self._disarm_btn,
                  self._test_btn, self._edit_btn):
            h.addWidget(b)
        v.addLayout(h)
        self.setCentralWidget(central)

        # --- worker state ------------------------------------------------
        self._detect_thread: Optional[QThread] = None
        self._detect_worker: Optional[_DetectWorker] = None
        self._calib_thread: Optional[QThread] = None
        self._calib_worker: Optional[_CalibrateWorker] = None
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
        if self._detect_thread is not None or self._calib_thread is not None:
            return
        self._badge.set_state("CALIB", "starting…")
        self._calibrate_btn.setEnabled(False)
        self._arm_btn.setEnabled(False)

        thread = QThread(self)
        worker = _CalibrateWorker(source=self._state.source)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.phase_changed.connect(self._on_calib_phase)
        worker.done.connect(self._on_calib_done)
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
        self._calibrate_btn.setEnabled(True)
        self._arm_btn.setEnabled(True)
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
        if self._detect_thread is not None:
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
            source=self._state.source,
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
        self._disarm_btn.setEnabled(True)
        self._badge.set_state("INIT")

    def _on_disarm(self) -> None:
        if self._detect_worker is None:
            return
        self._badge.set_state("DISARMED", "stopping…")
        self._detect_worker.request_stop()
        # iter_serial has up to ~1s of readline blocking, so disarm
        # takes effect within that window. UI re-enables in
        # _on_detect_finished.

    def _on_ratio(self, score: float, ratio: float) -> None:
        self._ratio_label.setText(
            f"Ratio: {ratio:5.2f}×  (threshold {self._state.enter_ratio:.2f}×)  "
            f"σ={score:.3f}")
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
        self._disarm_btn.setEnabled(False)
        self._badge.set_state("DISARMED")
        self._pkt_rate = 0.0
        self._refresh_stats()

    # ----------------------------------------------------------------------
    # Test alert + edit config
    # ----------------------------------------------------------------------

    def _on_test(self) -> None:
        if not self._state.alert_config_path:
            QMessageBox.warning(
                self, "No config",
                "Test alert needs an alert.toml — set --alert-config on "
                "the command line or 'Edit config' to point at one.")
            return
        try:
            cfg = load_config(self._state.alert_config_path)
            notifier = build_notifier(cfg)
        except SystemExit as exc:
            QMessageBox.warning(self, "Config error", str(exc))
            return
        ev = Event.now(kind="TEST", message="CSIDetector GUI test alert")
        try:
            notifier.send(ev)
            QMessageBox.information(
                self, "Test alert sent",
                "Queued for delivery. If your Telegram doesn't buzz "
                "within ~10s, check ~/.csidetector/alert-queue.db for "
                "a row with last_error set.")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Test alert failed", str(exc))
        finally:
            try:
                notifier.close()
            except Exception:  # noqa: BLE001
                pass

    def _on_edit(self) -> None:
        if not self._state.alert_config_path:
            QMessageBox.information(
                self, "No config path",
                "Pass --alert-config alert.toml to the GUI to use this "
                "feature.")
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

    def closeEvent(self, ev) -> None:  # noqa: N802
        if self._detect_worker is not None:
            self._detect_worker.request_stop()
            if self._detect_thread is not None:
                self._detect_thread.wait(2000)
        if self._calib_worker is not None:
            self._calib_worker.request_stop()
            if self._calib_thread is not None:
                self._calib_thread.wait(2000)
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
    app = QApplication(sys.argv)
    win = CSIDetectorWindow(alert_config_path=alert_config or "")
    # CLI overrides land in the persisted state so they survive a relaunch.
    if source:
        win._state.source = source
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
