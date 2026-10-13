"""Smoke tests for the alert-mode GUI.

PyQt6 is the heaviest dep in the desktop install and we don't ship it
in the slim Docker test image; tests gracefully skip the whole file
when it isn't importable. On the developer's desktop (where PyQt6
came in via matplotlib), the Qt-dependent tests run under the
offscreen platform plugin so no display is required.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest


# Module-level gate: GUI module top-level imports PyQt6, so any test
# touching it (even the persistence dataclass) needs Qt installed.
pytest.importorskip("PyQt6.QtWidgets")


# --------------------------------------------------------------------------
# Persistence — no Qt needed.
# --------------------------------------------------------------------------

def test_persisted_state_round_trip(home):
    """State written to disk reads back identically. Defaults fill in
    when a field is missing."""
    # Point _STATE_DIR/_STATE_PATH at the home dir BEFORE importing the
    # module so the dataclass picks up the override.
    import importlib
    gui = importlib.import_module("csidetector.modes.alert.gui")
    importlib.reload(gui)   # re-evaluate _STATE_DIR with the new HOME

    s = gui._PersistedState(
        source="/dev/ttyACM1",
        baseline=2.5,
        baseline_ts=1234567890.0,
        enter_ratio=1.7,
        exit_ratio=1.3,
        location="Lab",
        alert_config_path="/etc/foo.toml",
    )
    s.save()
    assert (home / ".csidetector" / "state.json").exists()

    loaded = gui._PersistedState.load()
    assert loaded.source == "/dev/ttyACM1"
    assert loaded.baseline == pytest.approx(2.5)
    assert loaded.location == "Lab"


def test_persisted_state_missing_file_yields_defaults(home):
    import importlib
    gui = importlib.import_module("csidetector.modes.alert.gui")
    importlib.reload(gui)

    s = gui._PersistedState.load()
    assert s.source == ("COM3" if os.name == "nt" else "/dev/ttyACM0")
    assert s.baseline == 0.0
    assert s.alert_config_path == ""


def test_persisted_state_corrupt_file_yields_defaults(home):
    """A truncated state.json (interrupted write) must not crash the GUI;
    falling back to defaults is the right behavior."""
    sdir = home / ".csidetector"
    sdir.mkdir()
    (sdir / "state.json").write_text("{not valid json")

    import importlib
    gui = importlib.import_module("csidetector.modes.alert.gui")
    importlib.reload(gui)

    s = gui._PersistedState.load()
    assert s.baseline == 0.0


# --------------------------------------------------------------------------
# Qt-dependent — gated on PyQt6 + offscreen platform.
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qt_app():
    """Create a single QApplication for the test session under the
    offscreen platform. Skips the entire fixture (and tests using it)
    when PyQt6 isn't importable."""
    pytest.importorskip("PyQt6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_main_window_constructs(qt_app):
    """The window builds without raising. Most regressions in this file
    would show up here (broken signal connections, missing widgets,
    bad parent ordering)."""
    from csidetector.modes.alert.gui import CSIDetectorWindow
    win = CSIDetectorWindow(alert_config_path="")
    try:
        assert win.windowTitle()
        # Disarm is disabled until armed.
        assert not win._disarm_btn.isEnabled()
        assert win._arm_btn.isEnabled()
    finally:
        win.close()


def test_closed_window_ignores_late_calibration_result(qt_app, tmp_path, monkeypatch):
    """A calibration result still queued when the monitor closes must not
    overwrite state saved after the close (the setup wizard rewrites it)."""
    import csidetector.modes.alert.gui as gui
    monkeypatch.setattr(gui, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(gui, "_STATE_PATH", tmp_path / "state.json")
    win = gui.CSIDetectorWindow(alert_config_path="")
    win.close()
    gui._PersistedState(source="/dev/ttyACM9").save()
    win._on_calib_done(3.0, 100)
    assert gui._PersistedState.load().source == "/dev/ttyACM9"
    assert gui._PersistedState.load().baseline == 0.0


def test_explicit_sources_are_not_replaced_by_the_receiver_lookup(qt_app, tmp_path,
                                                                    monkeypatch):
    """Only serial ports are looked up by the saved receiver MAC; a replay
    file or udp: source stays as given."""
    import csidetector.modes.alert.gui as gui
    from csidetector.provision import devices as dv
    monkeypatch.setattr(gui, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(gui, "_STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(dv, "list_boards", lambda comports=None: [
        dv.AttachedBoard("/dev/ttyACM0", 0x303A, 0x1001, "", "", mac="aa:bb:cc:dd:ee:01")])
    win = gui.CSIDetectorWindow(alert_config_path="")
    try:
        win._state.receiver_mac = "aa:bb:cc:dd:ee:01"
        for source in ("capture.log", "udp:5566"):
            win._state.source = source
            assert win._resolve_source() == source
        win._state.source = "/dev/ttyACM3"
        assert win._resolve_source() == "/dev/ttyACM0"     # moved: found by MAC
    finally:
        win.close()


def _fake_serial_lines(lines_per_s: float, line: str = ""):
    """Stands in for collector.iter_serial_lines: ``line`` at a rate (or
    nothing, with on_idle ticks) until stopped."""
    import time

    def fake(port, baud=921600, stop=None, on_idle=None):
        while not stop.is_set():
            if line:
                time.sleep(1.0 / lines_per_s)
                yield line
            else:
                time.sleep(0.05)
                on_idle()
    return fake


def _record(gui, tmp_path, delay_s, length_s, stop_after_s=None):
    worker = gui._RecordWorker("/dev/ttyACM1", tmp_path / "recs",
                               {"condition": "empty", "note": ""},
                               delay_s=delay_s, length_s=length_s)
    out = {"saved": [], "errors": [], "phases": set()}
    worker.saved.connect(lambda path, summary: out["saved"].append((path, summary)))
    worker.error_occurred.connect(out["errors"].append)
    worker.progress.connect(lambda phase, left, n: out["phases"].add(phase))
    if stop_after_s is not None:
        threading.Timer(stop_after_s, worker.request_stop, ("stop button",)).start()
    worker.run()                 # in this thread: the signals are delivered directly
    return out


def test_record_worker_waits_then_records_for_its_length(qt_app, tmp_path, monkeypatch,
                                                         csi_data_line):
    import csidetector.modes.alert.gui as gui
    from csidetector.core import recording
    monkeypatch.setattr(gui.csi_collector, "iter_serial_lines",
                        _fake_serial_lines(200, csi_data_line() + "\n"))
    out = _record(gui, tmp_path, delay_s=0.2, length_s=0.4)
    assert out["errors"] == [] and out["phases"] == {"WAIT", "REC"}
    (path, summary), = out["saved"]
    assert summary["stopped_by"] == "timer" and 0.35 <= summary["duration_s"] <= 1.0
    rec = recording.read_recording(path)
    assert rec.meta["condition"] == "empty" and "started" in rec.meta
    assert rec.end["samples"] == summary["samples"] > 20


def test_record_worker_ignores_wall_clock_jumps(qt_app, tmp_path, monkeypatch,
                                               csi_data_line):
    """The PC's clock syncing mid-recording (seen on a fresh CI machine)
    mustn't cut the recording short: lengths run on the monotonic clock."""
    import time as real
    import types
    import csidetector.modes.alert.gui as gui
    jumps = iter(range(10**6))
    monkeypatch.setattr(gui, "time", types.SimpleNamespace(
        time=lambda: real.time() + 3600 * next(jumps),     # an hour per read
        monotonic=real.monotonic, strftime=real.strftime, localtime=real.localtime))
    monkeypatch.setattr(gui.csi_collector, "iter_serial_lines",
                        _fake_serial_lines(200, csi_data_line() + "\n"))
    out = _record(gui, tmp_path, delay_s=0.2, length_s=0.4)
    (path, summary), = out["saved"]
    assert summary["stopped_by"] == "timer" and 0.35 <= summary["duration_s"] <= 1.0


def test_record_worker_stopped_during_the_countdown_saves_nothing_quietly(
        qt_app, tmp_path, monkeypatch, csi_data_line):
    import csidetector.modes.alert.gui as gui
    monkeypatch.setattr(gui.csi_collector, "iter_serial_lines",
                        _fake_serial_lines(200, csi_data_line() + "\n"))
    out = _record(gui, tmp_path, delay_s=0.5, length_s=5, stop_after_s=0.2)
    assert out["saved"] == [] and out["errors"] == []
    assert not (tmp_path / "recs").exists() or not list((tmp_path / "recs").glob("*"))


def test_recordings_fall_back_to_home_and_open_folder_follows(qt_app, tmp_path, home,
                                                             monkeypatch):
    """Documents not writable (e.g. Windows' Controlled folder access):
    recordings go to ~/CSIDetector recordings, and Open folder shows that."""
    import csidetector.modes.alert.gui as gui
    blocked = tmp_path / "Documents"
    blocked.write_text("")                       # a file: nothing can go under it
    monkeypatch.setattr(gui, "recordings_dir", lambda: blocked / "CSIDetector recordings")
    monkeypatch.setattr(gui, "_STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(gui, "_STATE_PATH", tmp_path / "state" / "state.json")
    folder, note = gui._writable_recordings_dir()
    assert folder == home / "CSIDetector recordings" and "can't be written" in note
    opened = []
    monkeypatch.setattr(gui.QDesktopServices, "openUrl",
                        lambda url: opened.append(url.toLocalFile()) or True)
    win = gui.CSIDetectorWindow(alert_config_path="")
    try:
        win._on_open_recordings()
        assert [Path(p) for p in opened] == [folder]
    finally:
        win.close()


def test_record_worker_saves_nothing_without_csi(qt_app, tmp_path, monkeypatch):
    import csidetector.modes.alert.gui as gui
    monkeypatch.setattr(gui, "NO_DATA_WARN_S", 0.1)
    monkeypatch.setattr(gui.csi_collector, "iter_serial_lines", _fake_serial_lines(0))
    out = _record(gui, tmp_path, delay_s=0.0, length_s=0.3)
    assert out["saved"] == [] and any("nothing was saved" in e for e in out["errors"])
    assert not list((tmp_path / "recs").glob("*"))


def test_status_badge_state_transitions(qt_app):
    from csidetector.modes.alert.gui import _StatusBadge
    b = _StatusBadge()
    for state in ("DISARMED", "INIT", "STILL", "MOTION", "ERROR", "CALIB", "REC"):
        b.set_state(state)
        assert b.text(), f"empty text for state {state}"


def test_sparkline_push_and_threshold(qt_app):
    from csidetector.modes.alert.gui import _Sparkline, _SPARKLINE_LEN
    s = _Sparkline()
    for v in range(_SPARKLINE_LEN + 50):
        s.push(float(v) / 100.0)
    # Buffer caps at _SPARKLINE_LEN, dropping oldest.
    assert len(s._values) == _SPARKLINE_LEN
    s.set_threshold(2.0)
    assert s._threshold == 2.0
