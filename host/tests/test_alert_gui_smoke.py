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
from pathlib import Path

import pytest


# Module-level gate: GUI module top-level imports PyQt6, so any test
# touching it (even the persistence dataclass) needs Qt installed.
pytest.importorskip("PyQt6.QtWidgets")


# --------------------------------------------------------------------------
# Persistence — no Qt needed.
# --------------------------------------------------------------------------

def test_persisted_state_round_trip(tmp_path, monkeypatch):
    """State written to disk reads back identically. Defaults fill in
    when a field is missing."""
    # Point _STATE_DIR/_STATE_PATH at the tmp dir BEFORE importing the
    # module so the dataclass picks up the override.
    monkeypatch.setenv("HOME", str(tmp_path))
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
    assert (tmp_path / ".csidetector" / "state.json").exists()

    loaded = gui._PersistedState.load()
    assert loaded.source == "/dev/ttyACM1"
    assert loaded.baseline == pytest.approx(2.5)
    assert loaded.location == "Lab"


def test_persisted_state_missing_file_yields_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    import importlib
    gui = importlib.import_module("csidetector.modes.alert.gui")
    importlib.reload(gui)

    s = gui._PersistedState.load()
    assert s.source == "/dev/ttyACM0"
    assert s.baseline == 0.0
    assert s.alert_config_path == ""


def test_persisted_state_corrupt_file_yields_defaults(tmp_path, monkeypatch):
    """A truncated state.json (interrupted write) must not crash the GUI;
    falling back to defaults is the right behavior."""
    monkeypatch.setenv("HOME", str(tmp_path))
    sdir = tmp_path / ".csidetector"
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


def test_status_badge_state_transitions(qt_app):
    from csidetector.modes.alert.gui import _StatusBadge
    b = _StatusBadge()
    for state in ("DISARMED", "INIT", "STILL", "MOTION", "ERROR", "CALIB"):
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
