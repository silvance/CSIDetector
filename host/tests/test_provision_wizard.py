"""Setup wizard flows under offscreen Qt.

The real Flasher runs end to end except for the process launch: its
QProcess start is stubbed and completion is driven with esptool-style
output, so registry recording, page wiring, and file writing are all
exercised for real.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

import pytest

pytest.importorskip("PyQt6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QWizard  # noqa: E402

import csidetector.modes.alert.gui as alert_gui  # noqa: E402
from csidetector.modes.alert.config import load_config  # noqa: E402
from csidetector.provision import devices as dv  # noqa: E402
from csidetector.provision import firmware as fw  # noqa: E402
from csidetector.provision import wizard as wz  # noqa: E402

MAC_A = "aa:aa:aa:aa:aa:a0"
MAC_B = "aa:aa:aa:aa:aa:b0"
TOKEN = "123456789:" + "A" * 35
EXAMPLE_LINKS = Path(__file__).resolve().parents[1] / "links.example.json"


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def env(qt_app, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(alert_gui, "_STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(alert_gui, "_STATE_PATH", tmp_path / "state" / "state.json")
    boards = [
        dv.AttachedBoard("/dev/ttyACM0", 0x303A, 0x1001, MAC_A.upper(), "", mac=MAC_A,
                         chip="ESP32-S3"),
        dv.AttachedBoard("/dev/ttyACM1", 0x303A, 0x1001, MAC_B.upper(), "", mac=MAC_B,
                         chip="ESP32-S3"),
    ]
    export = tmp_path / "export.sh"
    export.write_text("")
    host_dir = tmp_path / "host"
    host_dir.mkdir()
    hotspots = [fw.HotspotInfo("Hotspot", "wlp2s0u2", "CSIDetector", 6, "10.42.1.1")]
    reg = dv.Registry(tmp_path / "devices.json")
    wiz = wz.SetupWizard(reg, list_boards=lambda: list(boards),
                         detect_hotspots=lambda: hotspots,
                         idf_export=str(export), host_dir=host_dir)
    started = []
    monkeypatch.setattr(wiz.flasher._runner, "start",
                        lambda argv, environ: started.append(argv))
    # Never block a test on a modal dialog.
    monkeypatch.setattr(wiz, "confirm", lambda *a: True)
    shown = []
    monkeypatch.setattr(wz.QMessageBox, "warning", lambda *a, **k: shown.append(a))
    monkeypatch.setattr(wz.QMessageBox, "information", lambda *a, **k: shown.append(a))
    wiz.show()
    ns = type("Env", (), {})()
    ns.wiz, ns.reg, ns.boards, ns.host_dir, ns.started, ns.shown = (
        wiz, reg, boards, host_dir, started, shown)
    yield ns
    wiz.close()
    wiz.deleteLater()


def _flash(env, port, mac, label=""):
    """Click Flash on the current flash page and complete it."""
    page = env.wiz.currentPage()
    page.refresh_boards()
    page.board_combo.setCurrentIndex(page.board_combo.findData(port))
    page.label_edit.setText(label)
    page.flash_btn.click()
    job = env.wiz.flasher._pending["job"]
    env.wiz.flasher._on_finished(0, f"Chip is ESP32-S3 (QFN56)\nMAC: {mac}\n")
    return job


def test_alert_flow_end_to_end(env):
    wiz = env.wiz
    assert wiz.currentId() == wz.PAGE_MODE
    wiz.next()
    assert wiz.currentId() == wz.PAGE_ENV and wiz.currentPage().isComplete()
    wiz.next()

    assert wiz.currentId() == wz.PAGE_TX
    assert not wiz.currentPage().isComplete()          # nothing flashed yet
    tx_job = _flash(env, "/dev/ttyACM0", MAC_A, "Desk TX")
    assert tx_job.role == dv.ROLE_TX and tx_job.config.channel == 11
    assert env.reg.get(MAC_A).label == "Desk TX"
    assert env.reg.get(MAC_A).role == dv.ROLE_TX
    assert wiz.currentPage().isComplete()
    wiz.next()

    assert wiz.currentId() == wz.PAGE_RX
    assert "Desk TX" in wiz.currentPage().instructions.text()
    rx_job = _flash(env, "/dev/ttyACM1", MAC_B, "South RX")
    assert rx_job.config.tx_macs == [MAC_A]
    assert rx_job.config.uart_only
    wiz.next()

    assert wiz.currentId() == wz.PAGE_TELEGRAM
    tg = wiz.currentPage()
    assert not tg.isComplete()
    tg.token.setText(TOKEN)
    tg.chat.setText("8722080863")
    assert tg.location.text() == "South RX"             # defaulted from the RX
    assert tg.isComplete()
    wiz.next()

    assert wiz.currentId() == wz.PAGE_FINISH
    assert "/dev/ttyACM1" in wiz.currentPage().summary.text()
    wiz.button(QWizard.WizardButton.FinishButton).click()

    assert wiz.result() == QWizard.DialogCode.Accepted
    toml = env.host_dir / "alert.toml"
    assert stat.S_IMODE(os.stat(toml).st_mode) == 0o600
    cfg = load_config(str(toml))
    assert cfg["notifier"]["bot_token"] == TOKEN
    assert cfg["alert"]["location_label"] == "South RX"
    state = alert_gui._PersistedState.load()
    assert state.source == "/dev/ttyACM1"
    assert state.alert_config_path == str(toml)
    assert state.baseline == 0.0                          # forces recalibration
    assert wiz.open_monitor and wiz.alert_config_path == str(toml)


def test_localize_flow_merges_links(env):
    shutil.copy(EXAMPLE_LINKS, env.host_dir / "links.json")
    wiz = env.wiz
    wiz.page(wz.PAGE_MODE).localize.setChecked(True)
    wiz.next()
    envp = wiz.currentPage()
    assert envp.ssid.text() == "CSIDetector"            # auto-detected
    assert envp.host_ip.text() == "10.42.1.1"
    assert envp.channel.value() == 6
    envp.password.setText("hunter22")
    wiz.next()

    _flash(env, "/dev/ttyACM0", MAC_A, "Desk TX")
    wiz.next()
    rx_job = _flash(env, "/dev/ttyACM1", MAC_B, "Hall RX")
    assert rx_job.config.ssid == "CSIDetector"
    assert rx_job.config.host_ip == "10.42.1.1"
    assert rx_job.config.channel == 6
    assert rx_job.config.tx_macs == [MAC_A]
    # The password reaches the 0600 overlay, never the registry.
    assert "hunter22" not in json.dumps(env.reg.get(MAC_B).config)
    wiz.next()

    assert wiz.currentId() == wz.PAGE_LINKS
    wiz.next()
    wiz.button(QWizard.WizardButton.FinishButton).click()
    assert wiz.result() == QWizard.DialogCode.Accepted
    data = json.loads((env.host_dir / "links.json").read_text())
    assert data["room"] == json.loads(EXAMPLE_LINKS.read_text())["room"]
    assert {"mac": MAC_A, "label": "Desk TX", "placed": False}.items() <= data["txs"][-1].items()
    assert data["rxs"][-1]["label"] == "Hall RX"
    assert not wiz.open_monitor


def test_tx_page_still_works_after_going_back(env):
    """Regression guard: a page revisited via Back must still receive
    flash results (signals are connected once, not on page entry)."""
    wiz = env.wiz
    wiz.next()
    wiz.next()
    assert wiz.currentId() == wz.PAGE_TX
    wiz.back()
    wiz.next()
    assert wiz.currentId() == wz.PAGE_TX
    _flash(env, "/dev/ttyACM0", MAC_A, "Desk TX")
    assert MAC_A in wiz.currentPage().checked_macs()
    assert wiz.currentPage().isComplete()


def test_alert_mode_needs_exactly_one_attached_receiver(env):
    env.reg.upsert("cc:cc:cc:cc:cc:01", role=dv.ROLE_RX, label="Old RX 1")
    env.reg.upsert("cc:cc:cc:cc:cc:02", role=dv.ROLE_RX, label="Old RX 2")
    wiz = env.wiz
    wiz.next()
    wiz.next()
    _flash(env, "/dev/ttyACM0", MAC_A)
    wiz.next()
    page = wiz.currentPage()
    for i in range(page.ready.count()):
        page.ready.item(i).setCheckState(wz.Qt.CheckState.Checked)
    assert not page.isComplete()
    assert "exactly one" in page.ready_note.text()
    # One pre-registered RX that isn't plugged in can't be the alert receiver.
    for i in range(page.ready.count()):
        page.ready.item(i).setCheckState(
            wz.Qt.CheckState.Checked if i == 0 else wz.Qt.CheckState.Unchecked)
    assert not page.isComplete()
    assert "isn't plugged in" in page.ready_note.text()


def test_flashing_a_new_alert_receiver_replaces_the_tick(env):
    env.reg.upsert("cc:cc:cc:cc:cc:01", role=dv.ROLE_RX, label="Old RX")
    wiz = env.wiz
    wiz.next()
    wiz.next()
    _flash(env, "/dev/ttyACM0", MAC_A)
    wiz.next()
    page = wiz.currentPage()
    page.ready.item(0).setCheckState(wz.Qt.CheckState.Checked)
    _flash(env, "/dev/ttyACM1", MAC_B)
    assert page.checked_macs() == [MAC_B]
    assert page.isComplete()


def test_rx_page_defaults_away_from_the_transmitter(env, monkeypatch):
    wiz = env.wiz
    wiz.next()
    wiz.next()
    _flash(env, "/dev/ttyACM0", MAC_A)
    wiz.next()
    page = wiz.currentPage()
    assert page.board_combo.currentData() == "/dev/ttyACM1"   # not the TX
    asked = []
    monkeypatch.setattr(wiz, "confirm", lambda title, text: asked.append(text) or False)
    page.board_combo.setCurrentIndex(page.board_combo.findData("/dev/ttyACM0"))
    page.flash_btn.click()
    assert asked and "transmitter you chose" in asked[0]
    assert wiz.flasher._pending is None                       # declined → nothing ran


def test_telegram_skip_writes_no_credentials(env):
    wiz = env.wiz
    wiz.next()
    wiz.next()
    _flash(env, "/dev/ttyACM0", MAC_A)
    wiz.next()
    _flash(env, "/dev/ttyACM1", MAC_B)
    wiz.next()
    wiz.currentPage().skip.setChecked(True)
    assert wiz.currentPage().isComplete()
    wiz.next()
    wiz.button(QWizard.WizardButton.FinishButton).click()
    assert not (env.host_dir / "alert.toml").exists()
    assert alert_gui._PersistedState.load().alert_config_path == ""


def test_environment_page_requires_export_and_localize_fields(env, tmp_path):
    wiz = env.wiz
    wiz.page(wz.PAGE_MODE).localize.setChecked(True)
    wiz.next()
    page = wiz.currentPage()
    assert page.isComplete()
    page.host_ip.setText("10.42.1")
    assert not page.isComplete()
    page.host_ip.setText("10.42.1.1")
    page.export_edit.setText(str(tmp_path / "missing.sh"))
    assert not page.isComplete()


def test_failed_flash_is_not_ticked(env):
    wiz = env.wiz
    wiz.next()
    wiz.next()
    page = wiz.currentPage()
    page.refresh_boards()
    page.flash_btn.click()
    wiz.flasher._on_finished(2, "A fatal error occurred: Failed to connect to ESP32-S3")
    assert page.checked_macs() == []
    assert not page.isComplete()
    assert "[hint]" in page.log.toPlainText()
    assert env.shown                                   # failure dialog raised


# --------------------------------------------------------------------------
# Regression tests for the phase B review
# --------------------------------------------------------------------------

MAC_C = "aa:aa:aa:aa:aa:c0"
BACK = QWizard.WizardButton.BackButton


def _to_page(wiz, page_id):
    while wiz.currentId() != page_id:
        wiz.next()


def test_unknown_chip_is_identified_before_building(env):
    """A bridge board's chip isn't in its USB descriptor: the wizard must
    identify it and build for that chip, not silently for an S3."""
    env.boards.append(dv.AttachedBoard("/dev/ttyUSB0", 0x10C4, 0xEA60, "0001", "CP2102"))
    wiz = env.wiz
    _to_page(wiz, wz.PAGE_TX)
    page = wiz.currentPage()
    page.refresh_boards()
    page.board_combo.setCurrentIndex(page.board_combo.findData("/dev/ttyUSB0"))
    page.flash_btn.click()
    assert wiz.flasher._pending["kind"] == "identify"
    wiz.flasher._on_finished(0, "Chip is ESP32-C3 (QFN32)\nMAC: aa:aa:aa:aa:aa:c0\n")
    job = wiz.flasher._pending["job"]
    assert job.target == "esp32c3"
    assert env.reg.get(MAC_C).chip == "ESP32-C3"


def test_unsupported_chip_is_refused(env):
    env.boards.append(dv.AttachedBoard("/dev/ttyACM2", 0x303A, 0x1001, MAC_C.upper(), "",
                                       mac=MAC_C, chip="ESP32-H2"))
    wiz = env.wiz
    _to_page(wiz, wz.PAGE_TX)
    page = wiz.currentPage()
    page.refresh_boards()
    page.board_combo.setCurrentIndex(page.board_combo.findData("/dev/ttyACM2"))
    page.flash_btn.click()
    assert wiz.flasher._pending is None
    assert any("isn't supported" in a[2] for a in env.shown)


def test_alert_receiver_flashed_through_a_bridge_needs_its_native_port(env):
    wiz = env.wiz
    _to_page(wiz, wz.PAGE_TX)
    _flash(env, "/dev/ttyACM0", MAC_A)
    wiz.next()
    env.boards[:] = [b for b in env.boards if b.mac != MAC_B]
    env.boards.append(dv.AttachedBoard("/dev/ttyUSB0", 0x10C4, 0xEA60, "0001", "CP2102",
                                       chip="ESP32-S3"))
    _flash(env, "/dev/ttyUSB0", MAC_C)
    page = wiz.currentPage()
    assert page.checked_macs() == [MAC_C]
    assert not page.isComplete()
    assert "native USB port" in page.ready_note.text()
    # Plugging the native port in (MAC in its USB serial) completes the page,
    # and that's the port the monitor will read.
    env.boards.append(dv.AttachedBoard("/dev/ttyACM5", 0x303A, 0x1001, MAC_C.upper(), "",
                                       mac=MAC_C))
    page.refresh_boards()
    assert page.isComplete()
    wiz.next()
    assert wiz.state.rxs[MAC_C][1] == "/dev/ttyACM5"


def test_reused_boards_must_match_this_setup(env):
    env.reg.upsert(MAC_C, role=dv.ROLE_TX, label="Old TX", config={"channel": 6})
    wiz = env.wiz
    _to_page(wiz, wz.PAGE_TX)
    page = wiz.currentPage()
    page.ready.item(0).setCheckState(wz.Qt.CheckState.Checked)
    assert not page.isComplete()
    assert "channel 6" in page.ready_note.text()

    page.ready.item(0).setCheckState(wz.Qt.CheckState.Unchecked)
    _flash(env, "/dev/ttyACM0", MAC_A)
    wiz.next()
    # MAC_B was last flashed as a localize receiver: alert mode can't use it.
    env.reg.upsert(MAC_B, role=dv.ROLE_RX, label="Hall RX",
                   config={"channel": 11, "tx_macs": [MAC_A], "mode": "hotspot"})
    rx = wiz.currentPage()
    rx.initializePage()
    rx.ready.item(0).setCheckState(wz.Qt.CheckState.Checked)
    assert not rx.isComplete()
    assert "hotspot (localize) mode" in rx.ready_note.text()
    env.reg.upsert(MAC_B, config={"channel": 11, "tx_macs": [MAC_C], "mode": "uart"})
    rx.completeChanged.emit()
    assert not rx.isComplete()
    assert "different transmitters" in rx.ready_note.text()
    env.reg.upsert(MAC_B, config={"channel": 11, "tx_macs": [MAC_A], "mode": "uart"})
    assert rx.isComplete()


def test_ticks_survive_back_and_next(env):
    env.reg.upsert(MAC_C, role=dv.ROLE_TX, label="Old TX")
    wiz = env.wiz
    _to_page(wiz, wz.PAGE_TX)
    wiz.currentPage().ready.item(0).setCheckState(wz.Qt.CheckState.Checked)
    wiz.back()
    wiz.next()
    assert wiz.currentPage().checked_macs() == [MAC_C]


def test_back_is_hidden_while_flashing(env, monkeypatch):
    """Back runs cleanupPage, which kills the flash without asking."""
    wiz = env.wiz
    runner = wiz.flasher._runner
    monkeypatch.setattr(runner, "start",
                        lambda argv, environ: setattr(runner, "_proc", "running"))
    _to_page(wiz, wz.PAGE_TX)
    page = wiz.currentPage()
    assert wiz.button(BACK).isVisible()
    page.refresh_boards()
    page.flash_btn.click()
    assert wiz.flasher.busy
    assert not wiz.button(BACK).isVisible()
    page.completeChanged.emit()               # QWizard re-syncs buttons here
    assert not wiz.button(BACK).isVisible()
    runner._proc = None
    wiz.flasher._on_finished(0, f"Chip is ESP32-S3 (QFN56)\nMAC: {MAC_A}\n")
    assert wiz.button(BACK).isVisible()


def test_alert_mode_uses_exactly_one_transmitter(env):
    wiz = env.wiz
    _to_page(wiz, wz.PAGE_TX)
    page = wiz.currentPage()
    _flash(env, "/dev/ttyACM0", MAC_A)
    _flash(env, "/dev/ttyACM1", MAC_B)
    assert page.checked_macs() == [MAC_B]     # the new one replaces the tick
    for i in range(page.ready.count()):
        page.ready.item(i).setCheckState(wz.Qt.CheckState.Checked)
    assert not page.isComplete()
    assert "exactly one transmitter" in page.ready_note.text()


def test_open_monitor_is_retired_before_state_is_rewritten(env):
    seen = []
    env.wiz.before_alert_apply = lambda: seen.append(alert_gui._STATE_PATH.exists())
    wiz = env.wiz
    _to_page(wiz, wz.PAGE_TX)
    _flash(env, "/dev/ttyACM0", MAC_A)
    wiz.next()
    _flash(env, "/dev/ttyACM1", MAC_B)
    wiz.next()
    wiz.currentPage().skip.setChecked(True)
    wiz.next()
    wiz.button(QWizard.WizardButton.FinishButton).click()
    assert seen == [False]                    # called once, before the write
    assert alert_gui._PersistedState.load().source == "/dev/ttyACM1"


def test_unexpected_apply_error_is_reported_not_fatal(env, monkeypatch):
    wiz = env.wiz
    _to_page(wiz, wz.PAGE_TX)
    _flash(env, "/dev/ttyACM0", MAC_A)
    wiz.next()
    _flash(env, "/dev/ttyACM1", MAC_B)
    wiz.next()
    wiz.currentPage().skip.setChecked(True)
    wiz.next()

    def boom():
        raise KeyError("rxs")
    monkeypatch.setattr(wiz, "apply", boom)
    wiz.button(QWizard.WizardButton.FinishButton).click()
    assert wiz.isVisible() and wiz.currentId() == wz.PAGE_FINISH
    assert any("KeyError" in a[2] for a in env.shown)


@pytest.mark.parametrize("port, expect", [(5566, "  3. ./start.sh"),
                                          (5577, "  3. PORT=5577 ./start.sh")])
def test_localize_finish_steps_use_the_chosen_port(env, port, expect):
    wiz = env.wiz
    wiz.page(wz.PAGE_MODE).localize.setChecked(True)
    wiz.next()
    wiz.currentPage().port.setValue(port)
    wiz.next()
    _flash(env, "/dev/ttyACM0", MAC_A)
    wiz.next()
    _flash(env, "/dev/ttyACM1", MAC_B)
    wiz.next()
    wiz.next()
    text = wiz.currentPage().summary.text()
    assert f"calibrate-links udp:{port} " in text
    assert expect in text
