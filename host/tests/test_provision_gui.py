"""Setup GUI (Devices & Flashing) under offscreen Qt.

Hermetic: fake board enumeration, fake hotspot detection, a temp device
registry, and a temp settings file. Real QProcesses are used only for the
output-streaming and cancel tests (plain bash, no ESP-IDF).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PyQt6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEventLoop, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from csidetector.provision import devices as dv  # noqa: E402
from csidetector.provision import firmware as fw  # noqa: E402
from csidetector.provision import gui  # noqa: E402

TX_MAC = "a0:f2:62:f0:39:d8"
RX_MAC = "a0:f2:62:f0:39:28"
FAR_TX = "ac:a7:04:2c:24:cc"


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _boards():
    return [
        dv.AttachedBoard(port="/dev/ttyACM0", vid=0x303A, pid=0x1001,
                         usb_serial=RX_MAC.upper(), description="USB JTAG",
                         mac=RX_MAC),
        dv.AttachedBoard(port="/dev/ttyUSB0", vid=0x10C4, pid=0xEA60,
                         usb_serial="0001", description="CP2102"),
    ]


@pytest.fixture
def tab(qt_app, tmp_path):
    reg = dv.Registry(tmp_path / "devices.json")
    reg.upsert(FAR_TX, label="Corner TX", role=dv.ROLE_TX, chip="ESP32-S3",
               config={"channel": 11})
    export = tmp_path / "export.sh"
    export.write_text("")
    hotspots = [fw.HotspotInfo(connection="Hotspot", iface="wlp2s0u2",
                               ssid="CSIDetector", channel=11, ip="10.42.1.1")]
    t = gui.DevicesTab(reg, list_boards=_boards,
                       detect_hotspots=lambda: hotspots,
                       settings_path=tmp_path / "setup.json",
                       idf_export=str(export), auto_refresh=False)
    yield t
    t.shutdown()
    t.deleteLater()


def _select(tab, port_or_mac):
    for i, row in enumerate(tab._rows):
        if (row.board and row.board.port == port_or_mac) or row.mac == port_or_mac:
            tab.table.selectRow(i)
            return row
    raise AssertionError(f"no row for {port_or_mac}")


def test_table_lists_attached_and_registered_devices(tab):
    ports = [tab.table.item(i, 0).text() for i in range(tab.table.rowCount())]
    assert ports == ["/dev/ttyACM0", "/dev/ttyUSB0", "(not connected)"]
    assert tab.table.item(2, 4).text() == "Corner TX"
    assert "Identify" in tab.table.item(1, 3).text()   # bridge board, MAC unknown


def test_flash_disabled_for_unattached_rows(tab):
    _select(tab, FAR_TX)
    assert not tab.flash_btn.isEnabled()
    assert tab.save_details_btn.isEnabled()        # can still label it
    _select(tab, "/dev/ttyACM0")
    assert tab.flash_btn.isEnabled()


def test_rx_alert_plan_prefills_registered_tx_filter(tab):
    _select(tab, "/dev/ttyACM0")
    tab.mode_rx_alert.setChecked(True)
    assert tab.macs_edit.text() == FAR_TX
    plan = tab.plan_flash()
    assert plan.errors == [] and plan.warnings == []
    assert plan.job.role == dv.ROLE_RX
    assert plan.job.config.tx_macs == [FAR_TX]
    assert plan.job.config.uart_only


def test_localize_plan_warns_on_wrong_host_ip(tab):
    _select(tab, "/dev/ttyACM0")
    tab.mode_rx_loc.setChecked(True)
    tab.ssid_edit.setText("CSIDetector")
    tab.host_ip_edit.setText("10.42.0.1")      # the bug that ate an afternoon
    plan = tab.plan_flash()
    assert plan.job is not None
    assert any("isn't this machine's hotspot address" in w for w in plan.warnings)


def test_localize_plan_requires_ssid(tab):
    _select(tab, "/dev/ttyACM0")
    tab.mode_rx_loc.setChecked(True)
    tab.ssid_edit.setText("")
    plan = tab.plan_flash()
    assert plan.job is None
    assert any("SSID" in e for e in plan.errors)


def test_detect_hotspot_fills_form(tab):
    tab.mode_rx_loc.setChecked(True)
    tab.channel_spin.setValue(3)
    tab.detect_hotspot()
    assert tab.ssid_edit.text() == "CSIDetector"
    assert tab.host_ip_edit.text() == "10.42.1.1"
    assert tab.channel_spin.value() == 11


def test_rx_plan_warns_when_tx_on_other_channel(tab):
    _select(tab, "/dev/ttyACM0")
    tab.mode_rx_alert.setChecked(True)
    tab.channel_spin.setValue(6)
    plan = tab.plan_flash()
    assert any("channel 11" in w for w in plan.warnings)


def test_chip_target_mismatch_is_an_error(tab):
    row = _select(tab, "/dev/ttyACM0")
    row.board.chip = "ESP32-C3"
    tab.target_combo.setCurrentText("esp32s3")
    plan = tab.plan_flash()
    assert any("ESP32-C3" in e for e in plan.errors)


def test_missing_idf_export_is_an_error(tab, tmp_path):
    _select(tab, "/dev/ttyACM0")
    tab.export_edit.setText(str(tmp_path / "nope.sh"))
    assert any("export.sh" in e for e in tab.plan_flash().errors)


def test_save_details_writes_registry(tab):
    _select(tab, "/dev/ttyACM0")
    tab.label_edit.setText("South RX")
    tab.role_combo.setCurrentIndex(2)
    tab.save_details()
    rec = tab.registry.get(RX_MAC)
    assert rec.label == "South RX" and rec.role == dv.ROLE_RX


def test_bridge_board_cannot_be_labelled_until_identified(tab):
    _select(tab, "/dev/ttyUSB0")
    assert not tab.save_details_btn.isEnabled()
    assert tab.identify_btn.isEnabled()


def test_successful_flash_records_board(tab):
    _select(tab, "/dev/ttyACM0")
    tab.mode_tx.setChecked(True)
    plan = tab.plan_flash()
    assert plan.errors == []
    got = []
    tab.flashed.connect(lambda mac, role: got.append((mac, role)))
    tab._pending = {"kind": "flash", "job": plan.job, "mac": RX_MAC, "chip": "ESP32-S3"}
    tab._on_process_finished(0, "Hash of data verified.\n")
    rec = tab.registry.get(RX_MAC)
    assert rec.role == dv.ROLE_TX and rec.firmware == "csi_transmitter"
    assert rec.config == {"channel": 11, "rate_hz": 100} and rec.last_flashed > 0
    assert got == [(RX_MAC, dv.ROLE_TX)]


def test_failed_flash_shows_hints_without_recording(tab, monkeypatch):
    shown = []
    monkeypatch.setattr(gui.QMessageBox, "warning",
                        lambda *a, **k: shown.append(a[2]))
    _select(tab, "/dev/ttyACM0")
    plan = tab.plan_flash()
    tab._pending = {"kind": "flash", "job": plan.job, "mac": RX_MAC, "chip": ""}
    tab._on_process_finished(2, "serial.serialutil.SerialException: could not open port")
    assert tab.registry.get(RX_MAC) is None
    assert shown and "busy" in shown[0]
    assert "[hint]" in tab.log.toPlainText()


def test_identify_result_updates_row_and_registry(tab):
    _select(tab, "/dev/ttyUSB0")
    tab._pending = {"kind": "identify", "port": "/dev/ttyUSB0",
                    "usb_serial": "0001", "cmd": "read_mac"}
    tab._on_process_finished(0, "Chip is ESP32-S3 (QFN56)\nMAC: 34:85:18:aa:bb:cc\n")
    tab._force_refresh()
    row = _select(tab, "/dev/ttyUSB0")
    assert row.mac == "34:85:18:aa:bb:cc" and row.chip == "ESP32-S3"
    assert tab.registry.get("34:85:18:aa:bb:cc") is not None
    assert tab.save_details_btn.isEnabled()


def test_settings_persist_without_password(tab, tmp_path):
    tab.mode_rx_loc.setChecked(True)
    tab.ssid_edit.setText("CSIDetector")
    tab.pass_edit.setText("hunter22")
    tab._save_settings()
    text = (tmp_path / "setup.json").read_text()
    assert "CSIDetector" in text and "hunter22" not in text


# --------------------------------------------------------------------------
# ProcessRunner with real processes
# --------------------------------------------------------------------------

def _run_until_finished(runner, timeout_ms=10000):
    loop = QEventLoop()
    result = {}

    def done(code, out):
        result.update(code=code, out=out)
        loop.quit()

    runner.finished.connect(done)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    return result


def test_process_runner_streams_lines_and_progress(qt_app):
    runner = gui.ProcessRunner()
    lines, progress = [], []
    runner.line.connect(lines.append)
    runner.progress.connect(lambda ph, p: progress.append((ph, p)))
    script = (r"echo hello; echo '[5/10] Building x'; "
              r"printf 'Writing at 0x0001 [==] 40.0%%\rWriting at 0x0001 [===] 90.0%%\n'; "
              r"echo bye; exit 3")
    runner.start(["bash", "-c", script], dict(os.environ))
    res = _run_until_finished(runner)
    assert res["code"] == 3
    assert "hello" in lines and "bye" in lines and "[5/10] Building x" in lines
    assert not any(l.startswith("Writing at") for l in lines)
    assert ("build", 50) in progress and ("flash", 40) in progress and ("flash", 90) in progress
    assert not runner.running


def test_process_runner_cancel_kills_grandchildren(qt_app):
    runner = gui.ProcessRunner()
    child = {}

    def on_line(text):
        if text.startswith("child "):
            child["pid"] = int(text.split()[1])
            QTimer.singleShot(0, runner.cancel)

    runner.line.connect(on_line)
    runner.start(["bash", "-c", "sleep 60 & echo \"child $!\"; wait"], dict(os.environ))
    res = _run_until_finished(runner)
    assert res["code"] == -1 and runner.cancelled
    pid = child["pid"]
    stat = Path(f"/proc/{pid}/stat")
    # Gone, or a zombie awaiting reaping by PID 1 — either way not sleeping.
    assert not stat.exists() or stat.read_text().split(") ")[1].startswith("Z")


def test_process_runner_reports_failed_start(qt_app):
    runner = gui.ProcessRunner()
    runner.start(["/nonexistent/binary-xyz"], dict(os.environ))
    res = _run_until_finished(runner)
    assert res["code"] == -1 and "could not start" in res["out"]
