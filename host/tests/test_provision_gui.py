"""Setup GUI (Devices & Flashing) under offscreen Qt.

Hermetic: fake board enumeration, fake hotspot detection, a temp device
registry, and a temp settings file. Real QProcesses are used only for the
output-streaming and cancel tests (plain bash, no ESP-IDF).
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import pytest

pytest.importorskip("PyQt6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEventLoop, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from csidetector.provision import devices as dv  # noqa: E402
from csidetector.provision import firmware as fw  # noqa: E402
from csidetector.provision import gui  # noqa: E402

POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX-only (bash, file modes, symlinks or /proc)")

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


def _finish_flash(tab, job, code, output, mac="", chip=""):
    """Drive the Flasher's completion path as if idf.py had exited."""
    tab.flasher._pending = {"kind": "flash", "job": job, "mac": mac, "chip": chip}
    tab.flasher._on_finished(code, output)


def _finish_identify(tab, port, usb_serial, code, output, tmp_path):
    tab.flasher._pending = {"kind": "identify", "port": port, "usb_serial": usb_serial,
                            "cmd": "read_mac", "export": tmp_path / "export.sh"}
    tab.flasher._on_finished(code, output)


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
    _finish_flash(tab, plan.job, 0, "Hash of data verified.\n", mac=RX_MAC, chip="ESP32-S3")
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
    _finish_flash(tab, plan.job, 2, "serial.serialutil.SerialException: could not open port", mac=RX_MAC, chip="")
    assert tab.registry.get(RX_MAC) is None
    assert shown and "busy" in shown[0]
    assert "[hint]" in tab.log.toPlainText()


def test_identify_result_updates_row_and_registry(tab, tmp_path):
    _select(tab, "/dev/ttyUSB0")
    _finish_identify(tab, "/dev/ttyUSB0", "0001", 0,
                     "Chip is ESP32-S3 (QFN56)\nMAC: 34:85:18:aa:bb:cc\n", tmp_path)
    tab._force_refresh()
    row = _select(tab, "/dev/ttyUSB0")
    assert row.mac == "34:85:18:aa:bb:cc" and row.chip == "ESP32-S3"
    assert tab.registry.get("34:85:18:aa:bb:cc") is not None
    assert tab.save_details_btn.isEnabled()


def test_identify_button_click_runs_read_mac(tab, monkeypatch):
    """Regression: clicked(bool) used to arrive as esptool_cmd=False."""
    started = []
    monkeypatch.setattr(tab.flasher._runner, "start",
                        lambda argv, env: started.append(argv))
    _select(tab, "/dev/ttyUSB0")
    tab.identify_btn.click()
    assert started, "identify did not start"
    assert started[0][-1].endswith("--port /dev/ttyUSB0 read_mac")
    assert tab.flasher._pending["cmd"] == "read_mac"


def test_identify_retries_with_dashed_command_on_invalid_choice(tab, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(tab.flasher._runner, "start",
                        lambda argv, env: started.append(argv))
    _finish_identify(tab, "/dev/ttyUSB0", "0001", 2,
                     "esptool: error: argument operation: invalid choice: 'read_mac'",
                     tmp_path)
    assert started and started[0][-1].endswith("--port /dev/ttyUSB0 read-mac")


def test_selection_clears_when_selected_board_disappears(qt_app, tmp_path):
    attached = [
        dv.AttachedBoard("/dev/ttyACM0", 0x303A, 0x1001, "", "", mac="aa:aa:aa:aa:aa:00"),
        dv.AttachedBoard("/dev/ttyACM1", 0x303A, 0x1001, "", "", mac="aa:aa:aa:aa:aa:01"),
        dv.AttachedBoard("/dev/ttyACM2", 0x303A, 0x1001, "", "", mac="aa:aa:aa:aa:aa:02"),
    ]
    t = gui.DevicesTab(dv.Registry(tmp_path / "d.json"), list_boards=lambda: list(attached),
                       detect_hotspots=lambda: [], settings_path=tmp_path / "s.json",
                       auto_refresh=False)
    try:
        t.select_port("/dev/ttyACM1")
        del attached[1]                      # unplugged (BOOT + replug, reset…)
        t.refresh()
        assert t._selected_row() is None     # not silently ACM2
        assert not t.flash_btn.isEnabled()
    finally:
        t.shutdown()


def test_flashing_a_tx_adds_it_to_the_rx_filter(tab):
    _select(tab, "/dev/ttyACM0")
    tab.mode_tx.setChecked(True)
    plan = tab.plan_flash()
    _finish_flash(tab, plan.job, 0, "", mac=RX_MAC, chip="")
    assert tab.macs_edit.text() == f"{FAR_TX}, {RX_MAC}"


def test_rx_plan_warns_when_registered_tx_missing_from_filter(tab):
    tab.registry.upsert(TX_MAC, label="Desk TX", role=dv.ROLE_TX)
    _select(tab, "/dev/ttyACM0")
    tab.mode_rx_loc.setChecked(True)
    tab.ssid_edit.setText("CSIDetector")
    tab.host_ip_edit.setText("10.42.1.1")
    tab.macs_edit.setText(FAR_TX)            # stale: doesn't include Desk TX
    plan = tab.plan_flash()
    assert any("Desk TX" in w and "ignore" in w for w in plan.warnings)


def test_alert_rx_prefills_only_the_latest_tx(tab):
    tab.registry.upsert(TX_MAC, label="Desk TX", role=dv.ROLE_TX,
                        last_flashed=time.time())
    _select(tab, "/dev/ttyACM0")
    tab.mode_rx_loc.setChecked(True)
    tab.fill_tx_macs()
    assert set(tab._filter_macs()) == {FAR_TX, TX_MAC}
    tab.mode_rx_alert.setChecked(True)
    tab.fill_tx_macs()
    assert tab._filter_macs() == [TX_MAC]
    # Not-listed transmitters aren't nagged about in alert mode: one is right.
    assert not any("ignore" in w for w in tab.plan_flash().warnings)


def test_alert_rx_with_several_txs_warns_about_interleaving(tab):
    _select(tab, "/dev/ttyACM0")
    tab.mode_rx_alert.setChecked(True)
    tab.macs_edit.setText(f"{FAR_TX}, {TX_MAC}")
    assert any("exactly one transmitter" in w for w in tab.plan_flash().warnings)


def test_flash_output_mac_wins_over_inferred_mac(tab):
    _select(tab, "/dev/ttyUSB0")
    tab.mode_tx.setChecked(True)
    plan = tab.plan_flash()
    stale = "11:11:11:11:11:11"              # e.g. from an Identify on a swapped board
    _finish_flash(tab, plan.job, 0, "Chip is ESP32-S3 (QFN56)\nMAC: 34:85:18:aa:bb:cc\n", mac=stale, chip="")
    assert tab.registry.get(stale) is None
    rec = tab.registry.get("34:85:18:aa:bb:cc")
    assert rec.role == dv.ROLE_TX and rec.chip == "ESP32-S3"


def test_bridge_board_flash_registers_mac_without_identify(tab):
    _select(tab, "/dev/ttyUSB0")
    tab.mode_tx.setChecked(True)
    plan = tab.plan_flash()
    _finish_flash(tab, plan.job, 0, "Chip type:   ESP32-S3 (QFN56)\nMAC:   34:85:18:aa:bb:dd\n", mac="", chip="")
    assert tab.registry.get("34:85:18:aa:bb:dd").role == dv.ROLE_TX


def test_rx_alert_on_bridge_board_warns_about_native_usb(tab):
    _select(tab, "/dev/ttyUSB0")
    tab.mode_rx_alert.setChecked(True)
    tab.target_combo.setCurrentText("esp32s3")
    assert any("native USB" in w for w in tab.plan_flash().warnings)
    tab.target_combo.setCurrentText("esp32")      # classic ESP32: UART console
    assert not any("native USB" in w for w in tab.plan_flash().warnings)


def test_localize_plan_warns_when_no_hotspot_running(qt_app, tmp_path):
    export = tmp_path / "export.sh"
    export.write_text("")
    t = gui.DevicesTab(dv.Registry(tmp_path / "d.json"), list_boards=_boards,
                       detect_hotspots=lambda: [], settings_path=tmp_path / "s.json",
                       idf_export=str(export), auto_refresh=False)
    try:
        t.select_port("/dev/ttyACM0")
        t.mode_rx_loc.setChecked(True)
        t.ssid_edit.setText("CSIDetector")
        plan = t.plan_flash()
        assert plan.job is not None
        assert any("No running hotspot" in w for w in plan.warnings)
    finally:
        t.shutdown()


def test_tx_plan_warns_when_rx_on_other_channel(tab):
    tab.registry.upsert(TX_MAC, role=dv.ROLE_RX, label="North RX", config={"channel": 6})
    _select(tab, "/dev/ttyACM0")
    tab.mode_tx.setChecked(True)
    assert any("North RX" in w and "channel 6" in w for w in tab.plan_flash().warnings)


def test_fill_tx_macs_reports_truncation(tab):
    for i in range(5):
        tab.registry.upsert(f"aa:bb:cc:dd:ee:{i:02x}", role=dv.ROLE_TX)
    tab.mode_rx_loc.setChecked(True)
    tab.fill_tx_macs()
    assert len(tab._filter_macs()) == fw.MAX_FILTER_MACS
    assert "left out" in tab.log.toPlainText()


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


@POSIX_ONLY
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


@POSIX_ONLY
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
    stat = Path(f"/proc/{child['pid']}/stat")

    def dead() -> bool:
        # Gone, or a zombie awaiting reaping by PID 1 — either way not sleeping.
        try:
            return stat.read_text().split(") ")[1].startswith("Z")
        except (FileNotFoundError, ProcessLookupError):
            return True

    # SIGTERM delivery isn't instantaneous under load; allow a moment.
    deadline = time.monotonic() + 3.0
    while not dead() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert dead()


def test_process_runner_reports_failed_start(qt_app):
    runner = gui.ProcessRunner()
    runner.start(["/nonexistent/binary-xyz"], dict(os.environ))
    res = _run_until_finished(runner)
    assert res["code"] == -1 and "could not start" in res["out"]



def _live_sleeps(arg: str) -> list[int]:
    """PIDs (not zombies) whose argv is exactly ``sleep <arg>``."""
    want = b"sleep\0" + arg.encode() + b"\0"
    out = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes()
            if argv != want and not argv.endswith(b"/" + want):
                continue
            if (proc / "stat").read_text().split(") ")[1].startswith("Z"):
                continue
        except OSError:
            continue
        out.append(int(proc.name))
    return out


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc")
@POSIX_ONLY
def test_cancel_catches_children_forked_during_cancel(qt_app):
    """A /proc walk then kill misses a child forked in between (esptool
    starting up at the moment you click Cancel): it'd be orphaned holding
    the serial port. Signalling the process group can't miss it."""
    # A unique duration nothing else on the machine is sleeping for.
    arg = f"6000.{uuid.uuid4().int % 10**9:09d}"
    script = f"while true; do sleep {arg} & sleep 0.002; done"
    try:
        for _ in range(15):
            runner = gui.ProcessRunner()
            runner.start(["bash", "-c", script], dict(os.environ))
            QTimer.singleShot(60, runner.cancel)
            _run_until_finished(runner)
            deadline = time.monotonic() + 3
            while _live_sleeps(arg) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not _live_sleeps(arg)
    finally:
        for pid in _live_sleeps(arg):
            os.kill(pid, 9)


def test_wizard_retires_the_open_monitor_before_rewriting_its_state(qt_app, tmp_path):
    """An open monitor writes the settings it loaded at startup back on its
    next Calibrate, undoing the wizard's new receiver port."""
    win = gui.SetupWindow(dv.Registry(tmp_path / "devices.json"), idf_export="",
                          list_boards=_boards, detect_hotspots=lambda: [],
                          settings_path=tmp_path / "setup.json", auto_refresh=False)
    wiz = win.open_wizard()
    assert wiz.before_alert_apply == win.retire_monitor
    assert win.open_wizard() is wiz                  # a second click reuses it
    wiz.reject()
    assert win.wizard is None

    class FakeMonitor:
        closed = False

        def close(self):
            self.closed = True

    mon = FakeMonitor()
    win._monitor = mon
    win.retire_monitor()
    assert mon.closed and win._monitor is None
    assert win._retired_monitors == [mon]    # kept alive while its thread winds down
    win.retire_monitor()                     # nothing open: no-op
    win.devices.shutdown()
    win.deleteLater()


def test_excepthook_reports_instead_of_aborting(qt_app, monkeypatch, capsys):
    """PyQt6 aborts on an exception escaping a slot unless the hook is set."""
    shown = []
    monkeypatch.setattr(gui.QMessageBox, "critical", lambda *a: shown.append(a))
    monkeypatch.setattr(gui.sys, "excepthook", gui.sys.excepthook)   # restored after
    gui._install_excepthook()
    try:
        raise KeyError("rxs")
    except KeyError as exc:
        gui.sys.excepthook(type(exc), exc, exc.__traceback__)
    assert shown and "KeyError" in shown[0][2]
    assert "Traceback" in capsys.readouterr().err


def test_windows_flash_captures_the_idf_environment_once(qt_app, tmp_path, monkeypatch):
    """Windows: export.bat runs once through cmd.exe (native command line),
    its environment is captured and cached, then idf.py runs directly in
    it. The dumped environment never reaches the log."""
    monkeypatch.setattr(fw, "IS_WINDOWS", True)
    monkeypatch.setattr(fw, "_IDF_ENV_CACHE", {})
    export = tmp_path / "esp idf" / "export.bat"
    export.parent.mkdir()
    export.write_text("")
    (tmp_path / "idf" / "tools").mkdir(parents=True)
    flasher = gui.Flasher(dv.Registry(tmp_path / "devices.json"))
    started, lines = [], []
    flasher.line.connect(lines.append)
    monkeypatch.setattr(flasher._runner, "start",
                        lambda argv, env, native_args=None: started.append((argv, env, native_args)))
    job = fw.FlashJob(port="COM5", config=fw.TxConfig(), build_root=tmp_path / "build")
    flasher.flash(job, export)
    argv, _env, native = started[0]
    assert argv == ["cmd.exe"] and f'call "{export}"' in native
    dump = (f"Activating ESP-IDF 5.2\n{fw.ENV_MARKER} \n"
            f"IDF_PATH={tmp_path / 'idf'}\nPath={os.path.dirname(sys.executable)}\n"
            f"SECRET_TOKEN=hunter22\n")
    for text in dump.splitlines():
        flasher._on_line(text)
    flasher._on_finished(0, dump)
    argv, env, native = started[1]
    assert native is None and argv[1] == str(tmp_path / "idf" / "tools" / "idf.py")
    assert argv[-3:] == ["-p", "COM5", "flash"] and env["SECRET_TOKEN"] == "hunter22"
    assert "Activating ESP-IDF 5.2" in lines
    assert not any("hunter22" in l for l in lines)          # env dump not logged
    # Second use: cached, no cmd.exe stage.
    flasher._pending = None
    started.clear()
    flasher.identify("COM5", "", export)
    assert started[0][0][1:] == ["-m", "esptool", "--port", "COM5", "read_mac"]


def test_windows_failed_export_reports_a_flash_failure(qt_app, tmp_path, monkeypatch):
    monkeypatch.setattr(fw, "IS_WINDOWS", True)
    monkeypatch.setattr(fw, "_IDF_ENV_CACHE", {})
    export = tmp_path / "export.bat"
    export.write_text("")
    flasher = gui.Flasher(dv.Registry(tmp_path / "devices.json"))
    monkeypatch.setattr(flasher._runner, "start", lambda *a, **k: None)
    results = []
    flasher.flash_finished.connect(results.append)
    flasher.flash(fw.FlashJob(port="COM5", config=fw.TxConfig(),
                              build_root=tmp_path / "build"), export)
    flasher._on_finished(97, "ERROR: tool xtensa-esp32s3-elf has no installed versions\n"
                             "ESP-IDF export.bat failed to load\n")
    assert results and not results[0].ok
    assert any("environment didn't load" in h for h in fw.failure_hints(results[0].output))
