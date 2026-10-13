"""Setup GUI — device inventory and one-click flashing.

``python run.py setup`` opens a window that:

  * lists attached ESP32 boards (port, chip, MAC/serial) next to the ones
    already in the device registry, with editable labels and roles;
  * identifies bridge-based boards via esptool ``read_mac``;
  * flashes a selected board as a transmitter or receiver (alert-mode
    USB-only, or localize-mode hotspot), pre-filling the receiver's TX
    MAC filter from registered transmitters and the hotspot SSID /
    channel / host IP from NetworkManager;
  * warns before flashing values that can't work (host IP that isn't this
    machine's hotspot address, channel mismatches, empty MAC filter).

Builds and flashes run in a QProcess on the GUI thread's event loop (no
worker threads); output streams into the log pane.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QComboBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QRadioButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)

from csidetector.provision import devices as dv
from csidetector.provision import firmware as fw

_OUTPUT_CAP = 256 * 1024          # chars of process output kept for hints
_AUTO_REFRESH_MS = 2000

MODE_TX = "tx"
MODE_RX_ALERT = "rx_alert"
MODE_RX_LOCALIZE = "rx_localize"


def default_settings_path() -> Path:
    return Path(os.path.expanduser("~/.csidetector/setup.json"))


# --------------------------------------------------------------------------
# QProcess wrapper
# --------------------------------------------------------------------------

class ProcessRunner(QObject):
    """Runs one command at a time; streams lines and progress."""

    line = pyqtSignal(str)
    progress = pyqtSignal(str, int)       # phase, percent
    finished = pyqtSignal(int, str)       # exit code (-1 = killed/failed), output

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._proc: Optional[QProcess] = None
        self._buf = ""
        self._output = ""
        self.cancelled = False

    @property
    def running(self) -> bool:
        return self._proc is not None

    def start(self, argv: list[str], env: dict) -> None:
        if self._proc is not None:
            raise RuntimeError("a process is already running")
        proc = QProcess(self)
        qenv = QProcessEnvironment()
        for key, value in env.items():
            qenv.insert(key, value)
        proc.setProcessEnvironment(qenv)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_ready)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        self._buf = ""
        self._output = ""
        self.cancelled = False
        self._proc = proc
        proc.start(argv[0], argv[1:])

    def cancel(self) -> None:
        if self._proc is None:
            return
        self.cancelled = True
        pid = int(self._proc.processId() or 0)
        if pid:
            fw.terminate_tree(pid)
        self._proc.kill()

    def wait(self, msecs: int) -> None:
        if self._proc is not None:
            self._proc.waitForFinished(msecs)

    # ------------------------------------------------------------------

    def _on_ready(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        self._output = (self._output + data)[-_OUTPUT_CAP:]
        self._buf += data
        # esptool v5 redraws progress with '\r'; treat it as a line break.
        parts = re.split(r"[\r\n]", self._buf)
        self._buf = parts.pop()
        for part in parts:
            self._emit_line(part)

    def _emit_line(self, text: str) -> None:
        if not text.strip():
            return
        prog = fw.parse_progress(text)
        if prog is not None:
            self.progress.emit(*prog)
            if prog[0] == "flash":
                return          # hundreds of near-identical lines; bar shows it
        self.line.emit(text)

    def _on_error(self, error: QProcess.ProcessError) -> None:
        # FailedToStart is the one error after which finished() never fires.
        if error == QProcess.ProcessError.FailedToStart and self._proc is not None:
            msg = f"could not start {self._proc.program()}: {self._proc.errorString()}"
            self._finish(-1, msg)

    def _on_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        if self._proc is None:
            return
        self._on_ready()
        if self._buf:
            self._emit_line(self._buf)
            self._buf = ""
        crashed = status == QProcess.ExitStatus.CrashExit
        self._finish(-1 if (crashed or self.cancelled) else int(code), "")

    def _finish(self, code: int, extra: str) -> None:
        proc, self._proc = self._proc, None
        if extra:
            self._output += "\n" + extra
            self.line.emit(extra)
        if proc is not None:
            proc.deleteLater()
        self.finished.emit(code, self._output)


# --------------------------------------------------------------------------
# Devices & Flashing tab
# --------------------------------------------------------------------------

class _Row:
    __slots__ = ("board", "record")

    def __init__(self, board: Optional[dv.AttachedBoard],
                 record: Optional[dv.DeviceRecord]):
        self.board = board
        self.record = record

    @property
    def mac(self) -> str:
        if self.board is not None and self.board.mac:
            return self.board.mac
        return self.record.mac if self.record else ""

    @property
    def chip(self) -> str:
        if self.board is not None and self.board.chip:
            return self.board.chip
        return self.record.chip if self.record else ""

    def identity(self) -> tuple:
        return ("port", self.board.port) if self.board else ("mac", self.mac)


class FlashPlan:
    """Result of reading the flash form: a job, or why we can't flash."""

    def __init__(self):
        self.job: Optional[fw.FlashJob] = None
        self.errors: list[str] = []
        self.warnings: list[str] = []


class DevicesTab(QWidget):
    COLUMNS = ("Port", "Interface", "Chip", "MAC / USB serial", "Label",
               "Role", "Last flashed")

    flashed = pyqtSignal(str, str)        # mac, role — for the wizard

    def __init__(self, registry: dv.Registry,
                 list_boards: Callable[[], list[dv.AttachedBoard]] = dv.list_boards,
                 detect_hotspots: Callable[[], list[fw.HotspotInfo]] = fw.detect_hotspots,
                 settings_path: Optional[Path] = None,
                 idf_export: Optional[str] = None,
                 auto_refresh: bool = True,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.registry = registry
        self._list_boards = list_boards
        self._detect_hotspots = detect_hotspots
        self._settings_path = settings_path or default_settings_path()
        self._rows: list[_Row] = []
        self._table_sig: Optional[str] = None
        # (port, usb_serial) → (chip, mac) learned via esptool.
        self._identified: dict[tuple[str, str], tuple[str, str]] = {}
        self._pending: Optional[dict] = None

        self._runner = ProcessRunner(self)
        self._runner.line.connect(self._append_log)
        self._runner.progress.connect(self._on_progress)
        self._runner.finished.connect(self._on_process_finished)

        self._build_ui()
        self._load_settings(idf_export)
        self.refresh()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._auto_refresh)
        if auto_refresh:
            self._timer.start(_AUTO_REFRESH_MS)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        mono = QFont("monospace")
        mono.setStyleHint(QFont.StyleHint.Monospace)

        # Device table
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_selection)

        self.refresh_btn = QPushButton("Refresh")
        self.identify_btn = QPushButton("Identify (esptool)")
        self.refresh_btn.clicked.connect(self.refresh)
        self.identify_btn.clicked.connect(self.identify_selected)
        top = QHBoxLayout()
        top.addWidget(QLabel("Attached boards and registered devices"))
        top.addStretch(1)
        top.addWidget(self.refresh_btn)
        top.addWidget(self.identify_btn)

        # Details: label / role
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("e.g. South RX, Desk TX")
        self.role_combo = QComboBox()
        self.role_combo.addItems(["—", "TX", "RX"])
        self.save_details_btn = QPushButton("Save")
        self.save_details_btn.clicked.connect(self.save_details)
        details = QHBoxLayout()
        details.addWidget(QLabel("Label:"))
        details.addWidget(self.label_edit, 2)
        details.addWidget(QLabel("Role:"))
        details.addWidget(self.role_combo)
        details.addWidget(self.save_details_btn)

        # Flash form
        self.mode_tx = QRadioButton("Transmitter (TX)")
        self.mode_rx_alert = QRadioButton("Receiver — alert mode (USB only)")
        self.mode_rx_loc = QRadioButton("Receiver — localize mode (hotspot)")
        self._mode_group = QButtonGroup(self)
        for i, b in enumerate((self.mode_tx, self.mode_rx_alert, self.mode_rx_loc)):
            self._mode_group.addButton(b, i)
        self.mode_rx_alert.setChecked(True)
        self._mode_group.idToggled.connect(lambda *_: self._sync_form())
        modes = QHBoxLayout()
        for b in (self.mode_tx, self.mode_rx_alert, self.mode_rx_loc):
            modes.addWidget(b)
        modes.addStretch(1)

        self.target_combo = QComboBox()
        self.target_combo.addItems(fw.TARGETS)
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(1, 13)
        self.channel_spin.setValue(11)
        self.rate_spin = QSpinBox()
        self.rate_spin.setRange(1, 500)
        self.rate_spin.setValue(100)
        self.rate_spin.setSuffix(" Hz")

        self.macs_edit = QLineEdit()
        self.macs_edit.setPlaceholderText("aa:bb:cc:dd:ee:ff, … (up to 4; blank = any)")
        self.use_txs_btn = QPushButton("Use registered TXs")
        self.use_txs_btn.clicked.connect(self.fill_tx_macs)
        macs_row = QHBoxLayout()
        macs_row.addWidget(self.macs_edit, 1)
        macs_row.addWidget(self.use_txs_btn)
        self._macs_widget = QWidget()
        self._macs_widget.setLayout(macs_row)
        macs_row.setContentsMargins(0, 0, 0, 0)

        self.ssid_edit = QLineEdit()
        self.pass_edit = QLineEdit()
        self.pass_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.pass_edit.setPlaceholderText("blank for an open hotspot")
        self.host_ip_edit = QLineEdit("10.42.0.1")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(5566)
        self.detect_btn = QPushButton("Detect from hotspot")
        self.detect_btn.clicked.connect(self.detect_hotspot)
        ip_row = QHBoxLayout()
        ip_row.addWidget(self.host_ip_edit, 1)
        ip_row.addWidget(self.detect_btn)
        ip_row.setContentsMargins(0, 0, 0, 0)
        self._ip_widget = QWidget()
        self._ip_widget.setLayout(ip_row)

        form = QFormLayout()
        form.addRow("Firmware:", modes)
        form.addRow("Target chip:", self.target_combo)
        form.addRow("Wi-Fi channel:", self.channel_spin)
        form.addRow("Broadcast rate:", self.rate_spin)
        form.addRow("TX MAC filter:", self._macs_widget)
        form.addRow("Hotspot SSID:", self.ssid_edit)
        form.addRow("Hotspot password:", self.pass_edit)
        form.addRow("Host IP:", self._ip_widget)
        form.addRow("Host UDP port:", self.port_spin)
        self._form = form

        self.export_edit = QLineEdit()
        self.export_edit.setPlaceholderText("path to esp-idf/export.sh")
        self.export_edit.editingFinished.connect(self._save_settings)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_export)
        export_row = QHBoxLayout()
        export_row.addWidget(QLabel("ESP-IDF:"))
        export_row.addWidget(self.export_edit, 1)
        export_row.addWidget(browse)

        self.flash_btn = QPushButton("Flash selected board")
        self.cancel_btn = QPushButton("Cancel")
        self.flash_btn.clicked.connect(self.flash_selected)
        self.cancel_btn.clicked.connect(self._runner.cancel)
        self.cancel_btn.setEnabled(False)
        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(self.cancel_btn)
        actions.addWidget(self.flash_btn)

        flash_box = QGroupBox("Flash")
        fb = QVBoxLayout(flash_box)
        fb.addLayout(form)
        fb.addLayout(export_row)
        fb.addLayout(actions)

        # Progress + log
        self.phase_label = QLabel("Idle")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        self.log.setFont(mono)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.table, 2)
        layout.addLayout(details)
        layout.addWidget(flash_box)
        prog = QHBoxLayout()
        prog.addWidget(self.phase_label)
        prog.addWidget(self.progress, 1)
        layout.addLayout(prog)
        layout.addWidget(self.log, 2)

        self._sync_form()
        self._update_buttons()

    # ------------------------------------------------------------------
    # Settings (non-secret form values only)
    # ------------------------------------------------------------------

    def _load_settings(self, idf_export: Optional[str]) -> None:
        try:
            s = json.loads(self._settings_path.read_text())
            if not isinstance(s, dict):
                s = {}
        except (OSError, ValueError):
            s = {}
        mode = s.get("mode")
        {MODE_TX: self.mode_tx, MODE_RX_ALERT: self.mode_rx_alert,
         MODE_RX_LOCALIZE: self.mode_rx_loc}.get(mode, self.mode_rx_alert).setChecked(True)
        if s.get("target") in fw.TARGETS:
            self.target_combo.setCurrentText(s["target"])
        for spin, key in ((self.channel_spin, "channel"), (self.rate_spin, "rate_hz"),
                          (self.port_spin, "host_port")):
            if isinstance(s.get(key), int):
                spin.setValue(s[key])
        if isinstance(s.get("ssid"), str):
            self.ssid_edit.setText(s["ssid"])
        if isinstance(s.get("host_ip"), str):
            self.host_ip_edit.setText(s["host_ip"])
        export = idf_export or s.get("idf_export") or ""
        if not export:
            found = fw.find_idf_export()
            export = str(found) if found else ""
        self.export_edit.setText(export)
        self._sync_form()

    def _save_settings(self) -> None:
        s = {
            "mode": self.mode(),
            "target": self.target_combo.currentText(),
            "channel": self.channel_spin.value(),
            "rate_hz": self.rate_spin.value(),
            "ssid": self.ssid_edit.text(),
            "host_ip": self.host_ip_edit.text().strip(),
            "host_port": self.port_spin.value(),
            "idf_export": self.export_edit.text().strip(),
        }
        try:
            self._settings_path.parent.mkdir(parents=True, exist_ok=True)
            self._settings_path.write_text(json.dumps(s, indent=2))
        except OSError as exc:
            self._append_log(f"[setup] could not save settings: {exc}")

    def _browse_export(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ESP-IDF export.sh", os.path.expanduser("~/esp"),
            "export.sh (export.sh);;All files (*)")
        if path:
            self.export_edit.setText(path)
            self._save_settings()

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def mode(self) -> str:
        if self.mode_tx.isChecked():
            return MODE_TX
        if self.mode_rx_loc.isChecked():
            return MODE_RX_LOCALIZE
        return MODE_RX_ALERT

    def _auto_refresh(self) -> None:
        if not self._runner.running and self.isVisible():
            self.refresh()

    def refresh(self) -> None:
        if self._runner.running:
            return      # boards reset/re-enumerate mid-flash; don't flicker
        try:
            boards = self._list_boards()
        except Exception as exc:  # noqa: BLE001 — pyserial missing, sysfs errors
            boards = []
            self._append_log(f"[setup] could not list serial ports: {exc}")
        present = {(b.port, b.usb_serial) for b in boards}
        self._identified = {k: v for k, v in self._identified.items() if k in present}
        for b in boards:
            hit = self._identified.get((b.port, b.usb_serial))
            if hit:
                b.chip = hit[0] or b.chip
                b.mac = hit[1] or b.mac
        attached = {b.mac for b in boards if b.mac}
        rows = [_Row(b, self.registry.get(b.mac) if b.mac else None) for b in boards]
        rows += [_Row(None, r) for r in self.registry.all() if r.mac not in attached]

        sig = repr([(r.board and (r.board.port, r.board.mac, r.board.chip),
                     r.record and (r.record.mac, r.record.label, r.record.role,
                                   r.record.chip, r.record.last_flashed))
                    for r in rows])
        if sig == self._table_sig:
            return
        self._table_sig = sig

        selected = self._selected_row()
        keep = selected.identity() if selected else None
        self._rows = rows
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            b, r = row.board, row.record
            if b is not None:
                mac_text = row.mac or (f"unknown — Identify ({b.usb_serial})"
                                       if b.usb_serial else "unknown — Identify")
            else:
                mac_text = row.mac
            last = (time.strftime("%Y-%m-%d %H:%M", time.localtime(r.last_flashed))
                    if r and r.last_flashed else "")
            cells = (
                b.port if b else "(not connected)",
                b.interface if b else "",
                row.chip,
                mac_text,
                r.label if r else "",
                (r.role.upper() if r and r.role else ""),
                last,
            )
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if b is None:
                    item.setForeground(QBrush(QColor("gray")))
                if col == 3 and b is not None and b.usb_serial:
                    item.setToolTip(f"USB serial: {b.usb_serial}")
                self.table.setItem(i, col, item)
        self.table.blockSignals(False)

        if keep is not None:
            for i, row in enumerate(rows):
                if row.identity() == keep:
                    self.table.selectRow(i)
                    break
        self._on_selection()

    def _selected_row(self) -> Optional[_Row]:
        sel = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not sel:
            return None
        idx = sel[0].row()
        return self._rows[idx] if 0 <= idx < len(self._rows) else None

    def select_port(self, port: str) -> bool:
        for i, row in enumerate(self._rows):
            if row.board is not None and row.board.port == port:
                self.table.selectRow(i)
                return True
        return False

    def _on_selection(self) -> None:
        row = self._selected_row()
        rec = row.record if row else None
        self.label_edit.setText(rec.label if rec else "")
        self.role_combo.setCurrentIndex({"tx": 1, "rx": 2}.get(rec.role if rec else "", 0))
        target = dv.chip_to_target(row.chip) if row else ""
        if target in fw.TARGETS:
            self.target_combo.setCurrentText(target)
        self._update_buttons()

    def _update_buttons(self) -> None:
        busy = self._runner.running
        row = self._selected_row()
        attached = row is not None and row.board is not None
        self.refresh_btn.setEnabled(not busy)
        self.identify_btn.setEnabled(not busy and attached)
        self.flash_btn.setEnabled(not busy and attached)
        self.cancel_btn.setEnabled(busy)
        self.save_details_btn.setEnabled(row is not None and bool(row.mac))
        self.detect_btn.setEnabled(not busy)

    def save_details(self) -> None:
        row = self._selected_row()
        if row is None or not row.mac:
            return
        role = {1: dv.ROLE_TX, 2: dv.ROLE_RX}.get(self.role_combo.currentIndex(), "")
        changes = {"label": self.label_edit.text().strip(), "role": role}
        if row.chip:
            changes["chip"] = row.chip
        self.registry.upsert(row.mac, **changes)
        self._table_sig = None
        self.refresh()

    # ------------------------------------------------------------------
    # Form helpers
    # ------------------------------------------------------------------

    def _sync_form(self) -> None:
        mode = self.mode()
        rx = mode != MODE_TX
        loc = mode == MODE_RX_LOCALIZE
        for widget, visible in ((self.rate_spin, not rx),
                                (self._macs_widget, rx),
                                (self.ssid_edit, loc),
                                (self.pass_edit, loc),
                                (self._ip_widget, loc),
                                (self.port_spin, loc)):
            widget.setVisible(visible)
            label = self._form.labelForField(widget)
            if label is not None:
                label.setVisible(visible)
        if rx and not self.macs_edit.text().strip():
            self.fill_tx_macs()

    def fill_tx_macs(self) -> None:
        macs = self.registry.tx_macs()[:fw.MAX_FILTER_MACS]
        self.macs_edit.setText(", ".join(macs))

    def detect_hotspot(self) -> None:
        hotspots = self._detect_hotspots()
        if not hotspots:
            QMessageBox.information(
                self, "No hotspot found",
                "No active NetworkManager hotspot (AP-mode Wi-Fi connection) "
                "was found. Start it first (./start.sh), or type the values.")
            return
        hs = hotspots[0]
        self.ssid_edit.setText(hs.ssid)
        if hs.ip:
            self.host_ip_edit.setText(hs.ip)
        if 1 <= hs.channel <= 13:
            self.channel_spin.setValue(hs.channel)
        self._append_log(f"[setup] hotspot '{hs.connection}' on {hs.iface}: "
                         f"SSID={hs.ssid!r} channel={hs.channel or '?'} "
                         f"host IP={hs.ip or '?'}")
        for tx in self._txs_on_other_channel(self.channel_spin.value()):
            self._append_log(f"[setup] note: {tx} — receivers and transmitters "
                             f"must all share the hotspot channel.")

    def _txs_on_other_channel(self, channel: int) -> list[str]:
        out = []
        for rec in self.registry.all():
            ch = rec.config.get("channel") if isinstance(rec.config, dict) else None
            if rec.role == dv.ROLE_TX and isinstance(ch, int) and ch != channel:
                out.append(f"TX {rec.label or rec.mac} was flashed on channel {ch}")
        return out

    def plan_flash(self) -> FlashPlan:
        """Read the form into a FlashJob plus blocking errors and warnings.

        No dialogs here — flash_selected() presents the result — so the
        plan logic is testable headlessly.
        """
        plan = FlashPlan()
        row = self._selected_row()
        if row is None or row.board is None:
            plan.errors.append("Select an attached board first.")
            return plan
        mode = self.mode()
        channel = self.channel_spin.value()
        if mode == MODE_TX:
            config: fw.Config = fw.TxConfig(channel=channel,
                                            rate_hz=self.rate_spin.value())
        else:
            macs = [m for m in re.split(r"[,;\s]+", self.macs_edit.text()) if m]
            config = fw.RxConfig(channel=channel, tx_macs=macs)
            if mode == MODE_RX_LOCALIZE:
                config.ssid = self.ssid_edit.text()
                config.password = self.pass_edit.text()
                config.host_ip = self.host_ip_edit.text().strip()
                config.host_port = self.port_spin.value()
                if not config.ssid:
                    plan.errors.append("Localize mode needs the hotspot SSID "
                                       "(or choose alert mode for USB-only).")
        job = fw.FlashJob(port=row.board.port, config=config,
                          target=self.target_combo.currentText())
        plan.errors.extend(job.validate())
        if not self.export_edit.text().strip() or \
                not Path(self.export_edit.text().strip()).expanduser().is_file():
            plan.errors.append("ESP-IDF export.sh not found — set its path "
                               "below the form (usually ~/esp/esp-idf/export.sh).")
        if plan.errors:
            return plan

        if isinstance(config, fw.RxConfig):
            if not config.tx_macs:
                plan.warnings.append(
                    "The TX MAC filter is empty, so this receiver will report CSI "
                    "from ANY ESP-NOW transmitter nearby.")
            for note in self._txs_on_other_channel(channel):
                plan.warnings.append(f"{note}, but this receiver will use "
                                     f"channel {channel}.")
            if config.ssid:
                hotspots = self._detect_hotspots()
                if hotspots:
                    ips = {h.ip for h in hotspots if h.ip}
                    ssids = {h.ssid for h in hotspots}
                    chans = {h.channel for h in hotspots if h.channel}
                    if ips and config.host_ip not in ips:
                        plan.warnings.append(
                            f"Host IP {config.host_ip} isn't this machine's hotspot "
                            f"address ({', '.join(sorted(ips))}). Packets would go "
                            f"nowhere.")
                    if config.ssid not in ssids:
                        plan.warnings.append(
                            f"SSID {config.ssid!r} doesn't match the running hotspot "
                            f"({', '.join(sorted(ssids))}).")
                    if chans and channel not in chans:
                        plan.warnings.append(
                            f"Channel {channel} doesn't match the hotspot's channel "
                            f"({', '.join(str(c) for c in sorted(chans))}).")
        if row.chip:
            chip_target = dv.chip_to_target(row.chip)
            if chip_target and chip_target != job.target:
                plan.errors.append(f"This board is an {row.chip}; target is set "
                                   f"to {job.target}.")
        plan.job = job if not plan.errors else None
        return plan

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _export_path(self) -> Path:
        return Path(self.export_edit.text().strip()).expanduser()

    def flash_selected(self) -> None:
        if self._runner.running:
            return
        plan = self.plan_flash()
        if plan.errors:
            QMessageBox.warning(self, "Can't flash", "\n\n".join(plan.errors))
            return
        if plan.warnings:
            answer = QMessageBox.question(
                self, "Flash anyway?",
                "\n\n".join(plan.warnings) + "\n\nFlash anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.start_flash(plan.job)

    def start_flash(self, job: fw.FlashJob) -> None:
        row = self._selected_row()
        board = row.board if row and row.board and row.board.port == job.port else None
        try:
            changed = fw.prepare_build_dir(job)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Can't flash", f"Could not prepare build: {exc}")
            return
        self._save_settings()
        self.log.clear()
        self._append_log(
            f"[setup] flashing {job.project} ({job.target}) to {job.port}"
            + ("" if changed else " — config unchanged, reusing previous build"))
        self._pending = {
            "kind": "flash", "job": job,
            "mac": board.mac if board else "",
            "chip": board.chip if board else "",
        }
        self._start(fw.flash_command(job, self._export_path()), "Building…")

    def identify_selected(self, esptool_cmd: str = "read_mac") -> None:
        row = self._selected_row()
        if self._runner.running or row is None or row.board is None:
            return
        export = self._export_path()
        if not export.is_file():
            QMessageBox.warning(self, "ESP-IDF not found",
                                "Set the path to esp-idf/export.sh first.")
            return
        board = row.board
        self._pending = {"kind": "identify", "port": board.port,
                         "usb_serial": board.usb_serial, "cmd": esptool_cmd}
        self._append_log(f"[setup] identifying {board.port} …")
        self._start(fw.identify_command(board.port, export, esptool_cmd),
                    "Identifying…")

    def _start(self, argv: list[str], phase: str) -> None:
        self.progress.setValue(0)
        self.phase_label.setText(phase)
        self._runner.start(argv, fw.idf_env())
        self._update_buttons()

    def _on_progress(self, phase: str, pct: int) -> None:
        self.phase_label.setText("Building…" if phase == "build" else "Flashing…")
        self.progress.setValue(pct)

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def _on_process_finished(self, code: int, output: str) -> None:
        pending, self._pending = self._pending, None
        cancelled = self._runner.cancelled
        if pending is None:
            self._update_buttons()
            return
        if pending["kind"] == "identify":
            if code != 0 and "invalid choice" in output and pending["cmd"] == "read_mac":
                # A future esptool without the underscore alias.
                self._update_buttons()
                self.identify_selected("read-mac")
                return
            self._finish_identify(pending, code, output, cancelled)
        else:
            self._finish_flash(pending, code, output, cancelled)
        self._update_buttons()
        # Native-USB boards drop off the bus while resetting; look again
        # once they're back.
        QTimer.singleShot(1500, self._force_refresh)

    def _force_refresh(self) -> None:
        self._table_sig = None
        self.refresh()

    def _finish_identify(self, pending: dict, code: int, output: str,
                         cancelled: bool) -> None:
        chip, mac = dv.parse_esptool_identify(output)
        if code == 0 and mac:
            self._identified[(pending["port"], pending["usb_serial"])] = (chip, mac)
            rec = self.registry.get(mac)
            if rec is None or (chip and rec.chip != chip):
                self.registry.upsert(mac, chip=chip or (rec.chip if rec else ""))
            self.phase_label.setText(f"Identified {chip or 'board'} {mac}")
            self.progress.setValue(100)
            return
        self.phase_label.setText("Cancelled" if cancelled else "Identify failed")
        if not cancelled:
            self._show_failure("Identify failed", output)

    def _finish_flash(self, pending: dict, code: int, output: str,
                      cancelled: bool) -> None:
        job: fw.FlashJob = pending["job"]
        if code == 0:
            self.progress.setValue(100)
            self.phase_label.setText("Flashed ✓")
            mac = pending["mac"]
            if mac:
                rec = self.registry.get(mac)
                self.registry.upsert(
                    mac, role=job.role, firmware=job.project,
                    last_flashed=time.time(), config=job.config.summary(),
                    chip=pending["chip"] or (rec.chip if rec else ""))
                self._append_log(f"[setup] recorded {mac} as {job.role.upper()}")
                self.flashed.emit(mac, job.role)
            else:
                self._append_log("[setup] flashed OK. This board's MAC is unknown "
                                 "(USB-UART bridge) — click Identify to register it.")
                self.flashed.emit("", job.role)
            return
        if cancelled:
            self.phase_label.setText("Cancelled")
            self._append_log("[setup] cancelled")
            return
        self.phase_label.setText("Flash failed")
        self._show_failure("Flash failed", output)

    def _show_failure(self, title: str, output: str) -> None:
        hints = fw.failure_hints(output)
        for hint in hints:
            self._append_log(f"[hint] {hint}")
        body = "\n\n".join(hints) if hints else "See the log for details."
        QMessageBox.warning(self, title, body)

    # ------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self._runner.running

    def shutdown(self) -> None:
        self._timer.stop()
        if self._runner.running:
            self._runner.cancel()
            self._runner.wait(3000)


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class SetupWindow(QMainWindow):
    def __init__(self, registry: Optional[dv.Registry] = None,
                 idf_export: Optional[str] = None, **tab_kwargs):
        super().__init__()
        self.setWindowTitle("CSIDetector Setup")
        self.resize(980, 860)
        self.registry = registry or dv.Registry()
        self.devices = DevicesTab(self.registry, idf_export=idf_export, **tab_kwargs)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.devices, "Devices && Flashing")
        self.setCentralWidget(self.tabs)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.devices.busy:
            answer = QMessageBox.question(
                self, "Flash in progress",
                "A flash is still running. Cancel it and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.devices.shutdown()
        super().closeEvent(event)


def run_setup_gui(idf_export: Optional[str] = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    win = SetupWindow(idf_export=idf_export)
    win.show()
    return app.exec()
