"""Setup wizard: from bare boards to a working sensor.

    Mode → Environment → Transmitters → Receivers → (Telegram | links.json) → Finish

Alert mode ends with alert.toml written and the live monitor opened on
the receiver's port. Localize mode ends with the new boards merged into
links.json, ready for placement and per-link calibration.

Flashing goes through the same Flasher as the Devices tab, so the device
registry records boards identically whichever path flashed them.
"""

from __future__ import annotations

import ipaddress
import os
import threading
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QRadioButton, QSpinBox, QVBoxLayout, QWidget,
    QWizard, QWizardPage,
)

from csidetector.provision import devices as dv
from csidetector.provision import firmware as fw
from csidetector.provision import project_files as pf
from csidetector.provision.gui import Flasher, FlashResult

MODE_ALERT = "alert"
MODE_LOCALIZE = "localize"

PAGE_MODE, PAGE_ENV, PAGE_TX, PAGE_RX, PAGE_TELEGRAM, PAGE_LINKS, PAGE_FINISH = range(7)


def default_host_dir() -> Path:
    # host/csidetector/provision/wizard.py → host/ is parents[2]
    return Path(__file__).resolve().parents[2]


class WizardState:
    def __init__(self) -> None:
        self.mode = MODE_ALERT
        self.export = ""
        self.channel = 11
        self.ssid = ""
        self.password = ""
        self.host_ip = "10.42.0.1"
        self.host_port = 5566
        self.txs: dict[str, str] = {}                 # mac → label
        self.rxs: dict[str, tuple[str, str]] = {}     # mac → (label, port)


def _valid_ipv4(text: str) -> bool:
    try:
        ipaddress.IPv4Address(text)
        return True
    except (ipaddress.AddressValueError, ValueError):
        return False


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

class ModePage(QWizardPage):
    def __init__(self, wiz: "SetupWizard"):
        super().__init__()
        self.wiz = wiz
        self.setTitle("What are you setting up?")
        self.alert = QRadioButton("Alert sensor — one transmitter + one receiver, "
                                  "Telegram message on motion")
        self.localize = QRadioButton("Localize array — several transmitters and "
                                     "receivers over a hotspot, live floor-plan heatmap")
        self.alert.setChecked(True)
        lay = QVBoxLayout(self)
        lay.addWidget(self.alert)
        lay.addWidget(_hint("The receiver stays plugged into this computer by USB; "
                            "no Wi-Fi setup needed."))
        lay.addSpacing(12)
        lay.addWidget(self.localize)
        lay.addWidget(_hint("Receivers join this computer's hotspot and stream over "
                            "Wi-Fi. Needs the hotspot running (./start.sh)."))
        lay.addStretch(1)

    def validatePage(self) -> bool:  # noqa: N802
        self.wiz.state.mode = MODE_LOCALIZE if self.localize.isChecked() else MODE_ALERT
        return True


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: gray; margin-left: 24px;")
    return label


class EnvironmentPage(QWizardPage):
    def __init__(self, wiz: "SetupWizard"):
        super().__init__()
        self.wiz = wiz
        self.setTitle("Environment")
        self.export_edit = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.export_edit, 1)
        row.addWidget(browse)
        self.channel = QSpinBox()
        self.channel.setRange(1, 13)

        self.hotspot_status = QLabel()
        self.hotspot_status.setWordWrap(True)
        self.detect_btn = QPushButton("Detect from hotspot")
        self.detect_btn.clicked.connect(self.detect)
        self.ssid = QLineEdit()
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.password.setPlaceholderText("blank for an open hotspot")
        self.host_ip = QLineEdit()
        self.port = QSpinBox()
        self.port.setRange(1, 65535)

        form = QFormLayout(self)
        form.addRow("ESP-IDF export.sh:", row)
        form.addRow("Wi-Fi channel:", self.channel)
        form.addRow(self.hotspot_status)
        form.addRow("", self.detect_btn)
        form.addRow("Hotspot SSID:", self.ssid)
        form.addRow("Hotspot password:", self.password)
        form.addRow("Host IP:", self.host_ip)
        form.addRow("Host UDP port:", self.port)
        self._form = form
        self._hotspots: list[fw.HotspotInfo] = []
        for w in (self.export_edit, self.ssid, self.host_ip):
            w.textChanged.connect(self.completeChanged)

    def initializePage(self) -> None:  # noqa: N802
        st = self.wiz.state
        self.export_edit.setText(st.export)
        self.channel.setValue(st.channel)
        self.ssid.setText(st.ssid)
        self.password.setText(st.password)
        self.host_ip.setText(st.host_ip)
        self.port.setValue(st.host_port)
        loc = st.mode == MODE_LOCALIZE
        for w in (self.hotspot_status, self.detect_btn, self.ssid, self.password,
                  self.host_ip, self.port):
            w.setVisible(loc)
            label = self._form.labelForField(w)
            if label is not None:
                label.setVisible(loc)
        self.setSubTitle(
            "Receivers and transmitters must all use the hotspot's channel."
            if loc else "Transmitter and receiver must share a channel. "
                        "11 is a good default.")
        if loc:
            self.detect()

    def detect(self) -> None:
        self._hotspots = self.wiz.detect_hotspots()
        if not self._hotspots:
            self.hotspot_status.setText(
                "⚠ No running hotspot found. Start it (./start.sh), then click "
                "Detect — or fill the values in by hand.")
            return
        hs = self._hotspots[0]
        self.ssid.setText(hs.ssid)
        if hs.ip:
            self.host_ip.setText(hs.ip)
        if 1 <= hs.channel <= 13:
            self.channel.setValue(hs.channel)
        self.hotspot_status.setText(
            f"✓ Hotspot '{hs.connection}' on {hs.iface}: SSID {hs.ssid!r}, "
            f"channel {hs.channel or '?'}, this computer is {hs.ip or '?'}.")

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ESP-IDF export.sh", os.path.expanduser("~/esp"),
            "export.sh (export.sh);;All files (*)")
        if path:
            self.export_edit.setText(path)

    def isComplete(self) -> bool:  # noqa: N802
        if not Path(self.export_edit.text().strip()).expanduser().is_file():
            return False
        if self.wiz.state.mode == MODE_LOCALIZE:
            return bool(self.ssid.text()) and _valid_ipv4(self.host_ip.text().strip())
        return True

    def mismatches(self) -> list[str]:
        if self.wiz.state.mode != MODE_LOCALIZE or not self._hotspots:
            return []
        ips = {h.ip for h in self._hotspots if h.ip}
        out = []
        if ips and self.host_ip.text().strip() not in ips:
            out.append(f"Host IP {self.host_ip.text().strip()} isn't this computer's "
                       f"hotspot address ({', '.join(sorted(ips))}).")
        if self.ssid.text() not in {h.ssid for h in self._hotspots}:
            out.append("The SSID doesn't match the running hotspot.")
        chans = {h.channel for h in self._hotspots if h.channel}
        if chans and self.channel.value() not in chans:
            out.append("The channel doesn't match the running hotspot.")
        return out

    def validatePage(self) -> bool:  # noqa: N802
        problems = self.mismatches()
        if problems and not self.wiz.confirm(
                "Hotspot mismatch",
                "\n\n".join(problems) + "\n\nReceivers flashed with these values "
                "won't reach this computer. Continue anyway?"):
            return False
        st = self.wiz.state
        st.export = self.export_edit.text().strip()
        st.channel = self.channel.value()
        if st.mode == MODE_LOCALIZE:
            st.ssid = self.ssid.text()
            st.password = self.password.text()
            st.host_ip = self.host_ip.text().strip()
            st.host_port = self.port.value()
        else:
            st.ssid, st.password = "", ""
        return True


class FlashPage(QWizardPage):
    """Flash boards for one role and pick which ones the setup uses."""

    def __init__(self, wiz: "SetupWizard", role: str):
        super().__init__()
        self.wiz = wiz
        self.role = role
        self.setTitle("Transmitters" if role == dv.ROLE_TX else "Receivers")

        self.instructions = QLabel()
        self.instructions.setWordWrap(True)
        self.board_combo = QComboBox()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_boards)
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("Desk TX" if role == dv.ROLE_TX else "South RX")
        self.flash_btn = QPushButton(
            "Flash as transmitter" if role == dv.ROLE_TX else "Flash as receiver")
        self.flash_btn.clicked.connect(self.flash)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.wiz.flasher.cancel)
        self.cancel_btn.setEnabled(False)
        self.progress = QProgressBar()
        self.status = QLabel("")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(110)
        self.ready = QListWidget()
        self.ready.itemChanged.connect(lambda *_: self.completeChanged.emit())
        self.ready_note = QLabel("")
        self.ready_note.setWordWrap(True)

        board_row = QHBoxLayout()
        board_row.addWidget(self.board_combo, 1)
        board_row.addWidget(refresh)
        actions = QHBoxLayout()
        actions.addWidget(QLabel("Label:"))
        actions.addWidget(self.label_edit, 1)
        actions.addWidget(self.cancel_btn)
        actions.addWidget(self.flash_btn)
        lay = QVBoxLayout(self)
        lay.addWidget(self.instructions)
        lay.addLayout(board_row)
        lay.addLayout(actions)
        prog = QHBoxLayout()
        prog.addWidget(self.status)
        prog.addWidget(self.progress, 1)
        lay.addLayout(prog)
        lay.addWidget(self.log)
        lay.addWidget(QLabel("Use these in this setup:"))
        lay.addWidget(self.ready, 1)
        lay.addWidget(self.ready_note)

        self._boards: list[dv.AttachedBoard] = []
        self._ports: dict[str, str] = {}      # mac → port for boards flashed here
        self._flashing: tuple[Optional[dv.AttachedBoard], str] = (None, "")
        self._page_id = PAGE_TX if role == dv.ROLE_TX else PAGE_RX
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._auto_refresh)
        # Connected once, for the page's lifetime. QWizard doesn't re-run
        # initializePage when you go Back to a page, so connect-on-enter
        # would leave a revisited page deaf. Slots ignore events unless
        # this page is the current one.
        wiz.flasher.line.connect(self._on_line)
        wiz.flasher.progress.connect(self._on_progress)
        wiz.flasher.flash_finished.connect(self._on_finished)
        wiz.currentIdChanged.connect(self._on_page_changed)

    def _current(self) -> bool:
        return self.wiz.currentId() == self._page_id

    def _on_page_changed(self, page_id: int) -> None:
        if page_id == self._page_id:
            self._timer.start(2000)
        else:
            self._timer.stop()

    # -- lifecycle -------------------------------------------------------

    def initializePage(self) -> None:  # noqa: N802
        st = self.wiz.state
        if self.role == dv.ROLE_TX:
            self.instructions.setText(
                "Plug in a board to use as a transmitter, pick it below, and flash "
                "it. Afterwards it only needs power — any USB charger works. "
                "Already-flashed transmitters are listed at the bottom; tick the "
                "ones to use.")
        elif st.mode == MODE_ALERT:
            names = ", ".join(st.txs.values()) or "(none)"
            self.instructions.setText(
                f"Plug in the receiver — it stays connected to THIS computer, which "
                f"is where motion is detected. It will listen for: {names}.")
        else:
            names = ", ".join(st.txs.values()) or "(none)"
            self.instructions.setText(
                f"Flash each receiver. They join hotspot {st.ssid!r} and only need "
                f"power afterwards. They will listen for: {names}.")
        self._populate_ready()
        self.refresh_boards()

    def cleanupPage(self) -> None:  # noqa: N802 — Back pressed
        # Leaving mid-flash cancels it rather than letting it finish
        # unobserved and half-recorded.
        if self.wiz.flasher.busy:
            self.wiz.flasher.cancel()
        self._flashing = (None, "")

    def _populate_ready(self) -> None:
        chosen = self.wiz.state.txs if self.role == dv.ROLE_TX else self.wiz.state.rxs
        self.ready.blockSignals(True)
        self.ready.clear()
        for rec in self.wiz.registry.all():
            if rec.role == self.role:
                self._add_ready(rec.mac, rec.label, checked=rec.mac in chosen)
        self.ready.blockSignals(False)
        self.completeChanged.emit()

    def _add_ready(self, mac: str, label: str, checked: bool) -> None:
        for i in range(self.ready.count()):
            item = self.ready.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == mac:
                item.setText(f"{label or mac}   ({mac})")
                item.setCheckState(Qt.CheckState.Checked if checked
                                   else item.checkState())
                return
        item = QListWidgetItem(f"{label or mac}   ({mac})")
        item.setData(Qt.ItemDataRole.UserRole, mac)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.ready.addItem(item)

    def checked_macs(self) -> list[str]:
        return [self.ready.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.ready.count())
                if self.ready.item(i).checkState() == Qt.CheckState.Checked]

    # -- boards ----------------------------------------------------------

    def _auto_refresh(self) -> None:
        if not self.wiz.flasher.busy:
            self.refresh_boards()

    def refresh_boards(self) -> None:
        try:
            boards = self.wiz.list_boards()
        except Exception as exc:  # noqa: BLE001
            boards = []
            self.log.appendPlainText(f"could not list serial ports: {exc}")
        current = self.board_combo.currentData()
        key = [(b.port, b.mac, b.chip) for b in boards]
        if key == [(b.port, b.mac, b.chip) for b in self._boards]:
            return
        self._boards = boards
        self.board_combo.blockSignals(True)
        self.board_combo.clear()
        for b in boards:
            rec = self.wiz.registry.get(b.mac) if b.mac else None
            desc = " · ".join(x for x in (
                b.port, b.chip or (rec.chip if rec else ""), b.mac or "MAC unknown",
                (rec.label if rec and rec.label else ""),
                (f"currently {rec.role.upper()}" if rec and rec.role else "")) if x)
            self.board_combo.addItem(desc, b.port)
        idx = self.board_combo.findData(current)
        if idx < 0:
            # Default to a board that isn't already playing the other role
            # in this setup (the TX just flashed is usually still plugged in).
            taken = self._other_role_macs()
            idx = next((i for i, b in enumerate(boards) if b.mac not in taken),
                       0 if boards else -1)
        self.board_combo.setCurrentIndex(idx)
        self.board_combo.blockSignals(False)
        self._sync_buttons()

    def _other_role_macs(self) -> set[str]:
        st = self.wiz.state
        return set(st.rxs) if self.role == dv.ROLE_TX else set(st.txs)

    def _selected_board(self) -> Optional[dv.AttachedBoard]:
        port = self.board_combo.currentData()
        return next((b for b in self._boards if b.port == port), None)

    def _sync_buttons(self) -> None:
        busy = self.wiz.flasher.busy
        self.flash_btn.setEnabled(not busy and self._selected_board() is not None)
        self.cancel_btn.setEnabled(busy)
        # completeChanged makes QWizard refresh its buttons (re-enabling
        # Back), so emit first, then disable Back for the flash.
        self.completeChanged.emit()
        back = self.wiz.button(QWizard.WizardButton.BackButton)
        if back is not None and busy:
            back.setEnabled(False)

    # -- flashing --------------------------------------------------------

    def build_job(self, board: dv.AttachedBoard) -> fw.FlashJob:
        st = self.wiz.state
        if self.role == dv.ROLE_TX:
            config: fw.Config = fw.TxConfig(channel=st.channel)
        else:
            config = fw.RxConfig(channel=st.channel,
                                 tx_macs=list(st.txs)[:fw.MAX_FILTER_MACS])
            if st.mode == MODE_LOCALIZE:
                config.ssid = st.ssid
                config.password = st.password
                config.host_ip = st.host_ip
                config.host_port = st.host_port
        target = dv.chip_to_target(board.chip)
        if target not in fw.TARGETS:
            target = fw.DEFAULT_TARGET
        return fw.FlashJob(port=board.port, config=config, target=target)

    def flash(self) -> None:
        board = self._selected_board()
        if board is None or self.wiz.flasher.busy:
            return
        job = self.build_job(board)
        errors = job.validate()
        if errors:
            QMessageBox.warning(self, "Can't flash", "\n\n".join(errors))
            return
        if board.mac and board.mac in self._other_role_macs():
            other = "receiver" if self.role == dv.ROLE_TX else "transmitter"
            mine = "transmitter" if self.role == dv.ROLE_TX else "receiver"
            if not self.wiz.confirm(
                    "Board already in use",
                    f"{board.port} is the {other} you chose for this setup. "
                    f"Reflash it as a {mine} instead?"):
                return
        if (self.role == dv.ROLE_RX and job.config.uart_only
                and board.vid != dv.ESPRESSIF_VID and job.target in fw.USB_JTAG_TARGETS):
            if not self.wiz.confirm(
                    "Use the native USB port",
                    f"This receiver streams CSI over the chip's native USB port, not "
                    f"the {board.interface} you're flashing through. Afterwards, "
                    f"connect the board's native USB port to this computer.\n\n"
                    f"Flash anyway?"):
                return
        self.log.clear()
        self.progress.setValue(0)
        self.status.setText("Building…")
        self._flashing = (board, self.label_edit.text().strip())
        try:
            self.wiz.flasher.flash(job, Path(self.wiz.state.export).expanduser(),
                                   inferred_mac=board.mac, inferred_chip=board.chip)
        except (OSError, ValueError, RuntimeError) as exc:
            self.status.setText("Can't flash")
            QMessageBox.warning(self, "Can't flash", f"Could not start the flash: {exc}")
            return
        self._sync_buttons()

    def _on_line(self, text: str) -> None:
        if self._current():
            self.log.appendPlainText(text)

    def _on_progress(self, phase: str, pct: int) -> None:
        if self._current():
            self.status.setText("Building…" if phase == "build" else "Flashing…")
            self.progress.setValue(pct)

    def _on_finished(self, res: FlashResult) -> None:
        if not self._current():
            return
        board, label = self._flashing
        self._flashing = (None, "")
        if res.ok:
            self.progress.setValue(100)
            for note in res.notes:
                self.log.appendPlainText(f"note: {note}")
            if res.mac:
                if label:
                    self.wiz.registry.upsert(res.mac, label=label)
                rec = self.wiz.registry.get(res.mac)
                if board is not None:
                    self._ports[res.mac] = board.port
                if self.role == dv.ROLE_RX and self.wiz.state.mode == MODE_ALERT:
                    # One receiver in alert mode: the new one replaces any tick.
                    for i in range(self.ready.count()):
                        self.ready.item(i).setCheckState(Qt.CheckState.Unchecked)
                self._add_ready(res.mac, rec.label if rec else label, checked=True)
                self.status.setText(f"Flashed ✓ {rec.label if rec and rec.label else res.mac}")
                self.label_edit.clear()
            else:
                self.status.setText("Flashed, but the MAC wasn't reported")
                QMessageBox.information(
                    self, "MAC unknown",
                    "The board flashed, but its MAC wasn't reported, so it can't be "
                    "added to the setup. Use Identify in the Devices tab, then tick "
                    "it here.")
        elif res.cancelled:
            self.status.setText("Cancelled")
        else:
            self.status.setText("Flash failed")
            hints = fw.failure_hints(res.output)
            for hint in hints:
                self.log.appendPlainText(f"[hint] {hint}")
            QMessageBox.warning(self, "Flash failed",
                                "\n\n".join(hints) if hints else "See the log for details.")
        self._sync_buttons()
        QTimer.singleShot(1500, self.refresh_boards)

    # -- completion ------------------------------------------------------

    def _attached_port(self, mac: str) -> str:
        if mac in self._ports:
            return self._ports[mac]
        return next((b.port for b in self._boards if b.mac == mac), "")

    def problems(self) -> list[str]:
        macs = self.checked_macs()
        if self.wiz.flasher.busy:
            return ["Wait for the flash to finish."]
        if not macs:
            return [f"Flash or tick at least one "
                    f"{'transmitter' if self.role == dv.ROLE_TX else 'receiver'}."]
        if self.role == dv.ROLE_TX and len(macs) > fw.MAX_FILTER_MACS:
            return [f"A receiver can listen to at most {fw.MAX_FILTER_MACS} "
                    f"transmitters; untick some."]
        if self.role == dv.ROLE_RX and self.wiz.state.mode == MODE_ALERT:
            if len(macs) != 1:
                return ["Alert mode uses exactly one receiver."]
            if not self._attached_port(macs[0]):
                return ["The ticked receiver isn't plugged into this computer."]
        return []

    def isComplete(self) -> bool:  # noqa: N802
        problems = self.problems()
        self.ready_note.setText("" if not problems else "⚠ " + problems[0])
        return not problems

    def validatePage(self) -> bool:  # noqa: N802
        if self.problems():
            return False
        labels = {r.mac: r.label for r in self.wiz.registry.all()}
        if self.role == dv.ROLE_TX:
            self.wiz.state.txs = {m: labels.get(m, "") for m in self.checked_macs()}
        else:
            self.wiz.state.rxs = {m: (labels.get(m, ""), self._attached_port(m))
                                  for m in self.checked_macs()}
        return True

    def nextId(self) -> int:  # noqa: N802
        if self.role == dv.ROLE_TX:
            return PAGE_RX
        return PAGE_TELEGRAM if self.wiz.state.mode == MODE_ALERT else PAGE_LINKS


class TelegramPage(QWizardPage):
    test_result = pyqtSignal(bool, str)

    def __init__(self, wiz: "SetupWizard"):
        super().__init__()
        self.wiz = wiz
        self.setTitle("Telegram alerts")
        self.setSubTitle("Message @BotFather to create a bot and get its token. "
                         "Send your bot any message, then open "
                         "https://api.telegram.org/bot<TOKEN>/getUpdates to find "
                         "your chat ID.")
        self.skip = QCheckBox("Skip — detect motion without sending messages")
        self.skip.toggled.connect(self._sync)
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.chat = QLineEdit()
        self.location = QLineEdit()
        self.location.setPlaceholderText("e.g. Office — prefixed to every message")
        self.path = QLineEdit()
        self.test_btn = QPushButton("Send test message")
        self.test_btn.clicked.connect(self.send_test)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        form = QFormLayout(self)
        form.addRow(self.skip)
        form.addRow("Bot token:", self.token)
        form.addRow("Chat ID:", self.chat)
        form.addRow("Location label:", self.location)
        form.addRow("Save to:", self.path)
        form.addRow("", self.test_btn)
        form.addRow(self.status)
        for w in (self.token, self.chat):
            w.textChanged.connect(self._sync)
        self.test_result.connect(self._on_test_result)

    def initializePage(self) -> None:  # noqa: N802
        path = self.wiz.host_dir / "alert.toml"
        self.path.setText(str(path))
        if path.exists() and not self.token.text():
            try:
                from csidetector.modes.alert.config import load_config
                cfg = load_config(str(path))
                n, a = cfg.get("notifier", {}), cfg.get("alert", {})
                self.token.setText(str(n.get("bot_token", "")))
                self.chat.setText(str(n.get("chat_id", "")))
                self.location.setText(str(a.get("location_label", "")))
                self.status.setText(f"Loaded existing values from {path}.")
            except SystemExit:
                pass
        if not self.location.text():
            rx_labels = [label for label, _ in self.wiz.state.rxs.values() if label]
            self.location.setText(rx_labels[0] if rx_labels else "")
        self._sync()

    def errors(self) -> list[str]:
        if self.skip.isChecked():
            return []
        return pf.validate_telegram(self.token.text(), self.chat.text())

    def _sync(self) -> None:
        on = not self.skip.isChecked()
        for w in (self.token, self.chat, self.location, self.path):
            w.setEnabled(on)
        self.test_btn.setEnabled(on and not self.errors())
        self.completeChanged.emit()

    def isComplete(self) -> bool:  # noqa: N802
        return not self.errors()

    def send_test(self) -> None:
        token, chat = self.token.text().strip(), self.chat.text().strip()
        self.test_btn.setEnabled(False)
        self.status.setText("Sending…")

        def run() -> None:
            from csidetector.modes.alert.notifier import Event
            from csidetector.modes.alert.telegram import TelegramNotifier
            try:
                TelegramNotifier(token, chat).send(
                    Event.now("TEST", "CSIDetector setup: test message ✓"))
                self.test_result.emit(True, "Sent — check your phone.")
            except Exception as exc:  # noqa: BLE001 — surface any failure
                self.test_result.emit(False, f"Failed: {exc}")

        threading.Thread(target=run, daemon=True).start()

    def _on_test_result(self, ok: bool, message: str) -> None:
        self.status.setText(("✓ " if ok else "✗ ") + message)
        self._sync()

    def nextId(self) -> int:  # noqa: N802
        return PAGE_FINISH


class LinksPage(QWizardPage):
    def __init__(self, wiz: "SetupWizard"):
        super().__init__()
        self.wiz = wiz
        self.setTitle("Room layout (links.json)")
        self.setSubTitle("New boards are added to links.json without touching the "
                         "room shape or anything already placed.")
        self.path = QLineEdit()
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        form = QFormLayout(self)
        form.addRow("links.json:", self.path)
        form.addRow(self.summary)

    def initializePage(self) -> None:  # noqa: N802
        self.path.setText(str(self.wiz.host_dir / "links.json"))
        st = self.wiz.state
        tx = ", ".join(label or mac for mac, label in st.txs.items())
        rx = ", ".join(label or mac for mac, (label, _port) in st.rxs.items())
        self.summary.setText(
            f"Transmitters: {tx}\nReceivers: {rx}\n\n"
            "Boards not already in links.json get placeholder spots on the room's "
            "edge (marked \"placed\": false). Measure where they really are and "
            "update their x / y before calibrating.")

    def nextId(self) -> int:  # noqa: N802
        return PAGE_FINISH


class FinishPage(QWizardPage):
    def __init__(self, wiz: "SetupWizard"):
        super().__init__()
        self.wiz = wiz
        self.setTitle("Ready")
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.open_monitor = QCheckBox("Open the live monitor when I click Finish")
        self.open_monitor.setChecked(True)
        lay = QVBoxLayout(self)
        lay.addWidget(self.summary)
        lay.addWidget(self.open_monitor)
        lay.addStretch(1)

    def initializePage(self) -> None:  # noqa: N802
        st = self.wiz.state
        if st.mode == MODE_ALERT:
            mac, (label, port) = next(iter(st.rxs.items()))
            tg = self.wiz.page(PAGE_TELEGRAM)
            lines = [
                f"Receiver: {label or mac} on {port}",
                f"Transmitter(s): "
                f"{', '.join(label or mac for mac, label in st.txs.items())}",
                ("Telegram: messages off" if tg.skip.isChecked()
                 else f"Telegram: write {tg.path.text()}"),
                "",
                "In the monitor: put the transmitter 2–3 m from the receiver, click "
                "Calibrate and leave the room, then click Arm.",
            ]
            self.open_monitor.setVisible(True)
        else:
            lp = self.wiz.page(PAGE_LINKS)
            lines = [
                f"Update {lp.path.text()} with "
                f"{len(st.txs)} transmitter(s) and {len(st.rxs)} receiver(s).",
                "",
                "Next steps:",
                "  1. Measure each new board's position and set its x / y in links.json.",
                "  2. Leave the room and run:",
                "     python run.py localize calibrate-links udp:5566 --out baselines.json",
                "  3. ./start.sh  (hotspot + heatmap)",
            ]
            self.open_monitor.setVisible(False)
        self.summary.setText("\n".join(lines))

    def validatePage(self) -> bool:  # noqa: N802
        try:
            self.wiz.apply()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Couldn't save", str(exc))
            return False
        return True


# --------------------------------------------------------------------------
# Wizard
# --------------------------------------------------------------------------

class SetupWizard(QWizard):
    def __init__(self, registry: dv.Registry,
                 list_boards: Callable[[], list[dv.AttachedBoard]] = dv.list_boards,
                 detect_hotspots: Callable[[], list[fw.HotspotInfo]] = fw.detect_hotspots,
                 idf_export: str = "",
                 host_dir: Optional[Path] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("CSIDetector setup wizard")
        self.resize(760, 640)
        self.registry = registry
        self.list_boards = list_boards
        self.detect_hotspots = detect_hotspots
        self.host_dir = Path(host_dir) if host_dir else default_host_dir()
        self.flasher = Flasher(registry, self)
        self.state = WizardState()
        found = fw.find_idf_export()
        self.state.export = idf_export or (str(found) if found else "")
        # What the caller should do after accept().
        self.open_monitor = False
        self.alert_config_path = ""
        self.written: list[str] = []

        self.setPage(PAGE_MODE, ModePage(self))
        self.setPage(PAGE_ENV, EnvironmentPage(self))
        self.setPage(PAGE_TX, FlashPage(self, dv.ROLE_TX))
        self.setPage(PAGE_RX, FlashPage(self, dv.ROLE_RX))
        self.setPage(PAGE_TELEGRAM, TelegramPage(self))
        self.setPage(PAGE_LINKS, LinksPage(self))
        self.setPage(PAGE_FINISH, FinishPage(self))
        self.setStartId(PAGE_MODE)

    def confirm(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(
            self, title, text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def apply(self) -> None:
        """Write the config files for the chosen mode."""
        st = self.state
        self.written = []
        if st.mode == MODE_ALERT:
            from csidetector.modes.alert.gui import _PersistedState
            tg: TelegramPage = self.page(PAGE_TELEGRAM)
            config_path = ""
            if not tg.skip.isChecked():
                path = Path(tg.path.text().strip()).expanduser()
                pf.write_alert_toml(path, tg.token.text(), tg.chat.text(),
                                    tg.location.text().strip())
                config_path = str(path)
                self.written.append(config_path)
            _mac, (_label, port) = next(iter(st.rxs.items()))
            state = _PersistedState.load()
            state.source = port
            state.alert_config_path = config_path
            state.location = tg.location.text().strip()
            # A new receiver/placement invalidates the old baseline; the
            # monitor won't arm until it's recalibrated.
            state.baseline = 0.0
            state.baseline_ts = 0.0
            state.save()
            self.alert_config_path = config_path
            self.open_monitor = self.page(PAGE_FINISH).open_monitor.isChecked()
        else:
            lp: LinksPage = self.page(PAGE_LINKS)
            path = Path(lp.path.text().strip()).expanduser()
            pf.write_links(path, list(st.txs.items()),
                           [(mac, label) for mac, (label, _port) in st.rxs.items()])
            self.written.append(str(path))
            self.open_monitor = False

    def reject(self) -> None:
        if self.flasher.busy:
            if not self.confirm("Flash in progress",
                                "A flash is still running. Cancel it and close?"):
                return
            self.flasher.shutdown()
        super().reject()
