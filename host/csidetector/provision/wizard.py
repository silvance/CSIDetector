"""Setup wizard: from bare boards to a working sensor.

    Mode → Environment → Transmitters → Receivers → (Telegram | links.json) → Finish

Alert mode ends with alert.toml written and the live monitor opened on
the receiver's port. Localize mode ends with the new boards merged into
links.json, ready for placement and per-link calibration.

Flashing goes through the same Flasher as the Devices tab, so the device
registry records boards identically whichever path flashed them.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import json
import os
import shlex
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QRadioButton, QSpinBox, QVBoxLayout, QWidget,
    QWizard, QWizardPage,
)

from csidetector.provision import devices as dv
from csidetector.provision import firmware as fw
from csidetector.provision import project_files as pf
from csidetector.provision.gui import TIMING, Flasher, FlashResult

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
        self.hotspot_connection = ""                  # NetworkManager name, if detected
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
        lay.addWidget(_hint(
            "Receivers join a Wi-Fi network this computer is on and stream over it. "
            + ("Use your Wi-Fi router, or Windows' Mobile hotspot." if fw.IS_WINDOWS
               else "Needs the hotspot running (./start.sh).")))
        lay.addStretch(1)

    def validatePage(self) -> bool:  # noqa: N802
        self.wiz.state.mode = MODE_LOCALIZE if self.localize.isChecked() else MODE_ALERT
        return True


def _note() -> QLabel:
    """A line saying why Next is disabled (empty when it isn't)."""
    label = QLabel("")
    label.setWordWrap(True)
    return label


def _export_browse_dir() -> str:
    """Where ESP-IDF usually is, to start the file dialog there."""
    for d in ((os.path.expandvars(r"%SystemDrive%\Espressif\frameworks"),)
              if fw.IS_WINDOWS else ()) + (os.path.expanduser("~/esp"),):
        if os.path.isdir(d):
            return d
    return os.path.expanduser("~")


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
        form.addRow(f"ESP-IDF {fw.EXPORT_NAME}:", row)
        form.addRow("Wi-Fi channel:", self.channel)
        form.addRow(self.hotspot_status)
        form.addRow("", self.detect_btn)
        form.addRow("Hotspot SSID:", self.ssid)
        form.addRow("Hotspot password:", self.password)
        form.addRow("Host IP:", self.host_ip)
        form.addRow("Host UDP port:", self.port)
        self.note = _note()
        form.addRow(self.note)
        self._form = form
        self._hotspots: list[fw.HotspotInfo] = []
        for w in (self.export_edit, self.ssid, self.host_ip, self.password):
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
        if not self._hotspots and not fw.IS_WINDOWS:
            self.hotspot_status.setText(
                "⚠ No running hotspot found. Start it (./start.sh), then click "
                "Detect — or fill the values in by hand.")
            return
        if not self._hotspots:
            # Windows has no NetworkManager to ask: help with the host IP and
            # say what the rest must match.
            ips = fw.local_ipv4_addresses()
            if fw.WINDOWS_HOTSPOT_IP in ips:
                self.host_ip.setText(fw.WINDOWS_HOTSPOT_IP)
            elif len(ips) == 1:
                self.host_ip.setText(ips[0])
            self.hotspot_status.setText(
                "Enter the Wi-Fi network the receivers should join: its name, "
                "password and channel. A router with a fixed channel (set in its "
                "settings) works best; Windows' Mobile hotspot picks its own "
                "channel, which can change. This computer's addresses: "
                + (", ".join(ips) or "none found")
                + (f" ({fw.WINDOWS_HOTSPOT_IP} is the Mobile hotspot)."
                   if fw.WINDOWS_HOTSPOT_IP in ips else "."))
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
            self, f"Select ESP-IDF {fw.EXPORT_NAME}", _export_browse_dir(),
            f"{fw.EXPORT_NAME} ({fw.EXPORT_NAME});;All files (*)")
        if path:
            self.export_edit.setText(path)

    def export_path(self) -> Optional[Path]:
        """export.sh from the field; a folder containing one counts too."""
        text = self.export_edit.text().strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if path.is_dir() and (path / fw.EXPORT_NAME).is_file():
            path = path / fw.EXPORT_NAME
        return path if path.is_file() else None

    def problems(self) -> list[str]:
        out = []
        text = self.export_edit.text().strip()
        if self.export_path() is None:
            if not text:
                out.append(f"Select ESP-IDF's {fw.EXPORT_NAME} (usually "
                           f"{fw.EXPORT_EXAMPLE}). If ESP-IDF isn't installed "
                           "yet, see the README's Firmware section.")
            elif Path(text).expanduser().is_dir():
                out.append(f"That folder has no {fw.EXPORT_NAME}. Select the "
                           f"esp-idf folder, or its {fw.EXPORT_NAME}.")
            else:
                out.append(f"No {fw.EXPORT_NAME} at {text}.")
        if self.wiz.state.mode == MODE_LOCALIZE:
            if not self.ssid.text():
                out.append("Enter the hotspot's name (SSID).")
            elif not _valid_ipv4(self.host_ip.text().strip()):
                out.append("Host IP must be an IPv4 address like 10.42.0.1.")
            else:
                cfg = fw.RxConfig(channel=self.channel.value(), ssid=self.ssid.text(),
                                  password=self.password.text(),
                                  host_ip=self.host_ip.text().strip())
                out += cfg.validate()
        return out

    def isComplete(self) -> bool:  # noqa: N802
        problems = self.problems()
        self.note.setText("⚠ " + problems[0] if problems else "")
        return not problems

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
        if (self.wiz.state.mode == MODE_LOCALIZE and not self.password.text()
                and not self.wiz.confirm(
                    "No hotspot password",
                    "The password is blank, so receivers will look for an OPEN "
                    "network. Hotspots and routers normally have a password "
                    + ("(Windows shows the Mobile hotspot's in Settings → Network "
                       "& internet → Mobile hotspot)" if fw.IS_WINDOWS else
                       "(nmcli -s -g 802-11-wireless-security.psk connection show "
                       "<name> shows a NetworkManager one)")
                    + "; with a blank one here, receivers won't join.\n\n"
                    "Continue with an open network?")):
            return False
        st = self.wiz.state
        st.export = str(self.export_path() or self.export_edit.text().strip())
        st.channel = self.channel.value()
        if st.mode == MODE_LOCALIZE:
            st.ssid = self.ssid.text()
            st.password = self.password.text()
            st.host_ip = self.host_ip.text().strip()
            st.host_port = self.port.value()
            st.hotspot_connection = next(
                (h.connection for h in self._hotspots if h.ssid == st.ssid), "")
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
        self.board_combo.setAccessibleName("Board")
        self.board_combo.setPlaceholderText(
            "No ESP32 found. Plug one in with a data USB cable (not charge-only)."
            + (" Boards with a CP210x or CH340 USB chip need its driver."
               if fw.IS_WINDOWS else ""))
        self.board_combo.currentIndexChanged.connect(lambda *_: self._sync_buttons())
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_boards)
        self.identify_btn = QPushButton("Identify")
        self.identify_btn.setToolTip("Read the selected board's chip and MAC "
                                     "(boards on a USB-serial adapter don't report them).")
        self.identify_btn.clicked.connect(lambda: self.identify())
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("Desk TX" if role == dv.ROLE_TX else "South RX")
        # Enter here means "flash", not the wizard's default button (Next),
        # which would skip the page with the board unflashed.
        self.label_edit.installEventFilter(self)
        self.flash_btn = QPushButton(
            "Flash as transmitter" if role == dv.ROLE_TX else "Flash as receiver")
        self.flash_btn.clicked.connect(self.flash)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.wiz.flasher.cancel)
        self.cancel_btn.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setAccessibleName("Flash progress")
        self.progress.setVisible(False)    # until something runs
        self.status = QLabel("")
        self.log = QPlainTextEdit()
        self.log.setAccessibleName("Flash log")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(110)
        self.ready = QListWidget()
        self.ready.setAccessibleName("Use these in this setup")
        self.ready.itemChanged.connect(lambda *_: self.completeChanged.emit())
        self.ready_note = QLabel("")
        self.ready_note.setWordWrap(True)

        board_row = QHBoxLayout()
        board_row.addWidget(self.board_combo, 1)
        board_row.addWidget(refresh)
        board_row.addWidget(self.identify_btn)
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
        self._board_key: list = []
        self._last_process_end = 0.0
        self._flashing: tuple[Optional[dv.AttachedBoard], str] = (None, "")
        # (board, label, flash afterwards?)
        self._identifying: tuple[Optional[dv.AttachedBoard], str, bool] = (None, "", False)
        # Ticks survive Back → Next: initializePage rebuilds the list.
        # None until the page is first left via Back.
        self._remembered: Optional[set[str]] = None
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
        wiz.flasher.identify_finished.connect(self._on_identified)
        wiz.currentIdChanged.connect(self._on_page_changed)

    def _current(self) -> bool:
        return self.wiz.currentId() == self._page_id

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if (obj is self.label_edit and event.type() == QEvent.Type.KeyPress
                and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)):
            if self.flash_btn.isEnabled():
                self.flash()
            return True
        return super().eventFilter(obj, event)

    def _on_page_changed(self, page_id: int) -> None:
        if page_id == self._page_id:
            self._timer.start(TIMING.auto_refresh_ms)
        else:
            self._timer.stop()

    @property
    def _single(self) -> bool:
        """Alert mode: exactly one TX and one RX (the detector treats the
        receiver's samples as a single stream; several TXs would mix)."""
        return self.wiz.state.mode == MODE_ALERT

    # -- lifecycle -------------------------------------------------------

    def initializePage(self) -> None:  # noqa: N802
        st = self.wiz.state
        if self.role == dv.ROLE_TX:
            self.instructions.setText(
                ("Plug in the board to use as the transmitter (alert mode uses "
                 "exactly one), pick it below, and flash it. " if self._single else
                 "Plug in a board to use as a transmitter, pick it below, and flash "
                 "it. ")
                + "Afterwards it only needs power — any USB charger works. "
                "Already-flashed transmitters are listed at the bottom; tick to "
                "reuse one.")
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
        # Navigation is locked while busy, so this is a safety net: never
        # let a flash finish unobserved and half-recorded.
        if self.wiz.flasher.busy:
            self.wiz.flasher.cancel()
        self._flashing = (None, "")
        self._identifying = (None, "", False)
        self._remembered = set(self.checked_macs())

    def _populate_ready(self) -> None:
        # What the user last left ticked here wins: state.txs/rxs can be
        # from an earlier forward pass they've since changed on this page.
        chosen = (self._remembered if self._remembered is not None
                  else set(self.wiz.state.txs if self.role == dv.ROLE_TX
                           else self.wiz.state.rxs))
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
        # Right after a flash/identify a native-USB board is off the bus for
        # its reset; leave the re-scan to the post-process refresh, or the
        # page flashes "isn't plugged in" for a moment.
        settling = (time.monotonic() - self._last_process_end
                    < TIMING.post_process_refresh_ms / 1000)
        if not self.wiz.flasher.busy and not settling:
            self.refresh_boards()

    def refresh_boards(self) -> None:
        try:
            boards = self.wiz.list_boards()
        except Exception as exc:  # noqa: BLE001
            boards = []
            self.log.appendPlainText(f"could not list serial ports: {exc}")
        # A bridge port that's gone (or now shows a MAC) no longer maps.
        bridge = {b.port for b in boards if not b.mac}
        for mac, port in list(self.wiz.bridge_ports.items()):
            if port not in bridge:
                del self.wiz.bridge_ports[mac]
        current = self.board_combo.currentData()
        key = [(b.port, self._board_mac(b), self._board_chip(b), b.vid) for b in boards]
        if key == self._board_key:
            self._sync_buttons()
            return
        self._boards, self._board_key = boards, key
        self.board_combo.blockSignals(True)
        self.board_combo.clear()
        for b in boards:
            mac = self._board_mac(b)
            rec = self.wiz.registry.get(mac) if mac else None
            desc = " · ".join(x for x in (
                b.port, self._board_chip(b), mac or "MAC unknown (click Identify)",
                (rec.label if rec and rec.label else ""),
                (f"currently {rec.role.upper()}" if rec and rec.role else "")) if x)
            self.board_combo.addItem(desc, b.port)
        idx = self.board_combo.findData(current)
        if idx < 0:
            # Default to a board that isn't already playing the other role
            # in this setup (the TX just flashed is usually still plugged in).
            taken = self._other_role_macs()
            idx = next((i for i, b in enumerate(boards)
                        if self._board_mac(b) not in taken), 0 if boards else -1)
        self.board_combo.setCurrentIndex(idx)
        self.board_combo.blockSignals(False)
        self._sync_buttons()

    def _board_mac(self, board: dv.AttachedBoard) -> str:
        """The board's MAC: from its USB serial (native USB), else from an
        Identify or flash of that bridge port in this wizard; "" if unknown."""
        if board.mac:
            return board.mac
        return next((m for m, p in self.wiz.bridge_ports.items() if p == board.port), "")

    def _other_role_macs(self) -> set[str]:
        st = self.wiz.state
        return set(st.rxs) if self.role == dv.ROLE_TX else set(st.txs)

    def _selected_board(self) -> Optional[dv.AttachedBoard]:
        port = self.board_combo.currentData()
        return next((b for b in self._boards if b.port == port), None)

    def _board_chip(self, board: dv.AttachedBoard) -> str:
        """Chip from the live board, else from the registry; "" if unknown.
        list_boards() never knows the chip — it isn't in the USB descriptor."""
        if board.chip:
            return board.chip
        mac = self._board_mac(board)
        rec = self.wiz.registry.get(mac) if mac else None
        return rec.chip if rec else ""

    def _sync_buttons(self) -> None:
        busy = self.wiz.flasher.busy
        selected = self._selected_board() is not None
        self.flash_btn.setEnabled(not busy and selected)
        self.identify_btn.setEnabled(not busy and selected)
        self.cancel_btn.setEnabled(busy)
        self.wiz.set_navigation_locked(busy)
        self.completeChanged.emit()

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
        # No silent default: flashing an S3 build onto a C3 fails, and an
        # unknown chip is identified before we get here.
        return fw.FlashJob(port=board.port, config=config,
                           target=dv.chip_to_target(self._board_chip(board)))

    def flash(self) -> None:
        board = self._selected_board()
        if board is None or self.wiz.flasher.busy:
            return
        label = self.label_edit.text().strip()
        if not self._board_chip(board):
            self._start_identify(board, label, then_flash=True)
        else:
            self._flash_board(board, label)

    def identify(self) -> None:
        """Identify the selected board without flashing it (finds a bridge
        board's MAC, e.g. to reuse a board flashed in an earlier run)."""
        board = self._selected_board()
        if board is not None and not self.wiz.flasher.busy:
            self._start_identify(board, "", then_flash=False)

    def _set_running(self, text: str) -> None:
        self.status.setText(text)
        self.progress.setValue(0)
        self.progress.setVisible(True)

    def _not_flashed(self, text: str) -> None:
        """Clear a stale "Identifying…"/"Building…" after we stop short."""
        self.status.setText(text)
        self.progress.setValue(0)
        self.progress.setVisible(False)

    def _start_identify(self, board: dv.AttachedBoard, label: str, then_flash: bool) -> None:
        self.log.clear()
        self._set_running("Identifying the chip…")
        self._identifying = (board, label, then_flash)
        try:
            self.wiz.flasher.identify(board.port, board.usb_serial,
                                      Path(self.wiz.state.export).expanduser())
        except RuntimeError as exc:
            self._identifying = (None, "", False)
            self._not_flashed("Couldn't identify the chip")
            QMessageBox.warning(self, "Can't identify", str(exc))
            return
        self._sync_buttons()

    def _on_identified(self, res) -> None:
        if not self._current():
            return
        self._last_process_end = time.monotonic()
        board, label, then_flash = self._identifying
        self._identifying = (None, "", False)
        if board is None:
            return
        if res.ok and res.mac and board.vid != dv.ESPRESSIF_VID:
            # Remember which MAC is on this bridge port, for the board list,
            # the in-use check, and finding a ticked board's port.
            for mac, port in list(self.wiz.bridge_ports.items()):
                if port == board.port:
                    del self.wiz.bridge_ports[mac]
            self.wiz.bridge_ports[res.mac] = board.port
        if res.ok and res.chip and then_flash:
            board = dataclasses.replace(board, chip=res.chip)
            self._sync_buttons()
            self._flash_board(board, label)
            return
        if res.ok:
            self._board_key = []          # re-describe the boards
            self.refresh_boards()
            self.status.setText(f"Identified {res.chip or 'board'} {res.mac}")
            self.progress.setValue(100)
            self.completeChanged.emit()
            return
        self._sync_buttons()
        self._not_flashed("Cancelled" if res.cancelled else "Couldn't identify the chip")
        if not res.cancelled:
            hints = fw.failure_hints(res.output)
            for hint in hints:
                self.log.appendPlainText(f"[hint] {hint}")
            QMessageBox.warning(
                self, "Couldn't identify the chip",
                "The wizard needs to know which ESP32 this is to build the right "
                "firmware.\n\n" + ("\n\n".join(hints) if hints else
                                   "See the log for details."))

    def _flash_board(self, board: dv.AttachedBoard, label: str) -> None:
        job = self.build_job(board)
        errors = job.validate()
        if job.target not in fw.TARGETS:
            errors = [f"{self._board_chip(board) or 'This chip'} isn't supported "
                      f"by this firmware (supported: {', '.join(fw.TARGETS)})."]
        if errors:
            self._not_flashed("Not flashed")
            QMessageBox.warning(self, "Can't flash", "\n\n".join(errors))
            return
        mac = self._board_mac(board)
        if mac and mac in self._other_role_macs():
            other = "receiver" if self.role == dv.ROLE_TX else "transmitter"
            mine = "transmitter" if self.role == dv.ROLE_TX else "receiver"
            rec = self.wiz.registry.get(mac)
            name = f"{board.port} ({rec.label})" if rec and rec.label else board.port
            if not self.wiz.confirm(
                    "Board already in use",
                    f"{name} is the {other} you chose for this setup. "
                    f"Reflash it as a {mine} instead?"):
                self._not_flashed("Not flashed")
                return
        if (self.role == dv.ROLE_RX and job.config.uart_only
                and board.vid != dv.ESPRESSIF_VID and job.target in fw.USB_JTAG_TARGETS):
            if not self.wiz.confirm(
                    "Use the native USB port",
                    f"This receiver streams CSI over the chip's native USB port, not "
                    f"the {board.interface} you're flashing through. After flashing, "
                    f"connect the board's native USB port to this computer — the "
                    f"wizard waits for it before continuing.\n\nFlash anyway?"):
                self._not_flashed("Not flashed")
                return
        self.log.clear()
        self._set_running("Building…")
        self._flashing = (board, label)
        try:
            self.wiz.flasher.flash(job, Path(self.wiz.state.export).expanduser(),
                                   inferred_mac=mac, inferred_chip=board.chip)
        except (OSError, ValueError, RuntimeError) as exc:
            self._flashing = (None, "")
            self._not_flashed("Can't flash")
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
        self._last_process_end = time.monotonic()
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
                if board is not None and board.vid != dv.ESPRESSIF_VID:
                    # A bridge's USB serial isn't the MAC, so the live board
                    # list can't map this MAC back to its port.
                    for m, port in list(self.wiz.bridge_ports.items()):
                        if port == board.port:
                            del self.wiz.bridge_ports[m]
                    self.wiz.bridge_ports[res.mac] = board.port
                if self._single:
                    # One per role in alert mode: the new board replaces any tick.
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
            self._not_flashed("Cancelled")
        else:
            self._not_flashed("Flash failed")
            hints = fw.failure_hints(res.output)
            for hint in hints:
                self.log.appendPlainText(f"[hint] {hint}")
            QMessageBox.warning(self, "Flash failed",
                                "\n\n".join(hints) if hints else "See the log for details.")
        self._sync_buttons()
        QTimer.singleShot(TIMING.post_process_refresh_ms, self.refresh_boards)

    # -- completion ------------------------------------------------------

    def _attached_port(self, mac: str) -> str:
        """The port this board is on *now*, preferring its native USB port
        (where an S3/C3/C5 receiver's console — and CSI — goes)."""
        live = [b for b in self._boards if b.mac == mac]
        native = [b for b in live if b.vid == dv.ESPRESSIF_VID]
        if native:
            return native[0].port
        if live:
            return live[0].port
        # A bridge board flashed or identified in this wizard: its MAC isn't
        # in the USB serial, so trust the recorded port only while a
        # MAC-less board is still there.
        port = self.wiz.bridge_ports.get(mac, "")
        if port and any(b.port == port and not b.mac for b in self._boards):
            return port
        return ""

    def _config_problems(self, macs: list[str]) -> list[str]:
        """Boards reused without reflashing must have been flashed for this
        setup's channel / transmitters / mode, or they can't work."""
        st = self.wiz.state
        out = []
        for mac in macs:
            rec = self.wiz.registry.get(mac)
            cfg = rec.config if rec and isinstance(rec.config, dict) else {}
            if not cfg:
                continue            # flashed outside the setup app: unknown
            name = rec.label or mac
            if isinstance(cfg.get("channel"), int) and cfg["channel"] != st.channel:
                out.append(f"{name} was flashed for channel {cfg['channel']}, but this "
                           f"setup uses channel {st.channel}. Reflash it.")
                continue
            if self.role != dv.ROLE_RX:
                continue
            want = "uart" if st.mode == MODE_ALERT else "hotspot"
            if cfg.get("mode") and cfg["mode"] != want:
                kind = "USB-only (alert)" if cfg["mode"] == "uart" else "hotspot (localize)"
                out.append(f"{name} was flashed for {kind} mode. Reflash it.")
            elif set(cfg.get("tx_macs") or []) != set(st.txs):
                out.append(f"{name} listens for different transmitters than the ones "
                           f"chosen for this setup. Reflash it.")
            elif want == "hotspot" and (cfg.get("ssid"), cfg.get("host_ip"),
                                        cfg.get("host_port")) != (st.ssid, st.host_ip,
                                                                  st.host_port):
                out.append(f"{name} was flashed for a different hotspot or host "
                           f"address. Reflash it.")
        return out

    def problems(self) -> list[str]:
        macs = self.checked_macs()
        noun = "transmitter" if self.role == dv.ROLE_TX else "receiver"
        if self.wiz.flasher.busy:
            return ["Wait for the flash to finish."]
        if not macs:
            return [f"Flash or tick at least one {noun}."]
        if self._single and len(macs) != 1:
            return [f"Alert mode uses exactly one {noun}."]
        if self.role == dv.ROLE_TX and len(macs) > fw.MAX_FILTER_MACS:
            return [f"A receiver can listen to at most {fw.MAX_FILTER_MACS} "
                    f"transmitters; untick some."]
        if self.role == dv.ROLE_RX:
            both = [m for m in macs if m in self.wiz.state.txs]
            if both:
                name = self.wiz.state.txs[both[0]] or both[0]
                return [f"{name} is also this setup's transmitter. Flash a different "
                        f"board as the receiver, or go Back and choose another "
                        f"transmitter."]
        if self.role == dv.ROLE_RX and self._single:
            port = self._attached_port(macs[0])
            if not port:
                if any(not self._board_mac(b) for b in self._boards):
                    return ["Can't tell which port the ticked receiver is on: boards "
                            "on a USB-serial adapter don't report their MAC. Select "
                            "its port above and click Identify."]
                return ["The ticked receiver isn't plugged into this computer."]
            board = next(b for b in self._boards if b.port == port)
            rec = self.wiz.registry.get(macs[0])
            target = dv.chip_to_target(board.chip or (rec.chip if rec else ""))
            if board.vid != dv.ESPRESSIF_VID and target in fw.USB_JTAG_TARGETS:
                return [f"This receiver sends CSI over its native USB port, not the "
                        f"{board.interface}. Plug the native port into this computer."]
        return self._config_problems(macs)

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
        self.note = _note()
        form.addRow(self.note)
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
        errors = self.errors()
        if errors and not (self.token.text() or self.chat.text()):
            errors = ["Enter the bot token and chat ID, or tick Skip."]
        self.note.setText("⚠ " + errors[0] if errors else "")
        return not errors

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
                ok, message = True, "Sent — check your phone."
            except Exception as exc:  # noqa: BLE001 — surface any failure
                ok, message = False, f"Failed: {exc}"
            try:
                self.test_result.emit(ok, message)
            except RuntimeError:
                pass                      # the wizard was closed meanwhile

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
        self.path.textChanged.connect(self.completeChanged)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.note = _note()
        form = QFormLayout(self)
        form.addRow("links.json:", self.path)
        form.addRow(self.summary)
        form.addRow(self.note)

    def problems(self) -> list[str]:
        """An existing file must be one we can merge into — say so here,
        not as an error after Finish."""
        text = self.path.text().strip()
        if not text:
            return ["Choose where links.json goes."]
        path = Path(text).expanduser()
        if path.is_dir():
            return ["That's a folder; give the links.json file's path."]
        if path.exists():
            try:
                data = json.loads(path.read_text())
                pf.validate_links_shape(data)
            except (OSError, ValueError) as exc:
                return [f"The existing {path.name} can't be used: {exc}. Fix it, "
                        f"or choose another file."]
        return []

    def isComplete(self) -> bool:  # noqa: N802
        problems = self.problems()
        self.note.setText("⚠ " + problems[0] if problems else "")
        return not problems

    def initializePage(self) -> None:  # noqa: N802
        self.path.setText(str(self.wiz.host_dir / "links.json"))
        st = self.wiz.state
        tx = ", ".join(label or mac for mac, label in st.txs.items())
        rx = ", ".join(label or mac for mac, (label, _port) in st.rxs.items())
        self.summary.setText(
            f"Transmitters: {tx}\nReceivers: {rx}\n\n"
            "Boards not already in links.json get placeholder spots on the room's "
            "edge (marked \"placed\": false). After Finish, the Room layout tab "
            "opens: drag them to where they really are before calibrating.")

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
        self.next_steps: list[str] = []
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
                ("Same boards as your last calibration, so it's kept. Recalibrate "
                 "in the monitor if you moved either board, then click Arm."
                 if self.wiz.keeps_baseline() else
                 "In the monitor: put the transmitter 2–3 m from the receiver, "
                 "click Calibrate and leave the room, then click Arm."),
            ]
            self.open_monitor.setVisible(True)
        else:
            lp = self.wiz.page(PAGE_LINKS)
            links = Path(lp.path.text().strip()).expanduser()
            env = []
            if st.host_port != 5566:
                env.append(f"PORT={st.host_port}")
            if links.resolve() != (self.wiz.host_dir / "links.json").resolve():
                env.append(f"LINKS={shlex.quote(str(links))}")
            if st.hotspot_connection and st.hotspot_connection != "CSIDetector":
                env.append(f"HOTSPOT_NAME={shlex.quote(st.hotspot_connection)}")
            step1 = ("  1. Place each new board in the setup app's Room layout tab "
                     "(opens when you click Finish; dashed markers aren't placed yet).")
            if fw.IS_WINDOWS:
                # No start.sh (bash, NetworkManager) on Windows: the commands
                # it runs, for the Command Prompt.
                host = f'"{self.wiz.host_dir}"'
                py = f'"{sys.executable}"'
                self.next_steps = [
                    "Next steps:", step1,
                    "  2. Make sure the Wi-Fi network is on, leave the room and run "
                    "(in a Command Prompt):",
                    f"     cd /d {host} && {py} run.py localize calibrate-links "
                    f"udp:{st.host_port} --out baselines.json --settle 30",
                    "  3. Then, for the live heatmap:",
                    f"     cd /d {host} && {py} run.py localize heatmap "
                    f"udp:{st.host_port} --links \"{links}\" --baselines baselines.json",
                    "  The first time, Windows asks whether Python may use the "
                    "network: allow it on private networks, or no data arrives.",
                ]
            else:
                host = shlex.quote(str(self.wiz.host_dir))
                start = shlex.quote(str(self.wiz.host_dir.parent / "start.sh"))
                self.next_steps = [
                    "Next steps:", step1,
                    "  2. Leave the room and run:",
                    f"     cd {host} && python run.py localize calibrate-links "
                    f"udp:{st.host_port} --out baselines.json --settle 30",
                    f"  3. {' '.join(env + [start])}   (hotspot + heatmap)",
                ]
            lines = [f"Update {links} with {len(st.txs)} transmitter(s) and "
                     f"{len(st.rxs)} receiver(s).", ""] + self.next_steps
            self.open_monitor.setVisible(False)
        self.summary.setText("\n".join(lines))

    def validatePage(self) -> bool:  # noqa: N802
        try:
            self.wiz.apply()
        except Exception as exc:  # noqa: BLE001 — anything here would abort Qt
            QMessageBox.warning(self, "Couldn't save", f"{type(exc).__name__}: {exc}")
            return False
        if self.wiz.state.mode == MODE_LOCALIZE:
            # The wizard closes on Finish; keep the steps on screen.
            QMessageBox.information(self, "Saved: next steps", "\n".join(self.next_steps))
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
                 parent: Optional[QWidget] = None,
                 before_alert_apply: Optional[Callable[[], None]] = None):
        super().__init__(parent)
        self.setWindowTitle("CSIDetector setup wizard")
        self.resize(760, 640)
        self.registry = registry
        self.list_boards = list_boards
        self.detect_hotspots = detect_hotspots
        self.host_dir = Path(host_dir) if host_dir else default_host_dir()
        # Called before alert mode rewrites the monitor's state file, so an
        # open monitor is closed first and can't save over the new state.
        self.before_alert_apply = before_alert_apply
        self.set_navigation_locked(False)
        self.flasher = Flasher(registry, self)
        self.state = WizardState()
        # mac → port for boards on USB-serial bridges, learned by flashing or
        # identifying them here (their USB serial isn't the MAC). Shared by
        # both flash pages.
        self.bridge_ports: dict[str, str] = {}
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

    _NAV_BUTTONS = [QWizard.WizardButton.Stretch, QWizard.WizardButton.BackButton,
                    QWizard.WizardButton.NextButton, QWizard.WizardButton.FinishButton,
                    QWizard.WizardButton.CancelButton]

    def set_navigation_locked(self, locked: bool) -> None:
        """Hide Back while a flash runs.

        Back runs cleanupPage, which would kill the flash without asking,
        and QWizard re-enables Back whenever a page's completeChanged
        fires, so disabling the button doesn't stick. Removing it from the
        layout does. Next/Finish stay gated by isComplete; Cancel goes
        through reject(), which asks first.
        """
        layout = [b for b in self._NAV_BUTTONS
                  if not (locked and b == QWizard.WizardButton.BackButton)]
        self.setButtonLayout(layout)

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
            # Only now stop an open monitor (it would save its old settings
            # over the new state). Not before: a failed alert.toml write
            # shouldn't leave detection silently off.
            if self.before_alert_apply is not None:
                self.before_alert_apply()
            rx_mac, (_label, port) = next(iter(st.rxs.items()))
            tx_mac = next(iter(st.txs), "")
            state = _PersistedState.load()
            if not self.keeps_baseline(state):
                # Different boards: the old baseline doesn't apply, and the
                # monitor won't arm until it's recalibrated.
                state.baseline = 0.0
                state.baseline_ts = 0.0
            state.source = dv.stable_port(port)
            state.receiver_mac = rx_mac
            state.transmitter_mac = tx_mac
            state.alert_config_path = config_path
            state.location = tg.location.text().strip()
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

    def keeps_baseline(self, state=None) -> bool:
        """Alert mode with the same transmitter and receiver as the saved
        calibration: keep it rather than forcing a new one."""
        st = self.state
        if st.mode != MODE_ALERT or not st.rxs or not st.txs:
            return False
        if state is None:
            from csidetector.modes.alert.gui import _PersistedState
            state = _PersistedState.load()
        return (state.baseline > 0 and state.receiver_mac == next(iter(st.rxs))
                and state.transmitter_mac == next(iter(st.txs)))

    def reject(self) -> None:
        if self.flasher.busy:
            if not self.confirm("Flash in progress",
                                "A flash is still running. Cancel it and close?"):
                return
            self.flasher.shutdown()
        super().reject()
