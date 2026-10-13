"""Scripted walkthroughs of the setup GUI for tests/gui_driver.py.

Each scenario is a function taking a Session. They act like a user,
check the outcomes that must hold (``s.expect``), and leave screenshots
and widget dumps behind for a person or agent to look over.
test_gui_scenarios.py runs them all (fast, no screenshots) so they keep
working as the GUI changes.

Add a scenario: write ``def name(s: Session)`` with a one-line
docstring and add it to SCENARIOS.
"""

from __future__ import annotations

import json

import csidetector.modes.alert.gui as alert_gui
from csidetector.provision import firmware as fw
from tests.gui_driver import Session

TX_MAC = "30:30:f9:aa:00:01"
TX2_MAC = "30:30:f9:aa:00:04"
RX_MAC = "30:30:f9:aa:00:02"
RX2_MAC = "30:30:f9:aa:00:03"
TOKEN = "123456789:" + "A" * 35


def _start_wizard(s: Session, localize: bool = False, password: str = "hunter22") -> None:
    s.click("Setup wizard")
    s.expect(s.wizard is not None and s.wizard.isVisible(), "the wizard opens")
    if localize:
        s.fill("Localize", True)
    s.next()
    if localize:
        s.fill("Hotspot password", password)
    s.expect(s.next(), "Environment → Transmitters")


def _flash_here(s: Session, port: str, label: str = "", role: str = "transmitter") -> None:
    s.choose("Board", port)
    s.fill("Label", label)
    s.click(f"Flash as {role}")
    s.wait_idle()


def _telegram_skip_and_finish(s: Session) -> None:
    s.fill("Skip", True)
    s.expect(s.next(), "Telegram → Finish")
    s.expect(s.finish(), "Finish")


# --------------------------------------------------------------------------
# Wizard: alert mode
# --------------------------------------------------------------------------

def alert_happy_path(s: Session) -> None:
    """First-time alert setup: two native S3 boards, Telegram, monitor opens."""
    s.plug("/dev/ttyACM0", chip="ESP32-S3", mac=TX_MAC)
    s.plug("/dev/ttyACM1", chip="ESP32-S3", mac=RX_MAC)
    _start_wizard(s)
    s.expect(s.page_title() == "Transmitters", "on the transmitter page")
    s.expect(not s.nav_usable("next"), "Next is disabled before anything is flashed")
    _flash_here(s, "/dev/ttyACM0", "Desk TX")
    fwt = s.firmware(TX_MAC)
    s.expect(fwt and fwt["project"] == "csi_transmitter", "the TX got transmitter firmware")
    s.expect(fwt and fwt["target"] == "esp32s3", "built for the S3")
    s.expect(s.next(), "Transmitters → Receivers")
    s.expect("Desk TX" in s.dump(), "the receiver page names the transmitter")
    _flash_here(s, "/dev/ttyACM1", "South RX", "receiver")
    fwr = s.firmware(RX_MAC)
    s.expect(fwr and fwr["project"] == "csi_receiver", "the RX got receiver firmware")
    s.expect(fwr and fwr["config"].get("CONFIG_CSI_RX_FILTER_TX_MAC") == TX_MAC,
             "the receiver filters for the transmitter's MAC")
    s.expect(s.next(), "Receivers → Telegram")
    s.expect("South RX" in s.dump(), "location defaults to the receiver's label")
    s.fill("Bot token", TOKEN)
    s.fill("Chat ID", "8722080863")
    s.click("Send test message")
    s.wait_for(lambda: s.telegram_sent, what="the test message")
    s.snap("after the test message")
    s.expect(s.next(), "Telegram → Finish")
    s.expect("/dev/ttyACM1" in s.dump(), "the summary names the receiver port")
    s.expect(s.finish(), "Finish")
    s.expect(s.wizard is None, "the wizard closed")
    s.expect(s.window._monitor is not None, "the live monitor opened")
    s.expect(alert_gui._PersistedState.load().source == "/dev/ttyACM1",
             "the monitor reads the receiver's port")
    s.focus("monitor")
    s.snap("the live monitor")


def alert_telegram_failure(s: Session) -> None:
    """A rejected bot token shows a clear failure and can still be fixed or skipped."""
    s.plug("/dev/ttyACM0", mac=TX_MAC)
    s.plug("/dev/ttyACM1", mac=RX_MAC)
    _start_wizard(s)
    _flash_here(s, "/dev/ttyACM0", "TX")
    s.next()
    _flash_here(s, "/dev/ttyACM1", "RX", "receiver")
    s.next()
    s.telegram_ok = False
    s.fill("Bot token", "123:short")
    s.snap("malformed token")
    s.expect(not s.nav_usable("next"), "a malformed token blocks Next")
    s.fill("Bot token", TOKEN)
    s.fill("Chat ID", "8722080863")
    s.click("Send test message")
    s.wait_for(lambda: "failed" in s.texts(), what="the failure message")
    s.expect("401" in s.dump() or "unauthorized" in s.texts(),
             "the failure says why")
    _telegram_skip_and_finish(s)


def alert_bridge_receiver(s: Session) -> None:
    """Classic ESP32 on a CP2102 as the receiver: identified, built for esp32, CSI over the bridge."""
    s.plug("/dev/ttyACM0", chip="ESP32-S3", mac=TX_MAC)
    _start_wizard(s)
    _flash_here(s, "/dev/ttyACM0", "Desk TX")
    s.next()
    s.unplug("/dev/ttyACM0")
    s.plug("/dev/ttyUSB0", chip="ESP32-D0WD-V3", mac=RX_MAC, via="cp2102")
    s.click("Refresh")
    _flash_here(s, "/dev/ttyUSB0", "Bench RX", "receiver")
    fwr = s.firmware(RX_MAC)
    s.expect(fwr and fwr["target"] == "esp32", "built for the classic ESP32")
    s.wait_refresh("post-flash board refresh")
    s.expect(s.nav_usable("next"), "the bridge receiver completes the page")
    s.next()
    _telegram_skip_and_finish(s)
    s.expect(alert_gui._PersistedState.load().source == "/dev/ttyUSB0",
             "the monitor reads the bridge port")


def alert_s3_flashed_through_uart_port(s: Session) -> None:
    """S3 receiver flashed via its UART connector must be replugged on its native USB port."""
    s.plug("/dev/ttyACM0", chip="ESP32-S3", mac=TX_MAC)
    _start_wizard(s)
    _flash_here(s, "/dev/ttyACM0", "Desk TX")
    s.next()
    s.plug("/dev/ttyUSB0", chip="ESP32-S3", mac=RX_MAC, via="ch340")
    s.click("Refresh")
    s.answer("Yes")                       # "Use the native USB port … Flash anyway?"
    _flash_here(s, "/dev/ttyUSB0", "Bench RX", "receiver")
    s.expect(any("native" in d.text for d in s.dialogs), "warned about the native port first")
    s.wait_refresh("post-flash board refresh")
    s.expect(not s.nav_usable("next"), "can't continue on the UART port")
    s.expect("native usb port" in s.texts(), "says to plug in the native port")
    s.plug("/dev/ttyACM1", chip="ESP32-S3", mac=RX_MAC)          # native connector
    s.wait_refresh("auto refresh picks up the native port")
    s.expect(s.nav_usable("next"), "the native port completes the page")
    s.next()
    _telegram_skip_and_finish(s)
    s.expect(alert_gui._PersistedState.load().source == "/dev/ttyACM1",
             "the monitor reads the native port")


def alert_receiver_unplugged(s: Session) -> None:
    """Unplugging the flashed receiver blocks Next until it's back."""
    s.plug("/dev/ttyACM0", mac=TX_MAC)
    s.plug("/dev/ttyACM1", mac=RX_MAC)
    _start_wizard(s)
    _flash_here(s, "/dev/ttyACM0", "TX")
    s.next()
    _flash_here(s, "/dev/ttyACM1", "RX", "receiver")
    s.expect(s.nav_usable("next"), "complete after flashing")
    s.unplug("/dev/ttyACM1")
    s.wait_refresh("auto refresh")
    s.expect(not s.nav_usable("next"), "unplugged receiver blocks Next")
    s.expect("isn't plugged in" in s.texts(), "says it isn't plugged in")
    s.plug("/dev/ttyACM1", mac=RX_MAC)
    s.wait_refresh("auto refresh")
    s.expect(s.nav_usable("next"), "plugging it back in completes the page")


def alert_mode_switch_after_flashing(s: Session) -> None:
    """Flash a TX in alert mode, go back and switch to localize on another channel."""
    s.plug("/dev/ttyACM0", mac=TX_MAC)
    s.hotspot(channel=6)
    _start_wizard(s)
    _flash_here(s, "/dev/ttyACM0", "TX")
    s.back()
    s.back()
    s.fill("Localize", True)
    s.next()
    s.fill("Hotspot password", "hunter22")
    s.next()
    s.expect(s.page_title() == "Transmitters", "back on the transmitter page")
    s.expect("channel 11" in s.texts() or not s.nav_usable("next"),
             "the TX flashed for channel 11 can't be reused on the hotspot's channel 6")


# --------------------------------------------------------------------------
# Wizard: localize mode
# --------------------------------------------------------------------------

def localize_happy_path(s: Session) -> None:
    """Hotspot detected, two TXs and two RXs flashed, links.json written."""
    s.hotspot(ssid="CSIDetector", channel=6, ip="10.42.0.1", connection="Hotspot")
    for port, mac in (("/dev/ttyACM0", TX_MAC), ("/dev/ttyACM1", TX2_MAC),
                      ("/dev/ttyACM2", RX_MAC), ("/dev/ttyACM3", RX2_MAC)):
        s.plug(port, mac=mac)
    s.click("Setup wizard")
    s.fill("Localize", True)
    s.next()
    s.expect("10.42.0.1" in s.dump() and "CSIDetector" in s.dump(), "hotspot auto-detected")
    s.fill("Hotspot password", "hunter22")
    s.next()
    _flash_here(s, "/dev/ttyACM0", "North TX")
    _flash_here(s, "/dev/ttyACM1", "South TX")
    s.expect(s.next(), "Transmitters → Receivers")
    _flash_here(s, "/dev/ttyACM2", "East RX", "receiver")
    _flash_here(s, "/dev/ttyACM3", "West RX", "receiver")
    fwr = s.firmware(RX_MAC)
    cfg = fwr["config"] if fwr else {}
    s.expect(cfg.get("CONFIG_CSI_RX_WIFI_SSID") == "CSIDetector", "RX joins the hotspot")
    s.expect(cfg.get("CONFIG_CSI_RX_HOST_IP") == "10.42.0.1", "RX sends to this computer")
    s.expect(cfg.get("CONFIG_CSI_RX_CHANNEL") == "6", "RX on the hotspot's channel")
    heard = {m.strip() for m in cfg.get("CONFIG_CSI_RX_FILTER_TX_MAC", "").split(",")}
    s.expect(heard == {TX_MAC, TX2_MAC},
             "RX hears both transmitters")
    s.expect(cfg.get("CONFIG_CSI_RX_WIFI_PASS") == "<redacted>", "a password was set")
    s.expect(s.next(), "Receivers → links.json")
    s.expect(s.next(), "links.json → Finish")
    if fw.IS_WINDOWS:
        s.expect("cd /d" in s.dump() and "localize heatmap" in s.dump(),
                 "finish steps are Command Prompt commands")
    else:
        s.expect("HOTSPOT_NAME=Hotspot" in s.dump(), "finish steps name the hotspot connection")
    s.expect(s.finish(), "Finish")
    links = json.loads((s.host_dir / "links.json").read_text())
    s.expect(len(links["txs"]) == 2 and len(links["rxs"]) == 2, "links.json has all four")


def localize_no_hotspot(s: Session) -> None:
    """No hotspot running: the wizard says so and wants the values typed in."""
    s.plug("/dev/ttyACM0", mac=TX_MAC)
    s.click("Setup wizard")
    s.fill("Localize", True)
    s.next()
    if fw.IS_WINDOWS:      # nothing to detect with: says what to type in
        s.expect("enter the wi-fi network" in s.texts(), "says to enter the network by hand")
    else:
        s.expect("no running hotspot" in s.texts(), "says no hotspot was found")
    s.fill("Hotspot SSID", "")
    s.expect(not s.nav_usable("next"), "an empty SSID blocks Next")
    s.fill("Hotspot SSID", "MyHotspot")
    s.fill("Host IP", "10.42.0")
    s.expect(not s.nav_usable("next"), "a malformed IP blocks Next")
    s.fill("Host IP", "10.42.0.1")
    s.expect(s.nav_usable("next"), "valid values allow Next")


# --------------------------------------------------------------------------
# Wizard: failures and interruptions
# --------------------------------------------------------------------------

def flash_failures_then_success(s: Session) -> None:
    """No-connect, busy port and build failure each explain themselves; a retry then works."""
    s.plug("/dev/ttyACM0", mac=TX_MAC, behavior="no_connect")
    _start_wizard(s)
    s.choose("Board", "/dev/ttyACM0")
    s.click("Flash as transmitter")
    s.wait_idle()
    s.expect(s.dialogs and "BOOT" in s.dialogs[-1].text, "no-connect hint mentions BOOT")
    s.set_behavior("/dev/ttyACM0", "busy")
    s.dialogs.clear()
    # Identify failed above, so the chip is still unknown: identify runs again.
    s.click("Flash as transmitter")
    s.wait_idle()
    s.expect(s.dialogs and "busy" in s.dialogs[-1].text.lower(), "busy-port hint")
    s.set_behavior("/dev/ttyACM0", "build_fail")
    s.click("Flash as transmitter")
    s.wait_idle()
    s.expect("flash failed" in s.texts(), "status says the flash failed")
    s.expect(not s.nav_usable("next"), "a failed flash doesn't complete the page")
    s.set_behavior("/dev/ttyACM0", "ok")
    s.click("Flash as transmitter")
    s.wait_idle()
    s.expect(s.nav_usable("next"), "the retry completes the page")


def cancel_mid_flash(s: Session) -> None:
    """Cancel a hung flash: navigation locked meanwhile, no esptool left behind."""
    s.plug("/dev/ttyACM0", mac=TX_MAC, behavior="hang")
    _start_wizard(s)
    s.choose("Board", "/dev/ttyACM0")
    s.click("Flash as transmitter")
    s.wait_for(lambda: s.busy() and s.page().status.text().startswith("Flashing"),
               timeout=10, what="the flash to reach the (hung) write")
    s.expect(not s.nav("back").isVisible(), "Back is hidden while flashing")
    s.expect(not s.nav_usable("next"), "Next is disabled while flashing")
    s.click("Cancel", nth=0)              # the page's Cancel, not the wizard's
    s.wait_idle(timeout=10)
    s.expect("cancelled" in s.texts(), "status says cancelled")
    s.expect(s.nav("back").isVisible(), "Back is back")
    s.expect(not s.leftover_processes(), "no esptool process left holding the port")


def close_wizard_while_flashing(s: Session) -> None:
    """Closing the wizard mid-flash asks first; Yes stops the flash cleanly."""
    s.plug("/dev/ttyACM0", mac=TX_MAC, behavior="hang")
    _start_wizard(s)
    s.choose("Board", "/dev/ttyACM0")
    s.click("Flash as transmitter")
    s.wait_for(lambda: s.busy() and s.page().status.text().startswith("Flashing"),
               timeout=10, what="the flash to reach the (hung) write")
    s.answer("No")
    s.cancel_wizard()
    s.expect(s.wizard is not None and s.wizard.isVisible(), "No keeps the wizard open")
    s.expect(s.busy(), "and the flash running")
    s.answer("Yes")
    s.cancel_wizard()
    s.wait_for(lambda: s.wizard is None, what="the wizard to close")
    s.expect(not s.busy(), "the flash stopped")
    s.wait(0.5)
    s.expect(not s.leftover_processes(), "no esptool process left holding the port")


def back_and_forth(s: Session) -> None:
    """Back/Next round trips keep ticks, and unticking sticks."""
    s.plug("/dev/ttyACM0", mac=TX_MAC)
    s.plug("/dev/ttyACM1", mac=RX_MAC)
    _start_wizard(s)
    _flash_here(s, "/dev/ttyACM0", "TX")
    s.next()
    _flash_here(s, "/dev/ttyACM1", "RX", "receiver")
    s.back()
    s.back()
    s.next()
    s.expect("[x] TX" in s.dump(), "TX still ticked after Back/Next")
    s.next()
    s.expect("[x] RX" in s.dump(), "RX still ticked after Back/Next")
    s.tick("RX", False)
    s.back()
    s.next()
    s.expect("[ ] RX" in s.dump(), "the untick stuck")


def esptool_dash_only(s: Session) -> None:
    """A newer esptool without read_mac: identify retries with read-mac."""
    s.esptool_dash_only()
    s.plug("/dev/ttyACM0", mac=TX_MAC)
    _start_wizard(s)
    _flash_here(s, "/dev/ttyACM0", "TX")
    s.expect(s.nav_usable("next"), "flashed after the retry")


# --------------------------------------------------------------------------
# Devices & Flashing tab
# --------------------------------------------------------------------------

def devices_tab_tour(s: Session) -> None:
    """Identify a bridge board, flash it as TX, then flash an RX that hears it."""
    s.plug("/dev/ttyUSB0", chip="ESP32-C3", mac=TX_MAC, via="cp2102")
    s.plug("/dev/ttyACM0", chip="ESP32-S3", mac=RX_MAC)
    s.click("Refresh")
    s.select_row("/dev/ttyUSB0")
    s.click("Identify")
    s.wait_idle()
    s.expect("ESP32-C3" in s.dump(), "identify shows the chip")
    s.fill("Label", "Desk TX")
    s.choose("Role", "TX")
    s.click("Save")
    s.fill("Transmitter (TX)", True)
    s.expect(s.field("Target chip").currentText() == "esp32c3", "target set from the chip")
    s.click("Flash selected board")
    s.wait_idle()
    fwt = s.firmware(TX_MAC)
    s.expect(fwt and fwt["target"] == "esp32c3", "the TX was built for the C3")
    s.select_row("/dev/ttyACM0")
    s.fill("Receiver — alert mode", True)
    s.expect(TX_MAC in s.field("TX MAC filter").text(), "the RX filter is prefilled with the TX")
    s.click("Flash selected board")
    s.wait_idle()
    fwr = s.firmware(RX_MAC)
    s.expect(fwr and fwr["config"].get("CONFIG_CSI_RX_FILTER_TX_MAC") == TX_MAC,
             "the RX hears the TX")


def devices_tab_target_follows_chip(s: Session) -> None:
    """A leftover target doesn't build the wrong firmware: unknown chips are identified first."""
    s.plug("/dev/ttyUSB0", chip="ESP32-C3", mac=TX_MAC, via="cp2102")
    s.plug("/dev/ttyACM0", chip="ESP32-S3", mac=RX_MAC)
    s.click("Refresh")
    s.select_row("/dev/ttyUSB0")
    s.fill("Transmitter (TX)", True)
    s.choose("Target chip", "esp32s3")        # left over from another board
    s.click("Flash selected board")
    s.wait_idle()
    s.wait_idle()                             # identify, then the flash it chains to
    fwt = s.firmware(TX_MAC)
    s.expect(fwt and fwt["target"] == "esp32c3", "built for the identified C3")
    target = s.field("Target chip")
    s.expect(target.currentText() == "esp32c3" and not target.isEnabled(),
             "the target follows the identified chip")
    s.select_row("/dev/ttyACM0")
    s.expect(s.field("Target chip").isEnabled(), "an unidentified board's target is editable")


# --------------------------------------------------------------------------
# Regression walkthroughs for the GUI tester's findings
# --------------------------------------------------------------------------

def _alert_setup(s: Session, telegram: bool = False, open_monitor: bool = True,
                 tx_port: str = "/dev/ttyACM0", rx_port: str = "/dev/ttyACM1") -> None:
    """Plug two native S3 boards and run the alert wizard to Finish."""
    if tx_port not in s.boards:
        s.plug(tx_port, mac=TX_MAC)
    if rx_port not in s.boards:
        s.plug(rx_port, mac=RX_MAC)
    _start_wizard(s)
    _flash_here(s, tx_port, "Desk TX")
    s.next()
    _flash_here(s, rx_port, "South RX", "receiver")
    s.next()
    if telegram:
        s.fill("Bot token", TOKEN)
        s.fill("Chat ID", "8722080863")
    else:
        s.fill("Skip", True)
    s.next()
    s.fill("Open the live monitor", open_monitor)
    s.expect(s.finish(), "Finish")


def _monitor(s: Session):
    s.focus("monitor")
    return s.window._monitor


def _dialog(s: Session, title: str) -> bool:
    return any(d.title == title for d in s.dialogs)


def two_bridge_boards(s: Session) -> None:
    """Two classic ESP32s on CP2102s: the receiver page must not default to (or
    silently reflash) the transmitter just flashed."""
    s.plug("/dev/ttyUSB0", chip="ESP32-D0WD-V3", mac=TX_MAC, via="cp2102")
    s.plug("/dev/ttyUSB1", chip="ESP32-D0WD-V3", mac=RX_MAC, via="cp2102")
    _start_wizard(s)
    _flash_here(s, "/dev/ttyUSB0", "Desk TX")
    s.wait_idle()
    s.expect(s.next(), "Transmitters → Receivers")
    board = s.field("Board")
    s.expect(board.currentData() == "/dev/ttyUSB1", "defaults away from the transmitter")
    s.expect("Desk TX" in board.itemText(board.findData("/dev/ttyUSB0")),
             "the dropdown says which board is the transmitter")
    s.choose("Board", "/dev/ttyUSB0")
    s.answer("No")
    s.click("Flash as receiver")
    s.expect(_dialog(s, "Board already in use"), "asks before reflashing the transmitter")
    s.expect(not s.busy() and "not flashed" in s.texts(), "declining leaves it alone")
    _flash_here(s, "/dev/ttyUSB1", "South RX", "receiver")
    s.wait_idle()
    s.expect(s.firmware(TX_MAC)["project"] == "csi_transmitter", "the TX is still a TX")
    s.wait_refresh()
    s.expect(s.nav_usable("next"), "the bridge receiver completes the page")


def monitor_no_csi(s: Session) -> None:
    """Transmitter off: Calibrate times out with a reason, Disarm stops, the
    monitor says no data is arriving, and closing it lets go of the port."""
    _alert_setup(s)
    mon = _monitor(s)
    s.click("Calibrate")
    s.expect(s.button("Cancel calibration").isEnabled(), "calibration can be cancelled")
    s.wait_for(lambda: _dialog(s, "Calibration failed"), timeout=10,
               what="calibration to give up")
    s.expect(any("No CSI data" in d.text for d in s.dialogs), "says no CSI arrived")
    s.expect(mon._calibrate_btn.text() == "Calibrate", "the button is back")
    # Calibrate with data, then lose it while armed.
    s.csi("/dev/ttyACM1")
    s.click("Calibrate")
    s.wait_for(lambda: _dialog(s, "Calibration complete"), timeout=10,
               what="calibration with data")
    s.click("Arm")
    s.wait_for(lambda: "ARMED" in mon._badge.text(), what="armed and still")
    s.csi("/dev/ttyACM1", on=False)
    s.wait_for(lambda: mon._no_data.isVisible(), timeout=5, what="the no-data warning")
    s.snap("armed, no CSI")
    s.click("Disarm")
    s.wait_for(lambda: mon._detect_thread is None, timeout=3, what="disarm to finish")
    s.click("Calibrate")
    s.click("Cancel calibration")
    s.wait_for(lambda: mon._calib_thread is None, timeout=3, what="cancel to finish")
    s.expect(not _dialog(s, "Calibration failed") or
             sum(d.title == "Calibration failed" for d in s.dialogs) == 1,
             "a cancelled calibration isn't reported as a failure")


def monitor_detects_motion(s: Session) -> None:
    """The demo: calibrate a still room, arm, someone walks in, Telegram buzzes."""
    _alert_setup(s, telegram=True)
    mon = _monitor(s)
    s.csi("/dev/ttyACM1")
    s.click("Calibrate")
    s.wait_for(lambda: _dialog(s, "Calibration complete"), timeout=10, what="calibration")
    s.click("Arm")
    s.wait_for(lambda: "STILL" in mon._badge.text(), what="armed, still")
    s.motion("/dev/ttyACM1")
    s.wait_for(lambda: "MOTION" in mon._badge.text(), timeout=5, what="motion detected")
    s.snap("motion")
    s.wait_for(lambda: s.telegram_sent, timeout=5, what="the Telegram alert")
    s.click("Disarm")
    s.wait_for(lambda: mon._detect_thread is None, timeout=3, what="disarm")


def monitor_receiver_port_moved(s: Session) -> None:
    """Boards replugged in another order: the receiver is ttyACM0 now. The
    monitor finds it by MAC instead of failing on the old port."""
    _alert_setup(s)
    for port in ("/dev/ttyACM0", "/dev/ttyACM1"):
        s.unplug(port)
    s.plug("/dev/ttyACM0", mac=RX_MAC)
    s.csi("/dev/ttyACM0")
    mon = _monitor(s)
    s.click("Calibrate")
    s.wait_for(lambda: _dialog(s, "Calibration complete"), timeout=10,
               what="calibration on the new port")
    s.expect(mon._state.source == "/dev/ttyACM0", "the monitor follows the receiver")


def monitor_reopen_keeps_baseline(s: Session) -> None:
    """The monitor can be reopened from the setup app, and re-running the
    wizard with the same boards keeps the calibration."""
    _alert_setup(s, open_monitor=False)
    s.expect(s.window._monitor is None, "no monitor yet")
    s.click("Open alert monitor")
    s.expect(s.window._monitor is not None, "the button opens it")
    s.csi("/dev/ttyACM1")
    mon = _monitor(s)
    s.click("Calibrate")
    s.wait_for(lambda: _dialog(s, "Calibration complete"), timeout=10, what="calibration")
    baseline = mon._state.baseline
    s.focus("auto")
    s.click("Setup wizard")
    s.next()
    s.next()
    s.tick("Desk TX")
    s.next()
    s.tick("South RX")
    s.next()
    s.fill("Skip", True)
    s.next()
    s.expect("kept" in s.texts(), "the finish page says the calibration is kept")
    s.finish()
    s.expect(alert_gui._PersistedState.load().baseline == baseline, "baseline kept")


def monitor_test_alert(s: Session) -> None:
    """Test alert reports what Telegram actually said, and without Telegram
    points at the wizard, not at command-line flags."""
    _alert_setup(s, telegram=True)
    _monitor(s)
    s.telegram_ok = False
    s.click("Test alert")
    s.wait_for(lambda: _dialog(s, "Test alert failed"), what="the failure")
    s.expect(any("401" in d.text for d in s.dialogs), "shows Telegram's error")
    s.telegram_ok = True
    s.click("Test alert")
    s.wait_for(lambda: _dialog(s, "Test alert sent"), what="the success")
    s.window._monitor._state.alert_config_path = ""
    s.click("Edit config")
    s.click("Test alert")
    texts = [d.text for d in s.dialogs if d.title == "Telegram not set up"]
    s.expect(len(texts) == 2 and all("--alert-config" not in t for t in texts),
             "no-Telegram messages point at the wizard")


def environment_reasons(s: Session) -> None:
    """A disabled Next says why: ESP-IDF path, hotspot fields, short password,
    Telegram token."""
    s.plug("/dev/ttyACM0", mac=TX_MAC)
    s.hotspot()
    s.click("Setup wizard")
    s.fill("Localize", True)
    s.next()
    s.fill(f"ESP-IDF {fw.EXPORT_NAME}", "")
    s.expect(not s.nav_usable("next") and fw.EXPORT_NAME in s.page().note.text(),
             f"says to pick {fw.EXPORT_NAME}")
    s.fill(f"ESP-IDF {fw.EXPORT_NAME}", str(s.export_sh.parent))    # the folder is fine
    s.fill("Hotspot password", "abc12")
    s.expect(not s.nav_usable("next") and "8–64" in s.page().note.text(),
             "a short password is caught here, not at the first receiver")
    s.fill("Hotspot password", "")
    s.answer("No")
    s.next()
    s.expect(_dialog(s, "No hotspot password"), "warns that blank means open network")
    s.expect(s.page_title() == "Environment", "No stays on the page")


def enter_in_label_flashes(s: Session) -> None:
    """Enter in the Label box flashes the board instead of skipping the page."""
    s.plug("/dev/ttyACM0", mac=TX_MAC)
    _start_wizard(s)
    s.fill("Label", "Desk TX")
    s.press("Return", "Label")
    s.expect(s.page_title() == "Transmitters", "still on the transmitter page")
    s.wait_idle()
    s.wait_idle()
    s.expect(s.firmware(TX_MAC), "Enter flashed the board")


def rerun_with_bridge_receiver(s: Session) -> None:
    """Second wizard run with a bridge-board receiver from the first: the page
    says to Identify it, and Identify finds its port."""
    s.plug("/dev/ttyACM0", mac=TX_MAC)
    s.plug("/dev/ttyUSB0", chip="ESP32-D0WD-V3", mac=RX_MAC, via="cp2102")
    _start_wizard(s)
    _flash_here(s, "/dev/ttyACM0", "Desk TX")
    s.next()
    _flash_here(s, "/dev/ttyUSB0", "Bench RX", "receiver")
    s.wait_idle()
    s.next()
    _telegram_skip_and_finish(s)
    _start_wizard(s)
    s.tick("Desk TX")
    s.next()
    s.tick("Bench RX")
    s.expect(not s.nav_usable("next") and "identify" in s.texts(),
             "says to identify the adapter board")
    s.choose("Board", "/dev/ttyUSB0")
    s.click("Identify")
    s.wait_idle()
    s.expect(s.nav_usable("next"), "Identify finds the receiver's port")


def no_boards_plugged_in(s: Session) -> None:
    """Nothing plugged in: Flash is disabled and the list says what to check."""
    _start_wizard(s)
    s.expect(not s.button("Flash as transmitter").isEnabled(), "Flash is disabled")
    s.expect("data usb cable" in s.field("Board").placeholderText().lower(),
             "the empty list says what to check")


def unsupported_chip_status(s: Session) -> None:
    """An ESP32-H2 is refused, and the status says so instead of 'Identifying…'."""
    s.plug("/dev/ttyACM0", chip="ESP32-H2", mac=TX_MAC)
    _start_wizard(s)
    s.click("Flash as transmitter")
    s.wait_idle()
    s.expect(_dialog(s, "Can't flash"), "refused")
    s.expect("identifying" not in s.texts() and "not flashed" in s.texts(),
             "the status says it wasn't flashed")
    s.expect(not s.page().progress.isVisible(), "no stale progress bar")


def broken_links_json(s: Session) -> None:
    """A broken links.json is reported on its page, not as an error at Finish."""
    (s.host_dir / "links.json").write_text('{"room": {"width_m": 5,}}')
    s.plug("/dev/ttyACM0", mac=TX_MAC)
    s.plug("/dev/ttyACM1", mac=RX_MAC)
    s.hotspot()
    _start_wizard(s, localize=True)
    _flash_here(s, "/dev/ttyACM0", "TX")
    s.next()
    _flash_here(s, "/dev/ttyACM1", "RX", "receiver")
    s.next()
    s.expect(not s.nav_usable("next") and "can't be used" in s.texts(),
             "the links page explains the problem")
    s.fill("links.json", str(s.host_dir / "links-new.json"))
    s.expect(s.next(), "another path works")
    s.finish()
    s.expect(_dialog(s, "Saved: next steps"), "the next steps stay on screen after Finish")


def devices_label_saved_on_flash(s: Session) -> None:
    """A label typed in the Devices tab is kept when flashing without Save."""
    s.plug("/dev/ttyACM0", mac=TX_MAC)
    s.click("Refresh")
    s.select_row("/dev/ttyACM0")
    s.fill("Transmitter (TX)", True)
    s.fill("Label", "Kitchen TX")
    s.click("Flash selected board")
    s.wait_idle()
    s.wait_idle()
    s.expect(s.window.registry.get(TX_MAC).label == "Kitchen TX", "label saved")


def devices_identify_then_flash_native(s: Session) -> None:
    """Native board, chip unknown: the flash chained after Identify still runs
    although the board is off the bus for its reset."""
    s.plug("/dev/ttyACM0", mac=TX_MAC)
    s.click("Refresh")
    s.select_row("/dev/ttyACM0")
    s.fill("Transmitter (TX)", True)
    s.click("Flash selected board")
    s.wait_for(lambda: s.firmware(TX_MAC) is not None, timeout=15,
               what="the chained flash")


def devices_swapped_bridge_board(s: Session) -> None:
    """A board swapped on the same adapter port: the wrong-chip failure drops
    the stale identify, and the next Flash identifies and succeeds."""
    s.plug("/dev/ttyUSB0", chip="ESP32-C3", mac=TX_MAC, via="cp2102")
    s.click("Refresh")
    s.select_row("/dev/ttyUSB0")
    s.click("Identify")
    s.wait_idle()
    s.plug("/dev/ttyUSB0", chip="ESP32-S3", mac=TX2_MAC, via="cp2102")   # swapped
    s.fill("Transmitter (TX)", True)
    s.click("Flash selected board")
    s.wait_idle()
    s.expect(_dialog(s, "Flash failed"), "the stale target fails")
    s.wait_refresh()
    s.click("Flash selected board")
    s.wait_idle()
    s.wait_idle()
    s.expect(s.firmware(TX2_MAC) and s.firmware(TX2_MAC)["target"] == "esp32s3",
             "re-identified and flashed")


# --------------------------------------------------------------------------
# Room layout (floorplan editor)
# --------------------------------------------------------------------------

def _canvas(s: Session):
    return s.widget("Room canvas")


def _drag_node(s: Session, mac: str, x: float, y: float) -> None:
    canvas = _canvas(s)
    node = canvas.model.find(mac)
    s.drag("Room canvas", canvas.to_view(node.x, node.y), canvas.to_view(x, y))


def _links(s: Session) -> dict:
    return json.loads((s.host_dir / "links.json").read_text())


def room_place_boards_after_wizard(s: Session) -> None:
    """Localize wizard → the Room layout tab opens with the new boards marked
    unplaced; drag one, type the other's position, save."""
    s.hotspot()
    for port, mac in (("/dev/ttyACM0", TX_MAC), ("/dev/ttyACM1", RX_MAC)):
        s.plug(port, mac=mac)
    _start_wizard(s, localize=True)
    _flash_here(s, "/dev/ttyACM0", "Desk TX")
    s.next()
    _flash_here(s, "/dev/ttyACM1", "Hall RX", "receiver")
    s.next()
    s.next()
    s.finish()
    s.expect(s.window.tabs.currentWidget() is s.window.room, "the Room layout tab opens")
    s.expect("not placed" in s.texts(), "new boards are marked not placed")
    s.snap("room layout after the wizard")
    _drag_node(s, RX_MAC, 1.0, 0.0)
    node = _canvas(s).model.find(RX_MAC)
    s.expect((round(node.x, 2), round(node.y, 2)) == (1.0, 0.0) and node.placed,
             "dragging places the receiver (snapped to 0.1 m)")
    s.edit_cell("Boards", "Desk TX", "X", "4.5")
    s.edit_cell("Boards", "Desk TX", "Y", "3.2")
    s.expect("room layout *" in s.texts(), "the tab shows unsaved changes")
    s.click("Save")
    data = _links(s)
    tx = next(n for n in data["txs"] if n["mac"] == TX_MAC)
    s.expect((tx["x"], tx["y"]) == (4.5, 3.2) and "placed" not in tx,
             "typed position saved, now placed")
    s.expect(not s.window.room.dirty, "saved: no unsaved changes")


def room_draw_the_room(s: Session) -> None:
    """Make a rectangle, add a corner by double-clicking a wall, drag it in
    to make an L; a self-crossing outline can't be saved."""
    s.window.registry.upsert(TX_MAC, role="tx", label="TX")
    s.window.registry.upsert(RX_MAC, role="rx", label="RX")
    s.tab("Room layout")
    s.choose("Board to add", "TX")
    s.click("Add")
    s.choose("Board to add", "RX")
    s.click("Add")
    s.fill("Room width", 6.0)
    s.fill("Room depth", 4.0)
    s.click("Make rectangle")
    canvas = _canvas(s)
    s.click_at("Room canvas", canvas.to_view(6.0, 2.0), double=True)
    s.expect(len(canvas.model.corners) == 5, "double-clicking a wall adds a corner")
    i = canvas.model.corners.index((6.0, 2.0))
    s.drag("Room canvas", canvas.to_view(6.0, 2.0), canvas.to_view(3.0, 2.0))
    s.expect(canvas.model.corners[i] == (3.0, 2.0), "the corner moved")
    s.snap("L-shaped room")
    s.edit_cell("Corners", "3.00", "X", "-2")      # past the far wall: crosses
    s.expect("crosses itself" in s.texts(), "the panel says the outline crosses")
    s.click("Save")
    s.expect(any(d.title == "Can't save yet" and "crosses itself" in d.text
                 for d in s.dialogs), "a crossing outline is refused")
    s.expect(not (s.host_dir / "links.json").exists(), "and nothing is written")


def room_floorplan_image(s: Session) -> None:
    """Put a floor plan under the room, scale it from two clicks, save, reopen."""
    from PyQt6.QtGui import QColor, QImage, QPainter
    img = QImage(800, 600, QImage.Format.Format_RGB32)
    img.fill(QColor("white"))
    painter = QPainter(img)
    painter.drawRect(100, 100, 600, 400)            # a 600 px wall = 6 m below
    painter.end()
    plan = s.host_dir / "plans" / "office.png"
    plan.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(plan))
    s.window.registry.upsert(TX_MAC, role="tx", label="TX")
    s.window.registry.upsert(RX_MAC, role="rx", label="RX")
    s.tab("Room layout")
    for label in ("TX", "RX"):
        s.choose("Board to add", label)
        s.click("Add")
    s.choose_file(str(plan))
    s.click("Floor plan image")
    room = s.window.room
    fp = room.canvas.model.floorplan
    s.expect(fp is not None and fp.image == "plans/office.png",
             "image stored relative to links.json")
    # Two clicks on the plan's wall ends (pixels 100 and 700 on y=500), 6 m apart.
    a, b = fp.to_meters(100, 500), fp.to_meters(700, 500)
    canvas = _canvas(s)
    s.click("Set scale")
    s.click_at("Room canvas", canvas.to_view(*a))
    s.click_at("Room canvas", canvas.to_view(*b))
    s.fill("Distance between the points", 6.0)
    s.click("Apply scale")
    s.expect(abs(fp.px_per_m - 100.0) < 2.0, f"scale ≈ 100 px/m (got {fp.px_per_m:.1f})")
    s.snap("plan scaled")
    s.click("Save")
    s.expect(_links(s)["floorplan"]["image"] == "plans/office.png", "saved with the plan")
    s.click("Reload")
    s.expect(room.canvas.image_pixmap() is not None, "the plan comes back on reload")


def room_unsaved_changes_on_quit(s: Session) -> None:
    """Quitting with unsaved layout edits offers to save them."""
    s.window.registry.upsert(TX_MAC, role="tx", label="TX")
    s.tab("Room layout")
    s.click("Add")
    s.answer("Yes")
    s.window.close()
    s.expect(any(d.title == "Unsaved room layout" for d in s.dialogs), "asked to save")
    s.expect(any(d.title == "Can't save yet" for d in s.dialogs),
             "an incomplete layout (no receiver) isn't saved")
    s.expect(s.window.isVisible(), "so the window stays open")


def room_selection_and_image_move(s: Session) -> None:
    """Review regressions: reload with two boards selected; Move image drags
    only the image and is saved; Remove corner takes the canvas selection."""
    from PyQt6.QtGui import QColor, QImage
    img = QImage(500, 400, QImage.Format.Format_RGB32)
    img.fill(QColor("white"))
    plan = s.host_dir / "plan.png"
    img.save(str(plan))
    s.window.registry.upsert(TX_MAC, role="tx", label="TX")
    s.window.registry.upsert(RX_MAC, role="rx", label="RX")
    s.tab("Room layout")
    for label in ("TX", "RX"):
        s.choose("Board to add", label)
        s.click("Add")
    canvas = _canvas(s)
    s.drag("Room canvas", canvas.to_view(-0.4, -0.4), canvas.to_view(5.4, 4.4))   # rubber band
    s.expect(len([i for i in canvas.scene().selectedItems()
                  if hasattr(i, "node")]) == 2, "both boards selected")
    rx0 = canvas.model.find(RX_MAC)
    # Moves both; snapping to 2.9 gives 29 * 0.1 = 2.9000000000000004, which
    # no longer equals the 2.9 read back from the file.
    s.drag("Room canvas", canvas.to_view(rx0.x, rx0.y), canvas.to_view(2.9, rx0.y))
    s.click("Save")
    s.click("Reload")                    # used to raise from stale selection signals
    s.choose_file(str(plan))
    s.click("Floor plan image")
    rx = canvas.model.find(RX_MAC)
    canvas.node_items[RX_MAC].setSelected(True)
    before = (rx.x, rx.y, rx.placed)
    origin = canvas.model.floorplan.origin_px
    s.fill("Move image", True)
    s.drag("Room canvas", canvas.to_view(4.0, 1.0), canvas.to_view(4.5, 1.0))
    s.expect((rx.x, rx.y, rx.placed) == before, "the selected board didn't move with the image")
    s.expect(s.window.room.dirty, "moving the image is an unsaved change")
    s.click("Save")                      # Move image still ticked
    saved = _links(s)["floorplan"]["origin_px"]
    s.expect(saved[0] != origin[0], "the moved image position is saved")
    s.fill("Move image", False)
    canvas.corner_items[0].setSelected(True)
    first = canvas.model.corners[0]
    s.click("Remove corner")
    s.expect(first not in canvas.model.corners, "removed the corner selected on the plan")
    s.click("Add corner")                          # back to four corners
    canvas.corner_items[0].setSelected(True)       # picked on the plan, then
    s.widget("Corners").selectRow(2)               # a row picked in the table
    third = canvas.model.corners[2]
    s.click("Remove corner")
    s.expect(third not in canvas.model.corners and len(canvas.model.corners) == 3,
             "the latest pick (the table row) is removed")
    # A save that fails mustn't leave the image path rewritten in memory.
    image = canvas.model.floorplan.image
    s.fill("links.json path", str(s.host_dir))     # a folder: the save fails
    s.click("Save")
    s.expect(canvas.model.floorplan.image == image, "image path unchanged by a failed save")


# --------------------------------------------------------------------------
# Review of #36
# --------------------------------------------------------------------------

def identify_keeps_the_chosen_board(s: Session) -> None:
    """Identify resets a native board (off the bus for a moment): the board
    list mustn't fall back to the transmitter and stay there."""
    s.plug("/dev/ttyACM0", mac=TX_MAC)
    s.plug("/dev/ttyACM1", mac=RX_MAC)
    _start_wizard(s)
    _flash_here(s, "/dev/ttyACM0", "Desk TX")
    s.next()
    s.choose("Board", "/dev/ttyACM1")
    s.click("Identify")
    s.wait_idle()
    s.wait_refresh()
    s.expect(s.field("Board").currentData() == "/dev/ttyACM1",
             "still the board the user chose, not the transmitter")


def devices_swapped_board_keeps_its_label(s: Session) -> None:
    """A different board on the same cable doesn't inherit the old label."""
    s.plug("/dev/ttyACM0", mac=TX_MAC)
    s.click("Refresh")
    s.select_row("/dev/ttyACM0")
    s.fill("Label", "Desk TX")
    s.click("Save")
    s.unplug("/dev/ttyACM0")
    s.plug("/dev/ttyACM0", mac=RX_MAC)                # another board, same port
    s.click("Refresh")
    s.select_row("/dev/ttyACM0")
    s.fill("Receiver — alert mode", True)
    s.click("Flash selected board")
    s.wait_idle()
    s.wait_idle()
    rec = s.window.registry.get(RX_MAC)
    s.expect(rec is not None and rec.label != "Desk TX", "the new board isn't 'Desk TX'")
    s.expect(s.window.registry.get(TX_MAC).label == "Desk TX", "the old board keeps it")


def quit_with_the_monitor_armed(s: Session) -> None:
    """Ctrl+C / SIGTERM (app.exit, no close events) with the monitor armed:
    its detect thread must be stopped, or Qt aborts tearing it down."""
    _alert_setup(s)
    mon = _monitor(s)
    s.csi("/dev/ttyACM1")
    s.click("Calibrate")
    s.wait_for(lambda: _dialog(s, "Calibration complete"), timeout=10, what="calibration")
    s.click("Arm")
    s.wait_for(lambda: "ARMED" in mon._badge.text(), what="armed")
    thread = mon._detect_thread
    s.window.shutdown_processes()                    # what aboutToQuit runs
    s.expect(thread is not None and not thread.isRunning(), "the detect thread stopped")


def wizard_swapped_bridge_board(s: Session) -> None:
    """A different board on the same adapter port: the wrong-chip failure
    drops the old mapping and the next Flash identifies and works."""
    s.plug("/dev/ttyUSB0", chip="ESP32-C3", mac=TX_MAC, via="cp2102")
    _start_wizard(s)
    _flash_here(s, "/dev/ttyUSB0", "Desk TX")
    s.wait_idle()
    s.plug("/dev/ttyUSB0", chip="ESP32-D0WD-V3", mac=TX2_MAC, via="cp2102")   # swapped
    s.click("Flash as transmitter")
    s.wait_idle()
    s.expect(_dialog(s, "Flash failed"), "the old chip's build fails")
    s.dialogs.clear()
    s.click("Flash as transmitter")
    s.wait_idle()
    s.wait_idle()
    fwt = s.firmware(TX2_MAC)
    s.expect(fwt and fwt["target"] == "esp32", "identified again and flashed")


SCENARIOS = {f.__name__: f for f in (
    alert_happy_path,
    alert_telegram_failure,
    alert_bridge_receiver,
    alert_s3_flashed_through_uart_port,
    alert_receiver_unplugged,
    alert_mode_switch_after_flashing,
    localize_happy_path,
    localize_no_hotspot,
    flash_failures_then_success,
    cancel_mid_flash,
    close_wizard_while_flashing,
    back_and_forth,
    esptool_dash_only,
    devices_tab_tour,
    devices_tab_target_follows_chip,
    two_bridge_boards,
    monitor_no_csi,
    monitor_detects_motion,
    monitor_receiver_port_moved,
    monitor_reopen_keeps_baseline,
    monitor_test_alert,
    environment_reasons,
    enter_in_label_flashes,
    rerun_with_bridge_receiver,
    no_boards_plugged_in,
    unsupported_chip_status,
    broken_links_json,
    devices_label_saved_on_flash,
    devices_identify_then_flash_native,
    devices_swapped_bridge_board,
    room_place_boards_after_wizard,
    room_draw_the_room,
    room_floorplan_image,
    room_unsaved_changes_on_quit,
    room_selection_and_image_move,
    identify_keeps_the_chosen_board,
    devices_swapped_board_keeps_its_label,
    quit_with_the_monitor_armed,
    wizard_swapped_bridge_board,
)}
