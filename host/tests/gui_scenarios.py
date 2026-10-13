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
    s.wait_for(lambda: s.busy() and s.page().progress.value() >= 10,
               timeout=10, what="the flash to start writing")
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
    s.wait_for(lambda: s.busy() and s.page().progress.value() >= 10,
               timeout=10, what="the flash to start writing")
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
)}
