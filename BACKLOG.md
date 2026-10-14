# Backlog

Findings from the review and GUI-test passes on the setup app that fell
below the bar for a fix (a crash, a wrong result, data loss, or something
that would break a demo), plus the gaps we know about. Each entry says
where it came from. Add to it rather than opening another review round.

## Not verified yet

- **The prebuilt firmware on real boards.** CI compiles it; nobody has
  flashed it yet. On the first boards, check: the boot log says
  "settings from the setup app" with the board's label; a receiver
  streams CSI on the channel set in the app; a localize receiver joins
  the Wi-Fi typed into the app; a board flashed earlier with ESP-IDF
  (default partition table) boots after the app reflashes it.
- **ESP32-C5 / C6 CSI.** Their receiver code path (Wi-Fi 6 reception
  metadata, the CSI acquire config) compiles but has never run. Whether
  they produce CSI from the transmitter's HT20 broadcasts, and whether
  the detector copes with their subcarrier layout, is unknown until a
  recording from one is looked at.
- **Windows on real hardware.** CI runs the whole suite on windows-latest,
  but against a fake esptool. Nobody has yet run the setup app on a
  Windows PC with real boards, the USB drivers, the firewall prompt and a
  Mobile hotspot. Things to watch on that first run: COM port names in
  the Devices tab and monitor, esptool's progress output in the log, the
  firewall prompt for UDP 5566, and the Mobile hotspot's channel (it
  picks its own).
- **macOS.** Never run and not in CI. The monitor accepts `/dev/cu.*`
  ports; nothing else has been checked.

## Deferred findings (low severity)

- **Wizard Identify on a slow native board** (reviewer, #38 check).
  After Identify the page re-renders from its cached board list, and
  the board the user picked is remembered by MAC and port. If no board
  was picked by hand and the board stays off USB longer than the
  post-identify re-scan (1.5 s), the dropdown falls back to the first
  board listed, which can be the transmitter. The "already in use"
  confirmation still catches a wrong flash in alert mode.
  `host/csidetector/provision/wizard.py`, `FlashPage.refresh_boards`.
- **Room layout: new boards in a U-shaped room** (layout review). A new
  board is placed at the outline's centroid, which for a concave room
  can be outside it. `warnings()` reports the board as outside; the
  editor doesn't move it. `host/csidetector/provision/layout.py`.
- **Windows: Cancel is a hard kill** (Windows review). Cancel stops
  esptool with `taskkill /F`, like Ctrl+C in a terminal, so a flash
  cancelled mid-write leaves the board needing a reflash. There is no
  graceful stop on Windows.
- **Windows: alert.toml is not 0600** (Windows support). The Telegram
  token file is protected by the user profile's ACL only; NTFS has no
  owner-only mode bit to set.
- **links.json files saved on Windows before the forward-slash fix**
  (Windows CI review) keep a backslash image path and show "Floor plan
  image not found" when opened on Linux. Re-select the image once and
  save.

- **Recording: the wizard can end it unannounced** (recording review).
  Re-running the setup wizard closes an open alert monitor; a recording
  in progress is finished and kept, but the user isn't told.
- **Recording: closing the console window on Windows** (recording
  review) ends the process without a chance to finish the file. The
  file is flushed every 2 s, so it keeps everything but the last
  couple of seconds and has no end summary; it still replays.

- **Pi appliance not tried on a real Pi** (installer check). The
  installer was run end to end in a container (apt and systemd stubbed,
  a fake receiver on a pseudo-terminal): calibration, the service
  command, MOTION/STILL and the queued alert all work. Untested there:
  the apt packages on Raspberry Pi OS, systemd starting it at boot, the
  `dialout` group for the serial port, and a Pi Zero 2 W keeping up with
  100 samples/s.

## Ideas, not started

- **Read a board's settings back on Identify.** esptool `read-flash` of
  the csi_cfg partition would show what a board was set up with (role,
  label, channel, Wi-Fi name) on a computer that didn't flash it.
- **Change Wi-Fi without rewriting the app.** Writing only the 4 KB
  settings block would take a second; today the whole image is flashed
  again (still only seconds, nothing is rebuilt).
- **ESP32-S2** isn't in the prebuilt set (no USB-Serial/JTAG; it was a
  build-from-source target before).

## Test tooling

- **The fake esptool reads the bench once** (GUI tester). It loads
  `boards.json` when it starts, so unplugging a board mid-flash in a
  scenario doesn't fail the flash the way a real unplug would.
  `host/tests/fake_firmware.py`.
- **`Session.tick` misses some checkboxes** (room-tab review). A
  checkbox that isn't the field of a form row can't always be found by
  its label; `fill(name, True)` on the row's label works instead.
