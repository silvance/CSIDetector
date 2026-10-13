# Backlog

Findings from the review and GUI-test passes on the setup app that fell
below the bar for a fix (a crash, a wrong result, data loss, or something
that would break a demo), plus the gaps we know about. Each entry says
where it came from. Add to it rather than opening another review round.

## Not verified yet

- **Windows on real hardware.** CI runs the whole suite on windows-latest,
  but against fake idf.py/esptool. Nobody has yet run the setup app on a
  Windows PC with a real ESP-IDF install, the USB drivers, the firewall
  prompt and a Mobile hotspot. Things to watch on that first run: the
  export.bat capture with the installer layout (`C:\Espressif`), which
  Python the IDF virtualenv was made for, COM port names in the Devices
  tab and monitor, the firewall prompt for UDP 5566, and the Mobile
  hotspot's channel (it picks its own).
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
- **Windows: Cancel is a hard kill** (Windows review). Cancel stops the
  idf.py tree with `taskkill /F`, like Ctrl+C in a terminal, so a flash
  cancelled mid-write leaves the board needing a reflash. There is no
  graceful stop on Windows.
- **Windows: alert.toml is not 0600** (Windows support). The Telegram
  token file is protected by the user profile's ACL only; NTFS has no
  owner-only mode bit to set.
- **Windows: user names with spaces** (Windows support). The build
  directory lives under `%LOCALAPPDATA%` and uses the 8.3 short name
  when the path has spaces. On a volume with short names disabled,
  flashing refuses with the "path with spaces" message and asks for
  `CSIDETECTOR_BUILD_DIR`.
- **Windows: export.bat writes into the app's working directory**
  (Windows CI review). ESP-IDF's export.bat (v5.2) writes an
  `UNINSTALL_OUTPUT` file in the current directory, which is `host\`
  when started from setup-windows.bat. A read-only folder only adds an
  error line to the flash log.
- **links.json files saved on Windows before the forward-slash fix**
  (Windows CI review) keep a backslash image path and show "Floor plan
  image not found" when opened on Linux. Re-select the image once and
  save.

## Test tooling

- **The fake idf.py reads the bench once** (GUI tester). It loads
  `boards.json` when it starts, so unplugging a board mid-flash in a
  scenario doesn't fail the flash the way a real unplug would.
  `host/tests/gui_driver.py`, `_FAKE_IDF`.
- **`Session.tick` misses some checkboxes** (room-tab review). A
  checkbox that isn't the field of a form row can't always be found by
  its label; `fill(name, True)` on the row's label works instead.
