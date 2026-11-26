# Project review, November 2026

A look over the whole project at commit `4a45405` (the base branch after
#49). Five independent passes — the alert pipeline, localize mode and the
shell tooling, the firmware and how it ships, the documents against the
code, and a hands-on walk through the setup app — plus measurements of the
detector's cost. Every finding below says whether it was **confirmed** (by
running something) or **reasoned** (from reading the code).

No code was changed in this review. The test suite passes (332 tests, about
a minute); no secrets are committed in the tree or its history.

## Where the project stands

**Works, and has been seen working on real hardware:** two ESP32-S3 boards
flashed from the setup app on Windows 11 (Python 3.14) — no ESP-IDF, no
other tools — and the alert monitor recording clean CSI from them (about 99
samples a second, no gaps, the transmitter filter applied).

**Works in tests only, never on hardware:** the C3, C5, C6 and classic
ESP32 firmware; a wireless (localize) receiver joining Wi-Fi from settings
written by the app; the Raspberry Pi appliance; macOS.

**Not settled:** how good the detection is. The one set of recordings so
far had the boards in a different place for each take, so "sitting still"
against "walking" and the right thresholds are still open. A matched set
(boards fixed, calibrate, then empty / still / walk / edge) answers it.

## Findings worth fixing

The bar is the project's usual one: a crash, a wrong result, lost data, or
something that would break a demo. Grouped by what you would be doing when
you hit it.

### A. Alert mode on a desktop (the demo path)

1. **Swapping two boards' roles leaves the new receiver listening for
   itself.** Flash A as transmitter and B as receiver, then reflash A as
   receiver and B as transmitter from the Devices tab: A's transmitter
   filter is filled with A's own MAC, with no warning, so it never hears
   B. The table looks right. *Confirmed.* `provision/gui.py`,
   `fill_tx_macs`. Fix: leave the selected board out of the prefill and
   warn when a receiver's filter contains itself.
2. **"Open alert monitor" after setting boards up in the Devices tab reads
   the wrong port and offers no way to change it.** Without a wizard run
   the monitor has no receiver saved and opens the default port, which is
   often the transmitter; calibration then fails with "No CSI data".
   *Confirmed.* `modes/alert/gui.py`. Fix: fall back to the registry's
   newest attached receiver, or say to run the wizard.
3. **One glitch packet still raises a false MOTION on a quiet link.** The
   October fix (#47) covers links with normal noise. Where the room is
   very steady the outlier filter switches itself off (its spread estimate
   is zero on whole-number CSI), and a single low packet reaches a ratio
   of 2.5 to 9. With the monitor's 1.7 threshold that is a false alert for
   any baseline below about 0.6; the documents call 0.05–1.5 typical.
   *Confirmed in simulation on quantised data; not yet seen in a
   recording.* `core/detector.py`, `hampel_filter`. Fix: give the spread
   estimate a small floor instead of skipping; check it against the
   existing recordings.
4. **A garbled serial line stops detection.** If bytes are lost or a log
   line lands inside a CSI line, two numbers can run together
   (`12` + `34` → `1234`); the parser then raises instead of skipping the
   line. The monitor shows an error and stops watching; the Pi service
   exits and is blind for about 20 seconds. *Confirmed.*
   `core/collector.py`, `parse_line`. Fix: three lines, treat it like any
   other unparseable line.
5. **Re-running the wizard silently disarms an armed monitor.** The
   monitor is closed with no message; with "Open the live monitor"
   unticked nothing is watching afterwards. (The backlog has the milder
   recording variant.) *Confirmed.* Fix: ask first.
6. **The Devices tab loses the selected row and a typed label** when the
   2-second refresh runs while a native-USB board is resetting after
   Identify or Flash (3 tries in 8). *Confirmed.* The wizard already
   guards against this; the tab doesn't.
7. **A flash that fails halfway says only "See the log for details."**
   Nothing says the board is now half-written and must be flashed again.
   *Confirmed.*

### B. The Raspberry Pi appliance (headless `alert detect`)

8. **A Pi Zero 2 W cannot keep up with 100 samples a second.** Each sample
   costs about 4 ms of CPU on a 2.1 GHz Xeon: 2.3 ms reading the serial
   port one byte at a time, 1.1–1.9 ms re-filtering the whole half-second
   window. A core 7–10 times slower needs 30–40 ms against a 10 ms budget.
   *Measured here; the Pi figure is an estimate, not a measurement.* A
   Pi 4 or 5 should cope. `core/collector.py` (`readline`),
   `core/detector.py`. Fix: read the port in chunks (15× cheaper), and
   filter only the rows whose neighbours changed — a prototype gives
   identical scores for a quarter of the CPU.
9. **The Pi is less sensitive than the desktop it was tested on.** The
   monitor window enters MOTION at 1.7× and leaves at 1.3×; `alert detect`
   and so the Pi service use 3.0× and 1.5×. A walk reading 2.2× alerts on
   the desktop and is ignored by the Pi. Neither the installer nor its
   guide mentions thresholds. *Confirmed.* Fix: one pair of defaults, and
   the installer writes them where they can be changed.
10. **A transmitter that loses power looks like an empty room.** Headless
    detection logs nothing and sends nothing when CSI stops; only the
    monitor window warns. For an intrusion sensor, unplugging the
    transmitter defeats it silently. *Confirmed.* `modes/alert/detect.py`.
    Fix: log, and optionally alert, after N seconds of silence.
11. **After an internet outage, queued alerts arrive up to an hour late
    and look current.** Retry back-off isn't reset when sending works
    again, and the message carries no time: motion at 1:00 during an
    outage that ends at 2:05 was delivered at 3:03, after a newer alert.
    *Confirmed with a fake clock.* `modes/alert/queue.py`, `detect.py`.
12. **One database error kills alert delivery for good while the service
    looks healthy.** A locked queue database makes the delivery thread
    exit; detection carries on, nothing is sent, systemd sees nothing
    wrong. A corrupt database crash-loops the service at start.
    *Confirmed.* `modes/alert/queue.py`.
13. **A bot token pasted with a trailing space or newline is written to
    the system log and the queue database** in the error text, on every
    retry, forever. *Confirmed with a fake token.* `telegram.py`. The
    setup app's wizard already validates the token; the Pi installer and
    hand-edited files don't.

### C. Firmware

14. **A classic ESP32 flashed as a receiver doesn't work.** Its console
    falls back to 115200 baud while the host opens the port at 921600, so
    the host reads garbage and reports no CSI. *Confirmed by disassembling
    the shipped image.* Even at a matching speed 115200 carries about a
    fifth of the data. Classic ESP32 was added to the supported list in
    #44 without ever running. Fix: a per-chip defaults file setting the
    baud rate, or take the classic ESP32 receiver off the list until one
    has been tried. (As a transmitter it should be fine.)
15. **A receiver joins the hotspot on whatever channel the hotspot uses
    and says nothing when that differs from the transmitters'.** It
    associates, gets an address, and no CSI ever arrives; the log still
    prints the channel from its settings. Windows can't tell the app the
    hotspot's channel, so this is the likely first failure of a localize
    setup there. *Reasoned, about 75% sure.* `csi_receiver/main/main.c`.
    Fix: log an error when the two differ.

### D. Localize mode

The oldest code and the least looked at until now.

16. **The 2.5D room view never paints its heatmap.** The colours don't
    match the number of floor tiles, so every frame is one flat colour.
    *Confirmed (matplotlib 3.11).* `modes/localize/view3d.py`.
17. **The 2.5D view shows people in an empty room.** With every link at
    rest, up to three pins appear and jump around once a second.
    *Confirmed.*
18. **`./start.sh publish` starves the display publisher.** The publisher
    and the heatmap both listen on the same UDP port; only the last one
    started receives anything, so the remote display never leaves its
    start-up state. *Confirmed.* `start.sh`, `core/collector.py`.
19. **A receiver that dies keeps its last score forever.** In the 2.5D
    view its pins stay frozen at full strength; in the heatmap, pressing
    [C] "calibrates" the dead link from stale samples and saves the
    result. *Confirmed.*
20. **With every receiver silent the publisher reports EMPTY.** The remote
    display would go green. *Confirmed.* `modes/localize/publisher.py`.
21. **Without a baselines file the MOTION/EMPTY badge is meaningless**
    (raw values compared against ratio thresholds), and the hint to press
    [C] doesn't work without one. *Confirmed.*
22. **The README's hotspot recipe creates an open network that shares the
    host's internet**, and its next step copies the example file over a
    `links.json` the wizard may already have written. *Confirmed by
    reading.*
23. **On Linux the wizard always tells you to run `start.sh`**, which
    fails when the receivers share a router rather than a NetworkManager
    hotspot. *Confirmed.*

### E. Documents that would mislead

24. **Quickstart, "walking past doesn't trigger MOTION": says to try
    `--enter 4.0`.** Raising the threshold makes it trigger less.
25. **README says the monitor changes "nothing about the alert
    behavior".** It uses different thresholds (finding 9) and ignores
    `cooldown_s`, `clear_on_exit` and `--location`.
26. **A Windows newcomer following README → quickstart dead-ends** at
    shell-only steps (`cp`, `chmod`, `/dev/ttyACM0`, `BASELINE=$(…)`).
    The wizard does get them there, but no document says "on Windows, run
    the wizard, then Calibrate and Arm", or how to come back the next day
    (**Open alert monitor**, **Arm**), or that it only watches while the
    monitor is open and armed.
27. **README alert examples use `/dev/ttyUSB0`**; S3/C3/C5/C6 receivers
    stream only on the native port (`/dev/ttyACM0`).
28. **README: "install the same firmware on every board and pick the
    workflow at the host side".** Transmitter and receiver are different
    images, and the workflow is fixed when the receiver is flashed.
29. **Quickstart says no systemd unit is shipped**; the Pi installer
    ships one. It also says the wizard does "steps 6–7" (it does 6) and
    that calibration counts down (it prints nothing until it ends).
30. **"What is a bad baseline" differs in three places**: below 0.001 or
    above 1.0 (quickstart), 0.05–1.5 typical and above 2 (monitor),
    at most 0.005 or above 2.0 (installer).
31. **Supported chips are listed as equals.** For a demo the documents
    should say: use two ESP32-S3 boards; the others are untested.
32. **README: "~60 tests, <5 seconds"**; it is 332 and about a minute.
    The Docker test image copies only `host/`, so two tests fail in it.

## Below the bar

One line each; none of these loses data or breaks a demo.

- Setup app: no blank-password warning for a localize receiver in the
  Devices tab (the wizard asks); Role + Save lets a board with receiver
  firmware be offered as a transmitter; a board swapped on a bridge port
  between Identify and Flash leaves the table showing the old one; the
  Telegram "Save to" path isn't checked until Finish; the monitor shows
  the last ratio while disarmed or recording; Mark gives no feedback; raw
  exception names in monitor dialogs; the status stays "Identified" after
  a refused flash; the firmware line says "included" when a binary is
  damaged; a 300-character label widens the table; two "Cancel" buttons
  during a wizard flash; a broken existing `alert.toml` is replaced
  without a word.
- Alert pipeline: `clear_on_exit` sends a STILL for every flap, ignoring
  the cooldown; calibration prints `nan` if its first sample is all
  zeros; `--enter` at or below `--exit` is accepted and flaps; sent rows
  are never pruned from the queue; a 429's `retry_after` is ignored; the
  appliance guide's way to change the location label has no effect (the
  service passes `--location`); a capture file whose name starts with
  "com" is opened as a serial port.
- Firmware: a malformed MAC list in menuconfig logs "filter DISABLED" but
  leaves a partial filter; "capture-only mode" after a failed Wi-Fi join
  isn't real; transmitter rates that don't divide 1000 run fast (300 →
  333 Hz); the detector's window is 50 samples, so any rate other than
  100 Hz changes its timing and nothing says so; `c5_display` still
  truncates a 32-byte SSID or 64-character key.
- Localize: one odd-length CSI frame early on kills a link and leaks
  memory; relative `LINKS=` / `BASELINES=` can't work; the hotspot name
  is matched as a pattern; the watchdog unit runs a script from the
  user's checkout as root; hand-edited MACs without colons never match;
  `calibrate-links --settle 0` always fails; `baselines.json` is written
  non-atomically.
- Build: two firmware pushes in quick succession can leave prebuilt
  images built from the older sources.
- Housekeeping: no licence file; the default branch has a working-branch
  name; the merged branch `claude/live-recal-trustworthy` is still there.

## What checked out

- **The settings block.** 3,000,000 hostile blocks through the firmware's
  parser under memory checkers: clean, and a rejected block never changes
  anything. Everything the app will write parses back identically.
- **What gets written to a board.** All 30 shipped files match their
  checksums and are current with the sources; every partition table is
  valid and identical; nothing the app writes overlaps on any chip; the
  largest app uses 40% of its space.
- **Wire formats.** The CSI packet and the display packet match between
  firmware and host byte for byte. A `links.json` written by the wizard
  or the Room layout tab loads in both viewers.
- **The heatmap loop** itself: still → EMPTY, motion → MOTION, silent or
  unknown boards flagged, memory flat over a soak.
- **The alert service's basics.** Unplugging the receiver exits cleanly
  for systemd to restart; memory is flat over 100,000 samples; with a
  clean token nothing leaks; Telegram errors are classified correctly;
  the queue's threading is sound.
- **The setup app's main paths.** The alert wizard end to end; C5/C6
  show the right MAC; unsupported chips are refused by name; cancel and
  quit mid-flash leave nothing behind; a 5-board localize setup gets the
  right settings on every board with the password redacted; missing or
  damaged firmware is reported without a crash. All 39 scripted
  walkthroughs pass.
- **No screen still mentions ESP-IDF builds**, except the version in the
  firmware line.

## First power-up of an untested chip

- Every board: `csi_settings: settings from the setup app for "<label>"`.
  "using the built-in defaults" means the block was rejected.
- Transmitter: `TX up: mac=… ch=… rate=…Hz`, then `heartbeat: sent=N
  dropped=0` every 2 seconds.
- Receiver: `filtering on TX MAC[0] …`, a header line, `CSI capture
  started on channel N`, then `CSI_DATA` lines at about the transmitter's
  rate.
- C5: no abort at `esp_wifi_set_band_mode`; don't assume 128 values per
  line. C6: the sig_mode, mcs and bandwidth columns read 0 by design.
- Hotspot receivers: `STA associated to AP`, then `UDP forwarding to …`
  within 10 seconds. Repeating `reason=201`: network not found;
  `reason=15`: wrong password; associated but nothing arrives: compare
  the hotspot's channel with the transmitters'.
