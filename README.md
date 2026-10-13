# CSIDetector

WiFi Channel State Information (CSI) based motion sensing on ESP32 hardware.
One platform, two workflows:

- **Localize mode** — a small mesh of multiple TX + RX boards over a hotspot,
  with a live floor-plan heatmap and a 2.5D room viewer that pins the
  brightest cell. Best for room-scale motion *localization*.
- **Alert mode** — one TX + one RX, binary motion detection, optional
  Telegram alerts with a durable retry queue. Best for *occupancy /
  tamper notification* with minimal hardware and zero hotspot infrastructure.

Both modes share the same signal-processing core (parsing, filters,
sliding-window σ, hysteresis state machine), so improvements in one
benefit the other.

> **In a hurry?** If you just want the alert-mode demo working in 15
> minutes, follow [`docs/alert-quickstart.md`](docs/alert-quickstart.md).
> It's numbered steps, no narrative, with troubleshooting inline.
>
> **Ready to deploy as an appliance?** Once the desktop demo works, see
> [`docs/appliance-setup.md`](docs/appliance-setup.md) for the
> Pi Zero 2 W + ESP32-S3 single-box setup. One installer script handles
> the whole bootstrap (deps, venv, calibration, systemd unit).

## Which workflow fits me?

| Question | Localize | Alert |
|---|---|---|
| Boards needed | 1+ TX, 4+ RX | 1 TX + 1 RX |
| Network setup | Hotspot + UDP | UART-only — no Wi-Fi infra needed |
| Output | Live heatmap + person pin on a floor plan | `STILL` / `MOTION` events + optional Telegram message |
| Goal | "Where is the person?" | "Did someone enter the room while I was away?" |
| Setup time | 1-2 hours (calibration, geometry) | 5-10 minutes |

You can install the same firmware on every board and pick the workflow
at the host side — nothing about alert mode requires a separate firmware
build.

---

## Common: hardware + firmware

### Hardware

- **TX**: an ESP32-S3, C3, C5, C6 or classic ESP32 board (1 minimum; 2+
  for localize). Most testing so far has been on the S3.
- **RX**: the same chips (1 for alert; 4+ for localize).
- **Host**: any computer with Python 3.11+. Localize mode also needs
  a 2.4 GHz Wi-Fi network the receivers and the host share (a hotspot on
  the host, or a router).

### Firmware

You don't need ESP-IDF or any other tool to set boards up: the setup app
(below) flashes the **prebuilt firmware in `firmware/prebuilt/`** with
esptool, which comes with the Python requirements. It holds the
transmitter and receiver for each supported chip; what differs per board
(label, channel, transmitter MAC filter, hotspot SSID / password / host
IP) is written into a small settings partition on the board at flash
time, so nothing is compiled per board.

The firmware is built from `firmware/csi_transmitter` and
`firmware/csi_receiver` (ESP-IDF v5.5) by `scripts/build_firmware.py`.
The `firmware` GitHub workflow runs it whenever the firmware sources
change and commits the result to `firmware/prebuilt/`.

The TX firmware pins broadcasts to 11n HT20 MCS0 via
`esp_now_set_peer_rate_config` — without this, ESP-NOW falls back to
11b 1 Mbps DSSS (no HT-LTF) and the receiver's CSI engine never fires.
A receiver with no hotspot SSID runs **UART-only** — exactly what alert
mode wants: it prints CSI rows over USB and the host reads them directly.

**Working on the firmware itself** (needs ESP-IDF v5.3+): build and flash
from source as usual (`./idf-env.sh csi_receiver`, then `idf.py
set-target esp32s3`, `idf.py menuconfig`, `idf.py -p /dev/ttyACM0 flash
monitor`). The menuconfig values are only defaults: a board the setup
app has flashed keeps the settings it wrote (`idf.py erase-flash` clears
them). To flash your own build from the setup app instead, run
`python scripts/build_firmware.py --out /some/dir` and start the app with
`CSIDETECTOR_FIRMWARE_DIR=/some/dir`.

### Host setup

```sh
cd host
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`<source>` for both modes is one of:

- `udp:<port>` — multi-RX over the hotspot (localize mode)
- `/dev/ttyUSB0`, `/dev/ttyACM0`, `COM5` — single RX over USB-CDC (alert mode)
- a captured log file — replay
- `-` — stdin

The CLI is the same `python run.py …` you've always used; new subcommands
are added under `localize` and `alert` namespaces and the old top-level
names (`heatmap`, `detect`, …) continue to work as aliases.

### Setup app (flash and configure without menuconfig)

```sh
python run.py setup            # device inventory + one-click flashing
python run.py setup --wizard   # step-by-step: boards → alert or room layout
```

The **Devices & Flashing** tab lists attached boards with their MACs (read
from the USB serial number for native-USB boards; **Identify** runs
esptool for boards behind a USB-UART bridge), keeps a registry of labels
and roles in `~/.csidetector/devices.json`, and flashes a selected board
as a transmitter or receiver. The receiver's TX MAC filter fills in from
registered transmitters. **Detect from hotspot** reads SSID, channel, and
this computer's hotspot IP from NetworkManager, and the app warns before
flashing values that can't work (wrong host IP, channel mismatches, empty
filter).

The **Setup wizard** goes from bare boards to a working sensor: pick
alert or localize, flash transmitters and receivers, then it writes
`alert.toml` (alert mode, owner-only) and opens the live monitor, or
merges the new boards into `links.json` (localize mode) without touching
the room shape or nodes that are already placed.

The **Room layout** tab is a floorplan editor for `links.json`: drag the
boards to where they are (or type x / y), shape the room by dragging,
adding or removing corners, and put a floor-plan image underneath —
click two points a known distance apart to set its scale. It shows the
room exactly as the heatmap will (meters, y up), refuses to save layouts
the heatmap can't use (no receivers, a self-crossing outline), and keeps
a backup of the previous file. After a localize wizard run it opens with
the new boards marked as not yet placed.

Every flash writes the board's chip-specific firmware and its settings
in one esptool run; esptool checks that the board really is the chip the
firmware is for. The **Label** you give a board is stored on it too
(it prints it at boot). Changing a receiver's Wi-Fi name or password
means flashing it again from the app (a few seconds; nothing is
rebuilt). The Wi-Fi password is never saved on this computer: it goes
straight into the board's settings, through a temporary owner-only file
that is deleted as soon as esptool is done.

#### On Windows

1. Install **Python 3.11 or newer** from
   [python.org](https://www.python.org/downloads/) and tick *Add python.exe
   to PATH* in the installer.
2. Get CSIDetector (`git clone`, or download the ZIP and extract it).
3. Double-click **`setup-windows.bat`**. The first run creates a Python
   environment and installs the requirements (esptool included); after
   that it just opens the setup app.

No ESP-IDF, Arduino IDE or other tool is needed. Boards with native USB
(the ESP32-S3/C3/C5/C6 "USB" port) need no driver. Boards
whose USB goes through a CP210x or CH340 chip may need that chip's driver
from Silicon Labs or WCH before they show up.

Alert mode works the same as on Linux. Localize mode needs a Wi-Fi network
that the receivers and this computer share: Windows can't report a hotspot's
name and channel to the app, so you type them in. A router with a fixed
channel works best (Windows' Mobile hotspot picks its own channel, which can
change); the wizard lists this computer's IP addresses to choose from.
There's no `start.sh` on Windows: the wizard's last page gives the
Command Prompt commands for calibration and the heatmap. Allow Python
through Windows Firewall on private networks when asked, or the receivers'
data won't arrive.

### Run the test suite

Locally (uses your venv's matplotlib if installed; skips the matplotlib-
dependent localize smoke test otherwise):

```sh
cd host && python -m pytest tests/ -v
```

Or, hardware-free and reproducible, in a container:

```sh
./scripts/test-in-docker.sh                  # all layers
./scripts/test-in-docker.sh --rebuild        # force a fresh image build
```

The container installs numpy + matplotlib + pytest only — no Qt, no
pyserial, no real Telegram. It exercises:

- core (parsing, filters, motion detection),
- the CLI parser tree (every `--help` variant + back-compat alias dispatch),
- alert-mode config (TOML loader + `build_notifier` factory matrix),
- alert-mode durable queue (happy path, idempotency, transient-and-success,
  permanent dead, restart-survives),
- localize-mode module wiring (figure setup, file format round-trip).

Suite is ~60 tests, runs in <5 seconds.

### GUI walkthroughs (setup app)

`host/tests/gui_driver.py` drives the real setup window and wizard
against simulated boards: a fake esptool reads a model of what's
plugged in, so flashing runs the app's real process handling. Each step
saves a screenshot and a text dump of the screen.

```sh
cd host
python -m tests.gui_driver --list                     # scripted scenarios
python -m tests.gui_driver alert_happy_path --out /tmp/gui-run
# → /tmp/gui-run/alert_happy_path/report.md + screenshots
```

The scenarios also run headlessly in the normal suite
(`tests/test_gui_scenarios.py`). For an exploratory pass, the
`gui-tester` agent (`.claude/agents/gui-tester.md`) runs them, looks at
the screenshots, tries the awkward cases and reports what a user would
hit.

---

## Workflow A — Localized motion detection (multi-RX/TX)

### 1. Set up the hotspot (once)

NetworkManager creates a 2.4 GHz hotspot. Use a USB Wi-Fi dongle so the
host's built-in card is free for internet:

```sh
IFACE=wlp2s0u2   # whatever `nmcli device | grep wifi` shows
nmcli connection add type wifi ifname "$IFACE" con-name CSIDetector autoconnect no \
    ssid CSIDetector \
    802-11-wireless.mode ap 802-11-wireless.band bg 802-11-wireless.channel 11 \
    ipv4.method shared ipv6.method ignore
nmcli connection up CSIDetector

# Open the UDP port in the right firewalld zone (NM puts hotspot ifaces
# in `nm-shared`, not the default zone):
sudo firewall-cmd --zone=nm-shared --add-port=5566/udp --permanent
sudo firewall-cmd --reload
```

The shipped `./start.sh` brings the hotspot up + launches the heatmap
viewer in one command for everyday use.

If your hotspot drops when no clients are connected (some USB Wi-Fi
dongles / power-management configs do this), three quick fixes in
order of preference:

```sh
# Tell NM to bring it back up automatically.
nmcli connection modify CSIDetector connection.autoconnect yes \
                                    connection.autoconnect-retries 0

# Disable Wi-Fi radio power-save on the AP interface.
sudo iw dev wlp2s0u2 set power_save off    # match your iface name

# Last resort: install the watchdog timer that ticks every 30 s and
# revives the hotspot if it's down. See scripts/csidetector-hotspot.{service,timer}
# for the unit files; install with:
#
#   $EDITOR scripts/csidetector-hotspot.service       # set ExecStart= absolute path
#   sudo cp scripts/csidetector-hotspot.{service,timer} /etc/systemd/system/
#   sudo systemctl daemon-reload
#   sudo systemctl enable --now csidetector-hotspot.timer
#   journalctl -u csidetector-hotspot --since "5 min ago"   # verify
```

### 2. Calibrate per-link still-room baselines

Leave the room. The script counts down before recording starts:

```sh
python run.py localize calibrate-links udp:5566 \
    --settle 30 --seconds 30 --out baselines.json
```

`--settle` doubles as the walk-out timer.

### 3. Live floor-plan heatmap + 2.5D viewer

```sh
cp host/links.example.json host/links.json    # edit room polygon, TX/RX MACs, positions

python run.py localize heatmap udp:5566 --links links.json --baselines baselines.json
python run.py localize view3d  udp:5566 --links links.json --baselines baselines.json
```

The heatmap window has a `[C]` key that re-runs calibration in-place
(20 s walk-out + 15 s record by default — adjustable via
`--calibrate-settle` / `--calibrate-record`). New baselines overwrite
the file you passed via `--baselines`.

### 4. Optional: broadcast presence state to a remote display

```sh
python run.py localize publish udp:5566 --links links.json --baselines baselines.json \
    --c5-addr 10.42.0.255    # subnet broadcast on the hotspot
```

The shipped C5 firmware (`firmware/c5_display/`) listens for these and
shows EMPTY / MOTION DETECTED on its LCD.

---

## Workflow B — Single-room alerting (one TX + one RX)

This mode is intentionally minimal: no hotspot, no JSON config files,
no heatmap window. Just a baseline, a detector, and (optionally) a
Telegram bot that pings you when something moves.

### 1. Calibrate the still-room baseline

Power up the TX, plug the RX into the host via USB. Leave the room and:

```sh
BASELINE=$(python run.py alert calibrate /dev/ttyUSB0 --seconds 30 --settle 30)
echo "$BASELINE"   # e.g. 0.183421
```

The baseline is one float — no JSON file needed.

### 2. Detect motion (no notifications)

```sh
python run.py alert detect /dev/ttyUSB0 --baseline "$BASELINE"
```

Prints `STILL` / `MOTION` event lines on every transition. `--enter`
and `--exit` control the hysteresis ratios (defaults 3.0× / 1.5×).

### 3. Detect motion with Telegram alerts

Get a bot token from [@BotFather](https://t.me/BotFather), send your
bot a message, then GET `https://api.telegram.org/bot<token>/getUpdates`
to find your chat id. Drop both into a config file:

```sh
cp host/alert.example.toml host/alert.toml
chmod 600 host/alert.toml      # the file contains a credential
$EDITOR host/alert.toml        # fill in bot_token and chat_id
```

Then run detect with `--alert-config`:

```sh
python run.py alert detect /dev/ttyUSB0 --baseline "$BASELINE" \
    --alert-config alert.toml --location "Office"
```

You'll get a Telegram message on every `STILL → MOTION` transition.
Repeat alerts inside the same activity window are suppressed by a
configurable cooldown (default 60 s — set `--cooldown-s` or
`alert.cooldown_s` in the config). Pass `--clear-on-exit` to also send
a notification on `MOTION → STILL`.

### 3.5. (Optional) GUI instead of CLI

If you'd rather click than type:

```sh
python run.py alert gui --alert-config alert.toml
```

A small Qt window with **Calibrate / Arm / Disarm / Test alert** buttons,
a live ratio gauge, sparkline history, queue size, and the last
alert's timestamp. Reuses the same detect pipeline and Telegram queue;
nothing about the alert behavior changes. Settings (last source,
baseline, thresholds) persist at `~/.csidetector/state.json` so the
window relaunches with sensible defaults.

PyQt6 is already in `requirements.txt`.

### 3.6. Record data for the developers

Improving detection needs real captures, so the monitor has a
**Record data for the developers** panel. For each recording:

1. Pick the **Condition**: empty room, someone sitting still, someone
   walking through, someone walking along the walls, or other. Add a
   **Note** for anything unusual (door open, two people, where the
   boards are).
2. **Start in** gives you time to get into position (leave the room for
   an empty-room recording). **Length** is how long it records; 2
   minutes is plenty.
3. Click **Record**. Click **Mark** whenever something happens, such as
   someone walking in or sitting down. **Stop recording** ends it early
   and keeps what was recorded.

Each recording is one file in `Documents/CSIDetector recordings`, for
example `2026-10-13_213102_walk.csi.gz` (**Open folder** shows it). That
file is what to send: it holds the receiver's raw CSI lines plus what
they were recorded under (condition, note, marks, board labels and
chips, channel, baseline, app version). It holds no Wi-Fi name or
password, Telegram token or location label, and the firmware's other
log lines are left out. A useful set is one recording of each condition
with the boards where they'll be used. When a recording ends, the panel
says how much it got and warns if the stream paused, e.g. because the
transmitter lost power.

Recording uses the receiver's serial port, so it runs while the monitor
is disarmed. A recording also replays anywhere a capture file is
accepted. A file is read instantly, so turn off the start-up settle
time, e.g. `python run.py alert calibrate <file> --settle 0`.

### How offline alerts survive an internet outage

Each alert is enqueued in a SQLite file (default
`~/.csidetector/alert-queue.db`) before being handed to the Telegram
sender. A background worker drains the queue with exponential backoff:

- Transient failure (network down, Telegram 5xx): retry at
  60 s, 2 m, 4 m, 8 m, … up to 1 h, until success.
- Permanent failure (Telegram 4xx — bad token, bad chat id, malformed
  message): mark the row dead and stop retrying.
- Process restart: pending rows are picked up automatically.
- Idempotency: each event has a UUID; re-enqueueing the same id is a
  no-op, so a crash mid-send can't double-deliver.

So if your host loses internet for an hour and you walk past the sensor
twice during that hour, both alerts deliver as soon as the host is
back online — Telegram itself then handles delivery to your phone if
*it* was offline.

Disable the queue (rare; useful only for tests) by setting
`[queue] enabled = false` in `alert.toml`.

---

## Tuning notes

These mostly apply to localize mode but a few help alert mode too.

- **Channel**: pick the 2.4 GHz channel with the least neighbouring
  Wi-Fi traffic. Channel 11 is a reasonable default in North America.
  TX, RX, and the hotspot AP must all be on the same channel.
- **TX placement**: corners are best. A TX in the middle of the room
  makes every link's motion-σ rise on any motion (the long links
  dominate), which kills directional information. Two TXs in
  diagonally-opposite corners give the cleanest two-fan geometry.
- **Sample rate**: 100 Hz per TX is the sweet spot — high enough to
  catch hand motion, low enough that ESP-NOW airtime is comfortable
  with multiple TXs.
- **Antenna**: the WROOM-1U's u.fl connector takes any 2.4 GHz
  antenna (Wi-Fi Pineapple antennas, generic ESP32 dev-kit antennas).
  LoRa antennas (sub-GHz) will not work.
- **RF blast zones**: 3D printers, microwaves, USB3 cables, switching
  power supplies, and routers right next to a receiver will tank its
  packet rate and inflate σ. The heatmap's pkt-rate strip surfaces this
  immediately; in alert mode the RX's noisy σ shows up as constant
  false MOTION until you move the sensor.
- **Metal in the line of sight**: still kills the link. Metal furniture,
  appliances, and structural beams between TX and RX mask motion
  behind them.

## Roadmap

- [x] Single TX / single RX, live waterfall, single-stream binary detector
- [x] Multi-RX over Wi-Fi (UDP forwarding to host), per-link motion-σ heatmap
- [x] Multi-TX support, per-link baselines, 2.5D floor-plan viewer with person pin
- [x] Multi-person separation (top-K local maxima with non-max suppression)
- [x] Aggregator firmware on ESP32-C5-with-screen for an untethered display
- [x] Telegram alerting + durable outbound queue (alert mode)
- [ ] Optional: NBVI subcarrier auto-selection (espectre's MVS)
- [ ] Doppler/phase processing for sub-meter localization

## References

This project leans on prior work; in particular:

- [espressif/esp-csi](https://github.com/espressif/esp-csi) — the
  canonical CSI line format and `wifi_csi_config_t` defaults are taken
  directly from `examples/get-started/csi_recv`.
- [francescopace/espectre](https://github.com/francescopace/espectre) —
  AGC settle wait and Hampel outlier filter come from their MVS
  algorithm.
- [Rui-Chun/ESP32-CSI-Collection-and-Display](https://github.com/Rui-Chun/ESP32-CSI-Collection-and-Display)
  — useful reference for the host-side display loop.
- [euaziel/WiFi-CSI-Human-Pose-Detection](https://github.com/euaziel/WiFi-CSI-Human-Pose-Detection)
  — surveyed for pose-estimation approaches; not adopted because
  full-pose models require multi-antenna NICs (e.g. Intel 5300) that
  ESP32 doesn't expose.
