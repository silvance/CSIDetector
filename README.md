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

- **TX**: ESP32-S3 board (1 minimum; 2+ for localize). The S2 / C3 / C5
  also work in principle but the firmware has only been exercised on S3.
- **RX**: ESP32-S3 board (1 for alert; 4+ for localize).
- **Host**: any computer with Python 3.11+. Localize mode also needs
  a 2.4 GHz Wi-Fi adapter that supports AP mode (built-in card or any
  cheap USB dongle).

### Firmware

Both firmware projects target **ESP-IDF v5.3 or newer** (tested on v5.5).
A helper script drops you into a subshell with IDF activated:

```sh
./idf-env.sh csi_transmitter      # TX side
./idf-env.sh csi_receiver         # RX side
idf.py set-target esp32s3         # one-time per project
idf.py menuconfig                 # Wi-Fi creds, channel, MAC filter
idf.py -p /dev/ttyACM0 flash monitor
```

The TX firmware pins broadcasts to 11n HT20 MCS0 via
`esp_now_set_peer_rate_config` — without this, ESP-NOW falls back to
11b 1 Mbps DSSS (no HT-LTF) and the receiver's CSI engine never fires.
Note the TX MAC: every receiver needs it in its filter.

Required `menuconfig` values on the receiver depend on which workflow
you're running:

| Setting | Localize (multi-RX) | Alert (single-RX) |
|---|---|---|
| `CSI_RX_CHANNEL` | matches TX (e.g. 11) | matches TX (e.g. 11) |
| `CSI_RX_FILTER_TX_MAC` | comma-separated TX MAC list | the one TX's MAC |
| `CSI_RX_WIFI_SSID` | `CSIDetector` (the hotspot) | **leave blank** |
| `CSI_RX_WIFI_PASS` | blank (open hotspot) | n/a |
| `CSI_RX_HOST_IP` | `10.42.0.1` (host on hotspot) | n/a |

Leaving `CSI_RX_WIFI_SSID` blank disables Wi-Fi STA + UDP and falls
back to **UART-only** mode — exactly what alert mode wants. The receiver
then prints CSI rows over USB-CDC and the host reads them directly.

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
