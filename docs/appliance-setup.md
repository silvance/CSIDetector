# Pi appliance setup

End state: a single small box that you plug into a USB-C wall wart, and
get a Telegram message when somebody walks past it. Inside the box is a
Raspberry Pi Zero 2 W and a tiny ESP32-S3 (SuperMini-class), wired
together over USB. No PC needed at runtime.

This guide assumes you already have **alert mode working on a desktop**
([alert-quickstart.md](alert-quickstart.md)). If you haven't done the
desktop walkthrough yet, do that first — it confirms the firmware,
calibration, and Telegram credentials are all good before you bake
them into an always-on appliance.

## Hardware

- **Raspberry Pi Zero 2 W** (Pi 3 / 4 / 5 also work but are overkill).
  The original Pi Zero W is too slow for numpy + Python 3.11 — skip it.
- **microSD card**, 16 GB+ (Class 10 / A1 or better)
- **ESP32-S3 board** (any flavor; SuperMini variants are tiny and
  match the Pi Zero's footprint)
- **USB-C PD splitter** if you want to power Pi + ESP32 from one
  outlet, or two USB-C cables and a multi-port wall wart
- A short USB cable to connect ESP32 → Pi
- 3D-printed enclosure (optional; STL TBD)

## One-time setup

### 1. Flash Pi OS Lite (10 min)

1. Install Raspberry Pi Imager: <https://www.raspberrypi.com/software/>
2. Pick **Raspberry Pi OS Lite (64-bit)** for the Pi Zero 2 W.
3. Click the gear icon **before flashing**:
   - Set a hostname (e.g. `csidetector-01`)
   - **Enable SSH** with a password or your public key
   - **Set Wi-Fi credentials** (the home network the Pi will phone home
     from, *not* the CSIDetector hotspot)
   - Set username (`pi` is conventional; the install script honors
     whatever you pick)
4. Flash. Unmount when done.

### 2. First boot + SSH (2 min)

```sh
# Pop the card into the Pi, plug in power.
# After ~60s, find it:
ping csidetector-01.local       # if mDNS works
# or check your router's DHCP lease table

ssh pi@csidetector-01.local
```

### 3. Clone + install (5 min plus calibration time)

```sh
# On the Pi:
sudo apt-get install -y git
git clone https://github.com/silvance/CSIDetector.git
cd CSIDetector

# Plug the ESP32-S3 RX into one of the Pi's USB ports.
# Confirm it shows up:
ls /dev/ttyACM*    # should print /dev/ttyACM0 (or similar)

# Run the installer. It will:
#   - apt-install python3 + numpy
#   - create a venv
#   - prompt for Telegram bot token, chat id, location label
#   - prompt you to leave the room, then calibrate for 30s
#   - install + start the systemd service
sudo ./scripts/install-appliance.sh
```

When the installer asks you to leave the room, **actually leave** — the
baseline calibration captures whatever motion is in the room during the
record window. A baseline > 2.0× is the script's signal that the room
wasn't empty.

### 4. Confirm it's running

The installer ends by tailing the service log. You should see:

```
csidetector-alert[1234]: 2026-05-12T17:14:22 STILL score=0.142 baseline=0.183 ratio=0.78
```

Walk past the sensor — terminal prints `MOTION` and your phone buzzes.

`Ctrl-C` exits the log tail; the service keeps running.

## Day-to-day

```sh
sudo systemctl status csidetector-alert      # is it running?
journalctl -u csidetector-alert -n 50        # recent activity
journalctl -u csidetector-alert -f           # tail live
sudo systemctl restart csidetector-alert     # after editing config
sudo systemctl stop csidetector-alert        # quiet hours / debugging
```

Inspect the alert queue if a notification didn't arrive:

```sh
sqlite3 /var/lib/csidetector/alert-queue.db \
    "SELECT event_id, attempts, sent_ts, last_error FROM events ORDER BY created_ts DESC LIMIT 20"
```

## Reconfigure

To change the Telegram credentials, location label, or re-calibrate the
baseline (e.g. after moving the sensor), just re-run the installer:

```sh
sudo ./scripts/install-appliance.sh
```

It picks up the current values as defaults at each prompt, so hit
Enter through any unchanged fields. It restarts the service at the end.

## Uninstall

```sh
sudo ./scripts/install-appliance.sh --uninstall
```

Removes the systemd unit and `/etc/csidetector/`. Leaves the repo +
venv on disk in case you want to come back to it.

## Troubleshooting

### "device not found at /dev/ttyACM0"

The RX isn't enumerating. Common causes:

- Wrong USB port (some Pi Zero USB ports are power-only on some
  carrier boards — try the other one)
- ESP32 needs to be reset after plugging in: tap the reset button
- Bad / data-deficient cable — try a different USB cable

### Baseline keeps coming back > 2.0×

The Pi chassis + USB power supply is RF-noisy, and the ESP32 antenna
is sitting right next to it. Three things to try:

1. Reposition the ESP32 so the antenna points away from the Pi
2. Use a short USB extension cable to put the ESP32 ~30 cm from the
   Pi body
3. Move the Pi + ESP32 box itself away from other noise sources
   (3D printers, microwaves, USB3 hubs, screens)

Then re-run the installer to recalibrate.

### Phone doesn't buzz, but `journalctl` shows MOTION events

Check the alert queue:

```sh
sqlite3 /var/lib/csidetector/alert-queue.db "SELECT * FROM events"
```

Rows with `sent_ts NULL` and a `last_error` populated tell you what's
wrong (usually a bad Telegram token or network issue).

### `sudo: ./scripts/install-appliance.sh: command not found`

You're not in the repo root, or the file isn't executable:

```sh
cd ~/CSIDetector
chmod +x scripts/install-appliance.sh
sudo ./scripts/install-appliance.sh
```

## What's not yet built

- **OTA updates**: today you `git pull` + re-run the installer.
  A future cron-driven auto-update is on the roadmap.
- **Web UI**: no in-browser config / status panel yet. CLI + journalctl
  is the operator interface.
- **Multiple sensors**: each Pi+ESP32 unit is independent. Hand-edit
  `location_label` in `alert.toml` to disambiguate, or wait for the
  multi-sensor work.
