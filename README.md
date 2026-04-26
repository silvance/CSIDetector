# CSIDetector

WiFi Channel State Information (CSI) based motion sensing on ESP32 hardware,
with a host-side live viewer and binary detector.

The current build covers a single transmitter / single receiver pair. The
host shows a real-time waterfall of subcarrier amplitude vs. time — motion
in front of the link appears as vertical color streaks across the heatmap.

## Hardware

- Two ESP32 boards (S3 recommended; the S2 / C3 / C5 should work too).
- A USB cable for each board.
- A host computer with Python 3.10+.

A future milestone adds 1 transmitter ↔ N receivers and projects per-link
motion intensity onto a room layout. See `# Roadmap` below.

## Architecture

```
   ┌──────────────┐  ESP-NOW broadcast @ 100 Hz, ch 11   ┌──────────────┐
   │   ESP32-TX   │ ────────────────────────────────────▶│   ESP32-RX   │
   │ csi_transmit │                                      │  csi_receive │
   └──────────────┘                                      └──────┬───────┘
                                                                │ UART @ 921600
                                                                ▼
                                                         ┌──────────────┐
                                                         │  host (PC)   │
                                                         │  view/detect │
                                                         └──────────────┘
```

The TX broadcasts a small ESP-NOW packet at a fixed rate on a fixed
channel. The RX listens promiscuously on that channel; every received
broadcast generates one CSI sample, which the firmware prints over UART
in the [esp-csi](https://github.com/espressif/esp-csi) line format. The
host parses each line into a complex subcarrier vector and renders or
analyzes it.

## Firmware

Both firmware projects target ESP-IDF v5.x.

### Transmitter (`firmware/csi_transmitter/`)

```sh
cd firmware/csi_transmitter
idf.py set-target esp32s3
idf.py menuconfig    # adjust CSI_TX_CHANNEL / CSI_TX_RATE_HZ if needed
idf.py build flash monitor
```

You should see a log line like:

    I (412) csi_tx: TX up: mac=aa:bb:cc:dd:ee:ff ch=11 rate=100Hz

Note the MAC — you can pin the receiver to it for clean filtering.

### Receiver (`firmware/csi_receiver/`)

```sh
cd firmware/csi_receiver
idf.py set-target esp32s3
idf.py menuconfig    # set CSI_RX_CHANNEL = TX channel; optionally set
                     # CSI_RX_FILTER_TX_MAC = "aabbccddeeff" (12 hex chars)
idf.py build flash monitor
```

Once running, the UART stream looks like:

    type,seq,mac,rssi,rate,...,first_word,data
    CSI_DATA,0,aa:bb:cc:dd:ee:ff,-46,11,1,7,1,...,128,0,"[1,-2,0,3,...]"
    CSI_DATA,1,...

at roughly 100 lines/second.

## Host pipeline

```sh
cd host
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

All commands take a `<source>` that is either a serial port
(`/dev/ttyUSB0`, `COM5`), a saved log file, or `-` for stdin.

### Live heatmap

```sh
python run.py view /dev/ttyUSB0
```

Top panel: subcarrier (y) × time (x), color = |H| in dB. Bottom panel:
sliding-window standard deviation across active subcarriers (the motion
score). Wave a hand between the two boards; the streaks should be
unmistakable.

### Capture a log for offline analysis

```sh
python run.py capture /dev/ttyUSB0 still.log --seconds 30
```

### Calibrate a still-room baseline

Leave the room for the duration. The first 10 s of samples are dropped
to let the radio's AGC lock.

```sh
BASELINE=$(python run.py calibrate /dev/ttyUSB0 --seconds 30)
echo "baseline=$BASELINE"
```

### Live binary detection

```sh
python run.py detect /dev/ttyUSB0 --baseline "$BASELINE"
```

You'll see one event line per transition:

    2026-04-26T18:21:04 MOTION score=0.0182 baseline=0.0041 ratio=4.43
    2026-04-26T18:21:09 STILL  score=0.0049 baseline=0.0041 ratio=1.20

Tune `--enter` and `--exit` to taste; the defaults (3.0×, 1.5×) are
conservative.

## Tuning notes

- **Channel**: pick whichever 2.4 GHz channel has the least traffic.
  Channel 11 is a reasonable default for North America.
- **Sample rate**: 100 Hz is the sweet spot — high enough to catch hand
  motion, low enough that ESP-NOW transmission stays reliable.
- **Antenna placement**: the boards' built-in chip antennas are
  directional. Pointing them roughly at each other through the volume of
  interest gives the cleanest signal.
- **Metal in the line of sight**: kills it. Metal furniture, appliances,
  and structural beams between TX and RX will mask any motion behind
  them.
- **Range**: usable up to ~10–15 m indoors at 100 Hz.

## Roadmap

- [x] Single TX / single RX pair, live heatmap, binary detector
- [ ] Multi-RX (4–5 nodes) with per-link motion-intensity score
- [ ] Floor-plan overlay viewer (per-link blobs on a 2D room sketch)
- [ ] Aggregator firmware on ESP32-C5-with-screen for an
      untethered display
- [ ] Optional: NBVI subcarrier auto-selection (espectre's MVS)

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
  ESP32s do not have.
