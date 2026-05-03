# CSIDetector

WiFi Channel State Information (CSI) based motion sensing on ESP32 hardware.
A small mesh of ESP32-S3 transmitters and receivers cooperate over a
hotspot to localize motion in a room; the host renders a live floor-plan
heatmap and a 2.5D scene with a "person" pin tracking the brightest
spot.

## Hardware

- 1+ ESP32-S3 transmitter board (more = better localization).
- 4+ ESP32-S3 receiver boards distributed around the room.
- A host computer with Python 3.10+ and a 2.4 GHz WiFi adapter that
  supports AP mode (built-in card or any cheap USB dongle).
- USB chargers for the receivers (no host tether needed once flashed).

The S2 / C3 / C5 also work in principle but the firmware has only been
exercised on S3.

## Architecture

```
   ┌──────────┐  ESP-NOW broadcast (HT20 MCS0, ch 11)  ┌──────────┐
   │ ESP32-TX │ ──────────────────────────────────────▶│ ESP32-RX │  × N
   │   ×M     │                                        │ STA + CSI│
   └──────────┘                                        └────┬─────┘
                                                            │ UDP/WiFi
                                                            ▼
                                                     ┌────────────┐
                                                     │  host (PC) │
                                                     │ AP + viewer│
                                                     └────────────┘
```

Each TX broadcasts a small ESP-NOW frame at a fixed rate on a fixed
channel. Each RX listens for those broadcasts (filtered by source MAC),
and for every received broadcast emits one CSI sample. The RX then
sends each sample as a binary UDP packet to the host's hotspot IP. The
host listens on UDP, demuxes per-RX, computes per-link motion-σ on a
sliding window, and renders.

Single-stream UART is still supported for development on one board (the
firmware also prints CSI rows in the [esp-csi](https://github.com/espressif/esp-csi)
text format), but the full demo runs over UDP.

## Firmware

Both firmware projects target **ESP-IDF v5.3 or newer** (tested on
v5.5). Set the target once on a fresh checkout:

    idf.py set-target esp32s3

### Transmitter (`firmware/csi_transmitter/`)

```sh
cd firmware/csi_transmitter
idf.py menuconfig    # CSI_TX_CHANNEL / CSI_TX_RATE_HZ
idf.py -p /dev/ttyACM0 flash monitor   # or /dev/ttyUSB0 on USB-UART boards
```

Boot log:

    I (412) csi_tx: TX up: mac=ac:a7:04:2c:42:54 ch=11 rate=100Hz

The TX firmware pins broadcasts to 11n HT20 MCS0 via
`esp_now_set_peer_rate_config`. Without this, ESP-NOW falls back to
11b 1 Mbps DSSS (no HT-LTF), and the receiver's CSI engine never
fires for the broadcasts. Note the MAC — every receiver needs it in
its filter.

### Receiver (`firmware/csi_receiver/`)

```sh
cd firmware/csi_receiver
idf.py menuconfig    # see required values below
idf.py -p /dev/ttyACMx flash monitor
```

Required `menuconfig` values for the multi-RX UDP setup:

    CSI_RX_CHANNEL         = 11               # must match TX
    CSI_RX_FILTER_TX_MAC   = <TX1 MAC>, <TX2 MAC>     # comma-separated
    CSI_RX_WIFI_SSID       = CSIDetector
    CSI_RX_WIFI_PASS       = (blank, see Hotspot section)
    CSI_RX_HOST_IP         = 10.42.0.1        # host's hotspot IP
    CSI_RX_HOST_PORT       = 5566

Leaving `CSI_RX_WIFI_SSID` blank disables WiFi STA + UDP and falls back
to the original UART-only single-RX flow, which is useful for a quick
smoke test on one board.

> **Note on WiFi credentials**: `CSI_RX_WIFI_SSID` and `CSI_RX_WIFI_PASS`
> are baked into the receiver's `sdkconfig` and flashed in plain text to
> the chip's flash. For a closed demo network this is fine; for any
> other deployment, treat the receivers as having a recoverable
> password and don't reuse a credential you care about elsewhere.
> The recipe in the next section uses an open hotspot, which sidesteps
> this entirely.

### Host hotspot

NetworkManager creates a 2.4 GHz hotspot. Pick a USB dongle if you also
need internet on the host's built-in card:

```sh
IFACE=wlp2s0u2   # whatever `nmcli device | grep wifi` shows
nmcli connection add type wifi ifname "$IFACE" con-name Hotspot autoconnect no \
    ssid CSIDetector \
    802-11-wireless.mode ap \
    802-11-wireless.band bg \
    802-11-wireless.channel 11 \
    ipv4.method shared \
    ipv6.method ignore
nmcli connection up Hotspot

# Open the UDP port in the right firewalld zone (NM puts hotspot ifaces
# in `nm-shared`, not the default `FedoraWorkstation`).
sudo firewall-cmd --zone=nm-shared --add-port=5566/udp --permanent
sudo firewall-cmd --reload
```

Verify with `iw dev "$IFACE" info | grep channel` and
`ip addr show "$IFACE" | grep inet`. Channel 11 must match
`CSI_TX_CHANNEL` and `CSI_RX_CHANNEL`.

This recipe sets up an OPEN (no-password) hotspot. WPA2 also works, but
some USB AP adapters can't reliably complete the 4-way handshake with
multiple ESP32 STAs simultaneously, so for a closed demo network OPEN
is the path of least resistance.

## Host pipeline

```sh
cd host
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`<source>` is a serial port (`/dev/ttyUSB0`, `COM5`), a saved log file,
`udp:<port>` for the hotspot setup, or `-` for stdin. Multi-RX
subcommands (`heatmap`, `view3d`, `calibrate-links`) only make sense
with a `udp:<port>` source.

### Quick sanity check on packet rates

```sh
python -c "
import socket, time, collections
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(('0.0.0.0', 5566))
counts = collections.Counter(); end = time.time() + 5
while time.time() < end:
    data, _ = s.recvfrom(2048); counts[(data[2:8].hex(':'), data[8:14].hex(':'))] += 1
for (rx, tx), n in sorted(counts.items()):
    print(f'TX={tx}  RX={rx}  {n//5}/s')"
```

Expect one row per (TX, RX) pair at ~100/s. With 2 TX × 4 RX you'll see
8 pairs.

### Calibrate per-link still-room baselines

Leave the room. The script counts down before recording starts:

```sh
python run.py calibrate-links udp:5566 --settle 30 --seconds 30 --out baselines.json
```

`--settle` doubles as the walk-out timer. The output JSON maps each
RX MAC to its baseline motion-σ. RXs that didn't deliver enough
samples are flagged in stderr — silent drops would later show as
dead links in the heatmap.

### Live floor-plan heatmap

```sh
cp links.example.json links.json    # edit room polygon, TX/RX MACs, positions
python run.py heatmap udp:5566 --links links.json --baselines baselines.json
```

Each TX-RX line is tinted by its current motion-σ divided by its
still-room baseline (1× = idle, ≥5× = saturated). With multiple
TXs you get multiple fans of links crossing the room.

### 2.5D room viewer

```sh
python run.py view3d udp:5566 --links links.json --baselines baselines.json
```

Walls extruded from the polygon, floor surface coloured by per-cell
motion likelihood (Gaussian-weighted sum of per-link motion-σ along
each TX-RX line), red person pin at the brightest cell when
likelihood exceeds a threshold.

### Single-stream debug viewer

```sh
python run.py view /dev/ttyUSB0      # one RX, UART
python run.py view udp:5566          # pins to the first rx_id seen, drops the rest
```

Subcarrier × time waterfall + motion-σ trace. Useful for verifying
one board independently.

### Capture a log

```sh
python run.py capture /dev/ttyUSB0 still.log --seconds 30
```

### Single-stream binary detector

For a single-board (UART) setup. `calibrate` here emits one scalar
baseline σ; the multi-RX path uses `calibrate-links` instead, which
writes a per-RX JSON consumed by `heatmap` / `view3d`. The two
subcommands are not interchangeable.

```sh
BASELINE=$(python run.py calibrate /dev/ttyUSB0 --seconds 30)
python run.py detect /dev/ttyUSB0 --baseline "$BASELINE"
```

Prints `MOTION` / `STILL` events on transitions. `--enter` / `--exit`
control the hysteresis ratio (defaults 3.0× / 1.5×).

## Tuning notes

- **Channel**: pick the 2.4 GHz channel with the least neighbouring
  WiFi traffic. Channel 11 is a reasonable default in North America.
  TX, RX, and the hotspot AP must all be on the same channel.
- **TX placement**: corners are best. A TX in the middle of the room
  makes every link's motion-σ rise on any motion (the long links
  dominate), which kills directional information. Two TXs in
  diagonally-opposite corners give the cleanest two-fan geometry.
- **Sample rate**: 100 Hz per TX is the sweet spot — high enough to
  catch hand motion, low enough that ESP-NOW airtime is comfortable
  with multiple TXs.
- **Antenna**: the WROOM-1U's u.fl connector takes any 2.4 GHz
  antenna (WiFi Pineapple antennas, generic ESP32 dev-kit antennas).
  LoRa antennas (sub-GHz) will not work.
- **Metal in the line of sight**: still kills the link. Metal furniture,
  appliances, and structural beams between TX and RX mask any motion
  behind them.

## Roadmap

- [x] Single TX / single RX, live waterfall, single-stream binary detector
- [x] Multi-RX over WiFi (UDP forwarding to host), per-link motion-σ heatmap
- [x] Multi-TX support, per-link baselines, 2.5D floor-plan viewer with
      person pin
- [ ] Multi-person separation (top-K local maxima with non-max suppression)
- [ ] Aggregator firmware on ESP32-C5-with-screen for an untethered display
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
  ESP32s do not have.
