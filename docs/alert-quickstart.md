# Alert mode — 15-minute quickstart

You'll end up with: an ESP32 sensor that sends you a Telegram message
when someone walks past it.

This guide is **numbered steps, no narrative**. If you want to know
*why* any of this works, see the main [README](../README.md). Stop
reading and follow it linearly; troubleshooting is in §11 at the end.

## 0. What you need (5 min)

- 2× ESP32-S3 dev boards (one TX, one RX) and 2× USB-C cables
- A computer (Linux / macOS) with Python ≥3.11 and Docker (for tests, optional)
- A phone with the [Telegram](https://telegram.org/) app installed
- ESP-IDF v5.3+ installed and working (the `idf.py` command exists). If
  not, install it: <https://docs.espressif.com/projects/esp-idf/en/v5.5/esp32s3/get-started/index.html>
- This repo cloned somewhere

That's it. No hotspot, no router config, no Wi-Fi credentials anywhere.

---

## 1. Make a Telegram bot (2 min)

1. Open Telegram, search for `@BotFather`, start a chat.
2. Send `/newbot`. Pick a name (any) and a username ending in `bot`
   (e.g. `office_motion_test_bot`).
3. BotFather replies with a token — long string like
   `123456:AAH...`. **Save this**; you need it later.

## 2. Get your Telegram chat ID (1 min)

1. Search Telegram for the bot you just made and send it any message
   (e.g. `hi`).
2. In a browser, open this URL with your token substituted:
   ```
   https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   ```
3. Look for `"chat":{"id":1234567}` in the response. **That number is
   your chat ID.** Save it.

If `getUpdates` returns `{"ok":true,"result":[]}`, you didn't message
the bot — go back to step 2.1.

---

## 3. Flash the transmitter (3 min)

1. Plug **only the TX board** in. Note its `/dev/ttyACM*` (Linux) or
   `/dev/cu.usbmodem*` (macOS).
2. From repo root:
   ```sh
   ./idf-env.sh csi_transmitter
   idf.py set-target esp32s3
   idf.py menuconfig
   ```
3. In menuconfig, navigate to `CSI Transmitter Configuration`:
   - `CSI_TX_CHANNEL`: leave at default (11)
   - `CSI_TX_RATE_HZ`: leave at default (100)
4. Save (`S`), exit (`Q`).
5. Flash + watch boot:
   ```sh
   idf.py -p /dev/ttyACM0 flash monitor
   ```
6. **Copy the line that starts `TX up: mac=`.** You need this MAC for
   step 4.3. Example output:
   ```
   I (412) csi_tx: TX up: mac=ac:a7:04:2c:42:54 ch=11 rate=100Hz
   ```
7. `Ctrl-]` to exit monitor. Unplug TX.

## 4. Flash the receiver (3 min)

1. Plug **only the RX board** in.
2. From repo root:
   ```sh
   ./idf-env.sh csi_receiver
   idf.py set-target esp32s3
   idf.py menuconfig
   ```
3. In menuconfig, navigate to `CSI Receiver Configuration`:
   - `CSI_RX_CHANNEL`: must match TX (11)
   - `CSI_RX_FILTER_TX_MAC`: paste the TX MAC from step 3.6
   - `CSI_RX_WIFI_SSID`: **leave blank** ← this is what makes it
     UART-mode instead of hotspot-mode
4. Save, exit.
5. Flash + monitor:
   ```sh
   idf.py -p /dev/ttyACM0 flash monitor
   ```
6. You should see lines starting with `CSI_DATA,` streaming past at
   ~100/s. If yes: TX is broadcasting and RX is hearing it. **Hit
   `Ctrl-]` to exit.**
7. **Plug the TX into a USB charger** (any phone charger works). It
   needs power but not the host. Place it 2-3 m from where the RX
   will sit.

If step 4.6 shows no `CSI_DATA,` lines after 30 seconds, jump to §11.

---

## 5. Set up the host (2 min)

```sh
cd host
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## 6. Configure Telegram alerts

```sh
cp alert.example.toml alert.toml
chmod 600 alert.toml          # the file holds a credential
```

Open `alert.toml` in any editor, replace `REPLACE_WITH_YOUR_BOT_TOKEN`
and `REPLACE_WITH_YOUR_CHAT_ID` with the values from steps 1-2. Save.

Optional: set `location_label = "Office"` (or whatever) so alert
messages prefix that string.

---

## 7. Calibrate the still-room baseline (1 min)

1. Make sure the RX is plugged into your host via USB. Confirm the
   port:
   ```sh
   ls /dev/ttyACM* /dev/cu.usbmodem* 2>/dev/null
   ```
2. **Leave the room** (or the immediate area — keep stationary objects
   in place but get yourself out of the line of sight). Then run:
   ```sh
   BASELINE=$(python run.py alert calibrate /dev/ttyACM0 --seconds 30 --settle 30)
   echo "$BASELINE"
   ```
3. Wait the full 60 s. The script counts down. When it finishes, it
   prints **one float** like `0.183421`. **Save this number** — it's
   the only output the next step needs.

If the value is < 0.001 or > 1.0, the receiver isn't streaming
properly. Jump to §11.

## 8. Run detect with alerts armed

```sh
python run.py alert detect /dev/ttyACM0 \
    --baseline "$BASELINE" \
    --alert-config alert.toml \
    --location "Office"
```

The terminal sits quiet, printing `STILL` / `MOTION` lines as state
changes.

## 9. Confirm it works (the demo)

1. Walk past the RX board. Within ~1 second the terminal prints:
   ```
   2026-05-09T14:22:11 MOTION score=0.834 baseline=0.183 ratio=4.55
   ```
2. **Your Telegram phone should buzz** with `[MOTION] Office | ratio=4.55×`.
3. Stand still for 5 seconds — terminal prints `STILL` again.
4. Walk past again — terminal prints `MOTION` again. **Phone does NOT
   buzz a second time** (cooldown suppresses it for 60 s by default).

If all 4 happen, you're done. Hand the device to someone else and let
them try.

---

## 10. Stop / restart / leave running

- **Stop**: `Ctrl-C` in the detect terminal. Pending alerts in the
  queue persist.
- **Restart**: re-run the same command. Queued alerts deliver as soon
  as Telegram is reachable.
- **Leave running overnight**: use `tmux` / `screen` or
  `nohup python run.py alert detect … &` — no systemd unit shipped
  yet (coming).
- **Inspect the queue**:
  ```sh
  sqlite3 ~/.csidetector/alert-queue.db \
      "SELECT event_id, attempts, sent_ts, last_error FROM events"
  ```

## 11. Troubleshooting

### No `CSI_DATA,` lines from the RX (step 4.6)

- TX is on a different channel than RX. Re-run `idf.py menuconfig` on
  both, confirm `CSI_TX_CHANNEL` == `CSI_RX_CHANNEL`.
- TX MAC mismatch. The RX only forwards CSI from frames whose source
  MAC is in `CSI_RX_FILTER_TX_MAC`. Re-flash the TX with monitor and
  copy the MAC again — case-sensitive only for hex digits.
- TX is unpowered. Look for the blue boot LED on the TX.
- RX is in Wi-Fi-STA mode by accident. Confirm `CSI_RX_WIFI_SSID` is
  **empty**, not the literal string `""` or `CSIDetector`.

### Calibration baseline is < 0.001 or > 1.0 (step 7.2)

- < 0.001: the RX received nothing or near-nothing. See "no `CSI_DATA,`
  lines" above.
- \> 1.0: someone (you?) was moving during calibration. Repeat with
  `--settle 60` and actually leave the room.

### Walking past doesn't trigger MOTION (step 9.1)

- Check the terminal — does it print `STILL`/`MOTION` lines but no
  Telegram comes? See next entry.
- If the terminal stays silent: the baseline is too high (calibration
  caught some motion). Re-run §7 with the room genuinely empty.
- Check `--enter` ratio. Default is 3.0× — if you have a quiet
  environment, the baseline is small and crossing 3× is easy. If
  you're in a busy office with HVAC, real motion may need a higher
  multiple to cleanly distinguish. Try `--enter 4.0`.

### Terminal prints MOTION but no Telegram message arrives

- Is `alert.toml` actually being read? Add `--alert-config alert.toml`
  to the command (forgetting it puts you back in stdout-only mode).
- Bot token / chat ID typo. Run:
  ```sh
  curl -s "https://api.telegram.org/bot<TOKEN>/sendMessage" \
       -d "chat_id=<CHATID>&text=manual_test"
  ```
  If `"ok":true` comes back: the values are correct, problem is in
  the alert pipeline. If `"ok":false`: fix the values in `alert.toml`.
- Cooldown suppression. By default, after one MOTION alert the next
  MOTION transition within 60 s does not re-notify. Wait a minute and
  walk past again, or set `--cooldown-s 5` for testing.
- Network down. Check the queue:
  ```sh
  sqlite3 ~/.csidetector/alert-queue.db "SELECT * FROM events"
  ```
  Rows with `sent_ts` NULL and `attempts > 0` are pending retry.
  They'll deliver as soon as you have internet again.

### Constant false MOTION alerts when nothing is moving

- The RX is in an EMI hot zone (next to a 3D printer, microwave, USB3
  hub, switching power supply, fluorescent ballast). Move it.
- The baseline drifted because the RF environment changed. Re-run §7.

### "command not found: idf.py"

- ESP-IDF isn't on PATH. Run `./idf-env.sh` from repo root — it sources
  `~/esp/esp-idf/export.sh` for you. If your IDF lives elsewhere,
  set `IDF_EXPORT=/path/to/your/export.sh ./idf-env.sh`.

---

## 12. What's *not* in this guide

- Localize mode (multi-RX heatmap with floor plan). Different beast;
  see the main README §A.
- Running multiple sensors against the same Telegram chat. Possible
  (use `--location` to disambiguate) but out of scope here.
- Self-hosting Telegram alternatives (Matrix, Slack, generic webhook).
  Easy to add — `Notifier` is a small ABC, see
  `host/csidetector/modes/alert/notifier.py`. PRs welcome.
- Running the host on embedded Linux / a router. Doable but not yet
  packaged.
