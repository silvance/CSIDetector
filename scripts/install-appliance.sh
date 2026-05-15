#!/usr/bin/env bash
# Bootstrap a Raspberry Pi (or any Linux box) into a CSIDetector alert
# appliance. Idempotent — safe to re-run.
#
# What it does:
#   1. Installs system + Python deps (numpy via apt for speed).
#   2. Creates a venv in host/.venv with the alert-mode requirements.
#   3. Asks for Telegram bot token + chat id + a location label.
#   4. Writes /etc/csidetector/alert.toml (chmod 600).
#   5. Asks you to plug in the ESP32 RX, then leave the room. Runs a
#      30-second calibration and saves the baseline.
#   6. Installs scripts/csidetector-alert.service, enables it.
#   7. Tails the journal so you can confirm it's running.
#
# Usage:
#   sudo ./scripts/install-appliance.sh
#   sudo ./scripts/install-appliance.sh --uninstall
#
# Re-run the script any time to re-prompt + re-calibrate without
# losing the systemd unit.

set -euo pipefail

# --------------------------------------------------------------------------
# Resolve repo paths.
# --------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_DIR="$REPO_ROOT/host"
VENV_DIR="$HOST_DIR/.venv"
SYSCONF_DIR="/etc/csidetector"
ENV_FILE="$SYSCONF_DIR/env"
ALERT_TOML="$SYSCONF_DIR/alert.toml"
UNIT_NAME="csidetector-alert"
UNIT_FILE="/etc/systemd/system/${UNIT_NAME}.service"
SERVICE_USER="${SUDO_USER:-${USER:-pi}}"
SERVICE_HOME="$(getent passwd "$SERVICE_USER" | cut -d: -f6)"

UNINSTALL=0
for arg in "$@"; do
    case "$arg" in
        --uninstall) UNINSTALL=1 ;;
        -h|--help)
            sed -n '2,25p' "$0"
            exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    echo "ERROR: needs sudo (apt + systemd writes)." >&2
    exit 1
fi

heading() {
    printf '\n\033[1;36m== %s ==\033[0m\n' "$1"
}

# --------------------------------------------------------------------------
# Uninstall path: tear down systemd + sysconf, leave the repo + venv alone.
# --------------------------------------------------------------------------
if [[ "$UNINSTALL" -eq 1 ]]; then
    heading "Uninstalling appliance"
    systemctl disable --now "$UNIT_NAME" 2>/dev/null || true
    rm -f "$UNIT_FILE"
    systemctl daemon-reload
    rm -rf "$SYSCONF_DIR"
    echo "Removed: $UNIT_FILE, $SYSCONF_DIR"
    echo "Left in place: $REPO_ROOT, $VENV_DIR"
    exit 0
fi

# --------------------------------------------------------------------------
# 1. System deps.
# --------------------------------------------------------------------------
heading "Installing system deps"
apt-get update -qq
# python3-numpy via apt is dramatically faster than pip on a Pi (no
# OpenBLAS source build). Everything else falls out of stdlib in 3.11+.
apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-numpy

# --------------------------------------------------------------------------
# 2. Venv. Use --system-site-packages so the apt-installed numpy is visible.
# --------------------------------------------------------------------------
heading "Creating venv at $VENV_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
    sudo -u "$SERVICE_USER" python3 -m venv --system-site-packages "$VENV_DIR"
fi
# pyserial is needed for /dev/ttyACM* reads. matplotlib + PyQt6 are
# NOT needed for alert mode and are skipped (huge install on a Pi).
sudo -u "$SERVICE_USER" "$VENV_DIR/bin/pip" install --quiet \
    --upgrade pip pyserial

# --------------------------------------------------------------------------
# 3. Interactive config — bot token, chat id, location, device path.
# --------------------------------------------------------------------------
heading "Telegram + sensor config"
mkdir -p "$SYSCONF_DIR"

prompt() {
    local var_name="$1"
    local label="$2"
    local default="${3:-}"
    local current=""
    if [[ -f "$ENV_FILE" ]]; then
        current=$(awk -F= -v k="$var_name" '$1==k {sub(/^"/,"",$2); sub(/"$/,"",$2); print $2}' "$ENV_FILE")
    fi
    local prompt_default="${current:-$default}"
    local hint=""
    if [[ -n "$prompt_default" ]]; then
        hint=" [${prompt_default}]"
    fi
    printf '%s%s: ' "$label" "$hint"
    read -r value || value=""
    if [[ -z "$value" ]]; then
        value="$prompt_default"
    fi
    printf -v "$var_name" '%s' "$value"
}

prompt BOT_TOKEN  "Telegram bot token (from @BotFather)"
prompt CHAT_ID    "Telegram chat ID (from getUpdates)"
prompt LOCATION   "Location label (e.g. 'Office')" "Sensor"
prompt DEVICE     "Serial device for the RX" "/dev/ttyACM0"

if [[ -z "${BOT_TOKEN:-}" || -z "${CHAT_ID:-}" ]]; then
    echo "ERROR: bot token and chat id are required." >&2
    exit 1
fi

# Write alert.toml (read-only by root + service user; not world-readable).
cat >"$ALERT_TOML" <<EOF
# Managed by install-appliance.sh. Re-run the installer to edit.

[notifier]
type      = "telegram"
bot_token = "${BOT_TOKEN}"
chat_id   = "${CHAT_ID}"

[alert]
cooldown_s     = 60
clear_on_exit  = false
location_label = "${LOCATION}"

[queue]
enabled = true
path    = "/var/lib/csidetector/alert-queue.db"
EOF
chmod 600 "$ALERT_TOML"
chown root:root "$ALERT_TOML"

# Queue directory needs to be writable by the service user.
mkdir -p /var/lib/csidetector
chown "$SERVICE_USER":"$SERVICE_USER" /var/lib/csidetector

# --------------------------------------------------------------------------
# 4. Calibration. Block on the user being out of the room.
# --------------------------------------------------------------------------
heading "Calibrating still-room baseline"
echo "Plug the ESP32 RX into the Pi (or this host) now."
echo "Make sure your TX is powered on and in the room."
echo
echo "Confirming the device is present at ${DEVICE}..."
for _ in $(seq 1 20); do
    if [[ -c "$DEVICE" ]]; then
        break
    fi
    sleep 0.5
done
if [[ ! -c "$DEVICE" ]]; then
    echo "ERROR: $DEVICE not found. Is the RX plugged in?" >&2
    echo "       Available: $(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo none)" >&2
    exit 1
fi

echo "Device found. The next step records a 30-second still-room baseline"
echo "after a 30-second walk-out window. Press Enter, then leave the room."
read -r _ || true

BASELINE_RAW="$(sudo -u "$SERVICE_USER" "$VENV_DIR/bin/python" \
    "$HOST_DIR/run.py" alert calibrate "$DEVICE" --seconds 30 --settle 30 \
    2>/dev/null)"
BASELINE="$(printf '%s' "$BASELINE_RAW" | tail -n1 | tr -d '[:space:]')"

if [[ -z "$BASELINE" ]]; then
    echo "ERROR: calibration produced no output. Re-run after checking RX." >&2
    exit 1
fi

# Sanity-bound. Compare to typical still-room σ for this hardware
# (~0.05-1.5). Outside that range usually means the RX is in a noisy
# spot (chassis EMI, etc) or the room wasn't actually empty.
echo "Baseline: $BASELINE"
case "$BASELINE" in
    0.000*|0.001|0.002|0.003|0.004|0.005)
        echo "WARNING: baseline is very low — RX may not be receiving samples." ;;
esac
hi_check="$(printf '%s' "$BASELINE" | awk '{print ($1+0 > 2.0)}')"
if [[ "$hi_check" == "1" ]]; then
    echo "WARNING: baseline > 2.0× is unusually high (typical: 0.05-1.5)."
    echo "         The room may not have been empty, or the RX is in an"
    echo "         RF-noisy spot. Detection will still work, but consider"
    echo "         moving the RX and re-running the installer."
fi

# Write the EnvironmentFile that the unit references.
cat >"$ENV_FILE" <<EOF
# Managed by install-appliance.sh. Re-run the installer to refresh.
DEVICE=${DEVICE}
BASELINE=${BASELINE}
LOCATION=${LOCATION}
EOF
chmod 644 "$ENV_FILE"

# --------------------------------------------------------------------------
# 5. systemd unit.
# --------------------------------------------------------------------------
heading "Installing systemd unit ($UNIT_NAME)"
cat >"$UNIT_FILE" <<EOF
[Unit]
Description=CSIDetector alert sensor (single-RX motion → Telegram)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
EnvironmentFile=${ENV_FILE}
WorkingDirectory=${HOST_DIR}
ExecStart=${VENV_DIR}/bin/python ${HOST_DIR}/run.py alert detect \${DEVICE} \\
    --baseline \${BASELINE} \\
    --alert-config ${ALERT_TOML} \\
    --location \${LOCATION}
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "$UNIT_NAME" >/dev/null
systemctl restart "$UNIT_NAME"

# --------------------------------------------------------------------------
# 6. Status + tail.
# --------------------------------------------------------------------------
sleep 2
heading "Status"
systemctl --no-pager --full status "$UNIT_NAME" | head -12 || true

heading "Live log (Ctrl-C to exit; the service keeps running)"
echo "$ journalctl -u $UNIT_NAME -f"
exec journalctl -u "$UNIT_NAME" -f
