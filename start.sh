#!/usr/bin/env bash
# Bring the CSIDetector hotspot + host viewer up in one shot.
#
# Usage:
#   ./start.sh              # heatmap only
#   ./start.sh publish      # heatmap + presence publisher (broadcasts to C5)
#
# Env overrides (optional):
#   HOTSPOT_NAME=CSIDetector  nmcli connection name to activate
#   PORT=5566                 UDP port the receivers send to
#   LINKS=host/links.json
#   BASELINES=host/baselines.json
#   C5_ADDR=10.42.0.255       broadcast address of the hotspot subnet

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_DIR="$REPO_ROOT/host"

HOTSPOT_NAME="${HOTSPOT_NAME:-CSIDetector}"
PORT="${PORT:-5566}"
LINKS="${LINKS:-$HOST_DIR/links.json}"
BASELINES="${BASELINES:-$HOST_DIR/baselines.json}"
C5_ADDR="${C5_ADDR:-10.42.0.255}"

# --- 1. Hotspot ----------------------------------------------------------
if nmcli -t -f NAME,STATE connection show --active 2>/dev/null \
        | grep -qx "${HOTSPOT_NAME}:activated"; then
    echo "[start] hotspot '${HOTSPOT_NAME}' already up"
else
    echo "[start] activating hotspot '${HOTSPOT_NAME}'..."
    if ! nmcli connection up "${HOTSPOT_NAME}"; then
        echo
        echo "[start] ERROR: couldn't activate '${HOTSPOT_NAME}'."
        echo "        Available connections:"
        nmcli -t -f NAME connection show | sed 's/^/          /'
        echo
        echo "        If '${HOTSPOT_NAME}' isn't listed, create it once with:"
        echo "          nmcli con add type wifi ifname <iface> con-name '${HOTSPOT_NAME}' \\"
        echo "              ssid '${HOTSPOT_NAME}' mode ap"
        echo "          nmcli con modify '${HOTSPOT_NAME}' 802-11-wireless.band bg"
        echo "          nmcli con modify '${HOTSPOT_NAME}' 802-11-wireless.channel 11"
        echo "          nmcli con modify '${HOTSPOT_NAME}' ipv4.method shared"
        exit 1
    fi
    echo "[start] waiting 3s for the AP to settle..."
    sleep 3
fi

# --- 2. Sanity-check the host config -------------------------------------
if [[ ! -f "$LINKS" ]]; then
    echo "[start] ERROR: $LINKS not found. Copy host/links.example.json and edit MACs."
    exit 1
fi
if [[ ! -f "$BASELINES" ]]; then
    echo "[start] WARNING: $BASELINES not found — heatmap will run in σ-normalized mode."
    echo "        Run 'python run.py calibrate-links udp:${PORT} --out $BASELINES' first,"
    echo "        or press [C] in the heatmap window to live-recalibrate."
    BASELINES_ARG=()
else
    BASELINES_ARG=(--baselines "$BASELINES")
fi

# --- 3. Activate the host venv if present --------------------------------
cd "$HOST_DIR"
if [[ -d "$HOST_DIR/.venv" ]]; then
    # shellcheck source=/dev/null
    source "$HOST_DIR/.venv/bin/activate"
fi

# --- 4. Optional: presence publisher in the background -------------------
PUBLISHER_PID=""
cleanup() {
    if [[ -n "${PUBLISHER_PID:-}" ]] && kill -0 "$PUBLISHER_PID" 2>/dev/null; then
        echo "[start] stopping publisher (pid $PUBLISHER_PID)..."
        kill "$PUBLISHER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ "${1:-}" == "publish" ]]; then
    echo "[start] launching presence publisher → ${C5_ADDR}..."
    python run.py publish "udp:${PORT}" \
        --links "$LINKS" "${BASELINES_ARG[@]}" \
        --c5-addr "$C5_ADDR" &
    PUBLISHER_PID=$!
    sleep 1
fi

# --- 5. Heatmap viewer (foreground) --------------------------------------
echo "[start] launching heatmap viewer..."
python run.py heatmap "udp:${PORT}" \
    --links "$LINKS" "${BASELINES_ARG[@]}"
