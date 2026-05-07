#!/usr/bin/env bash
# CSIDetector hotspot watchdog: single-shot check, run on a systemd timer.
#
# Exits 0 if the named connection is already activated. Otherwise calls
# `nmcli connection up` to bring it back, logs to syslog under the
# `csidetector-watchdog` tag, and returns nonzero if the revive failed.
#
# Designed to be invoked by csidetector-hotspot.timer (every ~30s).
#
# Usage:
#   ./scripts/hotspot-watchdog.sh
#   HOTSPOT_NAME=MyOtherAP ./scripts/hotspot-watchdog.sh

set -euo pipefail

HOTSPOT_NAME="${HOTSPOT_NAME:-CSIDetector}"

if ! command -v nmcli >/dev/null 2>&1; then
    logger -t csidetector-watchdog -p user.err \
        "nmcli not found; cannot manage hotspot"
    exit 2
fi

if nmcli -t -f NAME,STATE connection show --active 2>/dev/null \
        | grep -qx "${HOTSPOT_NAME}:activated"; then
    # Already up — nothing to do.
    exit 0
fi

logger -t csidetector-watchdog -p user.notice \
    "hotspot '${HOTSPOT_NAME}' is down; bringing it up"
if nmcli connection up "$HOTSPOT_NAME" >/dev/null 2>&1; then
    logger -t csidetector-watchdog -p user.notice \
        "hotspot '${HOTSPOT_NAME}' restored"
    exit 0
else
    logger -t csidetector-watchdog -p user.err \
        "failed to bring '${HOTSPOT_NAME}' up; will retry next tick"
    exit 1
fi
