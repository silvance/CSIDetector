#!/usr/bin/env bash
# Drop into a subshell with ESP-IDF activated, ready for idf.py commands.
#
# Usage:
#   ./idf-env.sh                     # bare shell, generic cheat-sheet
#   ./idf-env.sh csi_receiver        # cd into firmware/csi_receiver first
#   ./idf-env.sh csi_transmitter
#   ./idf-env.sh c5_display
#
# Env overrides (optional):
#   IDF_EXPORT=$HOME/esp/esp-idf/export.sh   path to your IDF's export.sh
#
# Type 'exit' (or Ctrl-D) to return to your normal shell.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDF_EXPORT="${IDF_EXPORT:-$HOME/esp/esp-idf/export.sh}"

if [[ ! -f "$IDF_EXPORT" ]]; then
    echo "ERROR: ESP-IDF export.sh not found at $IDF_EXPORT" >&2
    echo "       Set IDF_EXPORT=/path/to/esp-idf/export.sh and re-run." >&2
    exit 1
fi

# Resolve the optional project argument.
PROJECT="${1:-}"
PROJECT_DIR=""
TARGET=""
if [[ -n "$PROJECT" ]]; then
    PROJECT_DIR="$REPO_ROOT/firmware/$PROJECT"
    if [[ ! -d "$PROJECT_DIR" ]]; then
        echo "ERROR: firmware/$PROJECT doesn't exist. Available projects:" >&2
        ls "$REPO_ROOT/firmware/" 2>/dev/null | sed 's/^/  /' >&2
        exit 1
    fi
    if [[ -f "$PROJECT_DIR/sdkconfig.defaults" ]]; then
        TARGET=$(grep -oP '^CONFIG_IDF_TARGET="\K[^"]+' \
                    "$PROJECT_DIR/sdkconfig.defaults" 2>/dev/null || true)
    fi
fi

# What's plugged in right now? Helps when you have multiple boards.
PORTS=()
shopt -s nullglob
for p in /dev/ttyACM* /dev/ttyUSB*; do PORTS+=("$p"); done
shopt -u nullglob
PORTS_STR="${PORTS[*]:-(none detected — plug in a board?)}"

# Tailor the build/flash hints if a project was given.
if [[ -n "$PROJECT" ]]; then
    TARGET_LINE="${TARGET:-esp32s3}"
    cat <<HINTS
==================================================================
  ESP-IDF env for: ${PROJECT}  (target: ${TARGET_LINE})
==================================================================

Working directory will be: firmware/${PROJECT}

First time on this project, or after a target change:
  idf.py set-target ${TARGET_LINE}
  idf.py menuconfig          # if the project has Kconfig (Wi-Fi creds, etc.)

Build / flash / monitor:
  idf.py build
  idf.py -p \$PORT flash
  idf.py -p \$PORT monitor
  idf.py -p \$PORT flash monitor      # combined

Recover from a broken state:
  idf.py fullclean           # nuke build/
  idf.py erase-flash -p \$PORT

USB ports currently visible:
  ${PORTS_STR}

HINTS

    if [[ "$PROJECT" == "csi_receiver" ]]; then
        cat <<RXHINTS
Multi-RX flash loop (one board at a time, replug between flashes):
  for p in ${PORTS_STR}; do
      idf.py -p "\$p" flash || break
  done

(Or flash each board manually, easier when serial-numbers aren't stable
 across replug.)
RXHINTS
    fi
else
    cat <<HINTS
==================================================================
  ESP-IDF environment ready.  Generic idf.py cheat-sheet:
==================================================================

cd to a project first:
  cd firmware/csi_transmitter   # esp32s3
  cd firmware/csi_receiver      # esp32s3
  cd firmware/c5_display        # esp32c5

Then:
  idf.py set-target esp32s3     # one-time per project
  idf.py menuconfig             # Kconfig (Wi-Fi creds, channel, MAC filter)
  idf.py build
  idf.py -p \$PORT flash monitor
  idf.py fullclean              # nuke build/, recover from a target mismatch
  idf.py erase-flash -p \$PORT  # wipe flash before re-flashing

USB ports currently visible:
  ${PORTS_STR}

Tip: re-run this script with a project name as argument for tailored
     hints — e.g.  ./idf-env.sh csi_receiver

HINTS
fi

# --- Strip an active venv from PATH so its python doesn't shadow IDF's. --
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "[idf-env] removing venv from PATH ($VIRTUAL_ENV)"
    PATH=$(echo ":$PATH:" \
        | sed -e "s|:${VIRTUAL_ENV}/bin:|:|g" \
              -e 's|^:||' -e 's|:$||')
    export PATH
    unset VIRTUAL_ENV PYTHONHOME
fi

# --- Source IDF and drop into a subshell with a tagged prompt. ----------
# shellcheck source=/dev/null
source "$IDF_EXPORT"

if [[ -n "$PROJECT_DIR" ]]; then
    cd "$PROJECT_DIR"
fi

echo "[idf-env] entering IDF subshell. 'exit' or Ctrl-D to leave."
RCFILE=$(mktemp)
trap 'rm -f "$RCFILE"' EXIT
cat >"$RCFILE" <<'RC'
# Inherit user's interactive niceties if present, then tag the prompt.
[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"
PS1="(idf) ${PS1:-\\w \$ }"
RC
exec bash --rcfile "$RCFILE" -i
