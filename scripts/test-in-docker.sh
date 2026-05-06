#!/usr/bin/env bash
# Run the full host-side test suite in a clean container.
#
# Usage:
#   ./scripts/test-in-docker.sh                  # all layers
#   ./scripts/test-in-docker.sh --rebuild        # force a fresh build
#   ./scripts/test-in-docker.sh --integration    # also run @pytest.mark.integration
#
# Exits nonzero on any failure. Prints clear section headers so output
# is scannable even with verbose pytest output.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-csidetector-test}"
DOCKERFILE="$REPO_ROOT/Dockerfile.test"

REBUILD=0
RUN_INTEGRATION=0
for arg in "$@"; do
    case "$arg" in
        --rebuild)     REBUILD=1 ;;
        --integration) RUN_INTEGRATION=1 ;;
        -h|--help)
            sed -n '2,11p' "$0"
            exit 0
            ;;
        *)
            echo "unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

heading() {
    printf '\n\033[1;36m== %s ==\033[0m\n' "$1"
}

# --------------------------------------------------------------------------
# Build (cache-friendly; skip unless --rebuild or no image yet).
# --------------------------------------------------------------------------
if [[ "$REBUILD" -eq 1 ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    heading "Building image ($IMAGE)"
    docker build -f "$DOCKERFILE" -t "$IMAGE" "$REPO_ROOT"
else
    echo "[test-in-docker] image '$IMAGE' already built; pass --rebuild to refresh"
fi

# --------------------------------------------------------------------------
# Run the layered test plan in a single container invocation. Each layer
# is a pytest selection so failures are obvious in the test report.
# --------------------------------------------------------------------------
PYTEST_BASE="python -m pytest -v --tb=short"
PYTEST_FILTER="-m 'not integration'"
if [[ "$RUN_INTEGRATION" -eq 1 ]]; then
    PYTEST_FILTER=""
fi

# We run pytest once with the full tests/ tree — pytest groups output by
# file, which already gives us the "by-layer" view the prompt asks for.
# The headings below are mostly for the human reader; pytest itself
# enumerates the tests inside each file.
heading "Layer 1 — core (parsing, filters, motion)"
heading "Layer 2 — CLI smoke (--help variants, back-compat)"
heading "Layer 3 — alert config parsing"
heading "Layer 4 — alert queue (durability, idempotency, retries)"
heading "Layer 5 — localize smoke (figure setup, file formats)"

docker run --rm "$IMAGE" sh -c "$PYTEST_BASE tests/ $PYTEST_FILTER"
status=$?

if [[ "$status" -eq 0 ]]; then
    heading "ALL TESTS PASSED"
else
    heading "TESTS FAILED (exit $status)"
fi
exit "$status"
