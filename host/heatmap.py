"""Back-compat shim — module relocated to ``csidetector.modes.localize.heatmap``.

New code should import from ``csidetector.modes.localize.heatmap``.
"""

from csidetector.modes.localize.heatmap import *  # noqa: F401, F403
# Private names that other modules import (publisher, etc.).
from csidetector.modes.localize.heatmap import (  # noqa: F401
    _Node, _LinkBuffer, _load_links, _load_baselines,
    _save_baselines_envelope, _reader_thread,
    RATIO_FLOOR, DEFAULT_RATIO_FULL_BRIGHT,
    SPARKLINE_HISTORY_LEN, PKT_RATE_WINDOW_S, BADGE_FLASH_DURATION_S,
    TX_LINESTYLES, MIN_LINK_HZ_FLOOR, MIN_LINK_HZ_FRAC,
    CALIB_SETTLE_S, CALIB_RECORD_S, CALIB_DONE_FLASH_S,
)
