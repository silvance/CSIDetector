"""Back-compat shim — module relocated to ``csidetector.core.detector``.

New code should import from ``csidetector.core.detector``.
"""

from csidetector.core.detector import *  # noqa: F401, F403
from csidetector.core.detector import (  # noqa: F401
    AGC_SETTLE_SECONDS_DEFAULT,
)
