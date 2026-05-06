"""Back-compat shim — module relocated to ``csidetector.core.collector``.

This shim exists so that existing imports (``import csi_collector``,
``from csi_collector import open_source``, etc.) continue to work after
the Phase 1 refactor. New code should import from
``csidetector.core.collector``.
"""

from csidetector.core.collector import *  # noqa: F401, F403
# Private/underscore names downstream callers occasionally reach for.
from csidetector.core.collector import (  # noqa: F401
    _UDP_HEADER, _UDP_IQ_MIN, _UDP_IQ_MAX,
)
