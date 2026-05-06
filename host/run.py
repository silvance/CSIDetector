"""Entry point. Dispatches into ``csidetector.cli``.

The CLI subcommand tree lives in the package; this file just bootstraps
it so ``python run.py …`` continues to work the way it always has.

New canonical form::

    python run.py localize heatmap udp:5566 --links links.json …
    python run.py alert    detect /dev/ttyUSB0 --baseline 0.42 …

The pre-refactor flat names (``heatmap``, ``view3d``, ``calibrate``,
``detect``, …) remain registered at the top level as back-compat
aliases.
"""

from __future__ import annotations

import sys

from csidetector.cli import main


if __name__ == "__main__":
    sys.exit(main())
