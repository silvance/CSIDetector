"""Run every GUI scenario headlessly (no screenshots, fast fake tools).

The scenarios drive the real setup GUI through tests/gui_driver.py; this
keeps them — and the behaviour they check — from silently rotting.
For screenshots and step-by-step reports, run them via the CLI instead:
``python -m tests.gui_driver <scenario> --out DIR``.
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("PyQt6.QtWidgets")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


from tests.gui_driver import DriverError, Session  # noqa: E402
from tests.gui_scenarios import SCENARIOS  # noqa: E402


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario(name):
    with Session(None, screenshots=False, line_delay=0.001, name=name,
                 fast_timers=True, calibrate_s=(0.3, 2.0)) as s:
        try:
            SCENARIOS[name](s)
        except DriverError as exc:
            pytest.fail(f"scenario couldn't run: {exc}")
    assert not s.exceptions, "\n\n".join(s.exceptions)
    assert not s.failed_expectations, "\n".join(s.failed_expectations)
    assert not s.qt_warnings, "\n".join(s.qt_warnings)
