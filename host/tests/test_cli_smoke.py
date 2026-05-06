"""CLI smoke: every --help variant reaches argparse and exits 0.

Argparse's ``--help`` raises ``SystemExit(0)`` after printing the help
text, which is the contract we rely on here. If a subcommand's parser
fails to register or a back-compat alias is missing, ``parse_args``
raises ``SystemExit(2)`` instead — caught and re-raised as a test
failure with a useful message.

These tests do **not** open any source, hardware, or network. They
exercise the parser tree only.
"""

from __future__ import annotations

import io
import contextlib

import pytest

from csidetector.cli import build_parser


HELP_PATHS = [
    # Top-level
    ["--help"],
    # Mode dispatchers
    ["localize", "--help"],
    ["alert", "--help"],
    # Localize subcommands (back-compat + canonical)
    ["heatmap", "--help"],
    ["localize", "heatmap", "--help"],
    ["view3d", "--help"],
    ["localize", "view3d", "--help"],
    ["calibrate-links", "--help"],
    ["localize", "calibrate-links", "--help"],
    ["publish", "--help"],
    ["localize", "publish", "--help"],
    # Alert subcommands (back-compat + canonical)
    ["calibrate", "--help"],
    ["alert", "calibrate", "--help"],
    ["detect", "--help"],
    ["alert", "detect", "--help"],
    ["view", "--help"],
    ["alert", "view-waterfall", "--help"],
    # Shared
    ["capture", "--help"],
]


@pytest.mark.parametrize("argv", HELP_PATHS,
                         ids=[" ".join(a) for a in HELP_PATHS])
def test_help_exits_zero(argv):
    p = build_parser()
    # Suppress argparse's stdout so the test output stays readable.
    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(SystemExit) as excinfo:
            p.parse_args(argv)
    assert excinfo.value.code == 0, (
        f"`{' '.join(argv)}` exited with {excinfo.value.code}; "
        "subcommand registration probably regressed.")


def test_top_level_subcommands_registered():
    """The full menu of top-level commands is reachable."""
    p = build_parser()
    actions = {a.dest: a for a in p._actions}
    # The first subparsers action holds the choices map.
    sub = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction")
    expected = {
        # Mode dispatchers
        "localize", "alert",
        # Back-compat aliases (must keep working)
        "heatmap", "view3d", "calibrate-links", "publish",
        "calibrate", "detect", "view",
        # Shared
        "capture",
    }
    missing = expected - set(sub.choices)
    assert not missing, f"missing top-level subcommands: {missing}"
