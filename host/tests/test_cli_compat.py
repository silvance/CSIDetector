"""Backward-compat: old + new command names dispatch to the same handler.

The two-mode refactor introduced ``localize <cmd>`` and ``alert <cmd>``
as the canonical forms while keeping the pre-refactor flat names as
top-level aliases. These tests pin that contract — if a future PR
moves a handler without updating both registration paths, these
catches it.
"""

from __future__ import annotations

import pytest

from csidetector.cli import build_parser


# (back-compat argv tail, canonical argv tail). Each pair must reach
# the same `args.func` and parse identical positional/optional values.
EQUIVALENT_FORMS = [
    # Localize family
    (["heatmap", "udp:5566", "--links", "l.json", "--baselines", "b.json"],
     ["localize", "heatmap", "udp:5566", "--links", "l.json", "--baselines", "b.json"]),
    (["view3d", "udp:5566", "--links", "l.json"],
     ["localize", "view3d", "udp:5566", "--links", "l.json"]),
    (["calibrate-links", "udp:5566", "--out", "b.json"],
     ["localize", "calibrate-links", "udp:5566", "--out", "b.json"]),
    (["publish", "udp:5566", "--links", "l.json", "--c5-addr", "10.42.0.255"],
     ["localize", "publish", "udp:5566", "--links", "l.json", "--c5-addr", "10.42.0.255"]),
    # Alert family
    (["calibrate", "udp:5566", "--seconds", "5"],
     ["alert", "calibrate", "udp:5566", "--seconds", "5"]),
    (["detect", "/dev/ttyUSB0", "--baseline", "0.1"],
     ["alert", "detect", "/dev/ttyUSB0", "--baseline", "0.1"]),
    (["view", "udp:5566"],
     ["alert", "view-waterfall", "udp:5566"]),
]


@pytest.mark.parametrize("legacy,canonical", EQUIVALENT_FORMS,
                         ids=[" ".join(a) for a, _ in EQUIVALENT_FORMS])
def test_legacy_and_canonical_share_handler(legacy, canonical):
    p = build_parser()
    a_old = p.parse_args(legacy)
    a_new = p.parse_args(canonical)
    assert a_old.func is a_new.func, (
        f"{legacy} → {a_old.func.__name__} vs "
        f"{canonical} → {a_new.func.__name__}: handler drift")


def test_detect_without_alert_config_preserves_legacy_behavior():
    """Old `detect` form must not require any new alert flags."""
    p = build_parser()
    args = p.parse_args(["detect", "/dev/ttyUSB0", "--baseline", "0.1"])
    # Legacy stdout-only mode: alert-config defaults to None (no notifier).
    assert args.alert_config is None
    # Tri-state bool: None means 'inherit from config'; with no config,
    # cmd_detect treats it as False.
    assert args.clear_on_exit is None
    assert args.cooldown_s is None
    assert args.location is None


def test_alert_detect_accepts_new_flags():
    p = build_parser()
    args = p.parse_args([
        "alert", "detect", "/dev/ttyUSB0", "--baseline", "0.1",
        "--alert-config", "alert.toml",
        "--cooldown-s", "30",
        "--clear-on-exit",
        "--location", "Office",
    ])
    assert args.alert_config == "alert.toml"
    assert args.cooldown_s == 30.0
    assert args.clear_on_exit is True
    assert args.location == "Office"


def test_heatmap_args_identical_between_forms():
    """Argument values land in the same namespace fields regardless of
    which form was used. (Catches drift where the two registrations
    accidentally use different ``add_argument`` flags / dests.)"""
    p = build_parser()
    legacy = p.parse_args([
        "heatmap", "udp:5566", "--links", "l.json", "--baselines", "b.json",
        "--motion-quantile", "0.8", "--motion-max-enter", "4.0",
    ])
    canonical = p.parse_args([
        "localize", "heatmap", "udp:5566", "--links", "l.json", "--baselines", "b.json",
        "--motion-quantile", "0.8", "--motion-max-enter", "4.0",
    ])
    for field in ("source", "links", "baselines", "motion_quantile",
                  "motion_max_enter", "motion_enter", "motion_exit"):
        assert getattr(legacy, field) == getattr(canonical, field), (
            f"field {field!r} differs between forms")
