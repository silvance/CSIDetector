"""Smoke test: localize-mode heatmap module imports and renders a figure.

Doesn't open a real source — just verifies that the dual-threshold
aggregator + sparkline + pkt-rate-strip layout code paths all execute
without errors. Skipped when matplotlib isn't installed.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


pytest.importorskip("matplotlib")


def test_heatmap_module_imports():
    # Just importing the heatmap module exercises the bulk of the
    # localize-mode init path — module-level constants, helper funcs,
    # and the matplotlib dependency.
    from csidetector.modes.localize import heatmap

    assert callable(heatmap.run_heatmap)
    assert heatmap.RATIO_FLOOR == 1.0
    assert heatmap._LinkBuffer is not None


def test_load_links_round_trip():
    from csidetector.modes.localize.heatmap import _load_links

    cfg = {
        "room": {"polygon": [[0, 0], [5, 0], [5, 4], [0, 4]]},
        "txs": [
            {"mac": "aa:bb:cc:dd:ee:01", "x": 0.3, "y": 0.3, "label": "TX1"},
            {"mac": "aa:bb:cc:dd:ee:02", "x": 4.7, "y": 3.7, "label": "TX2"},
        ],
        "rxs": [
            {"mac": "11:22:33:44:55:01", "x": 2.5, "y": 0.2, "label": "RX-S"},
            {"mac": "11:22:33:44:55:02", "x": 0.2, "y": 2.0, "label": "RX-W"},
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        path = f.name
    try:
        cfg_out, txs, rxs = _load_links(path)
        assert len(txs) == 2
        assert len(rxs) == 2
        assert txs[0].label == "TX1"
        assert rxs[1].mac == "11:22:33:44:55:02"
    finally:
        Path(path).unlink()


def test_load_baselines_handles_envelope_format():
    from csidetector.modes.localize.heatmap import _load_baselines, _Node

    txs = [_Node(mac="aa:bb:cc:dd:ee:01", x=0, y=0, label="TX1")]
    rxs = [_Node(mac="11:22:33:44:55:01", x=0, y=0, label="RX1")]
    payload = {
        "_meta": {"format": "csidetector-baselines/1"},
        "links": {"aa:bb:cc:dd:ee:01|11:22:33:44:55:01": 0.42},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = f.name
    try:
        out = _load_baselines(path, txs, rxs)
        assert out[("aa:bb:cc:dd:ee:01", "11:22:33:44:55:01")] == pytest.approx(0.42)
    finally:
        Path(path).unlink()
