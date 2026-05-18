"""Load-error tests for the localize-mode config + baseline files.

These exercise pure-Python validation paths in
``csidetector.modes.localize.heatmap`` that previously raised obscure
``KeyError`` / ``TypeError`` / ``JSONDecodeError`` exceptions. They now
``SystemExit`` with a message that names the file and the missing piece.

These tests don't need matplotlib (heatmap.py imports it lazily inside
``run_heatmap``), so they run in environments where the smoke-test
file is skipped.
"""

from __future__ import annotations

import json

import pytest

from csidetector.modes.localize.heatmap import (
    _Node, _load_baselines, _load_links,
)


def _write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f)


# --------------------------------------------------------------------------
# _load_links — missing top-level keys now SystemExit with context.
# --------------------------------------------------------------------------

def test_load_links_missing_rxs_raises_systemexit(tmp_path):
    p = tmp_path / "links.json"
    _write_json(p, {"room": {"polygon": [[0, 0]]},
                    "txs": [{"mac": "aa:bb:cc:dd:ee:01", "x": 0, "y": 0}]})
    with pytest.raises(SystemExit, match="rxs"):
        _load_links(str(p))


def test_load_links_missing_room_raises_systemexit(tmp_path):
    p = tmp_path / "links.json"
    _write_json(p, {"rxs": [], "txs": [{"mac": "aa:bb:cc:dd:ee:01",
                                         "x": 0, "y": 0}]})
    with pytest.raises(SystemExit, match="room"):
        _load_links(str(p))


def test_load_links_missing_txs_raises_systemexit(tmp_path):
    p = tmp_path / "links.json"
    _write_json(p, {"room": {"polygon": [[0, 0]]}, "rxs": []})
    with pytest.raises(SystemExit, match="txs"):
        _load_links(str(p))


def test_load_links_malformed_json_raises_systemexit(tmp_path):
    p = tmp_path / "links.json"
    p.write_text("not even close to json {{{")
    with pytest.raises(SystemExit, match="malformed JSON"):
        _load_links(str(p))


def test_load_links_missing_file_raises_systemexit(tmp_path):
    with pytest.raises(SystemExit):
        _load_links(str(tmp_path / "does-not-exist.json"))


# --------------------------------------------------------------------------
# _load_baselines — the real fix: envelope-without-`links` crash.
# --------------------------------------------------------------------------

def test_load_baselines_envelope_without_links_raises(tmp_path):
    """The bug: file has _meta block but the links map is truncated or
    missing. Old code fell into the "flat" branch, tried to coerce the
    metadata dict to a float, and produced a confusing TypeError."""
    p = tmp_path / "baselines.json"
    _write_json(p, {"_meta": {"format": "csidetector-baselines/1"}})   # no links
    with pytest.raises(SystemExit, match="missing 'links'"):
        _load_baselines(str(p), [_Node("aa", 0, 0, "TX")],
                        [_Node("bb", 0, 0, "RX")])


def test_load_baselines_envelope_with_non_dict_links_raises(tmp_path):
    p = tmp_path / "baselines.json"
    _write_json(p, {"_meta": {"format": "csidetector-baselines/1"},
                    "links": "not an object"})
    with pytest.raises(SystemExit, match="missing 'links'"):
        _load_baselines(str(p), [_Node("aa", 0, 0, "TX")],
                        [_Node("bb", 0, 0, "RX")])


def test_load_baselines_malformed_json_raises_systemexit(tmp_path):
    p = tmp_path / "baselines.json"
    p.write_text("{not valid json")
    with pytest.raises(SystemExit, match="malformed JSON"):
        _load_baselines(str(p), [_Node("aa", 0, 0, "TX")],
                        [_Node("bb", 0, 0, "RX")])


def test_load_baselines_top_level_not_object_raises(tmp_path):
    p = tmp_path / "baselines.json"
    _write_json(p, [1, 2, 3])   # JSON array, not object
    with pytest.raises(SystemExit, match="must be an object"):
        _load_baselines(str(p), [_Node("aa", 0, 0, "TX")],
                        [_Node("bb", 0, 0, "RX")])


def test_load_baselines_envelope_happy_path(tmp_path):
    p = tmp_path / "baselines.json"
    _write_json(p, {
        "_meta": {"format": "csidetector-baselines/1"},
        "links": {"aa|bb": 0.42},
    })
    out = _load_baselines(str(p), [_Node("aa", 0, 0, "TX")],
                          [_Node("bb", 0, 0, "RX")])
    assert out[("aa", "bb")] == pytest.approx(0.42)


def test_load_baselines_none_path_returns_empty():
    assert _load_baselines(None, [], []) == {}
