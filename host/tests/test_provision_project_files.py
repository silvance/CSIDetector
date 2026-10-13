"""alert.toml / links.json writers used by the setup wizard."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from csidetector.modes.alert.config import load_config
from csidetector.modes.localize.heatmap import _load_links
from csidetector.provision import project_files as pf

TOKEN = "123456789:" + "A" * 35
EXAMPLE_LINKS = Path(__file__).resolve().parents[1] / "links.example.json"


def test_validate_telegram():
    assert pf.validate_telegram(TOKEN, "8722080863") == []
    assert pf.validate_telegram(TOKEN, "-1001234567890") == []
    assert pf.validate_telegram(TOKEN, "@my_channel") == []
    assert len(pf.validate_telegram("nope", "x")) == 2


def test_write_alert_toml_is_owner_only_and_loadable(tmp_path):
    path = tmp_path / "alert.toml"
    assert pf.write_alert_toml(path, TOKEN, "8722080863", 'Lab "A"') is None
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    cfg = load_config(str(path))
    assert cfg["notifier"] == {"type": "telegram", "bot_token": TOKEN,
                               "chat_id": "8722080863"}
    assert cfg["alert"]["location_label"] == 'Lab "A"'
    assert cfg["queue"]["enabled"] is True


def test_write_alert_toml_carries_over_preferences_and_backs_up(tmp_path):
    path = tmp_path / "alert.toml"
    path.write_text('[alert]\ncooldown_s = 15\nclear_on_exit = true\n'
                    '[queue]\npath = "/var/lib/q.db"\n')
    backup = pf.write_alert_toml(path, TOKEN, "8722080863")
    assert backup is not None and "cooldown_s = 15" in backup.read_text()
    cfg = load_config(str(path))
    assert cfg["alert"]["cooldown_s"] == 15
    assert cfg["alert"]["clear_on_exit"] is True
    assert cfg["queue"]["path"] == "/var/lib/q.db"


def test_toml_string_escaping_survives_hostile_label(tmp_path):
    path = tmp_path / "alert.toml"
    label = 'x"\n[notifier]\nbot_token = "stolen'
    pf.write_alert_toml(path, TOKEN, "8722080863", label)
    cfg = load_config(str(path))
    assert cfg["alert"]["location_label"] == label
    assert cfg["notifier"]["bot_token"] == TOKEN


def test_merge_links_preserves_existing_layout():
    existing = json.loads(EXAMPLE_LINKS.read_text())
    data = pf.merge_links(
        existing,
        txs=[("ac:a7:04:2c:42:54", "ignored"), ("A0:F2:62:F0:39:D8", "Desk TX")],
        rxs=[("AC-A7-04-2C-24-CC", "ignored"), ("a0f262f03928", "")])
    assert data["room"] == existing["room"]
    assert data["_comment"] == existing["_comment"]
    south = next(n for n in data["rxs"] if n["mac"] == "ac:a7:04:2c:24:cc")
    assert (south["label"], south["x"], south["y"]) == ("south", 2.5, 0.0)
    assert "placed" not in south
    new_tx = data["txs"][-1]
    assert new_tx["mac"] == "a0:f2:62:f0:39:d8" and new_tx["label"] == "Desk TX"
    assert new_tx["placed"] is False
    assert data["rxs"][-1]["label"] == "RX5"
    assert len(data["txs"]) == 3 and len(data["rxs"]) == 5


def test_merge_links_converts_legacy_single_tx():
    data = pf.merge_links(
        {"room": {"width_m": 4, "height_m": 3},
         "tx": {"mac": "aa:bb:cc:dd:ee:ff", "x": 1, "y": 1}, "rxs": []},
        txs=[], rxs=[("11:22:33:44:55:66", "R")])
    assert "tx" not in data
    assert data["txs"] == [{"mac": "aa:bb:cc:dd:ee:ff", "x": 1, "y": 1}]
    assert data["rxs"][0]["placed"] is False


def test_new_nodes_land_on_room_perimeter():
    data = pf.merge_links(None, txs=[("aa:bb:cc:dd:ee:01", "")],
                          rxs=[(f"aa:bb:cc:dd:ee:{i:02x}", "") for i in range(2, 6)])
    for node in data["txs"] + data["rxs"]:
        x, y = node["x"], node["y"]
        assert 0 <= x <= 5 and 0 <= y <= 4
        assert x in (0, 5) or y in (0, 4)


def test_write_links_round_trips_through_viewer_loader(tmp_path):
    path = tmp_path / "links.json"
    path.write_text(EXAMPLE_LINKS.read_text())
    data, backup = pf.write_links(path, txs=[("a0:f2:62:f0:39:d8", "Desk TX")],
                                  rxs=[])
    assert backup is not None
    cfg, txs, rxs = _load_links(str(path))
    assert [t.label for t in txs] == ["TX1", "TX2", "Desk TX"]
    assert len(rxs) == 4


def test_write_links_refuses_to_clobber_invalid_json(tmp_path):
    path = tmp_path / "links.json"
    path.write_text("{hand-edited, broken")
    with pytest.raises(ValueError):
        pf.write_links(path, txs=[], rxs=[])
    assert path.read_text() == "{hand-edited, broken"


def test_backups_are_owner_only_and_never_overwrite_each_other(tmp_path):
    path = tmp_path / "alert.toml"
    path.write_text('[notifier]\nbot_token = "old"\n')
    os.chmod(path, 0o644)                    # world-readable before the wizard
    first = pf.write_alert_toml(path, TOKEN, "8722080863")
    second = pf.write_alert_toml(path, TOKEN, "8722080863")   # same second
    assert first != second
    assert 'bot_token = "old"' in first.read_text()           # not clobbered
    for p in (first, second, path):
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600


@pytest.mark.parametrize("bad", [
    [],
    {"room": None},
    {"room": {"polygon": [[0, 0], [5, 0], [5]]}},
    {"room": {"polygon": [[0, 0], [5, 0]]}},
    {"room": {"width_m": -1}},
    {"txs": {"mac": "aa:bb:cc:dd:ee:ff", "x": 0, "y": 0}},
    {"txs": [{"mac": 12, "x": 0, "y": 0}]},
    {"rxs": [{"mac": "not-a-mac", "x": 0, "y": 0}]},
    {"rxs": [{"mac": "aa:bb:cc:dd:ee:ff", "x": "left", "y": 0}]},
    {"rxs": [{"mac": "aa:bb:cc:dd:ee:ff", "x": float("nan"), "y": 0}]},
    {"room": {"width_m": 5}},
    {"tx": "aa:bb:cc:dd:ee:ff"},
])
def test_malformed_links_are_refused_and_left_alone(tmp_path, bad):
    with pytest.raises(ValueError):
        pf.validate_links_shape(bad)
    path = tmp_path / "links.json"
    text = json.dumps(bad)
    path.write_text(text)
    with pytest.raises(ValueError):
        pf.write_links(path, txs=[("aa:bb:cc:dd:ee:01", "")], rxs=[])
    assert path.read_text() == text
    assert list(tmp_path.iterdir()) == [path]          # no backup, no temp file


def test_example_links_pass_validation():
    pf.validate_links_shape(json.loads(EXAMPLE_LINKS.read_text()))


def test_reflashed_board_moves_role_and_keeps_its_position():
    existing = {"room": {"width_m": 5, "height_m": 4},
                "txs": [{"mac": "aa:bb:cc:dd:ee:01", "x": 1.0, "y": 2.0, "label": "old"}],
                "rxs": []}
    data = pf.merge_links(existing, txs=[], rxs=[("AA:BB:CC:DD:EE:01", "Hall RX")])
    assert data["txs"] == []
    assert data["rxs"] == [{"mac": "aa:bb:cc:dd:ee:01", "x": 1.0, "y": 2.0,
                            "label": "Hall RX"}]


def test_mac_in_both_roles_is_rejected():
    with pytest.raises(ValueError, match="both TX and RX"):
        pf.merge_links(None, txs=[("aa:bb:cc:dd:ee:01", "")],
                       rxs=[("aa-bb-cc-dd-ee-01", "")])


def test_placeholders_avoid_existing_nodes_and_each_other():
    # Example layout: nodes on the perimeter, including the corners where
    # evenly spaced placeholders used to land.
    existing = json.loads(EXAMPLE_LINKS.read_text())
    placed = [(n["x"], n["y"]) for n in existing["txs"] + existing["rxs"]]
    data = pf.merge_links(existing, txs=[],
                          rxs=[(f"aa:bb:cc:dd:ee:{i:02x}", "") for i in range(1, 4)])
    new = [(n["x"], n["y"]) for n in data["rxs"] if n.get("placed") is False]
    assert len(new) == 3 and len(set(new)) == 3
    for x, y in new:
        assert min((x - px) ** 2 + (y - py) ** 2 for px, py in placed) > 0.25


def test_numeric_strings_are_accepted_like_the_viewers_do(tmp_path):
    """heatmap/view3d call float() on these, so "0.3" is a working file."""
    path = tmp_path / "links.json"
    path.write_text(json.dumps({
        "room": {"width_m": "5", "height_m": "4"},
        "txs": [{"mac": "aa:bb:cc:dd:ee:01", "x": "0.3", "y": "1"}], "rxs": []}))
    pf.write_links(path, txs=[], rxs=[("aa:bb:cc:dd:ee:02", "R")])
    cfg, txs, rxs = _load_links(str(path))
    assert (txs[0].x, txs[0].y) == (0.3, 1.0) and len(rxs) == 1
