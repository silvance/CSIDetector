"""links.json model behind the floorplan editor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from csidetector.modes.localize.heatmap import _load_links
from csidetector.provision import layout as lo
from csidetector.provision.devices import ROLE_RX, ROLE_TX

EXAMPLE_LINKS = Path(__file__).resolve().parents[1] / "links.example.json"


def test_round_trip_keeps_everything_it_doesnt_edit(tmp_path):
    data = json.loads(EXAMPLE_LINKS.read_text())
    data["txs"][0]["height_m"] = 1.2                  # a key the editor doesn't know
    data["publisher"] = {"c5_addr": "10.42.0.255"}
    layout = lo.Layout.from_dict(data)
    out = layout.to_dict()
    assert out["_comment"] == data["_comment"]
    assert out["publisher"] == data["publisher"]
    assert out["room"]["polygon"] == data["room"]["polygon"]
    assert out["txs"][0] == data["txs"][0]
    assert [n["mac"] for n in out["rxs"]] == [n["mac"] for n in data["rxs"]]
    path = tmp_path / "links.json"
    layout.save(path)
    _cfg, txs, rxs = _load_links(str(path))           # the heatmap still loads it
    assert len(txs) == 2 and len(rxs) == 4


def test_rectangle_rooms_and_legacy_single_tx():
    layout = lo.Layout.from_dict({"room": {"width_m": 6, "height_m": 3},
                                  "tx": {"mac": "AA-BB-CC-DD-EE-01", "x": 1, "y": 1},
                                  "rxs": []})
    assert layout.corners == [(0, 0), (6, 0), (6, 3), (0, 3)]
    assert layout.nodes[0].role == ROLE_TX and layout.nodes[0].mac == "aa:bb:cc:dd:ee:01"
    out = layout.to_dict()
    assert "tx" not in out and len(out["txs"]) == 1


def test_moving_a_provisional_node_places_it():
    layout = lo.Layout.from_dict({"room": {"width_m": 5, "height_m": 4}, "txs": [],
                                  "rxs": [{"mac": "aa:bb:cc:dd:ee:02", "x": 0, "y": 0,
                                           "placed": False}]})
    node = layout.nodes[0]
    assert not node.placed and "placed" in node.to_dict()
    layout.move_node(node, 2.5, 1.0)
    assert node.placed and "placed" not in node.to_dict()
    assert (node.to_dict()["x"], node.to_dict()["y"]) == (2.5, 1.0)


def test_corner_edits():
    layout = lo.Layout()
    i = layout.insert_corner(1)                       # split the wall (5,0)→(5,4)
    assert layout.corners[i] == (5.0, 2.0) and len(layout.corners) == 5
    layout.move_corner(i, 6.0, 2.0)
    assert layout.corners[i] == (6.0, 2.0)
    assert layout.remove_corner(i)
    for _ in range(2):
        layout.remove_corner(0)
    assert len(layout.corners) == 3 and not layout.remove_corner(0)   # keeps a triangle
    with pytest.raises(ValueError):
        layout.set_rectangle(0, 3)


def test_add_node_lands_unplaced_in_the_room_and_rejects_duplicates():
    layout = lo.Layout()
    node = layout.add_node("aa:bb:cc:dd:ee:03", ROLE_RX, "Hall RX")
    assert (node.x, node.y) == pytest.approx((2.5, 2.0)) and not node.placed
    with pytest.raises(ValueError):
        layout.add_node("AA:BB:CC:DD:EE:03", ROLE_TX)


def test_problems_and_warnings():
    layout = lo.Layout(corners=[(0, 0), (4, 0), (0, 4), (4, 4)])     # bow tie
    assert any("crosses itself" in p for p in layout.problems())
    assert any("no receivers" in p for p in layout.problems())
    layout = lo.Layout()
    layout.add_node("aa:bb:cc:dd:ee:04", ROLE_TX)
    rx = layout.add_node("aa:bb:cc:dd:ee:05", ROLE_RX)
    assert layout.problems() == []
    assert any("Not placed" in w for w in layout.warnings())
    layout.move_node(rx, 9, 9)
    assert any("Outside the room" in w for w in layout.warnings())
    layout.move_node(rx, 5.02, 1.0)                    # on the wall: fine
    assert not any("Outside" in w for w in layout.warnings())


def test_floorplan_mapping_and_bad_values():
    fp = lo.Floorplan("plan.png", px_per_m=50, origin_px=(100, 400))
    assert fp.to_meters(*fp.to_pixels(3.0, 2.0)) == pytest.approx((3.0, 2.0))
    assert fp.to_pixels(0, 0) == (100, 400)
    assert lo.Floorplan.from_dict({"image": "x.png", "px_per_m": -1}) is None
    assert lo.Floorplan.from_dict({"image": "x.png", "origin_px": "nope"}) is None
    assert lo.Floorplan.from_dict(None) is None


def test_save_refuses_an_invalid_layout_and_keeps_the_file(tmp_path):
    path = tmp_path / "links.json"
    path.write_text(EXAMPLE_LINKS.read_text())
    layout = lo.Layout.load(path)
    layout.corners = [(0, 0), (1, 1)]                 # not a polygon
    with pytest.raises(ValueError):
        layout.save(path)
    assert path.read_text() == EXAMPLE_LINKS.read_text()


def test_load_missing_and_broken(tmp_path):
    assert lo.Layout.load(tmp_path / "none.json").nodes == []
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    with pytest.raises(ValueError):
        lo.Layout.load(bad)


def test_image_paths_relative_to_links(tmp_path):
    links = tmp_path / "host" / "links.json"
    links.parent.mkdir()
    img = links.parent / "plans" / "office.png"
    assert lo.relative_image_path(str(img), links) == "plans/office.png"
    assert lo.resolve_image_path("plans/office.png", links) == img
    outside = tmp_path / "elsewhere.png"
    assert lo.relative_image_path(str(outside), links) == str(outside)


def test_review_fixes(tmp_path):
    # Saving what the heatmap can't load is refused.
    layout = lo.Layout()
    layout.add_node("aa:bb:cc:dd:ee:10", ROLE_TX)
    with pytest.raises(ValueError, match="no receivers"):
        layout.save(tmp_path / "links.json")
    assert not (tmp_path / "links.json").exists()
    # Only transmitters and receivers can be in a layout.
    with pytest.raises(ValueError):
        layout.add_node("aa:bb:cc:dd:ee:11", "")
    # Room extras and a floorplan block we can't parse survive a round trip.
    data = {"room": {"width_m": 4, "height_m": 3, "_comment": "taped"},
            "txs": [{"mac": "aa:bb:cc:dd:ee:12", "x": 0, "y": 0}],
            "rxs": [{"mac": "aa:bb:cc:dd:ee:13", "x": 4, "y": 3}],
            "floorplan": {"image": "plan.png", "px_per_m": "120px"}}
    out = lo.Layout.from_dict(data).to_dict()
    assert out["room"]["_comment"] == "taped"
    assert out["floorplan"] == data["floorplan"]
    # A crossing through a corner counts.
    assert lo.self_intersects([(0, 4), (4, 0), (5, 5), (2, 2), (0, 0)])
    assert not lo.self_intersects([(0, 0), (5, 0), (5, 4), (2.5, 4), (2.5, 2.5), (0, 2.5)])


def test_save_writes_through_a_symlink(tmp_path):
    real = tmp_path / "rooms" / "office" / "links.json"
    real.parent.mkdir(parents=True)
    real.write_text(EXAMPLE_LINKS.read_text())
    link = tmp_path / "links.json"
    link.symlink_to(real)
    layout = lo.Layout.load(link)
    layout.move_node(layout.nodes[0], 1.0, 1.0)
    layout.save(link)
    assert link.is_symlink()
    assert json.loads(real.read_text())["txs"][0]["x"] == 1.0
