"""Room layout model for the floorplan editor: links.json in, links.json out.

links.json (see links.example.json) holds the room outline in meters,
the transmitters and receivers with their positions, and — written by
the editor — an optional floor-plan image underlay. The viewers read
only ``room``, ``txs``/``rxs`` (or the legacy single ``tx``); everything
else in the file is carried through untouched.

Coordinates are meters with y pointing up, as the heatmap draws them.
No Qt here: the editor widgets in floorplan.py drive this model, and the
tests drive it directly.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Optional

from csidetector.provision import project_files as pf
from csidetector.provision.devices import ROLE_RX, ROLE_TX, normalize_mac, try_normalize_mac

Point = tuple[float, float]

DEFAULT_SIZE = (5.0, 4.0)
MIN_CORNERS = 3


@dataclasses.dataclass
class Node:
    mac: str
    role: str                  # ROLE_TX / ROLE_RX
    x: float
    y: float
    label: str = ""
    placed: bool = True        # False: a provisional spot from the wizard
    extra: dict = dataclasses.field(default_factory=dict)   # other JSON keys

    def __post_init__(self) -> None:
        if self.role not in (ROLE_TX, ROLE_RX):
            raise ValueError(f"a board in the layout must be a transmitter or a "
                             f"receiver, not {self.role!r}")

    def to_dict(self) -> dict:
        out = dict(self.extra)
        out.update(mac=self.mac, x=round(self.x, 3), y=round(self.y, 3))
        if self.label:
            out["label"] = self.label
        else:
            out.pop("label", None)
        if self.placed:
            out.pop("placed", None)
        else:
            out["placed"] = False
        return out


@dataclasses.dataclass
class Floorplan:
    """A floor-plan image under the room, and how it maps to meters.

    Image pixel (u, v) — v pointing down, as in the file — sits at
    ``x = (u - origin_px[0]) / px_per_m``, ``y = (origin_px[1] - v) / px_per_m``,
    i.e. ``origin_px`` is the pixel at (0, 0) m.
    """
    image: str
    px_per_m: float = 100.0
    origin_px: Point = (0.0, 0.0)
    opacity: float = 0.6

    def to_meters(self, u: float, v: float) -> Point:
        return ((u - self.origin_px[0]) / self.px_per_m,
                (self.origin_px[1] - v) / self.px_per_m)

    def to_pixels(self, x: float, y: float) -> Point:
        return (self.origin_px[0] + x * self.px_per_m,
                self.origin_px[1] - y * self.px_per_m)

    def valid(self) -> bool:
        return (math.isfinite(self.px_per_m) and self.px_per_m > 0
                and all(math.isfinite(c) for c in self.origin_px))

    def to_dict(self) -> dict:
        if not self.valid():
            raise ValueError("the floor plan's scale or position is invalid; "
                             "set the scale again")
        return {"image": self.image, "px_per_m": round(self.px_per_m, 4),
                "origin_px": [round(self.origin_px[0], 2), round(self.origin_px[1], 2)],
                "opacity": round(self.opacity, 2)}

    @classmethod
    def from_dict(cls, data) -> Optional["Floorplan"]:
        """None for anything unusable (the editor just shows no image)."""
        if not isinstance(data, dict) or not isinstance(data.get("image"), str):
            return None
        try:
            ppm = float(data.get("px_per_m", 100.0))
            ox, oy = (float(c) for c in data.get("origin_px", (0.0, 0.0)))
            opacity = float(data.get("opacity", 0.6))
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(ppm) and ppm > 0 and math.isfinite(ox) and math.isfinite(oy)):
            return None
        return cls(data["image"], ppm, (ox, oy), min(1.0, max(0.05, opacity)))


class Layout:
    """The editable contents of one links.json."""

    def __init__(self, corners: Optional[list[Point]] = None,
                 nodes: Optional[list[Node]] = None,
                 floorplan: Optional[Floorplan] = None,
                 extra: Optional[dict] = None,
                 room_extra: Optional[dict] = None):
        w, h = DEFAULT_SIZE
        self.corners: list[Point] = list(corners or [(0, 0), (w, 0), (w, h), (0, h)])
        self.nodes: list[Node] = list(nodes or [])
        self.floorplan = floorplan
        self.extra: dict = dict(extra or {})     # top-level keys we don't edit
        self.room_extra: dict = dict(room_extra or {})   # e.g. a "_comment" in room

    # -- (de)serialisation -------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "Layout":
        """Raises ValueError (via validate_links_shape) for an unusable file."""
        pf.validate_links_shape(data)
        data = copy.deepcopy(data)
        room = data.pop("room", None) or {}
        room_extra = {k: v for k, v in room.items()
                      if k not in ("polygon", "width_m", "height_m")}
        if "polygon" in room:
            corners = [(float(x), float(y)) for x, y in room["polygon"]]
        elif "width_m" in room:
            w, h = float(room["width_m"]), float(room["height_m"])
            corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
        else:
            corners = None
        txs = data.pop("txs", None)
        legacy = data.pop("tx", None)
        if txs is None and isinstance(legacy, dict):
            txs = [legacy]
        rxs = data.pop("rxs", None) or []
        nodes = [cls._node(n, ROLE_TX) for n in txs or []]
        nodes += [cls._node(n, ROLE_RX) for n in rxs]
        floorplan = Floorplan.from_dict(data.get("floorplan"))
        if floorplan is not None:
            del data["floorplan"]        # else it stays in extra, untouched
        return cls(corners, nodes, floorplan, data, room_extra)

    @staticmethod
    def _node(raw: dict, role: str) -> Node:
        extra = {k: v for k, v in raw.items()
                 if k not in ("mac", "x", "y", "label", "placed")}
        return Node(mac=normalize_mac(raw["mac"]), role=role,
                    x=float(raw["x"]), y=float(raw["y"]),
                    label=str(raw.get("label", "") or ""),
                    placed=raw.get("placed", True) is not False, extra=extra)

    def to_dict(self) -> dict:
        out = dict(self.extra)
        out["room"] = dict(self.room_extra)
        out["room"]["polygon"] = [[round(x, 3), round(y, 3)] for x, y in self.corners]
        out["txs"] = [n.to_dict() for n in self.nodes if n.role == ROLE_TX]
        out["rxs"] = [n.to_dict() for n in self.nodes if n.role == ROLE_RX]
        if self.floorplan is not None:
            out["floorplan"] = self.floorplan.to_dict()
        return out

    @classmethod
    def load(cls, path: Path) -> "Layout":
        """A missing file is an empty default room; anything unreadable
        raises ValueError so the caller can say so instead of clobbering it."""
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path} isn't valid JSON ({exc})") from exc
        return cls.from_dict(data)

    def save(self, path: Path) -> Optional[Path]:
        """Validate and write (keeping a backup). Returns the backup path.
        Raises ValueError, writing nothing, if problems() isn't empty: the
        heatmap can't use a layout without a TX and an RX, say."""
        problems = self.problems()
        if problems:
            raise ValueError(" ".join(problems))
        return pf.save_links(Path(path), self.to_dict())

    # -- queries -------------------------------------------------------------

    def bbox(self) -> tuple[float, float, float, float]:
        xs = [x for x, _ in self.corners] + [n.x for n in self.nodes]
        ys = [y for _, y in self.corners] + [n.y for n in self.nodes]
        return min(xs), min(ys), max(xs), max(ys)

    def find(self, mac: str) -> Optional[Node]:
        mac = try_normalize_mac(mac)
        return next((n for n in self.nodes if n.mac == mac), None)

    def problems(self) -> list[str]:
        """Things the viewers would get wrong or choke on — shown before
        saving rather than discovered in the heatmap."""
        out = []
        if len(self.corners) < MIN_CORNERS:
            out.append(f"The room needs at least {MIN_CORNERS} corners.")
        elif self_intersects(self.corners):
            out.append("The room's outline crosses itself; list the corners in "
                       "order around the walls.")
        elif abs(polygon_area(self.corners)) < 1e-6:
            out.append("The room has no area (its corners are in a line).")
        if not any(n.role == ROLE_RX for n in self.nodes):
            out.append("There are no receivers in the layout.")
        if not any(n.role == ROLE_TX for n in self.nodes):
            out.append("There are no transmitters in the layout.")
        seen: set[str] = set()
        for n in self.nodes:
            if n.mac in seen:
                out.append(f"{n.mac} is in the layout twice.")
            seen.add(n.mac)
        return out

    def warnings(self) -> list[str]:
        """Allowed, but probably not what was meant."""
        out = []
        unplaced = [n.label or n.mac for n in self.nodes if not n.placed]
        if unplaced:
            out.append(f"Not placed yet (still on a provisional spot): "
                       f"{', '.join(unplaced)}.")
        if len(self.corners) >= MIN_CORNERS:
            outside = [n.label or n.mac for n in self.nodes
                       if not point_in_polygon((n.x, n.y), self.corners, edge_tol=0.05)]
            if outside:
                out.append(f"Outside the room: {', '.join(outside)}.")
        return out

    # -- edits ---------------------------------------------------------------

    def move_node(self, node: Node, x: float, y: float) -> None:
        node.x, node.y = float(x), float(y)
        node.placed = True

    def move_corner(self, i: int, x: float, y: float) -> None:
        self.corners[i] = (float(x), float(y))

    def insert_corner(self, after: int) -> int:
        """Split the wall from corner ``after`` to the next one at its
        midpoint. Returns the new corner's index."""
        a = self.corners[after % len(self.corners)]
        b = self.corners[(after + 1) % len(self.corners)]
        self.corners.insert(after + 1, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
        return after + 1

    def remove_corner(self, i: int) -> bool:
        if len(self.corners) <= MIN_CORNERS:
            return False
        del self.corners[i]
        return True

    def set_rectangle(self, width: float, height: float) -> None:
        if not (width > 0 and height > 0):
            raise ValueError("width and height must be positive")
        self.corners = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]

    def add_node(self, mac: str, role: str, label: str = "") -> Node:
        """Add a board near the room's middle, marked not placed (fanned out
        so several added in a row don't hide each other)."""
        if self.find(mac) is not None:
            raise ValueError(f"{mac} is already in the layout")
        cx, cy = centroid(self.corners)
        for k in range(len(self.nodes) + 1):
            # A loose spiral of 0.4 m steps around the centre.
            angle, radius = k * 2.4, 0.4 * math.sqrt(k)
            x, y = cx + radius * math.cos(angle), cy + radius * math.sin(angle)
            if all(math.hypot(n.x - x, n.y - y) > 0.3 for n in self.nodes):
                break
        node = Node(normalize_mac(mac), role, round(x, 2), round(y, 2), label, placed=False)
        self.nodes.append(node)
        return node

    def remove_node(self, node: Node) -> None:
        self.nodes.remove(node)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def polygon_area(pts: list[Point]) -> float:
    return 0.5 * sum(x0 * y1 - x1 * y0
                     for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))


def centroid(pts: list[Point]) -> Point:
    a = polygon_area(pts)
    if abs(a) < 1e-9:
        return (sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts))
    cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        cross = x0 * y1 - x1 * y0
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    return cx / (6 * a), cy / (6 * a)


def _segments_cross(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)
    o1, o2 = orient(p1, p2, p3), orient(p1, p2, p4)
    o3, o4 = orient(p3, p4, p1), orient(p3, p4, p2)
    return o1 * o2 < 0 and o3 * o4 < 0


def self_intersects(pts: list[Point]) -> bool:
    """Walls crossing each other — properly, or by one passing through a
    corner of a non-adjacent wall (easy to make with snapping)."""
    n = len(pts)
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue                     # adjacent through the closing wall
            if _segments_cross(*edges[i], *edges[j]):
                return True
    for k in range(n):
        # Corner k lies on a wall that doesn't end at it, and the outline
        # goes from one side of that wall to the other through it.
        prev, here, nxt = pts[k - 1], pts[k], pts[(k + 1) % n]
        for i, (a, b) in enumerate(edges):
            if i in (k, (k - 1) % n) or _dist_to_segment(here, a, b) > 1e-9:
                continue
            side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if side(prev) * side(nxt) < 0:
                return True
    return False


def _dist_to_segment(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def point_in_polygon(p: Point, pts: list[Point], edge_tol: float = 0.0) -> bool:
    """Ray casting; points within ``edge_tol`` m of a wall count as inside
    (boards are usually mounted on walls)."""
    n = len(pts)
    if edge_tol > 0 and any(_dist_to_segment(p, pts[i], pts[(i + 1) % n]) <= edge_tol
                            for i in range(n)):
        return True
    x, y = p
    inside = False
    for i in range(n):
        (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            if x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
                inside = not inside
    return inside


def relative_image_path(image: str, links_path: Path) -> str:
    """Store the image relative to links.json when it sits beside or below
    it (so the pair can be moved together), else as given."""
    try:
        rel = os.path.relpath(Path(image).resolve(), Path(links_path).resolve().parent)
    except ValueError:                       # different drive (Windows)
        return image
    return rel if not rel.startswith("..") else str(Path(image).resolve())


def resolve_image_path(image: str, links_path: Path) -> Path:
    path = Path(image).expanduser()
    return path if path.is_absolute() else Path(links_path).resolve().parent / path
