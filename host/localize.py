"""Per-cell motion likelihood from per-link motion-σ.

Each TX-RX link is a line segment through the room. Motion near that
line perturbs the multipath, which raises the link's motion-σ. Inverting
that: given the current per-link σ values, the cells of the room most
likely to contain motion are those that lie close to the bright links.

We discretize the room polygon into a grid, precompute per-cell distance
to each link (Gaussian-weighted), and at update time produce a heatmap
as Σ_links (link_score × kernel(distance to link)). Cells outside the
polygon are masked (None means outside, useful for L-shaped rooms).

Resolution and kernel width are tunable; defaults give ~10 cm grid and
0.3 m link "fatness" — enough for room-scale localization without
overfitting to noisy links.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np


@dataclasses.dataclass
class _LinkKernel:
    tx_mac: str
    rx_mac: str
    kernel: np.ndarray  # shape (ny, nx); precomputed Gaussian over the grid


def _point_to_segment_grid(X: np.ndarray, Y: np.ndarray,
                            p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Per-cell distance from (X, Y) to the segment p1-p2.

    Vectorized: clamps the projection to the segment so endpoints are
    handled correctly for cells whose closest point is past an end.
    """
    seg = p2 - p1
    seg_len_sq = float(seg @ seg)
    if seg_len_sq == 0.0:
        return np.hypot(X - p1[0], Y - p1[1])
    t = np.clip(((X - p1[0]) * seg[0] + (Y - p1[1]) * seg[1]) / seg_len_sq, 0.0, 1.0)
    proj_x = p1[0] + t * seg[0]
    proj_y = p1[1] + t * seg[1]
    return np.hypot(X - proj_x, Y - proj_y)


class Localizer:
    """Maintains a precomputed grid of per-link kernels and updates per-cell
    likelihood from incoming motion scores."""

    def __init__(self,
                 polygon: np.ndarray,
                 tx_positions: dict[str, np.ndarray],
                 rx_positions: dict[str, np.ndarray],
                 grid_step: float = 0.1,
                 link_sigma_m: float = 0.3):
        from matplotlib.path import Path

        self.polygon = polygon
        bbox_min = polygon.min(axis=0)
        bbox_max = polygon.max(axis=0)
        # Half-step pad so the grid covers a tiny margin past each wall;
        # makes the heatmap edges look smooth in the 2.5D render.
        nx = int(np.ceil((bbox_max[0] - bbox_min[0]) / grid_step)) + 1
        ny = int(np.ceil((bbox_max[1] - bbox_min[1]) / grid_step)) + 1
        x = np.linspace(bbox_min[0], bbox_max[0], nx)
        y = np.linspace(bbox_min[1], bbox_max[1], ny)
        self.X, self.Y = np.meshgrid(x, y, indexing="xy")
        self.x_axis = x
        self.y_axis = y

        path = Path(polygon)
        points = np.column_stack([self.X.ravel(), self.Y.ravel()])
        self.mask = path.contains_points(points).reshape(self.X.shape)

        self.tx_positions = tx_positions
        self.rx_positions = rx_positions
        self.links: list[_LinkKernel] = []
        denom = 2.0 * link_sigma_m * link_sigma_m
        for tx_mac, tx_pos in tx_positions.items():
            for rx_mac, rx_pos in rx_positions.items():
                d = _point_to_segment_grid(self.X, self.Y, tx_pos, rx_pos)
                kernel = np.exp(-(d * d) / denom)
                kernel[~self.mask] = 0.0
                self.links.append(_LinkKernel(tx_mac, rx_mac, kernel))

    def update(self, link_scores: dict[tuple[str, str], float]) -> np.ndarray:
        """Per-cell likelihood. Cells outside the polygon are zero."""
        grid = np.zeros_like(self.X)
        for link in self.links:
            score = link_scores.get((link.tx_mac, link.rx_mac), 0.0)
            if score > 0.0:
                grid += score * link.kernel
        return grid

    def argmax_xy(self, grid: np.ndarray) -> tuple[float, float, float]:
        """Position of the brightest cell. Returns (x, y, value)."""
        flat = grid.argmax()
        iy, ix = np.unravel_index(flat, grid.shape)
        return float(self.x_axis[ix]), float(self.y_axis[iy]), float(grid[iy, ix])
