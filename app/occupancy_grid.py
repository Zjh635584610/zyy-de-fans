"""二维占据栅格地图模块。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.special import expit, logit


def wrap_to_2pi(angle: np.ndarray) -> np.ndarray:
    return np.mod(angle, 2 * np.pi)


def find_overlap(base: np.ndarray, patch: np.ndarray, i_top: int, j_left: int):
    base_rows, base_cols = base.shape
    patch_rows, patch_cols = patch.shape

    i0 = max(0, i_top)
    j0 = max(0, j_left)
    i1 = min(base_rows, i_top + patch_rows)
    j1 = min(base_cols, j_left + patch_cols)

    if i0 >= i1 or j0 >= j1:
        return (slice(0, 0), slice(0, 0)), (slice(0, 0), slice(0, 0))

    patch_i0 = i0 - i_top
    patch_j0 = j0 - j_left
    patch_i1 = patch_i0 + (i1 - i0)
    patch_j1 = patch_j0 + (j1 - j0)

    return (slice(i0, i1), slice(j0, j1)), (slice(patch_i0, patch_i1), slice(patch_j0, patch_j1))


@dataclass
class OccupancyGridConfig:
    x_min: float = -4.0
    x_max: float = 4.0
    y_min: float = -4.0
    y_max: float = 8.0
    cell_width: float = 0.05
    r_max: float = 4.0
    r_res: float = 0.05
    p_low: float = 0.4
    p_high: float = 0.6
    phi_res_deg: float = 1.0


class OccupancyGrid2D:
    def __init__(self, config: OccupancyGridConfig | None = None) -> None:
        self.config = config or OccupancyGridConfig()

        self.p_low = self.config.p_low
        self.p_prior = 0.5
        self.p_high = self.config.p_high
        self.p_sat = 0.001

        self.l_low = logit(self.p_low)
        self.l_prior = logit(self.p_prior)
        self.l_high = logit(self.p_high)
        self.l_min = logit(self.p_sat)
        self.l_max = logit(1 - self.p_sat)

        self.phi_res = np.deg2rad(self.config.phi_res_deg)
        self.m_polar_patch = int(np.ceil((2 * np.pi) / self.phi_res))
        self.n_polar_patch = int(np.floor(self.config.r_max / self.config.r_res))
        self.polar_patch = np.full((self.m_polar_patch, self.n_polar_patch), self.l_prior, dtype=np.float32)

        self.n_patch = int(2 * np.ceil(self.config.r_max / self.config.cell_width) + 1)
        self.patch = np.full((self.n_patch, self.n_patch), self.l_prior, dtype=np.float32)

        self.x_length = self.config.x_max - self.config.x_min
        self.y_length = self.config.y_max - self.config.y_min
        self.m = int(np.ceil(self.y_length / self.config.cell_width))
        self.n = int(np.ceil(self.x_length / self.config.cell_width))
        self.map = np.full((self.m, self.n), self.l_prior, dtype=np.float32)

    def update_polar_grid(self, distances: np.ndarray) -> None:
        self.polar_patch.fill(self.l_prior)
        if distances.size == 0:
            return

        count = min(len(distances), self.m_polar_patch)
        r_idx = np.rint(np.asarray(distances[:count], dtype=float) / self.config.r_res).astype(int)
        for i in range(count):
            if r_idx[i] > 0:
                end_idx = min(r_idx[i], self.n_polar_patch - 1)
                self.polar_patch[i, :end_idx] = self.l_low
                self.polar_patch[i, end_idx:end_idx + 1] = self.l_high
                self.polar_patch[i, end_idx + 1:] = self.l_prior

    def generate_patch(self, yaw: float) -> None:
        cx = (self.n_patch * self.config.cell_width) / 2
        cy = cx
        x = np.linspace(-cx, cx, self.n_patch)
        y = np.linspace(-cy, cy, self.n_patch)
        xv, yv = np.meshgrid(x, y)

        r_patch = np.sqrt(np.square(xv) + np.square(yv)) / self.config.r_res
        phi_patch = wrap_to_2pi(np.arctan2(yv, xv) + yaw) / self.phi_res

        ndimage.map_coordinates(
            input=self.polar_patch,
            coordinates=[phi_patch, r_patch],
            output=self.patch,
            mode="nearest",
        )

    def xy_to_ij(self, x: float, y: float) -> tuple[int, int]:
        i = int(np.round((self.config.y_max - y) / self.config.cell_width))
        j = int(np.round((x - self.config.x_min) / self.config.cell_width))
        return i, j

    def update_map(self, x: float, y: float, yaw: float, distances: np.ndarray) -> None:
        self.update_polar_grid(distances)
        self.generate_patch(yaw)

        iy, jx = self.xy_to_ij(x, y)
        i_top = int(iy - np.round((self.n_patch - 1) / 2))
        j_left = int(jx - np.round((self.n_patch - 1) / 2))

        map_slice, patch_slice = find_overlap(self.map, self.patch, i_top, j_left)
        if map_slice[0].stop == map_slice[0].start or map_slice[1].stop == map_slice[1].start:
            return

        self.map[map_slice] = np.clip(
            self.map[map_slice] + self.patch[patch_slice],
            self.l_min,
            self.l_max,
        )

    def snapshot(self) -> dict:
        return {
            "polar_prob": expit(self.polar_patch),
            "local_prob": expit(self.patch),
            "global_prob": expit(self.map),
            "cell_width": self.config.cell_width,
            "r_res": self.config.r_res,
            "phi_res": self.phi_res,
            "x_min": self.config.x_min,
            "x_max": self.config.x_max,
            "y_min": self.config.y_min,
            "y_max": self.config.y_max,
        }
