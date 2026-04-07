"""障碍物感知模块。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config_loader import load_config


DEFAULT_CONFIG = {
    "lidar": {"max_range_m": 4.0, "angle_resolution_deg": 1.0},
    "preprocess": {
        "ground_threshold_m": 0.05,
        "voxel_size_m": 0.05,
        "min_valid_distance_m": 0.01,
        "neighbor_consistency_m": 0.35,
    },
    "clustering": {
        "method": "dbscan",
        "eps_m": 0.08,
        "adaptive_gain": 0.03,
        "min_points": 5,
    },
    "bbox": {"min_size_m": 0.10},
}


@dataclass
class PerceptionDebug:
    raw_points: np.ndarray
    filtered_points: np.ndarray
    downsampled_points: np.ndarray
    cluster_labels: np.ndarray


class PerceptionCore:
    """负责预处理、聚类和包围盒提取。"""

    def __init__(self, config_path: Path | None = None) -> None:
        if config_path is None:
            config_path = Path(__file__).resolve().parents[1] / "configs" / "detector.yaml"
        self.config = load_config(config_path, DEFAULT_CONFIG)
        self.last_debug: PerceptionDebug | None = None

    def _extract_points(self, sensor_frame: dict) -> np.ndarray:
        if "point_cloud_world" in sensor_frame:
            return np.asarray(sensor_frame["point_cloud_world"], dtype=float)
        if "point_cloud_vehicle" in sensor_frame:
            return np.asarray(sensor_frame["point_cloud_vehicle"], dtype=float)
        return np.empty((0, 2), dtype=float)

    def _remove_invalid_and_outliers(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.empty((0, 2), dtype=float)

        valid = np.all(np.isfinite(points), axis=1)
        points = points[valid]
        if len(points) <= 2:
            return points

        max_range = float(self.config["lidar"]["max_range_m"])
        min_valid = float(self.config["preprocess"]["min_valid_distance_m"])
        ranges = np.linalg.norm(points[:, :2], axis=1)
        points = points[(ranges >= min_valid) & (ranges <= max_range + 1.0)]
        if len(points) <= 2:
            return points

        jump_threshold = float(self.config["preprocess"].get("neighbor_consistency_m", 0.35))
        diffs_prev = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
        diffs_next = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
        keep = (diffs_prev < jump_threshold) | (diffs_next < jump_threshold)
        return points[keep]

    @staticmethod
    def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
        if points.size == 0 or voxel_size <= 0:
            return np.asarray(points, dtype=float)

        coords = np.floor(points / voxel_size).astype(int)
        accum: dict[tuple[int, int], list[np.ndarray]] = {}
        for coord, point in zip(coords, points):
            accum.setdefault(tuple(coord), []).append(point)
        return np.array([np.mean(bucket, axis=0) for bucket in accum.values()], dtype=float)

    @staticmethod
    def _dbscan(points: np.ndarray, base_eps: float, min_points: int, adaptive_gain: float) -> np.ndarray:
        n_points = len(points)
        if n_points == 0:
            return np.empty((0,), dtype=int)
        if n_points == 1:
            return np.array([0], dtype=int)

        ranges = np.linalg.norm(points, axis=1)
        adaptive_eps = base_eps + adaptive_gain * ranges
        deltas = points[:, None, :] - points[None, :, :]
        dist2 = np.sum(deltas * deltas, axis=2)
        eps_matrix = np.minimum(adaptive_eps[:, None], adaptive_eps[None, :])
        adjacency = dist2 <= (eps_matrix * eps_matrix)

        labels = np.full(n_points, -1, dtype=int)
        visited = np.zeros(n_points, dtype=bool)
        cluster_id = 0

        for i in range(n_points):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = np.flatnonzero(adjacency[i])
            if neighbors.size < min_points:
                continue

            labels[i] = cluster_id
            seeds = list(neighbors.tolist())
            while seeds:
                j = seeds.pop()
                if not visited[j]:
                    visited[j] = True
                    neighbors_j = np.flatnonzero(adjacency[j])
                    if neighbors_j.size >= min_points:
                        seeds.extend(neighbors_j.tolist())
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1

        return labels

    @staticmethod
    def _extract_bbox(cluster_points: np.ndarray, cluster_id: int) -> dict:
        min_xy = np.min(cluster_points, axis=0)
        max_xy = np.max(cluster_points, axis=0)
        center_xy = (min_xy + max_xy) / 2.0
        size_xy = max_xy - min_xy
        return {
            "id": None,
            "cluster_id": cluster_id,
            "center": [float(center_xy[0]), float(center_xy[1]), 0.0],
            "size": [float(size_xy[0]), float(size_xy[1]), 0.0],
            "distance": float(np.linalg.norm(center_xy)),
            "confidence": float(min(0.99, 0.4 + 0.03 * len(cluster_points))),
            "num_points": int(len(cluster_points)),
            "bbox_min": [float(min_xy[0]), float(min_xy[1]), 0.0],
            "bbox_max": [float(max_xy[0]), float(max_xy[1]), 0.0],
        }

    def detect(self, sensor_frame: dict, ego_state: dict) -> list[dict]:
        _ = ego_state
        raw_points = self._extract_points(sensor_frame)
        filtered = self._remove_invalid_and_outliers(raw_points)
        downsampled = self._voxel_downsample(filtered, float(self.config["preprocess"]["voxel_size_m"]))

        labels = self._dbscan(
            downsampled,
            base_eps=float(self.config["clustering"]["eps_m"]),
            min_points=int(self.config["clustering"]["min_points"]),
            adaptive_gain=float(self.config["clustering"].get("adaptive_gain", 0.0)),
        )

        self.last_debug = PerceptionDebug(
            raw_points=raw_points,
            filtered_points=filtered,
            downsampled_points=downsampled,
            cluster_labels=labels,
        )

        objects: list[dict] = []
        min_size = float(self.config["bbox"]["min_size_m"])

        for label in sorted(set(labels.tolist()) - {-1}):
            cluster_points = downsampled[labels == label]
            if len(cluster_points) == 0:
                continue
            bbox = self._extract_bbox(cluster_points, int(label))
            if bbox["size"][0] < min_size and bbox["size"][1] < min_size:
                continue
            objects.append(bbox)

        objects.sort(key=lambda item: item["distance"])
        return objects
