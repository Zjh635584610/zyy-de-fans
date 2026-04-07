"""目标跟踪模块。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackConfig:
    max_association_distance: float = 0.6
    max_missed_frames: int = 3


class TrackingCore:
    """负责多帧关联和运动状态估计。"""

    def __init__(self, config: TrackConfig | None = None) -> None:
        self.config = config or TrackConfig()
        self.next_track_id = 1
        self.tracks: dict[int, dict] = {}

    def update(self, detected_objects: list[dict], ego_state: dict) -> list[dict]:
        t = float(ego_state.get("t", 0.0))
        assigned_tracks: set[int] = set()
        tracked_objects: list[dict] = []

        for obj in detected_objects:
            center = np.array(obj["center"][:2], dtype=float)
            best_track_id = None
            best_distance = np.inf

            for track_id, track in self.tracks.items():
                if track_id in assigned_tracks:
                    continue
                track_center = np.array(track["center"][:2], dtype=float)
                distance = np.linalg.norm(center - track_center)
                if distance < best_distance and distance <= self.config.max_association_distance:
                    best_distance = distance
                    best_track_id = track_id

            tracked = dict(obj)
            if best_track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                velocity = [0.0, 0.0]
            else:
                track = self.tracks[best_track_id]
                dt = max(t - track["t"], 1e-3)
                prev_center = np.array(track["center"][:2], dtype=float)
                vel_xy = (center - prev_center) / dt
                velocity = [float(vel_xy[0]), float(vel_xy[1])]
                track_id = best_track_id
                assigned_tracks.add(track_id)

            tracked["id"] = track_id
            tracked["velocity"] = velocity
            tracked["speed"] = float(np.linalg.norm(velocity))
            tracked["t"] = t
            tracked["missed_frames"] = 0
            tracked_objects.append(tracked)

        new_tracks: dict[int, dict] = {tracked["id"]: tracked for tracked in tracked_objects}
        for track_id, track in self.tracks.items():
            if track_id in new_tracks:
                continue
            missed = int(track.get("missed_frames", 0)) + 1
            if missed <= self.config.max_missed_frames:
                updated = dict(track)
                updated["missed_frames"] = missed
                new_tracks[track_id] = updated

        self.tracks = new_tracks
        return tracked_objects
