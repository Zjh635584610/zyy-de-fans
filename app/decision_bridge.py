"""感知到控制的桥接模块。"""

from __future__ import annotations

import math
from pathlib import Path

from app.config_loader import load_config


DEFAULT_CONFIG = {
    "safety": {
        "warning_distance_m": 1.2,
        "emergency_brake_distance_m": 0.7,
        "forward_corridor_half_width_m": 0.8,
    },
    "control": {"default_safe_speed_mps": 0.3},
}


class DecisionBridge:
    """根据障碍物列表生成告警或控制建议。"""

    def __init__(self, config_path: Path | None = None) -> None:
        if config_path is None:
            config_path = Path(__file__).resolve().parents[1] / "configs" / "control.yaml"
        self.config = load_config(config_path, DEFAULT_CONFIG)

    def evaluate(self, tracked_objects: list[dict], ego_state: dict) -> dict:
        pose = ego_state.get("pose", {})
        ego_x = float(pose.get("x", 0.0))
        ego_y = float(pose.get("y", 0.0))
        ego_yaw = float(pose.get("yaw", 0.0))

        corridor_half_width = float(self.config["safety"].get("forward_corridor_half_width_m", 0.8))
        warning_distance = float(self.config["safety"]["warning_distance_m"])
        emergency_distance = float(self.config["safety"]["emergency_brake_distance_m"])
        safe_speed = float(self.config["control"]["default_safe_speed_mps"])

        min_forward_distance = None
        closest_object = None
        for obj in tracked_objects:
            cx, cy = obj["center"][:2]
            dx = float(cx) - ego_x
            dy = float(cy) - ego_y
            rel_forward = math.cos(ego_yaw) * dx + math.sin(ego_yaw) * dy
            rel_lateral = -math.sin(ego_yaw) * dx + math.cos(ego_yaw) * dy

            if rel_forward < 0:
                continue
            if abs(rel_lateral) > corridor_half_width:
                continue
            if min_forward_distance is None or rel_forward < min_forward_distance:
                min_forward_distance = rel_forward
                closest_object = obj

        if min_forward_distance is None:
            return {
                "warning": False,
                "emergency_brake": False,
                "safe_speed": safe_speed,
                "reason": "clear_path",
                "closest_object": None,
            }

        if min_forward_distance <= emergency_distance:
            return {
                "warning": True,
                "emergency_brake": True,
                "safe_speed": 0.0,
                "reason": "obstacle_too_close",
                "closest_object": closest_object,
            }

        if min_forward_distance <= warning_distance:
            return {
                "warning": True,
                "emergency_brake": False,
                "safe_speed": min(safe_speed, 0.1),
                "reason": "obstacle_in_warning_zone",
                "closest_object": closest_object,
            }

        return {
            "warning": False,
            "emergency_brake": False,
            "safe_speed": safe_speed,
            "reason": "clear_path",
            "closest_object": closest_object,
        }
