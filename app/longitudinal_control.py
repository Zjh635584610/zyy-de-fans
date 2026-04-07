"""纵向速度控制模块。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config_loader import load_config


DEFAULT_CONFIG = {
    "control": {
        "default_safe_speed_mps": 0.3,
        "cruise_speed_mps": 0.35,
        "kp": 0.8,
        "ki": 0.25,
        "max_throttle": 0.3,
    }
}


@dataclass
class SpeedControllerState:
    integral_error: float = 0.0


class LongitudinalController:
    def __init__(self, config_path: Path | None = None) -> None:
        if config_path is None:
            config_path = Path(__file__).resolve().parents[1] / "configs" / "control.yaml"
        self.config = load_config(config_path, DEFAULT_CONFIG)
        control_cfg = self.config["control"]
        self.cruise_speed = float(control_cfg.get("cruise_speed_mps", 0.35))
        self.kp = float(control_cfg.get("kp", 0.8))
        self.ki = float(control_cfg.get("ki", 0.25))
        self.max_throttle = float(control_cfg.get("max_throttle", 0.3))
        self.state = SpeedControllerState()

    def reset(self) -> None:
        self.state = SpeedControllerState()

    def compute_target_speed(self, decision: dict) -> float:
        if decision.get("emergency_brake", False):
            return 0.0
        if decision.get("warning", False):
            safe_speed = float(decision.get("safe_speed", self.cruise_speed))
            return min(self.cruise_speed, safe_speed if safe_speed > 0 else self.cruise_speed)
        return self.cruise_speed

    def update(self, measured_speed: float, decision: dict, dt: float) -> tuple[float, float]:
        target_speed = self.compute_target_speed(decision)
        error = target_speed - float(measured_speed)
        self.state.integral_error += error * max(dt, 1e-3)
        throttle = self.kp * error + self.ki * self.state.integral_error
        throttle = float(np.clip(throttle, -self.max_throttle, self.max_throttle))
        if decision.get("emergency_brake", False):
            throttle = min(throttle, -0.05)
        return throttle, target_speed
