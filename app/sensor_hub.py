"""传感器接入模块。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.quanser_shim import install_qcar2_shim


@dataclass
class SensorHubConfig:
    frequency: int = 30
    calibrate: bool = False
    attach_lidar: bool = True
    initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sensor_offset_xy: tuple[float, float] = (0.125, 0.0)


class SensorHub:
    """统一管理激光雷达、IMU、GPS 和编码器读取。"""

    def __init__(self, live: bool = False, config: SensorHubConfig | None = None) -> None:
        self.config = config or SensorHubConfig()
        self.live = bool(live)
        self.qcar: Any | None = None
        self.gps: Any | None = None
        self._last_time = time.time()
        self._opened = False

    def open(self) -> None:
        if self._opened or not self.live:
            return

        install_qcar2_shim()
        from pal.products.qcar import QCar, QCarGPS

        self.qcar = QCar(readMode=1, frequency=self.config.frequency)
        self.gps = QCarGPS(
            initialPose=np.array(self.config.initial_pose),
            calibrate=self.config.calibrate,
            attach_lidar=self.config.attach_lidar,
        )
        if hasattr(self.qcar, "__enter__"):
            self.qcar.__enter__()
        if hasattr(self.gps, "__enter__"):
            self.gps.__enter__()
        self._opened = True

    def close(self) -> None:
        if not self._opened:
            return
        self.stop()
        if self.gps is not None and hasattr(self.gps, "__exit__"):
            self.gps.__exit__(None, None, None)
        if self.qcar is not None and hasattr(self.qcar, "__exit__"):
            self.qcar.__exit__(None, None, None)
        self._opened = False

    def __enter__(self) -> "SensorHub":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read(self) -> dict:
        if not self.live:
            raise RuntimeError("当前 SensorHub 处于非 live 模式，请在测试中直接提供合成传感器帧。")

        assert self.qcar is not None
        assert self.gps is not None

        self.qcar.read()
        self.gps.readGPS()
        if self.config.attach_lidar:
            self.gps.readLidar()

        t_now = time.time()
        dt = max(t_now - self._last_time, 1e-3)
        self._last_time = t_now

        gps_position = None
        gps_orientation = None
        if hasattr(self.gps, "position") and hasattr(self.gps, "orientation"):
            gps_position = np.array(self.gps.position, dtype=float)
            gps_orientation = np.array(self.gps.orientation, dtype=float)

        lidar_ranges = np.array(getattr(self.gps, "distances", []), dtype=float)
        lidar_angles = np.array(getattr(self.gps, "angles", []), dtype=float)
        if lidar_angles.size:
            lidar_angles = np.mod(2.5 * np.pi - lidar_angles, 2 * np.pi)

        return {
            "t": t_now,
            "dt": dt,
            "lidar_ranges": lidar_ranges,
            "lidar_angles": lidar_angles,
            "imu": {
                "gyro": np.array(getattr(self.qcar, "gyroscope", [0.0, 0.0, 0.0]), dtype=float),
                "accel": np.array(getattr(self.qcar, "accelerometer", [0.0, 0.0, 0.0]), dtype=float),
            },
            "gps": {
                "position": gps_position,
                "orientation": gps_orientation,
                "scan_time": getattr(self.gps, "scanTime", t_now),
            },
            "encoder": {
                "counts": float(getattr(self.qcar, "motorEncoder", [0.0])[0]),
                "speed": float(getattr(self.qcar, "motorTach", 0.0)),
            },
            "steering": float(getattr(self.qcar, "lastSteering", 0.0)) if hasattr(self.qcar, "lastSteering") else 0.0,
            "sensor_offset_xy": np.array(self.config.sensor_offset_xy, dtype=float),
        }

    def write_command(self, throttle: float, steering: float = 0.0) -> None:
        if not self.live:
            return
        assert self.qcar is not None
        throttle = float(np.clip(throttle, -0.3, 0.3))
        steering = float(np.clip(steering, -np.pi / 6, np.pi / 6))
        self.qcar.write(throttle, steering)

    def stop(self) -> None:
        if not self.live or self.qcar is None:
            return
        try:
            self.qcar.read_write_std(throttle=0, steering=0)
        except Exception:
            try:
                self.qcar.write(0, 0)
            except Exception:
                pass
