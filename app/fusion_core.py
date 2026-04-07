"""位姿融合与坐标转换模块。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


try:
    from hal.content.qcar_functions import QCarEKF
    HAS_EKF = True
except ImportError:  # pragma: no cover
    HAS_EKF = False
    QCarEKF = object  # type: ignore[assignment]


@dataclass
class FusionConfig:
    use_ekf: bool = True
    initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)


class FusionCore:
    """负责时间同步、EKF 位姿估计和坐标统一。"""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()
        self.ekf = None
        if self.config.use_ekf and HAS_EKF:
            self.ekf = QCarEKF(x_0=np.array(self.config.initial_pose))

    def estimate_pose(self, sensor_frame: dict) -> dict:
        t = float(sensor_frame.get("t", 0.0))
        dt = float(sensor_frame.get("dt", 0.0))
        gps = sensor_frame.get("gps") or {}
        encoder = sensor_frame.get("encoder") or {}
        imu = sensor_frame.get("imu") or {}
        steering = float(sensor_frame.get("steering", 0.0))

        position = gps.get("position")
        orientation = gps.get("orientation")
        motor_speed = float(encoder.get("speed", 0.0))
        gyro = imu.get("gyro")
        yaw_rate = float(gyro[2]) if isinstance(gyro, np.ndarray) and gyro.size >= 3 else 0.0

        if self.ekf is not None:
            y_gps = None
            if position is not None and orientation is not None:
                y_gps = np.array([position[0], position[1], orientation[2]], dtype=float)
            self.ekf.update([motor_speed, steering], max(dt, 1e-3), y_gps, yaw_rate)
            x_hat = np.array(self.ekf.x_hat).reshape(-1)
            pose = {"x": float(x_hat[0]), "y": float(x_hat[1]), "yaw": float(x_hat[2])}
        elif position is not None and orientation is not None:
            pose = {"x": float(position[0]), "y": float(position[1]), "yaw": float(orientation[2])}
        else:
            pose = {
                "x": float(self.config.initial_pose[0]),
                "y": float(self.config.initial_pose[1]),
                "yaw": float(self.config.initial_pose[2]),
            }

        return {
            "t": t,
            "pose": pose,
            "velocity": {"vx": motor_speed, "vy": 0.0, "yaw_rate": yaw_rate},
        }

    @staticmethod
    def sensor_to_vehicle(points_sensor: np.ndarray, sensor_offset_xy: np.ndarray | tuple[float, float]) -> np.ndarray:
        if points_sensor.size == 0:
            return np.empty((0, 2), dtype=float)
        offset = np.asarray(sensor_offset_xy, dtype=float).reshape(1, 2)
        return points_sensor[:, :2] + offset

    @staticmethod
    def vehicle_to_world(points_vehicle: np.ndarray, ego_state: dict) -> np.ndarray:
        if points_vehicle.size == 0:
            return np.empty((0, 2), dtype=float)
        pose = ego_state["pose"]
        yaw = float(pose["yaw"])
        c = np.cos(yaw)
        s = np.sin(yaw)
        rot = np.array([[c, -s], [s, c]], dtype=float)
        translated = points_vehicle @ rot.T
        translated[:, 0] += float(pose["x"])
        translated[:, 1] += float(pose["y"])
        return translated

    @staticmethod
    def world_to_vehicle(points_world: np.ndarray, ego_state: dict) -> np.ndarray:
        if points_world.size == 0:
            return np.empty((0, 2), dtype=float)
        pose = ego_state["pose"]
        shifted = np.asarray(points_world, dtype=float).copy()
        shifted[:, 0] -= float(pose["x"])
        shifted[:, 1] -= float(pose["y"])
        yaw = float(pose["yaw"])
        c = np.cos(yaw)
        s = np.sin(yaw)
        rot_inv = np.array([[c, s], [-s, c]], dtype=float)
        return shifted @ rot_inv.T

    def transform_frame(self, sensor_frame: dict, ego_state: dict) -> dict:
        output = dict(sensor_frame)

        if "point_cloud_world" in sensor_frame:
            world_points = np.asarray(sensor_frame["point_cloud_world"], dtype=float)
            vehicle_points = self.world_to_vehicle(world_points, ego_state)
            output["point_cloud_vehicle"] = vehicle_points
            output["point_cloud_world"] = world_points
            return output

        if "point_cloud_vehicle" in sensor_frame:
            vehicle_points = np.asarray(sensor_frame["point_cloud_vehicle"], dtype=float)
        elif "point_cloud_sensor" in sensor_frame:
            vehicle_points = self.sensor_to_vehicle(
                np.asarray(sensor_frame["point_cloud_sensor"], dtype=float),
                sensor_frame.get("sensor_offset_xy", np.array([0.125, 0.0], dtype=float)),
            )
        else:
            lidar_ranges = np.asarray(sensor_frame.get("lidar_ranges", []), dtype=float)
            lidar_angles = np.asarray(sensor_frame.get("lidar_angles", []), dtype=float)
            valid = np.isfinite(lidar_ranges) & (lidar_ranges > 0) & np.isfinite(lidar_angles)
            lidar_ranges = lidar_ranges[valid]
            lidar_angles = lidar_angles[valid]
            if lidar_ranges.size:
                sensor_points = np.column_stack((lidar_ranges * np.cos(lidar_angles), lidar_ranges * np.sin(lidar_angles)))
            else:
                sensor_points = np.empty((0, 2), dtype=float)
            vehicle_points = self.sensor_to_vehicle(
                sensor_points,
                sensor_frame.get("sensor_offset_xy", np.array([0.125, 0.0], dtype=float)),
            )

        output["point_cloud_vehicle"] = vehicle_points
        output["point_cloud_world"] = self.vehicle_to_world(vehicle_points, ego_state)
        return output
