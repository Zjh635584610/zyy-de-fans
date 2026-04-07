"""项目统一入口。"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.decision_bridge import DecisionBridge
from app.fusion_core import FusionConfig, FusionCore
from app.longitudinal_control import LongitudinalController
from app.occupancy_grid import OccupancyGrid2D
from app.perception_core import PerceptionCore
from app.scene_manager import prepare_scene
from app.sensor_hub import SensorHub, SensorHubConfig
from app.tracking_core import TrackingCore
from app.visualizer import Visualizer, VisualizerConfig


class ObstacleDetectionApp:
    def __init__(
        self,
        live: bool = False,
        scene_name: str = "Cityscape Lite",
        launch_scene: bool = False,
        enable_gui: bool = False,
        enable_traffic: bool = False,
        initial_pose: tuple[float, float, float] | None = None,
    ) -> None:
        self.scene = prepare_scene(
            scene_name=scene_name,
            obstacle_layout="default",
            initial_pose=initial_pose,
            launch=launch_scene,
            enable_traffic=enable_traffic,
        )
        self.sensor_hub = SensorHub(live=live, config=SensorHubConfig(initial_pose=self.scene.initial_pose))
        self.fusion_core = FusionCore(config=FusionConfig(initial_pose=self.scene.initial_pose))
        self.occupancy_grid = OccupancyGrid2D()
        self.perception_core = PerceptionCore()
        self.tracking_core = TrackingCore()
        self.decision_bridge = DecisionBridge()
        self.longitudinal_controller = LongitudinalController()
        self.visualizer = Visualizer(VisualizerConfig(enable_gui=enable_gui))

    def process_frame(self, sensor_frame: dict) -> dict:
        ego_state = self.fusion_core.estimate_pose(sensor_frame)
        transformed_frame = self.fusion_core.transform_frame(sensor_frame, ego_state)

        pose = ego_state["pose"]
        sensor_offset = sensor_frame.get("sensor_offset_xy")
        if sensor_offset is None:
            sensor_offset = [0.125, 0.0]
        sensor_offset = sensor_offset[0]
        sensor_x = float(pose["x"]) + float(sensor_offset) * math.cos(float(pose["yaw"]))
        sensor_y = float(pose["y"]) + float(sensor_offset) * math.sin(float(pose["yaw"]))
        lidar_ranges = transformed_frame.get("lidar_ranges")
        if lidar_ranges is not None:
            self.occupancy_grid.update_map(sensor_x, sensor_y, float(pose["yaw"]), lidar_ranges)
        map_state = self.occupancy_grid.snapshot()

        detected_objects = self.perception_core.detect(transformed_frame, ego_state)
        tracked_objects = self.tracking_core.update(detected_objects, ego_state)
        decision = self.decision_bridge.evaluate(tracked_objects, ego_state)
        return self.visualizer.render(
            ego_state,
            tracked_objects,
            decision,
            map_state=map_state,
            point_cloud_world=transformed_frame.get("point_cloud_world"),
        )

    def run_live_steps(self, num_frames: int = 10, warmup_frames: int = 5) -> list[dict]:
        if not self.sensor_hub.live:
            raise RuntimeError("run_live_steps 只能在 live=True 时调用。")

        results: list[dict] = []
        with self.sensor_hub as sensor_hub:
            frames_to_collect = max(0, num_frames)
            total_frames = warmup_frames + frames_to_collect
            for index in range(total_frames):
                sensor_frame = sensor_hub.read()
                start = time.perf_counter()
                result = self.process_frame(sensor_frame)
                elapsed = time.perf_counter() - start
                self.visualizer.refresh()
                if index >= warmup_frames:
                    result = dict(result)
                    result["processing_time_s"] = elapsed
                    results.append(result)
        return results

    def run_live_drive(self, num_frames: int = 30, warmup_frames: int = 8, steering: float = 0.0) -> list[dict]:
        if not self.sensor_hub.live:
            raise RuntimeError("run_live_drive 只能在 live=True 时调用。")

        self.longitudinal_controller.reset()
        results: list[dict] = []
        with self.sensor_hub as sensor_hub:
            total_frames = warmup_frames + max(0, num_frames)
            for index in range(total_frames):
                sensor_frame = sensor_hub.read()
                start = time.perf_counter()
                result = self.process_frame(sensor_frame)
                elapsed = time.perf_counter() - start

                measured_speed = float(sensor_frame.get("encoder", {}).get("speed", 0.0))
                dt = float(sensor_frame.get("dt", 0.05))
                if index < warmup_frames:
                    throttle = 0.0
                    target_speed = 0.0
                else:
                    throttle, target_speed = self.longitudinal_controller.update(
                        measured_speed=measured_speed,
                        decision=result["decision"],
                        dt=dt,
                    )
                sensor_hub.write_command(throttle=throttle, steering=steering)
                self.visualizer.refresh()

                if index >= warmup_frames:
                    enriched = dict(result)
                    enriched["processing_time_s"] = elapsed
                    enriched["control"] = {
                        "throttle": throttle,
                        "steering": steering,
                        "target_speed": target_speed,
                        "measured_speed": measured_speed,
                    }
                    results.append(enriched)
            sensor_hub.stop()
        return results

    @staticmethod
    def summarize_results(results: list[dict]) -> dict:
        if not results:
            return {
                "frames": 0,
                "avg_detected_objects": 0.0,
                "warning_frames": 0,
                "emergency_frames": 0,
            }

        object_counts = [len(item["tracked_objects"]) for item in results]
        warning_frames = sum(1 for item in results if item["decision"]["warning"])
        emergency_frames = sum(1 for item in results if item["decision"]["emergency_brake"])
        processing_times = [float(item.get("processing_time_s", 0.0)) for item in results]
        throttles = [float(item.get("control", {}).get("throttle", 0.0)) for item in results]
        target_speeds = [float(item.get("control", {}).get("target_speed", 0.0)) for item in results]
        measured_speeds = [float(item.get("control", {}).get("measured_speed", 0.0)) for item in results]
        return {
            "frames": len(results),
            "avg_detected_objects": float(sum(object_counts) / len(object_counts)),
            "warning_frames": warning_frames,
            "emergency_frames": emergency_frames,
            "avg_processing_time_s": float(sum(processing_times) / len(processing_times)) if processing_times else 0.0,
            "max_processing_time_s": float(max(processing_times)) if processing_times else 0.0,
            "avg_throttle": float(sum(throttles) / len(throttles)) if throttles else 0.0,
            "avg_target_speed": float(sum(target_speeds) / len(target_speeds)) if target_speeds else 0.0,
            "avg_measured_speed": float(sum(measured_speeds) / len(measured_speeds)) if measured_speeds else 0.0,
        }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    print(f"工程目录: {project_root}")
    print("ObstacleDetectionApp 已就绪。请在 live 模式或测试脚本中调用 process_frame。")


if __name__ == "__main__":
    main()
