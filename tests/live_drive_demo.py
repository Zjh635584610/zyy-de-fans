"""CityscapeLite 场景下的感知驱动 demo。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import ObstacleDetectionApp


def parse_initial_pose(text: str | None):
    if not text:
        return None
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError("OBSTACLE_INITIAL_POSE 必须是 x,y,yaw 三个值")
    return (parts[0], parts[1], parts[2])


def main() -> None:
    scene_name = os.environ.get("OBSTACLE_SCENE", "CityscapeLite")
    initial_pose = parse_initial_pose(os.environ.get("OBSTACLE_INITIAL_POSE"))
    app = ObstacleDetectionApp(
        live=True,
        scene_name=scene_name,
        launch_scene=True,
        enable_gui=True,
        enable_traffic=True,
        initial_pose=initial_pose,
    )
    results = app.run_live_drive(num_frames=24, warmup_frames=8, steering=0.0)
    summary = app.summarize_results(results)
    summary["scene"] = scene_name
    summary["initial_pose"] = list(app.scene.initial_pose)
    summary["gui_enabled"] = app.visualizer.gui_enabled

    out_name = f"live_drive_summary_{scene_name.lower()}.json"
    out_path = ROOT / "docs" / out_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
