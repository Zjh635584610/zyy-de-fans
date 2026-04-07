"""带可视化的 QLabs live demo。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import ObstacleDetectionApp


def main() -> None:
    app = ObstacleDetectionApp(
        live=True,
        scene_name="Cityscape",
        launch_scene=True,
        enable_gui=True,
        enable_traffic=True,
    )
    results = app.run_live_steps(num_frames=20, warmup_frames=8)
    summary = app.summarize_results(results)
    summary["gui_enabled"] = app.visualizer.gui_enabled

    out_path = ROOT / "docs" / "live_visual_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
