"""QLabs live smoke test.

运行方式建议：
    python tests/live_smoke.py

在自动化环境中可配合 automation/run_with_timeout.py 使用。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import ObstacleDetectionApp


def main() -> None:
    app = ObstacleDetectionApp(live=True, scene_name="Cityscape", launch_scene=True)
    results = app.run_live_steps(num_frames=8, warmup_frames=6)
    summary = app.summarize_results(results)

    out_path = ROOT / "docs" / "live_smoke_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
