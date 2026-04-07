"""V1 验证脚本。

运行方式：
    python tests/run_validation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.decision_bridge import DecisionBridge
from app.main import ObstacleDetectionApp


def make_cluster(center, size, count, seed):
    rng = np.random.default_rng(seed)
    offsets = (rng.random((count, 2)) - 0.5) * np.array(size)
    return np.array(center, dtype=float) + offsets


def synthetic_frame(points, t=0.0):
    return {
        "t": t,
        "dt": 0.1,
        "point_cloud_world": points,
        "sensor_offset_xy": np.array([0.125, 0.0]),
    }


def compute_min_center_error(detected_centers, reference_centers):
    errors = []
    for ref in reference_centers:
        errors.append(float(np.min(np.linalg.norm(detected_centers - ref, axis=1))))
    return errors


def main() -> None:
    app = ObstacleDetectionApp(live=False)
    decision = DecisionBridge()

    ref_centers = np.array([[1.6, 0.0], [2.4, 1.0], [2.0, -1.0]], dtype=float)
    points = np.vstack(
        (
            make_cluster((1.6, 0.0), (0.25, 0.25), 30, 1),
            make_cluster((2.4, 1.0), (0.30, 0.20), 32, 2),
            make_cluster((2.0, -1.0), (0.20, 0.35), 28, 3),
            np.array([[3.5, 2.8], [3.8, -2.5], [0.2, 3.9]], dtype=float),
        )
    )

    result = app.process_frame(synthetic_frame(points, 0.0))
    objects = result["tracked_objects"]
    centers = np.array([obj["center"][:2] for obj in objects], dtype=float)
    center_errors = compute_min_center_error(centers, ref_centers)

    frame1 = synthetic_frame(make_cluster((0.6, 0.0), (0.15, 0.15), 35, 10), 0.0)
    frame2 = synthetic_frame(make_cluster((0.55, 0.02), (0.15, 0.15), 35, 11), 0.1)
    tracked1 = app.process_frame(frame1)["tracked_objects"]
    tracked2 = app.process_frame(frame2)["tracked_objects"]
    decision_output = decision.evaluate(tracked2, {"t": 0.1, "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}})

    perf_points = np.vstack(
        (
            make_cluster((1.6, 0.0), (0.25, 0.25), 80, 21),
            make_cluster((2.6, 1.2), (0.30, 0.20), 80, 22),
            make_cluster((2.1, -1.1), (0.25, 0.35), 80, 23),
            make_cluster((3.0, 0.4), (0.35, 0.30), 80, 24),
            np.random.default_rng(25).uniform(low=[-0.5, -2.5], high=[4.0, 2.5], size=(40, 2)),
        )
    )

    runs = 50
    times = []
    for i in range(runs):
        start = time.perf_counter()
        app.process_frame(synthetic_frame(perf_points, i * 0.1))
        times.append(time.perf_counter() - start)

    avg_time = float(np.mean(times))
    max_time = float(np.max(times))

    print("=== V1 Validation Summary ===")
    print(f"detected_objects={len(objects)}")
    print(f"center_errors_m={center_errors}")
    print(f"track_id_continuity={tracked1[0]['id'] == tracked2[0]['id'] if tracked1 and tracked2 else False}")
    print(f"warning={decision_output['warning']}, emergency_brake={decision_output['emergency_brake']}")
    print(f"avg_processing_time_s={avg_time:.6f}")
    print(f"max_processing_time_s={max_time:.6f}")


if __name__ == "__main__":
    main()
