"""让本车单独跑路线并记录若干位姿点。"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.scene_manager import load_scene_module
from hal.content.qcar_functions import QCarDriveController, QCarEKF
from hal.products.mats import SDCSRoadMap
from pal.products.qcar import IS_PHYSICAL_QCAR, QCar, QCarGPS


def main() -> None:
    controller_update_rate = 100
    tf = 180.0
    start_delay = 1.0
    v_ref = 0.3
    sample_period_s = 0.6
    min_distance_m = 0.12
    scene_name = "CityscapeLite"
    node_sequence = [0, 20, 0, 20, 0]

    roadmap = SDCSRoadMap()
    waypoint_sequence = roadmap.generate_path(node_sequence)
    initial_pose = roadmap.get_node_pose(node_sequence[0]).squeeze()

    if not IS_PHYSICAL_QCAR:
        load_scene_module(scene_name)
        import qlabs_setup

        qlabs_setup.setup(
            initialPosition=[float(initial_pose[0]), float(initial_pose[1]), 0.0],
            initialOrientation=[0.0, 0.0, float(initial_pose[2])],
            enableTraffic=False,
        )
        calibrate = False
    else:
        calibrate = False

    gps = QCarGPS(initialPose=initial_pose, calibrate=calibrate, attach_lidar=False)
    while gps.readGPS():
        pass

    ekf = QCarEKF(x_0=initial_pose)
    drive_controller = QCarDriveController(waypoint_sequence, cyclic=False)
    qcar = QCar(readMode=1, frequency=controller_update_rate)

    samples: list[dict] = []
    last_sample_time = None
    last_sample_pos = None

    with qcar:
        t0 = time.time()
        t = 0.0
        while t < tf:
            tp = t
            t = time.time() - t0
            dt = max(1e-3, t - tp)

            qcar.read()
            if gps.readGPS():
                y_gps = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
                ekf.update([qcar.motorTach, 0.0], dt, y_gps, qcar.gyroscope[2])
            else:
                ekf.update([qcar.motorTach, 0.0], dt, None, qcar.gyroscope[2])

            x = float(ekf.x_hat[0, 0])
            y = float(ekf.x_hat[1, 0])
            yaw = float(ekf.x_hat[2, 0])
            v = float(qcar.motorTach)
            p = np.array([x, y]) + np.array([math.cos(yaw), math.sin(yaw)]) * 0.2

            if t < start_delay:
                throttle = 0.0
                steering = 0.0
            else:
                throttle, steering = drive_controller.update(p, yaw, v, v_ref, dt)
            qcar.write(throttle, steering)

            should_sample = t >= start_delay
            if should_sample and last_sample_time is not None:
                should_sample = (t - last_sample_time) >= sample_period_s
            if should_sample and last_sample_pos is not None:
                should_sample = np.linalg.norm(p - last_sample_pos) >= min_distance_m

            if should_sample:
                sample = {
                    "index": len(samples),
                    "t": round(float(t), 3),
                    "x": round(float(p[0]), 6),
                    "y": round(float(p[1]), 6),
                    "yaw": round(float(yaw), 6),
                }
                samples.append(sample)
                last_sample_time = t
                last_sample_pos = p.copy()

            if drive_controller.steeringController.pathComplete:
                break

        qcar.write(0.0, 0.0)

    out_path = ROOT / "docs" / "route_seed_positions.json"
    payload = {
        "scene": scene_name,
        "node_sequence": node_sequence,
        "initial_pose": [float(initial_pose[0]), float(initial_pose[1]), float(initial_pose[2])],
        "sample_count": len(samples),
        "samples": samples,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
