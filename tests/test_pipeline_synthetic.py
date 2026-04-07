import time

import numpy as np

from app.decision_bridge import DecisionBridge
from app.fusion_core import FusionCore
from app.longitudinal_control import LongitudinalController
from app.perception_core import PerceptionCore
from app.tracking_core import TrackingCore


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


def test_fusion_transform_rotation():
    fusion = FusionCore()
    sensor_frame = {
        "t": 0.0,
        "point_cloud_sensor": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        "sensor_offset_xy": np.array([0.0, 0.0]),
    }
    ego_state = {
        "t": 0.0,
        "pose": {"x": 1.0, "y": 2.0, "yaw": np.pi / 2},
        "velocity": {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0},
    }
    transformed = fusion.transform_frame(sensor_frame, ego_state)
    points = transformed["point_cloud_world"]
    assert np.allclose(points[0], [1.0, 3.0], atol=1e-6)
    assert np.allclose(points[1], [0.0, 2.0], atol=1e-6)


def test_perception_detects_multiple_clusters():
    perception = PerceptionCore()
    cluster1 = make_cluster((1.6, 0.0), (0.25, 0.25), 30, 1)
    cluster2 = make_cluster((2.4, 1.0), (0.30, 0.20), 32, 2)
    cluster3 = make_cluster((2.0, -1.0), (0.20, 0.35), 28, 3)
    noise = np.array([[3.5, 2.8], [3.8, -2.5], [0.2, 3.9]], dtype=float)
    points = np.vstack((cluster1, cluster2, cluster3, noise))
    ego_state = {"t": 0.0, "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}, "velocity": {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}}

    objects = perception.detect(synthetic_frame(points), ego_state)
    centers = np.array([obj["center"][:2] for obj in objects])

    assert len(objects) == 3
    assert np.min(np.linalg.norm(centers - np.array([1.6, 0.0]), axis=1)) < 0.18
    assert np.min(np.linalg.norm(centers - np.array([2.4, 1.0]), axis=1)) < 0.20
    assert np.min(np.linalg.norm(centers - np.array([2.0, -1.0]), axis=1)) < 0.20


def test_tracking_and_decision():
    perception = PerceptionCore()
    tracking = TrackingCore()
    decision = DecisionBridge()
    ego_state_1 = {"t": 0.0, "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}, "velocity": {"vx": 0.2, "vy": 0.0, "yaw_rate": 0.0}}
    ego_state_2 = {"t": 0.1, "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}, "velocity": {"vx": 0.2, "vy": 0.0, "yaw_rate": 0.0}}

    frame1 = synthetic_frame(make_cluster((0.6, 0.0), (0.15, 0.15), 35, 10), 0.0)
    frame2 = synthetic_frame(make_cluster((0.55, 0.02), (0.15, 0.15), 35, 11), 0.1)

    objects1 = perception.detect(frame1, ego_state_1)
    tracked1 = tracking.update(objects1, ego_state_1)
    objects2 = perception.detect(frame2, ego_state_2)
    tracked2 = tracking.update(objects2, ego_state_2)
    output = decision.evaluate(tracked2, ego_state_2)

    assert len(tracked1) == 1
    assert len(tracked2) == 1
    assert tracked1[0]["id"] == tracked2[0]["id"]
    assert output["warning"] is True
    assert output["emergency_brake"] is False


def test_decision_uses_ego_heading_frame():
    decision = DecisionBridge()
    ego_state = {"t": 0.0, "pose": {"x": 0.0, "y": 0.0, "yaw": -np.pi / 2}}
    tracked_objects = [
        {"center": [0.8, 0.0, 0.0]},   # 世界坐标 x 正方向，不应视为前方
        {"center": [0.0, -0.4, 0.0]},  # 沿车辆朝向前方，应触发制动
    ]

    output = decision.evaluate(tracked_objects, ego_state)

    assert output["warning"] is True
    assert output["emergency_brake"] is True
    center = output["closest_object"]["center"][:2]
    assert np.linalg.norm(np.array(center) - np.array([0.0, -0.4])) < 1e-6


def test_perception_runtime_under_target():
    perception = PerceptionCore()
    cluster1 = make_cluster((1.6, 0.0), (0.25, 0.25), 80, 21)
    cluster2 = make_cluster((2.6, 1.2), (0.30, 0.20), 80, 22)
    cluster3 = make_cluster((2.1, -1.1), (0.25, 0.35), 80, 23)
    cluster4 = make_cluster((3.0, 0.4), (0.35, 0.30), 80, 24)
    noise = np.random.default_rng(25).uniform(low=[-0.5, -2.5], high=[4.0, 2.5], size=(40, 2))
    points = np.vstack((cluster1, cluster2, cluster3, cluster4, noise))
    ego_state = {"t": 0.0, "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}, "velocity": {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}}

    start = time.perf_counter()
    runs = 20
    for i in range(runs):
        perception.detect(synthetic_frame(points, i * 0.1), ego_state)
    elapsed = time.perf_counter() - start
    avg = elapsed / runs

    assert avg < 0.033


def test_longitudinal_controller_targets_and_braking():
    controller = LongitudinalController()

    throttle_clear, target_clear = controller.update(
        measured_speed=0.0,
        decision={"warning": False, "emergency_brake": False, "safe_speed": 0.3},
        dt=0.1,
    )
    throttle_warn, target_warn = controller.update(
        measured_speed=0.2,
        decision={"warning": True, "emergency_brake": False, "safe_speed": 0.1},
        dt=0.1,
    )
    throttle_brake, target_brake = controller.update(
        measured_speed=0.2,
        decision={"warning": True, "emergency_brake": True, "safe_speed": 0.0},
        dt=0.1,
    )

    assert target_clear > target_warn
    assert target_brake == 0.0
    assert throttle_brake <= -0.05
