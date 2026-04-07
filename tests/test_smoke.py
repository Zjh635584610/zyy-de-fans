from app.decision_bridge import DecisionBridge
from app.fusion_core import FusionCore
from app.main import ObstacleDetectionApp
from app.perception_core import PerceptionCore
from app.scene_manager import prepare_scene
from app.sensor_hub import SensorHub
from app.tracking_core import TrackingCore


def test_smoke_modules_initialize():
    scene = prepare_scene("Cityscape Lite", "default")
    sensor = SensorHub(live=False)
    fusion = FusionCore()
    perception = PerceptionCore()
    tracking = TrackingCore()
    decision = DecisionBridge()
    app = ObstacleDetectionApp(live=False)

    frame = {
        "t": 0.0,
        "point_cloud_world": [],
        "sensor_offset_xy": [0.125, 0.0],
    }
    ego_state = fusion.estimate_pose(frame)
    objects = perception.detect(frame, ego_state)
    tracked = tracking.update(objects, ego_state)
    output = decision.evaluate(tracked, ego_state)
    rendered = app.process_frame(frame)

    assert scene.status == "prepared"
    assert sensor.live is False
    assert isinstance(ego_state, dict)
    assert isinstance(objects, list)
    assert isinstance(tracked, list)
    assert isinstance(output, dict)
    assert isinstance(rendered, dict)
