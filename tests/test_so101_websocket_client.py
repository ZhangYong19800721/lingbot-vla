import numpy as np
import pytest

from deploy.so101_websocket_client import (
    JOINT_NAMES,
    CalibrationStatusMixin,
    action_row_to_robot_action,
    configure_camera_backend,
    disconnect_robot,
    make_policy_observation,
    validate_action_chunk,
)
from deploy.websocket_client_policy import _build_websocket_uri


def _robot_observation():
    observation = {
        f"{joint}.pos": float(index) for index, joint in enumerate((name.removesuffix(".pos") for name in JOINT_NAMES))
    }
    observation["side"] = np.zeros((32, 40, 3), dtype=np.uint8)
    observation["wrist"] = np.ones((24, 30, 3), dtype=np.uint8)
    return observation


def test_make_policy_observation_maps_so101_state_and_images():
    observation = make_policy_observation(_robot_observation(), "move the block")

    assert observation["observation.state"].tolist() == [0, 1, 2, 3, 4, 5]
    assert observation["observation.images.side"].shape == (32, 40, 3)
    assert observation["observation.images.wrist"].shape == (24, 30, 3)
    assert observation["task"] == "move the block"


def test_action_row_maps_to_lerobot_joint_keys_in_dataset_order():
    action = action_row_to_robot_action(np.arange(6, dtype=np.float32))

    assert list(action) == [f"{name}.pos" for name in JOINT_NAMES]
    assert list(action.values()) == [0, 1, 2, 3, 4, 5]


def test_validate_action_chunk_rejects_wrong_shapes_and_nonfinite_values():
    with pytest.raises(ValueError, match="Expected action chunk shape"):
        validate_action_chunk({"action": np.zeros((25, 7))}, chunk_length=25)

    actions = np.zeros((25, 6), dtype=np.float32)
    actions[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_action_chunk({"action": actions}, chunk_length=25)


def test_disconnect_robot_closes_devices_after_partial_connection():
    class Device:
        def __init__(self, is_connected):
            self.is_connected = is_connected
            self.disconnected = False

        def disconnect(self, *_args):
            self.disconnected = True
            self.is_connected = False

    class Robot:
        is_connected = False
        config = type("Config", (), {"disable_torque_on_disconnect": True})()

        def __init__(self):
            self.bus = Device(is_connected=True)
            self.cameras = {"side": Device(is_connected=False), "wrist": Device(is_connected=True)}

    robot = Robot()
    disconnect_robot(robot)

    assert robot.bus.disconnected
    assert not robot.cameras["side"].disconnected
    assert robot.cameras["wrist"].disconnected


def test_websocket_uri_supports_local_ws_and_remote_wss():
    assert _build_websocket_uri("127.0.0.1", 8006) == "ws://127.0.0.1:8006"
    assert _build_websocket_uri("wss://inference.example.com/infer", 30499) == (
        "wss://inference.example.com:30499/infer"
    )
    assert _build_websocket_uri("wss://inference.example.com:30499/infer", 8006) == (
        "wss://inference.example.com:30499/infer"
    )


def test_linux_camera_backend_uses_v4l2():
    import cv2

    camera = type("Camera", (), {"backend": cv2.CAP_ANY})()
    robot = type("Robot", (), {"cameras": {"side": camera}})()

    configure_camera_backend(robot, platform_name="linux")

    assert camera.backend == cv2.CAP_V4L2


def test_calibration_status_mixin_caches_the_connect_check():
    class Robot:
        calls = 0

        @property
        def is_calibrated(self):
            self.calls += 1
            return True

    class ClientRobot(CalibrationStatusMixin, Robot):
        pass

    robot = ClientRobot()

    assert robot.is_calibrated is True
    assert robot.calibration_check_result is True
    assert robot.calls == 1
