"""Run the SO-101 follower from a LingBot-VLA WebSocket policy server.

The client reads robot observations, asks the server for a 25-step action chunk,
then optionally executes that chunk on the robot. Execution is opt-in. Without
the flag, it connects to the robot and reads sensors but only prints predictions.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

from .websocket_client_policy import WebsocketClientPolicy


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
CHUNK_LENGTH = 25
logger = logging.getLogger(__name__)


class CalibrationStatusMixin:
    """Retain the calibration result already read by LeRobot's connect()."""

    calibration_check_result: bool | None = None

    @property
    def is_calibrated(self) -> bool:
        result = super().is_calibrated
        self.calibration_check_result = result
        return result


class SO101FollowerClient(CalibrationStatusMixin, SO101Follower):
    """SO101 follower that exposes connect()'s calibration check result."""


def _camera_source(value: str) -> int | Path:
    return int(value) if value.isdecimal() else Path(value)


def make_policy_observation(robot_observation: dict[str, Any], task: str) -> dict[str, Any]:
    """Map LeRobot SO-101 output to the keys configured by robot_configs/so101.yaml."""
    side_image = np.asarray(robot_observation["side"])
    wrist_image = np.asarray(robot_observation["wrist"])
    if side_image.ndim != 3 or side_image.shape[-1] != 3:
        raise ValueError(f"Expected side camera RGB image in HxWx3 format, got {side_image.shape}")
    if wrist_image.ndim != 3 or wrist_image.shape[-1] != 3:
        raise ValueError(f"Expected wrist camera RGB image in HxWx3 format, got {wrist_image.shape}")
    if side_image.dtype != np.uint8 or wrist_image.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB images from both cameras")
    if not task.strip():
        raise ValueError("Task instruction must not be empty")

    state = np.asarray([robot_observation[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError(f"Expected six finite joint positions, got {state}")

    return {
        "observation.images.side": side_image.astype(np.uint8, copy=False),
        "observation.images.wrist": wrist_image.astype(np.uint8, copy=False),
        "observation.state": state,
        "task": task,
    }


def action_row_to_robot_action(action_row: np.ndarray) -> dict[str, float]:
    """Convert the model's [5 arm joints, gripper] row to LeRobot motor keys."""
    action_row = np.asarray(action_row, dtype=np.float32)
    if action_row.shape != (len(JOINT_NAMES),):
        raise ValueError(f"Expected an action row with 6 values, got shape {action_row.shape}")
    if not np.isfinite(action_row).all():
        raise ValueError("The policy returned a non-finite action")
    return {f"{name}.pos": float(value) for name, value in zip(JOINT_NAMES, action_row, strict=True)}


def validate_action_chunk(result: dict[str, Any], chunk_length: int) -> np.ndarray:
    if "action" not in result:
        raise ValueError(f"Policy response has no 'action' field; received keys: {list(result)}")
    action = np.asarray(result["action"], dtype=np.float32)
    expected_shape = (chunk_length, len(JOINT_NAMES))
    if action.shape != expected_shape:
        raise ValueError(f"Expected action chunk shape {expected_shape}, got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("The policy returned a non-finite action chunk")
    return action


def disconnect_robot(robot: SO101Follower) -> None:
    """Close all connected SO-101 devices, including after a partial connect failure."""
    try:
        if robot.is_connected:
            robot.disconnect()
    except Exception:
        logger.exception("LeRobot robot.disconnect() failed; trying to close remaining devices")

    for camera_name, camera in robot.cameras.items():
        if camera.is_connected:
            try:
                camera.disconnect()
            except Exception:
                logger.exception("Failed to disconnect camera %s", camera_name)

    if robot.bus.is_connected:
        try:
            robot.bus.disconnect(robot.config.disable_torque_on_disconnect)
        except Exception:
            logger.exception("Failed to disconnect SO-101 motor bus")


def configure_camera_backend(robot: SO101Follower, platform_name: str | None = None) -> None:
    """Use V4L2 on Linux so OpenCV can select webcam FOURCC/resolution modes."""
    if (platform_name or sys.platform).startswith("linux"):
        import cv2

        for camera in robot.cameras.values():
            camera.backend = cv2.CAP_V4L2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Inference server host, LAN IP, or full ws:// / wss:// URL (may include a path)",
    )
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--robot-port", required=True, help="SO-101 serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--robot-id", required=True, help="ID used when this arm was calibrated")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="LeRobot calibration directory; use this if your LeRobot version stores it under another robot type",
    )
    parser.add_argument("--side-camera", required=True, help="OpenCV index or path for the side camera")
    parser.add_argument("--wrist-camera", required=True, help="OpenCV index or path for the wrist camera")
    parser.add_argument("--task", required=True, help="Task instruction, matching the training data language")
    parser.add_argument(
        "--fps", type=float, default=15, help="Action execution rate; defaults to the deployment guide's 15 Hz"
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=1,
        help="Number of chunks before exit; use 0 to continue until Ctrl-C",
    )
    parser.add_argument(
        "--steps-per-chunk",
        type=int,
        default=CHUNK_LENGTH,
        help="Number of predicted actions to execute from each chunk (1-25)",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=5.0,
        help="Per-command change cap in arm degrees / gripper units; not an absolute workspace limit; 0 disables it",
    )
    parser.add_argument("--side-camera-width", type=int, default=640)
    parser.add_argument("--side-camera-height", type=int, default=480)
    parser.add_argument("--side-camera-fps", type=int, default=30)
    parser.add_argument("--side-camera-fourcc", default=None, help="Optional camera format, e.g. MJPG")
    parser.add_argument("--wrist-camera-width", type=int, default=640)
    parser.add_argument("--wrist-camera-height", type=int, default=480)
    parser.add_argument("--wrist-camera-fps", type=int, default=30)
    parser.add_argument("--wrist-camera-fourcc", default=None, help="Optional camera format, e.g. MJPG")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send predicted actions to the robot; without this flag the client only prints predictions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")
    if args.max_chunks < 0:
        raise ValueError("--max-chunks must be zero (unlimited) or positive")
    if not 1 <= args.steps_per_chunk <= CHUNK_LENGTH:
        raise ValueError(f"--steps-per-chunk must be between 1 and {CHUNK_LENGTH}")
    if args.max_relative_target < 0:
        raise ValueError("--max-relative-target cannot be negative")

    max_relative_target = args.max_relative_target or None
    camera_config = {
        "side": OpenCVCameraConfig(
            index_or_path=_camera_source(args.side_camera),
            width=args.side_camera_width,
            height=args.side_camera_height,
            fps=args.side_camera_fps,
            fourcc=args.side_camera_fourcc,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=_camera_source(args.wrist_camera),
            width=args.wrist_camera_width,
            height=args.wrist_camera_height,
            fps=args.wrist_camera_fps,
            fourcc=args.wrist_camera_fourcc,
        ),
    }
    robot = SO101FollowerClient(
        SO101FollowerConfig(
            port=args.robot_port,
            id=args.robot_id,
            calibration_dir=args.calibration_dir,
            cameras=camera_config,
            use_degrees=True,
            max_relative_target=max_relative_target,
        )
    )
    configure_camera_backend(robot)
    policy: WebsocketClientPolicy | None = None

    try:
        policy = WebsocketClientPolicy(host=args.host, port=args.port)
        policy.reset("so101")

        # Use the saved calibration. This client intentionally never starts calibration itself.
        robot.connect(calibrate=False)
        if robot.calibration_check_result is not True:
            raise RuntimeError(
                f"SO-101 has no loaded calibration at {robot.calibration_fpath}. "
                "Calibrate it separately or pass the existing file's directory with --calibration-dir."
            )

        chunk_count = 0
        while args.max_chunks == 0 or chunk_count < args.max_chunks:
            raw_observation = robot.get_observation()
            policy_observation = make_policy_observation(raw_observation, args.task)
            action_chunk = validate_action_chunk(policy.infer(policy_observation), CHUNK_LENGTH)
            chunk_count += 1

            print(f"Chunk {chunk_count}: shape={action_chunk.shape}; execute={args.execute}")
            print("First predicted target:", action_row_to_robot_action(action_chunk[0]))
            if not args.execute:
                continue

            deadline = time.monotonic()
            for row in action_chunk[: args.steps_per_chunk]:
                robot.send_action(action_row_to_robot_action(row))
                deadline += 1.0 / args.fps
                time.sleep(max(0.0, deadline - time.monotonic()))

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        try:
            disconnect_robot(robot)
        finally:
            if policy is not None:
                policy.close()


if __name__ == "__main__":
    main()
