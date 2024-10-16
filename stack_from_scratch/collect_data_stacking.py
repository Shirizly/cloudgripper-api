import argparse
import logging
import os
import asyncio
import json
import traceback
from configparser import ConfigParser
from typing import Optional, Tuple

import numpy as np

from autograsper import Autograsper, RobotActivity
from client.cloudgripper_client import GripperRobot
from recording import Recorder
from stackingtask import StackingTask

# Initialize logger
logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'default': {'format': '%(asctime)s %(levelname)s:%(name)s:%(message)s'}
    },
    'handlers': {
        'console': {'class': 'logging.StreamHandler', 'formatter': 'default'}
    },
    'root': {'handlers': ['console'], 'level': 'INFO'},
})
logger = logging.getLogger(__name__)

ERROR_EVENT = asyncio.Event()
bottom_image_lock = asyncio.Lock()


class TaskManager:
    """Manages the lifecycle of asyncio tasks."""

    def __init__(self):
        self.tasks = []

    def create_task(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        return task

    async def cancel_all_tasks(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


class RecorderManager:
    """Manages the Recorder instance and recording sessions."""

    def __init__(self, output_dir: str, robot_idx: int, config: dict):
        self.output_dir = output_dir
        self.recorder = self.setup_recorder(output_dir, robot_idx, config)
        self.recording_task: Optional[asyncio.Task] = None

    def setup_recorder(self, output_dir: str, robot_idx: int, config: dict) -> Recorder:
        session_id = "test"
        camera_matrix = np.array(config["camera"]["m"])
        distortion_coefficients = np.array(config["camera"]["d"])
        token = os.getenv("ROBOT_TOKEN")
        if not token:
            logger.error("ROBOT_TOKEN environment variable not set")
            raise ValueError("ROBOT_TOKEN environment variable not set")
        return Recorder(session_id, output_dir, camera_matrix, distortion_coefficients, token, robot_idx)

    async def start_recording(self, task_dir: str):
        await self.recorder.start_new_recording(task_dir)
        # Start the recording loop in an asyncio task
        self.recording_task = asyncio.create_task(self.recorder.record())

    async def stop_recording(self):
        await self.recorder.stop()
        if self.recording_task:
            await self.recording_task  # Wait for the recording task to finish
            self.recording_task = None
        await self.recorder.write_final_image()


class SharedState:
    """Shared state between tasks."""

    def __init__(self):
        self.state: RobotActivity = RobotActivity.STARTUP
        self.recorder_manager: Optional[RecorderManager] = None


shared_state = SharedState()


def load_config(config_file: str) -> dict:
    """Loads and parses the configuration file."""
    config = ConfigParser()
    config.read(config_file)
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for key, value in config.items(section):
            if value.startswith("[") or value.startswith("{"):
                try:
                    config_dict[section][key] = json.loads(value)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in config for {key}: {value}")
                    config_dict[section][key] = value
            else:
                config_dict[section][key] = value
    return config_dict


def validate_config(config: dict) -> None:
    """Validates the configuration parameters."""
    required_keys = {
        "experiment": ["colors", "block_heights"],
        "camera": ["m", "d"],
    }
    for section, keys in required_keys.items():
        if section not in config:
            raise ValueError(f"Missing '{section}' section in configuration")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing '{key}' in {section} configuration")


def get_new_session_id(base_dir: str) -> int:
    """Generates a new session ID based on existing directories."""
    if not os.path.exists(base_dir):
        return 1
    session_ids = [
        int(dir_name) for dir_name in os.listdir(base_dir) if dir_name.isdigit()
    ]
    return max(session_ids, default=0) + 1


def handle_error(exception: Exception) -> None:
    """Handles exceptions by logging and setting the error event."""
    logger.error(f"Error occurred: {exception}")
    logger.error(traceback.format_exc())
    ERROR_EVENT.set()


def create_new_data_point(script_dir: str) -> Tuple[str, str, str]:
    """Creates directories for a new data point session."""
    recorded_data_dir = os.path.join(script_dir, "recorded_data")
    new_session_id = get_new_session_id(recorded_data_dir)
    new_session_dir = os.path.join(recorded_data_dir, str(new_session_id))
    task_dir = os.path.join(new_session_dir, "task")
    restore_dir = os.path.join(new_session_dir, "restore")
    try:
        os.makedirs(task_dir, exist_ok=True)
        os.makedirs(restore_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directories: {e}")
        raise
    return new_session_dir, task_dir, restore_dir


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Robot Controller")
    parser.add_argument("--robot_idx", type=str,
                        required=True, help="Robot index")
    parser.add_argument(
        "--config",
        type=str,
        default="stack_from_scratch/config.ini",
        help="Path to the configuration file",
    )
    return parser.parse_args()


def initialize(args: argparse.Namespace) -> Tuple[Autograsper, dict, str]:
    """Initializes the Autograsper and loads configuration."""
    config = load_config(args.config)
    validate_config(config)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    token = os.getenv("ROBOT_TOKEN")
    if not token:
        logger.error("ROBOT_TOKEN environment variable not set")
        raise ValueError("ROBOT_TOKEN environment variable not set")

    robot = GripperRobot(robot_idx=int(args.robot_idx), token=token)
    autograsper = Autograsper(
        robot=robot, config=config, robot_idx=int(args.robot_idx))
    return autograsper, config, script_dir


async def run_autograsper(autograsper: Autograsper, task: StackingTask) -> None:
    """Runs the autograsper asynchronously."""
    try:
        await autograsper.run_task(task)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        handle_error(e)


async def monitor_state(autograsper: Autograsper) -> None:
    """Monitors the state of the autograsper."""
    try:
        while not ERROR_EVENT.is_set():
            if shared_state.state != autograsper.state:
                shared_state.state = autograsper.state
                if shared_state.state == RobotActivity.FINISHED:
                    break
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        handle_error(e)


async def monitor_bottom_image(recorder: Recorder, autograsper: Autograsper) -> None:
    """Monitors and updates the bottom image from the recorder."""
    try:
        while not ERROR_EVENT.is_set():
            async with bottom_image_lock:
                if recorder and recorder.bottom_image is not None:
                    autograsper.bottom_image = np.copy(recorder.bottom_image)
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        handle_error(e)


async def start_new_recording(session_dir: str, task_dir: str, autograsper: Autograsper, args: argparse.Namespace, config: dict, task_manager: TaskManager) -> None:
    """Starts a new recording session."""
    autograsper.output_dir = task_dir
    if not shared_state.recorder_manager:
        shared_state.recorder_manager = RecorderManager(
            task_dir, int(args.robot_idx), config)
        task_manager.create_task(monitor_bottom_image(
            shared_state.recorder_manager.recorder, autograsper))
    await shared_state.recorder_manager.start_recording(task_dir)
    await asyncio.sleep(0.5)
    await autograsper.set_start_flag()


async def reset_experiment(session_dir: str, restore_dir: str, autograsper: Autograsper, task: StackingTask) -> None:
    """Resets the experiment after completion."""
    status_message = (
        "success"
        if not await task.detect_errors(autograsper)
        else "fail"
    )

    logger.info(f"Experiment status: {status_message}")
    try:
        with open(os.path.join(session_dir, "status.txt"), "w") as status_file:
            status_file.write(status_message)
    except OSError as e:
        logger.error(f"Failed to write status file: {e}")

    autograsper.output_dir = restore_dir
    await shared_state.recorder_manager.start_recording(restore_dir)


async def handle_state_changes(
    autograsper: Autograsper,
    config: dict,
    script_dir: str,
    task: StackingTask,
    args: argparse.Namespace,
    task_manager: TaskManager,
) -> None:
    """Handles changes in the robot's state."""
    prev_robot_activity = RobotActivity.STARTUP
    session_dir, task_dir, restore_dir = "", "", ""

    while not ERROR_EVENT.is_set():
        if shared_state.state != prev_robot_activity:
            if prev_robot_activity != RobotActivity.STARTUP and shared_state.recorder_manager:
                await shared_state.recorder_manager.stop_recording()

            if shared_state.state == RobotActivity.ACTIVE:
                session_dir, task_dir, restore_dir = create_new_data_point(
                    script_dir)
                await start_new_recording(session_dir, task_dir, autograsper, args, config, task_manager)

            elif shared_state.state == RobotActivity.RESETTING:
                await reset_experiment(session_dir, restore_dir, autograsper, task)

            prev_robot_activity = shared_state.state

        if shared_state.state == RobotActivity.FINISHED:
            if shared_state.recorder_manager:
                await shared_state.recorder_manager.stop_recording()
                await asyncio.sleep(1)
            break

        await asyncio.sleep(0.1)


async def main():
    args = parse_arguments()
    autograsper, config, script_dir = initialize(args)
    task_manager = TaskManager()

    colors = config["experiment"]["colors"]
    block_heights = np.array(config["experiment"]["block_heights"])

    stacking_task = StackingTask(colors, block_heights, config)

    autograsper_task = task_manager.create_task(
        run_autograsper(autograsper, stacking_task))
    monitor_state_task = task_manager.create_task(monitor_state(autograsper))

    try:
        await handle_state_changes(autograsper, config, script_dir, stacking_task, args, task_manager)
        await autograsper_task
        await monitor_state_task
    except Exception as e:
        handle_error(e)
    finally:
        ERROR_EVENT.set()
        await task_manager.cancel_all_tasks()
        await autograsper.shutdown()  # Ensure proper cleanup

if __name__ == "__main__":
    asyncio.run(main())
