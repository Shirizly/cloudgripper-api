import argparse
import logging
import os
import asyncio
import traceback
from configparser import ConfigParser
from typing import Optional, Tuple, Any, List

import numpy as np
from autograsper import Autograsper, RobotActivity
from recording import Recorder

from library.rgb_object_tracker import all_objects_are_visible

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ERROR_EVENT = asyncio.Event()


class SharedState:
    def __init__(self):
        self.state: RobotActivity = RobotActivity.STARTUP
        self.recorder: Optional[Recorder] = None


shared_state = SharedState()


def load_config(config_file: str = "stack_from_scratch/config.ini") -> dict:
    config = ConfigParser()
    config.read(config_file)
    return {section: dict(config.items(section)) for section in config.sections()}


def get_new_session_id(base_dir: str) -> int:
    if not os.path.exists(base_dir):
        return 1
    session_ids = [
        int(dir_name) for dir_name in os.listdir(base_dir) if dir_name.isdigit()
    ]
    return max(session_ids, default=0) + 1


def handle_error(exception: Exception) -> None:
    logger.error(f"Error occurred: {exception}")
    logger.error(traceback.format_exc())
    ERROR_EVENT.set()


def setup_recorder(output_dir: str, robot_idx: str, config: dict) -> Recorder:
    session_id = "test"
    camera_matrix = np.array(eval(config["camera"]["m"]))
    distortion_coefficients = np.array(eval(config["camera"]["d"]))
    token = os.getenv("ROBOT_TOKEN")
    if not token:
        raise ValueError("ROBOT_TOKEN environment variable not set")
    return Recorder(session_id, output_dir, camera_matrix, distortion_coefficients, token, robot_idx)


def create_new_data_point(script_dir: str) -> Tuple[str, str, str]:
    recorded_data_dir = os.path.join(script_dir, "recorded_data")
    new_session_id = get_new_session_id(recorded_data_dir)
    new_session_dir = os.path.join(recorded_data_dir, str(new_session_id))
    task_dir = os.path.join(new_session_dir, "task")
    restore_dir = os.path.join(new_session_dir, "restore")

    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(restore_dir, exist_ok=True)

    return new_session_dir, task_dir, restore_dir


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot Controller")
    parser.add_argument("--robot_idx", type=str, required=True, help="Robot index")
    parser.add_argument(
        "--config",
        type=str,
        default="config.ini",
        help="Path to the configuration file",
    )
    return parser.parse_args()


def initialize(args: argparse.Namespace) -> Tuple[Autograsper, dict, str]:
    config = load_config(args.config)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    autograsper = Autograsper(args, config)
    return autograsper, config, script_dir


async def run_autograsper(autograsper: Autograsper, colors: List[str], block_heights: np.ndarray, config: dict) -> None:
    try:
        await asyncio.to_thread(autograsper.run_grasping, colors, block_heights, config)
    except Exception as e:
        handle_error(e)


async def run_recorder(recorder: Recorder) -> None:
    try:
        await asyncio.to_thread(recorder.record)
    except Exception as e:
        handle_error(e)


async def monitor_state(autograsper: Autograsper) -> None:
    try:
        while not ERROR_EVENT.is_set():
            if shared_state.state != autograsper.state:
                shared_state.state = autograsper.state
                if shared_state.state == RobotActivity.FINISHED:
                    break
            await asyncio.sleep(0.1)
    except Exception as e:
        handle_error(e)


async def monitor_bottom_image(recorder: Recorder, autograsper: Autograsper) -> None:
    try:
        while not ERROR_EVENT.is_set():
            if recorder and recorder.bottom_image is not None:
                autograsper.bottom_image = np.copy(recorder.bottom_image)
            await asyncio.sleep(0.1)
    except Exception as e:
        handle_error(e)


async def handle_state_changes(
    autograsper: Autograsper,
    config: dict,
    script_dir: str,
    colors: List[str],
    args: argparse.Namespace,
) -> None:
    prev_robot_activity = RobotActivity.STARTUP
    session_dir, task_dir, restore_dir = "", "", ""

    while not ERROR_EVENT.is_set():
        if shared_state.state != prev_robot_activity:
            if (
                prev_robot_activity != RobotActivity.STARTUP
                and shared_state.recorder
            ):
                await asyncio.to_thread(shared_state.recorder.write_final_image)

            if shared_state.state == RobotActivity.ACTIVE:
                session_dir, task_dir, restore_dir = create_new_data_point(
                    script_dir
                )
                autograsper.output_dir = task_dir

                if not shared_state.recorder:
                    shared_state.recorder = setup_recorder(
                        task_dir, args.robot_idx, config
                    )
                    asyncio.create_task(run_recorder(shared_state.recorder))
                    asyncio.create_task(monitor_bottom_image(shared_state.recorder, autograsper))

                await asyncio.to_thread(shared_state.recorder.start_new_recording, task_dir)
                await asyncio.sleep(0.5)
                autograsper.start_flag = True

            elif shared_state.state == RobotActivity.RESETTING:
                status_message = (
                    "success"
                    if not all_objects_are_visible(colors, shared_state.recorder.bottom_image, debug=False)
                    else "fail"
                )
                if status_message == "fail":
                    autograsper.failed = True

                logger.info(status_message)
                with open(
                    os.path.join(session_dir, "status.txt"), "w"
                ) as status_file:
                    status_file.write(status_message)

                autograsper.output_dir = restore_dir
                await asyncio.to_thread(shared_state.recorder.start_new_recording, restore_dir)

            prev_robot_activity = shared_state.state

        if shared_state.state == RobotActivity.FINISHED:
            if shared_state.recorder:
                shared_state.recorder.stop()
                await asyncio.sleep(1)
            break

        await asyncio.sleep(0.1)


async def main():
    args = parse_arguments()
    autograsper, config, script_dir = initialize(args)

    colors = eval(config["experiment"]["colors"])
    block_heights = np.array(eval(config["experiment"]["block_heights"]))

    autograsper_task = asyncio.create_task(run_autograsper(autograsper, colors, block_heights, config))
    monitor_state_task = asyncio.create_task(monitor_state(autograsper))

    try:
        await handle_state_changes(autograsper, config, script_dir, colors, args)
    except Exception as e:
        handle_error(e)
    finally:
        ERROR_EVENT.set()
        await asyncio.gather(autograsper_task, monitor_state_task)


if __name__ == "__main__":
    asyncio.run(main())