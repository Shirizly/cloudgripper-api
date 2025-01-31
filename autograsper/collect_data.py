import argparse
import logging
import os
import threading
import time
import traceback
from configparser import ConfigParser
from typing import Optional, Tuple

import numpy as np

from grasper import AutograsperBase
from stacking_autograsper import StackingAutograsper, RobotActivity
from random_grasping_task import RandomGrasper
from recording import Recorder
from library.rgb_object_tracker import all_objects_are_visible

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants & Globals ---
STATE_LOCK = threading.Lock()
BOTTOM_IMAGE_LOCK = threading.Lock()
ERROR_EVENT = threading.Event()
TIME_BETWEEN_EXPERIMENTS = 10


class SharedState:
    """Holds shared references between threads (e.g., current robot activity, recorder)."""
    def __init__(self):
        self.state: RobotActivity = RobotActivity.STARTUP
        self.recorder: Optional[Recorder] = None
        self.recorder_thread: Optional[threading.Thread] = None
        self.bottom_image_thread: Optional[threading.Thread] = None


def load_config(config_file: str = "autograsper/config.ini") -> ConfigParser:
    config = ConfigParser()
    config.read(config_file)
    return config


def get_new_session_id(base_dir: str) -> int:
    """Returns a new numeric session ID based on existing directories."""
    if not os.path.exists(base_dir):
        return 1
    session_ids = [
        int(dir_name) for dir_name in os.listdir(base_dir) if dir_name.isdigit()
    ]
    return max(session_ids, default=0) + 1


def handle_error(exception: Exception) -> None:
    """Logs exception info and sets ERROR_EVENT."""
    logger.error(f"Error occurred: {exception}")
    logger.error(traceback.format_exc())
    ERROR_EVENT.set()


def monitor_state(autograsper: AutograsperBase, shared_state: SharedState) -> None:
    """Updates shared_state based on autograsper's state, and signals completion when FINISHED."""
    try:
        while not ERROR_EVENT.is_set():
            with STATE_LOCK:
                if shared_state.state != autograsper.state:
                    shared_state.state = autograsper.state
                    if shared_state.state == RobotActivity.FINISHED:
                        break
            time.sleep(0.1)
    except Exception as e:
        handle_error(e)


def monitor_bottom_image(recorder: Recorder, autograsper: AutograsperBase) -> None:
    """Copies the latest bottom image from the recorder to the autograsper."""
    try:
        while not ERROR_EVENT.is_set():
            if recorder and recorder.bottom_image is not None:
                with BOTTOM_IMAGE_LOCK:
                    autograsper.bottom_image = np.copy(recorder.bottom_image)
            time.sleep(0.1)
    except Exception as e:
        handle_error(e)


def create_new_data_point(script_dir: str) -> Tuple[str, str, str]:
    """Creates a new session folder with `task` and `restore` subfolders."""
    recorded_data_dir = os.path.join(script_dir, "recorded_data")
    new_session_id = get_new_session_id(recorded_data_dir)
    new_session_dir = os.path.join(recorded_data_dir, str(new_session_id))
    task_dir = os.path.join(new_session_dir, "task")
    restore_dir = os.path.join(new_session_dir, "restore")

    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(restore_dir, exist_ok=True)

    return new_session_dir, task_dir, restore_dir


def setup_recorder(output_dir: str, robot_idx: str, config: ConfigParser) -> Recorder:
    """Configures and returns a Recorder instance."""
    camera_matrix = np.array(eval(config["camera"]["m"]))
    distortion_coeffs = np.array(eval(config["camera"]["d"]))
    token = os.getenv("CLOUDGRIPPER_TOKEN")
    if not token:
        raise ValueError("CLOUDGRIPPER_TOKEN environment variable not set")

    # In practice, you might want a unique session_id per run or folder name
    return Recorder(
        session_id="test",
        output_dir=output_dir,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        token=token,
        robot_idx=robot_idx
    )


class RobotController:
    """Main controller that orchestrates the autograsper, recording, and state management."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = load_config(args.config)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.shared_state = SharedState()
        self.autograsper: AutograsperBase = self._initialize_autograsper()
        self.autograsper_thread: Optional[threading.Thread] = None
        self.monitor_thread: Optional[threading.Thread] = None

    def _initialize_autograsper(self) -> AutograsperBase:
        """Instantiates the appropriate Autograsper with config-based parameters."""
        import ast

        # Verify required config sections
        if "experiment" not in self.config:
            raise KeyError("The 'experiment' section is missing from config.")
        if "camera" not in self.config:
            raise KeyError("The 'camera' section is missing from config.")

        # Read experiment parameters
        colors = ast.literal_eval(self.config["experiment"]["colors"])
        block_heights = np.array(ast.literal_eval(self.config["experiment"]["block_heights"]))
        position_bank = ast.literal_eval(self.config["experiment"]["position_bank"])
        stack_position = ast.literal_eval(self.config["experiment"]["stack_position"])
        object_size = self.config.getfloat("experiment", "object_size")

        # Read camera calibration
        camera_matrix = np.array(ast.literal_eval(self.config["camera"]["m"]))
        distortion_coeffs = np.array(ast.literal_eval(self.config["camera"]["d"]))

        # Create an Autograsper (RandomGrasper or StackingAutograsper)
        return RandomGrasper(
            self.args,
            output_dir="",
            colors=colors,
            block_heights=block_heights,
            position_bank=position_bank,
            stack_position=stack_position,
            object_size=object_size,
            camera_matrix=camera_matrix,
            distortion_coeffs=distortion_coeffs,
        )

    def start_threads(self):
        """Starts the main autograsper thread and a monitor thread for state changes."""
        self.autograsper_thread = threading.Thread(target=self.autograsper.run_grasping)
        self.monitor_thread = threading.Thread(
            target=monitor_state,
            args=(self.autograsper, self.shared_state),
        )

        self.autograsper_thread.start()
        self.monitor_thread.start()

    def handle_state_changes(self):
        """Core loop that reacts to changes in robot state and manages recorder logic."""
        prev_state = RobotActivity.STARTUP
        session_dir, task_dir, restore_dir = "", "", ""

        while not ERROR_EVENT.is_set():
            with STATE_LOCK:
                curr_state = self.shared_state.state
                if curr_state != prev_state:
                    # Handle transitions from one state to another
                    self._on_state_transition(prev_state, curr_state)

                    # If the new state is ACTIVE, create new data point, set up recording
                    if curr_state == RobotActivity.ACTIVE:
                        session_dir, task_dir, restore_dir = create_new_data_point(self.script_dir)
                        self._start_active_session(task_dir)

                    # If the new state is RESETTING, finalize the task session, record outcome
                    if curr_state == RobotActivity.RESETTING:
                        self._finalize_task_session(session_dir, restore_dir)

                    # If the new state is FINISHED, stop everything
                    if curr_state == RobotActivity.FINISHED:
                        self._stop_recording()
                        break

                    prev_state = curr_state

            time.sleep(0.1)

    def _on_state_transition(self, prev_state: RobotActivity, new_state: RobotActivity):
        """Hook for additional actions on specific state transitions."""
        # Example: Pause recording briefly between experiments
        if new_state == RobotActivity.STARTUP and prev_state != RobotActivity.STARTUP:
            if self.shared_state.recorder:
                self.shared_state.recorder.pause = True
                time.sleep(TIME_BETWEEN_EXPERIMENTS)
                self.shared_state.recorder.pause = False

    def _start_active_session(self, task_dir: str):
        """Configures the autograsper and recorder for a new active session."""
        # Assign output directory for data from this session
        self.autograsper.output_dir = task_dir

        # Create a recorder if it doesn't exist yet
        if not self.shared_state.recorder:
            self.shared_state.recorder = setup_recorder(
                task_dir, self.args.robot_idx, self.config
            )
            self.shared_state.recorder_thread = threading.Thread(
                target=self.shared_state.recorder.record, args=()
            )
            self.shared_state.recorder_thread.start()

            # Start the bottom image monitoring thread
            self.shared_state.bottom_image_thread = threading.Thread(
                target=monitor_bottom_image,
                args=(self.shared_state.recorder, self.autograsper),
            )
            self.shared_state.bottom_image_thread.start()

        # Start a new recording in the same recorder
        self.shared_state.recorder.start_new_recording(task_dir)
        time.sleep(0.5)

        # Signal the autograsper to begin
        self.autograsper.start_flag = True

    def _finalize_task_session(self, session_dir: str, restore_dir: str):
        """Records success/fail status of the task, then switches recorder to restore_dir."""
        status_message = "fail" if self.autograsper.failed else "success"
        logger.info(f"Task finished with status: {status_message}")

        # Save the status to a text file
        with open(os.path.join(session_dir, "status.txt"), "w") as status_file:
            status_file.write(status_message)

        # Prepare to record the restore operation
        self.autograsper.output_dir = restore_dir
        self.shared_state.recorder.start_new_recording(restore_dir)

    def _stop_recording(self):
        """Stops the recorder and joins threads."""
        if self.shared_state.recorder:
            self.shared_state.recorder.stop()
            time.sleep(1)
        if self.shared_state.recorder_thread and self.shared_state.recorder_thread.is_alive():
            self.shared_state.recorder_thread.join()
        if self.shared_state.bottom_image_thread and self.shared_state.bottom_image_thread.is_alive():
            self.shared_state.bottom_image_thread.join()

    def cleanup(self):
        """Sets the error event and joins main threads."""
        ERROR_EVENT.set()
        if self.autograsper_thread and self.autograsper_thread.is_alive():
            self.autograsper_thread.join()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join()

    def run(self):
        """Main entry point to start threads, watch for state changes, and handle errors."""
        try:
            self.start_threads()
            self.handle_state_changes()
        except Exception as e:
            handle_error(e)
        finally:
            self.cleanup()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot Controller")
    parser.add_argument("--robot_idx", type=str, required=True, help="Robot index")
    parser.add_argument(
        "--config",
        type=str,
        default="autograsper/config.ini",
        help="Path to the configuration file",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    controller = RobotController(args)
    controller.run()


if __name__ == "__main__":
    main()
