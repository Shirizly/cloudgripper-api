import os
import logging
import threading
import time
import traceback
import ast
import numpy as np
from dataclasses import dataclass
from typing import Optional
import cv2

# -------------
from grasper import RobotActivity, AutograsperBase
from recording import Recorder
from library.utils import parse_config

logger = logging.getLogger(__name__)

# Global event for error handling across threads
ERROR_EVENT = threading.Event()

# Lock for thread-safe access to bottom image for display
BOTTOM_IMAGE_LOCK = threading.Lock()


@dataclass
class SharedState:
    """
    Holds shared references between threads, e.g.,
    the current robot activity and the recorder.
    """

    state: str = RobotActivity.STARTUP
    recorder: Optional[Recorder] = None
    recorder_thread: Optional[threading.Thread] = None


class DataCollectionCoordinator:
    """
    Orchestrates the autograsper, manages recording,
    and coordinates changes between states.
    """

    def __init__(self, config_file, grasper: AutograsperBase):
        self._load_config(config_file)
        self.shared_state = SharedState()
        self.autograsper = grasper

    def run(self):
        """
        Main controller entry point: starts threads,
        monitors states, and displays bottom image safely in the main thread.
        """
        # Threads
        self.autograsper_thread: Optional[threading.Thread] = None
        self.monitor_thread: Optional[threading.Thread] = None

        try:
            self._start_threads()
            # Main loop: monitor state changes and display the bottom image
            self._handle_state_changes()
        except Exception as e:
            self._handle_error(e)
        finally:
            self._cleanup()
            # Ensure any OpenCV windows are closed in the main thread
            cv2.destroyAllWindows()

    def _load_config(self, config_file: str):
        self.config = parse_config(config_file)
        try:
            experiment_cfg = self.config["experiment"]
            camera_cfg = self.config["camera"]
            self.experiment_name = ast.literal_eval(experiment_cfg["name"])
            self.timeout_between_experiments = ast.literal_eval(
                experiment_cfg["timeout_between_experiments"]
            )
            self.save_data = bool(ast.literal_eval(camera_cfg["record"]))
        except Exception as e:
            raise ValueError("ERROR reading from config.ini: ", e)

    def _start_threads(self):
        """Starts the main autograsper thread and state monitor thread."""
        self.autograsper_thread = threading.Thread(target=self.autograsper.run_grasping)
        self.monitor_thread = threading.Thread(target=self._monitor_state)
        self.autograsper_thread.start()
        self.monitor_thread.start()

    def _monitor_state(self):
        """Monitors and updates shared_state based on autograsper's state."""
        try:
            while not ERROR_EVENT.is_set():
                # Check if a record is requested.
                self._check_if_record_is_requested()
                self.shared_state.state = self.autograsper.state
                # Update robot state based on the recorder's state.
                if self.shared_state.recorder is not None:
                    self.autograsper.robot_state = self.shared_state.recorder.state

                time.sleep(0.1)
        except Exception as e:
            self._handle_error(e)

    def _check_if_record_is_requested(self):
        if (
            self.autograsper.request_state_record
            and self.shared_state.recorder is not None
        ):
            self.shared_state.recorder.take_snapshot += 1
            while self.shared_state.recorder.take_snapshot > 0:
                time.sleep(0.1)
            self.autograsper.request_state_record = False

    def _handle_state_changes(self):
        """Main loop that responds to changes in robot state and displays bottom image."""
        prev_state = RobotActivity.STARTUP
        self.session_dir, self.task_dir, self.restore_dir = "", "", ""

        while not ERROR_EVENT.is_set():
            # Process state changes in a thread-safe manner.
            with threading.Lock():
                current_state = self.shared_state.state
                if current_state != prev_state:
                    self._on_state_transition(prev_state, current_state)

                    if current_state == RobotActivity.ACTIVE:
                        if self.save_data:
                            self._create_new_data_point()
                        self._on_active_state()

                    if current_state == RobotActivity.RESETTING:
                        self._on_resetting_state()

                    if current_state == RobotActivity.FINISHED:
                        self._on_finished_state()
                        break

                    prev_state = current_state

            # Safely display the bottom image in the main thread.
            if self.shared_state.recorder is not None:
                with BOTTOM_IMAGE_LOCK:
                    bottom_img = None
                    if self.shared_state.recorder.bottom_image is not None:
                        # Copy the image under lock to avoid concurrent modifications.
                        bottom_img = self.shared_state.recorder.bottom_image.copy()
                if bottom_img is not None:
                    try:
                        # Display the bottom image in a window.
                        # Import here to ensure it's used only in the main thread.
                        import cv2

                        cv2.imshow("Bottom Image", bottom_img)
                        # cv2.waitKey(1) returns immediately, allowing our loop to continue.
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            ERROR_EVENT.set()
                    except Exception as e:
                        logger.error("Error during image display: %s", e)
            time.sleep(0.1)

    def _on_state_transition(self, old_state, new_state):
        """Optional hook for additional logic during state transitions."""
        if new_state == RobotActivity.STARTUP and old_state != RobotActivity.STARTUP:
            if self.shared_state.recorder:
                self.shared_state.recorder.pause = True
                time.sleep(self.timeout_between_experiments)
                self.shared_state.recorder.pause = False

    def _create_new_data_point(self):
        """Creates a new session folder with `task` and `restore` subfolders."""
        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "recorded_data",
            self.experiment_name,
        )
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

        session_ids = [int(x) for x in os.listdir(base_dir) if x.isdigit()]
        new_id = max(session_ids, default=0) + 1
        self.session_dir = os.path.join(base_dir, str(new_id))

        self.task_dir = os.path.join(self.session_dir, "task")
        self.restore_dir = os.path.join(self.session_dir, "restore")
        os.makedirs(self.task_dir, exist_ok=True)
        os.makedirs(self.restore_dir, exist_ok=True)

    def _on_active_state(self):
        """Actions to perform when transitioning to ACTIVE."""
        if self.save_data:
            self.autograsper.output_dir = self.task_dir
        self._ensure_recorder_running(self.task_dir)
        time.sleep(0.5)
        self.autograsper.start_event.set()

    def _ensure_recorder_running(self, output_dir: str):
        """Ensures the recorder is created and running."""
        if not self.shared_state.recorder:
            self.shared_state.recorder = self._setup_recorder(output_dir)
            self.shared_state.recorder_thread = threading.Thread(
                target=self.shared_state.recorder.record
            )
            self.shared_state.recorder_thread.start()
        if self.save_data:
            self.shared_state.recorder.start_new_recording(output_dir)

    def _setup_recorder(self, output_dir: str) -> Recorder:
        """Initializes and returns a Recorder instance based on config."""
        return Recorder(
            self.config,
            output_dir=output_dir,
        )

    def _on_resetting_state(self):
        """Actions when the robot transitions into RESETTING."""
        # Save success/fail status
        status = "fail" if self.autograsper.failed else "success"
        logger.info(f"Task result: {status}")

        if self.save_data:
            with open(os.path.join(self.session_dir, "status.txt"), "w") as f:
                f.write(status)
            self.autograsper.output_dir = self.restore_dir
            if self.shared_state.recorder:
                self.shared_state.recorder.start_new_recording(self.restore_dir)

    def _on_finished_state(self):
        """Actions when the robot transitions to FINISHED."""
        # Stop recorder
        if self.shared_state.recorder:
            self.shared_state.recorder.stop()
            time.sleep(1)
            if (
                self.shared_state.recorder_thread
                and self.shared_state.recorder_thread.is_alive()
            ):
                self.shared_state.recorder_thread.join()

    def _cleanup(self):
        """Sets the global error event and joins main threads."""
        ERROR_EVENT.set()
        if self.autograsper_thread and self.autograsper_thread.is_alive():
            self.autograsper_thread.join()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join()

    def _handle_error(self, exception: Exception) -> None:
        """Logs exception info and sets ERROR_EVENT."""
        logger.error(f"Error occurred: {exception}")
        logger.error(traceback.format_exc())
        ERROR_EVENT.set()
