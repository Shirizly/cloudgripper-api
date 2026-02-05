# coordinator.py
import os
import logging
import time
import concurrent.futures
import threading
from dataclasses import dataclass, field
from queue import Queue, Empty
import numpy as np
import cv2

from grasper import RobotActivity, AutograsperBase
from image_collector.cv2displayer import update_images
from recording import Recorder
from file_manager import FileManager

logger = logging.getLogger(__name__)


@dataclass
class SharedState:
    """
    Holds shared references between threads.
    """
    state: str = RobotActivity.STARTUP
    latest_top_image: np.ndarray | None = None
    latest_bottom_image: np.ndarray | None = None
    latest_robot_state: dict | None = None
    timestamp: float | None = None
    image_lock: threading.RLock = field(default_factory=threading.RLock)


class DataCollectionCoordinator:
    """
    Orchestrates the autograsper, manages recording, and coordinates state changes
    and image updates via a centralized message queue.
    """

    def __init__(
        self, config, grasper: AutograsperBase, shutdown_event: threading.Event, visualize: bool = False):
        self.config = config
        self.shutdown_event = shutdown_event
        self.shared_state = SharedState()
        self.recorder = None
        self.autograsper = grasper
        self.autograsper.connect_shared_state(self.shared_state)
        self.visualize = visualize
        cv2.startWindowThread()
        # Message queue for non-UI messages.
        self.msg_queue = Queue()
        # Separate UI queue so that image updates can be handled in the main thread.
        self.ui_queue = Queue()


        # Read configuration with explicit error handling.
        try:
            experiment_config = config["experiment"]
            camera_config = config["camera"]
            self.experiment_name = experiment_config["name"]
            self.timeout_between_experiments = experiment_config[
                "timeout_between_experiments"
            ]
            self.save_data = camera_config["record"]
        except KeyError as e:
            raise ValueError(f"Missing configuration key in coordinator: {e}") from e
        except TypeError as e:
            raise ValueError(f"Invalid configuration format in coordinator: {e}") from e

    def _monitor_state(self):
        """
        Polls the autograsper and recorder for updates and posts messages to the message queue.
        """
        while not self.shutdown_event.is_set():
            try:
                # Post state update message.
                state_msg = {"type": "state_update", "state": self.autograsper.state}
                self.msg_queue.put(state_msg)

                self._check_if_record_is_requested()
                # If a recorder exists, push an image update message onto the UI queue.
                # also, push latest images to update_images for visualization
                if self.recorder is not None:
                    with self.shared_state.image_lock:
                        top_img = self.shared_state.latest_top_image
                        bottom_img = self.shared_state.latest_bottom_image
                    if self.visualize and top_img is not None and bottom_img is not None:
                        top_img_np = top_img.copy()
                        bottom_img_np = bottom_img.copy()
                        # bottom_img_np = np.transpose(bottom_img_np, (1, 0, 2))

                        update_images(
                            [bottom_img_np, top_img_np],
                            window_name="Live Robot Feed"
                        )
                        cv2.waitKey(1)
                    if bottom_img is not None:
                        ui_msg = {"type": "image_update", "image": bottom_img.copy()}
                        self.ui_queue.put(ui_msg)
                        # TODO make this safe against race conditions
                        # push latest robot state to autograsper for use in grasper logic, this is legacy from before shared_state
                        self.autograsper.robot_state = self.shared_state.latest_robot_state
                self.shutdown_event.wait(timeout=0.1)
            except Exception as e:
                logger.exception("Error in state monitoring: %s", e)
                self.shutdown_event.set()

    def _check_if_record_is_requested(self):
        if (
            self.autograsper.request_state_record
            and self.recorder is not None
        ):
            with self.recorder.snapshot_cond:
                self.recorder.take_snapshot += 1
                while self.recorder.take_snapshot > 0:
                    self.recorder.snapshot_cond.wait(timeout=1.0)
            self.autograsper.request_state_record = False
            self.autograsper.state_recorded_event.set()

    def _process_messages(self):
        """
        Processes non-UI messages from the message queue.
        """
        prev_state = RobotActivity.STARTUP
        self.on_startup()
        self.session_dir, self.task_dir, self.restore_dir = "", "", ""
        while not self.shutdown_event.is_set():
            try:
                msg = self.msg_queue.get(timeout=0.2)
            except Empty:
                continue

            if msg["type"] == "state_update":
                current_state = msg["state"]
                if current_state != prev_state:
                    self._on_state_transition(prev_state, current_state)
                    if current_state == RobotActivity.ACTIVE:
                        logger.info(f"State transition to ACTIVE. save_data={self.save_data}")
                        if self.save_data:
                            logger.info(f"save_data is True, calling _create_new_data_point()")
                            self._create_new_data_point()
                            logger.info(f"After _create_new_data_point: task_dir={self.task_dir}")
                        else:
                            logger.warning("save_data is False, skipping _create_new_data_point()")
                        self._on_active_state()
                    elif current_state == RobotActivity.RESETTING:
                        self._on_resetting_state()
                    elif current_state == RobotActivity.FINISHED:
                        self._on_finished_state()
                        break
                    prev_state = current_state
            self.msg_queue.task_done()

    def _on_state_transition(self, old_state, new_state):
        if new_state == RobotActivity.STARTUP and old_state != RobotActivity.STARTUP:
            if self.recorder:
                self.recorder.pause = True
                time.sleep(self.timeout_between_experiments)
                self.recorder.pause = False

    def _create_new_data_point(self):
        base_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "recorded_data",
            self.experiment_name,
        )
        self.session_dir, self.task_dir, self.restore_dir = (
            FileManager.get_session_dirs(base_dir)
        )
        logger.info(f"Created data point: session_dir={self.session_dir}, task_dir={self.task_dir}, restore_dir={self.restore_dir}")

    def on_startup(self):
        self._ensure_recorder_running("")
        self.recorder._update()  # Initial update to get images/state

    def _on_active_state(self):
        logger.info(f"Entering ACTIVE state. task_dir={self.task_dir}, save_data={self.save_data}")
        if self.save_data:
            if not self.task_dir:
                logger.error(f"CRITICAL: task_dir is empty! save_data={self.save_data}. This likely means _create_new_data_point() was not called or failed.")
                return
            self.autograsper.output_dir = self.task_dir
            logger.info(f"Starting new recording in directory: {self.task_dir}")
            self._ensure_recorder_running(self.task_dir)
            if self.recorder:
                try:
                    logger.info(f"Calling start_new_recording with task_dir: {self.task_dir}")
                    self.recorder.start_new_recording(self.task_dir)
                    logger.info(f"Successfully called start_new_recording")
                except Exception as e:
                    logger.error(f"Exception in start_new_recording: {e}", exc_info=True)
            else:
                logger.error("Recorder failed to initialize!")
        else:
            logger.warning("save_data is False, not initializing recording")
        # Allow some time for initialization.
        time.sleep(0.5)
        self.autograsper.start_event.set()

    def _ensure_recorder_running(self, output_dir: str):
        if not self.recorder:
            logger.info(f"Creating recorder with output_dir: {output_dir}")
            self.recorder = self._setup_recorder(output_dir)
            # Start the recorder in its own thread.
            logger.info("Submitting recorder.record() to executor")
            self.executor.submit(self.recorder.record)
        else:
            logger.info(f"Recorder already exists, not recreating")

    def _setup_recorder(self, output_dir: str):
        return Recorder(
            self.config, output_dir=output_dir, shutdown_event=self.shutdown_event, shared_state=self.shared_state
        )

    def _on_resetting_state(self):
        status = "fail" if self.autograsper.failed else "success"
        logger.info(f"Task result: {status}")
        if self.save_data:
            # Ensure data directories exist, even if ACTIVE state was skipped
            if not self.task_dir:
                logger.info("task_dir not initialized, calling _create_new_data_point() in RESETTING state")
                self._create_new_data_point()
            
            status_file = os.path.join(self.session_dir, "status.txt")
            with open(status_file, "w") as f:
                f.write(status)
            self.autograsper.output_dir = self.restore_dir
            
            # Ensure recorder exists and start new recording in restore_dir
            if not self.recorder:
                logger.info("Recorder not initialized, creating it for RESETTING state")
                self._ensure_recorder_running(self.restore_dir)
            
            if self.recorder and self.restore_dir:
                logger.info(f"Starting new recording in restore_dir: {self.restore_dir}")
                try:
                    self.recorder.start_new_recording(self.restore_dir)
                except Exception as e:
                    logger.error(f"Exception in start_new_recording during RESETTING: {e}", exc_info=True)
            else:
                logger.warning(f"Cannot start recording: restore_dir={'empty' if not self.restore_dir else 'ok'}, recorder={'exists' if self.recorder else 'missing'}")

    def _on_finished_state(self):
        if self.recorder:
            self.recorder.stop()
            time.sleep(1)  # Allow recorder to finish up

    # --- Public API for running coordinator tasks ---
    def start(self):
        """
        Starts the coordinator's background tasks.
        """
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="Coord"
        )
        self.futures = [
            self.executor.submit(self.autograsper.run_grasping),
            self.executor.submit(self._monitor_state),
            self.executor.submit(self._process_messages),
        ]

    def join(self):
        """
        Blocks until the coordinator's background tasks have finished.
        """
        try:
            concurrent.futures.wait(
                self.futures, return_when=concurrent.futures.FIRST_EXCEPTION
            )
        except Exception as e:
            logger.error("Exception in coordinator tasks: %s", e)
            self.shutdown_event.set()
        finally:
            self.shutdown_event.set()
            self.executor.shutdown(wait=True)

    def get_ui_update(self, timeout: float = 0.1):
        """
        Retrieves an image update from the UI queue.
        Returns the message dict if available, or None.
        """
        try:
            return self.ui_queue.get(timeout=timeout)
        except Empty:
            return None
