# coordinator.py
import os
import logging
import time
import cv2
from dataclasses import dataclass
from typing import Optional
from queue import Queue, Empty
import concurrent.futures
import threading

from grasper import RobotActivity, AutograsperBase
from recording import Recorder

logger = logging.getLogger(__name__)


@dataclass
class SharedState:
    """
    Holds shared references between threads.
    """

    state: str = RobotActivity.STARTUP
    recorder: Optional[Recorder] = None


class DataCollectionCoordinator:
    """
    Orchestrates the autograsper, manages recording, and coordinates state changes
    and image updates via a centralized message queue.
    """

    def __init__(
        self, config, grasper: AutograsperBase, shutdown_event: threading.Event
    ):
        self.config = config
        self.shutdown_event = shutdown_event
        self.shared_state = SharedState()
        self.autograsper = grasper
        self.message_queue = Queue()

        # Read configuration with explicit error handling
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
            # Post state update message.
            state_msg = {"type": "state_update", "state": self.autograsper.state}
            self.message_queue.put(state_msg)

            self._check_if_record_is_requested()

            # Post image update message if recorder is available.
            if self.shared_state.recorder is not None:
                bottom_img = self.shared_state.recorder.bottom_image
                if bottom_img is not None:
                    img_msg = {"type": "image_update", "image": bottom_img.copy()}
                    self.message_queue.put(img_msg)
            self.shutdown_event.wait(timeout=0.1)

    def _check_if_record_is_requested(self):
        if (
            self.autograsper.request_state_record
            and self.shared_state.recorder is not None
        ):
            with self.shared_state.recorder.snapshot_cond:
                self.shared_state.recorder.take_snapshot += 1
                while self.shared_state.recorder.take_snapshot > 0:
                    self.shared_state.recorder.snapshot_cond.wait(timeout=0.1)
            self.autograsper.request_state_record = False
            self.autograsper.state_recorded_event.set()

    def _process_messages(self):
        """
        Processes messages from the message queue.
        """
        prev_state = RobotActivity.STARTUP
        self.session_dir, self.task_dir, self.restore_dir = "", "", ""
        while not self.shutdown_event.is_set():
            try:
                msg = self.message_queue.get(timeout=0.2)
            except Empty:
                continue

            if msg["type"] == "state_update":
                current_state = msg["state"]
                if current_state != prev_state:
                    self._on_state_transition(prev_state, current_state)
                    if current_state == RobotActivity.ACTIVE:
                        if self.save_data:
                            self._create_new_data_point()
                        self._on_active_state()
                    elif current_state == RobotActivity.RESETTING:
                        self._on_resetting_state()
                    elif current_state == RobotActivity.FINISHED:
                        self._on_finished_state()
                        break
                    prev_state = current_state
            elif msg["type"] == "image_update":
                bottom_img = msg["image"]
                try:
                    cv2.imshow("Bottom Image", bottom_img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.shutdown_event.set()
                except Exception as e:
                    logger.error("Error during image display: %s", e)

            self.message_queue.task_done()

    def _on_state_transition(self, old_state, new_state):
        if new_state == RobotActivity.STARTUP and old_state != RobotActivity.STARTUP:
            if self.shared_state.recorder:
                self.shared_state.recorder.pause = True
                time.sleep(self.timeout_between_experiments)
                self.shared_state.recorder.pause = False

    def _create_new_data_point(self):
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
        if self.save_data:
            self.autograsper.output_dir = self.task_dir
        self._ensure_recorder_running(self.task_dir)
        # Allow some time for initialization.
        time.sleep(0.5)
        self.autograsper.start_event.set()

    def _ensure_recorder_running(self, output_dir: str):
        if not self.shared_state.recorder:
            self.shared_state.recorder = self._setup_recorder(output_dir)
            # Start the recorder in its own thread.
            self.recorder_future = self.executor.submit(
                self.shared_state.recorder.record
            )
        if self.save_data:
            self.shared_state.recorder.start_new_recording(output_dir)

    def _setup_recorder(self, output_dir: str):
        return Recorder(
            self.config, output_dir=output_dir, shutdown_event=self.shutdown_event
        )

    def _on_resetting_state(self):
        status = "fail" if self.autograsper.failed else "success"
        logger.info(f"Task result: {status}")
        if self.save_data:
            with open(os.path.join(self.session_dir, "status.txt"), "w") as f:
                f.write(status)
            self.autograsper.output_dir = self.restore_dir
            if self.shared_state.recorder:
                self.shared_state.recorder.start_new_recording(self.restore_dir)

    def _on_finished_state(self):
        if self.shared_state.recorder:
            self.shared_state.recorder.stop()
            # Wait briefly for the recorder thread to finish.
            time.sleep(1)

    def run(self):
        """
        Main entry point: Uses a ThreadPoolExecutor to run the autograsper, state monitor,
        and message processor concurrently.
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="Coord"
        ) as executor:
            self.executor = executor  # Save reference for later submissions.
            # Submit the three primary tasks.
            futures = []
            futures.append(executor.submit(self.autograsper.run_grasping))
            futures.append(executor.submit(self._monitor_state))
            futures.append(executor.submit(self._process_messages))

            try:
                concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_EXCEPTION
                )
            except Exception as e:
                logger.error("Exception in one of the coordinator tasks: %s", e)
                self.shutdown_event.set()
            finally:
                self.shutdown_event.set()
                for future in futures:
                    try:
                        future.result(timeout=5)
                    except Exception as e:
                        logger.error("Error waiting for task to finish: %s", e)
                cv2.destroyAllWindows()
                logger.info("Coordinator shutdown complete.")
