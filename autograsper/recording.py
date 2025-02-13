# recording.py
import os
import sys
import time
import json
import logging
from typing import Any, Tuple, List, Dict
import cv2
import numpy as np
import threading

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from client.cloudgripper_client import GripperRobot
from library.utils import convert_ndarray_to_list, get_undistorted_bottom_image

logger = logging.getLogger(__name__)


class Recorder:
    FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

    def __init__(self, config: Any, output_dir: str, shutdown_event: threading.Event):
        self.shutdown_event = shutdown_event
        try:
            camera_config = config["camera"]
            experiment_config = config["experiment"]
            self.camera_matrix = np.array(camera_config["m"])
            self.distortion_coeffs = np.array(camera_config["d"])
            self.record_only_after_action = bool(
                camera_config["record_only_after_action"]
            )
            self.robot_idx = experiment_config["robot_idx"]
            self.save_data = bool(camera_config["record"])
            self.FPS = int(camera_config["fps"])
            self.save_images_individually = bool(
                camera_config["save_images_individually"]
            )
            self.clip_length = camera_config.get("clip_length", None)
        except KeyError as e:
            raise ValueError(f"Missing configuration key in Recorder: {e}") from e
        except TypeError as e:
            raise ValueError(f"Invalid configuration format in Recorder: {e}") from e

        self.token = os.getenv("CLOUDGRIPPER_TOKEN")
        if not self.token:
            raise ValueError("CLOUDGRIPPER_TOKEN environment variable not set")

        self.output_dir = output_dir
        self.robot = GripperRobot(self.robot_idx, self.token)

        self.image_top = None
        self.bottom_image = None
        self.pause = False

        # For when record_only_after_action is True.
        self.take_snapshot = 0

        # Use reentrant locks for nested locking.
        self.image_lock = threading.RLock()
        self.writer_lock = threading.RLock()
        # Condition variable to synchronize snapshot requests.
        self.snapshot_cond = threading.Condition(threading.RLock())

        # State and writer variables.
        self.stop_flag = False
        self.frame_counter = 0
        self.video_counter = 0
        self.video_writer_top = None
        self.video_writer_bottom = None

        self._initialize_directories()

    def _initialize_directories(self) -> None:
        if self.save_images_individually:
            self.output_images_dir = os.path.join(self.output_dir, "Images")
            self.output_bottom_images_dir = os.path.join(
                self.output_dir, "Bottom_Images"
            )
            os.makedirs(self.output_images_dir, exist_ok=True)
            os.makedirs(self.output_bottom_images_dir, exist_ok=True)
        else:
            self.output_video_dir = os.path.join(self.output_dir, "Video")
            self.output_bottom_video_dir = os.path.join(self.output_dir, "Bottom_Video")
            os.makedirs(self.output_video_dir, exist_ok=True)
            os.makedirs(self.output_bottom_video_dir, exist_ok=True)

    def _start_new_video(self) -> Tuple[cv2.VideoWriter, cv2.VideoWriter]:
        if not self.ensure_images():
            return None, None

        video_filename_top = os.path.join(
            self.output_video_dir, f"video_{self.video_counter}.mp4"
        )
        video_filename_bottom = os.path.join(
            self.output_bottom_video_dir, f"video_{self.video_counter}.mp4"
        )

        if self.save_data:
            with self.image_lock:
                top_shape = self.image_top.shape[1::-1]
                bottom_shape = self.bottom_image.shape[1::-1]
            video_writer_top = cv2.VideoWriter(
                video_filename_top, self.FOURCC, self.FPS, top_shape
            )
            video_writer_bottom = cv2.VideoWriter(
                video_filename_bottom, self.FOURCC, self.FPS, bottom_shape
            )
            return video_writer_top, video_writer_bottom
        else:
            return None, None

    def record(self) -> None:
        """Record video or images. Note that image display is now handled by the coordinator."""
        self._prepare_new_recording()
        try:
            while not self.stop_flag and not self.shutdown_event.is_set():
                if not self.pause:
                    self._update()
                    if not self.ensure_images():
                        continue

                    if (not self.record_only_after_action) or (self.take_snapshot > 0):
                        if self.save_data:
                            self._capture_frame()
                        if (
                            self.clip_length
                            and (self.frame_counter % self.clip_length == 0)
                            and (self.frame_counter != 0)
                            and not self.save_images_individually
                        ):
                            self.video_counter += 1
                            self._start_or_restart_video_writers()
                        time.sleep(1 / self.FPS)
                        if self.save_data:
                            self.save_state()
                        self.frame_counter += 1
                    else:
                        time.sleep(1 / self.FPS)
                else:
                    time.sleep(1 / self.FPS)
        except Exception as e:
            logger.error("An error occurred in Recorder.record: %s", e)
            self.shutdown_event.set()
        finally:
            self._release_writers()

    def _update(self) -> None:
        data = self.robot.get_all_states()
        with self.image_lock:
            self.image_top = data[0]
            self.bottom_image = get_undistorted_bottom_image(
                self.robot, self.camera_matrix, self.distortion_coeffs
            )
        self.state = data[2]
        self.timestamp = data[3]

    def _capture_frame(self) -> None:
        try:
            if not self.ensure_images():
                return
            with self.image_lock:
                top_image = self.image_top.copy()
                bottom_image = self.bottom_image.copy()
            if self.save_images_individually:
                top_filename = os.path.join(
                    self.output_images_dir, f"image_top_{self.frame_counter}.jpeg"
                )
                bottom_filename = os.path.join(
                    self.output_bottom_images_dir,
                    f"image_bottom_{self.frame_counter}.jpeg",
                )
                cv2.imwrite(top_filename, top_image)
                cv2.imwrite(bottom_filename, bottom_image)
            else:
                with self.writer_lock:
                    if (
                        self.video_writer_top is not None
                        and self.video_writer_bottom is not None
                    ):
                        self.video_writer_top.write(top_image)
                        self.video_writer_bottom.write(bottom_image)
                    else:
                        logger.warning("Video writers not initialized.")
        except Exception as e:
            logger.error("Error capturing frame: %s", e)
        finally:
            with self.snapshot_cond:
                if self.take_snapshot > 0:
                    self.take_snapshot -= 1
                    if self.take_snapshot == 0:
                        self.snapshot_cond.notify_all()

    def _start_or_restart_video_writers(self) -> None:
        if not self.save_images_individually:
            with self.writer_lock:
                self._release_writers()
                self.video_writer_top, self.video_writer_bottom = (
                    self._start_new_video()
                )

    def _release_writers(self) -> None:
        with self.writer_lock:
            if self.video_writer_top:
                self.video_writer_top.release()
                self.video_writer_top = None
            if self.video_writer_bottom:
                self.video_writer_bottom.release()
                self.video_writer_bottom = None

    def start_new_recording(self, new_output_dir: str) -> None:
        self.output_dir = new_output_dir
        self._initialize_directories()
        self._prepare_new_recording()
        logger.info("Started new recording in directory: %s", new_output_dir)

    def _prepare_new_recording(self) -> None:
        self.stop_flag = False
        if not self.save_images_individually:
            self._start_or_restart_video_writers()

    def stop(self) -> None:
        self.stop_flag = True
        logger.info("Stop flag set to True in Recorder")

    def save_state(self) -> None:
        try:
            state = self.state.copy() if isinstance(self.state, dict) else self.state
            timestamp = self.timestamp
            state = convert_ndarray_to_list(state)
            if not isinstance(state, dict):
                state = {"state": state}
            state["time"] = timestamp

            state_file = os.path.join(self.output_dir, "states.json")
            data: List[Dict[str, Any]] = []

            if os.path.exists(state_file):
                with open(state_file, "r") as file:
                    data = json.load(file)

            data.append(state)

            with open(state_file, "w") as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            logger.error("Error saving state: %s", e)

    def ensure_images(self) -> bool:
        with self.image_lock:
            if self.image_top is None or self.bottom_image is None:
                self._update()
            if self.image_top is None or self.bottom_image is None:
                logger.error(
                    "ensure_images: Failed to obtain valid images from the robot after update."
                )
                return False
        return True
