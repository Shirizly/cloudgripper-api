import os
import sys
import time
import json
import logging
from typing import Any, Tuple, List, Dict
import cv2
import ast
import numpy as np
import copy

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

#CG1Specific
from client.cloudgripper_client import (
    GripperRobot,
)  # Assuming this is the correct import
from library.utils import (
    convert_ndarray_to_list,
    get_undistorted_bottom_image,
)  # Assuming these are the correct imports

logging.basicConfig(level=logging.INFO)


class Recorder:
    FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

    def __init__(self, config: Any, output_dir: str):
        try:
            experiment_cfg = config["experiment"]
            self.experiment_name = ast.literal_eval(experiment_cfg["name"])
            self.robot_idx = ast.literal_eval(experiment_cfg["robot_idx"])

            camera_cfg = config["camera"]
            self.camera_matrix = np.array(ast.literal_eval(camera_cfg["m"]))
            self.distortion_coeffs = np.array(ast.literal_eval(camera_cfg["d"]))

            self.save_data = bool(ast.literal_eval(camera_cfg["record"]))
            self.FPS = int(ast.literal_eval(camera_cfg["fps"]))

            self.record_only_after_action = bool(
                ast.literal_eval(camera_cfg["record_only_after_action"])
            )

            self.save_images_individually = bool(
                ast.literal_eval(camera_cfg["save_images_individually"])
            )

            self.clip_length = None
            if "clip_length" in camera_cfg:
                self.clip_length = ast.literal_eval(camera_cfg["clip_length"])
        except Exception as e:
            raise ValueError("Recorder config.ini ERROR") from e

        self.token = os.getenv("CLOUDGRIPPER_TOKEN")
        if not self.token:
            raise ValueError("CLOUDGRIPPER_TOKEN environment variable not set")

        self.output_dir = output_dir
        self.robot = GripperRobot(self.robot_idx, self.token)
        self.image_top = None
        self.bottom_image = None
        self.pause = False

        self.state = None

        # For when record_only_after_action is True
        self.take_snapshot = 0

        
        self._update()

        # Initialize state variables
        self.stop_flag = False
        self.frame_counter = 0
        self.video_counter = 0
        self.video_writer_top = None
        self.video_writer_bottom = None
        self._initialize_directories()

    def _initialize_directories(self) -> None:
        """Initialize output directories."""
        if self.save_images_individually:
            # Create directories for individual image files
            self.output_images_dir = os.path.join(self.output_dir, "Images")
            self.output_bottom_images_dir = os.path.join(
                self.output_dir, "Bottom_Images"
            )
            os.makedirs(self.output_images_dir, exist_ok=True)
            os.makedirs(self.output_bottom_images_dir, exist_ok=True)
        else:
            # Create directories for video files
            self.output_video_dir = os.path.join(self.output_dir, "Video")
            self.output_bottom_video_dir = os.path.join(self.output_dir, "Bottom_Video")
            os.makedirs(self.output_video_dir, exist_ok=True)
            os.makedirs(self.output_bottom_video_dir, exist_ok=True)

    def _start_new_video(self) -> Tuple[cv2.VideoWriter, cv2.VideoWriter]:
        """Start new video writers for top and bottom cameras."""
        if not self.ensure_images():
            return

        video_filename_top = os.path.join(
            self.output_video_dir, f"video_{self.video_counter}.mp4"
        )
        video_filename_bottom = os.path.join(
            self.output_bottom_video_dir, f"video_{self.video_counter}.mp4"
        )

        if self.save_data:
            video_writer_top = cv2.VideoWriter(
                video_filename_top, self.FOURCC, self.FPS, self.image_top.shape[1::-1]
            )
            video_writer_bottom = cv2.VideoWriter(
                video_filename_bottom,
                self.FOURCC,
                self.FPS,
                self.bottom_image.shape[1::-1],
            )

            return video_writer_top, video_writer_bottom

        else:
            return None, None

    def record(self) -> None:
        """Record video or images with optional periodic video restarts."""
        self._prepare_new_recording()
        try:
            while not self.stop_flag:
                if not self.pause:
                    self._update()

                    if not self.ensure_images():
                        return

                    if self.record_only_after_action is False or self.take_snapshot > 0:
                        if self.save_data:
                            self._capture_frame()

                        # Only restart video writers for video mode, not when saving images individually.
                        if (
                            self.clip_length
                            and self.frame_counter % self.clip_length == 0
                            and self.frame_counter != 0
                            and not self.save_images_individually
                        ):
                            self.video_counter += 1
                            self._start_or_restart_video_writers()

                        time.sleep(1 / self.FPS)
                        if self.save_data:
                            self.save_state()
                        self.frame_counter += 1

                        if self.take_snapshot > 0:
                            self.take_snapshot -= 1
                    else:
                        time.sleep(1 / self.FPS)

                    image_to_show = copy.copy(self.bottom_image)
                    cv2.imshow(f"ImageBottom_{self.robot_idx}", image_to_show)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.stop_flag = True
        except Exception as e:
            logging.error("An error occurred: %s", e)
        finally:
            self._release_writers()
            cv2.destroyAllWindows()

    def _update(self) -> None:
        """Update images and state from the robot."""
        data = self.robot.get_all_states()
        self.image_top = data[0]
        self.bottom_image = get_undistorted_bottom_image(
            self.robot, self.camera_matrix, self.distortion_coeffs
        )
        self.state = data[2]
        self.timestamp = data[3]

    def _capture_frame(self) -> None:
        """
        Capture frames from the robot's cameras and either write directly to a video file
        or save each frame as an individual JPEG image.
        """
        try:
            if not self.ensure_images():
                return
            if self.save_images_individually:
                # Save each image as a separate JPEG file
                top_filename = os.path.join(
                    self.output_images_dir, f"image_top_{self.frame_counter}.jpeg"
                )
                bottom_filename = os.path.join(
                    self.output_bottom_images_dir,
                    f"image_bottom_{self.frame_counter}.jpeg",
                )
                cv2.imwrite(top_filename, self.image_top)
                cv2.imwrite(bottom_filename, self.bottom_image)
            else:
                if self.video_writer_top and self.video_writer_bottom:
                    self.video_writer_top.write(self.image_top)
                    self.video_writer_bottom.write(self.bottom_image)
                else:
                    logging.warning("Video writers not initialized.")
        except Exception as e:
            logging.error("Error capturing frame: %s", e)

    def _start_or_restart_video_writers(self) -> None:
        """Start or restart video writers (only used when saving video)."""
        if not self.save_images_individually:
            self._release_writers()  # Ensure the old writers are released
            self.video_writer_top, self.video_writer_bottom = self._start_new_video()

    def _release_writers(self) -> None:
        """Release the video writers."""
        if self.video_writer_top:
            self.video_writer_top.release()
            self.video_writer_top = None
        if self.video_writer_bottom:
            self.video_writer_bottom.release()
            self.video_writer_bottom = None

    def start_new_recording(self, new_output_dir: str) -> None:
        """Start a new recording session with a new output directory."""
        self.output_dir = new_output_dir
        self._initialize_directories()
        self._prepare_new_recording()
        logging.info("Started new recording in directory: %s", new_output_dir)

    def _prepare_new_recording(self) -> None:
        """Prepare for a new recording session."""
        self.stop_flag = False
        if not self.save_images_individually:
            self._start_or_restart_video_writers()

    def stop(self) -> None:
        """Set the stop flag to terminate recording."""
        self.stop_flag = True
        logging.info("Stop flag set to True")

    def save_state(self) -> None:
        """Save the state of the robot to a JSON file."""
        try:
            state = self.state.copy()
            timestamp = self.timestamp
            state = convert_ndarray_to_list(state)
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
            logging.error("Error saving state: %s", e)

    def ensure_images(self) -> bool:
        """
        Ensure that both top and bottom images are valid.

        Checks if self.image_top or self.bottom_image are None.
        If either is None, calls self._update() to refresh the images.
        If after update either image is still None, logs an error and returns False.
        Otherwise, returns True.
        """
        if self.image_top is None or self.bottom_image is None:
            self._update()

        if self.image_top is None or self.bottom_image is None:
            logging.error("ensure_images: Failed to obtain valid images from the robot after update.")
            return False

        return True
