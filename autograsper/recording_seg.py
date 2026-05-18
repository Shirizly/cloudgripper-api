import os
import sys
import json
import logging
import time
from typing import Any, Tuple, List, Dict, Optional
import cv2
import numpy as np
import threading

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from client.cloudgripper_client import GripperRobot
from library.utils import convert_ndarray_to_list, get_undistorted_bottom_image
from file_manager import FileManager

logger = logging.getLogger(__name__)


class SegmentationThread(threading.Thread):
    """
    Dedicated thread for running chickpea segmentation on bottom images.
    
    Runs continuously, processing bottom images from shared state and 
    storing segmentation masks back to shared state.
    """
    
    def __init__(self, segmenter, shared_state: Any, shutdown_event: threading.Event, crop_size: int = 360):
        super().__init__(name="SegmentationThread", daemon=True)
        self.segmenter = segmenter
        self.shared_state = shared_state
        self.shutdown_event = shutdown_event
        self.crop_size = crop_size
        self.last_processed_timestamp = None
        logger.info(f"SegmentationThread initialized with crop_size={crop_size}")
    
    def _crop_center(self, image: np.ndarray, crop_size: int) -> np.ndarray:
        """Crop image to center region of specified size."""
        h, w = image.shape[:2]
        start_y = (h - crop_size) // 2
        start_x = (w - crop_size) // 2
        return image[start_y:start_y + crop_size, start_x:start_x + crop_size]
    
    def run(self):
        """Main segmentation loop."""
        logger.info("SegmentationThread started")
        try:
            while not self.shutdown_event.is_set():
                # Get latest bottom image from shared state
                with self.shared_state.image_lock:
                    bottom_image = self.shared_state.latest_bottom_image
                    current_timestamp = self.shared_state.timestamp
                
                # Only process if we have a new image
                if bottom_image is not None and current_timestamp != self.last_processed_timestamp:
                    try:
                        # Crop to center 360x360
                        cropped = self._crop_center(bottom_image, self.crop_size)
                        
                        # Run segmentation
                        mask = self.segmenter.predict(cropped, return_format='combined')
                        
                        # Store result in shared state
                        with self.shared_state.image_lock:
                            self.shared_state.latest_mask = mask
                            self.shared_state.latest_mask_saved = False
                        
                        self.last_processed_timestamp = current_timestamp
                    except Exception as e:
                        logger.exception(f"Error in segmentation: {e}")
                else:
                    # No new image, sleep briefly
                    self.shutdown_event.wait(0.01)
        except Exception as e:
            logger.exception(f"Fatal error in SegmentationThread: {e}")
            self.shutdown_event.set()
        finally:
            logger.info("SegmentationThread stopped")


class Recorder:
    FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

    def __init__(self, config: Any, output_dir: str, shutdown_event: threading.Event, shared_state: Any, segmenter=None):
        self.shutdown_event = shutdown_event
        self.shared_state = shared_state
        self.segmenter = segmenter
        self.segmentation_thread: Optional[SegmentationThread] = None

        try:
            camera_config = config["camera"]
            experiment_config = config["experiment"]
            robot_config = config["robot"]
            self.angle_bias = robot_config.get(experiment_config["robot_idx"], {}).get("rotation_bias", 0)
            self.camera_matrix = np.array(camera_config["m"])
            self.distortion_coeffs = np.array(camera_config["d"])
            print("Camera Matrix:", self.camera_matrix)
            print("Distortion Coefficients:", self.distortion_coeffs)
            
            self.H_matrix = np.array(camera_config["H"])
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

        self.image_top: Optional[np.ndarray] = None
        self.bottom_image: Optional[np.ndarray] = None
        self.bottom_image_raw: Optional[np.ndarray] = None
        self.save_bottom_raw = bool(camera_config.get("save_bottom_raw", False))
        self.pause = False

        # For when record_only_after_action is True.
        self.take_snapshot = 0

        # Reentrant locks for nested locking.
        self.image_lock = threading.RLock()
        self.writer_lock = threading.RLock()
        # Condition variable to synchronize snapshot requests.
        self.snapshot_cond = threading.Condition(threading.RLock())

        # State and writer variables.
        self.stop_flag = False
        self.frame_counter = 0
        self.video_counter = 0
        self.video_writer_top: Optional[cv2.VideoWriter] = None
        self.video_writer_bottom: Optional[cv2.VideoWriter] = None
        self.disk_enabled = False # no saving images/videos on startup, not advancing counters
        
        # Start segmentation thread if segmenter provided
        if self.segmenter is not None:
            self.segmentation_thread = SegmentationThread(
                self.segmenter, 
                self.shared_state, 
                self.shutdown_event,
                crop_size=360
            )
            self.segmentation_thread.start()
            logger.info("Segmentation thread started")

    def _initialize_directories(self) -> None:
        logger.info(f"_initialize_directories called. save_images_individually={self.save_images_individually}")
        if self.save_images_individually:
            self.output_images_dir, self.output_bottom_images_dir, self.output_mask_dir = (
                FileManager.create_image_dirs(self.output_dir)
            )
            logger.info(f"Created image directories: {self.output_images_dir}, {self.output_bottom_images_dir}, {self.output_mask_dir}")
        else:
            self.output_video_dir, self.output_bottom_video_dir = (
                FileManager.create_video_dirs(self.output_dir)
            )
            logger.info(f"Created video directories: {self.output_video_dir}, {self.output_bottom_video_dir}")

    def _start_new_video(
        self,
    ) -> Tuple[Optional[cv2.VideoWriter], Optional[cv2.VideoWriter]]:
        if not self.ensure_images():
            return None, None

        video_filename_top = os.path.join(
            self.output_video_dir, f"video_{self.video_counter}.mp4"
        )
        video_filename_bottom = os.path.join(
            self.output_bottom_video_dir, f"video_{self.video_counter}.mp4"
        )

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

    def record(self) -> None:
        """Record video or images. Image display is handled externally."""
        self._prepare_new_recording()
        try:
            frame_start: float = time.perf_counter()
            while not self.stop_flag and not self.shutdown_event.is_set():
                inbetween_time = time.perf_counter() - frame_start
                frame_start = time.perf_counter()
                if not self.pause:
                    self._update()
                    api_time = time.perf_counter() - frame_start
                    while not self.ensure_images():
                        # Wait briefly for images to become available.
                        self.shutdown_event.wait(0.1 / self.FPS)
                        if time.perf_counter() - frame_start > 1/self.FPS:  # If images aren't available after 1/FPS seconds, log a warning and skip this frame.
                            logger.warning("Images not available after 1/FPS-rate seconds, skipping frame.")
                            break
                    api_time2 = time.perf_counter() - frame_start # total time including ensure_images
                    if (not self.record_only_after_action) or (self.take_snapshot > 0):
                        if self.save_data and self.disk_enabled:
                            self._capture_frame()
                        capture_time = time.perf_counter() - frame_start - api_time2 # time spent on capture after ensuring images
                        if (
                            self.clip_length
                            and (self.frame_counter % self.clip_length == 0)
                            and (self.frame_counter != 0)
                            and not self.save_images_individually
                        ):
                            self.video_counter += 1
                            self._start_or_restart_video_writers()
                        # Use shutdown_event.wait to allow prompt shutdown.
                        self.shutdown_event.wait(max(0, 1 / self.FPS - (time.perf_counter() - frame_start)-0.0002)) # subtract small buffer time to improve chances of hitting target FPS
                        if self.save_data and self.disk_enabled:
                            self.save_state()
                        self.frame_counter += 1
                    else:
                        self.shutdown_event.wait(max(0, 1 / self.FPS - (time.perf_counter() - frame_start)-0.0002))
                else:
                    self.shutdown_event.wait(max(0, 1 / self.FPS - (time.perf_counter() - frame_start)-0.0002))
                final_time = time.perf_counter() - frame_start
                # print(f"Recorder frame time: total={final_time:.4f}s, api={api_time:.4f}s, capture={capture_time if 'capture_time' in locals() else 0:.4f}s, inbetween={inbetween_time:.4f}s")
        except Exception as e:
            logger.exception("An error occurred in Recorder.record:", e)
            self.shutdown_event.set()
        finally:
            self._release_writers()

    def _update(self) -> None:
        """Update image and state data from the robot."""
        try:
            data = self.robot.get_all_states()
            with self.image_lock:
                self.image_top = data[0]
                if self.save_bottom_raw:
                    self.bottom_image_raw = data[1]
                self.bottom_image = get_undistorted_bottom_image(
                    data[1], self.camera_matrix, self.distortion_coeffs, self.H_matrix
                )
            self.state = data[2]
            if self.state is not None and isinstance(self.state, dict) and "rotation" in self.state:
                self.state['rotation'] -= self.angle_bias # apply rotation bias to recorded state for better alignment with actual gripper pose
                self.state['rotation'] = self.state['rotation'] % 180 # ensure rotation stays within [0, 180) range after bias correction
            self.timestamp = data[3]
            with self.shared_state.image_lock:
                self.shared_state.latest_top_image = self.image_top
                self.shared_state.latest_bottom_image = self.bottom_image
                self.shared_state.latest_robot_state = self.state
                self.shared_state.timestamp = data[3]
            # print(f"latest robot state is: {data[2]}")
        except Exception as e:
            logger.exception("Error updating images/state:", e)
            raise

    def _capture_frame(self) -> None:
        """Capture and save the current frame as an image or add it to the video writer."""
        try:
            if not self.ensure_images():
                return
            
            # Update shared state with current frame index
            # This synchronizes the grasper with the recorder for precise action boundaries
            with self.shared_state.frame_index_lock:
                self.shared_state.frame_index = self.frame_counter
            
            bottom_image_raw = None
            mask = None
            with self.image_lock:
                top_image = self.image_top.copy()
                bottom_image = self.bottom_image.copy()
                if self.bottom_image_raw is not None:
                    bottom_image_raw = self.bottom_image_raw.copy()
                if self.shared_state.latest_mask is not None and not self.shared_state.latest_mask_saved:
                    mask = self.shared_state.latest_mask
                    self.shared_state.latest_mask_saved = True
            if self.save_images_individually:
                self._save_individual_images(top_image, bottom_image, bottom_image_raw,mask)
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
            logger.exception("Error capturing frame:", e)
        finally:
            with self.snapshot_cond:
                if self.take_snapshot > 0:
                    self.take_snapshot -= 1
                    if self.take_snapshot == 0:
                        self.snapshot_cond.notify_all()

    def _save_individual_images(
        self, top_image: np.ndarray, bottom_image: np.ndarray, bottom_image_raw: np.ndarray = None, mask: np.ndarray = None) -> None:
        """Save the top and bottom images as individual JPEG files and mask as binary .npy file."""
        try:
            top_filename = os.path.join(
                self.output_images_dir, f"image_top_{self.frame_counter}.jpeg"
            )
            bottom_filename = os.path.join(
                self.output_bottom_images_dir, f"image_bottom_{self.frame_counter}.jpeg"
            )
            if bottom_image_raw is not None:
                bottom_raw_filename = os.path.join(
                    self.output_bottom_images_dir, f"image_bottom_raw_{self.frame_counter}.jpeg"
                )
                cv2.imwrite(bottom_raw_filename, bottom_image_raw)
            cv2.imwrite(top_filename, top_image)
            cv2.imwrite(bottom_filename, bottom_image)
            if mask is not None:
                # Save mask as binary .npy file for efficient storage
                mask_filename = os.path.join(
                    self.output_mask_dir, f"mask_{self.frame_counter}.npy"
                )
                np.save(mask_filename, mask)
        except Exception as e:
            logger.exception("Error saving individual images:", e)

    def _start_or_restart_video_writers(self) -> None:
        """Restart the video writers if not saving images individually."""
        if not self.save_images_individually:
            with self.writer_lock:
                self._release_writers()
                self.video_writer_top, self.video_writer_bottom = (
                    self._start_new_video()
                )

    def _release_writers(self) -> None:
        """Release the video writers if they have been initialized."""
        with self.writer_lock:
            if self.video_writer_top:
                self.video_writer_top.release()
                self.video_writer_top = None
            if self.video_writer_bottom:
                self.video_writer_bottom.release()
                self.video_writer_bottom = None
        
        # Save action summary when recording ends
        if self.disk_enabled:
            self.save_action_summary()

    def start_new_recording(self, new_output_dir: str) -> None:
        """Start a new recording session in the specified directory."""
        logger.info(f"start_new_recording called with: {new_output_dir}")
        if not new_output_dir:
            logger.error(f"start_new_recording called with empty output_dir! This is a critical error.")
            raise ValueError("start_new_recording called with empty output_dir")
        try:
            self.output_dir = new_output_dir
            self.disk_enabled = True
            self.frame_counter = 0
            self.video_counter = 0
            # Clear action tracker for new session to ensure actions.json only contains actions from this session
            self.shared_state.action_tracker.clear()
            logger.info(f"disk_enabled set to True. Calling _initialize_directories()")
            self._initialize_directories()
            self._prepare_new_recording()
            logger.info("Started new recording in directory: %s", new_output_dir)
        except Exception as e:
            logger.error(f"Exception in start_new_recording: {e}", exc_info=True)
            raise

    def disable_recording(self) -> None:
        """Disable disk saving without starting a new recording session."""
        if self.disk_enabled:
            # Save action summary before disabling
            self.save_action_summary()
        self.disk_enabled = False
        logger.info("Disk saving disabled")

    def _prepare_new_recording(self) -> None:
        """Prepare for a new recording session."""
        self.stop_flag = False
        if self.disk_enabled and not self.save_images_individually:
            self._start_or_restart_video_writers()

    def stop(self) -> None:
        """Stop the recorder, segmentation thread, and save action summary."""
        if self.disk_enabled:
            # Save action summary before stopping
            self.save_action_summary()
            self.disk_enabled = False
        self.stop_flag = True
        
        # Wait for segmentation thread to finish
        if self.segmentation_thread is not None and self.segmentation_thread.is_alive():
            logger.info("Waiting for segmentation thread to stop...")
            self.segmentation_thread.join(timeout=2.0)
            if self.segmentation_thread.is_alive():
                logger.warning("Segmentation thread did not stop in time")
        
        logger.info("Stop flag set to True in Recorder")

    def save_state(self) -> None:
        """
        Save the current state to a JSON file, including action metadata if available.
        
        Each frame entry includes:
        - Robot state
        - Timestamp
        - Frame index
        - Associated action information (if any)
        """
        try:
            state = self.state.copy() if isinstance(self.state, dict) else self.state
            timestamp = self.timestamp
            state = convert_ndarray_to_list(state)
            if not isinstance(state, dict):
                state = {"state": state}
            state["time"] = timestamp
            state["frame_index"] = self.frame_counter

            # Add action metadata if available
            action = self.shared_state.action_tracker.get_action_for_frame(self.frame_counter)
            if action is not None:
                state["action"] = {
                    "action_id": action.action_id,
                    "action_type": action.action_type.value,
                    "phase": action.phase.value,
                    "start_frame": action.start_frame,
                    "is_planar_2d": action.is_planar_2d,
                    "action_details": action.action_details,
                    "description": action.description,
                }

            state_file = os.path.join(self.output_dir, "states.json")
            data: List[Dict[str, Any]] = []
            if os.path.exists(state_file):
                with open(state_file, "r") as file:
                    data = json.load(file)
            data.append(state)
            with open(state_file, "w") as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            logger.exception("Error saving state:", e)

    def save_action_summary(self) -> None:
        """
        Save action summary to a separate JSON file.
        
        This provides a high-level overview of all actions performed during recording,
        useful for dataset creation and analysis.
        """
        try:
            actions = self.shared_state.action_tracker.get_all_actions()
            action_data = {
                "total_actions": len(actions),
                "actions": [action.to_dict() for action in actions],
            }
            action_file = os.path.join(self.output_dir, "actions.json")
            if len(actions) == 0:
                return # either no actions to save or actions were saved and cleared already, so skip saving empty file
            with open(action_file, "w") as f:
                json.dump(action_data, f, indent=4)
            logger.info(
                f"Saved {len(actions)} actions to {action_file}"
            )
        except Exception as e:
            logger.exception("Error saving action summary:", e)


    def ensure_images(self) -> bool:
        """Ensure that valid images are available. Try updating if not."""
        with self.image_lock:
            if self.image_top is None or self.bottom_image is None:
                self._update()
            if self.image_top is None or self.bottom_image is None:
                logger.error(
                    "ensure_images: Failed to obtain valid images from the robot after update."
                )
                return False
        return True
