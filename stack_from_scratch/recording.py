from library.utils import (
    convert_ndarray_to_list,
    get_undistorted_bottom_image,
)  # Assuming these are the correct imports
# Assuming this is the correct import
from client.cloudgripper_client import GripperRobot
import os
import sys
import time
import json
import logging
import asyncio
from typing import Any, Tuple, List, Dict
import cv2

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Recorder:
    FPS = 3
    FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

    def __init__(
        self, session_id: str, output_dir: str, m: Any, d: Any, token: str, idx: int
    ):
        self.session_id = session_id
        self.output_dir = output_dir
        self.m = m
        self.d = d
        self.token = token
        self.robot_idx = idx
        self.stop_flag = False
        self.frame_counter = 0
        self.video_counter = 0
        self.video_writer_top = None
        self.video_writer_bottom = None
        self.lock = asyncio.Lock()  # For thread safety

        self.robot = GripperRobot(self.robot_idx, self.token)
        self.image_top, _ = self.robot.get_image_top()
        self.bottom_image = get_undistorted_bottom_image(
            self.robot, self.m, self.d)

        self._initialize_directories()

    def _initialize_directories(self) -> None:
        """Initialize output directories."""
        self.output_video_dir = os.path.join(self.output_dir, "Video")
        self.output_bottom_video_dir = os.path.join(
            self.output_dir, "Bottom_Video")
        self.final_image_dir = os.path.join(self.output_dir, "Final_Image")
        os.makedirs(self.output_video_dir, exist_ok=True)
        os.makedirs(self.output_bottom_video_dir, exist_ok=True)
        os.makedirs(self.final_image_dir, exist_ok=True)

    def _start_new_video(self) -> Tuple[cv2.VideoWriter, cv2.VideoWriter]:
        """Start new video writers for top and bottom cameras."""
        video_filename_top = os.path.join(
            self.output_video_dir, f"video_{self.video_counter}.mp4"
        )
        video_writer_top = cv2.VideoWriter(
            video_filename_top, self.FOURCC, self.FPS, self.image_top.shape[1::-1]
        )

        video_filename_bottom = os.path.join(
            self.output_bottom_video_dir, f"video_{self.video_counter}.mp4"
        )
        video_writer_bottom = cv2.VideoWriter(
            video_filename_bottom, self.FOURCC, self.FPS, self.bottom_image.shape[1::-1]
        )

        return video_writer_top, video_writer_bottom

    async def record(self, start_new_video_every: int = 30) -> None:
        """Asynchronous recording loop with optional periodic video restarts."""
        self._prepare_new_recording()
        try:
            while not self.stop_flag:
                await self._capture_frame()
                if (
                    start_new_video_every
                    and self.frame_counter % start_new_video_every == 0
                    and self.frame_counter != 0
                ):
                    self.video_counter += 1
                    await self._start_or_restart_video_writers()

                await asyncio.sleep(1 / self.FPS)
                await self.save_state(self.robot)
                self.frame_counter += 1
                logger.info("Frames recorded: %d", self.frame_counter)

                # Optionally display the bottom image
                # cv2.imshow(f"ImageBottom_{self.robot_idx}", self.bottom_image)
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     self.stop_flag = True
        except Exception as e:
            logger.error("An error occurred: %s", e)
        finally:
            await self._release_writers()
            # cv2.destroyAllWindows()

    async def _capture_frame(self) -> None:
        """Capture frames from the robot's cameras and write directly to video file."""
        async with self.lock:
            try:
                image_top, _ = await self.robot.get_image_top_async()
                bottom_image = await get_undistorted_bottom_image_async(self.robot, self.m, self.d)

                if self.video_writer_top and self.video_writer_bottom:
                    self.video_writer_top.write(image_top)
                    self.video_writer_bottom.write(bottom_image)
                else:
                    logger.warning("Video writers not initialized.")

                self.image_top = image_top
                self.bottom_image = bottom_image
            except Exception as e:
                logger.error("Error capturing frame: %s", e)

    async def _start_or_restart_video_writers(self) -> None:
        """Start or restart video writers."""
        await self._release_writers()  # Ensure the old writers are released
        self.video_writer_top, self.video_writer_bottom = self._start_new_video()

    async def _release_writers(self) -> None:
        """Release the video writers."""
        if self.video_writer_top:
            self.video_writer_top.release()
            self.video_writer_top = None
        if self.video_writer_bottom:
            self.video_writer_bottom.release()
            self.video_writer_bottom = None

    async def write_final_image(self) -> None:
        """Write the final image from the top camera."""
        try:
            logger.info("Writing final image")
            image_top, _ = await self.robot.get_image_top_async()
            final_image_path = os.path.join(
                self.final_image_dir, f"final_image_{self.video_counter}.jpg"
            )
            cv2.imwrite(final_image_path, image_top)
        except Exception as e:
            logger.error("Error writing final image: %s", e)

    async def start_new_recording(self, new_output_dir: str) -> None:
        """Start a new recording session with a new output directory."""
        self.output_dir = new_output_dir
        self._initialize_directories()
        await self._prepare_new_recording()
        logger.info("Started new recording in directory: %s", new_output_dir)

    async def _prepare_new_recording(self) -> None:
        """Prepare for a new recording session."""
        self.frame_counter = 0
        self.video_counter = 0
        self.stop_flag = False
        await self._start_or_restart_video_writers()

    async def stop(self) -> None:
        """Set the stop flag to terminate recording."""
        self.stop_flag = True
        await self._release_writers()
        logger.info("Recording stopped.")

    async def save_state(self, robot: GripperRobot) -> None:
        """Save the state of the robot to a JSON file."""
        try:
            state, timestamp = await robot.get_state_async()
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
            logger.error("Error saving state: %s", e)
