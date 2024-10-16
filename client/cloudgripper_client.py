import base64
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from requests import exceptions, get
import asyncio

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from robot_interface import IRobot

class GripperRobot(IRobot):
    """
    A class to represent a gripper robot and interact with its API.

    Attributes:
        api_address_robots (dict): Dictionary mapping robot names to their API addresses.
        name (str): The name of the robot.
        headers (dict): The headers to be sent with each API request.
        base_api (str): The base API URL for the robot.
        order_count (int): Counter for the number of orders sent to the robot.
    """

    api_address_robots = {f"robot{i}": f"https://cloudgripper.eecs.kth.se:8443/robot{i}/api/v1.1/robot" for i in range(1, 33)}

    def __init__(self, robot_idx: int, token: str):
        """
        Initialize the GripperRobot with a robot index and API token.

        Args:
            robot_idx (int): The index of the robot.
            token (str): The API token for authentication.
        """
        self.name = f"robot{robot_idx}"
        self.headers = {"apiKey": token}
        self.base_api = self.api_address_robots[self.name]
        self.order_count = 0

    def _make_request(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """
        Make a GET request to the robot's API.

        Args:
            endpoint (str): The API endpoint to call.

        Returns:
            Optional[dict]: The JSON response from the API if successful, otherwise None.
        """
        try:
            response = get(f"{self.base_api}/{endpoint}", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except exceptions.RequestException as e:
            print(f"Request to {endpoint} failed:", e)
            return None

    def _safe_get(self, response: Optional[Dict[str, Any]], key: str) -> Optional[Any]:
        """
        Safely get a value from the response dictionary.

        Args:
            response (Optional[dict]): The response dictionary.
            key (str): The key to retrieve the value for.

        Returns:
            Optional[Any]: The value if the key exists, otherwise None.
        """
        if response and key in response:
            return response[key]
        return None

    async def move_xy(self, position: Tuple[float, float]) -> None:
        """
        Move the robot along the X and Y axes asynchronously.

        Args:
            position (Tuple[float, float]): The (x, y) position to move to.
        """
        x, y = position
        # Convert positions to integers if necessary
        x_int = int(x)
        y_int = int(y)
        await asyncio.to_thread(self._move_xy_sync, x_int, y_int)

    def _move_xy_sync(self, x: int, y: int) -> None:
        """
        Move the robot along the X and Y axes synchronously.

        Args:
            x (int): The distance to move along the X-axis.
            y (int): The distance to move along the Y-axis.
        """
        self._make_request(f"gcode/{x}/{y}")

    async def move_z(self, height: float) -> None:
        """
        Move the robot along the Z-axis asynchronously.

        Args:
            height (float): The height to move to.
        """
        z_int = int(height)
        await asyncio.to_thread(self._move_z_sync, z_int)

    def _move_z_sync(self, z: int) -> None:
        """
        Move the robot along the Z-axis synchronously.

        Args:
            z (int): The height to move to.
        """
        self._make_request(f"up_down/{z}")

    async def gripper_open(self) -> None:
        """
        Open the robot's gripper asynchronously.
        """
        await asyncio.to_thread(self._gripper_open_sync)

    def _gripper_open_sync(self) -> None:
        """
        Open the robot's gripper synchronously.
        """
        self.move_gripper(1)

    async def gripper_close(self) -> None:
        """
        Close the robot's gripper asynchronously.
        """
        await asyncio.to_thread(self._gripper_close_sync)

    def _gripper_close_sync(self) -> None:
        """
        Close the robot's gripper synchronously.
        """
        self.move_gripper(0)

    def move_gripper(self, angle: int) -> None:
        """
        Move the robot's gripper to a specified angle.

        Args:
            angle (int): The angle to move the gripper to.
        """
        self._make_request(f"grip/{angle}")

    async def get_bottom_image(self) -> np.ndarray:
        """
        Get the bottom image from the robot's camera asynchronously.

        Returns:
            np.ndarray: The bottom image as a numpy array.
        """
        return await asyncio.to_thread(self._get_bottom_image_sync)

    def _get_bottom_image_sync(self) -> np.ndarray:
        """
        Get the bottom image from the robot's camera synchronously.

        Returns:
            np.ndarray: The bottom image as a numpy array.
        """
        image, _ = self.get_image_base()
        return image

    # Existing methods from the original GripperRobot class
    def get_state(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the current state of the robot.

        Returns:
            tuple: The state and timestamp of the robot.
        """
        response = self._make_request("getState")
        return self._safe_get(response, "state"), self._safe_get(response, "timestamp")

    def get_image_base(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Get the base (bottom) image from the robot's camera.

        Returns:
            Tuple[Optional[np.ndarray], Optional[str]]: The image as a numpy array and the timestamp.
        """
        return self._get_image("getImageBase")

    def get_image_top(self) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Get the top image from the robot's camera.

        Returns:
            Tuple[Optional[np.ndarray], Optional[str]]: The image as a numpy array and the timestamp.
        """
        return self._get_image("getImageTop")

    def _decode_image(self, image_str: str) -> Optional[np.ndarray]:
        """
        Decode a base64-encoded image string into a numpy array.

        Args:
            image_str (str): The base64-encoded image string.

        Returns:
            Optional[np.ndarray]: The decoded image as a numpy array.
        """
        try:
            img_bytes = base64.b64decode(image_str.encode("latin1"))
            np_img = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f"Image decoding failed: {e}")
            return None

    def _get_image(
        self, endpoint: str
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Retrieve an image from the robot's camera.

        Args:
            endpoint (str): The API endpoint to call for the image.

        Returns:
            Tuple[Optional[np.ndarray], Optional[str]]: The image as a numpy array and the timestamp.
        """
        response = self._make_request(endpoint)
        image_data = self._safe_get(response, "data")
        time_stamp = self._safe_get(response, "time")

        if image_data:
            image = self._decode_image(image_data)
            return image, time_stamp
        else:
            print("Image not available")
            return None, None
