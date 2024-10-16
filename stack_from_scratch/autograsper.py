import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional
from enum import Enum
import asyncio
import logging
import time
from Task import Task
from library.rgb_object_tracker import get_object_pos
from library.utils import (
    OrderType,
    get_undistorted_bottom_image,
    queue_orders,
)

from client.robot_interface import IRobot

# Set up logging configuration
logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'default': {'format': '%(asctime)s %(levelname)s:%(name)s:%(message)s'}
    },
    'handlers': {
        'console': {'class': 'logging.StreamHandler', 'formatter': 'default'}
    },
    'root': {'handlers': ['console'], 'level': 'INFO'},
})
logger = logging.getLogger(__name__)


class RobotActivity(Enum):
    STARTUP = 1
    ACTIVE = 2
    RESETTING = 3
    FINISHED = 4


# Custom exception classes
class AutograsperError(Exception):
    """Base class for Autograsper exceptions."""
    pass


class RobotInitializationError(AutograsperError):
    """Exception raised for errors during robot initialization."""
    pass


class InvalidStateTransitionError(AutograsperError):
    """Exception raised for invalid state transitions."""
    pass


@dataclass
class Autograsper:
    """
    Autograsper class controls the robotic gripper to pick up and place objects.
    """
    robot: IRobot
    config: dict
    robot_idx: int  # Now explicitly passed
    output_dir: str = ""
    start_time: float = field(default_factory=time.time)
    camera_matrix: np.ndarray = field(init=False)
    distortion_coefficients: np.ndarray = field(init=False)
    state: RobotActivity = RobotActivity.STARTUP
    _start_event: asyncio.Event = field(
        default_factory=asyncio.Event, init=False)
    bottom_image: np.ndarray = field(init=False)

    DEFAULT_STACK_POSITION: List[float] = field(
        default_factory=lambda: [0.5, 0.5])
    DEFAULT_TIME_BETWEEN_ORDERS: float = 1.5
    GRIPPER_OFFSET: float = 0.20
    MINIMUM_GRASP_HEIGHT: float = 0.02

    def __post_init__(self):
        self.camera_matrix = np.array(self.config['camera']['m'])
        self.distortion_coefficients = np.array(self.config['camera']['d'])

        try:
            # Robot is already injected via dependency injection
            self.bottom_image = self.get_undistorted_bottom_image()
        except Exception as e:
            logger.error(f"Failed to get undistorted bottom image: {e}")
            raise

    async def set_start_flag(self) -> None:
        """Sets the start flag to signal the robot to begin."""
        self._start_event.set()

    async def clear_start_flag(self) -> None:
        """Clears the start flag."""
        self._start_event.clear()

    async def is_start_flag_set(self) -> bool:
        """Checks if the start flag is set."""
        return self._start_event.is_set()

    async def queue_robot_orders(self, orders: List[Tuple[OrderType, List[Any]]], delay: float) -> None:
        """
        Queue a list of orders to the robot asynchronously.

        :param orders: List of orders to queue.
        :param delay: Time delay between orders.
        """
        await queue_orders(self.robot, orders, delay, output_dir=self.output_dir)

    async def pickup_and_place_object(self, object_position: Tuple[float, float], object_height: float, target_height: float,
                                      target_position: List[float] = None,
                                      time_between_orders: float = None) -> None:
        """
        Pickup and place an object from one position to another asynchronously.

        :param object_position: Position of the object to pick up.
        :param object_height: Height of the object.
        :param target_height: Target height for placing the object.
        :param target_position: Target position for placing the object.
        :param time_between_orders: Time to wait between orders.
        """
        if target_position is None:
            target_position = self.DEFAULT_STACK_POSITION
        if time_between_orders is None:
            time_between_orders = self.DEFAULT_TIME_BETWEEN_ORDERS

        if not isinstance(object_position, (list, tuple)) or len(object_position) != 2:
            raise ValueError("object_position must be a tuple of two floats")
        if not isinstance(target_position, (list, tuple)) or len(target_position) != 2:
            raise ValueError("target_position must be a list of two floats")

        orders = self._generate_pickup_and_place_orders(
            object_position, object_height, target_height, target_position)
        await self.queue_robot_orders(orders, time_between_orders)

    def _generate_pickup_and_place_orders(self, object_position: Tuple[float, float], object_height: float,
                                          target_height: float, target_position: List[float]) -> List[Tuple[OrderType, List[Any]]]:
        """
        Generate orders for picking up and placing an object.

        :param object_position: Position of the object to pick up.
        :param object_height: Height of the object.
        :param target_height: Target height for placing the object.
        :param target_position: Target position for placing the object.
        :return: List of orders.
        """
        return [
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, object_position),
            (OrderType.GRIPPER_OPEN, []),
            (OrderType.MOVE_Z, [object_height]),
            (OrderType.GRIPPER_CLOSE, []),
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, target_position),
            (OrderType.MOVE_Z, [target_height]),
            (OrderType.GRIPPER_OPEN, []),
        ]

    async def reset(self, block_positions: List[List[float]], block_heights: np.ndarray,
                    stack_position: List[float] = None,
                    time_between_orders: float = None) -> None:
        """
        Reset the blocks to their initial positions asynchronously.

        :param block_positions: Positions of the blocks.
        :param block_heights: Heights of the blocks.
        :param stack_position: Position for stacking the blocks.
        :param time_between_orders: Time to wait between orders.
        """
        if stack_position is None:
            stack_position = self.DEFAULT_STACK_POSITION
        if time_between_orders is None:
            time_between_orders = self.DEFAULT_TIME_BETWEEN_ORDERS

        rev_heights = np.flip(block_heights.copy())
        rev_positions = block_positions[::-1]
        target_z = sum(rev_heights)

        for index, (block_pos, block_height) in enumerate(zip(rev_positions, rev_heights)):
            target_z -= block_height
            orders = self._generate_reset_orders(
                index, block_pos, stack_position, target_z, rev_heights)
            await self.queue_robot_orders(orders, time_between_orders)

    def _generate_reset_orders(self, index: int, block_pos: List[float], stack_position: List[float],
                               target_z: float, rev_heights: np.ndarray) -> List[Tuple[OrderType, List[Any]]]:
        """
        Generate orders for resetting a block.

        :param index: Index of the block.
        :param block_pos: Position of the block.
        :param stack_position: Position for stacking the blocks.
        :param target_z: Target height for placing the block.
        :param rev_heights: Reversed block heights.
        :return: List of orders.
        """
        orders = []
        if index == 0:
            orders += [
                (OrderType.MOVE_Z, [1]),
                (OrderType.MOVE_XY, stack_position),
                (OrderType.GRIPPER_OPEN, []),
            ]
        orders += [
            (OrderType.MOVE_Z, [target_z]),
            (OrderType.GRIPPER_CLOSE, []),
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, block_pos),
            (OrderType.MOVE_Z, [0]),
            (OrderType.GRIPPER_OPEN, []),
        ]
        if index != len(rev_heights) - 1:
            orders += [
                (OrderType.MOVE_Z, [target_z]),
                (OrderType.MOVE_XY, stack_position),
            ]
        return orders

    def prepare_experiment(self, config: dict) -> Tuple[np.ndarray, List[float]]:
        """
        Prepare the experiment by setting default positions from configuration.

        :param config: Configuration dictionary.
        :return: Tuple containing position_bank and stack_position.
        """
        position_bank = np.array(config['experiment']['position_bank'])
        stack_position = config['experiment']['stack_position']
        return position_bank, stack_position

    def get_undistorted_bottom_image(self) -> np.ndarray:
        """
        Gets the undistorted bottom image from the robot's camera.

        :return: Undistorted bottom image.
        """
        try:
            image = get_undistorted_bottom_image(
                self.robot, self.camera_matrix, self.distortion_coefficients)
            return image
        except Exception as e:
            logger.error(f"Failed to get undistorted bottom image: {e}")
            raise

    def update_bottom_image(self) -> None:
        """
        Updates the bottom image by capturing a new undistorted image from the robot's camera.
        """
        self.bottom_image = self.get_undistorted_bottom_image()

    async def run_task(self, task: Task) -> None:
        """
        Runs the provided task, handling errors and state transitions.
        """
        while self.state != RobotActivity.FINISHED:
            try:
                await self.go_to_start()
                self.set_state(RobotActivity.ACTIVE)

                await self.wait_for_start_signal()
                await self.clear_start_flag()

                # Execute the task
                await task.execute(self)

                await self.go_to_start()
                await asyncio.sleep(1)
                self.set_state(RobotActivity.RESETTING)
                await asyncio.sleep(1)

                # Detect errors
                if await task.detect_errors(self):
                    logger.info("Errors detected, recovering")
                    await task.recover_from_errors(self)
                else:
                    await task.reset_scene(self)

                self.set_state(RobotActivity.STARTUP)

            except Exception as e:
                logger.error(f"Run task loop: An exception occurred: {e}")
                raise

    async def go_to_start(self) -> None:
        """
        Move the robot to the start position asynchronously.
        """
        positions = np.array([[1, 0.7], [0, 0.7]])
        position = positions[np.random.choice(len(positions))]

        orders = [
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, position),
        ]
        await self.queue_robot_orders(orders, 2)

    async def wait_for_start_signal(self) -> None:
        """
        Waits for the start signal to begin the operation asynchronously.
        """
        await self._start_event.wait()

    def set_state(self, new_state: RobotActivity) -> None:
        """
        Sets the new state of the robot, ensuring valid transitions.

        :param new_state: The new state to transition to.
        """
        valid_transitions = {
            RobotActivity.STARTUP: [RobotActivity.ACTIVE],
            RobotActivity.ACTIVE: [RobotActivity.RESETTING, RobotActivity.FINISHED],
            RobotActivity.RESETTING: [RobotActivity.STARTUP, RobotActivity.FINISHED],
        }
        if new_state in valid_transitions.get(self.state, []):
            self.state = new_state
        else:
            raise InvalidStateTransitionError(
                f"Invalid state transition from {self.state} to {new_state}")

    async def shutdown(self) -> None:
        """
        Performs cleanup and resource release asynchronously.
        """
        pass

    async def __aenter__(self):
        # Asynchronous initialization if needed
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.shutdown()