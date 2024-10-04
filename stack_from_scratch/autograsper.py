import os
import sys
import time
from enum import Enum
from typing import List, Tuple, Any

import numpy as np
from dotenv import load_dotenv

# Ensure the project root is in the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import project-specific modules
from client.cloudgripper_client import GripperRobot
from library.rgb_object_tracker import all_objects_are_visible, get_object_pos
from library.utils import (OrderType, get_undistorted_bottom_image,
                           pick_random_positions, queue_orders, clear_center)

# Load environment variables
load_dotenv()

class RobotActivity(Enum):
    ACTIVE = 1
    RESETTING = 2
    FINISHED = 3
    STARTUP = 4


class Autograsper:
    def __init__(self, args, config: dict, output_dir: str = ""):
        """
        Initialize the Autograsper with the provided arguments and configuration.

        :param args: Command-line arguments
        :param config: Configuration dictionary
        :param output_dir: Directory to save state data
        """
        self.token = os.getenv("ROBOT_TOKEN")
        if not self.token:
            raise ValueError("ROBOT_TOKEN environment variable not set")

        self.output_dir = output_dir
        self.start_time = time.time()
        self.failed = False
        self.camera_matrix = np.array(config['camera']['m'])
        self.distortion_coefficients = np.array(config['camera']['d'])

        self.state = RobotActivity.STARTUP
        self.start_flag = False

        self.robot = self.initialize_robot(args.robot_idx, self.token)
        self.robot_idx = args.robot_idx
        self.bottom_image = get_undistorted_bottom_image(self.robot, self.camera_matrix, self.distortion_coefficients)

    @staticmethod
    def initialize_robot(robot_idx: int, token: str) -> GripperRobot:
        """
        Initialize the GripperRobot instance.

        :param robot_idx: Index of the robot.
        :param token: Authentication token for the robot.
        :return: GripperRobot instance.
        """
        try:
            return GripperRobot(robot_idx, token)
        except Exception as e:
            raise ValueError("Invalid robot ID or token") from e

    def queue_robot_orders(self, orders: List[Tuple[OrderType, List]], delay: float):
        """
        Queue a list of orders to the robot.

        :param orders: List of orders to queue.
        :param delay: Time delay between orders.
        """
        queue_orders(self.robot, orders, delay, output_dir=self.output_dir)

    def pickup_and_place_object(self, object_position: Tuple[float, float], object_height: float, target_height: float,
                                target_position: List[float] = [0.5, 0.5], time_between_orders: float = 1.5):
        """
        Pickup and place an object from one position to another.

        :param object_position: Position of the object to pick up.
        :param object_height: Height of the object.
        :param target_height: Target height for placing the object.
        :param target_position: Target position for placing the object.
        :param time_between_orders: Time to wait between orders.
        """
        orders = self._generate_pickup_and_place_orders(object_position, object_height, target_height, target_position)
        self.queue_robot_orders(orders, time_between_orders)

    def _generate_pickup_and_place_orders(self, object_position: Tuple[float, float], object_height: float,
                                          target_height: float, target_position: List[float]) -> List[Tuple[OrderType, List]]:
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

    def reset(self, block_positions: List[List[float]], block_heights: np.ndarray,
              stack_position: List[float] = [0.5, 0.5], time_between_orders: float = 1.5):
        """
        Reset the blocks to their initial positions.

        :param block_positions: Positions of the blocks.
        :param block_heights: Heights of the blocks.
        :param stack_position: Position for stacking the blocks.
        :param time_between_orders: Time to wait between orders.
        """
        rev_heights = np.flip(block_heights.copy())
        target_z = sum(rev_heights)

        for index, block_pos in enumerate(block_positions):
            target_z -= rev_heights[index]
            orders = self._generate_reset_orders(index, block_pos, stack_position, target_z, rev_heights)
            self.queue_robot_orders(orders, time_between_orders)

    def _generate_reset_orders(self, index: int, block_pos: List[float], stack_position: List[float],
                               target_z: float, rev_heights: np.ndarray) -> List[Tuple[OrderType, List]]:
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

    def run_grasping(self, colors, block_heights, config: dict, object_size: float = 2):
        """
        Run the main grasping loop.

        :param colors: List of colors for the blocks.
        :param block_heights: List of heights corresponding to each block.
        :param config: Configuration dictionary.
        :param object_size: Size of the objects.
        """
        position_bank, stack_position = self.prepare_experiment(config)

        if not all_objects_are_visible(colors, self.bottom_image):
            print("All blocks not visible")

        while self.state is not RobotActivity.FINISHED:
            try:
                self.go_to_start()
                self.state = RobotActivity.ACTIVE

                self.wait_for_start_signal()
                self.start_flag = False

                self.robot.move_xy(0.5, 0.5)
                time.sleep(1)
                self.robot.move_xy(0.9, 0.2)
                time.sleep(1)
                self.robot.move_xy(0.1, 0.5)
                time.sleep(2)

                self.go_to_start()
                time.sleep(1)
                self.state = RobotActivity.RESETTING
                time.sleep(1)

                if self.failed:
                    print("Experiment failed, recovering")
                    self.recover_after_fail()
                    self.failed = False
                else:
                    random_reset_positions = pick_random_positions(
                        position_bank, len(block_heights), object_size
                    )
                    self.reset(
                        random_reset_positions,
                        block_heights,
                        stack_position=stack_position,
                    )
                    self.go_to_start()

                self.state = RobotActivity.STARTUP

            except Exception as e:
                print(f"Run grasping loop: An exception of type {type(e).__name__} occurred. Arguments: {e.args}")
                raise Exception("Autograsping failed")

    def go_to_start(self):
        """
        Move the robot to the start position.
        """
        positions = [[1, 0.7], [0, 0.7]]
        positions = np.array(positions)

        position = np.choose(1, positions)

        orders = [
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, position),
        ]
        self.queue_robot_orders(orders, 2)