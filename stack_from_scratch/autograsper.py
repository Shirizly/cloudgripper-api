import argparse
import os
import random
import sys
import time
from enum import Enum
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv
from pynput import keyboard

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from client.cloudgripper_client import GripperRobot
from library.calibration import order2movement
from library.Camera2Robot import Camera2Robot, cam_to_robot
from library.rgb_object_tracker import all_objects_are_visible, object_tracking
from library.utils import (
    OrderType,
    generate_position_grid,
    get_undistorted_bottom_image,
    pick_random_positions,
    queue_orders,
    queue_orders_with_input,
    save_state,
    sweep_straight,
)

load_dotenv()


class RobotActivity(Enum):
    ACTIVE = 1
    RESETTING = 2
    FINISHED = 3
    STARTUP = 4


class Autograsper:
    def __init__(self, args, output_dir):
        """
        Initialize the Autograsper with the provided arguments.

        :param args: Command-line arguments
        :param output_dir: Directory to save state data
        """
        self.token = os.getenv("ROBOT_TOKEN", "default_token")
        if self.token == "default_token":
            raise ValueError("ROBOT_TOKEN environment variable not set")

        self.output_dir = output_dir
        self.start_time = time.time()

        self.m = np.array(
            [
                [505.24537524391866, 0.0, 324.5096286632362],
                [0.0, 505.6456651337437, 233.54118730278543],
                [0.0, 0.0, 1.0],
            ]
        )
        self.d = np.array(
            [
                -0.07727407195057368,
                -0.047989733519315944,
                0.12157420705123315,
                -0.09667542135039282,
            ]
        )

        self.state = RobotActivity.STARTUP

        try:
            self.robot = GripperRobot(args.robot_idx, self.token)
        except Exception as e:
            raise ValueError("Invalid robot ID or token") from e

        self.robot_idx = args.robot_idx

        
    def pickup_and_place_object(
        self,
        object_position: Tuple[float, float],
        object_height: float,
        target_height: float,
        target_position: List[float] = [0.5, 0.5],
        time_between_orders: int = 2,
    ):
        """
        Pickup and place an object from one position to another.

        :param object_position: Position of the object to pick up
        :param object_height: Height of the object
        :param target_height: Target height for placing the object
        :param target_position: Target position for placing the object
        :param time_between_orders: Time to wait between orders
        """
        movement = object_position

        order_list = [
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, movement),
            (OrderType.GRIPPER_OPEN, []),
            (OrderType.MOVE_Z, [object_height]),
            (OrderType.GRIPPER_CLOSE, []),
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, target_position),
            (OrderType.MOVE_Z, [target_height]),
            (OrderType.GRIPPER_OPEN, []),
        ]

        queue_orders_with_input(
            self.robot, order_list, self.output_dir, self.start_time
        )

    def reset(
        self,
        block_positions: List[List[float]],
        block_heights: np.ndarray,
        stack_position: List[float] = [0.5, 0.5],
        time_between_orders: int = 2,
    ):
        """
        Reset the blocks to their initial positions.

        :param block_positions: Positions of the blocks
        :param block_heights: Heights of the blocks
        :param stack_position: Position for stacking the blocks
        :param time_between_orders: Time to wait between orders
        """
        rev_heights = np.flip(block_heights.copy())

        target_z = sum(rev_heights)

        for index, block_pos in enumerate(block_positions):
            target_z -= rev_heights[index]

            order_list = []

            # this can be enabled if before resetting the gripper moves to a random position, to make the destacking data more general
            # if index == 0:
            #     order_list += [
            #         (OrderType.MOVE_Z, [1]),
            #         (OrderType.MOVE_XY, stack_position),
            #         (OrderType.GRIPPER_OPEN, None),
            #     ]

            order_list += [
                (OrderType.MOVE_Z, [target_z]),
                (OrderType.GRIPPER_CLOSE, []),
                (OrderType.MOVE_Z, [1]),
                (OrderType.MOVE_XY, block_pos),
                (OrderType.MOVE_Z, [0]),
                (OrderType.GRIPPER_OPEN, None),
                (OrderType.MOVE_Z, [1]),
                (OrderType.MOVE_XY, stack_position),
            ]

            queue_orders_with_input(
                self.robot, order_list, self.output_dir, self.start_time
            )

    def clear_center(self):
        """
        Clear the center area of the workspace.
        """
        commands = [
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, [0.3, 0.3]),
            (OrderType.MOVE_Z, [0.2]),
            (OrderType.MOVE_XY, [0.3, 0.8]),
            (OrderType.MOVE_XY, [0.5, 0.8]),
            (OrderType.MOVE_XY, [0.5, 0.4]),
            (OrderType.MOVE_XY, [0.7, 0.4]),
            (OrderType.MOVE_XY, [0.7, 0.8]),
            (OrderType.MOVE_Z, [1]),
        ]

        queue_orders(self.robot, commands, 2, self.output_dir, self.start_time)

    def startup(self, position: List[float]):
        """
        Perform startup sequence at a given position.

        :param position: Position to start up at
        """
        self.robot.rotate(0)
        time.sleep(0.5)
        startup_commands = [
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, position),
        ]

        queue_orders(self.robot, startup_commands, 1)
        time.sleep(1)

    def run_calibration(self):
        commands = [
            (OrderType.GRIPPER_CLOSE, None),
            (OrderType.MOVE_Z, [1.0]),
            (OrderType.MOVE_XY, [0.0, 0.0]),
            (OrderType.MOVE_Z, [0.3]),
            (OrderType.MOVE_Z, [1.0]),
            (OrderType.MOVE_XY, [1.0, 0.0]),
            (OrderType.MOVE_Z, [0.3]),
            (OrderType.MOVE_Z, [1.0]),
            (OrderType.MOVE_XY, [0.0, 1.0]),
            (OrderType.MOVE_Z, [0.3]),
            (OrderType.MOVE_Z, [1.0]),
            (OrderType.MOVE_XY, [1.0, 1.0]),
            (OrderType.MOVE_Z, [0.3]),
            (OrderType.MOVE_Z, [1.0]),
            (OrderType.MOVE_XY, [1.0, 1.0]),
        ]

        queue_orders_with_input(self.robot, commands, self.output_dir, self.start_time)

    def run_grasping(self):
        #self.manual_control()
        #return

        """
        Run the main grasping loop.
        """
        robot = self.robot
        m = self.m
        d = self.d
        robot_idx = self.robot_idx

        position_bank = generate_position_grid()
        block_height = 0.3

        blocks = [
            ("red", block_height),
            ("green", block_height),
        ]
        n_layers = len(blocks)

        block_heights = np.repeat([block_height], n_layers)

        if not all_objects_are_visible(
            blocks, get_undistorted_bottom_image(robot, m, d)
        ):
            print("all blocks not visible")
            return

        while self.state is not RobotActivity.FINISHED:

            
            try:

                # Reset robot
                stack_height = 0
                startup_position = random.choice(position_bank)
                self.startup(startup_position)
                # we only want to start recording after this is finished

                # Start main task
                self.state = RobotActivity.ACTIVE

                save_state(self.robot, self.output_dir, self.start_time)

                for color, block_height in blocks:
                    camera_position = object_tracking(
                        get_undistorted_bottom_image(robot, m, d), color, DEBUG=False
                    )

                    object_position = cam_to_robot(robot_idx, camera_position)

                    self.pickup_and_place_object(
                        object_position,
                        max(block_height - 0.25, 0.02),  # grip at max grip surface area
                        stack_height,
                        time_between_orders=1,
                    )

                    stack_height += block_height

                self.state = RobotActivity.RESETTING

                random_reset_positions = pick_random_positions(
                    position_bank, n_layers, 0.2
                )

                save_state(self.robot, self.output_dir, self.start_time)  # Save initial stat

                self.reset(random_reset_positions, block_heights, time_between_orders=3)

                self.state = RobotActivity.STARTUP

                if not all_objects_are_visible(
                    blocks, get_undistorted_bottom_image(robot, m, d)
                ):
                    print("not all blocks found after reset, sweeping")
                    sweep_straight(robot)

            except Exception as e:
                print(
                    f"PAP loop: An exception of type {type(e).__name__} occurred. Arguments: {e.args}"
                )

    def manual_control(self):
        """
        Manually control the robot using keyboard inputs.
        """
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_rotation = 0
        self.current_angle = 0.4

        def on_press(key):
            try:
                if key.char == "w":
                    self.current_y += 0.1
                    self.current_y = min(max(self.current_y, 0), 1)
                    self.robot.move_xy(self.current_x, self.current_y)
                elif key.char == "a":
                    self.current_x -= 0.1
                    self.current_x = min(max(self.current_x, 0), 1)
                    self.robot.move_xy(self.current_x, self.current_y)
                elif key.char == "s":
                    self.current_y -= 0.1
                    self.current_y = min(max(self.current_y, 0), 1)
                    self.robot.move_xy(self.current_x, self.current_y)

                elif key.char == "x":
                    self.current_y -= 0.05
                    self.current_y = min(max(self.current_y, 0), 1)
                    self.robot.move_xy(self.current_x, self.current_y)

                elif key.char == "z":
                    self.current_y -= 0.01
                    self.current_y = min(max(self.current_y, 0), 1)
                    self.robot.move_xy(self.current_x, self.current_y)

                elif key.char == "d":
                    self.current_x += 0.1
                    self.current_x = min(max(self.current_x, 0), 1)
                    self.robot.move_xy(self.current_x, self.current_y)
                elif key.char == "r":
                    self.current_z += 0.1
                    self.current_z = min(max(self.current_z, 0), 1)
                    print(self.current_z)
                    self.robot.move_z(self.current_z)
                elif key.char == "f":
                    self.current_z -= 0.1
                    self.current_z = min(max(self.current_z, 0), 1)
                    print(self.current_z)
                    self.robot.move_z(self.current_z)
                elif key.char == "i":
                    self.current_angle += 0.05
                    self.current_angle = min(self.current_angle, 1)
                    print(self.current_angle)
                    self.robot.move_gripper(self.current_angle)
                elif key.char == "o":
                    self.current_angle += 0.01
                    self.current_angle = min(self.current_angle, 1)
                    print(self.current_angle)
                    self.robot.move_gripper(self.current_angle)
                elif key.char == "p":
                    self.current_angle -= 0.01
                    self.current_angle = max(self.current_angle, 0.2)
                    print(self.current_angle)
                    self.robot.move_gripper(self.current_angle)
                elif key.char == "q":
                    self.current_rotation -= 10
                    self.robot.rotate(self.current_rotation)
                elif key.char == "e":
                    self.current_rotation += 10
                    self.robot.rotate(self.current_rotation)

                elif key.char == "n":
                    self.robot.gripper_open()
                    time.sleep(1)
                    self.robot.move_z(0)
                    time.sleep(1)
                    self.robot.move_gripper(0.5)
                    time.sleep(1)
                    self.robot.move_z(1)
                    time.sleep(1)
                    self.robot.move_xy(
                        min(self.current_x + 0.2, 1), min(self.current_y + 0.2, 1)
                    )
                    time.sleep(1)
                    self.robot.move_xy(self.current_x, self.current_y)
                    time.sleep(1)
                    self.robot.move_z(0)
                    time.sleep(1)

                elif key.char == "b":
                    self.state = RobotActivity.FINISHED

            except Exception as e:
                print(e)

        def on_release(key):
            if key == keyboard.Key.esc:
                # Stop listener
                return False

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autograsper Robot Controller")
    parser.add_argument("--robot_idx", type=str, required=True, help="Robot index")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    args = parser.parse_args()

    autograsper = Autograsper(args, args.output_dir)
    autograsper.run_grasping()
