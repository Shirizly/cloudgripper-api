from abc import ABC, abstractmethod
import os
import sys
import time
from enum import Enum
from typing import List, Tuple
import ast

import numpy as np
from dotenv import load_dotenv

# Ensure the project root is in the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import project-specific modules
from client.cloudgripper_client import GripperRobot
from library.utils import (
    OrderType,
    get_undistorted_bottom_image,
    execute_order,
    parse_config,
)

# Load environment variables
load_dotenv()


class RobotActivity(Enum):
    ACTIVE = 1
    RESETTING = 2
    FINISHED = 3
    STARTUP = 4


class AutograsperBase(ABC):
    def __init__(
        self,
        config_file,
        output_dir: str = "",
    ):
        self.token = os.getenv("CLOUDGRIPPER_TOKEN")
        if not self.token:
            raise ValueError("CLOUDGRIPPER_TOKEN environment variable not set")

        self.output_dir = output_dir
        self.start_time = time.time()
        self.failed = False

        self.state = RobotActivity.STARTUP
        self.start_flag = False

        self.request_state_record = False

        # fudge time to ensure frames at the start/finish of task/resetting
        self.task_time_margin = 2
        self.robot_state = None

        config = parse_config(config_file)

        try:
            camera_cfg = config["camera"]
            self.camera_matrix = np.array(ast.literal_eval(camera_cfg["m"]))
            self.distortion_coeffs = np.array(ast.literal_eval(camera_cfg["d"]))
            self.record_only_after_action = bool(
                ast.literal_eval(camera_cfg["record_only_after_action"])
            )

            experiment_cfg = config["experiment"]
            self.robot_idx = ast.literal_eval(experiment_cfg["robot_idx"])
            self.time_between_orders = ast.literal_eval(
                experiment_cfg["time_between_orders"]
            )

        except Exception as e:
            raise ValueError("Grasper config.ini ERROR: ", e) from e

        self.robot = self.initialize_robot(self.robot_idx, self.token)

        self.bottom_image = get_undistorted_bottom_image(
            self.robot, self.camera_matrix, self.distortion_coeffs
        )

    @staticmethod
    def initialize_robot(robot_idx: int, token: str) -> GripperRobot:
        try:
            return GripperRobot(robot_idx, token)
        except Exception as e:
            raise ValueError("Invalid robot ID or token: ", e) from e

    def record_current_state(self):
        self.request_state_record = True
        while self.request_state_record:
            time.sleep(0.05)

    def wait_for_start_signal(self):
        while not self.start_flag:
            time.sleep(0.05)

    def run_grasping(self):
        while self.state != RobotActivity.FINISHED:
            self.startup()
            self.state = RobotActivity.ACTIVE

            self.wait_for_start_signal()
            self.start_flag = False

            try:
                self.perform_task()
            except Exception as e:
                print(f"Unexpected error during perform_task: {e}")
                self.failed = True
                raise

            if self.state == RobotActivity.FINISHED:
                break

            time.sleep(self.task_time_margin)
            self.state = RobotActivity.RESETTING
            time.sleep(self.task_time_margin)

            if self.failed:
                print("Experiment failed, recovering")
                self.recover_after_fail()
                self.failed = False
            else:
                self.reset_task()

            self.state = RobotActivity.STARTUP

    def recover_after_fail(self):
        pass

    @abstractmethod
    def perform_task(self):
        while True:
            print(
                "GRASPER: No task defined. Override `perform_task` function to perform robot actions."
            )
            time.sleep(30)

    def reset_task(self):
        pass

    def startup(self):
        pass

    def queue_orders(
        self,
        order_list: List[Tuple[OrderType, List[float]]],
        time_between_orders: float,
        output_dir: str = "",
        reverse_xy: bool = False,
    ):
        """
        Queue a list of orders for the robot to execute sequentially and save state after each order.

        :param robot: The robot to execute the orders
        :param order_list: A list of tuples containing OrderType and the associated values
        :param time_between_orders: Time to wait between executing orders
        :param output_dir: Directory to save state data
        :param start_time: The start time of the autograsper process
        """
        for order in order_list:
            execute_order(self.robot, order, output_dir, reverse_xy)
            time.sleep(time_between_orders)
            if self.record_only_after_action and (
                self.state is RobotActivity.ACTIVE
                or self.state is RobotActivity.RESETTING
            ):
                self.record_current_state()

    def manual_control(self, step_size=0.1, state = None, time_between_orders=None):
        """
        Manually control the robot using keyboard inputs.
        """
        from pynput import keyboard

        if self.robot_state is None:
            self.robot_state, _ = self.robot.get_state()
        if time_between_orders is None:
            time_between_orders = self.time_between_orders

        current_x = self.robot_state["x_norm"]
        current_y = self.robot_state["y_norm"]
        current_z = self.robot_state["z_norm"]
        current_rotation = self.robot_state["rotation"]
        current_angle = self.robot_state["claw_norm"]

        def on_press(key):
            nonlocal current_x, current_y, current_z, current_rotation, current_angle
            try:

                # == XY axis ==
                if key.char == "w":
                    current_y += step_size
                    current_y = min(max(current_y, 0), 1)
                    self.queue_orders([(OrderType.MOVE_XY, [current_x, current_y])], time_between_orders)
                elif key.char == "a":
                    current_x -= step_size
                    current_x = min(max(current_x, 0), 1)
                    self.queue_orders([(OrderType.MOVE_XY, [current_x, current_y])], time_between_orders)
                elif key.char == "s":
                    current_y -= step_size
                    current_y = min(max(current_y, 0), 1)
                    self.queue_orders([(OrderType.MOVE_XY, [current_x, current_y])], time_between_orders)
                elif key.char == "d":
                    current_x += step_size
                    current_x = min(max(current_x, 0), 1)
                    self.queue_orders([(OrderType.MOVE_XY, [current_x, current_y])], time_between_orders)

                # == Z axis ==
                elif key.char == "r":
                    current_z += step_size
                    current_z = min(max(current_z, 0), 1)
                    print(current_z)
                    self.queue_orders([(OrderType.MOVE_Z, [current_z])], time_between_orders)
                elif key.char == "f":
                    current_z -= step_size
                    current_z = min(max(current_z, 0), 1)
                    print(current_z)
                    self.queue_orders([(OrderType.MOVE_Z, [current_z])], time_between_orders)

                # == Gripper open ==
                elif key.char == "i":
                    current_angle += step_size / 100
                    current_angle = min(current_angle, 1)
                    print(current_angle)
                    self.queue_orders([(OrderType.GRIPPER_CLOSE, [current_angle])], time_between_orders)
                # == Gripper open small steps==
                elif key.char == "o":
                    current_angle += step_size / 200
                    current_angle = min(current_angle, 1)
                    print(current_angle)
                    self.queue_orders([(OrderType.GRIPPER_CLOSE, [current_angle])], time_between_orders)
                # == Gripper close ==
                elif key.char == "k":
                    current_angle -= step_size / 100
                    current_angle = max(current_angle, 0.2)
                    print(current_angle)
                    self.queue_orders([(OrderType.GRIPPER_CLOSE, [current_angle])], time_between_orders)
                # == Gripper close small steps==
                elif key.char == "l":
                    current_angle -= step_size / 200
                    current_angle = max(current_angle, 0.2)
                    print(current_angle)
                    self.queue_orders([(OrderType.GRIPPER_CLOSE, [current_angle])], time_between_orders)

                # == Rotate ==
                elif key.char == "z":
                    current_rotation -= int(step_size * 100)
                    current_rotation = np.clip(current_rotation, 0, 360)
                    print(current_rotation)
                    self.queue_orders([(OrderType.ROTATE, [current_rotation])], time_between_orders)
                elif key.char == "x":
                    print(current_rotation)
                    current_rotation += int(step_size * 100)
                    current_rotation = np.clip(current_rotation, 0, 360)
                    self.queue_orders([(OrderType.ROTATE, [current_rotation])], time_between_orders)

                # == Quit ==
                elif key.char == "q":
                    return False

            except Exception as e:
                print(e)
                print(
                    "Make sure that the runtime has access to an X server. If running in a container on Wayland, you might need to perform `xhost local:root` in the host terminal."
                )

        def on_release(key):
            if key == keyboard.Key.esc:
                # Stop listener
                return False

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()