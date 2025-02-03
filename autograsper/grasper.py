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
from library.utils import OrderType, queue_orders, clear_center

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
        config,
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

        # fudge time to ensure frames at the start/finish of task/resetting
        self.task_time_margin = 2

        try:
            camera_cfg = config["camera"]
            self.camera_matrix = np.array(ast.literal_eval(camera_cfg["m"]))
            self.distortion_coeffs = np.array(ast.literal_eval(camera_cfg["d"]))

            experiment_cfg = config["experiment"]
            self.robot_idx = ast.literal_eval(experiment_cfg["robot_idx"])
            self.default_action_delay = ast.literal_eval(experiment_cfg["default_action_delay"])
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

    def queue_robot_orders(
        self, orders: List[Tuple[OrderType, List]], delay = None 
    ):
        if not delay:
            delay = self.default_action_delay

        queue_orders(self.robot, orders, delay, output_dir=self.output_dir)


    def wait_for_start_signal(self):
        # TODO this solution can probably be improved
        while not self.start_flag:
            time.sleep(0.05)


    def run_grasping(self):
        while self.state != RobotActivity.FINISHED:
            self.go_to_start()
            self.state = RobotActivity.ACTIVE

            self.wait_for_start_signal()
            self.start_flag = False

            try:
                self.perform_task()
            except Exception as e:
                print(f"Unexpected error during perform_task: {e}")
                self.failed = True
                raise

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

    @abstractmethod
    def recover_after_fail(self):
        pass

    @abstractmethod
    def perform_task(self):
        pass

    @abstractmethod
    def reset_task(self):
        pass

    @abstractmethod
    def startup(self):
        pass

    @abstractmethod
    def go_to_start(self):
        pass