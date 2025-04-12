from grasper import AutograsperBase
from library.utils import OrderType
from library.rgb_object_tracker import get_object_pos
import os
import json
import time
import numpy as np
from typing import List, Tuple
import math


class EvalGrasper(AutograsperBase):
    def __init__(self, config, shutdown_event):
        super().__init__(config, shutdown_event=shutdown_event)

        self.policy_path = os.path.join(
            os.getcwd(), "eval_bash/Staged_learning/stage_1_100.sh"
        )

        if os.path.isfile(self.policy_path):
            print("found")
        else:
            print("not found")

    def perform_policy_action(self):
        # Read the JSON to retrieve the proposed actions
        action_path = "/workspaces/cloudgripper-api/proposed_action.json"
        if not os.path.exists(action_path):
            print(
                f"No proposed action file found at {action_path}. No actions to perform."
            )
            return

        with open(action_path, "r") as f:
            data = json.load(f)

        x = data.get("x", None)
        y = data.get("y", None)
        z = data.get("z", None)

        if z is not None and z > 0.3 and z < 0.6:
            z = 0.55

        grip = data.get("grip", None)

        orders = []

        if z is not None:
            orders.append((OrderType.MOVE_Z, [z]))

        if x is not None and y is not None:
            orders.append((OrderType.MOVE_XY, [x, y]))

        if grip is not None:
            if grip < 0.5:
                # orders.append((OrderType.GRIPPER_CLOSE, [0]))
                print("gripper close")
            else:
                orders.append((OrderType.GRIPPER_OPEN, []))

        # Execute the queued orders with a delay
        self.queue_orders(orders, time_between_orders=1, record=False)
        self.record_current_state()

    def center_sweep(self):
        orders = [
            (OrderType.GRIPPER_CLOSE, [0]),
            (OrderType.MOVE_Z, [0]),
            (OrderType.MOVE_XY, [0.5, 0.0]),
            (OrderType.MOVE_XY, [0.5, 0.4]),
            (OrderType.MOVE_XY, [0.5, 0.5]),
            (OrderType.MOVE_XY, [0.5, 0.6]),
            (OrderType.MOVE_XY, [0.4, 0.6]),
            (OrderType.MOVE_XY, [0.4, 0.4]),
            (OrderType.MOVE_XY, [0.6, 0.4]),
        ]
        self.queue_robot_orders(orders, delay=self.time_between_orders)

    def get_color_pos(self, color):
        return get_object_pos(self.bottom_image, self.robot_idx, color)

    def check_grasping_success(self):
        state = self.robot_state

        gripper_pos = [state["x_norm"], state["y_norm"]]

        object_position = self.get_color_pos("green")

        if object_position is None:
            return False

        gripper_is_close_enough = (
            np.linalg.norm(np.array(gripper_pos) - np.array(object_position)) < 0.10
        )

        return gripper_is_close_enough

    def check_stacking_success(self):
        if self.get_color_pos("green") is None:
            print("stacking appears successful")
            return True
        print("stacking appears failed")
        return False

    def evaluate_policy(self, n_actions):
        for _ in range(n_actions):
            self.call_real_eval()

            time.sleep(5)

            self.perform_policy_action()

            time.sleep(3)

            if self.check_stacking_success():
                print("look stacked")

    def subtask1(self):
        n_actions = 3
        self.evaluate_policy(n_actions)

        if self.check_grasping_success():
            self.failed = False
        else:
            self.failed = True

        return self.failed is True

    def subtask2(self):
        self.evaluate_policy(10)
        if self.check_stacking_success():
            self.failed = False
        else:
            self.failed = True

    def perform_task(self):
        time.sleep(2)

        if self.subtask1():
            print("subtask 2")
            self.subtask2()

        print("task complete")

    def reset_task(self):
        self._reset_target_block()
        return

    def startup(self):
        position = [0, 0.7]
        orders = [
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, position),
            (OrderType.GRIPPER_CLOSE, [0]),
        ]
        self.queue_orders(orders, time_between_orders=1, record=False)

        self.robot.gripper_close()

    def _reset_target_block(self):
        block_pos = self.generate_new_block_position()

        target_color = "green"
        object_position = get_object_pos(
            self.bottom_image, self.robot_idx, target_color
        )
        self.pickup_and_place_object(
            object_position,
            0,
            0,
            target_position=block_pos,
        )

        self.make_sure_red_box_is_center()

    def make_sure_red_box_is_center(self):
        margin = 0.15
        red_position = get_object_pos(
            self.bottom_image, self.robot_idx, "red", debug=False
        )

        if abs(red_position[0] - 0.5) > margin and abs(red_position[1] - 0.5) > margin:
            print("red moved, moving back to center")
            self.move_red_to_center()
        else:
            print("red at correct position")

    def move_red_to_center(self):
        target_color = "red"
        object_position = get_object_pos(
            self.bottom_image, self.robot_idx, target_color
        )
        self.pickup_and_place_object(object_position, 0, 0, target_position=[0.5, 0.5])

    def generate_new_block_position(self):
        margin = 0.20
        while True:
            x = np.linspace(0.2, 0.8, 0.05)
            y = np.linspace(0.2, 0.8, 0.05)
            avoid_position = [0.5, 0.5]
            dist = np.sqrt(
                math.pow(x - avoid_position[0], 2) + math.pow(y - avoid_position[1], 2)
            )

            if dist > margin:
                break
        return [x, y]

    def pickup_and_place_object(
        self,
        object_position: Tuple[float, float],
        object_height: float,
        target_height: float,
        target_position: List[float] = [0.5, 0.5],
    ):
        orders = [
            (OrderType.GRIPPER_OPEN, []),
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, object_position),
            (OrderType.MOVE_Z, [object_height]),
            (OrderType.GRIPPER_CLOSE, []),
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, target_position),
            (OrderType.MOVE_Z, [target_height]),
            (OrderType.GRIPPER_OPEN, []),
        ]
        self.queue_orders(orders, time_between_orders=self.time_between_orders)

    def call_real_eval(self):
        import subprocess

        # Define the arguments from the bash command
        args = [
            "bash",
            self.policy_path,
        ]

        try:
            _ = subprocess.run(args, check=True)
            print("Policy action calculated.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
