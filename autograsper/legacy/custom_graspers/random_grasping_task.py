from grasper import AutograsperBase
from library.utils import OrderType
from library.rgb_object_tracker import get_object_pos
import os
import json
import time
import numpy as np
from typing import List, Tuple
import math


class RandomGrasper(AutograsperBase):
    def __init__(self, config, shutdown_event):
        super().__init__(config, shutdown_event=shutdown_event)

        self.policy_path = os.path.join(
            os.getcwd(), "eval_bash/Staged_learning/stage_1_100.sh"
        )

        if os.path.isfile(self.policy_path):
            print("found")
        else:
            print("not found")

    def sweep(self):
        print("sweeping")
        orders = [
            (OrderType.MOVE_Z, [1]),
            (OrderType.GRIPPER_CLOSE, [0]),
            (OrderType.MOVE_XY, self.get_color_pos("red") + [0.0, 0.1]),
            (OrderType.MOVE_Z, [0]),
            (OrderType.MOVE_XY, self.get_color_pos("red") + [0.0, -0.1]),
        ]
        self.queue_orders(orders)

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

    def random_grasp(self):
        import numpy as np

        random_coordinates = [np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)]

        orders = [
            (OrderType.MOVE_XY, random_coordinates),
            (OrderType.GRIPPER_OPEN, [0]),
            (OrderType.MOVE_Z, [0]),
            (OrderType.GRIPPER_CLOSE, [0]),
        ]
        self.queue_orders(orders)

    def get_color_pos(self, color):
        import numpy as np

        return np.array(get_object_pos(self.bottom_image, self.robot_idx, color))

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

    def perform_task(self):
        time.sleep(2)

        self.random_grasp()

        if self.check_grasping_success():
            self.failed = False
        else:
            self.failed = True

        return

    def reset_task(self):
        if math.dist(self.get_color_pos("red"), self.get_color_pos("green")) < 0.2:
            self.sweep()
        self.make_sure_red_box_is_center()

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

        self.pickup_and_place_object(
            self.get_color_pos("green"),
            0,
            0,
            target_position=block_pos,
        )
        if math.dist(self.get_color_pos("red"), self.get_color_pos("green")) < 0.2:
            self.sweep()
        self.make_sure_red_box_is_center()

    def recover_after_fail(self):
        self._reset_target_block()
        self.make_sure_red_box_is_center()

    def make_sure_red_box_is_center(self):
        margin = 0.10

        if math.dist(self.get_color_pos("red"), [0.5, 0.5]) > margin:
            print("red moved, moving back to center")
            self.move_red_to_center()
        else:
            print("red at correct position")

    def move_red_to_center(self):
        self.pickup_and_place_object(
            self.get_color_pos("red"), 0, 0, target_position=[0.5, 0.5]
        )

    def generate_new_block_position(self):
        margin = 0.20
        while True:
            x = np.random.uniform(0.25, 0.75)
            y = np.random.uniform(0.25, 0.75)
            avoid_position = [0.5, 0.5]
            dist = math.dist([x, y], avoid_position)

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
        self.queue_orders(orders)
