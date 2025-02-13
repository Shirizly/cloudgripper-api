from grasper import AutograsperBase
from library.utils import OrderType
from library.rgb_object_tracker import get_object_pos
import os
import json
import time
import numpy as np
from typing import List, Tuple
import random
import math

class EvalGrasper(AutograsperBase):
    def __init__(self, config):
        super().__init__(config)

    def perform_policy_action(self):
            # Read the JSON to retrieve the proposed actions
            json_file_path = "/workspaces/cloudgripper-api/proposed_action.json"
            if not os.path.exists(json_file_path):
                print(f"No proposed action file found at {json_file_path}. No actions to perform.")
                return

            with open(json_file_path, "r") as f:
                data = json.load(f)

            x = data.get("x", None)
            y = data.get("y", None)
            z = data.get("z", None)
            grip = data.get("grip", None)

            # Build a list of robot orders based on which JSON keys exist
            orders = []
            # Move X/Y if present
            if x is not None and y is not None:
                orders.append((OrderType.MOVE_XY, [x, y]))

            # Move Z if present
            if z is not None:
                orders.append((OrderType.MOVE_Z, [z]))

            # Gripper logic if "grip" is present
            # e.g., if grip < 0.4 => close, if grip > 0.7 => open
            if grip is not None:
                if grip < 0.4:
                    orders.append((OrderType.GRIPPER_CLOSE, [0]))
                elif grip > 0.7:
                    orders.append((OrderType.GRIPPER_OPEN, []))

            # Execute the queued orders with a delay
            self.queue_orders(orders,time_between_orders=1, record=False)
            self.record_current_state()


    def perform_task(self):

        # self.make_sure_red_box_is_center()

        n_actions = 3
        for _ in range(n_actions):
            self.call_real_eval()  

            time.sleep(5)

            self.perform_policy_action()

            time.sleep(2)

        # After actions execute, check state and determine success
        state = self.robot_state
        gripper_pos = [state[0]["x_norm"], state[0]["y_norm"]]

        def check_success():
            state = self.robot_state
            gripper_pos = [state[0]["x_norm"], state[0]["y_norm"]]

            object_position = get_object_pos(
                self.bottom_image, self.robot_idx, "green", debug=True
            )

            gripper_is_close_enough = np.linalg.norm(np.array(gripper_pos) - np.array(object_position)) < 0.12
            gripper_is_closed = state[0]["claw_norm"] < 0.4

            return gripper_is_close_enough and gripper_is_closed

        # perform trivial movement just to add a duplicate action for resnet training purposes
        if check_success():
            print("seems successful, staying")
            self.queue_orders([(OrderType.MOVE_XY, gripper_pos)], time_between_orders=1)
            self.record_current_state()
        else: 
            print("seems unsuccessful, performing one more action")
            self.call_real_eval()
            time.sleep(5)
            self.perform_policy_action()
        time.sleep(5)


        if (
            check_success()
        ):
            self.failed = False
            print("successful grasp")
        else:
            print("failed grasp")
            self.failed = True

        print("task complete")


    def reset_task(self):

        self._reset_target_block()
        self.go_to_start()
        
        return


    def go_to_start(self):
        positions = [[1, 0.7], [0, 0.7]]
        positions = np.array(positions)
        position = positions[1]
        orders = [
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, position),
        ]
        self.queue_orders(orders,time_between_orders=2,record=False)

        self.robot.gripper_close()

    def _reset_target_block(self):

        block_pos = self.generate_new_block_position()

        target_color = "green"
        object_position = get_object_pos(
            self.bottom_image, self.robot_idx, target_color)
        self.pickup_and_place_object(
            object_position,
            0,
            0,
            target_position=block_pos,
        )

        self.make_sure_red_box_is_center()

    def make_sure_red_box_is_center(self):
        margin = 0.05
        red_position = get_object_pos(
            self.bottom_image, self.robot_idx, "red", debug=False)

        if abs(red_position[0] - 0.5) > margin and abs(red_position[1] - 0.5) > margin:
            print("red moved, moving back to center")
            self.move_red_to_center()
        else:
            print("red at correct position")

    def move_red_to_center(self):
 
        target_color = "red"
        object_position = get_object_pos(
            self.bottom_image, self.robot_idx, target_color)
        self.pickup_and_place_object(
            object_position,
            0,
            0,
            target_position=[0.5,0.5]
        )


    def generate_new_block_position(self):

        margin = 0.20
        while True:
            x = random.uniform(0.2, 0.8)
            y = random.uniform(0.2, 0.8)
            avoid_position = [0.5, 0.5]
            dist = np.sqrt( math.pow(x - avoid_position[0], 2) + math.pow(y-avoid_position[1], 2) )

            if dist > margin:
                break

        print("new random pos: ", x, y)
        return [x, y]

    def pickup_and_place_object(
        self,
        object_position: Tuple[float, float],
        object_height: float,
        target_height: float,
        target_position: List[float] = [0.5, 0.5],
        time_between_orders: float = 2.0,
    ):
        orders = [
            (OrderType.GRIPPER_OPEN, []),
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, object_position),
            #(OrderType.GRIPPER_OPEN, []),
            (OrderType.MOVE_Z, [object_height]),
            (OrderType.GRIPPER_CLOSE, []),
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, target_position),
            (OrderType.MOVE_Z, [target_height]),
            (OrderType.GRIPPER_OPEN, []),
        ]
        self.queue_orders(orders, time_between_orders=time_between_orders)

    def call_real_eval(self):
        import subprocess

        # Define the arguments from the bash command
        args = [
            "bash", "eval_bash/Staged_learning/stage_1_50_v2.sh",
        ]

        try:
            _ = subprocess.run(args, check=True)
            print("Policy action calculated.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}") 