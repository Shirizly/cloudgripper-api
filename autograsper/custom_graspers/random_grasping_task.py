from grasper import AutograsperBase, RobotActivity
from library.rgb_object_tracker import (
    get_object_pos,
    all_objects_are_visible,
    test_calibration,
)
from library.utils import (
    OrderType,
    pick_random_positions,
    get_undistorted_bottom_image,
    clear_center,
    manual_control,
    run_calibration,
)
import numpy as np
from typing import List, Tuple
import time
import random
import ast


class RandomGrasper(AutograsperBase):
    def __init__(
        self,
        config,
        output_dir: str = "",
    ):
        super().__init__(config, output_dir)
        # Task-specific initialization
        experiment_cfg = config["experiment"]

        self.colors = ast.literal_eval(experiment_cfg["colors"])
        self.block_heights = np.array(ast.literal_eval(experiment_cfg["block_heights"]))
        self.position_bank = ast.literal_eval(experiment_cfg["position_bank"])
        self.stack_position = ast.literal_eval(experiment_cfg["stack_position"])
        self.object_size = config.getfloat("experiment", "object_size")

        self.time_between_orders = 1

    def startup(self):
        position = [0, 0.7]
        orders = [
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, position),
            (OrderType.GRIPPER_CLOSE, []),
        ]
        self.queue_robot_orders(orders, 1)

    def prepare_experiment(self, position_bank, stack_position):
        if position_bank is None:
            position_bank = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8]]
        position_bank = np.array(position_bank)

        if stack_position is None:
            stack_position = [0.5, 0.5]

        return position_bank, stack_position

    def perform_task(self):
        random_position = self.generate_new_block_position()

        self.queue_robot_orders(
            [
                (OrderType.MOVE_XY, random_position),
                (OrderType.GRIPPER_OPEN, []),
                (OrderType.MOVE_Z, [0]),
                (OrderType.GRIPPER_CLOSE, []),
            ],
            delay=2.0,
        )

        object_position = get_object_pos(
            self.bottom_image, self.robot_idx, "green", debug=True
        )
        if np.linalg.norm(np.array(random_position) - np.array(object_position)) < 0.12:
            self.failed = False
            print("succesful grasp")
        else:
            print("failed grasp")
            self.failed = True

        print("task complete")

    def generate_new_block_position(self):
        import random

        margin = 0.25
        while True:
            x = random.uniform(0.2, 0.8)
            y = random.uniform(0.2, 0.8)
            avoid_position = [0.5, 0.5]
            dist = np.sqrt(
                np.pow(x - avoid_position[0], 2) + np.pow(y - avoid_position[1], 2)
            )

            if dist > margin:
                break

        return [x, y]

    def recover_after_fail(self):
        self.robot.gripper_open()
        time.sleep(0.5)
        self.startup()

    def reset_target_block(self):
        block_pos = self.generate_new_block_position()

        target_color = "green"
        object_position = get_object_pos(
            self.bottom_image, self.robot_idx, target_color
        )
        self.pickup_and_place_object(
            0,
            0,
            object_position=object_position,
            target_position=block_pos,
        )

        margin = 0.1
        red_position = get_object_pos(
            self.bottom_image, self.robot_idx, "red", debug=False
        )

        if abs(red_position[0] - 0.5) > margin and abs(red_position[1] - 0.5) > margin:
            self.move_red_to_center()

    def reset_task(self):
        self._reset_target_block()
        self.startup()

        return

    def pickup_and_place_object(
        self,
        object_height: float,
        target_height: float,
        object_position: Tuple[float, float] = None,
        object_color: str = None,
        target_position: List[float] = [0.5, 0.5],
        time_between_orders: float = 2.0,
    ):
        if not object_position and not object_color:
            print("Pick and place: object position and color required")
            return

        if not object_position:
            object_position = get_object_pos(
                self.bottom_image, self.robot_idx, object_color
            )

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
        self.queue_robot_orders(orders, time_between_orders)

    def stack_objects(self, colors, block_heights, stack_position):
        blocks = list(zip(colors, block_heights))
        bottom_color = colors[0]

        stack_height = 0

        for color, block_height in blocks:
            try:
                bottom_block_position = get_object_pos(
                    self.bottom_image, self.robot_idx, bottom_color
                )
                object_position = get_object_pos(
                    self.bottom_image, self.robot_idx, color, debug=False
                )
            except ValueError as e:
                print(f"Error finding object position for color '{color}': {e}")
                self.failed = True
                return  # Exit the function if an object is not found

            target_pos = (
                bottom_block_position if color != bottom_color else stack_position
            )

            self.pickup_and_place_object(
                max(block_height - 0.20, 0.02),
                stack_height,
                object_position=object_position,
                target_position=target_pos,
            )

            stack_height += block_height

    def reset_blocks(
        self,
        block_positions: List[List[float]],
        block_heights: np.ndarray,
        stack_position: List[float] = [0.5, 0.5],
        time_between_orders: float = 1.5,
    ):
        rev_heights = np.flip(block_heights.copy())
        target_z = sum(rev_heights)

        for index, block_pos in enumerate(block_positions):
            target_z -= rev_heights[index]
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

            self.queue_robot_orders(orders, time_between_orders)
