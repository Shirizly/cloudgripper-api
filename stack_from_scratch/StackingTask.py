# stackingtask.py

from task import Task
from typing import List, Tuple, Any, TYPE_CHECKING
import numpy as np

from library.rgb_object_tracker import all_objects_are_visible, get_object_pos
from library.utils import pick_random_positions

if TYPE_CHECKING:
    from autograsper import Autograsper  # For type hinting without circular imports


class StackingTask(Task):
    def __init__(self, colors: List[str], block_heights: np.ndarray, config: dict):
        self.colors = colors
        self.block_heights = block_heights
        self.config = config

    async def execute(self, autograsper: 'Autograsper') -> None:
        # Implement the stacking logic here
        await self.stack_objects(autograsper)

    async def detect_errors(self, autograsper: 'Autograsper') -> bool:
        # Implement error detection logic
        autograsper.update_bottom_image()
        if not all_objects_are_visible(self.colors, autograsper.bottom_image):
            return True
        return False

    async def recover_from_errors(self, autograsper: 'Autograsper') -> None:
        # Implement recovery logic
        await self.recover_after_fail(autograsper)

    async def reset_scene(self, autograsper: 'Autograsper') -> None:
        # Implement scene resetting logic
        position_bank, stack_position = autograsper.prepare_experiment(
            self.config)
        random_reset_positions = pick_random_positions(
            position_bank, len(self.block_heights), object_size=2
        )
        await autograsper.reset(
            random_reset_positions,
            self.block_heights,
            stack_position=stack_position,
        )
        await autograsper.go_to_start()

    async def stack_objects(self, autograsper: 'Autograsper') -> None:
        autograsper.update_bottom_image()  # Ensure we have the latest image

        blocks = list(zip(self.colors, self.block_heights))
        bottom_color = self.colors[0]

        stack_height = 0

        for color, block_height in blocks:
            bottom_block_position = get_object_pos(
                autograsper.bottom_image, autograsper.robot_idx, bottom_color
            )
            object_position = get_object_pos(
                autograsper.bottom_image, autograsper.robot_idx, color, debug=True
            )

            target_pos = (
                bottom_block_position if color != bottom_color else autograsper.DEFAULT_STACK_POSITION
            )

            grasp_height = max(
                block_height - autograsper.GRIPPER_OFFSET, autograsper.MINIMUM_GRASP_HEIGHT
            )

            await autograsper.pickup_and_place_object(
                object_position,
                grasp_height,
                stack_height,
                target_position=target_pos,
            )

            stack_height += block_height

    async def recover_after_fail(self, autograsper: 'Autograsper') -> None:
        """
        Handles recovery after a failed experiment asynchronously.
        """
        # Implement recovery logic here
        # For example, move robot to a safe position
        await autograsper.go_to_start()
