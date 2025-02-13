from grasper import AutograsperBase, RobotActivity
from library.utils import OrderType
import time


class ExampleGrasper(AutograsperBase):
    def __init__(self, config):
        super().__init__(config)

    def perform_task(self):
        self.queue_orders(
            [
                (OrderType.MOVE_XY, [0.9, 0.9]),
                (OrderType.ROTATE, [30]),
                (OrderType.MOVE_Z, [0.7]),
                (OrderType.GRIPPER_OPEN, []),
            ],

            time_between_orders=self.time_between_orders,  # set in config.ini file
        )

        self.state = RobotActivity.FINISHED  # stop data recording

    def startup(self):
        # This method will execute at the beginning of every experiment.
        # During this phase, data will not be recorded or shown in real time.

        print("performing startup tasks...")

        self.queue_orders(
            [
                (OrderType.MOVE_XY, [0.1, 0.1]),
                (OrderType.ROTATE, [0]),
                (OrderType.MOVE_Z, [1]),
                (OrderType.GRIPPER_CLOSE, []),
            ],
            time_between_orders=self.time_between_orders,  # set in config.ini file
        )

    def reset_task(self):
        self.queue_orders(
            [
                (OrderType.MOVE_XY, [0.1, 0.1]),
                (OrderType.ROTATE, [0]),
                (OrderType.MOVE_Z, [1]),
                (OrderType.GRIPPER_CLOSE, []),
            ],

            time_between_orders=self.time_between_orders,  # set in config.ini file
        )


