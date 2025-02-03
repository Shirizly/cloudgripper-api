from grasper import AutograsperBase, RobotActivity
from library.utils import OrderType
import time


class ExampleGrasper(AutograsperBase):
    def __init__(self, config, output_dir: str = ""):
        super().__init__(config, output_dir)

    def perform_task(self):
        # Method 1: Using queue_orders to send a batch of commands
        self.queue_orders(
            [
                (OrderType.MOVE_XY, [0.5, 0.5]),
                (OrderType.ROTATE, [30]),
                (OrderType.MOVE_Z, [0.7]),
                (OrderType.GRIPPER_OPEN, []),
            ],
            time_between_orders=self.time_between_orders  # set in config.ini file
        )

        # Method 2: Sending individual commands with state recording
        self.robot.move_xy(0.5, 0.5)
        time.sleep(2)
        self.record_current_state()

        self.robot.rotate(30)
        time.sleep(2)
        self.record_current_state()

        self.robot.move_z(0.7)
        time.sleep(2)
        self.record_current_state()

        self.robot.gripper_open()
        time.sleep(2)
        self.record_current_state()

        # comment or remove if you want multiple experiments to run
        self.state = RobotActivity.FINISHED  # stop data recording

    def startup(self):
        # This method will execute at the beginning of every experiment.
        # During this phase, data will not be recorded.

        print("performing startup tasks...")

        self.queue_orders(
            [
                (OrderType.MOVE_XY, [0.1, 0.1]),
                (OrderType.ROTATE, [0]),
                (OrderType.MOVE_Z, [1]),
                (OrderType.GRIPPER_CLOSE, []),
            ],
            time_between_orders=self.time_between_orders  # set in config.ini file
        )

    def reset_task(self):
        # replace with your own resetting if needed
        return super().reset_task()
