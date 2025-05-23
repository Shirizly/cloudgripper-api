from grasper import AutograsperBase, RobotActivity
from library.utils import run_calibration

class CalibrateGrasper(AutograsperBase):
    def __init__(self, config):
        super().__init__(config)

    def perform_task(self):

        run_calibration(0.2, self.robot)

        self.state = RobotActivity.FINISHED  # stop data recording

    def reset_task(self):
        # replace with your own resetting if needed
        return super().reset_task()
