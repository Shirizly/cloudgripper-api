from grasper import AutograsperBase, RobotActivity
from library.utils import manual_control


class ManualGrasper(AutograsperBase):
    def __init__(self, config):
        super().__init__(config)

    def perform_task(self):

        manual_control(self.robot)

        self.state = RobotActivity.FINISHED  # stop data recording

    def reset_task(self):
        # replace with your own resetting if needed
        return super().reset_task()
