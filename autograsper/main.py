import logging
import sys

from coordinator import DataCollectionCoordinator
from custom_graspers.example_grasper import ExampleGrasper
from custom_graspers.manual_grasper import ManualGrasper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config_file = "autograsper/config.ini"
    exampleGrasper = ManualGrasper(config_file)
    coordinator = DataCollectionCoordinator(config_file, exampleGrasper)
    coordinator.run()


if __name__ == "__main__":
    sys.exit(main())
