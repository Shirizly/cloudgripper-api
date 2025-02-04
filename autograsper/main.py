import logging
import sys

from coordinator import DataCollectionCoordinator
from custom_graspers.example_grasper import ExampleGrasper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config_file = "autograsper/config.ini"
    exampleGrasper = ExampleGrasper(config_file)
    coordinator = DataCollectionCoordinator(config_file, exampleGrasper)
    coordinator.run()


if __name__ == "__main__":
    sys.exit(main())
