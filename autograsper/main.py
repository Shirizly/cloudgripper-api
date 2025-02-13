import logging
import sys

from coordinator import DataCollectionCoordinator
from custom_graspers.example_grasper import ExampleGrasper
from utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config = load_config("autograsper/config.yaml")
    exampleGrasper = ExampleGrasper(config)
    coordinator = DataCollectionCoordinator(config, exampleGrasper)
    coordinator.run()


if __name__ == "__main__":
    sys.exit(main())
