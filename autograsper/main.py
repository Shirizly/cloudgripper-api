import logging
import sys
import threading

from coordinator import DataCollectionCoordinator
from custom_graspers.example_grasper import ExampleGrasper
from utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config = load_config("autograsper/config.yaml")
    shutdown_event = threading.Event()
    exampleGrasper = ExampleGrasper(config, shutdown_event)
    coordinator = DataCollectionCoordinator(config, exampleGrasper, shutdown_event)
    coordinator.run()


if __name__ == "__main__":
    sys.exit(main())
