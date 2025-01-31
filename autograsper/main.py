import logging
import argparse
import sys

from robot_controller import RobotController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Robot Controller")
    parser.add_argument("--robot_idx", type=str, required=True, help="Robot index")
    parser.add_argument(
        "--config",
        type=str,
        default="autograsper/config.ini",
        help="Path to the configuration file",
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    controller = RobotController(args)
    controller.run()

if __name__ == "__main__":
    sys.exit(main())
