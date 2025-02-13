import threading
import cv2
import logging
import time

from coordinator import DataCollectionCoordinator
from utils import load_config
from custom_graspers.example_grasper import ExampleGrasper

# Configure centralized logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def main():
    # Load configuration as a dict.
    config = load_config('autograsper/config.yaml')
    shutdown_event = threading.Event()

    # Dependency injection: pass shutdown_event to all components.
    grasper = ExampleGrasper(config, shutdown_event=shutdown_event)
    coordinator = DataCollectionCoordinator(config, grasper, shutdown_event)

    # Start the background coordinator tasks.
    coordinator.start()

    try:
        # Run UI loop in the main thread.
        while not shutdown_event.is_set():
            ui_msg = coordinator.get_ui_update(timeout=0.1)
            if ui_msg and ui_msg["type"] == "image_update":
                bottom_img = ui_msg["image"]
                cv2.imshow("Bottom Image", bottom_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    shutdown_event.set()
            else:
                # No image update; sleep briefly.
                time.sleep(0.01)
    except Exception as e:
        logging.error("Error in UI loop: %s", e)
    finally:
        # Signal shutdown and wait for background tasks to finish.
        shutdown_event.set()
        coordinator.join()
        cv2.destroyAllWindows()
        logging.info("Application shutdown complete.")

if __name__ == "__main__":
    main()
