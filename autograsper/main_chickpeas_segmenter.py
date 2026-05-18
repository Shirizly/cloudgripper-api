import sys
import threading
import time
from attr import dataclass
import cv2
import logging

import os

from flask import Flask, Response
from werkzeug.serving import make_server

from coordinator import DataCollectionCoordinator
from autograsper.custom_graspers.granular_pusher import GranularPusher
from autograsper.custom_graspers.segmenting_granular_pusher import SegGranularPusher
from utils import load_config
from image_collector.chickpea_segmenter import ChickpeaSegmenter

import traceback

def thread_exception_handler(args):
    print("\n Unhandled thread exception")
    print(f"Thread: {args.thread.name}")
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
    sys.exit(1)   # hard crash so you SEE it



# Configure logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Create the Flask application.
app = Flask(__name__)

# Global reference to the coordinator (set in main()).
global_coordinator = None


@app.route("/video_feed")
def video_feed():
    """
    Route that streams the video feed as an MJPEG stream.
    """
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def generate_frames():
    """
    Generator function that continuously retrieves image frames from the
    coordinator's UI update queue, encodes them as JPEG, and yields them.
    """
    assert global_coordinator is not None, "Global coordinator must be initialized before starting the Flask app."
    while not global_coordinator.shutdown_event.is_set():
        ui_msg = global_coordinator.get_ui_update(timeout=0.1)
        if ui_msg and ui_msg.get("type") == "image_update":
            frame = ui_msg.get("image")
            if frame is not None:
                # Encode frame as JPEG.
                ret, jpeg = cv2.imencode(".jpg", frame)
                if ret:
                    frame_bytes = jpeg.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n"
                    )
        else:
            # If no frame is available, sleep briefly.
            time.sleep(0.02)




def main():
    global global_coordinator
    threading.excepthook = thread_exception_handler
    config_path = os.path.join(os.getcwd(), "autograsper", "chickpea-config.yaml")
    # config_path = os.path.join(os.getcwd(), "backgammon-config.yaml")
    config = load_config(config_path)
    shutdown_event = threading.Event()
    
    # Initialize chickpea segmenter
    segmenter_weights_path = os.path.join(os.getcwd(), "image_collector", "chickpeas_segmentation_best.pt")
    logging.info(f"Loading chickpea segmenter from {segmenter_weights_path}")
    segmenter = ChickpeaSegmenter(
        weights_path=segmenter_weights_path,
        conf_threshold=0.25,
        iou_threshold=0.5
    )
    logging.info("Chickpea segmenter loaded successfully")

    grasper = SegGranularPusher(config, shutdown_event=shutdown_event, N_pushes=10)
    # time.sleep(100)
    global_coordinator = DataCollectionCoordinator(config, grasper, shutdown_event, visualize=True, segmenter=segmenter)
    global_coordinator.start()

    # Create Werkzeug server instead of using app.run()
    server = make_server("0.0.0.0", 3000, app, threaded=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=False)
    server_thread.start()
    logging.info("Flask server started on http://0.0.0.0:3000")

    try:
        # Monitor shutdown_event and gracefully shutdown when it's set
        while not shutdown_event.is_set():
            shutdown_event.wait(timeout=0.5)
    except Exception as e:
        logging.error("Error in main thread: %s", e)
    finally:
        logging.info("Initiating graceful shutdown...")
        shutdown_event.set()
        
        # Shutdown the Flask server
        server.shutdown()
        server_thread.join(timeout=5.0)
        
        # Wait for coordinator to shutdown
        global_coordinator.join()
        
        logging.info("Application shutdown complete.")
        sys.exit(0)

if __name__ == "__main__":
    main()
