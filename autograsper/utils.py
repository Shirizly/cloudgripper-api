import os
import logging
import traceback
import threading
from typing import Optional

logger = logging.getLogger(__name__)
ERROR_EVENT = threading.Event()


def handle_error(exception: Exception) -> None:
    """Logs exception info and sets ERROR_EVENT."""
    logger.error(f"Error occurred: {exception}")
    logger.error(traceback.format_exc())
    ERROR_EVENT.set()


def get_new_session_id(base_dir: str) -> int:
    """Returns a new numeric session ID based on existing directories."""
    if not os.path.exists(base_dir):
        return 1
    session_ids = [
        int(dir_name)
        for dir_name in os.listdir(base_dir) if dir_name.isdigit()
    ]
    return max(session_ids, default=0) + 1


def start_thread(target, *args, **kwargs) -> threading.Thread:
    """Creates and starts a new thread, returning the Thread object."""
    thread = threading.Thread(target=target, args=args, kwargs=kwargs)
    thread.start()
    return thread
