"""`MJPEGServer` — Flask/werkzeug debug MJPEG stream reading the bottom camera from
`ObservationSource` (Wave 5).

Design reference: `autograsper/design/02_proposed_architecture.md` §2 (`ui/stream.py # MJPEG
server, reads from ObservationSource`) and §6 ("What deliberately does not change ... Flask MJPEG
debugging UI").

Porting notes (copied and adapted, not imported): `main_chickpeas.py`'s Flask app (`/video_feed`
route, `generate_frames` reading `coordinator.get_ui_update()`) + `coordinator.py`'s `ui_queue`
handling. The hand-rolled `Queue(maxsize=2)` drop-oldest queue is replaced by
`source.subscribe('ui', maxsize=2)` (the same `LatestWinsQueue` primitive every other subscriber
uses — `observation.source`'s doc explicitly lists the UI stream as an intended consumer).

Flask/werkzeug are imported **lazily** (inside `start()`/`_build_app()`), never at module import
time — this module must be importable even in an environment without them installed (as of this
wave's implementation, `flask`/`werkzeug` are in fact NOT installed in the `cge` conda env despite
`docs/CONVENTIONS.md` listing `flask` as present; logged in `design/IMPLEMENTATION_LOG.md`). No
test in this repo starts an `MJPEGServer` for this reason; `main_granular.py --no-ui` (or
`ui.enabled: false` in config) skips constructing one entirely.

Threading: `start()` spawns one background thread running werkzeug's `serve_forever()`; `stop()`
signals it via the server's own `shutdown()` and joins. The frame generator itself runs on
werkzeug's own request-handling thread (one per open connection), each with its own
`source.subscribe('ui', ...)` queue — safe, since `ObservationSource.subscribe` is documented safe
to call from any thread.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import cv2

logger = logging.getLogger(__name__)

_SUBSCRIBER_NAME = "ui"


class MJPEGServer:
    """Thread running a werkzeug server, streaming `source`'s bottom images as MJPEG at
    `http://0.0.0.0:<port>/video_feed`."""

    def __init__(self, source, port: int, shutdown_event: threading.Event) -> None:
        self._source = source
        self._port = port
        self._shutdown_event = shutdown_event
        self._server = None
        self._thread: Optional[threading.Thread] = None

    def _generate_frames(self):
        queue = self._source.subscribe(_SUBSCRIBER_NAME, maxsize=2)
        try:
            while not self._shutdown_event.is_set():
                obs = queue.get(timeout=0.5)
                if obs is None:
                    continue
                ok, buf = cv2.imencode(".jpg", obs.bottom_image)
                if not ok:
                    continue
                frame = buf.tobytes()
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        finally:
            self._source.unsubscribe(_SUBSCRIBER_NAME)

    def _build_app(self):
        from flask import Flask, Response

        app = Flask(__name__)

        @app.route("/video_feed")
        def video_feed():
            return Response(
                self._generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
            )

        return app

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("MJPEGServer already started")
        from werkzeug.serving import make_server

        app = self._build_app()
        self._server = make_server("0.0.0.0", self._port, app)
        self._thread = threading.Thread(target=self._server.serve_forever, name="MJPEGServer", daemon=True)
        self._thread.start()
        logger.info("MJPEGServer: serving on port %d", self._port)

    def stop(self, timeout: Optional[float] = None) -> None:
        if self._server is not None:
            self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._server = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
