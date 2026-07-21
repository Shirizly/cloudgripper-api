"""`ObservationSource` — the single polling thread + pub/sub for `Observation` snapshots.

Design reference: `autograsper/design/02_proposed_architecture.md` §3.2.

Porting notes (legacy sources copied/adapted, not imported):
- `autograsper/recording.py::Recorder.record()` / `Recorder._update()` — the FPS-paced polling
  loop: `perf_counter()`-based timing, one `robot.get_all_states()` call per cycle, and
  `shutdown_event.wait(max(0, 1/FPS - elapsed - 0.0002))` pacing (the `0.0002` buffer is
  subtracted "to improve chances of hitting target FPS", per the legacy comment, and is kept
  verbatim here). `Recorder._update` is also where the bottom image got undistorted/rectified and
  where `rotation` bias was subtracted from state — the bias step now lives one layer down (Wave 1
  `CloudGripperRobot`, see `hardware.md`), so this module does not touch rotation.
- `autograsper/coordinator.py`'s `ui_queue` (`Queue(maxsize=2)`, drop-oldest-on-full: `put_nowait`,
  on `Full` do `get_nowait()` then `put_nowait()` again) — ported into `LatestWinsQueue._put_latest`
  for every subscriber, generalizing what was one hardcoded UI queue into the `subscribe()`/
  `unsubscribe()` pub/sub design 02 §3.2 calls for (recorder, segmentation worker, and UI stream
  all get their own instance).

`ObservationSource` deliberately imports `autograsper.perception.frames.CameraPipeline`even though
the informal package-layout arrows in design 02 §2 list `perception -> observation` (perception
may import observation) and not the reverse: design 02 §3.2's prose is explicit and specific
("applies the camera pipeline (undistort + homography rectify from `perception.frames`)") and is
treated as the binding requirement over the more general layout diagram, which does not call out
observation/perception the way it explicitly forbids planner -> hardware/execution imports. Logged
in `design/IMPLEMENTATION_LOG.md`.

Threading model:
- One background thread (name `"ObservationSource"`, daemon) owns all reads from `robot` and
  `camera`. No other thread should call `robot.get_all_states()` while a source for that robot is
  running (design 02 §3.2: "one observation pipeline, one robot connection").
- `latest()`, `await_next()`, `subscribe()`/`unsubscribe()`, `set_frame_index_provider()`, and
  `.failed` are safe to call from any thread.
- Publishing: the poll thread builds one `Observation`, stores it as `_latest` and
  `notify_all()`s under `_cond` (this is what `await_next` blocks on), then pushes a reference to
  every subscriber's `LatestWinsQueue` (drop-oldest on full) — a slow subscriber can never block
  the poll loop or any other subscriber, since each has its own independent `queue.Queue`.

Failure policy (design 02 §3.5 — "Separated failure policy ... No `shutdown_event.set()` inside
behaviors"): an exception raised by `robot.get_all_states()` or `camera.process_bottom()` during a
poll cycle is caught, logged, and counted. After `max_consecutive_failures` (default 5) in a row,
`.failed` becomes `True`, the internal `failed_event` is set, and the poll loop **exits its own
thread** (no further polling is attempted) — but `shutdown_event` is never touched by this class.
The session layer (Wave 5+) is the one that decides what a failed source means (retry with a fresh
`ObservationSource`, abort, human intervention, ...); it should watch `.failed`/`failed_event`
alongside its own `shutdown_event`.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Dict, Optional

import numpy as np

from autograsper.observation.types import Observation, RobotState
from autograsper.perception.frames import CameraPipeline

logger = logging.getLogger(__name__)

# Subtracted from the per-cycle sleep duration; ported verbatim from `recording.py::Recorder.record`
# ("subtract small buffer time to improve chances of hitting target FPS").
_PACING_BUFFER = 0.0002


class LatestWinsQueue:
    """Bounded observation queue with drop-oldest-on-full semantics.

    Not meant to be constructed directly — returned by `ObservationSource.subscribe()`. Ports
    legacy `coordinator.py`'s `ui_queue` handling (see module docstring) into a reusable,
    per-subscriber primitive.
    """

    def __init__(self, maxsize: int = 2):
        self._q: "queue.Queue[Observation]" = queue.Queue(maxsize=maxsize)

    def _put_latest(self, obs: Observation) -> None:
        try:
            self._q.put_nowait(obs)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(obs)
            except queue.Full:
                # Lost a race with another producer/consumer; drop this frame rather than block
                # the poll loop. Only ObservationSource ever produces into this queue, so this is
                # only reachable under concurrent get()/put() races and is harmless to skip.
                logger.debug("LatestWinsQueue: still full after drop-oldest, dropping observation")

    def get(self, timeout: Optional[float] = None) -> Optional[Observation]:
        """Blocking get with optional timeout. Returns `None` on timeout instead of raising."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self) -> int:
        return self._q.qsize()


class ObservationSource:
    """Owns the single FPS-paced polling loop against `robot`/`camera`, publishing immutable
    `Observation` snapshots to `latest()`/`await_next()`/subscriber queues.

    `robot` is duck-typed against `hardware.robot_interface.RobotInterface` (only
    `get_all_states()` is used here); `camera` is duck-typed against
    `perception.frames.CameraPipeline` (only `process_bottom(raw_bottom)` is used here) — neither
    is `runtime_checkable`, matching Wave 1's convention (see `docs/hardware.md`).
    """

    def __init__(
        self,
        robot,
        camera: CameraPipeline,
        fps: float,
        shutdown_event: threading.Event,
        *,
        max_consecutive_failures: int = 5,
    ):
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps!r}")
        self._robot = robot
        self._camera = camera
        self._fps = fps
        self._shutdown_event = shutdown_event
        self._max_consecutive_failures = max_consecutive_failures

        self._cond = threading.Condition()
        self._latest: Optional[Observation] = None
        self._seq = 0

        self._consecutive_failures = 0
        self._failed = False
        self.failed_event = threading.Event()

        self._subscribers_lock = threading.Lock()
        self._subscribers: Dict[str, LatestWinsQueue] = {}

        self._frame_index_provider: Optional[Callable[[], Optional[int]]] = None

        self._thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        """Start the polling thread. Raises `RuntimeError` if already running."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("ObservationSource already started")
        self._stop_requested.clear()
        self._thread = threading.Thread(target=self._run, name="ObservationSource", daemon=True)
        self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        """Signal the polling thread to stop and join it (does not touch `shutdown_event`)."""
        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def failed(self) -> bool:
        """True once `max_consecutive_failures` poll cycles in a row have raised. See module
        docstring's "Failure policy" section — the session layer decides what to do about it."""
        return self._failed

    # -- configuration -----------------------------------------------------------

    def set_frame_index_provider(self, provider: Optional[Callable[[], Optional[int]]]) -> None:
        """Register (or clear, with `None`) a callable the source polls once per cycle to stamp
        `Observation.frame_index`. Wired by the recorder in Wave 5; `None` (the default) means
        every observation's `frame_index` stays `None`."""
        self._frame_index_provider = provider

    # -- consumers ----------------------------------------------------------------

    def latest(self) -> Optional[Observation]:
        """Non-blocking read of the newest published `Observation`, or `None` before the first
        successful poll cycle."""
        with self._cond:
            return self._latest

    def await_next(self, after_seq: int, timeout: Optional[float] = None) -> Optional[Observation]:
        """Block until an `Observation` with `seq > after_seq` is published, or `timeout` elapses.

        Pass `after_seq=0` (or the seq of the last observation you've already consumed) to wait
        for "the next new one". Returns `None` on timeout; `timeout=None` blocks indefinitely.
        """
        deadline = None if timeout is None else time.perf_counter() + timeout
        with self._cond:
            while self._latest is None or self._latest.seq <= after_seq:
                if deadline is not None:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        return None
                    self._cond.wait(timeout=remaining)
                else:
                    self._cond.wait()
            return self._latest

    def subscribe(self, name: str, maxsize: int = 2) -> LatestWinsQueue:
        """Register a new subscriber and return its `LatestWinsQueue` handle. Re-subscribing under
        an existing `name` replaces the previous queue (the old handle stops receiving updates)."""
        q = LatestWinsQueue(maxsize=maxsize)
        with self._subscribers_lock:
            self._subscribers[name] = q
        return q

    def unsubscribe(self, name: str) -> None:
        """Remove a subscriber by name. No-op if `name` isn't currently subscribed."""
        with self._subscribers_lock:
            self._subscribers.pop(name, None)

    # -- internal polling loop -----------------------------------------------------

    def _run(self) -> None:
        while not self._stop_requested.is_set() and not self._shutdown_event.is_set():
            frame_start = time.perf_counter()
            try:
                self._poll_once()
                self._consecutive_failures = 0
            except Exception:
                self._consecutive_failures += 1
                logger.exception(
                    "ObservationSource: poll cycle failed (%d/%d consecutive failures)",
                    self._consecutive_failures,
                    self._max_consecutive_failures,
                )
                if self._consecutive_failures >= self._max_consecutive_failures:
                    self._failed = True
                    self.failed_event.set()
                    logger.error(
                        "ObservationSource: %d consecutive failures (limit %d) -- marking "
                        "source failed and stopping the poll loop. shutdown_event is NOT set; "
                        "the session layer decides what happens next.",
                        self._consecutive_failures,
                        self._max_consecutive_failures,
                    )
                    return
            elapsed = time.perf_counter() - frame_start
            remaining = max(0.0, 1.0 / self._fps - elapsed - _PACING_BUFFER)
            self._shutdown_event.wait(remaining)

    def _poll_once(self) -> None:
        raw = self._robot.get_all_states()

        if raw.top_image is None or raw.bottom_image_raw is None:
            raise RuntimeError(
                "ObservationSource: get_all_states() returned no image(s) "
                f"(top={raw.top_image is None!r} missing, bottom_raw={raw.bottom_image_raw is None!r} missing)"
            )

        top = np.asarray(raw.top_image)
        bottom = self._camera.process_bottom(raw.bottom_image_raw)
        if bottom is None:
            raise RuntimeError("ObservationSource: camera.process_bottom returned None")
        bottom = np.asarray(bottom)

        robot_state = RobotState.from_legacy_dict(raw.robot_state_dict)

        frame_index: Optional[int] = None
        if self._frame_index_provider is not None:
            try:
                frame_index = self._frame_index_provider()
            except Exception:
                logger.exception("ObservationSource: frame_index_provider raised, using None")
                frame_index = None

        with self._cond:
            self._seq += 1
            obs = Observation(
                seq=self._seq,
                frame_index=frame_index,
                timestamp=raw.timestamp,
                top_image=top,
                bottom_image=bottom,
                robot_state=robot_state,
            )
            self._latest = obs
            self._cond.notify_all()

        with self._subscribers_lock:
            subscribers = list(self._subscribers.values())
        for q in subscribers:
            q._put_latest(obs)
