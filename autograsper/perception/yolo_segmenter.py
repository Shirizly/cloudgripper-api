"""`YoloOccupancyProvider` + `SegmentationWorker` — segmenter-native perception (Wave 3a).

Design reference: `autograsper/design/03_segmenter_native_design.md` §2.1 (`SegmentationWorker` is
the successor of `recording_seg.SegmentationThread`, "promoted from a recorder detail to the
standard perception provider") and §7.3 (confidence policy), §7.4 (degraded fallback).

`SegmentationWorker` is generic over any `OccupancyProvider` (duck-typed against
`perception.occupancy.OccupancyProvider`) -- it is written and tested against
`YoloOccupancyProvider` here because that is design 03's payoff case (no move-aside needed), but it
works identically wrapping `BackgroundDiffProvider` if a future session wants continuous background-
diff masks (that combination is still only valid with the robot out of frame -- see
`background_diff.py`'s docstring; `SegmentationWorker` has no opinion on that, it just calls
`provider.compute(obs)` on whatever the newest observation is).

Porting notes:
- `recording_seg.py::SegmentationThread` -- the loop shape (poll shared state, skip unchanged
  frames, run segmentation, publish result) is the direct ancestor of `SegmentationWorker._run`.
  Differences: (1) subscribes to `ObservationSource` instead of polling a shared-state timestamp
  field under a lock; (2) publishes a structured `OccupancyResult` instead of a bare mask array
  into mutable shared state; (3) has explicit `latest()`/`await_result()` consumer API instead of
  callers reaching into `shared_state.latest_mask` directly; (4) tracks a `.degraded` signal (no
  legacy equivalent -- `SegmentationThread` only logged exceptions and kept looping forever).
- `image_collector/chickpea_segmenter.py::ChickpeaSegmenter` -- reused as-is via a **lazy** import
  inside `YoloOccupancyProvider.__init__` (never at module import time), so importing this module
  never requires `ultralytics`/`torch` to be installed -- only actually constructing the provider
  does (CONVENTIONS.md / task spec: "no GPU ... in tests"; the construction-is-lazy test in
  `tests/test_perception_yolo_segmenter.py` asserts this via `sys.modules`).

Threading:
- `YoloOccupancyProvider.compute()` is synchronous and, per `ChickpeaSegmenter`, not documented as
  thread-safe for concurrent calls on the same instance -- only `SegmentationWorker`'s single
  thread calls it for a given provider instance, matching design 03 §2.1's "GPU note: ... one
  worker thread suffices".
- `SegmentationWorker` owns exactly one background thread (name `"SegmentationWorker"`), mirroring
  `ObservationSource`'s threading model: `latest()`/`await_result()`/`.degraded`/`start()`/`stop()`
  are safe from any thread.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional, Tuple

import cv2
import numpy as np

from autograsper.observation.source import ObservationSource
from autograsper.observation.types import Observation
from autograsper.perception.occupancy import (
    ClumpStats,
    Instance,
    OccupancyProvider,
    OccupancyResult,
    clump_stats_from_mask,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from autograsper.config_schema import YoloConfig
    from autograsper.observation.debug import DebugSink
    from autograsper.perception.frames import CoordinateFrames

logger = logging.getLogger(__name__)

# design 03 §7.3: "proposed default: occupied for safety, threshold 0.25 [model conf_threshold];
# counted for statistics at 0.5".
_DEFAULT_STATS_CONF_THRESHOLD = 0.5


class YoloOccupancyProvider:
    """`ChickpeaSegmenter`-backed occupancy provider (design 03 §2.1).

    Confidence policy (design 03 §7.3): the underlying model call already excludes detections below
    `yolo_config.conf_threshold` (default 0.25) -- that filtering happens inside
    `ChickpeaSegmenter.predict`, not here. On top of that:
    - `crop_mask` (and `instances`) include **every** instance the model returned (after the
      `min_instance_area_px` sanity filter below) -- safety-conservative, matching design 03 §7.3's
      "occupied for safety" side of the policy. Consumers doing safety checks (e.g. the `LowerTool`
      guard) should read `crop_mask`/`instances`.
    - `stats` (`ClumpStats`) is computed only from instances at or above `stats_conf_threshold`
      (default 0.5) -- the "counted for statistics" side. Consumers doing counting/reset-decision
      logic (e.g. `check_reset_needed`-style heuristics) should read `stats`, not
      `cv2.countNonZero(crop_mask)`.
    - `min_instance_area_px` is a light sanity filter (default 0, i.e. off) dropping instances
      whose mask has fewer than that many non-zero pixels *before* either of the above -- this
      replaces legacy's removed `Granuler_detection.min_granule_size` config key (design 03 §6:
      "instance filtering is the model's job; a `min_instance_area_px` safety filter remains").
      `config_schema.YoloConfig` does not carry either of these two knobs (Wave 1 schema); both are
      read via `getattr(yolo_config, ..., default)` so a future config-schema addition is picked up
      without code changes here. Logged in `design/IMPLEMENTATION_LOG.md`.
    """

    def __init__(
        self,
        frames: "CoordinateFrames",
        yolo_config: "YoloConfig",
        debug: Optional["DebugSink"] = None,
    ):
        # Lazy import: constructing this class is the only thing that requires `ultralytics`/
        # `torch` to be installed. Importing this module does not.
        from image_collector.chickpea_segmenter import ChickpeaSegmenter

        self._frames = frames
        self._debug = debug
        self._min_instance_area_px = int(getattr(yolo_config, "min_instance_area_px", 0) or 0)
        self._stats_conf_threshold = float(
            getattr(yolo_config, "stats_conf_threshold", _DEFAULT_STATS_CONF_THRESHOLD)
        )
        self._segmenter = ChickpeaSegmenter(
            weights_path=yolo_config.weights_path,
            conf_threshold=yolo_config.conf_threshold,
            iou_threshold=yolo_config.iou_threshold,
            imgsz=yolo_config.imgsz,
        )

    def compute(self, obs: Observation) -> OccupancyResult:
        crop_img = self._frames.crop(obs.bottom_image)
        result = self._segmenter.predict(crop_img, return_format="dict")

        instances: list = []
        for mask, box, conf in zip(result["individual"], result["boxes"], result["confidences"]):
            if self._min_instance_area_px > 0 and cv2.countNonZero(mask) < self._min_instance_area_px:
                continue
            instances.append(
                Instance(
                    mask=mask,
                    box=tuple(float(v) for v in box),
                    confidence=float(conf),
                )
            )

        h, w = crop_img.shape[:2]
        crop_mask = _union_masks(instances, (h, w))
        stats_instances = [inst for inst in instances if inst.confidence >= self._stats_conf_threshold]
        stats_mask = _union_masks(stats_instances, (h, w))
        stats: ClumpStats = clump_stats_from_mask(stats_mask)

        grid_mask = self._frames.crop_to_grid(crop_mask)

        if self._debug is not None and self._debug.enabled:
            self._debug.save("yolo_crop_mask", crop_mask)
            self._debug.save("yolo_stats_mask", stats_mask)

        return OccupancyResult(
            source_seq=obs.seq,
            frame_index=obs.frame_index,
            timestamp=obs.timestamp,
            grid_mask=grid_mask,
            crop_mask=crop_mask,
            instances=tuple(instances),
            stats=stats,
        )


def _union_masks(instances, shape: Tuple[int, int]) -> np.ndarray:
    """Binary uint8 {0, 255} union of `instances[*].mask`, `shape`-sized (all-zero if none)."""
    combined = np.zeros(shape, dtype=np.uint8)
    for inst in instances:
        if inst.mask is not None:
            combined = np.maximum(combined, inst.mask)
    return combined


def occupied_mask_at_conf(result: OccupancyResult, min_conf: float) -> np.ndarray:
    """Recompute a crop-frame occupancy mask from `result.instances` at an arbitrary confidence
    cutoff, for consumers that want something other than the two fixed cutoffs baked into
    `crop_mask` (all instances) / `stats` (>= `stats_conf_threshold`) -- e.g. a planner that wants
    to sweep several thresholds without recomputing the model call. Returns an all-zero mask shaped
    like `result.crop_mask` if no instance clears `min_conf` or `result.instances` is empty
    (background-diff results always take this path).
    """
    shape = result.crop_mask.shape[:2]
    selected = [inst for inst in result.instances if inst.confidence >= min_conf]
    return _union_masks(selected, shape)


class SegmentationWorker:
    """Background thread wrapping any `OccupancyProvider`, publishing `OccupancyResult`s with
    latest-wins semantics against an `ObservationSource` (design 03 §2.1).

    API mirrors `ObservationSource`: `latest()` / `await_result(min_source_seq, timeout)` /
    `start()` / `stop()`, plus a `.degraded` flag (see "Degraded detection" below).

    Latest-wins: subscribes to `source` with a maxsize-1 `LatestWinsQueue` (same drop-oldest
    primitive the observation layer already uses for the UI stream), so a `compute()` slower than
    the observation rate never causes the worker to fall behind processing a backlog -- it always
    picks up the newest observation once it's ready for another (design 03 §2.1: "if it ever lags,
    latest-wins semantics keep it from queueing stale work").

    Degraded detection (design 03 §7.4): the worker never raises out of its own thread and never
    touches `shutdown_event` -- it only flips `.degraded` to `True` and keeps running, for the
    session layer (Wave 5+) to observe and act on. Two independent triggers, either one is enough
    to set it (it never resets once `True` -- that's the session layer's call, e.g. by constructing
    a fresh worker):
    - `provider.compute()` raises on `max_consecutive_failures` (default 3) observations in a row.
    - The clump count (`OccupancyResult.stats.num_clumps`, chosen over `len(instances)` so this
      works identically for background-diff, whose `instances` is always empty) drops to 0 for
      `zero_detection_streak_for_degraded` (default 5, i.e. ~2s at the template's 2.5 FPS)
      consecutive successful results, but only counted once at least one prior result showed
      `num_clumps > 0` (a workspace that starts empty and stays empty -- e.g. before setup -- is not
      "degraded", it's just empty; design 03 §7.4's trigger is specifically "0 detections while the
      previous frames showed many"). Pass `zero_detection_streak_for_degraded=None` to disable this
      trigger entirely (still get the consecutive-failure trigger). Both defaults are this task's
      choice (not dictated by the design docs) -- logged in `design/IMPLEMENTATION_LOG.md`.
    """

    def __init__(
        self,
        provider: OccupancyProvider,
        source: ObservationSource,
        shutdown_event: threading.Event,
        *,
        max_consecutive_failures: int = 3,
        zero_detection_streak_for_degraded: Optional[int] = 5,
    ):
        self._provider = provider
        self._source = source
        self._shutdown_event = shutdown_event
        self._max_consecutive_failures = max_consecutive_failures
        self._zero_streak_limit = zero_detection_streak_for_degraded

        self._cond = threading.Condition()
        self._latest: Optional[OccupancyResult] = None
        self._last_processed_seq = 0

        self._consecutive_failures = 0
        self._ever_seen_detections = False
        self._zero_streak = 0
        self._degraded = False

        self._queue = None
        self._subscriber_name = f"segmentation_worker_{id(self)}"
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Start the worker thread. Raises `RuntimeError` if already running."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("SegmentationWorker already started")
        self._stop_requested.clear()
        self._queue = self._source.subscribe(self._subscriber_name, maxsize=1)
        self._thread = threading.Thread(target=self._run, name="SegmentationWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        """Signal the worker thread to stop, join it, and unsubscribe from `source`. Does not
        touch `shutdown_event`."""
        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._source.unsubscribe(self._subscriber_name)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def degraded(self) -> bool:
        """See class docstring's "Degraded detection" section. Sticky once `True`."""
        return self._degraded

    # -- consumers --------------------------------------------------------------

    def latest(self) -> Optional[OccupancyResult]:
        """Non-blocking read of the newest published `OccupancyResult`, or `None` before the first
        successful `compute()`."""
        with self._cond:
            return self._latest

    def await_result(
        self, min_source_seq: int, timeout: Optional[float] = None
    ) -> Optional[OccupancyResult]:
        """Block until an `OccupancyResult` with `source_seq >= min_source_seq` is published, or
        `timeout` elapses. Returns `None` on timeout; `timeout=None` blocks indefinitely.

        `>=` (not `>`, unlike `ObservationSource.await_next`'s `seq > after_seq`): callers typically
        pass "the seq of the observation right after my last motion completed" and want a mask
        computed from *that* observation or a later one to count, per design 03 §3.2's freshness
        rule.
        """
        deadline = None if timeout is None else time.perf_counter() + timeout
        with self._cond:
            while self._latest is None or self._latest.source_seq < min_source_seq:
                if deadline is not None:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        return None
                    self._cond.wait(timeout=remaining)
                else:
                    self._cond.wait()
            return self._latest

    # -- internal loop ------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_requested.is_set() and not self._shutdown_event.is_set():
            obs = self._queue.get(timeout=0.2)
            if obs is None:
                continue  # timed out waiting for a new observation; recheck stop/shutdown flags
            if obs.seq <= self._last_processed_seq:
                continue  # already processed (shouldn't happen with a maxsize=1 subscription, but
                # cheap to guard -- design 03 §2.1: "processes the newest unprocessed observation")
            self._last_processed_seq = obs.seq

            try:
                result = self._provider.compute(obs)
            except Exception:
                logger.exception(
                    "SegmentationWorker: provider.compute() raised for source_seq=%d", obs.seq
                )
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._max_consecutive_failures:
                    if not self._degraded:
                        logger.error(
                            "SegmentationWorker: %d consecutive compute() failures (limit %d) -- "
                            "marking degraded.",
                            self._consecutive_failures,
                            self._max_consecutive_failures,
                        )
                    self._degraded = True
                continue

            self._consecutive_failures = 0
            self._update_zero_streak(result)

            with self._cond:
                self._latest = result
                self._cond.notify_all()

    def _update_zero_streak(self, result: OccupancyResult) -> None:
        num_detections = result.stats.num_clumps
        if num_detections > 0:
            self._ever_seen_detections = True
            self._zero_streak = 0
            return
        if not self._ever_seen_detections:
            return  # empty workspace before anything was ever seen is not "degraded"
        self._zero_streak += 1
        if self._zero_streak_limit is not None and self._zero_streak >= self._zero_streak_limit:
            if not self._degraded:
                logger.error(
                    "SegmentationWorker: %d consecutive zero-detection results after having seen "
                    "occupancy (limit %d) -- marking degraded.",
                    self._zero_streak,
                    self._zero_streak_limit,
                )
            self._degraded = True
