"""`Recorder` — pure frame/state/mask sink (Wave 5).

Design reference: `autograsper/design/02_proposed_architecture.md` §3.6 ("demoted to a pure sink
... subscribes to `ObservationSource`; writes frames/states/masks for whatever directory the
session runner points it at; assigns frame indices. It no longer owns a robot connection, the
camera pipeline, segmentation, or pacing decisions"); `autograsper/design/03_segmenter_native_design.md`
§4 (mask + `masks_meta.jsonl` saving).

Porting notes (copied and adapted, not imported — legacy `recording.py`/`recording_seg.py` are
frozen per CONVENTIONS.md):
- `recording.py::Recorder.record`/`_update`/`_capture_frame`/`_save_individual_images` — the
  per-frame capture shape (`image_top_<f>.jpeg`/`image_bottom_<f>.jpeg` under `Images/`/
  `Bottom_Images/`, created lazily) is ported into `_capture`. The FPS pacing loop itself moved to
  `observation.source.ObservationSource` (Wave 2) — this class is a *subscriber*, not a poller: it
  runs its own consumer thread over `source.subscribe('recorder', maxsize=8)` and never blocks the
  source.
- `recording_seg.py::Recorder._capture_frame`'s `.npy` mask saving (`Masks/mask_<f>.npy`) — ported
  into `_maybe_save_mask`, keyed by `frame_index` instead of legacy's `latest_mask_saved` boolean
  flag (design 03 §4: "masks are keyed by `frame_index`, saved when their source frame is saved,
  once" — realized here by only ever consulting `occupancy_supplier.latest()` and comparing its
  `source_seq` against what was last saved, monotonically).
- `recording.py::Recorder.save_state`/`save_action_summary` — `states.json`/`actions.json` row
  shapes now live in `session.storage.StatesWriter`/`ActionsWriter`; this module only calls them.
- `recording.py::Recorder.start_new_recording`'s `shared_state.action_tracker.clear()` — ported
  into `_reset_dir_state` (fires on both `start()` and `retarget()`, matching legacy clearing the
  tracker at the top of every new recording directory so `actions.json` only ever contains that
  directory's own actions).
- `recording.py::Recorder._start_new_video`/`_start_or_restart_video_writers` (video mode, used
  when `camera.save_images_individually` is `False`) — ported into `_write_video_frame`, same
  `video_<n>.mp4` naming under `Video/`/`Bottom_Video/`, same `clip_length`-triggered rollover.

**Frame-index assignment** (design 02 §3.2/§3.6: "the recorder assigns frame indices and reports
them into observations"): `start()`/`retarget()` register `_frame_index_provider` with the shared
`ObservationSource` via `set_frame_index_provider`. That callback runs on the *source's* polling
thread once per cycle (not this class's own consumer thread) and is the sole place the frame
counter is incremented — this avoids a race between two independently-incrementing counters (the
provider callback and this class's consumer thread processing the same observations slightly
later). The consumer thread that actually writes to disk just reads back `Observation.frame_index`,
already stamped by the time it sees the observation.

Threading: `Recorder` owns exactly one background thread (name `"Recorder"`), consuming
`source.subscribe('recorder', maxsize=8)`. `start()`/`retarget()`/`pause()`/`stop_recording()`/
`snapshot()` are meant to be called from the session runner's single control thread; `mask_for_frame`
and `first_frame_event` are safe to read from any thread.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np

from autograsper.execution.actions import ActionTracker
from autograsper.observation.source import ObservationSource
from autograsper.observation.types import Observation
from autograsper.session.storage import ActionsWriter, MasksMetaWriter, StatesWriter

logger = logging.getLogger(__name__)

_DEFAULT_MASK_RING_SIZE = 512


class Recorder:
    """Pure sink: subscribes to `source`, writes frames/states/masks into whichever directory
    `start()`/`retarget()` last pointed at. See module docstring for the porting map."""

    def __init__(
        self,
        source: ObservationSource,
        camera_config,
        *,
        frames=None,
        occupancy_supplier=None,
        action_tracker: Optional[ActionTracker] = None,
        mask_ring_size: int = _DEFAULT_MASK_RING_SIZE,
        subscriber_name: str = "recorder",
    ) -> None:
        """
        Args:
            source: `observation.source.ObservationSource`-duck-typed.
            camera_config: `config_schema.CameraConfig`-duck-typed (`save_images_individually`,
                `clip_length`, `record_only_after_action`, `fps`).
            frames: `perception.frames.CoordinateFrames`, or `None` — accepted for interface
                symmetry with the rest of the stack (e.g. a future debug overlay); not required by
                any current capture path (masks are saved verbatim in whatever frame the occupancy
                supplier already produced them in).
            occupancy_supplier: `SegmentationWorker`-shaped (`.latest()`), or `None` to skip mask
                saving entirely (e.g. `perception.provider == "none"`).
            action_tracker: shared `ActionTracker` this recorder reads
                (`get_action_for_frame`)/clears (on `start()`/`retarget()`, matching legacy) — not
                writes; only the executor writes actions.
            mask_ring_size: bounded in-memory `frame_index -> grid_mask` map exposed via
                `mask_for_frame`, for `session.storage.TransitionWriter`.
        """
        self._source = source
        self._camera_config = camera_config
        self._frames = frames
        self._occupancy_supplier = occupancy_supplier
        self._action_tracker = action_tracker
        self._ring_size = mask_ring_size
        self._subscriber_name = subscriber_name

        self._save_images_individually = bool(getattr(camera_config, "save_images_individually", True))
        self._clip_length = getattr(camera_config, "clip_length", None)
        self._record_only_after_action = bool(getattr(camera_config, "record_only_after_action", False))
        self._fps = float(getattr(camera_config, "fps", 2.5))

        self._counter_lock = threading.Lock()
        self._frame_counter = 0
        self._pending_snapshots = 0

        self._output_dir: Optional[str] = None
        self._images_dir: Optional[str] = None
        self._bottom_images_dir: Optional[str] = None
        self._masks_dir: Optional[str] = None
        self._video_dir: Optional[str] = None
        self._bottom_video_dir: Optional[str] = None

        self._states_writer: Optional[StatesWriter] = None
        self._masks_meta_writer: Optional[MasksMetaWriter] = None

        self._video_writer_top: Optional[cv2.VideoWriter] = None
        self._video_writer_bottom: Optional[cv2.VideoWriter] = None
        self._video_counter = 0

        self._mask_ring: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._last_saved_source_seq: Optional[int] = None

        self._recording_enabled = False
        self.first_frame_event = threading.Event()
        self._processed_cond = threading.Condition()
        self._last_processed_frame_index: Optional[int] = None

        self._queue = None
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()
        self._last_processed_seq = 0

    # -- lifecycle ------------------------------------------------------------

    def start(self, target_dir: str) -> None:
        """Begin recording into `target_dir`. Idempotent w.r.t. the consumer thread: the first
        call spawns it; later calls (after `pause()`) just re-enable + retarget."""
        self._reset_dir_state(target_dir)
        self._recording_enabled = True
        self._source.set_frame_index_provider(self._frame_index_provider)
        if self._thread is None:
            self._queue = self._source.subscribe(self._subscriber_name, maxsize=8)
            self._stop_requested.clear()
            self._thread = threading.Thread(target=self._run, name="Recorder", daemon=True)
            self._thread.start()

    def retarget(self, new_dir: str) -> None:
        """Finalize the current directory's writers (`states.json`/`actions.json`), then point at
        `new_dir` (`task/` -> `restore/`)."""
        self._finalize_current_dir()
        self._reset_dir_state(new_dir)
        self._recording_enabled = True

    def pause(self) -> None:
        """Finalize the current directory's writers and stop capturing (frame_index_provider
        starts returning `None`) without tearing down the consumer thread — the between-episode
        gap (legacy `disable_recording()` + `pause=True`)."""
        self._finalize_current_dir()
        self._recording_enabled = False

    def stop_recording(self) -> None:
        """Finalize writers, stop the consumer thread, and unsubscribe/un-register from `source`.
        Call once at the end of the whole run (not between episodes — use `pause()` for that)."""
        self._finalize_current_dir()
        self._recording_enabled = False
        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._queue is not None:
            self._source.unsubscribe(self._subscriber_name)
            self._queue = None
        self._source.set_frame_index_provider(None)
        self._release_video_writers()

    def snapshot(self) -> None:
        """`record_only_after_action` mode: request that the next observation be captured (simple
        port — see module/class docstrings; the executor-side record hook design 02 dropped is not
        wired up automatically anywhere in this wave, callers invoke this directly if they want
        it)."""
        with self._counter_lock:
            self._pending_snapshots += 1

    # -- synchronization for callers that need "has this frame been written yet" ----

    def wait_until_frame_processed(self, frame_index: Optional[int], timeout: Optional[float] = None) -> bool:
        """Block until the consumer thread has finished processing (images/states/mask all
        written for) a frame `>= frame_index`, or `timeout` elapses. Returns `True` if satisfied,
        `False` on timeout. `frame_index=None` returns `True` immediately (nothing to wait for —
        e.g. no `Observation` was available at all).

        Exists to close a real race for callers like `session.coordinator.SessionRunner`, which
        needs the recorder to have durably saved the mask for a plan's last frame *before* calling
        `session.storage.TransitionWriter.finalize()` — `Executor.execute_primitive`'s own
        `source.await_next(...)` only guarantees the *Observation* exists, not that this class's
        independent consumer thread has already processed it.
        """
        if frame_index is None:
            return True
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._processed_cond:
            while self._last_processed_frame_index is None or self._last_processed_frame_index < frame_index:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._processed_cond.wait(timeout=remaining)
                else:
                    self._processed_cond.wait()
            return True

    # -- consumer API for TransitionWriter -------------------------------------

    def mask_for_frame(self, frame_index: int, mode: str) -> Optional[np.ndarray]:
        """`mode="before"`: latest saved grid mask with frame `<= frame_index`. `mode="after"`:
        earliest saved grid mask with frame `>= frame_index`. `None` if no saved mask satisfies
        that. Bounded to the last `mask_ring_size` saved masks of the *current* directory (cleared
        on `start()`/`retarget()` — see module docstring)."""
        if mode not in ("before", "after"):
            raise ValueError(f"mask_for_frame: mode must be 'before'/'after', got {mode!r}")
        if not self._mask_ring:
            return None
        keys = sorted(self._mask_ring.keys())
        if mode == "before":
            candidates = [k for k in keys if k <= frame_index]
            if not candidates:
                return None
            return self._mask_ring[max(candidates)]
        candidates = [k for k in keys if k >= frame_index]
        if not candidates:
            return None
        return self._mask_ring[min(candidates)]

    # -- internal: directory (re)targeting -------------------------------------

    def _reset_dir_state(self, target_dir: str) -> None:
        if not target_dir:
            raise ValueError("Recorder: target_dir must be non-empty")
        self._output_dir = target_dir
        self._images_dir = os.path.join(target_dir, "Images")
        self._bottom_images_dir = os.path.join(target_dir, "Bottom_Images")
        self._masks_dir = os.path.join(target_dir, "Masks")
        self._video_dir = os.path.join(target_dir, "Video")
        self._bottom_video_dir = os.path.join(target_dir, "Bottom_Video")
        with self._counter_lock:
            self._frame_counter = 0
            self._pending_snapshots = 0
        self.first_frame_event.clear()
        with self._processed_cond:
            self._last_processed_frame_index = None
        self._last_saved_source_seq = None
        self._mask_ring.clear()
        if self._action_tracker is not None:
            self._action_tracker.clear()
        self._states_writer = StatesWriter(target_dir)
        self._masks_meta_writer = MasksMetaWriter(target_dir)
        self._release_video_writers()
        self._video_counter = 0

    def _finalize_current_dir(self) -> None:
        if self._output_dir is None:
            return
        if self._states_writer is not None:
            self._states_writer.finalize()
        if self._action_tracker is not None:
            ActionsWriter.finalize(self._output_dir, self._action_tracker.get_all_actions())
        self._release_video_writers()

    # -- internal: frame index assignment (runs on the ObservationSource thread) ----

    def _frame_index_provider(self) -> Optional[int]:
        if not self._recording_enabled:
            return None
        with self._counter_lock:
            if self._record_only_after_action:
                if self._pending_snapshots <= 0:
                    return None
                self._pending_snapshots -= 1
            idx = self._frame_counter
            self._frame_counter += 1
            return idx

    # -- internal: consumer thread ----------------------------------------------

    def _run(self) -> None:
        while not self._stop_requested.is_set():
            obs = self._queue.get(timeout=0.2)
            if obs is None:
                continue
            if obs.seq <= self._last_processed_seq:
                continue
            self._last_processed_seq = obs.seq
            if obs.frame_index is None:
                continue
            try:
                self._capture(obs)
            except Exception:
                logger.exception("Recorder: error capturing frame_index=%s", obs.frame_index)

    def _capture(self, obs: Observation) -> None:
        frame_index = obs.frame_index
        assert frame_index is not None

        if self._save_images_individually:
            os.makedirs(self._images_dir, exist_ok=True)
            os.makedirs(self._bottom_images_dir, exist_ok=True)
            cv2.imwrite(os.path.join(self._images_dir, f"image_top_{frame_index}.jpeg"), obs.top_image)
            cv2.imwrite(
                os.path.join(self._bottom_images_dir, f"image_bottom_{frame_index}.jpeg"), obs.bottom_image
            )
        else:
            self._write_video_frame(obs, frame_index)

        robot_state_dict = dataclasses.asdict(obs.robot_state) if obs.robot_state is not None else {}
        action = (
            self._action_tracker.get_action_for_frame(frame_index)
            if self._action_tracker is not None
            else None
        )
        self._states_writer.record(robot_state_dict, obs.timestamp, frame_index, action)

        self._maybe_save_mask(frame_index, obs.seq)

        if not self.first_frame_event.is_set():
            self.first_frame_event.set()

        with self._processed_cond:
            self._last_processed_frame_index = frame_index
            self._processed_cond.notify_all()

    # -- internal: video mode -----------------------------------------------------

    def _write_video_frame(self, obs: Observation, frame_index: int) -> None:
        if self._clip_length and frame_index != 0 and frame_index % self._clip_length == 0:
            self._video_counter += 1
            self._release_video_writers()
        if self._video_writer_top is None or self._video_writer_bottom is None:
            os.makedirs(self._video_dir, exist_ok=True)
            os.makedirs(self._bottom_video_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            top_shape = obs.top_image.shape[1::-1]
            bottom_shape = obs.bottom_image.shape[1::-1]
            self._video_writer_top = cv2.VideoWriter(
                os.path.join(self._video_dir, f"video_{self._video_counter}.mp4"), fourcc, self._fps, top_shape
            )
            self._video_writer_bottom = cv2.VideoWriter(
                os.path.join(self._bottom_video_dir, f"video_{self._video_counter}.mp4"),
                fourcc,
                self._fps,
                bottom_shape,
            )
        self._video_writer_top.write(obs.top_image)
        self._video_writer_bottom.write(obs.bottom_image)

    def _release_video_writers(self) -> None:
        if self._video_writer_top is not None:
            self._video_writer_top.release()
            self._video_writer_top = None
        if self._video_writer_bottom is not None:
            self._video_writer_bottom.release()
            self._video_writer_bottom = None

    # -- internal: mask saving ----------------------------------------------------

    def _maybe_save_mask(self, frame_index: int, obs_seq: int) -> None:
        if self._occupancy_supplier is None:
            return
        occ = self._occupancy_supplier.latest()
        if occ is None:
            return
        if self._last_saved_source_seq is not None:
            if occ.source_seq <= self._last_saved_source_seq:
                return  # already saved (keyed by source_seq, once per new mask -- design 03 §4)
            if occ.source_seq > obs_seq:
                return  # not yet caught up to this observation; try again next frame
        # Bootstrap: the very first mask saved in a directory is saved unconditionally (even if the
        # occupancy supplier is already "ahead" of this frame) so `mask_for_frame`'s "before" lookup
        # has something to find as early as possible in the episode, instead of only ever catching
        # up once the recorder's own processing has drifted back in sync with the live supplier.

        os.makedirs(self._masks_dir, exist_ok=True)
        mask_path = os.path.join(self._masks_dir, f"mask_{frame_index}.npy")
        np.save(mask_path, occ.grid_mask)
        self._last_saved_source_seq = occ.source_seq
        self._store_ring(frame_index, occ.grid_mask)

        num_instances = len(occ.instances) if getattr(occ, "instances", None) is not None else None
        stats = getattr(occ, "stats", None)
        mask_area = int(stats.total_area_px) if stats is not None else None
        self._masks_meta_writer.record(frame_index, occ.source_seq, num_instances, mask_area)

    def _store_ring(self, frame_index: int, mask: np.ndarray) -> None:
        self._mask_ring[frame_index] = mask.copy()
        while len(self._mask_ring) > self._ring_size:
            self._mask_ring.popitem(last=False)
