# `autograsper/recording/`, `autograsper/ui/` — Recorder, MJPEGServer

Status: **implemented** (Wave 5). Design reference:
[`../design/02_proposed_architecture.md`](../design/02_proposed_architecture.md) §3.6 (`Recorder`
— core scope), §6 ("Flask MJPEG debugging UI" stays); [`../design/03_segmenter_native_design.md`](../design/03_segmenter_native_design.md)
§4 (mask + `masks_meta.jsonl` saving). See [session.md](session.md) for how `SessionRunner` drives
this class through an episode, and [dataset_formats.md](dataset_formats.md) for the exact on-disk
shapes it produces.

## `recording/recorder.py` — `Recorder`

### Purpose

A pure frame/state/mask sink: subscribes to `observation.source.ObservationSource`, writes
whatever `session.coordinator.SessionRunner` points it at, and assigns frame indices. Owns no
robot connection, camera pipeline, segmentation, or pacing decisions of its own — `recording.py`
and `recording_seg.py` collapse into this one class (segmentation is a perception worker, not a
recorder concern, per design 03).

### Public API

`Recorder(source, camera_config, *, frames=None, occupancy_supplier=None, action_tracker=None,
mask_ring_size=512, subscriber_name="recorder")`:

- `start(target_dir)` — begin recording into `target_dir`. The **first** call spawns the
  consumer thread (subscribes to `source.subscribe('recorder', maxsize=8)`, registers
  `set_frame_index_provider`); later calls (after `pause()`) just re-target + re-enable.
- `retarget(new_dir)` — finalize the current directory's writers (`states.json`/`actions.json`),
  then point at `new_dir` (`task/` -> `restore/`).
- `pause()` — finalize the current directory's writers and stop capturing (the between-episode
  gap) without tearing down the consumer thread.
- `stop_recording()` — finalize writers, stop the consumer thread, unsubscribe/un-register from
  `source`. Call once at the end of the whole run.
- `snapshot()` — `record_only_after_action` mode: request the next observation be captured (a
  simple, direct port — see "Known limitations" below).
- `mask_for_frame(frame_index, mode) -> Optional[np.ndarray]` — `mode="before"`: latest saved grid
  mask with frame `<= frame_index`; `mode="after"`: earliest saved with frame `>= frame_index`.
  Bounded to the last `mask_ring_size` saved masks of the *current* directory (cleared on
  `start()`/`retarget()`). Feeds `session.storage.TransitionWriter`.
- `wait_until_frame_processed(frame_index, timeout=None) -> bool` — block until the consumer
  thread has finished processing a frame `>= frame_index` (or `frame_index is None`, which returns
  `True` immediately). Closes a real race between `Executor`'s own end-of-primitive
  `source.await_next(...)` (which only guarantees the *Observation* exists) and this class's
  independently-scheduled consumer thread actually having written it — see
  [session.md](session.md)'s "mask-finalization race" section for why `SessionRunner` calls this.
- `first_frame_event: threading.Event` — set once the first frame of the current directory has
  been fully captured; `SessionRunner` waits on this before sending any order of an episode
  (design 02 §3.5's ordering guarantee).

### Frame-index assignment

`start()`/`retarget()` register an internal counter with the shared `ObservationSource` via
`set_frame_index_provider`. That callback runs on the **source's own polling thread**, once per
poll cycle — not this class's consumer thread — and is the sole place the frame counter
increments. This avoids a race between two independently-incrementing counters: by the time the
consumer thread sees an `Observation`, its `frame_index` is already stamped, so `_capture` just
reads it back. In `record_only_after_action` mode, the provider returns `None` (skipping both the
frame-index assignment and the eventual capture) unless a `snapshot()` request is pending.

### Mask saving

Every captured frame calls `occupancy_supplier.latest()` (a `SegmentationWorker`-shaped duck type,
or `None` to skip mask saving entirely). A mask is saved (`Masks/mask_<f>.npy`, `.npy` binary
`grid_mask`) once per distinct `source_seq`, with one bootstrap exception: the very **first** mask
ever saved in a directory is saved unconditionally (even if the supplier is already "ahead" of the
current frame), so `mask_for_frame`'s "before" lookup has something to find as early as possible in
an episode rather than only catching up once the recorder's own processing drifts back in sync
with a fast supplier. Every saved mask also gets a `masks_meta.jsonl` row (`frame_index,
source_seq, num_instances, mask_area`) and a bounded in-memory ring entry
(`frame_index -> grid_mask`, default last 512, cleared per directory).

### Known limitations / deferred (logged in `design/IMPLEMENTATION_LOG.md`)

- **Video mode** (`camera.save_images_individually: false`) — implemented (`_write_video_frame`,
  ports legacy `_start_new_video`'s `video_<n>.mp4` naming + `clip_length` rollover under
  `Video/`/`Bottom_Video/`), but has no dedicated test in this wave — only the image-mode path is
  exercised by `test_integration_dryrun.py`.
- **`record_only_after_action`'s executor-side hook** — design 02 explicitly drops the legacy
  snapshot-request handshake (`shared_state.frame_index`/`snapshot_cond` round trip); `snapshot()`
  is exposed for a caller to invoke directly, but nothing in this wave's `Executor`/`SessionRunner`
  calls it automatically. Continuous recording (`record_only_after_action: false`) is the primary,
  tested mode.

## `ui/stream.py` — `MJPEGServer`

### Purpose

Debug MJPEG stream: `http://0.0.0.0:<port>/video_feed`, reading `source`'s bottom images via
`source.subscribe('ui', maxsize=2)` — the same `LatestWinsQueue` primitive every other subscriber
uses, replacing legacy's hand-rolled `Queue(maxsize=2)` (`coordinator.py`'s `ui_queue`).

### Public API

`MJPEGServer(source, port, shutdown_event)` — `.start()` (spawns a background thread running
werkzeug's `serve_forever()`), `.stop(timeout=None)`, `.is_running()`.

### Dependency note

Flask/werkzeug are imported **lazily**, inside `start()`/`_build_app()` — never at module import
time. As of this wave, `flask`/`werkzeug` are **not installed** in the `cge` conda env (verified:
`import flask` raises `ModuleNotFoundError`), despite `docs/CONVENTIONS.md` listing `flask` as
present in that environment — logged in `design/IMPLEMENTATION_LOG.md`. Consequently:
- No test in this repo constructs an `MJPEGServer`.
- `main_granular.py --no-ui` (or `ui.enabled: false` in config) skips constructing one entirely,
  and is the default used by every test/example per the hard safety/no-extra-deps posture.
- Installing `flask`/`werkzeug` into the `cge` env would make `--ui`/`ui.enabled: true` usable;
  this wave does not do so (out of scope — no test needs it).

## Threading

- `Recorder` owns exactly one background thread (name `"Recorder"`); `start()`/`retarget()`/
  `pause()`/`stop_recording()`/`snapshot()` are meant to be called from `SessionRunner`'s single
  control thread; `mask_for_frame`/`first_frame_event`/`wait_until_frame_processed` are safe from
  any thread.
- `MJPEGServer` owns one background thread running the werkzeug server; each open HTTP connection's
  frame generator runs on werkzeug's own per-request thread, each with its own `subscribe()` queue.

## Config keys consumed

- `Recorder`: `camera.save_images_individually`, `camera.clip_length`,
  `camera.record_only_after_action`, `camera.fps` (duck-typed off `config_schema.CameraConfig`).
- `MJPEGServer`: `ui.port` (read by the composition root, passed as a plain `int`).

## Porting notes

- `recording.py::Recorder.record`/`_update`/`_capture_frame`/`_save_individual_images` — the FPS
  pacing loop moved to `observation.source.ObservationSource` (Wave 2); this class is a
  *subscriber*, never a poller. Image filenames (`image_top_<f>.jpeg`/`image_bottom_<f>.jpeg`)
  unchanged.
- `recording_seg.py::Recorder._capture_frame`'s `.npy` mask saving — ported into
  `_maybe_save_mask`, keyed by `frame_index`/`source_seq` instead of legacy's `latest_mask_saved`
  boolean flag.
- `recording.py::Recorder.start_new_recording`'s `shared_state.action_tracker.clear()` — ported
  into `_reset_dir_state` (fires on both `start()` and `retarget()`).
- `recording.py::Recorder._start_new_video`/`_start_or_restart_video_writers` — ported into
  `_write_video_frame` (see "Known limitations" above).
- `main_chickpeas.py`'s Flask app + `coordinator.py`'s `ui_queue`/`get_ui_update` -> `MJPEGServer`.

## Testing

`autograsper/tests/test_integration_dryrun.py` exercises `Recorder` end-to-end (images/masks/
states written, frame-index assignment, the mask-finalization race fix) as part of the full
session lifecycle — there is no standalone `test_recording_*.py` in this wave (the recorder has no
meaningful behavior to test in isolation from a real or dry-run `ObservationSource`). `MJPEGServer`
has no test (see "Dependency note" above). Run the integration test:

```
cd /home/alon/Code/cloudgripper-api
/home/alon/anaconda3/envs/cge/bin/python -m pytest autograsper/tests/test_integration_dryrun.py -q
```
