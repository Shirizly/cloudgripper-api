# `autograsper/observation/` — Observation, ObservationSource, DebugSink

Status: **implemented** (Wave 2). Design reference:
[`../design/02_proposed_architecture.md`](../design/02_proposed_architecture.md) §3.2.

## Purpose

The single source of truth for "what does the world look like right now" — one polling thread
per robot connection, publishing immutable snapshots that the recorder, perception providers,
planner, and debug UI (Waves 3-6) all read from. This:

- collapses the legacy pattern of **two independent robot connections** (the grasper's for
  commands, the recorder's for `get_all_states()` polling — design 01 §2) down to one connection
  shared via [`hardware.md`](hardware.md)'s `RobotInterface`, consumed by exactly one poller;
- replaces the mutable, multi-producer `SharedState` blackboard (design 01 §3.1, defect #10 — two
  competing mask producers with different coordinate frames) with immutable, sequence-numbered
  snapshots and an explicit pub/sub API, so "wait for a fresh observation" is a real primitive
  instead of ad hoc `time.sleep(1.5)` calls scattered through `custom_graspers/granular_pusher.py`;
- gives debug artifacts (mask visualizations, tool-mask overlays, etc.) one no-op-by-default sink
  (`DebugSink`) instead of `cv2.imwrite` calls to the current working directory.

## Public API

`autograsper/observation/types.py`:

- `RobotState(x, y, z, rotation, claw)` — frozen dataclass, all fields `Optional[float]`.
  `x`/`y`/`z`/`claw` are robot-normalized `[0, 1]`; `rotation` is bias-corrected degrees (the
  correction is applied one layer down, at the hardware layer — see "Porting notes"). Build one
  with `RobotState.from_legacy_dict(d)`, which accepts the legacy `x_norm`/`y_norm`/`z_norm`/
  `rotation`/`claw_norm` key set (`hardware.RawObservation.robot_state_dict`'s shape) and is
  **tolerant of missing or non-numeric keys**: any gap becomes `None` in the corresponding field,
  with a `logging.warning` naming exactly which key(s) were missing/invalid (never raises — see
  `design/IMPLEMENTATION_LOG.md` for why this was chosen over raising).
- `Observation(seq, frame_index, timestamp, top_image, bottom_image, robot_state)` — frozen
  dataclass, one snapshot from `ObservationSource`.
  - `seq: int` — monotonically increasing, source-assigned (starts at 1).
  - `frame_index: Optional[int]` — recorder-assigned frame number; `None` until a provider is
    registered (`ObservationSource.set_frame_index_provider`, wired by the recorder in Wave 5).
  - `timestamp: float` — local wall-clock seconds, passed through from
    `hardware.RawObservation.timestamp`.
  - `top_image: np.ndarray` — raw top-camera frame, untouched.
  - `bottom_image: np.ndarray` — bottom-camera frame **after** the camera pipeline
    (`perception.frames.CameraPipeline.process_bottom`: fisheye undistort + homography rectify).
    Never the raw camera frame.
  - `robot_state: RobotState`.
  - Both image arrays have `flags.writeable = False` set in `__post_init__` — a best-effort
    read-only guard (see the module docstring: `dataclass(frozen=True)` only prevents rebinding
    the field name, not in-place mutation of the array object it points to). Treat both fields as
    read-only; `arr.copy()` before any in-place edit. Writing into them raises `ValueError:
    assignment destination is read-only`.

`autograsper/observation/source.py`:

- `LatestWinsQueue` — bounded per-subscriber queue with drop-oldest-on-full semantics. Returned by
  `subscribe()`; not constructed directly by consumers.
  - `get(timeout: Optional[float] = None) -> Optional[Observation]` — blocking with timeout;
    returns `None` on timeout instead of raising `queue.Empty`.
  - `qsize() -> int`.
- `ObservationSource(robot, camera, fps, shutdown_event, *, max_consecutive_failures=5)` — the one
  polling loop.
  - `robot` is duck-typed against `hardware.robot_interface.RobotInterface` (only
    `get_all_states()` is called). `camera` is duck-typed against
    `perception.frames.CameraPipeline` (only `process_bottom(raw_bottom)` is called) — neither
    protocol is `runtime_checkable`, matching Wave 1's convention.
  - `start()` — spawns the poll thread (name `"ObservationSource"`, daemon). Raises `RuntimeError`
    if already running.
  - `stop(timeout=None)` — signals the thread to stop and joins it. Does **not** touch
    `shutdown_event`.
  - `is_running() -> bool`.
  - `latest() -> Optional[Observation]` — non-blocking; `None` before the first successful poll.
  - `await_next(after_seq: int, timeout: Optional[float] = None) -> Optional[Observation]` —
    condition-variable wait; returns the first `Observation` with `seq > after_seq`, or `None` on
    timeout (`timeout=None` blocks indefinitely). This is the replacement for both the legacy
    `record_current_state()` snapshot handshake and ad hoc `time.sleep(1.5)` waits: "act, then wait
    for an observation newer than the command completion."
  - `subscribe(name: str, maxsize: int = 2) -> LatestWinsQueue` — register a named subscriber
    (recorder, segmentation worker, UI stream, ...); re-subscribing under an existing name
    replaces the previous queue. `unsubscribe(name)` removes it (no-op if absent).
  - `set_frame_index_provider(provider: Optional[Callable[[], Optional[int]]])` — registered by
    the recorder (Wave 5); called once per poll cycle to stamp `Observation.frame_index`. `None`
    (the default) means every observation's `frame_index` stays `None`.
  - `.failed: bool` / `.failed_event: threading.Event` — see "Failure policy" below.

`autograsper/observation/debug.py`:

- `DebugSink` — no-op by default.
  - `enable(directory)` / `disable()` — start/stop writing artifacts under `directory` (created if
    missing); `enable` resets the filename counter.
  - `enabled: bool` (property).
  - `save(name, image) -> Optional[str]` — writes `<dir>/<counter>_<name>.png` via `cv2.imwrite`;
    returns the path, or `None` if disabled.
  - `save_json(name, obj) -> Optional[str]` — writes `<dir>/<counter>_<name>.json`; returns the
    path, or `None` if disabled.
  - Thread-safe (internal lock); meant to be shared across the observation source, perception
    workers, and planner (Waves 3+).

## Data flow

```
RobotInterface.get_all_states()  (hardware layer, Wave 1)
    │  RawObservation(top_image, bottom_image_raw, robot_state_dict, timestamp)
    ▼
ObservationSource._poll_once()
    │  camera.process_bottom(bottom_image_raw)   (perception.frames.CameraPipeline)
    │  RobotState.from_legacy_dict(robot_state_dict)
    │  frame_index_provider() if registered
    ▼
Observation(seq=N, frame_index, timestamp, top_image, bottom_image, robot_state)
    │
    ├─► latest()               non-blocking read
    ├─► await_next(after_seq)  condition-variable wait, woken on every publish
    └─► subscriber queues       one LatestWinsQueue per subscribe() call, drop-oldest on full
```

## Threading

- Exactly one background thread per `ObservationSource` instance, named `"ObservationSource"`,
  owns every `robot.get_all_states()` / `camera.process_bottom()` call. No other thread should
  call `get_all_states()` on the same robot while a source for it is running (design 02 §3.2:
  "one observation pipeline, one robot connection").
- `latest()`, `await_next()`, `subscribe()`/`unsubscribe()`, `set_frame_index_provider()`, and
  `.failed`/`.failed_event` are safe to call from any thread.
- Publishing an `Observation` is a single critical section under an internal `threading.Condition`
  (`notify_all()` wakes every `await_next()` waiter), followed by pushing into each subscriber's
  independent `queue.Queue` — a slow subscriber can never block the poll loop or any other
  subscriber.
- `DebugSink` has its own internal lock and is safe to share across threads.

## Failure policy

Design 02 §3.5 ("Separated failure policy ... No `shutdown_event.set()` inside behaviors"):
an exception raised by `robot.get_all_states()`, a missing image in the returned
`RawObservation`, or `camera.process_bottom()` during one poll cycle is caught, logged, and
counted. After `max_consecutive_failures` (default 5) **in a row**, `.failed` becomes `True`, the
internal `failed_event` is set, and the poll loop **exits its own thread** — no further polling is
attempted. `shutdown_event` is never touched by `ObservationSource` itself. The session layer
(Wave 5+) is the one that decides what a failed source means (retry with a fresh
`ObservationSource`, abort, human intervention, ...) — it should watch `.failed`/`.failed_event`
alongside its own `shutdown_event`. A successful cycle resets the consecutive-failure counter to
zero.

## Config keys consumed

None directly — `ObservationSource` takes an already-constructed `robot`
(`hardware.robot_interface.RobotInterface`) and `camera`
(`perception.frames.CameraPipeline`, itself built from `config_schema.CameraConfig` — see
[perception.md](perception.md)) plus a plain `fps: float`. The composition root (Wave 6+) is
expected to read `config.camera.fps` and pass it through.

## Porting notes

- `autograsper/recording.py::Recorder.record()` / `Recorder._update()` — the FPS-paced polling
  loop (`perf_counter()`-based timing, one `get_all_states()` call per cycle,
  `shutdown_event.wait(max(0, 1/FPS - elapsed - 0.0002))` pacing — the `0.0002` buffer is kept
  verbatim, "to improve chances of hitting target FPS" per the legacy comment). `Recorder._update`
  also applied fisheye undistortion + homography rectification to the bottom image and subtracted
  the rotation bias from reported state; both of those now live one layer down (Wave 1
  `CloudGripperRobot` for the bias — see `hardware.md`'s "Rotation bias lives here and only here" —
  and `perception.frames.CameraPipeline` for the camera pipeline, invoked from here).
- `autograsper/coordinator.py`'s `ui_queue` (`Queue(maxsize=2)`, drop-oldest: `put_nowait`, on
  `Full` do `get_nowait()` then `put_nowait()` again) → generalized into
  `LatestWinsQueue._put_latest`, one instance per `subscribe()` call instead of one hardcoded UI
  queue — recorder, segmentation worker, and UI stream (Waves 3, 5, 6) each get their own.
- `autograsper/legacy/library/utils.py::manual_control`'s state-dict shape
  (`x_norm`/`y_norm`/`z_norm`/`rotation`/`claw_norm`) → `RobotState.from_legacy_dict`'s expected
  input, unchanged from Wave 1's `RawObservation.robot_state_dict`.
- Debug `cv2.imwrite` calls scattered through library code (e.g.
  `custom_graspers/fence_utils.py::make_tool_mask`/`check_wall_reset_needed`,
  `object_tracker/granular_utils.py::process_image`) → `DebugSink.save`/`.save_json`, no-op unless
  a session directory has been wired in (Waves 3+ take a `DebugSink` and call it instead of
  `cv2.imwrite` directly).
- `ObservationSource` imports `perception.frames.CameraPipeline` even though the informal
  package-layout arrow list in design 02 §2 reads as "`perception` may import `observation`" (not
  the reverse). Design 02 §3.2's prose is explicit ("applies the camera pipeline (undistort +
  homography rectify from `perception.frames`)") and is treated as binding over the more general
  layout diagram, which — unlike its explicit "the planner never imports `hardware` or
  `execution`" rule — does not call out observation/perception as a hard-forbidden direction.
  Logged in `design/IMPLEMENTATION_LOG.md`.

## Testing

`autograsper/tests/test_observation_source.py` — monotonic `seq` at high fps (via `DryRunRobot` +
a pass-through fake camera), `latest()`/`await_next()` semantics (wakes promptly on a new
observation; times out to `None` when none is forthcoming), subscriber drop-oldest behavior,
read-only image arrays, `frame_index` provider stamping, and the consecutive-failure path (both a
raising fake robot and a robot returning no images) reaching `.failed`/`.failed_event` without ever
setting `shutdown_event`. Run:

```
cd /home/alon/Code/cloudgripper-api
/home/alon/anaconda3/envs/cge/bin/python -m pytest autograsper/tests/test_observation_source.py -q
```
