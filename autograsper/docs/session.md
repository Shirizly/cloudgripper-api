# `autograsper/session/` — EpisodeStateMachine, SessionRunner, storage writers

Status: **implemented** (Wave 5). Design reference:
[`../design/02_proposed_architecture.md`](../design/02_proposed_architecture.md) §3.5 (core scope
of this wave), §3.6 (dataset writers), §6; [`../design/03_segmenter_native_design.md`](../design/03_segmenter_native_design.md)
§4-5 (masks_meta.jsonl, online transition emission).

## Purpose

The layer that turns the pure `planning`/`execution` machinery into a running data-collection
process: an explicit, validated episode lifecycle (`session/episode.py`), the single-threaded
control loop that drives it and centralizes failure policy (`session/coordinator.py`), and the
dataset writers that turn `Executor`/`Recorder` output into the on-disk dataset contract
(`session/storage.py`). Replaces `coordinator.py::DataCollectionCoordinator` (10 Hz poll +
message-queue state machine) and `grasper.py`'s activity state machine / `file_manager.py`.

## Public API

### `session/episode.py`

- `EpisodeState` — `{STARTUP, ACTIVE, EVALUATING, RESETTING, INTERVENTION, FINISHED, ABORTED}`.
- `EpisodeStateMachine(initial=STARTUP, on_transition=None)` — `.state`, `.is_terminal()`,
  `.transition(new_state)` (raises `InvalidTransition` for any edge not in design 02 §3.5's
  diagram; self-transitions are always allowed; `FINISHED`/`ABORTED` are reachable from any
  non-terminal state and are themselves terminal — nothing transitions out of them).
  `on_transition(old, new)` fires on every successful non-self transition; a raising callback is
  logged and does not affect the transition.

### `session/storage.py`

- `SessionDirs(session_dir, task_dir, restore_dir)` + `create_session_dirs(base_dir,
  experiment_name) -> SessionDirs` — ports `file_manager.py::FileManager.get_session_dirs`'s "next
  integer directory" numbering; creates `task/`/`restore/` eagerly, `Images/`/`Bottom_Images/`/
  `Masks/` are created lazily by the recorder.
- `write_status(session_dir, "success"|"fail") -> path`.
- `JsonlWriter(path)` — generic append-mode `.jsonl` sink (`append(row)`, flushed per row;
  `.rows()` returns everything appended so far, for a `.json`-array `finalize()` step).
- `StatesWriter(dir_path)` — `states.jsonl` (streamed) + `states.json` (legacy array, at
  `finalize()`). `record(robot_state, time, frame_index, action=None)` row shape ported verbatim
  from `recording.py::Recorder.save_state`.
- `OrdersWriter(dir_path)` — `orders.jsonl` + `orders.json`, same pattern; `record(order_record)`
  takes exactly the dict shape `execution.executor.Executor`'s `order_sink` contract produces.
- `OrderSinkRouter()` — mutable indirection (`.set(sink)`/`__call__`) so one long-lived `Executor`
  (constructed once for the whole run) can be redirected to a fresh `OrdersWriter` every
  episode/phase without being reconstructed. Pass an instance as `Executor(order_sink=router)`.
- `ActionsWriter.finalize(dir_path, actions) -> Optional[path]` — `actions.json`
  (`{"total_actions": N, "actions": [...]}`), skipped entirely when `actions` is empty (matches
  legacy).
- `MasksMetaWriter(dir_path)` — `masks_meta.jsonl`, one row per saved mask:
  `{frame_index, source_seq, num_instances, mask_area}` (design 03 §4).
- `TransitionWriter(frames, grid_height, grid_width, tool_size_px, mask_for_frame,
  experiment_meta=None)` — online emission of the `RealData` transition-dataset format (design 03
  §5, `MD files/transition_dataset_design.md`). `register_completion_callback`-compatible
  `on_action_completed(action)` filters for completed top-level `Push` actions
  (`action.parent_id is None and action.action_type is ActionType.MOVE_XY and
  action.is_planar_2d`); `finalize(transitions_dir, episode_id) -> Optional[(data_path,
  config_path)]` writes `_{episode_id}_data.pt` (`torch.save`) + `_{episode_id}_config.yaml`
  once per episode (task phase only) and clears its pending-push buffer. See
  [dataset_formats.md](dataset_formats.md) for the exact tensor/YAML shapes.

### `session/coordinator.py`

- `SessionRunner(*, robot, source, planner, executor, recorder, tracker, workspace, config,
  storage_base_dir, experiment_name, shutdown_event, order_sink_router, occupancy_source=None,
  tool_grip_checker=None, transition_writer=None, debug=None, episode_budget=None,
  intervention_wait_seconds=60.0, max_reset_iterations=8, first_frame_timeout=5.0,
  on_state_transition=None)` — every component is constructed and wired by the caller (the
  composition root, `main_granular.py`); this class owns only the control loop and failure policy.
  - `.world_state() -> WorldState` — `source.latest()` (blocking briefly via `await_next` if
    nothing has been published yet) + occupancy (see below) + last-known `ToolStatus` +
    `workspace`.
  - `.run()` — blocking episode loop; honors `episode_budget` (`None` = until `shutdown_event`).
  - `.start()`/`.join(timeout=None)`/`.is_alive()`/`.stop()` — run `.run()` on a dedicated thread.
  - `.state` / `.episodes_run` — inspection.

## Occupancy plumbing (`occupancy_source`)

One constructor argument serves both perception pipelines (design 03 §1's "worker_or_provider"),
distinguished by duck typing:

- **Segmenter-native** (`SegmentationWorker`-shaped, has `.latest()`): `world_state()` always calls
  `occupancy_source.latest()` — continuously fresh, no runner-side bookkeeping needed.
- **Cautious/background-diff** (`OccupancyProvider`-shaped, has only `.compute(obs)`):
  `world_state()` returns the last value the runner itself computed and cached
  (`_current_occupancy`); the runner recomputes it via `_refresh_occupancy_from_provider(obs)`
  right after `plan_startup`'s `RefreshMask`-driven move-aside completes (a fresh observation is
  captured with `source.await_next(pre_startup_seq, ...)`, then `.compute(obs)` is called and
  cached — this is the "provider" half of design 03 §5's "cautious pipeline: last computed
  provider result held by runner" instruction).
- `None` (`perception.provider == "none"`): `world_state().occupancy` stays `None` forever.

## Episode control loop (`_run_loop`)

A plain dispatch table keyed by `EpisodeState`, looping until `FINISHED`/`ABORTED`:

```
while not terminal:
    if shutdown_event.is_set() or episode_budget reached or source.failed:
        -> FINISHED
    dispatch handler for current state
```

- **`_handle_startup`**: runs `planner.plan_startup(w)` via the executor (`ActionPhase.STARTUP`);
  on success, refreshes cautious-pipeline occupancy (see above), then re-derives `WorldState` and
  transitions to `RESETTING` (`needs_reset`) or `ACTIVE` (ready). `ToolLost` -> `INTERVENTION`;
  `NoGranulesDetected`/`PerceptionDegraded`/unexpected exception -> `ABORTED`.
- **`_handle_intervention`**: shutdown-aware wait (`intervention_wait_seconds`, default 60s,
  tiny in tests) then one `RegraspTool` primitive execution, looping until it succeeds (no
  `ToolLost`) or `shutdown_event` is set. The scripted grasp sequence + post-grasp grip re-check
  are already ported inside `RegraspTool`'s executor expansion (see
  [execution.md](execution.md)) — this loop only owns the wait/retry-until-success shape (the
  legacy "warn by gripper" human-intervention loop). Success -> `STARTUP`.
- **`_handle_active`**: **ordering guarantee** (design 02 §3.5, fixes design 01 §5 defect #8):
  (1) `session.storage.create_session_dirs`, (2) `recorder.start(task_dir)` and **wait for
  `recorder.first_frame_event`** (bounded by `first_frame_timeout`), (3) wire a fresh
  `OrdersWriter` through `order_sink_router`, **only then** (4) `planner.plan_task(w)` +
  `executor.run_plan(..., ActionPhase.TASK)`. Failure policy: `UnsafeLower` -> replan `plan_task`
  once (persists -> episode `fail`); `StaleMaskTimeout` -> retry once (persists -> episode
  `fail`); `ToolLost` -> episode `fail`, finalize writers, `INTERVENTION`; `NoGranulesDetected`/
  `PerceptionDegraded`/unexpected exception -> episode `fail`, finalize writers, `ABORTED`. On any
  outcome, finalizes `OrdersWriter` + (if configured) `TransitionWriter.finalize()` into
  `session_dir/transitions/` — waiting first (`recorder.wait_until_frame_processed(...)`, see
  below) for the recorder to have durably processed the plan's last frame, then -> `EVALUATING`.
- **`_handle_evaluating`**: `write_status(session_dir, status)`; re-derives `WorldState` and goes
  to `RESETTING` (`needs_reset`) or, "next episode": `recorder.pause()` +
  `timeout_between_experiments` (shutdown-aware) + -> `STARTUP` (the one legacy-reachable
  `-> STARTUP` pause, `coordinator.py::_on_state_transition`'s `new_state == STARTUP` branch).
- **`_handle_resetting`**: `recorder.retarget(restore_dir)`, then loops `plan_reset` +
  `executor.run_plan(..., ActionPhase.RESET)` up to `max_reset_iterations` (default 8, an
  undictated but logged choice), honoring `Plan.meta["replan_after_each"]` (`SegPushPlanner`'s
  contract, see [planning.md](planning.md)'s "Replanning contract"): re-derives `WorldState` and
  calls `plan_reset` again until `needs_reset()` is `False`, an empty plan comes back, or the
  iteration cap is hit. Always -> `ACTIVE` on completion (never back through `STARTUP` — matches
  legacy's `reset_task()` -> `perform_task()` transition, no re-check).

### The mask-finalization race (and its fix)

`Executor.execute_primitive`'s own end-of-primitive `source.await_next(...)` only guarantees the
last `Observation` of a plan *exists* — it says nothing about whether `Recorder`'s independent
consumer thread has already processed it (written the image/state row/mask). Calling
`TransitionWriter.finalize()` immediately after `run_plan()` returns can therefore race ahead of
the recorder and find the last push's "after" mask unresolvable. Fixed with
`Recorder.wait_until_frame_processed(frame_index, timeout)` (a condition variable the recorder's
consumer thread notifies after every `_capture()`): `_handle_active` calls it (bounded by
`first_frame_timeout`) right before `TransitionWriter.finalize()`.

## Failure-policy table

| Exception | Raised from | Policy |
|---|---|---|
| `execution.errors.ToolLost` | `CheckToolGrip`/`RegraspTool` grip checks (startup or task) | `-> INTERVENTION` |
| `execution.errors.UnsafeLower` | task execution (`LowerTool` guard) | replan `plan_task` once; persists -> episode `fail` |
| `execution.errors.StaleMaskTimeout` | task execution (freshness gate/guard) | retry once; persists -> episode `fail` |
| `planning.planner.NoGranulesDetected` | `SegPushPlanner.plan_startup`/`plan_task` | abort the whole run |
| `perception.occupancy.PerceptionDegraded` | a directly-called `OccupancyProvider.compute()` | abort the whole run |
| `ObservationSource.failed` | checked at every loop-boundary via `_should_finish` | finish the run (`FINISHED`, not `ABORTED` — a clean stop) |
| unexpected `Exception` | anywhere | episode `fail` (if mid-episode) + abort the whole run |
| `shutdown_event` set / `KeyboardInterrupt` (composition root) | any loop boundary | `FINISHED` |

`status.txt` is `"fail"` for every aborted/failed episode.

## `recording/` — `Recorder` (pure sink)

### Purpose

Subscribes to `ObservationSource`; writes frames/states/masks for whichever directory the session
runner points it at; assigns frame indices. No robot connection, camera pipeline, segmentation, or
pacing decisions of its own (design 02 §3.6). See [recording.md](recording.md) for the full API.

## `ui/` — `MJPEGServer`

Thread running a werkzeug server, streaming `source`'s bottom images as MJPEG. Flask/werkzeug are
imported lazily (not installed in this repo's `cge` conda env as of this wave, despite
`docs/CONVENTIONS.md` listing `flask` as present — logged in `design/IMPLEMENTATION_LOG.md`); no
test constructs one. `main_granular.py --no-ui` (or `ui.enabled: false`) skips it entirely.

## Threading

- `SessionRunner.run()` is the single control thread (design 02 §3.5) — it drives `Executor`,
  reads `ObservationSource`/`Recorder`/occupancy-source APIs, and is the only writer of
  `EpisodeStateMachine`'s state.
- `Recorder` owns one consumer thread (name `"Recorder"`) over `source.subscribe('recorder',
  maxsize=8)`; `start()`/`retarget()`/`pause()`/`stop_recording()`/`snapshot()` are meant to be
  called from `SessionRunner`'s thread; `mask_for_frame`/`first_frame_event`/
  `wait_until_frame_processed` are safe from any thread.
- `session.storage` writers are single-writer (whichever thread constructs/uses them — the
  recorder's consumer thread for `StatesWriter`/`MasksMetaWriter`, `SessionRunner`'s thread for
  `OrdersWriter`/`TransitionWriter`/`ActionsWriter`); `OrderSinkRouter` is the one exception,
  documented safe to `set()`/call from different threads.

## Config keys consumed

- `SessionRunner`: `config.experiment.timeout_between_experiments` (between-episode pause).
- `Recorder`: `config.camera.save_images_individually`/`.clip_length`/
  `.record_only_after_action`/`.fps`.
- `create_session_dirs`: `config.storage.base_dir`, `config.experiment.name`.
- `TransitionWriter`: `config.perception.grid.height`/`.width`, `config.storage.
  tool_size_px_for_transitions`; only constructed at all when `config.storage.
  emit_transitions_online` is `True`.
- Everything else (`workspace.*`, `perception.*`, `tool_check.*`) is baked into the injected
  `planner`/`executor`/`workspace`/`tool_grip_checker` objects by the composition root
  (`main_granular.py`), not read directly by this layer.

## Porting notes

See each module's own docstring for the full per-method legacy call-site map; highlights:

- `coordinator.py::DataCollectionCoordinator._monitor_state`/`_process_messages`/
  `_on_state_transition` (10 Hz poll + message queue) -> `SessionRunner._run_loop`'s direct
  state-dispatch (design 02 §3.5's stated goal: "no message queue, no polling, no missed states").
- `coordinator.py::_on_active_state`'s directory-creation-then-`start_event` sequence ->
  `_handle_active`'s "create dirs, arm recorder, wait for first frame, wire order sink, only then
  run_plan" — the design 01 §5 defect #8 fix.
- `file_manager.py::FileManager` -> `session/storage.py`'s `create_session_dirs`/writers.
- `recording.py::Recorder.save_state`/`save_action_summary` -> `StatesWriter`/`ActionsWriter`
  (row/document shapes unchanged; O(n²) per-frame rewrite replaced by jsonl-then-finalize).
- `grasper.py::RobotActivity`/`RandomPushGrasper.run_grasping`'s loop shape ->
  `session/episode.py::EpisodeState`/`EpisodeStateMachine` + `SessionRunner`'s dispatch table.

## Testing

`autograsper/tests/test_integration_dryrun.py` — full end-to-end `DryRunRobot` run (2 episodes:
directory layout, `states.json`/`.jsonl` consistency, `orders.json` legacy+additive keys,
`actions.json` parent/child actions, `status.txt`, online transition emission shapes/dtypes) and a
scripted `ToolLost -> INTERVENTION -> recovery` path. Run:

```
cd /home/alon/Code/cloudgripper-api
/home/alon/anaconda3/envs/cge/bin/python -m pytest autograsper/tests/test_integration_dryrun.py -q
```
