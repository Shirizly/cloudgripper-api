"""`SessionRunner` — single-threaded episode control loop (Wave 5).

Design reference: `autograsper/design/02_proposed_architecture.md` §3.5 (core scope of this
module: "Replaces `DataCollectionCoordinator._monitor_state/_process_messages` and the
`run_grasping` override. Single-threaded control loop ... Ordering guarantee that fixes the
start-signal race ... Failure policy is centralized here").

Porting notes (copied and adapted, not imported — legacy `coordinator.py`/`grasper.py` are frozen
per CONVENTIONS.md):
- `coordinator.py::DataCollectionCoordinator._monitor_state`/`_process_messages`/
  `_on_state_transition` (10 Hz poll + message-queue state machine) -> replaced entirely by
  `_run_loop`'s direct state-dispatch (design 02 §3.5: "Transitions are function calls in one
  loop — no message queue, no polling, no missed states").
- `coordinator.py::_on_active_state`'s directory-creation + recorder-arming-before-`start_event`
  sequence -> `_handle_active`'s "create dirs, `recorder.start()`, **wait for first frame**, wire
  order sink, only then `run_plan`" sequence — this is the exact fix for design 01 §5 defect #8
  (the start-signal race: legacy's subclass bypassed the `start_event` handshake entirely).
- `coordinator.py::_on_state_transition`'s "`new_state == STARTUP and old != STARTUP` ->
  `disable_recording()` + `pause=True` + `sleep(timeout_between_experiments)` + `pause=False`" ->
  `_handle_evaluating`'s "no reset needed" branch (`recorder.pause()` + a shutdown-aware wait) —
  this is the one transition into `STARTUP` legacy's actual reachable state graph ever exercised
  (see this module's own state-diagram note below for why `INTERVENTION -> STARTUP` does not also
  get this pause).
- `custom_graspers/granular_pusher.py::RandomPushGrasper.run_grasping`'s `STARTUP -> {ACTIVE |
  RESETTING}`, `ACTIVE -> STARTUP` (recheck), `RESETTING -> ACTIVE` (direct, no recheck) loop
  shape -> `_run_loop`'s state dispatch table, now with an explicit `EVALUATING` step between
  `ACTIVE` and the `{RESETTING | STARTUP}` decision (design 02 §3.5's diagram) and an explicit
  `INTERVENTION` state (legacy's tool-grip retry loop lived *inside* `startup()` without a visible
  state change; `session.episode.EpisodeState` promotes it to a first-class, externally observable
  state).
- The legacy warn-by-gripper human-intervention loop (`startup()`'s "wait 60s, warn by
  closing/opening the gripper, then `perform_grab_tool()`, re-check, repeat") is already ported
  *inside* `execution.executor.Executor._expand_regrasp_tool` (per `docs/execution.md`) — this
  module's `_handle_intervention` only owns the *retry loop around* one `RegraspTool` primitive
  execution (the wait + loop-until-success-or-shutdown part), not the scripted grasp sequence
  itself.

Failure-policy mapping (design 02 §3.5's typed-exception table, realized in `_handle_startup`/
`_handle_active`/`_handle_resetting`):

| Exception | Where it can occur | Policy |
|---|---|---|
| `ToolLost` | `plan_startup`'s `CheckToolGrip`, `plan_task`'s `PlaceTool`/pushes | -> `INTERVENTION` |
| `UnsafeLower` | task execution (`LowerTool` guard) | replan `plan_task` once; persists -> episode `fail`, evaluate normally (may lead to `RESETTING`) |
| `StaleMaskTimeout` | task execution (freshness gate/guard) | retry once; persists -> episode `fail` |
| `NoGranulesDetected` | `SegPushPlanner.plan_startup`/`plan_task` | abort the whole run |
| `perception.occupancy.PerceptionDegraded` | a directly-called `OccupancyProvider.compute()` (cautious-pipeline mask refresh) | abort the whole run |
| `ObservationSource.failed` | checked at loop boundaries via `_should_finish`-adjacent checks | abort the whole run (see `_handle_active`/`_handle_startup`'s explicit checks) |
| unexpected `Exception` | anywhere | mark episode `fail` (if mid-episode) and abort the whole run |
| `shutdown_event` set / `KeyboardInterrupt` (caught by the composition root, sets `shutdown_event`) | any loop boundary | `FINISHED` |

`status.txt` is written `"fail"` for every aborted/failed episode (design 02 §3.5's "`status.txt`
'fail' on aborted episodes").

Threading: `SessionRunner.run()` is meant to be driven by exactly one thread (its own, if
`start()`ed, or the caller's if `run()` is called directly) — it is the "single-threaded control
loop" design 02 §3.5 describes; all the concurrency in the system lives in `ObservationSource`, the
`Recorder`'s consumer thread, and `SegmentationWorker`, none of which this class synchronizes with
beyond their already-documented thread-safe APIs (`latest()`/`await_next()`/events).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

from autograsper.execution.actions import ActionPhase
from autograsper.execution.errors import StaleMaskTimeout, ToolLost, UnsafeLower
from autograsper.perception.occupancy import PerceptionDegraded
from autograsper.planning.planner import NoGranulesDetected
from autograsper.planning.types import RegraspTool, ToolStatus, WorldState
from autograsper.session.episode import EpisodeState, EpisodeStateMachine
from autograsper.session.storage import (
    OrderSinkRouter,
    OrdersWriter,
    SessionDirs,
    create_session_dirs,
    write_status,
)

logger = logging.getLogger(__name__)


class SessionRunner:
    """Wires every already-constructed component together and owns the episode control loop and
    failure policy (design 02 §3.5). See module docstring for the full porting/failure-policy map.
    """

    def __init__(
        self,
        *,
        robot,
        source,
        planner,
        executor,
        recorder,
        tracker,
        workspace,
        config,
        storage_base_dir: str,
        experiment_name: str,
        shutdown_event: threading.Event,
        order_sink_router: OrderSinkRouter,
        occupancy_source: Optional[Any] = None,
        tool_grip_checker: Optional[Any] = None,
        transition_writer: Optional[Any] = None,
        debug: Optional[Any] = None,
        episode_budget: Optional[int] = None,
        intervention_wait_seconds: float = 60.0,
        max_reset_iterations: int = 8,
        first_frame_timeout: float = 5.0,
        on_state_transition: Optional[Any] = None,
    ) -> None:
        """
        Args:
            robot: `hardware.robot_interface.RobotInterface`-duck-typed (kept for completeness /
                future direct use — every actual command goes through `executor`).
            source: `observation.source.ObservationSource`-duck-typed.
            planner: `planning.planner.Planner`-duck-typed (`RandomPushPlanner`/`SegPushPlanner`).
            executor: `execution.executor.Executor` — constructed with `order_sink=
                order_sink_router` by the caller (composition root); this class redirects the
                router per episode/phase.
            recorder: `recording.recorder.Recorder`.
            tracker: the same `execution.actions.ActionTracker` instance `executor`/`recorder` were
                built with (used to read `get_all_actions()` indirectly via `recorder`; kept here
                for future direct use, e.g. intervention-retry-count escalation into
                `NeedsHumanHelp` — not implemented this wave, see `docs/execution.md`'s note on
                that exception being reserved for the session layer).
            workspace: `planning.workspace.Workspace`.
            config: `config_schema.Config`-duck-typed; reads `.experiment.timeout_between_orders`
                is NOT read here (that's the executor's job) — only
                `.experiment.timeout_between_experiments`.
            storage_base_dir / experiment_name: passed to `session.storage.create_session_dirs`
                every `ACTIVE` entry.
            shutdown_event: the one external abort channel (Ctrl-C/UI), per design 02 §3.5.
            order_sink_router: `session.storage.OrderSinkRouter` the same `executor` was
                constructed with — this class calls `.set(writer.record)`/`.set(None)` per episode
                phase so one long-lived `Executor` can write to a fresh `OrdersWriter` every
                episode without being reconstructed.
            occupancy_source: either a `SegmentationWorker`-shaped object (`.latest()` — the
                segmenter-native pipeline, where occupancy is always continuously fresh) or an
                `OccupancyProvider`-shaped object (`.compute(obs)` only — the cautious
                background-diff pipeline, where this class caches the last manually-computed
                result and refreshes it after every `RefreshMask`-driven `plan_startup`), or `None`
                (`perception.provider == "none"`).
            tool_grip_checker: kept for interface symmetry with the executor's own copy; not
                called directly by this class (the executor calls it as part of `CheckToolGrip`/
                `RegraspTool`'s expansion) — accepted so a future escalation policy (e.g. reading
                `grip_quality` trends) has it available without re-wiring.
            transition_writer: `session.storage.TransitionWriter`, or `None` if
                `storage.emit_transitions_online` is `False`. Registered as an executor completion
                callback by the caller; this class only calls `.finalize()` once per episode (task
                phase only, per design 03 §5) right after `plan_task` execution, before the
                recorder retargets to `restore/` (whose mask ring reset would otherwise make later
                masks unresolvable).
            episode_budget: `None` = run until `shutdown_event` is set (design 02 §3.5:
                "episode loop honoring experiment.episode_budget (None = until shutdown)").
            intervention_wait_seconds: shutdown-aware wait before each `RegraspTool` retry in
                `INTERVENTION` (legacy: a hardcoded 60s; overridable so tests can use ~0).
            max_reset_iterations: cap on `SegPushPlanner`'s `replan_after_each` loop in
                `RESETTING` (design 03 §3.3's "adaptive sweeps" contract, `docs/planning.md`'s
                "Replanning contract" section) — a logged, undictated choice (see
                `design/IMPLEMENTATION_LOG.md`).
            first_frame_timeout: how long to wait for `recorder.first_frame_event` before treating
                a directory as un-recordable (episode marked `fail`, or a logged warning during
                `RESETTING`).
            on_state_transition: optional `Callable[[EpisodeState, EpisodeState], None]` fired on
                every successful `EpisodeState` transition (passthrough to
                `session.episode.EpisodeStateMachine`) — a test/observability hook, not read by
                any behavior in this class itself.
        """
        self._robot = robot
        self._source = source
        self._planner = planner
        self._executor = executor
        self._recorder = recorder
        self._tracker = tracker
        self._workspace = workspace
        self._config = config
        self._storage_base_dir = storage_base_dir
        self._experiment_name = experiment_name
        self._shutdown_event = shutdown_event
        self._order_sink_router = order_sink_router
        self._occupancy_source = occupancy_source
        self._tool_grip_checker = tool_grip_checker
        self._transition_writer = transition_writer
        self._debug = debug
        self._episode_budget = episode_budget
        self._intervention_wait_seconds = intervention_wait_seconds
        self._max_reset_iterations = max_reset_iterations
        self._first_frame_timeout = first_frame_timeout
        self._timeout_between_experiments = float(
            getattr(getattr(config, "experiment", object()), "timeout_between_experiments", 0.0)
        )

        self._current_occupancy: Optional[Any] = None
        self._tool_status = ToolStatus(held=None, grip_quality=None)
        self._current_dirs: Optional[SessionDirs] = None
        self._episode_status = "fail"
        self._episodes_run = 0

        self._sm = EpisodeStateMachine(EpisodeState.STARTUP, on_transition=on_state_transition)
        self._thread: Optional[threading.Thread] = None
        self._done_event = threading.Event()

        self._handlers = {
            EpisodeState.STARTUP: self._handle_startup,
            EpisodeState.INTERVENTION: self._handle_intervention,
            EpisodeState.ACTIVE: self._handle_active,
            EpisodeState.EVALUATING: self._handle_evaluating,
            EpisodeState.RESETTING: self._handle_resetting,
        }

    # -- public API ---------------------------------------------------------------

    @property
    def state(self) -> EpisodeState:
        return self._sm.state

    @property
    def episodes_run(self) -> int:
        return self._episodes_run

    def world_state(self) -> WorldState:
        """Build a `WorldState` from the latest `Observation` + whichever occupancy source is
        configured + the last-known tool status + the static `Workspace`."""
        obs = self._source.latest()
        if obs is None:
            obs = self._source.await_next(0, timeout=self._first_frame_timeout)
        if obs is None:
            raise RuntimeError("SessionRunner.world_state: no Observation available from source")
        return WorldState(
            obs=obs, occupancy=self._get_occupancy(), tool=self._tool_status, workspace=self._workspace
        )

    def run(self) -> None:
        """Blocking episode loop. Call directly, or via `start()` to run it on a dedicated
        thread."""
        try:
            self._run_loop()
        except Exception:
            logger.exception("SessionRunner: unhandled exception escaped the run loop")
            try:
                self._sm.transition(EpisodeState.ABORTED)
            except Exception:
                pass
        finally:
            self._done_event.set()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("SessionRunner already started")
        self._thread = threading.Thread(target=self.run, name="SessionRunner", daemon=True)
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        else:
            self._done_event.wait(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        """Request a clean stop (the external abort channel — design 02 §3.5). Does not block;
        call `join()` afterward."""
        self._shutdown_event.set()

    # -- main loop ------------------------------------------------------------------

    def _run_loop(self) -> None:
        while True:
            if self._sm.is_terminal():
                return
            if self._should_finish():
                self._sm.transition(EpisodeState.FINISHED)
                continue
            handler = self._handlers.get(self._sm.state)
            if handler is None:
                logger.error("SessionRunner: no handler for state %s; aborting", self._sm.state)
                self._sm.transition(EpisodeState.ABORTED)
                continue
            handler()

    def _should_finish(self) -> bool:
        if self._shutdown_event.is_set():
            return True
        if self._episode_budget is not None and self._episodes_run >= self._episode_budget:
            return True
        source_failed = getattr(self._source, "failed", False)
        if source_failed:
            logger.error("SessionRunner: ObservationSource reports failed; finishing run")
            return True
        return False

    def _wait_with_shutdown(self, seconds: float) -> None:
        self._shutdown_event.wait(timeout=max(0.0, seconds))

    def _latest_seq(self) -> int:
        obs = self._source.latest()
        return obs.seq if obs is not None else 0

    # -- occupancy plumbing -----------------------------------------------------------

    def _get_occupancy(self) -> Optional[Any]:
        if self._occupancy_source is None:
            return self._current_occupancy
        if hasattr(self._occupancy_source, "latest"):
            return self._occupancy_source.latest()
        return self._current_occupancy

    def _refresh_occupancy_from_provider(self, obs) -> None:
        """After a `RefreshMask`-driven move (cautious/background-diff pipeline only — a `latest()`
        -shaped `occupancy_source` already keeps itself fresh in the background and this is a
        no-op for it), compute a fresh occupancy result from `obs` and cache it."""
        if self._occupancy_source is None or hasattr(self._occupancy_source, "latest"):
            return
        if not hasattr(self._occupancy_source, "compute"):
            return
        try:
            self._current_occupancy = self._occupancy_source.compute(obs)
        except PerceptionDegraded:
            logger.warning("SessionRunner: occupancy provider raised PerceptionDegraded on refresh")
        except Exception:
            logger.exception("SessionRunner: occupancy_source.compute() raised")

    # -- STARTUP ------------------------------------------------------------------

    def _handle_startup(self) -> None:
        pre_seq = self._latest_seq()
        try:
            plan = self._planner.plan_startup(self.world_state())
            self._executor.run_plan(plan, ActionPhase.STARTUP)
        except ToolLost as exc:
            logger.warning(
                "SessionRunner: tool-grip check failed at startup (quality=%s) -> INTERVENTION",
                exc.grip_quality,
            )
            self._tool_status = ToolStatus(held=False, grip_quality=exc.grip_quality)
            self._sm.transition(EpisodeState.INTERVENTION)
            return
        except NoGranulesDetected:
            logger.error("SessionRunner: no granular material detected at startup; aborting run")
            self._sm.transition(EpisodeState.ABORTED)
            return
        except PerceptionDegraded:
            logger.error("SessionRunner: perception degraded at startup; aborting run")
            self._sm.transition(EpisodeState.ABORTED)
            return
        except Exception:
            logger.exception("SessionRunner: unexpected error during startup; aborting run")
            self._sm.transition(EpisodeState.ABORTED)
            return

        self._tool_status = ToolStatus(held=True, grip_quality=None)

        fresh_obs = self._source.await_next(pre_seq, timeout=self._first_frame_timeout)
        if fresh_obs is None:
            fresh_obs = self._source.latest()
        if fresh_obs is not None:
            self._refresh_occupancy_from_provider(fresh_obs)

        if self._shutdown_event.is_set():
            return  # next loop iteration's _should_finish() moves to FINISHED

        try:
            needs_reset = self._planner.needs_reset(self.world_state())
        except Exception:
            logger.exception("SessionRunner: needs_reset raised after startup; aborting run")
            self._sm.transition(EpisodeState.ABORTED)
            return

        self._sm.transition(EpisodeState.RESETTING if needs_reset else EpisodeState.ACTIVE)

    # -- INTERVENTION ---------------------------------------------------------------

    def _handle_intervention(self) -> None:
        """Wait, then retry one `RegraspTool`, until it succeeds or `shutdown_event` is set
        (design 02 §3.5: `Intervention -> Startup: human confirmed / regrasp ok`). The scripted
        grasp + post-grasp grip re-check are already ported inside `RegraspTool`'s executor
        expansion (see module docstring) — this loop only owns the wait + retry-until-success
        shape."""
        while not self._shutdown_event.is_set():
            self._wait_with_shutdown(self._intervention_wait_seconds)
            if self._shutdown_event.is_set():
                break
            try:
                self._executor.execute_primitive(RegraspTool(), ActionPhase.STARTUP)
            except ToolLost as exc:
                self._tool_status = ToolStatus(held=False, grip_quality=exc.grip_quality)
                logger.warning(
                    "SessionRunner: RegraspTool did not recover grip (quality=%.2f); retrying",
                    exc.grip_quality,
                )
                continue
            except Exception:
                logger.exception("SessionRunner: unexpected error during RegraspTool intervention; aborting run")
                self._sm.transition(EpisodeState.ABORTED)
                return
            self._tool_status = ToolStatus(held=True, grip_quality=None)
            self._sm.transition(EpisodeState.STARTUP)
            return
        # shutdown observed mid-wait/retry: leave state as-is, _should_finish() handles it next.

    # -- ACTIVE ---------------------------------------------------------------------

    def _handle_active(self) -> None:
        dirs = create_session_dirs(self._storage_base_dir, self._experiment_name)
        self._current_dirs = dirs

        # Ordering guarantee (design 02 §3.5): arm the recorder and wait for its first frame
        # BEFORE any order of this episode is sent.
        self._recorder.start(dirs.task_dir)
        if not self._recorder.first_frame_event.wait(timeout=self._first_frame_timeout):
            logger.error(
                "SessionRunner: recorder produced no frames in task/ within %.1fs; failing episode",
                self._first_frame_timeout,
            )
            self._episode_status = "fail"
            self._sm.transition(EpisodeState.EVALUATING)
            return

        orders_writer = OrdersWriter(dirs.task_dir)
        self._order_sink_router.set(orders_writer.record)

        status = "success"
        unsafe_lower_retried = False
        stale_retried = False
        while True:
            try:
                plan = self._planner.plan_task(self.world_state())
                self._executor.run_plan(plan, ActionPhase.TASK)
                break
            except UnsafeLower:
                if unsafe_lower_retried:
                    logger.error("SessionRunner: UnsafeLower persisted after one replan; failing episode")
                    status = "fail"
                    break
                logger.warning("SessionRunner: UnsafeLower during task; replanning once")
                unsafe_lower_retried = True
                continue
            except StaleMaskTimeout:
                if stale_retried:
                    logger.error("SessionRunner: StaleMaskTimeout persisted after one retry; failing episode")
                    status = "fail"
                    break
                logger.warning("SessionRunner: StaleMaskTimeout during task; retrying once")
                stale_retried = True
                continue
            except ToolLost as exc:
                self._tool_status = ToolStatus(held=False, grip_quality=exc.grip_quality)
                self._episode_status = "fail"
                self._finalize_episode_writers(dirs, orders_writer)
                self._sm.transition(EpisodeState.INTERVENTION)
                return
            except NoGranulesDetected:
                logger.error("SessionRunner: no granules detected during task; aborting run")
                self._episode_status = "fail"
                self._finalize_episode_writers(dirs, orders_writer)
                self._sm.transition(EpisodeState.ABORTED)
                return
            except PerceptionDegraded:
                logger.error("SessionRunner: perception degraded during task; aborting run")
                self._episode_status = "fail"
                self._finalize_episode_writers(dirs, orders_writer)
                self._sm.transition(EpisodeState.ABORTED)
                return
            except Exception:
                logger.exception("SessionRunner: unexpected error during task execution; aborting run")
                self._episode_status = "fail"
                self._finalize_episode_writers(dirs, orders_writer)
                self._sm.transition(EpisodeState.ABORTED)
                return

        self._episode_status = status
        self._finalize_episode_writers(dirs, orders_writer)
        self._sm.transition(EpisodeState.EVALUATING)

    def _finalize_episode_writers(self, dirs: SessionDirs, orders_writer: OrdersWriter) -> None:
        orders_writer.finalize()
        self._order_sink_router.set(None)
        if self._transition_writer is not None:
            # `Executor.execute_primitive`'s own end-of-primitive `source.await_next(...)` only
            # guarantees the last Observation exists -- not that the recorder's independent
            # consumer thread has already written/saved its frame (images/states/mask) yet. Wait
            # for it (bounded) before asking the TransitionWriter to resolve masks by frame index,
            # closing that race.
            last_obs = self._source.latest()
            frame_index = last_obs.frame_index if last_obs is not None else None
            if not self._recorder.wait_until_frame_processed(frame_index, timeout=self._first_frame_timeout):
                logger.warning(
                    "SessionRunner: recorder did not catch up to frame_index=%s within %.1fs before "
                    "finalizing transitions; some masks may be unresolvable",
                    frame_index,
                    self._first_frame_timeout,
                )
            try:
                self._transition_writer.finalize(
                    os.path.join(dirs.session_dir, "transitions"), os.path.basename(dirs.session_dir)
                )
            except Exception:
                logger.exception("SessionRunner: TransitionWriter.finalize raised")

    # -- EVALUATING -------------------------------------------------------------------

    def _handle_evaluating(self) -> None:
        assert self._current_dirs is not None
        write_status(self._current_dirs.session_dir, self._episode_status)
        self._episodes_run += 1

        if self._should_finish():
            self._sm.transition(EpisodeState.FINISHED)
            return

        try:
            needs_reset = self._planner.needs_reset(self.world_state())
        except Exception:
            logger.exception("SessionRunner: needs_reset raised while evaluating; aborting run")
            self._sm.transition(EpisodeState.ABORTED)
            return

        if needs_reset:
            self._sm.transition(EpisodeState.RESETTING)
        else:
            # Legacy's one reachable "-> STARTUP" pause (coordinator.py::_on_state_transition):
            # disable recording and wait out `timeout_between_experiments` before the next
            # episode's tool-grip recheck.
            self._recorder.pause()
            self._wait_with_shutdown(self._timeout_between_experiments)
            self._sm.transition(EpisodeState.STARTUP)

    # -- RESETTING --------------------------------------------------------------------

    def _handle_resetting(self) -> None:
        assert self._current_dirs is not None
        self._recorder.retarget(self._current_dirs.restore_dir)
        if not self._recorder.first_frame_event.wait(timeout=self._first_frame_timeout):
            logger.warning(
                "SessionRunner: recorder produced no frames in restore/ within %.1fs", self._first_frame_timeout
            )

        orders_writer = OrdersWriter(self._current_dirs.restore_dir)
        self._order_sink_router.set(orders_writer.record)

        iterations = 0
        while iterations < self._max_reset_iterations:
            if self._shutdown_event.is_set():
                break
            try:
                plan = self._planner.plan_reset(self.world_state())
            except Exception:
                logger.exception("SessionRunner: plan_reset raised; aborting run")
                orders_writer.finalize()
                self._order_sink_router.set(None)
                self._sm.transition(EpisodeState.ABORTED)
                return

            if not plan.primitives:
                break

            try:
                self._executor.run_plan(plan, ActionPhase.RESET)
            except Exception:
                logger.exception("SessionRunner: error executing reset plan; aborting run")
                orders_writer.finalize()
                self._order_sink_router.set(None)
                self._sm.transition(EpisodeState.ABORTED)
                return

            iterations += 1
            if not plan.meta.get("replan_after_each"):
                break  # batch planner (RandomPushPlanner): one call handles every needy wall.
            try:
                if not self._planner.needs_reset(self.world_state()):
                    break
            except Exception:
                logger.exception("SessionRunner: needs_reset raised mid-reset; aborting run")
                orders_writer.finalize()
                self._order_sink_router.set(None)
                self._sm.transition(EpisodeState.ABORTED)
                return

        orders_writer.finalize()
        self._order_sink_router.set(None)
        self._sm.transition(EpisodeState.ACTIVE)
