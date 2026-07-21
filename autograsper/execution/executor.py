"""`Executor` — the only component that both talks to the robot and writes actions (Wave 4).

Design reference: `autograsper/design/02_proposed_architecture.md` §3.4 (`Executor` — core scope
of this module) and §3.3's primitive table; `autograsper/design/03_segmenter_native_design.md`
§3.2 (freshness rule), §3.3 (`LowerTool`'s real-time guard), §5 (what a completed `Push` action
must carry for online transition emission).

Porting notes (source legacy files, copied and adapted per CONVENTIONS.md — not imported):
- `autograsper/grasper.py::AutograsperBase.queue_orders`/`execute_order`/`action_type_from_order`/
  `action_details_from_order`/`start_action`/`end_action` — the per-order action-tracking +
  `time_between_orders` pacing shape this module's `_send_order` ports, extended with the
  parent/child nesting `execution.actions.ActionTracker` now supports (one top-level primitive
  action wrapping N per-order child actions, instead of legacy's single flat action per order).
  Legacy's `if action_type is not ActionType.GRIPPER_CLOSE: start_action(...)` (skipping action
  tracking for gripper orders) is **not** carried over — every order here becomes a child action
  (see `_child_action_info`'s `ActionType.GRIPPER_OPEN`/`GRIPPER_CLOSE` split by opening value)
  since the task's `RegraspTool` port needs its two gripper orders tracked like any other. Logged
  in `design/IMPLEMENTATION_LOG.md`.
- `autograsper/custom_graspers/granular_pusher.py::perform_task` (clearance raise + placement
  approach), `sweep_wall`/`sweep` (targeted/fallback sweep choreography, verbatim order sequences
  and constants — `sweep_height +/- 0.02/0.01`, 3-step `already_at_wall` dance), `perform_grab_tool`
  (scripted rack grasp sequence), `update_mask_and_process` (move-aside choreography), `startup`
  (tool-grip inspection pose) — each ported into the matching `_expand_*` method below; see each
  method's docstring for the exact legacy call site.
- `autograsper/library/utils.py::execute_order`'s `np.clip`/rotation int-cast — already ported into
  `hardware.robot_interface.Order.validate()` (Wave 1); this module does not repeat that logic, it
  calls `Order(...).validate()` once per order before dispatch.
- `autograsper/library/utils.py::write_order`'s `orders.json` row shape (`order_type`,
  `order_value`, `time`) — ported into `_emit_order_record`'s dict, with `frame_index` and
  `robot_reported_time` added (additive; legacy tooling reading the three original keys keeps
  working). Written through an injected `order_sink` callable rather than an unbounded
  read-modify-write JSON file (Wave 5's `session.storage` owns the actual sink/persistence; this
  module only calls it).

Primitive -> top-level `ActionType` mapping (decision, logged in `design/IMPLEMENTATION_LOG.md`):
legacy's dataset consumers expect `ActionType` values from the existing enum (unchanged, per
`execution.actions`'s porting notes) — adding new members (e.g. `PLACE_TOOL`) would be a dataset
schema change out of scope for this wave. Instead: `Push` (the one primitive design 02 §3.3
explicitly calls "one `is_planar_2d` action ⇒ maps 1:1 to a dataset transition") maps to
`ActionType.MOVE_XY` with `is_planar_2d=True`; `SweepWall` maps to `ActionType.SWEEP` (already a
legacy member for exactly this); every other primitive (`MoveTo`, `PlaceTool`, `LowerTool`,
`RegraspTool`, `RefreshMask`, `CheckToolGrip`) maps to `ActionType.OTHER`. Each top-level action's
`description` is `primitive.describe()` and `action_details` is the primitive's own dataclass
fields (`dataclasses.asdict(primitive)`) — this is where `Push`'s start/end xy, angle, and height
live (design 03 §5's transition-writer needs); the surrounding `Action.start_frame`/`end_frame`/
`start_robot_state`/`end_robot_state` (already present on every `Action`, no duplication needed)
supply the "start/end robot state AND frames" half of that same requirement.

Freshness/guard design decisions (logged in `design/IMPLEMENTATION_LOG.md`):
- A generic freshness gate (`_enforce_freshness_gate`) runs before *every* primitive whose
  `FreshnessPolicy.requires_fresh()` is `True` and an `occupancy_supplier` is configured: it awaits
  a result with `source_seq >= source.latest().seq`, raising `StaleMaskTimeout` on timeout. This
  covers `PlaceTool`/`SweepWall` (categories `"placement"`/`"wall_check"`) as well as a standalone
  `LowerTool` (category `"lower_tool"`).
- Independently, `_expand_lower_tool` always re-checks (fresh-if-required-by-policy, else
  best-effort `latest()`) and runs the actual `safety.tool_footprint_clear` guard at the *current*
  commanded pose immediately before sending `MOVE_Z` down, whenever `guarded=True` and an
  `occupancy_supplier` is configured — this runs for `LowerTool` both standalone and nested inside
  `PlaceTool`'s expansion (whose approach motion changes the pose between the generic gate above
  and the actual lower). For a standalone guarded `LowerTool` this means two occupancy fetches (the
  generic gate, then the guard's own) — a deliberate, harmless duplication favoring one
  self-contained guard implementation over conditionally skipping it. If `occupancy_supplier` is
  `None` entirely, a guarded `LowerTool` proceeds unguarded (documented: this is the cautious/
  background-diff pipeline's mode, where planned placements already came from a mask the planner
  trusted at plan-build time, and there is no live per-frame mask to re-check against).

Threading: `Executor` is not internally synchronized — it is meant to be driven by exactly one
thread (the session runner's control loop, design 02 §3.5: "single-threaded control loop"), the
same way legacy's `run_grasping` loop was. It reads from `source`/`occupancy_supplier` (both
independently thread-safe) and writes to `robot`/`tracker` (assumed single-writer, matching
`ActionTracker`'s and `RobotInterface`'s own documented threading models).
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from autograsper.execution.actions import Action, ActionPhase, ActionTracker, ActionType
from autograsper.execution.errors import (
    ExecutionError,
    RobotAPIError,
    StaleMaskTimeout,
    ToolLost,
    UnsafeLower,
)
from autograsper.execution.safety import SafetyValidator, tool_footprint_clear
from autograsper.hardware.robot_interface import CommandReceipt, Order, OrderType
from autograsper.planning.planner import FreshnessPolicy
from autograsper.planning.workspace import Workspace, sample_tool_pose
from autograsper.planning.types import (
    CheckToolGrip,
    LowerTool,
    MoveTo,
    PlaceTool,
    Plan,
    Primitive,
    Push,
    RefreshMask,
    RegraspTool,
    SweepWall,
)

logger = logging.getLogger(__name__)

# -- tunables that are not (yet) config-schema fields, ported verbatim from legacy call sites -----

# `RandomPushGrasper.sweep()` always samples tool pose with these two constants (distinct from
# `Workspace`'s own fence-wall tool_width_robot/safety_margin, which are for *building* the walls,
# not for the sweep-pose sample itself) — kept as module constants exactly as legacy hardcoded them.
_SWEEP_TOOL_WIDTH = 0.02
_SWEEP_SAFETY_MARGIN = 0.02

# `RandomPushGrasper.perform_grab_tool`'s scripted rack sequence, verbatim.
_RACK_CLEARANCE_Z = 1.0
_RACK_APPROACH_XY: Tuple[float, float] = (0.03, 0.49)
_RACK_GRASP_Z = 0.27
_POST_RACK_XY: Tuple[float, float] = (0.5, 0.41)
_POST_RACK_ROTATE = 90
_GRIPPER_OPEN_VALUE = 1.0
_GRIPPER_CLOSE_VALUE = 0.0

# `RandomPushGrasper.startup`'s tool-grip inspection pose, verbatim.
_INSPECTION_Z = 1.0
_INSPECTION_XY: Tuple[float, float] = (0.5, 0.41)
_INSPECTION_ROTATE = 90

# `RandomPushGrasper.update_mask_and_process`'s move-aside pose, verbatim.
_REFRESH_Z = 1.0
_REFRESH_XY: Tuple[float, float] = (0.0, 1.0)
_REFRESH_ROTATE = 90

# `find_tool_placements`/legacy default MIN_CLEARANCE_PX, reused as this module's default
# footprint-guard clearance requirement (see `planning/_push_policy.py`'s
# `DEFAULT_MIN_CLEARANCE_PX`, not imported here to avoid a private-module dependency).
_DEFAULT_MIN_CLEARANCE_PX = 10

_POSE_EPS = 1e-3
_Z_EPS = 1e-3

_TOP_ACTION_TYPE = {
    Push: ActionType.MOVE_XY,
    SweepWall: ActionType.SWEEP,
}


@dataclass(frozen=True)
class PlanResult:
    """Result of `Executor.run_plan`.

    - `completed`: number of primitives fully executed (in plan order).
    - `aborted`: `True` only when `run_plan` stopped early because `shutdown_event` was set
      *between* primitives (a clean stop, not a failure) — `completed` is then `< len(plan.
      primitives)` and `error` is `None`.
    - `error`: always `None` in this implementation. Typed failures (`ExecutionError` subclasses)
      and unexpected exceptions are **raised** out of `run_plan`/`execute_primitive` (after closing
      any open `Action`s cleanly — see module docstring), not captured here; the field is kept on
      `PlanResult` for forward compatibility / to make the "no error" case self-documenting at call
      sites that only check the return value. Decision logged in `design/IMPLEMENTATION_LOG.md`
      ("decide and document" per the task spec).
    """

    completed: int
    aborted: bool
    error: Optional[BaseException] = None


def _sleep_with_shutdown(duration: float, shutdown_event: Optional[threading.Event]) -> None:
    """Ported from `autograsper/grasper.py::sleep_with_shutdown` — sleep in small increments,
    checking for shutdown, instead of one uninterruptible `time.sleep(duration)`."""
    end_time = time.time() + duration
    while time.time() < end_time:
        if shutdown_event is not None and shutdown_event.is_set():
            return
        time.sleep(min(0.05, max(0.0, end_time - time.time())))


def _near(a: float, b: float, eps: float) -> bool:
    return abs(a - b) <= eps


def _child_action_info(
    order_type: OrderType, values: Tuple[float, ...]
) -> Tuple[ActionType, Dict[str, Any], bool]:
    """Map one `Order` to its child `ActionType`/`action_details`/`is_planar_2d` — ported from
    `grasper.py::action_type_from_order`/`action_details_from_order`, extended: `GRIPPER` (the
    collapsed Wave 1 order type, see `hardware/robot_interface.py`) is split back into
    `ActionType.GRIPPER_OPEN`/`GRIPPER_CLOSE` by opening value (`>= 0.5` -> open) so the legacy
    two-member split survives despite `OrderType` itself now having only one gripper member.
    Logged in `design/IMPLEMENTATION_LOG.md`.
    """
    if order_type == OrderType.MOVE_XY:
        return ActionType.MOVE_XY, {"x": values[0], "y": values[1]}, True
    if order_type == OrderType.MOVE_Z:
        return ActionType.MOVE_Z, {"z": values[0]}, False
    if order_type == OrderType.ROTATE:
        return ActionType.ROTATE, {"angle": values[0]}, True
    if order_type == OrderType.GRIPPER:
        opening = values[0]
        action_type = ActionType.GRIPPER_OPEN if opening >= 0.5 else ActionType.GRIPPER_CLOSE
        return action_type, {"position": opening}, False
    raise ExecutionError(f"Executor: unknown order type {order_type!r}")


class Executor:
    """Expands `Plan`/`Primitive`s into `RobotInterface` orders, owning choreography, safety
    validation, freshness/guard enforcement, action tracking, and pacing (design 02 §3.4)."""

    def __init__(
        self,
        robot,
        source,
        tracker: ActionTracker,
        safety: SafetyValidator,
        freshness: FreshnessPolicy,
        workspace: Workspace,
        config,
        *,
        occupancy_supplier=None,
        tool_grip_checker=None,
        order_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        debug=None,
        shutdown_event: Optional[threading.Event] = None,
        freshness_timeout: float = 5.0,
        end_frame_timeout: float = 2.0,
        min_clearance_px: float = _DEFAULT_MIN_CLEARANCE_PX,
    ) -> None:
        """
        Args:
            robot: `hardware.robot_interface.RobotInterface`-duck-typed backend (`DryRunRobot` in
                every test/example per CONVENTIONS.md's hard safety rule).
            source: `observation.source.ObservationSource`-duck-typed (`latest()`/`await_next()`).
            tracker: `ActionTracker` this executor is the sole writer of.
            safety: `execution.safety.SafetyValidator` for order-level checks.
            freshness: `planning.planner.FreshnessPolicy` deciding which primitives need a fresh
                occupancy result before running.
            workspace: `planning.workspace.Workspace` (fence walls, manipulation boundary, tool
                dims, heights via `workspace.config`, and `workspace.frames` for the footprint
                guard's pixel conversion).
            config: duck-typed; only `config.experiment.time_between_orders` is read (matching
                `config_schema.ExperimentConfig`, but any object with that attribute path works —
                see the tests).
            occupancy_supplier: `SegmentationWorker`-shaped (`latest()`/`await_result(
                min_source_seq, timeout)`), or `None` for a pipeline with no live per-frame mask
                (see module docstring's guard-design note).
            tool_grip_checker: `perception.tool_grip.ToolGripChecker`-shaped (`check(obs) ->
                GripCheckResult`), or `None` (checks become no-ops, logged).
            order_sink: optional `callable(order_record: dict)` for persistence (Wave 5 wires
                `session.storage`); exceptions from it are logged and swallowed (never abort a
                plan because persistence failed).
            debug: `observation.debug.DebugSink` or `None` — accepted for interface symmetry with
                other layers; not currently used by any `_expand_*` method (no debug artifacts are
                produced at this layer yet).
            shutdown_event: checked between primitives (`run_plan`) and during per-order pacing
                sleeps (`_sleep_with_shutdown`); never set by this class itself (design 02 §3.5).
            freshness_timeout: default timeout (seconds) for `occupancy_supplier.await_result(...)`
                calls (both the generic gate and the `LowerTool` guard).
            end_frame_timeout: timeout (seconds) for the `source.await_next(...)` call used to
                capture a primitive's precise end frame; falls back to `source.latest()` on
                timeout.
            min_clearance_px: default clearance (pixels) required by the `LowerTool` footprint
                guard (`execution.safety.tool_footprint_clear`'s `min_clearance_px`).
        """
        self._robot = robot
        self._source = source
        self.tracker = tracker
        self._safety = safety
        self._freshness = freshness
        self._workspace = workspace
        self._config = config
        self._occupancy_supplier = occupancy_supplier
        self._tool_grip_checker = tool_grip_checker
        self._order_sink = order_sink
        self._debug = debug
        self._shutdown_event = shutdown_event
        self._freshness_timeout = freshness_timeout
        self._end_frame_timeout = end_frame_timeout
        self._min_clearance_px = min_clearance_px

        self._time_between_orders = config.experiment.time_between_orders

        self._last_x, self._last_y, self._last_z, self._last_angle = self._seed_last_pose()

    # -- completed-action stream (passthrough; see execution.actions.ActionTracker) -------------

    def register_completion_callback(self, fn: Callable[[Action], None]) -> None:
        """Passthrough to `self.tracker.register_completion_callback` (kept here so callers don't
        need to know the tracker is where the registry actually lives)."""
        self.tracker.register_completion_callback(fn)

    # -- public API ------------------------------------------------------------------------------

    def run_plan(self, plan: Plan, phase: ActionPhase) -> PlanResult:
        """Execute every primitive in `plan.primitives`, in order.

        Stops early (returning `PlanResult(aborted=True)`) if `shutdown_event` becomes set between
        primitives. A typed (`ExecutionError` subclass) or unexpected exception from
        `execute_primitive` propagates out of this call (after that primitive's own `Action`(s)
        have been closed) rather than being captured in the returned `PlanResult` — see
        `PlanResult.error`'s docstring.
        """
        completed = 0
        for primitive in plan.primitives:
            if self._shutdown_event is not None and self._shutdown_event.is_set():
                return PlanResult(completed=completed, aborted=True, error=None)
            self.execute_primitive(primitive, phase)
            completed += 1
        return PlanResult(completed=completed, aborted=False, error=None)

    def execute_primitive(self, primitive: Primitive, phase: ActionPhase) -> None:
        """Execute one primitive: freshness gate, open a top-level `Action`, expand to orders
        (each a child `Action`), close the top-level `Action` with a fresh end frame.

        Raises `StaleMaskTimeout` before opening any action if the freshness gate times out.
        Raises whatever `_dispatch_primitive` raises (`UnsafeLower`/`ToolLost`/`RobotAPIError`/
        `ExecutionError`/anything else) after closing the top-level `Action` cleanly.
        """
        self._enforce_freshness_gate(primitive)

        top_type = _TOP_ACTION_TYPE.get(type(primitive), ActionType.OTHER)
        is_planar = isinstance(primitive, Push)
        start_frame = self._current_frame_index()
        top_id = self.tracker.start_action(
            action_type=top_type,
            phase=phase,
            start_frame=start_frame,
            start_robot_state=self._robot_state_dict(),
            is_planar_2d=is_planar,
            action_details=dataclasses.asdict(primitive),
            description=primitive.describe(),
        )

        try:
            self._dispatch_primitive(primitive, phase, top_id)
        except BaseException:
            end_frame = self._current_frame_index()
            self.tracker.end_action(top_id, end_frame, end_robot_state=self._robot_state_dict())
            raise

        end_obs = None
        last_obs = self._source.latest()
        if last_obs is not None:
            end_obs = self._source.await_next(last_obs.seq, timeout=self._end_frame_timeout)
        if end_obs is None:
            end_obs = self._source.latest()

        if end_obs is not None and end_obs.frame_index is not None:
            end_frame = end_obs.frame_index
        elif end_obs is not None:
            end_frame = end_obs.seq
        else:
            end_frame = start_frame
        end_state = dataclasses.asdict(end_obs.robot_state) if end_obs is not None else None
        self.tracker.end_action(top_id, end_frame, end_robot_state=end_state)

    # -- freshness gate ----------------------------------------------------------------------

    def _enforce_freshness_gate(self, primitive: Primitive) -> None:
        """Design 03 §3.2's freshness rule, generic across every freshness-categorized primitive
        (`LowerTool`/`PlaceTool`/`SweepWall` per `FreshnessPolicy`). No-op if `occupancy_supplier`
        is `None` or `primitive` has no freshness requirement configured."""
        if self._occupancy_supplier is None:
            return
        if not self._freshness.requires_fresh(primitive):
            return
        obs = self._source.latest()
        min_seq = obs.seq if obs is not None else 0
        occ = self._occupancy_supplier.await_result(min_source_seq=min_seq, timeout=self._freshness_timeout)
        if occ is None:
            raise StaleMaskTimeout(
                primitive=type(primitive).__name__, min_source_seq=min_seq, timeout=self._freshness_timeout
            )

    # -- primitive dispatch -------------------------------------------------------------------

    def _dispatch_primitive(self, primitive: Primitive, phase: ActionPhase, parent_id: int) -> None:
        if isinstance(primitive, MoveTo):
            self._expand_move_to(primitive, phase, parent_id)
        elif isinstance(primitive, PlaceTool):
            self._expand_place_tool(primitive, phase, parent_id)
        elif isinstance(primitive, LowerTool):
            self._expand_lower_tool(primitive, phase, parent_id)
        elif isinstance(primitive, Push):
            self._expand_push(primitive, phase, parent_id)
        elif isinstance(primitive, SweepWall):
            self._expand_sweep_wall(primitive, phase, parent_id)
        elif isinstance(primitive, RegraspTool):
            self._expand_regrasp_tool(primitive, phase, parent_id)
        elif isinstance(primitive, RefreshMask):
            self._expand_refresh_mask(primitive, phase, parent_id)
        elif isinstance(primitive, CheckToolGrip):
            self._expand_check_tool_grip(primitive, phase, parent_id)
        else:
            raise ExecutionError(f"Executor: no expansion for primitive {type(primitive).__name__!r}")

    # -- MoveTo ---------------------------------------------------------------------------------

    def _expand_move_to(self, move: MoveTo, phase: ActionPhase, parent_id: int) -> None:
        """`z` given and raising (or equal) -> raise before translate; `z` given and lowering ->
        translate (+ rotate) before lowering; `z=None` -> translate (+ rotate) only. Design 02
        §3.3: "ordering rules (raise before translate) live in the executor, not the planner"."""
        if move.z is not None and move.z >= self._last_z - _Z_EPS:
            self._send_order(OrderType.MOVE_Z, (move.z,), phase=phase, parent_id=parent_id)
            self._send_order(OrderType.MOVE_XY, (move.x, move.y), phase=phase, parent_id=parent_id)
            if move.angle is not None:
                self._send_order(OrderType.ROTATE, (move.angle,), phase=phase, parent_id=parent_id)
        elif move.z is not None:
            self._send_order(OrderType.MOVE_XY, (move.x, move.y), phase=phase, parent_id=parent_id)
            if move.angle is not None:
                self._send_order(OrderType.ROTATE, (move.angle,), phase=phase, parent_id=parent_id)
            self._send_order(OrderType.MOVE_Z, (move.z,), phase=phase, parent_id=parent_id)
        else:
            self._send_order(OrderType.MOVE_XY, (move.x, move.y), phase=phase, parent_id=parent_id)
            if move.angle is not None:
                self._send_order(OrderType.ROTATE, (move.angle,), phase=phase, parent_id=parent_id)

    # -- PlaceTool --------------------------------------------------------------------------

    def _expand_place_tool(self, place: PlaceTool, phase: ActionPhase, parent_id: int) -> None:
        """Ported from `granular_pusher.py::perform_task`'s clearance raise +
        `MOVE_XY(pos_world) / ROTATE(angle) / MOVE_Z(grasp_height)` sequence: approach at
        `workspace.config.clearance_height`, translate, rotate, then a guarded `LowerTool`."""
        cfg = self._workspace.config
        self._send_order(OrderType.MOVE_Z, (cfg.clearance_height,), phase=phase, parent_id=parent_id)
        self._send_order(OrderType.MOVE_XY, (place.x, place.y), phase=phase, parent_id=parent_id)
        self._send_order(OrderType.ROTATE, (place.angle,), phase=phase, parent_id=parent_id)
        self._expand_lower_tool(LowerTool(z=place.lower_to, guarded=True), phase, parent_id)

    # -- LowerTool (+ real-time footprint guard) --------------------------------------------

    def _expand_lower_tool(self, lower: LowerTool, phase: ActionPhase, parent_id: int) -> None:
        """The mask-aware `LowerTool` guard (design 02 §3.4, design 03 §3.3), then `MOVE_Z(z)`.

        See module docstring's "Freshness/guard design decisions" for why this always re-checks
        occupancy itself (rather than relying solely on `_enforce_freshness_gate`), and for the
        `occupancy_supplier is None` unguarded-proceed behavior.
        """
        if lower.guarded and self._occupancy_supplier is not None:
            if self._freshness.requires_fresh(lower):
                obs = self._source.latest()
                min_seq = obs.seq if obs is not None else 0
                occ = self._occupancy_supplier.await_result(
                    min_source_seq=min_seq, timeout=self._freshness_timeout
                )
                if occ is None:
                    raise StaleMaskTimeout(
                        primitive="LowerTool", min_source_seq=min_seq, timeout=self._freshness_timeout
                    )
            else:
                occ = self._occupancy_supplier.latest()
                if occ is None:
                    raise StaleMaskTimeout(
                        primitive="LowerTool",
                        min_source_seq=0,
                        timeout=self._freshness_timeout,
                        reason="no occupancy result available from supplier yet",
                    )

            ok, clearance = tool_footprint_clear(
                occ,
                self._last_x,
                self._last_y,
                self._last_angle,
                self._workspace.frames,
                self._workspace.tool_dims_px,
                self._min_clearance_px,
            )
            if not ok:
                raise UnsafeLower(pose=(self._last_x, self._last_y), clearance_px=clearance)

        self._send_order(OrderType.MOVE_Z, (lower.z,), phase=phase, parent_id=parent_id)

    # -- Push -----------------------------------------------------------------------------------

    def _expand_push(self, push: Push, phase: ActionPhase, parent_id: int) -> None:
        """`ROTATE(angle)`, `MOVE_XY(start)` (skipped if already there), `MOVE_Z(height)` (skipped
        if already at that height), `MOVE_XY(end)` — the last order is the planar push itself."""
        self._send_order(OrderType.ROTATE, (push.angle,), phase=phase, parent_id=parent_id)
        if not (_near(self._last_x, push.start_x, _POSE_EPS) and _near(self._last_y, push.start_y, _POSE_EPS)):
            self._send_order(
                OrderType.MOVE_XY, (push.start_x, push.start_y), phase=phase, parent_id=parent_id
            )
        if not _near(self._last_z, push.height, _Z_EPS):
            self._send_order(OrderType.MOVE_Z, (push.height,), phase=phase, parent_id=parent_id)
        self._send_order(OrderType.MOVE_XY, (push.end_x, push.end_y), phase=phase, parent_id=parent_id)

    # -- SweepWall --------------------------------------------------------------------------

    def _expand_sweep_wall(self, sweep: SweepWall, phase: ActionPhase, parent_id: int) -> None:
        """`mode="targeted"`: approach (clearance raise / `MOVE_XY(approach)` / `ROTATE` /
        `MOVE_Z(grasp_height)` / `MOVE_XY(sweep_pos)`) then every `t` pass with
        `already_at_wall=True` (no cautious pre-sweep dance). `mode="fallback"`: every `t` pass
        directly, with the *first* pass using `already_at_wall=False` (legacy's cautious multi-
        height pre-sweep dance) and every subsequent pass `already_at_wall=True`. Ported from
        `granular_pusher.py::sweep_wall`'s two branches."""
        wall = self._workspace.wall(sweep.wall_label)
        cfg = self._workspace.config

        if sweep.mode == "targeted":
            self._send_order(OrderType.MOVE_Z, (cfg.clearance_height,), phase=phase, parent_id=parent_id)
            self._send_order(
                OrderType.MOVE_XY, (sweep.approach_x, sweep.approach_y), phase=phase, parent_id=parent_id
            )
            self._send_order(OrderType.ROTATE, (sweep.angle,), phase=phase, parent_id=parent_id)
            self._send_order(OrderType.MOVE_Z, (cfg.grasp_height,), phase=phase, parent_id=parent_id)
            self._send_order(
                OrderType.MOVE_XY, (sweep.sweep_pos_x, sweep.sweep_pos_y), phase=phase, parent_id=parent_id
            )
            for t in sweep.t_values:
                self._sweep_pass(wall, t, sweep.step, phase, parent_id, already_at_wall=True)
        else:
            already_at_wall = False
            for t in sweep.t_values:
                self._sweep_pass(wall, t, sweep.step, phase, parent_id, already_at_wall=already_at_wall)
                already_at_wall = True

    def _sweep_pass(
        self,
        wall,
        t: float,
        step: float,
        phase: ActionPhase,
        parent_id: int,
        *,
        already_at_wall: bool,
    ) -> None:
        """One `t`-value sweep pass. Ported verbatim from `granular_pusher.py::sweep`: the
        `already_at_wall=False` preamble (cautious pre-sweep at `sweep_height +/- 0.02/0.01` with
        clearance raises between each) followed by the grasp-height triple
        `MOVE_XY(pos) / MOVE_XY(pos + dir*step) / MOVE_XY(pos)` every pass runs regardless."""
        cfg = self._workspace.config
        pose = sample_tool_pose(wall, tool_width=_SWEEP_TOOL_WIDTH, safety_margin=_SWEEP_SAFETY_MARGIN, t=t)
        x, y = float(pose["x"]), float(pose["y"])
        dx, dy = pose["perpendicular_dir"]
        dx, dy = float(dx), float(dy)
        rotation_angle = pose["angle"]

        if not already_at_wall:
            self._send_order(OrderType.MOVE_Z, (cfg.clearance_height,), phase=phase, parent_id=parent_id)
            self._send_order(OrderType.MOVE_XY, (x, y), phase=phase, parent_id=parent_id)
            self._send_order(OrderType.ROTATE, (rotation_angle,), phase=phase, parent_id=parent_id)
            self._send_order(OrderType.MOVE_Z, (cfg.sweep_height + 0.02,), phase=phase, parent_id=parent_id)
            self._send_order(
                OrderType.MOVE_XY, (x + dx * 0.02, y + dy * 0.02), phase=phase, parent_id=parent_id
            )
            self._send_order(OrderType.MOVE_Z, (cfg.clearance_height,), phase=phase, parent_id=parent_id)
            self._send_order(OrderType.MOVE_XY, (x, y), phase=phase, parent_id=parent_id)
            self._send_order(OrderType.MOVE_Z, (cfg.sweep_height - 0.01,), phase=phase, parent_id=parent_id)
            self._send_order(
                OrderType.MOVE_XY, (x + dx * 0.02, y + dy * 0.02), phase=phase, parent_id=parent_id
            )
            self._send_order(OrderType.MOVE_Z, (cfg.clearance_height,), phase=phase, parent_id=parent_id)
            self._send_order(OrderType.MOVE_XY, (x, y), phase=phase, parent_id=parent_id)
            self._send_order(OrderType.MOVE_Z, (cfg.grasp_height,), phase=phase, parent_id=parent_id)

        self._send_order(OrderType.MOVE_XY, (x, y), phase=phase, parent_id=parent_id)
        self._send_order(OrderType.MOVE_XY, (x + dx * step, y + dy * step), phase=phase, parent_id=parent_id)
        self._send_order(OrderType.MOVE_XY, (x, y), phase=phase, parent_id=parent_id)

    # -- RegraspTool ----------------------------------------------------------------------------

    def _expand_regrasp_tool(self, regrasp: RegraspTool, phase: ActionPhase, parent_id: int) -> None:
        """Ported verbatim from `granular_pusher.py::perform_grab_tool`: z 1.0, gripper open,
        rotate 0, xy (0.03, 0.49), z 0.27, gripper close, z 1.0, xy (0.5, 0.41), rotate 90 — every
        order sent with `allow_rack=True` (the whole sequence is a known-safe scripted routine,
        including the sub-grasp-height `z=0.27` and the off-manipulation-boundary rack xy).
        Followed by a post-grasp grip check (if `tool_grip_checker` is configured) on a fresh
        observation, raising `ToolLost` below threshold."""
        self._send_order(
            OrderType.MOVE_Z, (_RACK_CLEARANCE_Z,), phase=phase, parent_id=parent_id, allow_rack=True
        )
        self._send_order(
            OrderType.GRIPPER, (_GRIPPER_OPEN_VALUE,), phase=phase, parent_id=parent_id, allow_rack=True
        )
        self._send_order(OrderType.ROTATE, (0,), phase=phase, parent_id=parent_id, allow_rack=True)
        self._send_order(OrderType.MOVE_XY, _RACK_APPROACH_XY, phase=phase, parent_id=parent_id, allow_rack=True)
        self._send_order(OrderType.MOVE_Z, (_RACK_GRASP_Z,), phase=phase, parent_id=parent_id, allow_rack=True)
        self._send_order(
            OrderType.GRIPPER, (_GRIPPER_CLOSE_VALUE,), phase=phase, parent_id=parent_id, allow_rack=True
        )
        self._send_order(
            OrderType.MOVE_Z, (_RACK_CLEARANCE_Z,), phase=phase, parent_id=parent_id, allow_rack=True
        )
        self._send_order(OrderType.MOVE_XY, _POST_RACK_XY, phase=phase, parent_id=parent_id, allow_rack=True)
        self._send_order(
            OrderType.ROTATE, (_POST_RACK_ROTATE,), phase=phase, parent_id=parent_id, allow_rack=True
        )

        if self._tool_grip_checker is None:
            return
        check_obs = self._fresh_observation()
        if check_obs is None:
            logger.warning("RegraspTool: no observation available for post-grasp grip check, skipping")
            return
        result = self._tool_grip_checker.check(check_obs)
        if not result.ok:
            raise ToolLost(grip_quality=result.quality)

    # -- RefreshMask ------------------------------------------------------------------------

    def _expand_refresh_mask(self, refresh: RefreshMask, phase: ActionPhase, parent_id: int) -> None:
        """Move-aside choreography only (z 1.0, xy (0.0, 1.0), rotate 90 — ported from
        `update_mask_and_process`), then wait for a fresh observation. The actual mask computation
        is the perception/session layer's job (a `SegmentationWorker`/`BackgroundDiffProvider`
        consuming that fresh observation), not the executor's — this primitive only performs the
        robot motion + wait."""
        if not refresh.move_aside:
            return
        self._send_order(OrderType.MOVE_Z, (_REFRESH_Z,), phase=phase, parent_id=parent_id)
        self._send_order(OrderType.MOVE_XY, _REFRESH_XY, phase=phase, parent_id=parent_id)
        self._send_order(OrderType.ROTATE, (_REFRESH_ROTATE,), phase=phase, parent_id=parent_id)
        self._fresh_observation()

    # -- CheckToolGrip --------------------------------------------------------------------------

    def _expand_check_tool_grip(self, check: CheckToolGrip, phase: ActionPhase, parent_id: int) -> None:
        """Move to the fixed inspection pose (z 1.0, xy (0.5, 0.41), rotate 90 — ported from
        `RandomPushGrasper.startup`), then run `tool_grip_checker` on a fresh observation, raising
        `ToolLost` below threshold. A no-op (with a warning) if no checker is configured."""
        self._send_order(OrderType.MOVE_Z, (_INSPECTION_Z,), phase=phase, parent_id=parent_id)
        self._send_order(OrderType.MOVE_XY, _INSPECTION_XY, phase=phase, parent_id=parent_id)
        self._send_order(OrderType.ROTATE, (_INSPECTION_ROTATE,), phase=phase, parent_id=parent_id)

        if self._tool_grip_checker is None:
            logger.warning("CheckToolGrip: no tool_grip_checker configured, skipping check")
            return
        check_obs = self._fresh_observation()
        if check_obs is None:
            logger.warning("CheckToolGrip: no observation available, skipping check")
            return
        result = self._tool_grip_checker.check(check_obs)
        if not result.ok:
            raise ToolLost(grip_quality=result.quality)

    # -- order execution ------------------------------------------------------------------------

    def _send_order(
        self,
        order_type: OrderType,
        values: Tuple[float, ...],
        *,
        phase: ActionPhase,
        parent_id: int,
        allow_rack: bool = False,
    ) -> CommandReceipt:
        """Validate, dispatch, pace, and track one order as a child `Action` of `parent_id`.

        Safety validation happens before the child `Action` is opened (a rejected order never gets
        tracked); an unexpected exception from the actual robot call closes the child `Action`
        before propagating as `RobotAPIError` (or, if it was already an `ExecutionError`, as-is).
        """
        order = Order(order_type, tuple(values)).validate()
        current_z = self._last_z if order_type == OrderType.MOVE_XY else None
        self._safety.validate_order(order, current_z=current_z, allow_rack=allow_rack)

        start_frame = self._current_frame_index()
        child_type, details, is_planar = _child_action_info(order_type, order.values)
        child_id = self.tracker.start_action(
            action_type=child_type,
            phase=phase,
            start_frame=start_frame,
            start_robot_state=self._robot_state_dict(),
            is_planar_2d=is_planar,
            action_details=details,
            parent_id=parent_id,
        )

        try:
            receipt = self._execute_order_on_robot(order)
        except Exception as exc:
            self.tracker.end_action(
                child_id, self._current_frame_index(), end_robot_state=self._robot_state_dict()
            )
            if isinstance(exc, ExecutionError):
                raise
            raise RobotAPIError(f"RobotInterface call failed for {order.type.name}: {exc}") from exc

        self._update_last_pose(order_type, order.values)
        _sleep_with_shutdown(self._time_between_orders, self._shutdown_event)

        end_frame = self._current_frame_index()
        self.tracker.end_action(child_id, end_frame, end_robot_state=self._robot_state_dict())
        self._emit_order_record(order, receipt, start_frame)
        return receipt

    def _execute_order_on_robot(self, order: Order) -> CommandReceipt:
        if order.type == OrderType.MOVE_XY:
            return self._robot.move_xy(order.values[0], order.values[1])
        if order.type == OrderType.MOVE_Z:
            return self._robot.move_z(order.values[0])
        if order.type == OrderType.ROTATE:
            return self._robot.rotate(int(order.values[0]))
        if order.type == OrderType.GRIPPER:
            return self._robot.set_gripper(order.values[0])
        raise ExecutionError(f"Executor: unknown order type {order.type!r}")

    def _update_last_pose(self, order_type: OrderType, values: Tuple[float, ...]) -> None:
        if order_type == OrderType.MOVE_XY:
            self._last_x, self._last_y = values
        elif order_type == OrderType.MOVE_Z:
            self._last_z = values[0]
        elif order_type == OrderType.ROTATE:
            self._last_angle = values[0]

    def _emit_order_record(self, order: Order, receipt: CommandReceipt, frame_index: int) -> None:
        """`orders.json` row shape (legacy keys `order_type`/`order_value`/`time` unchanged, per
        `library/utils.py::write_order`), plus `robot_reported_time`/`frame_index` (additive)."""
        if self._order_sink is None:
            return
        record = {
            "order_type": order.type.name,
            "order_value": [float(v) for v in order.values],
            "time": receipt.send_time,
            "robot_reported_time": receipt.robot_reported_time,
            "frame_index": frame_index,
        }
        try:
            self._order_sink(record)
        except Exception:
            logger.exception("Executor: order_sink raised, dropping this order record")

    # -- helpers ---------------------------------------------------------------------------------

    def _seed_last_pose(self) -> Tuple[float, float, float, float]:
        obs = self._source.latest()
        if obs is not None and obs.robot_state is not None:
            st = obs.robot_state
            return (
                st.x if st.x is not None else 0.5,
                st.y if st.y is not None else 0.5,
                st.z if st.z is not None else 1.0,
                st.rotation if st.rotation is not None else 0.0,
            )
        return (0.5, 0.5, 1.0, 0.0)

    def _current_frame_index(self) -> int:
        """Best-available integer frame index: `Observation.frame_index` if a recorder has
        registered one (Wave 5), else `Observation.seq` as a monotonic stand-in, else `0` if no
        observation has been published yet."""
        obs = self._source.latest()
        if obs is None:
            return 0
        if obs.frame_index is not None:
            return obs.frame_index
        return obs.seq

    def _robot_state_dict(self) -> Optional[Dict[str, Any]]:
        obs = self._source.latest()
        if obs is None or obs.robot_state is None:
            return None
        return dataclasses.asdict(obs.robot_state)

    def _fresh_observation(self):
        """`source.await_next(after last-known seq, timeout=end_frame_timeout)`, falling back to
        `source.latest()` on timeout/no prior observation. Used wherever a primitive's expansion
        needs "the observation captured after this motion completed" (tool-grip checks,
        `RefreshMask`'s wait)."""
        last_obs = self._source.latest()
        if last_obs is None:
            return None
        fresh = self._source.await_next(last_obs.seq, timeout=self._end_frame_timeout)
        return fresh if fresh is not None else self._source.latest()
