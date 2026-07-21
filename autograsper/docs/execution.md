# `autograsper/execution/` — ActionTracker, SafetyValidator, Executor

Status: **implemented** (Wave 4). Design reference:
[`../design/02_proposed_architecture.md`](../design/02_proposed_architecture.md) §3.3 (primitive
table), §3.4 (`Executor` — core scope of this wave), §3.5 (typed exceptions);
[`../design/03_segmenter_native_design.md`](../design/03_segmenter_native_design.md) §3.2
(freshness rule), §3.3 (`LowerTool` real-time guard), §5 (online transition emission
requirements).

## Purpose

The only layer that both talks to the robot and writes actions (design 02 §3.4). It consumes
`planning.types.Plan`/`Primitive` (Wave 3b) and `hardware.robot_interface.RobotInterface`
(Wave 1), and produces: `RobotInterface` calls (via `Order`/`CommandReceipt`), a tree of
`execution.actions.Action`s (top-level primitive + child per-order, per `ActionTracker`), and
optional `orders.json`-shaped records through an injected sink. It replaces
`grasper.py::AutograsperBase.queue_orders`/`execute_order` and
`custom_graspers/granular_pusher.py`'s hand-rolled order-sequence methods
(`sweep`/`sweep_wall`/`perform_grab_tool`/`update_mask_and_process`/`startup`'s inspection pose),
centralizing every one of those choreographies as `Executor._expand_*` methods driven by the
planning layer's semantic primitives instead.

**Architectural note (deviation from design 02 §2's dependency-arrow summary, logged in
`design/IMPLEMENTATION_LOG.md`):** the informal layout diagram reads `hardware ← execution ←
session → planning → perception → observation` ("arrows = may import"), which taken literally
would forbid `execution` importing `planning`. This is unworkable: the executor's whole job is to
consume `planning.types.Primitive` subclasses and reuse `planning.workspace`'s pure geometry
helpers (`make_tool_mask`, `check_placement`, `sample_tool_pose`, `Wall`, `Workspace`) — duplicating
that math in `execution/` would violate "one definition" for the exact same reason design 03 §2.1
gives for `CoordinateFrames`. `execution/executor.py` and `execution/safety.py` import
`autograsper.planning.types` and `autograsper.planning.workspace` (pure functions/value types
only — no planner classes, no `random_push_planner`/`seg_push_planner` imports). This mirrors the
precedent already set in Wave 2 (`observation/source.py` importing `perception/frames.py` against
the same diagram's general direction) and Wave 3b (`planning/workspace.py` importing
`perception/frames.py`'s `CoordinateFrames` type).

## Public API

### `execution/actions.py`

Port of `autograsper/legacy/action_tracker.py`, extended for composite (parent/child) actions:

- `ActionType` — `{GRIPPER_OPEN, GRIPPER_CLOSE, MOVE_Z, MOVE_XY, ROTATE, SWEEP, OTHER}` (unchanged
  from legacy — legacy already had `SWEEP`, no new members added, preserving dataset compatibility
  for any tooling reading `action_type` string values).
- `ActionPhase` — `{TASK, RESET, STARTUP, OTHER}` (unchanged from legacy).
- `Action` — frozen-in-spirit dataclass (mutable fields updated in place by `ActionTracker`, same
  as legacy) with every legacy field plus **`parent_id: Optional[int]`** (`None` for a top-level/
  primitive action; the top-level action's `action_id` for a child/per-order action).
  `to_dict()`/`from_dict()` round-trip `parent_id`; a JSON row recorded before this wave (no
  `parent_id` key) loads fine via the dataclass field default (`None`).
- `ActionTracker` — thread-safe (one `threading.RLock`), now tracks **two** simultaneously
  in-flight actions: `current_action` (top-level) and `current_child_action` (nested). Public API:
  - `start_action(..., parent_id=None)` — `parent_id=None` opens/replaces the top-level slot;
    passing the open top-level action's id opens/replaces the child slot.
  - `end_action(action_id, end_frame, end_robot_state=None)` — checked against the child slot
    first, then the top-level slot (so ending a child while its parent is still open works
    correctly); appends the completed `Action` to `self.actions` and fires every registered
    completion callback (see below), then returns it (`None` + a logged warning if `action_id`
    matches neither open slot).
  - `get_action_for_frame(frame_index)` — prefers the **deepest** match: in-flight child, then
    in-flight top-level, then (among completed actions) a completed child match over a completed
    top-level match.
  - `get_current_action()` / `get_current_child_action()`, `get_all_actions()`,
    `get_actions_by_type()`, `get_actions_by_phase()`, `get_planar_2d_actions()`, `to_dict()` (now
    also includes a `"current_child_action"` key — additive), `to_json()`, `clear()` — all
    unchanged in shape from legacy otherwise.
  - `register_completion_callback(fn: Callable[[Action], None])` — new. `fn` is called once per
    `end_action()`, for every completed action (top-level or child), **outside** the tracker's
    lock (the completed-action list is already updated and the lock released before any callback
    runs) — a slow/reentrant callback (Wave 5's `TransitionWriter` subscribing to completed `Push`
    actions per design 03 §5) can never deadlock a concurrent `start_action`/`end_action` call. A
    raising callback is logged and does not affect any other callback or the tracker's own state.

### `execution/errors.py`

Typed exceptions (design 02 §3.5) — none are caught anywhere in this wave; the session layer
(Wave 5) is expected to catch each and decide retry/replan/intervention/abort:

- `ExecutionError(Exception)` — base class; also raised **directly** (not via a subclass) by
  `SafetyValidator.validate_order` for generic manipulation-boundary/z-floor violations (no
  dedicated subclass was introduced for those — logged in `design/IMPLEMENTATION_LOG.md`).
- `UnsafeLower(pose: Tuple[float, float], clearance_px: float)` — the `LowerTool` real-time
  footprint-guard failure.
- `ToolLost(grip_quality: float)` — a tool-grip check (`CheckToolGrip`, or `RegraspTool`'s
  post-grasp check) came back below threshold.
- `NeedsHumanHelp(reason: str)` — defined per the design 02 §3.5 vocabulary; **not raised anywhere
  in this wave's `Executor`** — reserved for the session layer to escalate into (e.g. repeated
  `ToolLost`). Logged in `design/IMPLEMENTATION_LOG.md`.
- `RobotAPIError` — wraps any non-`ExecutionError` exception a `RobotInterface` call itself raises.
- `StaleMaskTimeout(primitive: str, min_source_seq: int, timeout: Optional[float], reason: str="")`
  — the freshness gate or the `LowerTool` guard could not obtain a fresh-enough (or any) occupancy
  result within `timeout`.

### `execution/safety.py`

- `SafetyValidator(workspace: Workspace, *, boundary_margin=None, high_z_threshold=None)` —
  order-level checks **beyond** `hardware.robot_interface.Order.validate()`'s clipping:
  - `validate_order(order, *, current_z=None, allow_rack=False) -> None` — raises `ExecutionError`
    on violation:
    - `MOVE_Z` below `workspace.config.grasp_height` (the z floor) is rejected.
    - `MOVE_XY` is checked against `workspace.manip_x`/`manip_y` (± `boundary_margin`, default
      `workspace.config.safety_margin`) **only** when `current_z < high_z_threshold` (default
      `workspace.config.clearance_height`) — i.e. only "low" moves are boundary-constrained; a
      move at/above the threshold (e.g. the `RefreshMask`/`CheckToolGrip` corner poses, always
      sent at `z=1.0`) may go anywhere, including outside the manipulation boundary.
    - `current_z=None` (unknown) is treated as `0.0` — conservative, "assume low" — since the
      caller not knowing the current z is itself a reason to be cautious.
    - `allow_rack=True` bypasses both checks entirely (the `RegraspTool` scripted sequence, whose
      `z=0.27` grasp step is legitimately below the floor and whose rack-approach xy sits outside
      the manipulation boundary while at `z=1.0` anyway).
    - `ROTATE`/`GRIPPER` orders are never checked (no boundary/z-floor implication).
- `tool_footprint_clear(occ_result, x_robot, y_robot, angle_deg, frames, tool_dims_px,
  min_clearance_px) -> (bool, clearance_px)` — module-level function (no safety-policy state
  needed): rasterizes the tool footprint at the given pose via
  `planning.workspace.make_tool_mask`, converts the pose to `crop_px` via
  `frames.robot_to_crop_px`, and checks it against `occ_result.crop_mask`'s distance transform via
  `planning.workspace.check_placement` — `True` only if the footprint doesn't overlap any occupied
  pixel **and** the minimum clearance is `>= min_clearance_px`.

### `execution/executor.py`

- `PlanResult(completed: int, aborted: bool, error: Optional[BaseException] = None)` — frozen.
  `error` is **always `None`** in this implementation (see "PlanResult semantics" below).
- `Executor(robot, source, tracker, safety, freshness, workspace, config, *, occupancy_supplier=
  None, tool_grip_checker=None, order_sink=None, debug=None, shutdown_event=None,
  freshness_timeout=5.0, end_frame_timeout=2.0, min_clearance_px=10)`:
  - `run_plan(plan: Plan, phase: ActionPhase) -> PlanResult` — executes every primitive in order;
    see "PlanResult semantics" below for the abort/raise contract.
  - `execute_primitive(primitive: Primitive, phase: ActionPhase) -> None` — freshness gate, open a
    top-level `Action`, dispatch (expand to orders / nested primitives), close the top-level
    `Action` with a fresh end frame (`source.await_next(...)`, falling back to `source.latest()`
    on timeout).
  - `register_completion_callback(fn)` — passthrough to `self.tracker.register_completion_callback`
    (also exposed as `executor.tracker` directly, a public attribute).

## `PlanResult` semantics (decision, logged in `design/IMPLEMENTATION_LOG.md`)

The task's own framing is self-tensioned: `PlanResult` carries an `error` field, but typed failures
must also "raise ... after closing actions cleanly." Resolution: **`run_plan`/`execute_primitive`
raise on any failure** (typed `ExecutionError` subclass, or an unexpected exception) — after the
top-level (and, if applicable, child) `Action` has been cleanly closed via `end_action()`, so no
action is ever left dangling in `tracker.current_action`/`current_child_action`. `PlanResult` is
only ever *returned* in the two non-failure cases:
- the whole plan ran: `PlanResult(completed=len(plan.primitives), aborted=False, error=None)`.
- `shutdown_event` was observed set **between** primitives (never mid-primitive): `PlanResult(
  completed=N, aborted=True, error=None)` — a clean, intentional stop, not a failure.

`PlanResult.error` is therefore always `None` in practice; it exists so a caller that only checks
the return value (rather than wrapping `run_plan` in `try/except`) has a self-documenting field to
confirm "no error" against, and for forward compatibility if a future wave decides some failure
class should be captured instead of raised.

## Executor public API contract for Wave 5 (session/recorder/transitions)

- **Completed-action stream**: call `executor.register_completion_callback(fn)` (or
  `executor.tracker.register_completion_callback(fn)` directly) to receive every completed
  `Action` — top-level and child — as it closes, outside any lock. A `TransitionWriter` (design 03
  §5) should filter for `action.parent_id is None and action.action_type is ActionType.MOVE_XY and
  action.is_planar_2d` to isolate completed `Push` actions; each such `Action` carries everything
  design 03 §5 lists: `action_details` has the primitive's own fields (`start_x`, `start_y`,
  `angle`, `end_x`, `end_y`, `height`), and `start_frame`/`end_frame` +
  `start_robot_state`/`end_robot_state` (already on every `Action`, no duplication needed) give the
  frame range and robot pose at both ends.
- **Order records**: pass `order_sink=callable(record_dict)` to receive one dict per order sent —
  `{"order_type": str, "order_value": List[float], "time": float, "robot_reported_time":
  Optional[str], "frame_index": int}` (legacy `orders.json` keys `order_type`/`order_value`/`time`
  unchanged; `robot_reported_time`/`frame_index` additive). Exceptions from `order_sink` are logged
  and swallowed — persistence failures never abort a plan.
- **Frame indices**: until a recorder registers a `frame_index_provider` on the shared
  `ObservationSource` (Wave 5), every `Action.start_frame`/`end_frame` and order record's
  `frame_index` fall back to `Observation.seq` (monotonic, but not a recorder frame number) — see
  `Executor._current_frame_index`'s docstring. Once a provider is registered, indices become real
  recorder frame numbers with no code change needed here.
- **`Workspace`/`SafetyValidator`/`FreshnessPolicy` are constructed by the session layer** (Wave 5
  owns config wiring) and passed in; `Executor` never constructs its own.

## Primitive expansion tables

Every expansion is a direct, documented port of a `custom_graspers/granular_pusher.py` call site
(see `execution/executor.py`'s per-`_expand_*`-method docstrings for the exact legacy method).
`x`/`y`/`z` are robot-normalized `[0, 1]`; `angle` is degrees.

### `MoveTo(x, y, z=None, angle=None)`

| `z` | Order sequence |
|---|---|
| `None` | `MOVE_XY(x, y)`, then `ROTATE(angle)` if `angle is not None` |
| `z >= last_z` (raising or level) | `MOVE_Z(z)`, `MOVE_XY(x, y)`, then `ROTATE(angle)` if given |
| `z < last_z` (lowering) | `MOVE_XY(x, y)`, `ROTATE(angle)` if given, `MOVE_Z(z)` |

"Raise before translate" ordering (design 02 §3.3) lives here, not in the planner.

### `PlaceTool(x, y, angle, lower_to)`

`MOVE_Z(workspace.config.clearance_height)`, `MOVE_XY(x, y)`, `ROTATE(angle)`, then a nested
guarded `LowerTool(lower_to, guarded=True)` expansion (see below).

### `LowerTool(z, guarded=True)`

If `guarded` and an `occupancy_supplier` is configured: fetch an occupancy result (fresh, awaited
per `FreshnessPolicy`, if `"lower_tool"` is in the configured freshness categories; otherwise
`occupancy_supplier.latest()`), then `execution.safety.tool_footprint_clear` at the **current
tracked pose** (`self._last_x`/`_last_y`/`_last_angle`) — raise `UnsafeLower` on failure, raise
`StaleMaskTimeout` if no result is obtainable at all. Then always: `MOVE_Z(z)`. If
`occupancy_supplier is None`, the guard is skipped entirely and `MOVE_Z(z)` proceeds unguarded —
this is the cautious/background-diff pipeline's mode, where the planner already read a mask at
plan-build time and there is no live per-frame mask to re-check against (documented, not a bug).

### `Push(start_x, start_y, angle, end_x, end_y, height)`

`ROTATE(angle)`; `MOVE_XY(start_x, start_y)` **unless** already there (within `1e-3`); `MOVE_Z(
height)` **unless** already at that height (within `1e-3`); `MOVE_XY(end_x, end_y)` — this last
order is the tracked planar push. Tracked as one top-level `ActionType.MOVE_XY`, `is_planar_2d=
True` action (see "Primitive → top-level ActionType mapping" below).

### `SweepWall(wall_label, mode, angle, sweep_dir, step, t_values, approach_x/y=None,
sweep_pos_x/y=None)`

- `mode="targeted"`: `MOVE_Z(clearance_height)`, `MOVE_XY(approach_x, approach_y)`,
  `ROTATE(angle)`, `MOVE_Z(grasp_height)`, `MOVE_XY(sweep_pos_x, sweep_pos_y)`, then every `t` in
  `t_values` gets one direct sweep pass (`already_at_wall=True`, see below).
- `mode="fallback"`: every `t` in `t_values` gets one sweep pass; the **first** pass uses
  `already_at_wall=False` (the cautious pre-sweep dance), every subsequent pass
  `already_at_wall=True`.
- One sweep pass at parameter `t`, pose from `planning.workspace.sample_tool_pose(wall,
  tool_width=0.02, safety_margin=0.02, t=t)` → `(x, y)`, `perpendicular_dir` → `(dx, dy)`:
  - if `already_at_wall=False` (ported verbatim from `granular_pusher.py::sweep`): `MOVE_Z(
    clearance_height)`, `MOVE_XY(x, y)`, `ROTATE(wall.angle)`, `MOVE_Z(sweep_height + 0.02)`,
    `MOVE_XY(x + dx*0.02, y + dy*0.02)`, `MOVE_Z(clearance_height)`, `MOVE_XY(x, y)`, `MOVE_Z(
    sweep_height - 0.01)`, `MOVE_XY(x + dx*0.02, y + dy*0.02)`, `MOVE_Z(clearance_height)`,
    `MOVE_XY(x, y)`, `MOVE_Z(grasp_height)`.
  - always then: `MOVE_XY(x, y)`, `MOVE_XY(x + dx*step, y + dy*step)`, `MOVE_XY(x, y)` — the
    contact sweep triple.

Tracked as one top-level `ActionType.SWEEP` action.

### `RegraspTool()`

Ported verbatim from `perform_grab_tool`, every order sent with `allow_rack=True`: `MOVE_Z(1.0)`,
`GRIPPER(1.0)` (open), `ROTATE(0)`, `MOVE_XY(0.03, 0.49)`, `MOVE_Z(0.27)`, `GRIPPER(0.0)` (close),
`MOVE_Z(1.0)`, `MOVE_XY(0.5, 0.41)`, `ROTATE(90)`. If `tool_grip_checker` is configured, a fresh
observation is checked afterward and `ToolLost` is raised below threshold (a no-op with a log line
if no checker is configured).

### `RefreshMask(move_aside=True)`

If `move_aside`: `MOVE_Z(1.0)`, `MOVE_XY(0.0, 1.0)`, `ROTATE(90)`, then wait for a fresh
observation (`source.await_next(...)`). The actual mask computation is the perception/session
layer's job (a provider consuming that fresh observation) — this primitive is motion + wait only.
A no-op if `move_aside=False`.

### `CheckToolGrip()`

`MOVE_Z(1.0)`, `MOVE_XY(0.5, 0.41)`, `ROTATE(90)` (the legacy startup inspection pose), then a
fresh-observation grip check exactly like `RegraspTool`'s post-grasp check; a no-op with a warning
if no `tool_grip_checker` is configured.

## Primitive → top-level `ActionType` mapping (decision, logged in `design/IMPLEMENTATION_LOG.md`)

Legacy dataset consumers expect `ActionType` values from the existing enum; adding new members
(e.g. a `PLACE_TOOL` type) would be a dataset-schema change out of this wave's scope. Decision:

| Primitive | Top-level `ActionType` | `is_planar_2d` |
|---|---|---|
| `Push` | `MOVE_XY` | `True` |
| `SweepWall` | `SWEEP` | `False` |
| everything else (`MoveTo`, `PlaceTool`, `LowerTool`, `RegraspTool`, `RefreshMask`,
  `CheckToolGrip`) | `OTHER` | `False` |

Every top-level action's `description` is `primitive.describe()`; `action_details` is
`dataclasses.asdict(primitive)` (the primitive's own fields verbatim). Child (per-order) actions
use the same mapping `grasper.py::action_type_from_order`/`action_details_from_order` used, with
one extension: `OrderType.GRIPPER` (the Wave 1 collapsed order type) is split back into
`ActionType.GRIPPER_OPEN`/`GRIPPER_CLOSE` by opening value (`>= 0.5` → open) so the two-member
legacy split survives despite the order type itself only having one gripper member. Unlike legacy
(`grasper.py::queue_orders`'s `if action_type is not ActionType.GRIPPER_CLOSE:` — skipping action
tracking for gripper orders entirely), **every** order here becomes a tracked child action,
including gripper opens/closes (`RegraspTool`'s scripted sequence needs them tracked like any
other order). Logged in `design/IMPLEMENTATION_LOG.md`.

## Freshness / `LowerTool` guard design (logged in `design/IMPLEMENTATION_LOG.md`)

Two independent mechanisms, both consulting `occupancy_supplier` (a `SegmentationWorker`-shaped
duck type: `latest()` / `await_result(min_source_seq, timeout)`):

1. **Generic freshness gate** (`Executor._enforce_freshness_gate`, called once per primitive before
   its top-level `Action` is even opened): if `occupancy_supplier` is configured and
   `FreshnessPolicy.requires_fresh(primitive)` is `True` (i.e. the primitive's category —
   `"lower_tool"`/`"placement"`/`"wall_check"` — is in `perception.freshness_require_zero_for`),
   await a result with `source_seq >= source.latest().seq`, raising `StaleMaskTimeout` on timeout.
   No-op otherwise (including when `occupancy_supplier is None`).
2. **`LowerTool`-specific guard** (`Executor._expand_lower_tool`, always runs whenever `guarded=
   True` and `occupancy_supplier` is configured, regardless of what the generic gate did): fetches
   its own occupancy result (fresh-if-required-by-policy, else best-effort `latest()`) and runs
   `tool_footprint_clear` at the pose the tool is about to lower at. This runs both for a
   standalone `LowerTool` primitive and for the one nested inside `PlaceTool`'s expansion (whose
   approach motion changes the pose between the generic gate and the actual lower).

For a standalone guarded `LowerTool` whose category is configured as freshness-required, this means
two independent occupancy fetches (the generic gate, then the guard's own) — a deliberate, harmless
duplication favoring one self-contained, always-correct guard implementation over conditionally
skipping it based on which primitive triggered the generic gate.

## Data flow

```
planning.types.Plan(primitives, meta)
    │
    ▼
Executor.run_plan(plan, phase)
    │  for each primitive:
    │    _enforce_freshness_gate(primitive)         -- may raise StaleMaskTimeout
    │    tracker.start_action(..., parent_id=None)  -- top-level Action opens
    │    _dispatch_primitive(...)                    -- _expand_* method
    │        │  for each order:
    │        │    safety.validate_order(order, current_z=..., allow_rack=...)
    │        │    tracker.start_action(..., parent_id=top_id)  -- child Action opens
    │        │    robot.move_xy/move_z/rotate/set_gripper(...)  -- CommandReceipt
    │        │    _sleep_with_shutdown(time_between_orders, shutdown_event)
    │        │    tracker.end_action(child_id, ...)   -- child Action closes, callbacks fire
    │        │    order_sink(record)                  -- optional persistence
    │        ▼
    │    source.await_next(...)  -- fresh end-of-primitive observation
    │    tracker.end_action(top_id, ...)  -- top-level Action closes, callbacks fire
    ▼
PlanResult(completed, aborted, error=None)
```

## Threading

`Executor` is not internally synchronized — it is meant to be driven by exactly one thread (the
session runner's single-threaded control loop, design 02 §3.5), the same way legacy's
`run_grasping` loop drove `queue_orders`. It reads from `source`/`occupancy_supplier` (both
independently thread-safe per their own docs) and writes to `robot`/`tracker` under the assumption
of a single writer (matching both `RobotInterface` implementations' and `ActionTracker`'s own
documented threading models). `execution.safety.SafetyValidator`/`tool_footprint_clear` are pure
and safe to call from any thread; `execution.actions.ActionTracker` is internally thread-safe (see
its own section above) since `get_action_for_frame`/etc. are meant to be read concurrently (e.g. by
a recorder assigning frame indices on a different thread — Wave 5).

## Config keys consumed

- `Executor` reads `config.experiment.time_between_orders` (any duck-typed object with that
  attribute path works — see the tests' `SimpleNamespace` stand-ins).
- `SafetyValidator`/`Executor` read workspace heights/margins off `workspace.config`
  (`config_schema.WorkspaceConfig`): `grasp_height`, `clearance_height`, `sweep_height`,
  `safety_margin`.
- Neither reads `perception.*`/`tool_check.*` directly — those are baked into the injected
  `occupancy_supplier`/`tool_grip_checker` objects by whoever constructs them (Wave 5's
  composition root).

## Porting notes (source legacy files, intentional behavior changes)

See `execution/executor.py`'s module docstring and each `_expand_*` method's docstring for the
exact legacy call site each expansion ports; summary of every intentional/uncertain decision is in
`design/IMPLEMENTATION_LOG.md`, highlights:

- `execution/executor.py` and `execution/safety.py` import `autograsper.planning.types`/
  `autograsper.planning.workspace` — see "Architectural note" above.
- `PlanResult.error` is always `None`; failures raise instead (see "`PlanResult` semantics" above).
- Every order becomes a tracked child action, including gripper orders (legacy skipped tracking
  gripper orders entirely) — see "Primitive → top-level `ActionType` mapping" above.
- `SafetyValidator`'s manipulation-boundary check applies uniformly to every "low" `MOVE_XY`
  order, including `SweepWall`'s approach/sweep passes near the fence walls. A real deployment's
  `workspace.manipulation_boundary_robot` must be sized to include the fence-wall reach (not just
  a tighter "random push sampling only" region — legacy's own comment calls this the "hardcoded
  safe X,Y ranges (example, adjust to your setup)"), since sweep positions are already
  geometrically bounded by `build_fence_walls`'s `t_min`/`t_max` (which itself accounts for the
  fence extent and a safety margin) and should legitimately fall inside a correctly sized
  manipulation boundary. This wave's tests use a wide `(0.0, 1.0)` boundary for
  `SweepWall`/general-choreography tests and a tighter one only for `SafetyValidator`'s own
  dedicated boundary-violation unit tests.
- `current_z=None` (unknown) is treated as `0.0` in `SafetyValidator.validate_order` — the
  conservative "assume low" choice, since `Executor` always tracks and passes the last commanded
  z; `None` should only occur if some future caller invokes `SafetyValidator` directly without that
  context.
- `NeedsHumanHelp` is defined in `execution/errors.py` per the design 02 §3.5 vocabulary but not
  raised anywhere in this wave — reserved for the session layer (Wave 5) to escalate into (e.g.
  after N repeated `ToolLost` failures from `RegraspTool`).

## Testing

`autograsper/tests/test_execution_actions.py` — parent/child simultaneous tracking, `end_action`
matching child before top-level, `get_action_for_frame`'s child-preference (in-flight and
completed), `to_dict`/`from_dict` round-tripping `parent_id` (and tolerating its absence in old
data), completion callbacks firing outside the lock (including a reentrant-call proof) and
surviving one callback raising.

`autograsper/tests/test_execution_safety.py` — `validate_order`'s z-floor / manipulation-boundary /
`allow_rack` / unknown-`current_z` behaviors, and `tool_footprint_clear` on an empty mask, a
directly-overlapping obstacle, and an obstacle within footprint range but below the clearance
threshold.

`autograsper/tests/test_execution_executor.py` — exact order-sequence expansion for every
primitive (verified against `DryRunRobot.command_log`, including both `SweepWall` modes, computed
against the same `sample_tool_pose` helper the executor itself uses so the assertions track the
real geometry rather than hand-copied numbers), action-tree correctness (parent/child linkage,
monotonic frames, `Push`'s complete `action_details`+robot-state+frame data), the freshness gate
(timeout → `StaleMaskTimeout`; fresh result → proceeds), the `LowerTool` guard (clear mask →
proceeds; obstacle → `UnsafeLower`; `occupancy_supplier=None` → proceeds unguarded),
`RegraspTool`/`CheckToolGrip` with a stub grip checker (ok and `ToolLost` paths), `order_sink`
record shape (and exception-swallowing), and `run_plan`/`PlanResult` (full completion,
clean-shutdown abort, typed-error propagation with actions closed cleanly). Run:

```
cd /home/alon/Code/cloudgripper-api
/home/alon/anaconda3/envs/cge/bin/python -m pytest autograsper/tests/test_execution_actions.py autograsper/tests/test_execution_safety.py autograsper/tests/test_execution_executor.py -q
```
