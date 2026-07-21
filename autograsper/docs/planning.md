# `autograsper/planning/` — WorldState, Plan/Primitive, Workspace, push planners

Status: **implemented** (Wave 3b). Design reference:
[`../design/02_proposed_architecture.md`](../design/02_proposed_architecture.md) §3.3 (core scope)
and [`../design/03_segmenter_native_design.md`](../design/03_segmenter_native_design.md) §3 (seg
planner deltas).

## Purpose

The planning layer is a **pure policy**: `WorldState` in, `Plan` (a tuple of semantic
`Primitive`s) out. It replaces the mixed decision/actuation/perception/safety/state-machine logic
of `custom_graspers/granular_pusher.py::RandomPushGrasper` (design 01 §3.5, defect #11: "Planner
is not separable") with an isolated object containing no robot calls, no sleeps, no `cv2` GUI
calls, and no `shutdown_event` — everything I/O-shaped (robot commands, fresh perception reads,
retry/intervention policy) is an execution/session-layer concern (Wave 4/5).

This wave does **not** touch `autograsper/perception/` (occupancy providers were being built by a
parallel effort at the same time). Instead, `planning/types.py` defines `OccupancyLike`/
`ClumpStatsLike` as `typing.Protocol`s describing the exact shape the perception layer's
`OccupancyResult`/`ClumpStats` are expected to have; planning code never imports or
`isinstance`-checks against the real perception types, only reads the documented attributes.

## Public API

### `planning/types.py`

- `OccupancyLike` / `ClumpStatsLike` — `Protocol`s (see module docstring for the exact field
  list). Any object with these attributes satisfies them.
- `ToolStatus(held: Optional[bool], grip_quality: Optional[float])`.
- `WorldState(obs: Observation, occupancy: Optional[OccupancyLike], tool: ToolStatus, workspace:
  Workspace)` — frozen dataclass; `.staleness -> Optional[int]` is `obs.seq -
  occupancy.source_seq` (`None` if `occupancy` is `None`).
- `Primitive` — base class (not a dataclass itself; `describe() -> str`). Concrete primitives
  (all `@dataclass(frozen=True)`):

  | Primitive | Fields (all robot frame unless `_px`) | Notes |
  |---|---|---|
  | `MoveTo` | `x, y, z=None, angle=None` | generic move; ordering owned by the executor |
  | `PlaceTool` | `x, y, angle, lower_to` | placement-search pose + lower target |
  | `LowerTool` | `z, guarded=True` | `guarded` asks the executor for a mask-aware guard before `MOVE_Z` down |
  | `Push` | `start_x, start_y, angle, end_x, end_y, height` | one planar push; maps 1:1 to a transition-dataset row |
  | `SweepWall` | `wall_label, mode, angle, sweep_dir, step, t_values=(), approach_x/y=None, sweep_pos_x/y=None` | `mode` is `"targeted"` or `"fallback"` — see the class docstring for exactly which executor choreography each implies |
  | `RegraspTool` | (none) | scripted rack grasp; may raise `NeedsHumanHelp` (execution/session concern) |
  | `RefreshMask` | `move_aside=True` | background-diff pipeline only; never emitted by `SegPushPlanner` |
  | `CheckToolGrip` | (none) | move to inspection pose + grip-quality check is an execution/perception concern |

- `Plan(primitives: Tuple[Primitive, ...], meta: Dict[str, Any])` — `meta` keys used by this
  wave's planners: `"phase"` (`"startup"`/`"task"`/`"reset"`), `"reason"` (branch taken), and
  `"replan_after_each"` (see "Replanning contract" below).

### `planning/workspace.py`

- `Wall(label, origin, tangent, normal, t_min, t_max, angle)` — one fence wall, robot frame.
- `build_fence_walls(fence_center, fence_size, tool_length, tool_width, safety_margin) ->
  List[Wall]`, `sample_tool_pose(wall, tool_width, safety_margin, t=None) -> dict`,
  `get_pos_sweep_from_optimal(pos_optimal, wall, margin=0.02) -> np.ndarray` — ported verbatim
  from `custom_graspers/fence_utils.py` (same math, same angle conventions).
- `find_tool_placements(obstacle_mask_crop, tool_dims_px, angles_deg, search_region_px,
  min_clearance_px=10, debug=None) -> Optional[dict]` — placement search; `None` (not legacy's
  `{}`) when nothing is found.
- `check_wall_reset_needed(mask_crop, wall, tool_dims_px, min_granule_size, frames, debug=None) ->
  (bool, dict)` — band-occupancy check + free-spot search near a wall; `details["pos_robot"]` is
  added relative to legacy (the `frames: CoordinateFrames` parameter converts `pos_px` for the
  caller).
- `check_center_reset_needed(mask_crop) -> bool` — the workspace-central 30% heuristic (ported
  from `object_tracker/granular_utils.py::check_reset_needed`).
- `check_placement`, `make_tool_mask`, `in_region_pix` — supporting geometry, ported.
- `Workspace(config: WorkspaceConfig, frames: CoordinateFrames)` — `.walls`, `.wall(label)`,
  `.manip_x`/`.manip_y`/`.manipulation_boundary_robot` (robot frame),
  `.manipulation_boundary_px` (crop-pixel frame; derived from the robot-frame boundary via
  `frames.robot_to_crop_px` if `config.manipulation_boundary_px` isn't set), `.tool_dims_robot`,
  `.tool_dims_px`.

### `planning/planner.py`

- `Planner` — `Protocol` with `plan_startup`, `needs_reset`, `plan_task`, `plan_reset` (all
  `WorldState -> Plan`, except `needs_reset -> bool`), matching design 02 §3.3 verbatim.
- `NoGranulesDetected(Exception)` — raised by a planner when `occupancy` shows no material
  (design 02 §3.5's failure-policy table; ported from legacy's
  `shutdown_event.set()`-on-no-granules path as a typed exception instead).
- `FreshnessPolicy(freshness_require_zero_for: Iterable[str])` — `.requires_fresh(primitive) ->
  bool`, using an explicit primitive-class -> category mapping (`LowerTool -> "lower_tool"`,
  `PlaceTool -> "placement"`, `SweepWall -> "wall_check"`; every other primitive is never
  "fresh-required"). This mapping is a documented decision (not a mechanical
  lowercase-the-classname transform — the category names don't textually match any primitive's
  class name), see `design/IMPLEMENTATION_LOG_planning.md`.

### `planning/random_push_planner.py` / `planning/seg_push_planner.py`

- `RandomPushPlanner(config, workspace, rng, debug=None)` — the cautious, background-diff-era
  policy, re-expressing `custom_graspers/granular_pusher.py::RandomPushGrasper`.
- `SegPushPlanner(config, workspace, rng, debug=None)` — the segmenter-native policy (design 03),
  the never-written `SegGranularPusher`. Same placement search / push sampling / wall-reset math
  as `RandomPushPlanner` (shared via `planning/_push_policy.py`, an internal helper module), minus
  every `RefreshMask` and with an adaptive, one-wall-at-a-time `plan_reset`.

Both constructors take a `config` object exposing `.experiment.n_pushes`, `.workspace.
grasp_height`, and `.perception.background_diff.min_granule_size` (typically the full
`config_schema.Config`, but any duck-typed object with those attributes works — see the tests).

## Data flow

```
Observation + OccupancyLike + ToolStatus + Workspace
    │
    ▼
WorldState  (planning/types.py)
    │
    ▼
Planner.plan_startup / needs_reset / plan_task / plan_reset   (random_push_planner.py /
    │                                                           seg_push_planner.py)
    ▼
Plan(primitives, meta)
    │
    ▼
Execution layer (Wave 4) — expands each Primitive into RobotInterface calls,
    owns choreography/ordering/safety guards, tracks actions.
```

## Primitive semantics for the executor (Wave 4)

- **`MoveTo`**: raise-before-translate ordering is the executor's job, not the planner's.
- **`PlaceTool`**: expands to an approach (raise/translate/rotate) + `LowerTool(lower_to)`.
- **`LowerTool(guarded=True)`**: immediately before `MOVE_Z` down, verify the tool footprint
  region of the *current* mask is clear; raise `UnsafeLower` (an execution-layer exception, not
  defined in `planning/`) instead of moving down if not (design 02 §3.4, design 03 §3.3).
- **`Push`**: one `is_planar_2d` action; `(start_x, start_y) -> (end_x, end_y)` at `height`, tool
  held at `angle` throughout. The executor decides whether to send `ROTATE` before or after
  `MOVE_XY` for a given push (see `random_push_planner.py`'s `sample_pushes` docstring for why
  this ordering freedom is safe relative to legacy's slightly different bookkeeping).
- **`SweepWall`**: `mode="targeted"` expands to
  `(approach_x, approach_y) -> rotate(angle) -> lower(grasp_height) -> (sweep_pos_x, sweep_pos_y)`
  then `len(t_values)` direct sweep passes (no cautious pre-sweep dance — legacy's
  `already_at_wall=True` path). `mode="fallback"` has no approach fields; its **first** pass gets
  legacy's cautious multi-height pre-sweep dance (`already_at_wall=False`), subsequent passes are
  direct. See `SweepWall`'s docstring in `types.py` for the exact legacy call this maps to.
- **`RegraspTool`**/**`CheckToolGrip`**/**`RefreshMask`**: thin markers; the actual robot
  choreography (fixed poses, ROI/color thresholds, move-aside-and-recapture) is execution/
  perception-layer configuration (`tool_check.*`, `perception.crop`/`.grid`), not a planning
  decision — the planner only sequences *that* one of these should run here.

## Replanning contract for the session layer (Wave 5)

- `RandomPushPlanner.plan_reset` is a **one-shot batch**: it evaluates every wall against the
  *same* `WorldState.occupancy` snapshot and returns `SweepWall`/`RefreshMask` pairs for every
  wall that needed it. This mirrors the fact that refreshing a mask is expensive in the
  background-diff pipeline (a real move-aside round trip) — see the "KNOWN ARCHITECTURE
  DIFFERENCE" note in `random_push_planner.py`'s `plan_reset` docstring for the one behavior gap
  this introduces relative to legacy (which recaptured the mask *between* wall checks within one
  `reset_task()` call). The session layer doesn't need special handling: it just calls
  `needs_reset` again next cycle, same as legacy's own episode loop.
- `SegPushPlanner.plan_reset` returns **at most one** `SweepWall`, tagged
  `Plan.meta["replan_after_each"] = True`. The session layer must, whenever it sees that flag:
  execute the returned plan (0 or 1 primitives), then re-derive `WorldState` (fresh occupancy is
  cheap for the segmenter-native pipeline) and call `plan_reset` again, looping until
  `needs_reset(w)` is `False` or the plan comes back empty. This is how design 03 §3.3's "adaptive
  sweeps" (stop early / add passes per wall, instead of a fixed batch) is realized without the
  planner needing any internal state between calls.
- Neither planner's `plan_task` sets `replan_after_each` — one `plan_task()` call is always a
  complete episode's primitives.

## Config keys consumed

- `experiment.n_pushes`, `workspace.grasp_height`, `workspace.fence_center`/`.fence_size`/
  `.tool_length_robot`/`.tool_width_robot`/`.tool_dims_px`/`.safety_margin`/
  `.manipulation_boundary_robot`/`.manipulation_boundary_px`, `perception.background_diff.
  min_granule_size` (optional; falls back to `300`, matching legacy's
  `Granuler_detection.min_granule_size` default), `perception.freshness_require_zero_for` (read by
  `FreshnessPolicy`, constructed by the executor/session layer — not read by the planners
  themselves).
- `Workspace` additionally depends on `perception.frames.CoordinateFrames` (Wave 2) for
  `crop_px_to_robot`/`robot_to_crop_px` conversions.

## Porting notes (source legacy files, intentional behavior changes)

See the module docstrings of `planning/workspace.py`, `planning/random_push_planner.py`,
`planning/seg_push_planner.py`, and `planning/_push_policy.py` for the full per-function porting
notes (they cite exact legacy call sites). Summary of every intentional/uncertain decision is in
[`../design/IMPLEMENTATION_LOG_planning.md`](../design/IMPLEMENTATION_LOG_planning.md); highlights:

- `build_fence_walls`'s angle formula was **verified numerically** (not assumed): for the
  template fence, `top`/`bottom` walls get `angle=90`, `left`/`right` get `angle=0`.
- `find_tool_placements` returns `None` instead of legacy's inconsistent `{}`.
- `check_wall_reset_needed` drops the dead `margin_of_safety` parameter (never referenced in the
  legacy function body) but keeps `min_granule_size` (also unused in the ported math, matching
  legacy) since the task's specified signature includes it.
- `check_center_reset_needed` preserves the `occupancy_ratio = occupied_area / mask_area`
  (total-mask-area, not central-box-area) quirk verbatim — see `workspace.py`'s module docstring
  for exactly what this means behaviorally.
- `Push`'s single `(start, angle, end)` shape is a deliberate re-expression of legacy's
  interleaved `MOVE_XY`/`ROTATE` chaining (see `_push_policy.py::sample_pushes`'s docstring) —
  same multiset of commands sent, same physical hazard profile, different index-to-angle
  bookkeeping.
- `SweepWall.step` is sampled once per wall (from the planner's `rng`) rather than once per `t`
  pass like legacy's `sweep()` — see `_push_policy.py::build_sweep_primitive`'s docstring.
- No `DebugSink`-writable artifact is *required* — every ported function accepts an optional
  `debug: Optional[DebugSink] = None` and no-ops if omitted, matching Wave 2's convention.

## Testing

`autograsper/tests/test_planning_workspace.py` — `Wall`/`build_fence_walls` (4 walls, angle
values verified against direct computation), `sample_tool_pose` (`t` explicit and random stay
within `[t_min, t_max]`), `get_pos_sweep_from_optimal`, `find_tool_placements` (obvious free
region found; fully-occupied mask -> `None`), `check_wall_reset_needed` (piled-wall ->
`True`/`False` for opposite wall; empty mask -> `False`), `check_center_reset_needed`.

`autograsper/tests/test_planning_planners.py` — stub `OccupancyLike`/`ClumpStatsLike` objects
(no perception-provider import), `WorldState.staleness`, `FreshnessPolicy`, both planners'
`plan_startup`/`needs_reset`/`plan_task`/`plan_reset` shapes and edge cases (no-placement sweep
fallback, push count/bounds/chaining, wall-reset sweep emission, `SegPushPlanner`'s
`NoGranulesDetected` and absence of `RefreshMask` everywhere, `replan_after_each` tagging). Run:

```
cd /home/alon/Code/cloudgripper-api
/home/alon/anaconda3/envs/cge/bin/python -m pytest autograsper/tests/test_planning_workspace.py autograsper/tests/test_planning_planners.py -q
```
