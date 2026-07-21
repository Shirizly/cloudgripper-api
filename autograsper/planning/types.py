"""`WorldState`, `Plan`, `Primitive` value types + the `OccupancyLike`/`ClumpStatsLike` shared
contract (Wave 3b — planning layer).

Design reference: `autograsper/design/02_proposed_architecture.md` §3.3 (`WorldState`/`Plan`/
`Primitive` + the primitive table) and `autograsper/design/03_segmenter_native_design.md` §3
(seg-planner deltas: no `RefreshMask`, freshness instead of staleness bookkeeping, adaptive
sweeps).

IMPORTANT — cross-team contract, duck typing only
--------------------------------------------------
A parallel agent is implementing `autograsper/perception/` (the occupancy providers) at the same
time this module was written. This module does **not** import anything from
`autograsper.perception` (other than the already-complete, read-only `frames.py`/`docs/
perception.md`, and this module doesn't even need those). Instead, `OccupancyLike`/
`ClumpStatsLike` below are `typing.Protocol`s that describe the exact field shape the perception
layer's `OccupancyResult`/`ClumpStats` are expected to have (per design 02 §3.3 / design 03 §2.1
and the shared contract agreed for this wave). Any object with these attributes — the real
`OccupancyResult`, a test stub, a future provider's result type — satisfies the protocol. Planning
code must never do `isinstance(x, OccupancyResult)` or `from autograsper.perception... import
OccupancyResult`; it only ever reads `.source_seq`, `.grid_mask`, `.crop_mask`, `.stats`, etc.

Units / coordinate frames (see `docs/perception.md` for the full frame definitions):
- `grid_mask`: uint8 {0, 255}, canonical dataset grid frame (`perception.grid.height/.width`).
- `crop_mask`: uint8 {0, 255}, full-resolution crop frame (`perception.crop.center_px/.size`) —
  this is the frame `planning/workspace.py`'s ported `find_tool_placements`/
  `check_wall_reset_needed`/`check_center_reset_needed` operate on (matching legacy
  `object_tracker/granular_utils.py::process_image`'s un-downscaled mask).
- `centroids`: `(x, y)` pairs in `crop_px`.
- All primitive geometric fields are in the `robot` frame (normalized `[0, 1]^2` xy, robot z)
  unless the field name is suffixed `_px` (`crop_px`) — documented per-field below.

Threading: every type here is an immutable (`frozen=True`) dataclass or a `Protocol`; no shared
mutable state, safe to construct/read from any thread. Planners built on these types must be pure
(see `planner.py`, `random_push_planner.py`, `seg_push_planner.py` module docstrings): no robot
calls, no sleeps, no `cv2` GUI/`imwrite`, no `shutdown_event` — planning is world-state-in,
`Plan`-out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from autograsper.observation.types import Observation

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing workspace.py at runtime
    from autograsper.planning.workspace import Workspace


# ---------------------------------------------------------------------------
# Shared contract with the perception layer (duck-typed, not imported)
# ---------------------------------------------------------------------------


@runtime_checkable
class ClumpStatsLike(Protocol):
    """Shape of `perception.occupancy.ClumpStats` (connected-component stats on `grid_mask`)."""

    num_clumps: int
    total_area_px: int
    areas: Tuple[int, ...]
    centroids: Tuple[Tuple[float, float], ...]  # crop_px


@runtime_checkable
class OccupancyLike(Protocol):
    """Shape of `perception.occupancy.OccupancyResult` (design 02 §3.3, design 03 §2.1)."""

    source_seq: int  # Observation.seq this mask was computed from
    frame_index: Optional[int]
    timestamp: float
    grid_mask: np.ndarray  # uint8 {0, 255}, grid frame
    crop_mask: np.ndarray  # uint8 {0, 255}, crop frame
    instances: tuple
    stats: ClumpStatsLike


# ---------------------------------------------------------------------------
# WorldState
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolStatus:
    """Tool-in-hand status (from `perception.tool_grip` + robot state).

    - `held`: `True`/`False` if a grip-quality check has run and produced a verdict, `None` if no
      check has happened yet this run (e.g. before the first `CheckToolGrip` primitive executes).
    - `grip_quality`: raw grip-quality score in `[0, 1]` (see legacy
      `object_tracker/tool_user_utils.py::analyze_tool_grip`), or `None` if unavailable.
    """

    held: Optional[bool]
    grip_quality: Optional[float]


@dataclass(frozen=True)
class WorldState:
    """Immutable snapshot planners decide against (design 02 §3.3).

    - `obs`: the `Observation` this decision is based on (robot pose, frame indices, `seq`).
    - `occupancy`: latest available `OccupancyLike`, or `None` if no perception provider has
      produced one yet (always non-`None` after startup for the segmenter-native pipeline —
      design 03 §2.2 — but may be `None` transiently for the background-diff pipeline before the
      first mask capture).
    - `tool`: tool-in-hand status.
    - `workspace`: static geometry (fence walls, manipulation boundary, tool dims) — see
      `planning.workspace.Workspace`.
    """

    obs: Observation
    occupancy: Optional[OccupancyLike]
    tool: ToolStatus
    workspace: "Workspace"

    @property
    def staleness(self) -> Optional[int]:
        """`obs.seq - occupancy.source_seq`, or `None` if there is no occupancy yet.

        0 means the mask was computed from exactly this observation; a positive number counts how
        many observations have been published since. See `planner.FreshnessPolicy` for how this
        feeds the freshness rule (design 03 §3.2).
        """
        if self.occupancy is None:
            return None
        return self.obs.seq - self.occupancy.source_seq


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class Primitive:
    """Base class for all planning primitives (design 02 §3.3 primitive table).

    Primitives are semantic units, not raw orders — the executor (Wave 4) expands each into a
    sequence of `RobotInterface` calls, owning choreography/ordering/safety-guard details the
    planner does not need to know (e.g. "raise before translate", the multi-height cautious
    pre-sweep dance for a fallback `SweepWall`'s first pass). Every concrete primitive is a
    `@dataclass(frozen=True)` subclass; `describe()` gives a short human-readable summary for
    logging/`orders.json`-style traces.
    """

    def describe(self) -> str:  # pragma: no cover - overridden by every subclass
        raise NotImplementedError


@dataclass(frozen=True)
class MoveTo(Primitive):
    """Move to `(x, y)` [, raise/lower to `z`] [, rotate to `angle`]. Robot frame.

    - `x`, `y`: robot-normalized xy, `[0, 1]`.
    - `z`: robot-normalized height, `[0, 1]`, or `None` to leave z unchanged.
    - `angle`: tool rotation in degrees, or `None` to leave rotation unchanged.
    """

    x: float
    y: float
    z: Optional[float] = None
    angle: Optional[float] = None

    def describe(self) -> str:
        return f"MoveTo(x={self.x:.3f}, y={self.y:.3f}, z={self.z}, angle={self.angle})"


@dataclass(frozen=True)
class PlaceTool(Primitive):
    """Place the tool at a granule-free spot found by placement search, then lower it.

    Expands to an approach (raise/translate/rotate) + `LowerTool` (design 02 §3.3 table). Ported
    from `custom_graspers/granular_pusher.py::perform_task`'s
    `MOVE_XY(pos_world) / ROTATE(angle) / MOVE_Z(grasp_height)` sequence.

    - `x`, `y`: robot-normalized xy of the placement (from
      `workspace.find_tool_placements`'s `pos_px`, converted via
      `CoordinateFrames.crop_px_to_robot`).
    - `angle`: tool orientation in degrees (from the placement search's candidate angle).
    - `lower_to`: robot-normalized z to lower the tool to (`workspace.grasp_height` in practice).
    """

    x: float
    y: float
    angle: float
    lower_to: float

    def describe(self) -> str:
        return (
            f"PlaceTool(x={self.x:.3f}, y={self.y:.3f}, angle={self.angle:.1f}, "
            f"lower_to={self.lower_to:.3f})"
        )


@dataclass(frozen=True)
class LowerTool(Primitive):
    """Lower the tool to `z` (robot-normalized height).

    - `guarded`: if `True` (the default), the executor must verify the tool footprint region of
      the *current* occupancy mask is clear immediately before the `MOVE_Z` down (design 02 §3.4
      "mask-aware guard hooks"; design 03 §3.3 "`LowerTool` gains a real-time guard") and raise
      `UnsafeLower` (an execution-layer concern, not defined here) instead of moving down if not.
    """

    z: float
    guarded: bool = True

    def describe(self) -> str:
        return f"LowerTool(z={self.z:.3f}, guarded={self.guarded})"


@dataclass(frozen=True)
class Push(Primitive):
    """One planar push: move from `(start_x, start_y)` to `(end_x, end_y)` at fixed `height`,
    with the tool held at `angle` degrees throughout.

    Tracked by the executor as one `is_planar_2d` action ⇒ maps 1:1 to a dataset transition
    (design 02 §3.3 table; design 03 §5 online transition emission wants exactly
    `start`/`end`/`angle` per push). Ported from
    `custom_graspers/granular_pusher.py::perform_task`'s per-iteration
    `MOVE_XY(x, y) / ROTATE(orientation)` pair — re-expressed as an explicit, self-contained
    (start, angle, end) unit rather than legacy's implicit chaining through mutable orientation
    state; see `random_push_planner.py` module docstring for why this is behavior-preserving.

    All of `start_x`, `start_y`, `end_x`, `end_y`, `height` are robot-normalized `[0, 1]`; `angle`
    is degrees.
    """

    start_x: float
    start_y: float
    angle: float
    end_x: float
    end_y: float
    height: float

    def describe(self) -> str:
        return (
            f"Push(start=({self.start_x:.3f},{self.start_y:.3f}), "
            f"end=({self.end_x:.3f},{self.end_y:.3f}), angle={self.angle:.1f}, "
            f"height={self.height:.3f})"
        )


@dataclass(frozen=True)
class SweepWall(Primitive):
    """One full per-wall reset sweep (ported from
    `custom_graspers/granular_pusher.py::sweep_wall`/`sweep`).

    `mode` selects the choreography the executor expands this into:
    - `"targeted"`: a granule-free entry point was found near the wall
      (`workspace.check_wall_reset_needed`'s non-fallback branch). `approach_x/approach_y` is the
      free-space pose found by the placement search (`pos_optimal`, robot frame); `sweep_pos_x/
      sweep_pos_y` is the position after moving toward the wall to contact
      (`workspace.get_pos_sweep_from_optimal`). The executor approaches via
      `(approach_x, approach_y) -> rotate(angle) -> lower(grasp_height) -> (sweep_pos_x,
      sweep_pos_y)`, then sweeps at every `t` in `t_values`, all with `already_at_wall=True`
      choreography (legacy `sweep(..., already_at_wall=True)`: no cautious pre-sweep dance).
    - `"fallback"`: no free entry point existed
      (`workspace.check_wall_reset_needed`'s `use_fallback` branch); `approach_x/approach_y`/
      `sweep_pos_x/sweep_pos_y` are `None`. The executor's *first* `t` pass gets legacy's cautious
      multi-height pre-sweep dance (`already_at_wall=False`); subsequent passes are direct
      (`already_at_wall=True`), matching `sweep_wall`'s fallback loop
      (`already_at_wall = False` then flipped to `True` after the first iteration).

    - `wall_label`: which `Workspace` wall this targets (`"top"`/`"right"`/`"bottom"`/`"left"`).
    - `angle`: tool orientation in degrees for every pass (`Wall.angle` — legacy `sweep_wall` uses
      `wall.angle` for the `ROTATE` order, NOT the `tool_angle`/`details["angle"]` value
      `check_wall_reset_needed` also computes and returns for its internal free-space search; see
      `workspace.py` module docstring for that quirk).
    - `sweep_dir`: unit vector (robot frame `(dx, dy)`) the tool is pushed along during each pass
      — `Wall.normal` (legacy's `perpendicular_dir` from `sample_tool_pose`).
    - `step`: push distance along `sweep_dir`, robot-normalized units (legacy
      `reset_step_size * (1 + 0.5 * rand())`, sampled once per `SweepWall` by the planner's rng
      rather than once per `t` pass — a deliberate simplification, see
      `random_push_planner.py`/`design/IMPLEMENTATION_LOG_planning.md`).
    - `t_values`: tangent-parameter positions along the wall to sweep at (robot-normalized units,
      via `Wall.tangent`), computed by the planner as `np.linspace(wall.t_min, wall.t_max, 3)`
      (legacy's fixed 3-pass schedule; design 03 §3.3 flags this as an upgrade target for the
      segmenter-native planner, not changed at this wave).
    """

    wall_label: str
    mode: str  # "targeted" | "fallback"
    angle: float
    sweep_dir: Tuple[float, float]
    step: float
    t_values: Tuple[float, ...] = ()
    approach_x: Optional[float] = None
    approach_y: Optional[float] = None
    sweep_pos_x: Optional[float] = None
    sweep_pos_y: Optional[float] = None

    def describe(self) -> str:
        return (
            f"SweepWall(wall={self.wall_label!r}, mode={self.mode!r}, angle={self.angle:.1f}, "
            f"passes={len(self.t_values)})"
        )


@dataclass(frozen=True)
class RegraspTool(Primitive):
    """Scripted rack grasp (ported from
    `custom_graspers/granular_pusher.py::perform_grab_tool`). May raise `NeedsHumanHelp`
    (execution/session-layer concern, not defined here) if the post-grasp grip check still fails.
    """

    def describe(self) -> str:
        return "RegraspTool()"


@dataclass(frozen=True)
class RefreshMask(Primitive):
    """Move the tool out of frame and force a fresh occupancy capture (background-diff provider
    only — design 03 §1 removes this entirely for the segmenter-native planner). Ported from
    `custom_graspers/granular_pusher.py::update_mask_and_process`'s move-aside choreography.

    - `move_aside`: always `True` in the current ported behavior (kept as a field for forward
      compatibility with a future non-moving refresh, e.g. a cheap re-poll without repositioning).
    """

    move_aside: bool = True

    def describe(self) -> str:
        return f"RefreshMask(move_aside={self.move_aside})"


@dataclass(frozen=True)
class CheckToolGrip(Primitive):
    """Move to the fixed inspection pose and verify the tool is held correctly (ported from
    `custom_graspers/granular_pusher.py::startup`'s inspection-pose + `check_tool_grip` call). No
    fields: the inspection pose, ROI, and color thresholds are execution/perception-layer
    configuration (`tool_check.*`), not planning decisions. On failure the executor/session layer
    runs the human-intervention loop (`RegraspTool` + wait), not the planner.
    """

    def describe(self) -> str:
        return "CheckToolGrip()"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """An ordered sequence of primitives plus free-form metadata for the session/executor layer.

    `meta` is intentionally an open dict (not a fixed schema) so individual planners can attach
    whatever the session layer needs without changing this type; keys used by planners in this
    wave:
    - `"phase"`: `"startup"` | `"task"` | `"reset"` (which `Planner` method produced this plan).
    - `"reason"`: short string explaining a branch taken (e.g. `"no_placement"`).
    - `"replan_after_each"`: `True` on `SegPushPlanner.plan_reset`'s single-`SweepWall` plans — a
      signal (design 03 §3.3 "adaptive sweeps") that the session layer should re-invoke
      `plan_reset` with a fresh `WorldState` after executing this plan and check `needs_reset`
      again, looping until it returns `False`, rather than assuming one `plan_reset()` call
      resets every wall that needed it.
    """

    primitives: Tuple[Primitive, ...]
    meta: Dict[str, Any] = field(default_factory=dict)
