"""Shared building blocks for `random_push_planner.RandomPushPlanner` and
`seg_push_planner.SegPushPlanner` — both re-express the same underlying push/reset policy
(design 03 §3: "The policy logic is *mostly identical* to `random_push_planner` — same walls,
same placement search, same random pushes — minus the caution choreography"). Factored out here
so the two planner modules stay thin and the shared math has one place to fix bugs, rather than
being copy-pasted twice.

Not a public planning API surface on its own (leading underscore); both planner modules import
these directly. Pure functions of their arguments — no robot calls, no I/O, no global RNG use
(every source of randomness is the caller-supplied `np.random.Generator`), per CONVENTIONS.md /
design 02 §3.3.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from autograsper.planning.types import Push, SweepWall
from autograsper.planning.workspace import (
    Wall,
    Workspace,
    find_tool_placements,
    get_pos_sweep_from_optimal,
)

# Candidate tool angles tried by the placement search — ported verbatim from
# `custom_graspers/granular_pusher.py::perform_task`'s
# `find_tool_placements(self.latest_mask, self.image_space_tool_dimensions,
# [0,30,45,60,90,120,135,150], ...)` call.
PLACEMENT_ANGLES_DEG: Tuple[float, ...] = (0, 30, 45, 60, 90, 120, 135, 150)

# Sweep schedule: 3 equally spaced tangent-parameter passes along a wall, matching legacy's fixed
# `np.linspace(wall.t_min, wall.t_max, num=3)` (both `sweep_wall` branches use this exact call).
_SWEEP_PASSES = 3

# Shared defaults for both planner modules (`random_push_planner.py`/`seg_push_planner.py`) —
# collected here rather than duplicated in each:
# - `DEFAULT_RESET_STEP_SIZE`: legacy `RandomPushGrasper.__init__`'s hardcoded
#   `self.reset_step_size = 0.15` (not config-driven in legacy either).
# - `DEFAULT_MIN_GRANULE_SIZE`: legacy `config.get('Granuler_detection',
#   {}).get('min_granule_size', 300)`'s fallback, used when
#   `config.perception.background_diff` isn't populated (e.g. the `yolo`/segmenter-native
#   pipeline has no reason to configure it) — inconsequential in practice since
#   `workspace.check_wall_reset_needed`'s `min_granule_size` parameter is dead in the ported math
#   too (see `workspace.py` module docstring).
# - `DEFAULT_MIN_CLEARANCE_PX`: legacy `find_tool_placements`'s default `MIN_CLEARANCE_PX=10`,
#   used by `RandomPushGrasper.perform_task`'s placement-search call (no explicit override there).
DEFAULT_RESET_STEP_SIZE = 0.15
DEFAULT_MIN_GRANULE_SIZE = 300
DEFAULT_MIN_CLEARANCE_PX = 10


def find_placement_pose(
    workspace: Workspace,
    mask_crop: "np.ndarray",
    *,
    min_clearance_px: int = 10,
    debug=None,
) -> Optional[Tuple[float, float, float]]:
    """Run the placement search over `mask_crop` and convert the result to a robot-frame
    `(x, y, angle_deg)` pose, or `None` if no placement was found.

    Ported from `custom_graspers/granular_pusher.py::perform_task`'s
    `find_tool_placements(...)` call + `self.pix2robtrans.pix_to_robot(*pos_pixel)` conversion.
    """
    placement = find_tool_placements(
        mask_crop,
        workspace.tool_dims_px,
        PLACEMENT_ANGLES_DEG,
        workspace.manipulation_boundary_px,
        min_clearance_px=min_clearance_px,
        debug=debug,
    )
    if placement is None:
        return None
    px_x, px_y = placement["pos_px"]
    x, y = workspace.frames.crop_px_to_robot(px_x, px_y)
    return (x, y, float(placement["angle"]))


def sample_pushes(
    rng: np.random.Generator,
    workspace: Workspace,
    n_pushes: int,
    start_x: float,
    start_y: float,
    height: float,
) -> List[Push]:
    """Sample `n_pushes` chained `Push` primitives, uniform in the manipulation boundary, random
    angle in `[0, 180)` degrees, at fixed `height` — ported from
    `custom_graspers/granular_pusher.py::perform_task`'s push-sampling loop:

    ```python
    x = np.random.uniform(self.manip_x[0], self.manip_x[1])
    y = np.random.uniform(self.manip_y[0], self.manip_y[1])
    orientation = np.random.uniform(0, 180)
    orders.append((OrderType.MOVE_XY, [x, y]))
    orders.append((OrderType.ROTATE, [orientation]))
    ```

    Legacy sends `MOVE_XY(x, y)` (the actual push, executed at the tool's *current* orientation)
    then `ROTATE(orientation)` (preparing the angle for the *next* push) — i.e. push *i*'s
    on-the-ground contact angle is whatever the *previous* iteration's `ROTATE` set (or the
    placement angle, for the first push), while the `orientation` sampled during iteration *i*
    only takes effect starting with push *i+1*. This module instead treats each `Push` as an
    atomic, explicit `(start, angle, end)` unit — the angle sampled for push *i* is stored as
    *that push's* `angle`, and consecutive pushes chain `start = previous.end`. This is a
    deliberate re-expression, not a literal transcription, logged in
    `design/IMPLEMENTATION_LOG_planning.md`: the *set* of positions/angles sent to the robot is
    the same multiset either way (each of the `n_pushes` sampled `(x, y, angle)` triples still
    gets sent to the robot exactly once, in the same order), and since rotation happens in place
    at a fixed `grasp_height`, the physical hazard/behavior is unchanged — only the bookkeeping of
    "which push claims which angle" shifts by one index. The executor is free to still send
    `ROTATE` before or after `MOVE_XY` for a given `Push`; that ordering is an execution-layer
    choice (design 02 §3.3: "ordering rules ... live in the executor, not the planner"), not a
    planning decision.
    """
    pushes: List[Push] = []
    cur_x, cur_y = start_x, start_y
    manip_x = workspace.manip_x
    manip_y = workspace.manip_y
    for _ in range(n_pushes):
        end_x = float(rng.uniform(manip_x[0], manip_x[1]))
        end_y = float(rng.uniform(manip_y[0], manip_y[1]))
        angle = float(rng.uniform(0, 180))
        pushes.append(
            Push(start_x=cur_x, start_y=cur_y, angle=angle, end_x=end_x, end_y=end_y, height=height)
        )
        cur_x, cur_y = end_x, end_y
    return pushes


def build_sweep_primitive(
    rng: np.random.Generator, wall: Wall, details: Dict[str, object], reset_step_size: float
) -> SweepWall:
    """Build the `SweepWall` primitive for a wall that `workspace.check_wall_reset_needed` flagged
    as needing a reset, from its returned `details` dict. Ported from
    `custom_graspers/granular_pusher.py::sweep_wall`'s two branches (targeted / fallback)."""
    t_values = tuple(np.linspace(wall.t_min, wall.t_max, num=_SWEEP_PASSES))
    # Legacy re-samples `reset_step_size * (1 + 0.5 * rand())` inside `sweep()`, once per `t` pass
    # (i.e. a fresh random jitter for each of the 3 passes). A `SweepWall` primitive is a single
    # immutable value, so this port samples ONE jittered step per `SweepWall` (per wall), from the
    # planner's own `rng`, rather than per pass — a deliberate simplification (logged in
    # `design/IMPLEMENTATION_LOG_planning.md`): it slightly reduces the amplitude variance across
    # a wall's 3 passes but changes nothing about safety (the step is still within the same
    # jittered range legacy used) and keeps the primitive a plain value the executor can replay
    # deterministically instead of re-invoking randomness at execution time.
    step = float(reset_step_size * (1 + 0.5 * rng.random()))

    if details.get("use_fallback", False):
        return SweepWall(
            wall_label=wall.label,
            mode="fallback",
            angle=wall.angle,
            sweep_dir=(float(wall.normal[0]), float(wall.normal[1])),
            step=step,
            t_values=t_values,
        )

    pos_robot = details["pos_robot"]
    pos_sweep = get_pos_sweep_from_optimal(np.array(pos_robot, dtype=float), wall, margin=0.01)
    return SweepWall(
        wall_label=wall.label,
        mode="targeted",
        angle=wall.angle,
        sweep_dir=(float(wall.normal[0]), float(wall.normal[1])),
        step=step,
        t_values=t_values,
        approach_x=float(pos_robot[0]),
        approach_y=float(pos_robot[1]),
        sweep_pos_x=float(pos_sweep[0]),
        sweep_pos_y=float(pos_sweep[1]),
    )
