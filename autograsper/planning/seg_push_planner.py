"""`SegPushPlanner` — the segmenter-native push/reset policy (no `RefreshMask`, adaptive resets).

Design reference: `autograsper/design/03_segmenter_native_design.md` §3 ("The policy logic is
*mostly identical* to `random_push_planner` — same walls, same placement search, same random
pushes — minus the caution choreography"). This is the never-written `SegGranularPusher` (design
01 §1's entry-point table: "the intended class was never committed"), expressed directly as a
`Planner` rather than another grasper subclass.

Deltas from `random_push_planner.RandomPushPlanner` (design 03 §3.1/§3.2/§3.3):
- No `RefreshMask` anywhere — the segmenter-native perception provider keeps a continuously fresh
  mask (design 03 §2.1's `SegmentationWorker`), so there is no move-aside round trip to plan.
- `plan_startup` additionally verifies granular material is present via `occupancy.stats`
  (`num_clumps == 0` -> raise `planner.NoGranulesDetected`), matching legacy `startup`'s "abort if
  no granules detected" path, ported as a typed exception instead of `shutdown_event.set()`.
- `plan_reset` emits **one** `SweepWall` per call (for one needy wall) with
  `Plan.meta["replan_after_each"] = True` instead of batching every needy wall against one
  snapshot (contrast `RandomPushPlanner.plan_reset`, which is a legitimate one-shot batch job for
  the cautious pipeline since refreshing a mask there is expensive). Design 03 §3.3 explicitly
  wants continuous re-verification ("needs_reset / wall checks after every episode become cheap,
  so they run every cycle" and "the planner can stop sweeping a wall early... instead of the
  fixed 3-pass schedule") — since masks are cheap here, the natural way to realize that at this
  layer (without adding planner-internal state) is: plan one wall, hand control back, let the
  session layer re-derive `WorldState` (a fresh mask is already available/cheap) and call
  `plan_reset` again, looping until `needs_reset()` is `False`. See `types.Plan`'s docstring for
  the `replan_after_each` contract.

Pure, per CONVENTIONS.md / design 02 §3.3/§3.5: no robot calls, no sleeps, no `cv2` GUI/`imwrite`,
no `shutdown_event`. All randomness goes through the caller-supplied `np.random.Generator`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from autograsper.observation.debug import DebugSink
from autograsper.planning._push_policy import (
    DEFAULT_MIN_CLEARANCE_PX,
    DEFAULT_MIN_GRANULE_SIZE,
    DEFAULT_RESET_STEP_SIZE,
    build_sweep_primitive,
    find_placement_pose,
    sample_pushes,
)
from autograsper.planning.planner import NoGranulesDetected
from autograsper.planning.types import CheckToolGrip, PlaceTool, Plan, WorldState
from autograsper.planning.workspace import Workspace, check_center_reset_needed, check_wall_reset_needed

if TYPE_CHECKING:  # pragma: no cover - typing only
    from autograsper.config_schema import Config


class SegPushPlanner:
    """The segmenter-native push/reset policy. Implements `planning.planner.Planner` by duck
    typing (structural, no explicit inheritance needed)."""

    def __init__(
        self,
        config: "Config",
        workspace: Workspace,
        rng: np.random.Generator,
        debug: Optional[DebugSink] = None,
    ) -> None:
        self._config = config
        self.workspace = workspace
        self.rng = rng
        self._debug = debug

        self.n_pushes = config.experiment.n_pushes
        self.grasp_height = config.workspace.grasp_height
        self.reset_step_size = DEFAULT_RESET_STEP_SIZE
        self._min_granule_size = (
            config.perception.background_diff.min_granule_size
            if config.perception.background_diff is not None
            else DEFAULT_MIN_GRANULE_SIZE
        )

    # -- Planner protocol -----------------------------------------------------

    def plan_startup(self, w: WorldState) -> Plan:
        """Tool-grip check only — no mask refresh (design 03 §3.1: "`plan_startup` shrinks to:
        tool-grip check pose -> grip check -> verify granules present in `occupancy.stats`").
        Raises `NoGranulesDetected` if the (always-available, per design 03 §2.2) occupancy shows
        no material, matching legacy `startup`'s abort path."""
        if w.occupancy is None or w.occupancy.stats.num_clumps == 0:
            raise NoGranulesDetected(
                "SegPushPlanner.plan_startup: occupancy stats show no granular material "
                f"({'no occupancy yet' if w.occupancy is None else 'num_clumps == 0'})"
            )
        return Plan(primitives=(CheckToolGrip(),), meta={"phase": "startup"})

    def needs_reset(self, w: WorldState) -> bool:
        """Same central-workspace heuristic as `RandomPushPlanner.needs_reset` — design 03 does
        not change `needs_reset`'s math, only how cheaply/often it can be called."""
        if w.occupancy is None:
            return False
        return check_center_reset_needed(w.occupancy.crop_mask)

    def plan_task(self, w: WorldState) -> Plan:
        """Same placement search + `n_pushes` random pushes as `RandomPushPlanner.plan_task`,
        minus the trailing `RefreshMask` (design 03 §3.1)."""
        if w.occupancy is None:
            raise NoGranulesDetected(
                "SegPushPlanner.plan_task: WorldState.occupancy is None (should never happen "
                "after a successful plan_startup per design 03 §2.2)"
            )
        mask = w.occupancy.crop_mask

        pose = find_placement_pose(
            self.workspace, mask, min_clearance_px=DEFAULT_MIN_CLEARANCE_PX, debug=self._debug
        )

        if pose is None:
            wall = self.workspace.walls[int(self.rng.integers(len(self.workspace.walls)))]
            sweep = build_sweep_primitive(self.rng, wall, {"use_fallback": True}, self.reset_step_size)
            return Plan(primitives=(sweep,), meta={"phase": "task", "reason": "no_placement"})

        x, y, angle = pose
        primitives = [PlaceTool(x=x, y=y, angle=angle, lower_to=self.grasp_height)]
        primitives.extend(
            sample_pushes(self.rng, self.workspace, self.n_pushes, x, y, self.grasp_height)
        )
        return Plan(primitives=tuple(primitives), meta={"phase": "task", "reason": "placed"})

    def plan_reset(self, w: WorldState) -> Plan:
        """Emit a `SweepWall` for (at most) ONE needy wall, tagged
        `meta["replan_after_each"] = True` — see module/`types.Plan` docstrings for why. If no
        wall needs a reset (e.g. `needs_reset()` was stale by the time this ran), returns an empty
        plan with the same tag so the session layer's loop terminates cleanly."""
        if w.occupancy is None:
            return Plan(primitives=(), meta={"phase": "reset", "replan_after_each": True})
        mask = w.occupancy.crop_mask

        order = self.rng.permutation(len(self.workspace.walls))
        for idx in order:
            wall = self.workspace.walls[int(idx)]
            needed, details = check_wall_reset_needed(
                mask,
                wall,
                self.workspace.tool_dims_px,
                self._min_granule_size,
                self.workspace.frames,
                debug=self._debug,
            )
            if not needed:
                continue
            sweep = build_sweep_primitive(self.rng, wall, details, self.reset_step_size)
            return Plan(
                primitives=(sweep,),
                meta={"phase": "reset", "wall": wall.label, "replan_after_each": True},
            )

        return Plan(primitives=(), meta={"phase": "reset", "replan_after_each": True})
