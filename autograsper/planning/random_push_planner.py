"""`RandomPushPlanner` — the cautious (background-diff) push/reset policy, re-expressed as a pure
`Planner` (Wave 3b).

Design reference: `autograsper/design/02_proposed_architecture.md` §3.3 ("The current
`RandomPushGrasper` policy maps cleanly: `plan_task` = placement search + N random pushes;
`plan_reset` = per-wall band checks + sweeps; `needs_reset` = `check_reset_needed`.").

Ported from `custom_graspers/granular_pusher.py::RandomPushGrasper` — specifically `startup`
(minus the tool-grip color-threshold check and human-intervention loop, which stay
execution/perception-layer concerns — see `planning.types.CheckToolGrip`/`RegraspTool`),
`perform_task`, `reset_task`, `sweep_wall`/`sweep` (re-expressed as `SweepWall` primitives — see
`planning.types.SweepWall`'s docstring for the exact choreography mapping), and
`update_mask_and_process` (re-expressed as the `RefreshMask` primitive; the actual move-aside +
recapture + mask computation is an execution/perception-layer concern here, not a planning one).

This planner is **pure**: every method is `WorldState -> Plan`/`bool`, no robot calls, no
`time.sleep`, no `cv2` GUI/`imwrite`, no `shutdown_event` (CONVENTIONS.md, design 02 §3.3/§3.5).
All randomness goes through the caller-supplied `np.random.Generator` (never the numpy global
RNG), so a seeded planner is fully reproducible.
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
from autograsper.planning.types import CheckToolGrip, PlaceTool, Plan, RefreshMask, WorldState
from autograsper.planning.workspace import Workspace, check_center_reset_needed, check_wall_reset_needed

if TYPE_CHECKING:  # pragma: no cover - typing only
    from autograsper.config_schema import Config


class RandomPushPlanner:
    """The cautious push/reset policy for the background-diff perception pipeline. Implements
    `planning.planner.Planner` by duck typing (structural, no explicit inheritance needed)."""

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
        """Tool-grip check, then force a fresh mask capture before deciding `needs_reset`/
        `plan_task` (ported from `RandomPushGrasper.startup`'s
        `check_tool_grip()` + `update_mask_and_process()` calls; the human-intervention regrasp
        loop on a failed grip check is an execution/session-layer concern, triggered by whatever
        the executor reports back from `CheckToolGrip`, not something the planner decides)."""
        return Plan(
            primitives=(CheckToolGrip(), RefreshMask(move_aside=True)),
            meta={"phase": "startup"},
        )

    def needs_reset(self, w: WorldState) -> bool:
        """Ported from `RandomPushGrasper.startup`'s
        `check_reset_needed(self.latest_mask)` call. `False` (no reset — nothing to decide yet)
        if no occupancy mask is available."""
        if w.occupancy is None:
            return False
        return check_center_reset_needed(w.occupancy.crop_mask)

    def plan_task(self, w: WorldState) -> Plan:
        """Ported from `RandomPushGrasper.perform_task`: find a granule-free tool placement (or
        sweep a random wall first if none exists), then `n_pushes` random pushes, then force a
        mask refresh (`interaction_since_last_mask = True` in legacy is realized here as an
        explicit trailing `RefreshMask`)."""
        if w.occupancy is None:
            raise ValueError(
                "RandomPushPlanner.plan_task requires WorldState.occupancy to be populated "
                "(run plan_startup's RefreshMask first)"
            )
        mask = w.occupancy.crop_mask

        pose = find_placement_pose(
            self.workspace, mask, min_clearance_px=DEFAULT_MIN_CLEARANCE_PX, debug=self._debug
        )

        if pose is None:
            # No granule-free spot: sweep a random wall first (legacy:
            # `self.sweep_wall(random.sample(self.walls, 1))` when `find_tool_placements` returns
            # nothing). We don't have wall-band `details` here (no wall was checked against the
            # mask for reset need — legacy doesn't check either in this branch, it just sweeps),
            # so this uses the fallback choreography unconditionally.
            wall = self.workspace.walls[int(self.rng.integers(len(self.workspace.walls)))]
            sweep = build_sweep_primitive(self.rng, wall, {"use_fallback": True}, self.reset_step_size)
            return Plan(
                primitives=(sweep, RefreshMask(move_aside=True)),
                meta={"phase": "task", "reason": "no_placement"},
            )

        x, y, angle = pose
        primitives = [PlaceTool(x=x, y=y, angle=angle, lower_to=self.grasp_height)]
        primitives.extend(
            sample_pushes(self.rng, self.workspace, self.n_pushes, x, y, self.grasp_height)
        )
        primitives.append(RefreshMask(move_aside=True))
        return Plan(primitives=tuple(primitives), meta={"phase": "task", "reason": "placed"})

    def plan_reset(self, w: WorldState) -> Plan:
        """Ported from `RandomPushGrasper.reset_task`: for each wall, in random order, check if it
        needs a reset sweep; if so, emit a `SweepWall` + trailing `RefreshMask`.

        KNOWN ARCHITECTURE DIFFERENCE from legacy (logged in
        `design/IMPLEMENTATION_LOG_planning.md`): legacy's `reset_task` calls
        `update_mask_and_process()` (a real, synchronous mask recapture) *between* wall checks
        within the same loop, so wall *i+1*'s `check_wall_reset_needed` sees a mask reflecting
        wall *i*'s just-completed sweep. This planner is a pure function of a single `WorldState`
        snapshot — it cannot re-observe mid-`plan_reset()` — so every wall's check in one
        `plan_reset()` call is evaluated against the *same* (pre-reset) `w.occupancy.crop_mask`.
        The `RefreshMask` primitives are still emitted between sweeps (so the executor/session
        layer gets an updated observation for whatever comes *next*), but a single `plan_reset()`
        call may plan a sweep for a wall that a fresher mask would already show as clear. This
        converges the same way legacy's episode loop does regardless: the session state machine
        rechecks `needs_reset` on the next cycle (design 02 §3.5's `Resetting -> Active ->
        Evaluating -> Resetting` loop), so an over- or under-swept wall is corrected on the next
        pass rather than within this one call.
        """
        if w.occupancy is None:
            return Plan(primitives=(), meta={"phase": "reset", "reason": "no_occupancy"})
        mask = w.occupancy.crop_mask

        order = self.rng.permutation(len(self.workspace.walls))
        primitives = []
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
            primitives.append(build_sweep_primitive(self.rng, wall, details, self.reset_step_size))
            primitives.append(RefreshMask(move_aside=True))

        return Plan(primitives=tuple(primitives), meta={"phase": "reset"})
