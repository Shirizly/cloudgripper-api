"""`Planner` protocol + `FreshnessPolicy` (Wave 3b).

Design reference: `autograsper/design/02_proposed_architecture.md` §3.3:

```python
class Planner(Protocol):
    def plan_startup(self, w: WorldState) -> Plan: ...
    def needs_reset(self, w: WorldState) -> bool: ...
    def plan_task(self, w: WorldState) -> Plan: ...       # one episode worth of primitives
    def plan_reset(self, w: WorldState) -> Plan: ...
```

and `autograsper/design/03_segmenter_native_design.md` §3.2 (the freshness rule that replaces
`RefreshMask`/`interaction_since_last_mask` for the segmenter-native planner):

> A primitive that *reads* the mask (placement search, wall band check, lower-tool guard) must
> use an `OccupancyResult` whose `source_seq` is newer than the completion of the last
> scene-touching primitive.

Implementations (`random_push_planner.RandomPushPlanner`, `seg_push_planner.SegPushPlanner`) must
be **pure**: world state in, `Plan` out — no robot calls, no `time.sleep`, no `cv2` GUI/`imwrite`,
no `shutdown_event` (design 02 §3.3/§3.5, CONVENTIONS.md). Any I/O-shaped decision (a robot
call's outcome, a fresh perception read) belongs to the executor/session layers (Wave 4/5), which
own retry/replan/intervention policy.
"""

from __future__ import annotations

from typing import Iterable, Protocol

from autograsper.planning.types import Plan, Primitive, WorldState


class NoGranulesDetected(Exception):
    """Raised by a planner when `WorldState.occupancy` shows no granular material present (design
    02 §3.5's failure-policy table lists this as one of the typed exceptions the session layer
    maps to retry/replan/intervention/abort). Ported semantically from
    `custom_graspers/granular_pusher.py::startup`'s `"Granular material not detected in
    workspace... shutdown_event.set()"` path — re-expressed as a typed exception instead of a
    global shutdown flag, per design 02 §3.5 ("Separated failure policy... No
    `shutdown_event.set()` inside behaviors")."""


class Planner(Protocol):
    """A pure policy: `WorldState` in, `Plan` out. See module docstring."""

    def plan_startup(self, w: WorldState) -> Plan:
        """Primitives to run at the start of an episode cycle (tool-grip check, mask refresh for
        the cautious pipeline, etc.) before deciding `needs_reset`/`plan_task`."""
        ...

    def needs_reset(self, w: WorldState) -> bool:
        """Should the session layer transition to `Resetting` (sweep granules back toward the
        workspace center) instead of running another task episode?"""
        ...

    def plan_task(self, w: WorldState) -> Plan:
        """One episode's worth of primitives (placement search + pushes)."""
        ...

    def plan_reset(self, w: WorldState) -> Plan:
        """Primitives to sweep granules away from the fence walls back toward the center."""
        ...


# ---------------------------------------------------------------------------
# FreshnessPolicy
# ---------------------------------------------------------------------------

# Maps a primitive's concrete class name to the freshness *category* it belongs to, matching
# `config_schema.PerceptionConfig.freshness_require_zero_for` / design 03 §6's
# `freshness.require_staleness_0_for: [lower_tool, placement, wall_check]`. This is NOT a generic
# "snake_case the class name" transform (the category names don't correspond letter-for-letter to
# any primitive's class name — `PlaceTool` bareword-lowered would be `place_tool`, not
# `placement`; `SweepWall` would be `sweep_wall`, not `wall_check`) — it is an explicit, documented
# mapping from primitive to "which mask-reading decision produced this primitive", decided here
# since neither the config schema nor the design docs spell out the mapping (logged in
# `design/IMPLEMENTATION_LOG_planning.md`):
#   - "lower_tool" -> `LowerTool` (design 02 §3.4's mask-aware guard immediately before `MOVE_Z`
#     down; design 03 §3.3 "LowerTool gains a real-time guard").
#   - "placement"  -> `PlaceTool` (its pose came from a placement search over the mask; design 02
#     §3.3 table: "PlaceTool(pose) | approach + LowerTool | pose from placement search").
#   - "wall_check"  -> `SweepWall` (its target/mode came from `check_wall_reset_needed`'s band
#     check over the mask).
_FRESHNESS_CATEGORY = {
    "LowerTool": "lower_tool",
    "PlaceTool": "placement",
    "SweepWall": "wall_check",
}


class FreshnessPolicy:
    """Should the executor demand a fresh (`staleness == 0`) occupancy mask before running a given
    primitive? Driven by `config_schema.PerceptionConfig.freshness_require_zero_for` (a list of
    category names, matched case-insensitively) via the `_FRESHNESS_CATEGORY` mapping above.

    This is a plain policy object, not a `Planner` method, because it is consulted by the
    executor (Wave 4) at primitive-dispatch time, not by the planner at plan-build time — the
    planner has already committed to *which* primitive to run; freshness governs *when* the
    executor is allowed to actually run it.
    """

    def __init__(self, freshness_require_zero_for: Iterable[str]) -> None:
        self._required = {s.strip().lower() for s in freshness_require_zero_for}

    def requires_fresh(self, primitive: Primitive) -> bool:
        """`True` if `primitive`'s freshness category is in the configured
        `freshness_require_zero_for` list (i.e. the executor must ensure `staleness == 0` before
        running it); `False` for primitives with no freshness category (e.g. `Push`,
        `RefreshMask`, `CheckToolGrip`) or when the category isn't in the configured list."""
        category = _FRESHNESS_CATEGORY.get(type(primitive).__name__)
        if category is None:
            return False
        return category in self._required
