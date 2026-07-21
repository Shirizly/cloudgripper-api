"""Typed exceptions raised by the execution layer (Wave 4).

Design reference: `autograsper/design/02_proposed_architecture.md` §3.5 ("Separated failure
policy... Components raise typed errors; only the session layer decides between retry, human
intervention, or abort. No `shutdown_event.set()` inside behaviors.") and §3.4 (`UnsafeLower`).

`execution.executor.Executor` raises these when a primitive cannot be executed safely or
successfully. None of them are caught anywhere in this wave — the session layer (Wave 5) is
expected to catch each subclass and map it to retry / replan / human-intervention-wait / abort per
its own policy; this module only defines the vocabulary both sides agree on.

Threading: plain exception classes, no shared state.
"""

from __future__ import annotations

from typing import Optional, Tuple


class ExecutionError(Exception):
    """Base class for every typed exception raised by the execution layer.

    Also raised directly (not via a subclass) by `execution.safety.SafetyValidator.validate_order`
    for generic order-level safety violations (manipulation-boundary / z-floor) that aren't one of
    the more specific conditions below — see that module's docstring for why a dedicated subclass
    wasn't introduced for those (logged in `design/IMPLEMENTATION_LOG.md`).
    """


class UnsafeLower(ExecutionError):
    """Raised when a guarded `LowerTool` primitive's real-time footprint check (design 02 §3.4,
    design 03 §3.3's "LowerTool gains a real-time guard") finds the tool footprint is not clear of
    granules at the commanded pose.

    - `pose`: `(x, y)` robot-frame xy the tool was about to lower at.
    - `clearance_px`: the measured clearance in pixels (0.0 if the footprint directly overlapped
      an occupied pixel; otherwise the minimum distance-transform value under the footprint, which
      was below the configured safety threshold).
    """

    def __init__(self, pose: Tuple[float, float], clearance_px: float):
        self.pose = pose
        self.clearance_px = clearance_px
        super().__init__(
            f"UnsafeLower: tool footprint at pose={pose!r} has clearance {clearance_px:.2f}px, "
            "below the safety threshold"
        )


class ToolLost(ExecutionError):
    """Raised when a tool-grip check (`CheckToolGrip`, or the post-grasp check inside
    `RegraspTool`) reports a grip quality below the configured detection threshold.

    - `grip_quality`: the measured grip quality, `[0, 1]` (see
      `perception.tool_grip.GripCheckResult.quality`).
    """

    def __init__(self, grip_quality: float):
        self.grip_quality = grip_quality
        super().__init__(f"ToolLost: grip quality {grip_quality:.2f} below detection threshold")


class NeedsHumanHelp(ExecutionError):
    """Raised when a scripted recovery sequence cannot proceed without a human's intervention.

    Not raised anywhere by `execution.executor.Executor` in this wave — defined here per the
    design 02 §3.5 typed-exception vocabulary and reserved for the session layer (Wave 5), which
    is expected to escalate repeated `ToolLost` failures (e.g. `RegraspTool` retried N times with
    no improvement) into this exception as part of its own retry/intervention policy. Logged in
    `design/IMPLEMENTATION_LOG.md`.

    - `reason`: short human-readable explanation.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"NeedsHumanHelp: {reason}")


class RobotAPIError(ExecutionError):
    """Raised when a `RobotInterface` call itself raises an unexpected exception (e.g. a real
    `CloudGripperRobot`'s HTTP call failing, or any non-`ExecutionError` exception from `move_xy`/
    `move_z`/`rotate`/`set_gripper`) — wraps the original exception (via `raise RobotAPIError(...)
    from exc`) so callers can catch one execution-layer type regardless of the underlying
    transport."""


class StaleMaskTimeout(ExecutionError):
    """Raised when the freshness policy (design 03 §3.2: "A primitive that reads the mask ... must
    use an `OccupancyResult` whose `source_seq` is newer than the completion of the last
    scene-touching primitive. If not yet available, `await_result()`") requires a fresh
    (`staleness == 0`) occupancy result before running a primitive, and none becomes available
    within `timeout`.

    - `primitive`: the primitive class name that triggered the freshness requirement (e.g.
      `"LowerTool"`, `"PlaceTool"`, `"SweepWall"`).
    - `min_source_seq`: the `Observation.seq` the awaited occupancy result needed to be computed
      from (or newer) — i.e. what was passed to `occupancy_supplier.await_result(min_source_seq=)`.
    - `timeout`: how long (seconds) the executor waited before giving up (`None` if it would have
      blocked indefinitely, though the executor always passes a finite default).
    - `reason`: optional extra context (e.g. "no occupancy result available from supplier yet" for
      the guarded-`LowerTool` case where no result has ever been published).
    """

    def __init__(
        self,
        primitive: str,
        min_source_seq: int,
        timeout: Optional[float],
        reason: str = "",
    ):
        self.primitive = primitive
        self.min_source_seq = min_source_seq
        self.timeout = timeout
        self.reason = reason
        msg = (
            f"StaleMaskTimeout: no occupancy result with source_seq >= {min_source_seq} for "
            f"{primitive} within {timeout}s"
        )
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)
