"""`SafetyValidator` — order-level safety checks beyond hardware clipping, plus the `LowerTool`
mask-aware guard primitive (Wave 4).

Design reference: `autograsper/design/02_proposed_architecture.md` §3.4 ("Expands primitives to
orders; validates against `safety.py` limits (workspace bounds, z floor by region, rotation range)
before sending ... Mask-aware guard hooks: before executing `LowerTool`, ask perception for a
fresh-enough occupancy check under the tool footprint; on failure raise `UnsafeLower`") and
`autograsper/design/03_segmenter_native_design.md` §3.3 ("`LowerTool` gains a real-time guard:
immediately before MOVE_Z down, verify the tool footprint region of the *current* mask is clear").

`hardware.robot_interface.Order.validate()` (Wave 1) already clips xy/z/gripper values to `[0, 1]`
and normalizes rotation — that is "hardware clipping" and is not repeated here. This module adds
the *execution-layer* safety policy on top of an already-clipped order:

- **Manipulation-boundary check for low moves**: a `MOVE_XY` order is only checked against
  `workspace.manip_x`/`manip_y` (+ a configurable margin) when the commanded/current z is *below*
  a configurable `high_z_threshold` (default: `workspace.config.clearance_height`) — "low moves
  must stay inside manipulation boundary + margin; high moves may go anywhere" (task spec),
  including the legacy move-aside corner `(0.0, 1.0)` (`RefreshMask`'s choreography, always sent
  at `z=1.0 >= clearance_height`) and the tool-rack approach `(0.03, 0.49)` in `RegraspTool`
  (sent while still at `z=1.0`, i.e. also "high" — see `execution/executor.py`'s module docstring
  for why `RegraspTool` still passes `allow_rack=True` for every one of its orders regardless).
  `current_z` is the caller's responsibility to supply (the executor tracks the last commanded z);
  when omitted (`None`), this validator treats it as **unknown and therefore low** (`0.0`) — the
  conservative choice, logged in `design/IMPLEMENTATION_LOG.md`.
- **z floor**: a `MOVE_Z` order below `workspace.config.grasp_height` is rejected — "low moves"
  should never go lower than the grasp height during normal task/reset operation. The
  `RegraspTool` scripted rack sequence legitimately goes to `z=0.27 < grasp_height=0.34` (legacy
  `perform_grab_tool`); callers pass `allow_rack=True` to bypass *both* checks for that whole
  scripted sequence (a context flag, per the task spec: "make this a context flag").
- **`tool_footprint_clear`**: the actual mask-aware `LowerTool` guard math, reusing
  `planning.workspace.make_tool_mask` (rotated tool-footprint rasterization) +
  `planning.workspace.check_placement` (distance-transform overlap/clearance check) — the exact
  same geometry `planning.workspace.find_tool_placements` uses for placement search, now evaluated
  at one specific candidate pose instead of searched over a region. This is a free function (not a
  `SafetyValidator` method) since it only needs an `OccupancyResult`-shaped object + geometry
  inputs, no safety-policy state.

Threading: `SafetyValidator` is stateless after construction (reads `workspace`/its own
constructor-time thresholds only); `validate_order`/`tool_footprint_clear` are pure functions of
their arguments and safe to call from any thread. Only `execution.executor.Executor` is expected
to call these (design 02 §3.4: the executor is the one place that validates before sending).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from autograsper.execution.errors import ExecutionError
from autograsper.hardware.robot_interface import Order, OrderType
from autograsper.planning.workspace import Workspace, check_placement, make_tool_mask

logger = logging.getLogger(__name__)

_EPS = 1e-9


class SafetyValidator:
    """Order-level safety checks beyond hardware clipping (see module docstring)."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        boundary_margin: Optional[float] = None,
        high_z_threshold: Optional[float] = None,
    ) -> None:
        self._workspace = workspace
        cfg = workspace.config
        # Default boundary margin: workspace.config.safety_margin (the same margin
        # build_fence_walls/sample_tool_pose use for keeping the tool footprint off the fence
        # corners) — a reasonable "configurable safe boundary" default absent a dedicated config
        # field of its own; overridable per `SafetyValidator` instance. Logged in
        # `design/IMPLEMENTATION_LOG.md`.
        self._boundary_margin = boundary_margin if boundary_margin is not None else cfg.safety_margin
        self._high_z_threshold = (
            high_z_threshold if high_z_threshold is not None else cfg.clearance_height
        )
        self._grasp_height = cfg.grasp_height

    def validate_order(
        self,
        order: Order,
        *,
        current_z: Optional[float] = None,
        allow_rack: bool = False,
    ) -> None:
        """Raise `ExecutionError` if `order` violates the manipulation-boundary or z-floor policy.

        - `current_z`: the z the robot is (or will be) at when a `MOVE_XY` order executes — only
          relevant for `MOVE_XY` orders; ignored otherwise. `None` is treated as "unknown, assume
          low" (conservative).
        - `allow_rack`: bypasses both checks entirely, for the scripted `RegraspTool` sequence
          (see module docstring).

        Only `MOVE_XY`/`MOVE_Z` orders are checked; `ROTATE`/`GRIPPER` have no boundary/z-floor
        implications and always pass.
        """
        if allow_rack:
            return

        if order.type == OrderType.MOVE_Z:
            z = order.values[0]
            if z < self._grasp_height - _EPS:
                raise ExecutionError(
                    f"MOVE_Z to {z:.3f} is below the grasp-height floor "
                    f"({self._grasp_height:.3f}); pass allow_rack=True for scripted rack "
                    "sequences that legitimately go lower"
                )
        elif order.type == OrderType.MOVE_XY:
            z_ref = current_z if current_z is not None else 0.0
            if z_ref < self._high_z_threshold - _EPS:
                x, y = order.values
                (xmin, xmax) = self._workspace.manip_x
                (ymin, ymax) = self._workspace.manip_y
                m = self._boundary_margin
                if not (xmin - m <= x <= xmax + m and ymin - m <= y <= ymax + m):
                    raise ExecutionError(
                        f"MOVE_XY to ({x:.3f}, {y:.3f}) at z={z_ref:.3f} (below "
                        f"high_z_threshold={self._high_z_threshold:.3f}) is outside the "
                        f"manipulation boundary x={self._workspace.manip_x}, "
                        f"y={self._workspace.manip_y} (+/- margin {m:.3f})"
                    )


def tool_footprint_clear(
    occ_result,
    x_robot: float,
    y_robot: float,
    angle_deg: float,
    frames,
    tool_dims_px: Tuple[int, int],
    min_clearance_px: float,
) -> Tuple[bool, float]:
    """Is the tool footprint at robot pose `(x_robot, y_robot)`, oriented `angle_deg`, clear of
    occupied pixels in `occ_result.crop_mask`, with at least `min_clearance_px` clearance?

    Returns `(is_clear, clearance_px)`:
    - `is_clear=True` only if the rotated tool footprint (rasterized by
      `planning.workspace.make_tool_mask`) does not overlap any occupied `crop_mask` pixel AND the
      minimum distance-transform value under the footprint is `>= min_clearance_px`.
    - `clearance_px` is the measured minimum clearance (0.0 on direct overlap).

    `occ_result` is duck-typed against `perception.occupancy.OccupancyResult` (only `.crop_mask` is
    read) / `planning.types.OccupancyLike` — this module does not import either concrete type.
    `frames` is duck-typed against `perception.frames.CoordinateFrames` (only `.robot_to_crop_px`
    is used) so this module has no hard perception-layer import either.

    This is the primitive check `execution.executor.Executor` calls from its guarded-`LowerTool`
    expansion (see that module); it is exposed as a free function here (not a `SafetyValidator`
    method) since it needs no safety-policy state, only geometry inputs.
    """
    crop_mask = occ_result.crop_mask
    free = ~crop_mask.astype(bool)
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)

    tool_mask, r = make_tool_mask(tool_dims_px[0], tool_dims_px[1], angle_deg)
    cx, cy = frames.robot_to_crop_px(x_robot, y_robot)

    fits, clearance = check_placement(dist, tool_mask, cx, cy, r)
    if not fits:
        return False, clearance
    if clearance < min_clearance_px:
        return False, clearance
    return True, clearance
