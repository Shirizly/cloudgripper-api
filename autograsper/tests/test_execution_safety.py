"""Tests for `autograsper.execution.safety` (Wave 4): `SafetyValidator.validate_order` (boundary +
z-floor checks beyond hardware clipping, the `allow_rack` bypass) and `tool_footprint_clear` (the
`LowerTool` real-time mask-aware guard's underlying geometry check).
"""

from __future__ import annotations

import numpy as np
import pytest

from autograsper.config_schema import (
    CropConfig,
    GridConfig,
    ManipulationBoundaryRobot,
    PerceptionConfig,
    WorkspaceConfig,
)
from autograsper.execution.errors import ExecutionError
from autograsper.execution.safety import SafetyValidator, tool_footprint_clear
from autograsper.hardware.robot_interface import Order, OrderType
from autograsper.perception.frames import CoordinateFrames
from autograsper.planning.workspace import Workspace

_CROP_SIZE = (300, 300)
_TOOL_DIMS_PX = (8, 40)


def _synthetic_H(crop_w: int, crop_h: int) -> np.ndarray:
    """Simple invertible homography: crop_px -> robot xy, robot y=1 at row 0 (top) per the
    documented y-axis convention (see `perception/frames.py`)."""
    return np.array(
        [
            [1.0 / crop_w, 0.0, 0.0],
            [0.0, -1.0 / crop_h, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _make_workspace(tmp_path) -> Workspace:
    crop = CropConfig(center_px=(_CROP_SIZE[0] // 2, _CROP_SIZE[1] // 2), size=_CROP_SIZE)
    grid = GridConfig(height=32, width=32)
    H = _synthetic_H(*_CROP_SIZE)
    npz_path = tmp_path / "homography.npz"
    np.savez(npz_path, H)

    workspace_config = WorkspaceConfig(
        fence_center=(0.5, 0.49),
        fence_size=(0.94, 0.955),
        manipulation_boundary_robot=ManipulationBoundaryRobot(x=(0.1, 0.9), y=(0.1, 0.9)),
        tool_length_robot=0.36,
        tool_width_robot=0.017,
        tool_dims_px=_TOOL_DIMS_PX,
        homography_npz_path=str(npz_path),
        grasp_height=0.34,
        sweep_height=0.57,
        clearance_height=0.8,
        safety_margin=0.01,
    )
    perception_config = PerceptionConfig(provider="none", crop=crop, grid=grid)
    frames = CoordinateFrames.from_config(workspace_config, perception_config)
    return Workspace(workspace_config, frames)


# --- validate_order: z floor -----------------------------------------------------------------


def test_move_z_below_grasp_height_raises(tmp_path):
    validator = SafetyValidator(_make_workspace(tmp_path))
    order = Order(OrderType.MOVE_Z, (0.2,)).validate()  # below grasp_height=0.34
    with pytest.raises(ExecutionError):
        validator.validate_order(order)


def test_move_z_at_or_above_grasp_height_ok(tmp_path):
    validator = SafetyValidator(_make_workspace(tmp_path))
    order = Order(OrderType.MOVE_Z, (0.34,)).validate()
    validator.validate_order(order)  # must not raise
    order = Order(OrderType.MOVE_Z, (0.8,)).validate()
    validator.validate_order(order)


def test_move_z_below_floor_allowed_with_allow_rack(tmp_path):
    validator = SafetyValidator(_make_workspace(tmp_path))
    order = Order(OrderType.MOVE_Z, (0.27,)).validate()  # RegraspTool's rack-grasp z
    validator.validate_order(order, allow_rack=True)  # must not raise


# --- validate_order: manipulation-boundary check for low moves --------------------------------


def test_low_move_outside_boundary_raises(tmp_path):
    validator = SafetyValidator(_make_workspace(tmp_path))
    # manip boundary is x,y in [0.1, 0.9]; z below clearance_height (0.8) is "low".
    order = Order(OrderType.MOVE_XY, (0.02, 0.5)).validate()
    with pytest.raises(ExecutionError):
        validator.validate_order(order, current_z=0.34)


def test_low_move_inside_boundary_ok(tmp_path):
    validator = SafetyValidator(_make_workspace(tmp_path))
    order = Order(OrderType.MOVE_XY, (0.5, 0.5)).validate()
    validator.validate_order(order, current_z=0.34)  # must not raise


def test_high_move_to_legacy_corner_ok(tmp_path):
    validator = SafetyValidator(_make_workspace(tmp_path))
    # (0.0, 1.0) is the legacy RefreshMask move-aside corner, well outside the manip boundary --
    # only safe because it is always sent at high z (>= clearance_height).
    order = Order(OrderType.MOVE_XY, (0.0, 1.0)).validate()
    validator.validate_order(order, current_z=1.0)  # must not raise (high move)


def test_unknown_current_z_is_treated_as_low_conservatively(tmp_path):
    validator = SafetyValidator(_make_workspace(tmp_path))
    order = Order(OrderType.MOVE_XY, (0.0, 1.0)).validate()
    with pytest.raises(ExecutionError):
        validator.validate_order(order)  # current_z omitted -> conservative low-move check


def test_allow_rack_bypasses_boundary_check_too(tmp_path):
    validator = SafetyValidator(_make_workspace(tmp_path))
    order = Order(OrderType.MOVE_XY, (0.03, 0.49)).validate()  # tool-rack approach xy
    validator.validate_order(order, current_z=0.27, allow_rack=True)  # must not raise


def test_rotate_and_gripper_orders_are_never_checked(tmp_path):
    validator = SafetyValidator(_make_workspace(tmp_path))
    validator.validate_order(Order(OrderType.ROTATE, (45,)).validate())
    validator.validate_order(Order(OrderType.GRIPPER, (0.0,)).validate())


# --- tool_footprint_clear ----------------------------------------------------------------------


def test_tool_footprint_clear_on_empty_mask(tmp_path):
    workspace = _make_workspace(tmp_path)
    crop_mask = np.zeros((_CROP_SIZE[1], _CROP_SIZE[0]), dtype=np.uint8)
    occ = _StubOccupancy(crop_mask)
    ok, clearance = tool_footprint_clear(
        occ, 0.5, 0.5, 0.0, workspace.frames, workspace.tool_dims_px, min_clearance_px=10
    )
    assert ok is True
    assert clearance >= 10


def test_tool_footprint_blocked_by_obstacle_under_footprint(tmp_path):
    workspace = _make_workspace(tmp_path)
    crop_mask = np.zeros((_CROP_SIZE[1], _CROP_SIZE[0]), dtype=np.uint8)
    cx, cy = workspace.frames.robot_to_crop_px(0.5, 0.5)
    crop_mask[cy - 5 : cy + 5, cx - 5 : cx + 5] = 255  # a blob right under the footprint center
    occ = _StubOccupancy(crop_mask)
    ok, clearance = tool_footprint_clear(
        occ, 0.5, 0.5, 0.0, workspace.frames, workspace.tool_dims_px, min_clearance_px=10
    )
    assert ok is False
    assert clearance == 0.0


def test_tool_footprint_blocked_when_clearance_below_threshold(tmp_path):
    workspace = _make_workspace(tmp_path)
    crop_mask = np.zeros((_CROP_SIZE[1], _CROP_SIZE[0]), dtype=np.uint8)
    cx, cy = workspace.frames.robot_to_crop_px(0.5, 0.5)
    # A blob just outside the footprint but close enough that clearance < a very large threshold.
    crop_mask[cy + 25 : cy + 27, cx - 1 : cx + 1] = 255
    occ = _StubOccupancy(crop_mask)
    ok, clearance = tool_footprint_clear(
        occ, 0.5, 0.5, 0.0, workspace.frames, workspace.tool_dims_px, min_clearance_px=10_000
    )
    assert ok is False
    assert clearance < 10_000


class _StubOccupancy:
    """Minimal `OccupancyLike` stand-in: only `.crop_mask` is read by `tool_footprint_clear`."""

    def __init__(self, crop_mask: np.ndarray):
        self.crop_mask = crop_mask
