"""Tests for `autograsper.planning.workspace` (Wave 3b).

Synthetic data only (no network/GPU/robot), per CONVENTIONS.md. Uses the same
tmp_path-homography-npz pattern as `tests/test_frames.py`.
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
from autograsper.perception.frames import CoordinateFrames
from autograsper.planning.workspace import (
    Workspace,
    build_fence_walls,
    check_center_reset_needed,
    check_wall_reset_needed,
    find_tool_placements,
    get_pos_sweep_from_optimal,
    sample_tool_pose,
)

# Template fence values from autograsper/granular-config.yaml.
_FENCE_CENTER = (0.5, 0.49)
_FENCE_SIZE = (0.94, 0.955)
_TOOL_LENGTH = 0.36
_TOOL_WIDTH = 0.017
_SAFETY_MARGIN = 0.01


def _synthetic_H(crop_w: int, crop_h: int) -> np.ndarray:
    """Affine-like homography crop_px (u, v) -> robot (x, y): x = u/crop_w, y = 1 - v/crop_h
    (matches the documented convention: robot y=1 is the top row, y=0 the bottom row) — same
    construction as `tests/test_frames.py::_synthetic_H`."""
    return np.array(
        [
            [1.0 / crop_w, 0.0, 0.0],
            [0.0, -1.0 / crop_h, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _make_workspace(tmp_path, crop_size=(300, 300), tool_dims_px=(8, 60)) -> Workspace:
    crop = CropConfig(center_px=(crop_size[0] // 2, crop_size[1] // 2), size=crop_size)
    grid = GridConfig(height=32, width=32)
    H = _synthetic_H(crop_size[0], crop_size[1])
    npz_path = tmp_path / "homography.npz"
    np.savez(npz_path, H)

    workspace_config = WorkspaceConfig(
        fence_center=_FENCE_CENTER,
        fence_size=_FENCE_SIZE,
        manipulation_boundary_robot=ManipulationBoundaryRobot(x=(0.1, 0.9), y=(0.1, 0.9)),
        tool_length_robot=_TOOL_LENGTH,
        tool_width_robot=_TOOL_WIDTH,
        tool_dims_px=tool_dims_px,
        homography_npz_path=str(npz_path),
        grasp_height=0.34,
        sweep_height=0.57,
        clearance_height=0.8,
        safety_margin=_SAFETY_MARGIN,
    )
    perception_config = PerceptionConfig(provider="none", crop=crop, grid=grid)
    frames = CoordinateFrames.from_config(workspace_config, perception_config)
    return Workspace(workspace_config, frames)


# --- build_fence_walls --------------------------------------------------------------


def test_build_fence_walls_returns_four_walls_with_expected_labels():
    walls = build_fence_walls(
        fence_center=_FENCE_CENTER,
        fence_size=_FENCE_SIZE,
        tool_length=_TOOL_LENGTH,
        tool_width=_TOOL_WIDTH,
        safety_margin=_SAFETY_MARGIN,
    )
    assert len(walls) == 4
    assert {w.label for w in walls} == {"top", "right", "bottom", "left"}


def test_build_fence_walls_angle_values_match_verified_legacy_math():
    """Legacy `build_fence_walls`'s angle formula
    (`angle = (atan2(tangent.y, tangent.x) + pi/2) % pi`, scaled to degrees, wrapped to
    `[0, 180)`) was evaluated directly (not assumed) for the template fence values: it produces
    `top=90`, `right=0`, `bottom=90`, `left=0` — the horizontal (top/bottom) walls get a 90-degree
    tool orientation and the vertical (left/right) walls get 0, i.e. the OPPOSITE pairing from
    what "top/bottom parallel to x-axis => angle 0" naive intuition would suggest. This is a
    numerically verified property of the formula, not a restatement of the task brief (which
    described the pairing the other way around) — see
    `design/IMPLEMENTATION_LOG_planning.md` for the verification.
    """
    walls = build_fence_walls(
        fence_center=_FENCE_CENTER,
        fence_size=_FENCE_SIZE,
        tool_length=_TOOL_LENGTH,
        tool_width=_TOOL_WIDTH,
        safety_margin=_SAFETY_MARGIN,
    )
    by_label = {w.label: w for w in walls}
    assert by_label["top"].angle == 90
    assert by_label["bottom"].angle == 90
    assert by_label["right"].angle == 0
    assert by_label["left"].angle == 0


def test_workspace_walls_match_build_fence_walls(tmp_path):
    workspace = _make_workspace(tmp_path)
    direct = build_fence_walls(
        fence_center=_FENCE_CENTER,
        fence_size=_FENCE_SIZE,
        tool_length=_TOOL_LENGTH,
        tool_width=_TOOL_WIDTH,
        safety_margin=_SAFETY_MARGIN,
    )
    assert len(workspace.walls) == 4
    for w1, w2 in zip(workspace.walls, direct):
        assert w1.label == w2.label
        assert w1.angle == w2.angle
    assert workspace.wall("top").label == "top"


# --- sample_tool_pose ----------------------------------------------------------------


def test_sample_tool_pose_explicit_t_stays_within_wall_range():
    walls = build_fence_walls(
        fence_center=_FENCE_CENTER,
        fence_size=_FENCE_SIZE,
        tool_length=_TOOL_LENGTH,
        tool_width=_TOOL_WIDTH,
        safety_margin=_SAFETY_MARGIN,
    )
    rng = np.random.default_rng(0)
    for wall in walls:
        for _ in range(10):
            t = float(rng.uniform(wall.t_min, wall.t_max))
            pose = sample_tool_pose(wall, tool_width=0.02, safety_margin=0.02, t=t)
            center = np.array([pose["x"], pose["y"]])
            tangential = np.dot(center - wall.origin, wall.tangent)
            normal_offset = np.dot(center - wall.origin, wall.normal)
            assert wall.t_min - 1e-9 <= tangential <= wall.t_max + 1e-9
            assert normal_offset == pytest.approx(0.02 / 2 + 0.02)
            assert pose["angle"] == wall.angle


def test_sample_tool_pose_random_t_stays_within_wall_range():
    walls = build_fence_walls(
        fence_center=_FENCE_CENTER,
        fence_size=_FENCE_SIZE,
        tool_length=_TOOL_LENGTH,
        tool_width=_TOOL_WIDTH,
        safety_margin=_SAFETY_MARGIN,
    )
    wall = walls[0]
    np.random.seed(42)  # sample_tool_pose(t=None) uses the numpy global RNG, verbatim from legacy
    for _ in range(20):
        pose = sample_tool_pose(wall, tool_width=0.02, safety_margin=0.02, t=None)
        center = np.array([pose["x"], pose["y"]])
        tangential = np.dot(center - wall.origin, wall.tangent)
        assert wall.t_min - 1e-9 <= tangential <= wall.t_max + 1e-9


def test_get_pos_sweep_from_optimal_moves_to_margin_distance():
    walls = build_fence_walls(
        fence_center=_FENCE_CENTER,
        fence_size=_FENCE_SIZE,
        tool_length=_TOOL_LENGTH,
        tool_width=_TOOL_WIDTH,
        safety_margin=_SAFETY_MARGIN,
    )
    wall = walls[0]  # top
    pos_optimal = wall.origin + wall.normal * 0.3  # somewhere well inside the workspace
    pos_sweep = get_pos_sweep_from_optimal(pos_optimal, wall, margin=0.02)
    dist_to_wall = np.dot(pos_sweep - wall.origin, wall.normal)
    assert dist_to_wall == pytest.approx(0.02, abs=1e-9)


# --- find_tool_placements --------------------------------------------------------------


def test_find_tool_placements_finds_obvious_free_region():
    mask = np.full((120, 120), 255, dtype=np.uint8)
    mask[40:80, 40:80] = 0  # a clear 40x40 free square in the middle

    placement = find_tool_placements(
        mask,
        tool_dims_px=(6, 20),
        angles_deg=[0],
        search_region_px=((0, 119), (0, 119)),
        min_clearance_px=3,
    )
    assert placement is not None
    x, y = placement["pos_px"]
    assert 40 <= x <= 80
    assert 40 <= y <= 80
    assert placement["clearance_px"] > 0


def test_find_tool_placements_fully_occupied_returns_none():
    mask = np.full((120, 120), 255, dtype=np.uint8)
    placement = find_tool_placements(
        mask,
        tool_dims_px=(6, 20),
        angles_deg=[0],
        search_region_px=((0, 119), (0, 119)),
        min_clearance_px=3,
    )
    assert placement is None


# --- check_wall_reset_needed -----------------------------------------------------------


def test_check_wall_reset_needed_true_for_wall_with_piled_material(tmp_path):
    workspace = _make_workspace(tmp_path, crop_size=(300, 300))
    mask = np.zeros((300, 300), dtype=np.uint8)
    # Top wall (robot y near 1) maps to LOW row indices (top of the image) per the documented
    # y-flip convention; pile material across the top rows.
    mask[0:40, :] = 255

    top_wall = workspace.wall("top")
    needed, details = check_wall_reset_needed(
        mask, top_wall, workspace.tool_dims_px, min_granule_size=300, frames=workspace.frames
    )
    assert needed is True
    assert details["wall"] is top_wall


def test_check_wall_reset_needed_false_for_opposite_wall(tmp_path):
    workspace = _make_workspace(tmp_path, crop_size=(300, 300))
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[0:40, :] = 255  # material piled at the top only

    bottom_wall = workspace.wall("bottom")
    needed, details = check_wall_reset_needed(
        mask, bottom_wall, workspace.tool_dims_px, min_granule_size=300, frames=workspace.frames
    )
    assert needed is False
    assert details == {}


def test_check_wall_reset_needed_empty_mask_is_false(tmp_path):
    workspace = _make_workspace(tmp_path)
    mask = np.zeros((300, 300), dtype=np.uint8)
    needed, details = check_wall_reset_needed(
        mask, workspace.wall("top"), workspace.tool_dims_px, min_granule_size=300, frames=workspace.frames
    )
    assert needed is False
    assert details == {}


# --- check_center_reset_needed ----------------------------------------------------------


def test_check_center_reset_needed_true_when_material_is_peripheral():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:10, 0:10] = 255  # far corner, well outside the central 30-70% box
    assert check_center_reset_needed(mask) is True


def test_check_center_reset_needed_false_when_material_is_central():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255  # inside the central 30-70% box
    assert check_center_reset_needed(mask) is False


def test_check_center_reset_needed_empty_mask_is_false():
    mask = np.zeros((100, 100), dtype=np.uint8)
    assert check_center_reset_needed(mask) is False
