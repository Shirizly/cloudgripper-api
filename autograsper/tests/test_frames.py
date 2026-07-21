"""Tests for `autograsper.perception.frames` (`CameraPipeline`, `CoordinateFrames`) — Wave 2.

Synthetic data only (no network/GPU/robot), per CONVENTIONS.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from autograsper.config_schema import CropConfig, GridConfig
from autograsper.perception.frames import CameraPipeline, CoordinateFrames, FramesError

# Real camera intrinsics/distortion from autograsper/granular-config.yaml (fisheye, 4 coeffs) —
# not calibration secrets, just the checked-in template values; used here to exercise the real
# fisheye branch of CameraPipeline rather than a degenerate stand-in.
_TEMPLATE_M = [
    [505.24537524391866, 0.0, 324.5096286632362],
    [0.0, 505.6456651337437, 233.54118730278543],
    [0.0, 0.0, 1.0],
]
_TEMPLATE_D = [
    -0.07727407195057368,
    -0.047989733519315944,
    0.12157420705123315,
    -0.09667542135039282,
]


# --- CameraPipeline -----------------------------------------------------------------


def test_camera_pipeline_process_bottom_preserves_shape_and_channels():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)

    pipeline = CameraPipeline(m=_TEMPLATE_M, d=_TEMPLATE_D, H=np.eye(3))
    out = pipeline.process_bottom(image)

    assert out is not None
    assert out.ndim == 3
    assert out.shape[2] == 3
    # calibration.py::undistort's baked-in 90-degree rotation swaps height/width.
    assert out.shape[0] == image.shape[1]
    assert out.shape[1] == image.shape[0]


def test_camera_pipeline_process_bottom_none_is_passthrough():
    pipeline = CameraPipeline(m=_TEMPLATE_M, d=_TEMPLATE_D, H=np.eye(3))
    assert pipeline.process_bottom(None) is None


def test_camera_pipeline_process_top_is_passthrough():
    pipeline = CameraPipeline(m=_TEMPLATE_M, d=_TEMPLATE_D, H=np.eye(3))
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert pipeline.process_top(image) is image


def test_camera_pipeline_without_homography_still_undistorts():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    pipeline = CameraPipeline(m=_TEMPLATE_M, d=_TEMPLATE_D, H=None)
    out = pipeline.process_bottom(image)
    assert out is not None
    assert out.shape[0] == image.shape[1]
    assert out.shape[1] == image.shape[0]


# --- CoordinateFrames: construction / errors -----------------------------------------


def _crop_grid(crop_size=(360, 360), grid_dim=(36, 36), center=(180, 180)):
    crop = CropConfig(center_px=center, size=crop_size)
    grid = GridConfig(height=grid_dim[0], width=grid_dim[1])
    return crop, grid


def _synthetic_H(crop_w=360, crop_h=360):
    """Affine-like homography mapping crop_px (u, v) -> robot (x, y):
    x = u / crop_w
    y = 1 - v / crop_h
    (matches the documented convention: robot y=1 is the top row, y=0 the bottom row.)
    """
    return np.array(
        [
            [1.0 / crop_w, 0.0, 0.0],
            [0.0, -1.0 / crop_h, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_coordinate_frames_rejects_non_3x3_matrix():
    crop, grid = _crop_grid()
    with pytest.raises(FramesError):
        CoordinateFrames(crop, grid, H_crop_to_robot=np.eye(2))


def test_coordinate_frames_from_config_missing_file_raises_clear_error(tmp_path):
    from autograsper.config_schema import WorkspaceConfig, ManipulationBoundaryRobot, PerceptionConfig

    crop, grid = _crop_grid()
    workspace = WorkspaceConfig(
        fence_center=(0.5, 0.5),
        fence_size=(0.9, 0.9),
        manipulation_boundary_robot=ManipulationBoundaryRobot(x=(0.1, 0.9), y=(0.1, 0.9)),
        tool_length_robot=0.36,
        tool_width_robot=0.017,
        tool_dims_px=(8, 120),
        homography_npz_path=str(tmp_path / "does_not_exist.npz"),
        grasp_height=0.34,
        sweep_height=0.57,
        clearance_height=0.8,
        safety_margin=0.02,
    )
    perception = PerceptionConfig(provider="none", crop=crop, grid=grid)

    with pytest.raises(FramesError, match="homography file not found"):
        CoordinateFrames.from_config(workspace, perception)


def test_coordinate_frames_from_config_loads_arr_0(tmp_path):
    from autograsper.config_schema import WorkspaceConfig, ManipulationBoundaryRobot, PerceptionConfig

    crop, grid = _crop_grid()
    H = _synthetic_H()
    npz_path = tmp_path / "homography.npz"
    np.savez(npz_path, H)  # positional arg -> key "arr_0", matching legacy format

    workspace = WorkspaceConfig(
        fence_center=(0.5, 0.5),
        fence_size=(0.9, 0.9),
        manipulation_boundary_robot=ManipulationBoundaryRobot(x=(0.1, 0.9), y=(0.1, 0.9)),
        tool_length_robot=0.36,
        tool_width_robot=0.017,
        tool_dims_px=(8, 120),
        homography_npz_path=str(npz_path),
        grasp_height=0.34,
        sweep_height=0.57,
        clearance_height=0.8,
        safety_margin=0.02,
    )
    perception = PerceptionConfig(provider="none", crop=crop, grid=grid)

    frames = CoordinateFrames.from_config(workspace, perception)
    x, y = frames.crop_px_to_robot(180, 180)
    assert x == pytest.approx(0.5, abs=1e-9)
    assert y == pytest.approx(0.5, abs=1e-9)


# --- CoordinateFrames: px <-> robot round-trip ----------------------------------------


def test_crop_px_to_robot_and_back_round_trip():
    crop, grid = _crop_grid()
    H = _synthetic_H()
    frames = CoordinateFrames(crop, grid, H_crop_to_robot=H)

    for u, v in [(0, 0), (360, 360), (180, 180), (90, 270)]:
        x, y = frames.crop_px_to_robot(u, v)
        u2, v2 = frames.robot_to_crop_px(x, y)
        # robot_to_crop_px truncates to int (ported from legacy PixelRobotTransform.robot_to_pix)
        assert u2 == pytest.approx(u, abs=1)
        assert v2 == pytest.approx(v, abs=1)


def test_robot_y_axis_convention_top_is_y_equals_1():
    # Documented convention: robot y=1 is the TOP of the image (row 0); y=0 is the BOTTOM.
    crop, grid = _crop_grid()
    H = _synthetic_H(crop_w=360, crop_h=360)
    frames = CoordinateFrames(crop, grid, H_crop_to_robot=H)

    x_top, y_top = frames.crop_px_to_robot(180, 0)  # row 0 = top
    x_bottom, y_bottom = frames.crop_px_to_robot(180, 360)  # row crop_h = bottom
    assert y_top == pytest.approx(1.0)
    assert y_bottom == pytest.approx(0.0)


# --- CoordinateFrames: crop / grid geometry -------------------------------------------


def test_crop_extracts_expected_region():
    crop, grid = _crop_grid(crop_size=(100, 60), center=(200, 150), grid_dim=(10, 10))
    H = _synthetic_H()
    frames = CoordinateFrames(crop, grid, H_crop_to_robot=H)

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # crop.size = (w=100, h=60), center=(x=200, y=150) -> x1=150, y1=120
    image[120:180, 150:250] = 255

    cropped = frames.crop(image)
    assert cropped.shape[:2] == (60, 100)  # (h, w)
    assert np.all(cropped == 255)


def test_crop_to_grid_nearest_neighbor_preserves_binary():
    crop, grid = _crop_grid(crop_size=(100, 100), grid_dim=(10, 10))
    H = _synthetic_H()
    frames = CoordinateFrames(crop, grid, H_crop_to_robot=H)

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:50, :] = 1  # top half occupied

    grid_mask = frames.crop_to_grid(mask)
    assert grid_mask.shape == (10, 10)
    assert set(np.unique(grid_mask)).issubset({0, 1})
    assert np.all(grid_mask[:5, :] == 1)
    assert np.all(grid_mask[5:, :] == 0)


def test_grid_crop_px_round_trip():
    crop, grid = _crop_grid(crop_size=(360, 360), grid_dim=(36, 36))
    H = _synthetic_H()
    frames = CoordinateFrames(crop, grid, H_crop_to_robot=H)

    for pt in [(0, 0), (18, 18), (35, 35)]:
        crop_pt = frames.grid_to_crop_px(pt)
        back = frames.crop_px_to_grid(crop_pt)
        assert back[0] == pytest.approx(pt[0])
        assert back[1] == pytest.approx(pt[1])


def test_full_crop_px_round_trip():
    crop, grid = _crop_grid(crop_size=(100, 100), center=(200, 150))
    H = _synthetic_H()
    frames = CoordinateFrames(crop, grid, H_crop_to_robot=H)

    full_pt = (210, 140)
    crop_pt = frames.full_to_crop_px(full_pt)
    back = frames.crop_to_full_px(crop_pt)
    assert back[0] == pytest.approx(full_pt[0])
    assert back[1] == pytest.approx(full_pt[1])
