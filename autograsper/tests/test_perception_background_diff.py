"""Tests for `autograsper.perception.background_diff.BackgroundDiffProvider` (Wave 3a).

Synthetic reference + scene images only (no network/GPU/robot), per CONVENTIONS.md.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from autograsper.config_schema import BackgroundDiffConfig, CropConfig, GridConfig
from autograsper.observation.types import Observation, RobotState
from autograsper.perception.background_diff import BackgroundDiffProvider
from autograsper.perception.frames import CoordinateFrames


def _synthetic_H(crop_w=200, crop_h=200):
    """Affine-like homography, crop_px -> robot, same shape as test_frames.py's helper. Not
    exercised by these tests directly (background_diff never calls crop_px_to_robot) but required
    to construct a `CoordinateFrames`."""
    return np.array(
        [
            [1.0 / crop_w, 0.0, 0.0],
            [0.0, -1.0 / crop_h, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _make_frames(crop_size=(200, 200), grid_dim=(20, 20), center=(100, 100)):
    crop = CropConfig(center_px=center, size=crop_size)
    grid = GridConfig(height=grid_dim[0], width=grid_dim[1])
    H = _synthetic_H(crop_size[0], crop_size[1])
    return CoordinateFrames(crop, grid, H_crop_to_robot=H)


def _make_observation(bottom_image: np.ndarray, seq: int = 1) -> Observation:
    top = np.zeros((10, 10, 3), dtype=np.uint8)
    state = RobotState(x=0.5, y=0.5, z=1.0, rotation=0.0, claw=1.0)
    return Observation(
        seq=seq,
        frame_index=None,
        timestamp=0.0,
        top_image=top,
        bottom_image=bottom_image,
        robot_state=state,
    )


def _reference_image() -> np.ndarray:
    # Solid mid-gray "empty plate" -- also falls inside the background gray-rejection range
    # ([170,170,170]-[240,240,240]) is NOT required for the reference itself (only the diffed
    # scene needs rejection), but keeping it plausible.
    return np.full((200, 200, 3), 200, dtype=np.uint8)


def _scene_with_blobs() -> np.ndarray:
    img = _reference_image().copy()
    # Granule-like blob: distinct BGR color, chosen to avoid every color-rejection range in
    # create_occupancy_mask (not primary red/green/blue/yellow, not gray/black).
    img[20:60, 20:60] = (90, 110, 130)
    # Robot-part blob: pure "red" in BGR, matches create_occupancy_mask's red rejection range.
    img[100:140, 100:140] = (0, 0, 255)
    return img


@pytest.fixture
def reference_path(tmp_path):
    path = tmp_path / "reference.png"
    cv2.imwrite(str(path), _reference_image())
    return str(path)


def _make_provider(reference_path, min_granule_size=100, frames=None):
    frames = frames or _make_frames()
    config = BackgroundDiffConfig(reference_image_path=reference_path, min_granule_size=min_granule_size)
    return BackgroundDiffProvider(frames, config)


def test_finds_granule_blob_and_rejects_robot_color(reference_path):
    provider = _make_provider(reference_path)
    result = provider.compute(_make_observation(_scene_with_blobs()))

    assert result.source_seq == 1
    assert result.frame_index is None
    assert result.instances == ()
    assert result.stats.num_clumps == 1
    assert result.stats.total_area_px > 500  # blob survives cleanup, roughly its 1600px area

    # robot-colored region must not show up as occupied
    assert np.all(result.crop_mask[100:140, 100:140] == 0)
    # granule-colored region must show up as occupied
    assert np.any(result.crop_mask[20:60, 20:60] != 0)


def test_clump_stats_centroid_within_blob(reference_path):
    provider = _make_provider(reference_path)
    result = provider.compute(_make_observation(_scene_with_blobs()))

    assert len(result.stats.centroids) == 1
    cx, cy = result.stats.centroids[0]
    assert 20 <= cx <= 60
    assert 20 <= cy <= 60


def test_grid_mask_shape_and_binary(reference_path):
    provider = _make_provider(reference_path, frames=_make_frames(grid_dim=(20, 20)))
    result = provider.compute(_make_observation(_scene_with_blobs()))

    assert result.grid_mask.shape == (20, 20)
    assert set(np.unique(result.grid_mask)).issubset({0, 255})


def test_empty_scene_has_no_clumps(reference_path):
    provider = _make_provider(reference_path)
    result = provider.compute(_make_observation(_reference_image()))  # identical to reference

    assert result.stats.num_clumps == 0
    assert cv2.countNonZero(result.crop_mask) == 0


def test_missing_reference_image_raises_filenotfounderror(tmp_path):
    frames = _make_frames()
    config = BackgroundDiffConfig(
        reference_image_path=str(tmp_path / "missing.png"), min_granule_size=100
    )
    with pytest.raises(FileNotFoundError):
        BackgroundDiffProvider(frames, config)


def test_result_arrays_are_read_only(reference_path):
    provider = _make_provider(reference_path)
    result = provider.compute(_make_observation(_scene_with_blobs()))

    with pytest.raises(ValueError):
        result.crop_mask[0, 0] = 5
    with pytest.raises(ValueError):
        result.grid_mask[0, 0] = 5
