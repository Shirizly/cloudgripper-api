"""Tests for `autograsper.perception.tool_grip.ToolGripChecker` (Wave 3a).

Synthetic top-camera images only (no network/GPU/robot), per CONVENTIONS.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from autograsper.config_schema import RoiConfig, ToolCheckConfig
from autograsper.observation.types import Observation, RobotState
from autograsper.perception.tool_grip import ToolGripChecker


def _make_config(
    x=(0.4, 0.6),
    y=(0.4, 0.6),
    threshold=0.5,
    lower=(100, 100, 100),
    upper=(200, 200, 200),
):
    return ToolCheckConfig(
        enabled=True,
        roi=RoiConfig(x=x, y=y),
        detection_threshold=threshold,
        color_lower_bgr=lower,
        color_upper_bgr=upper,
    )


def _make_observation(top_image: np.ndarray) -> Observation:
    bottom = np.zeros((10, 10, 3), dtype=np.uint8)
    state = RobotState(x=0.5, y=0.5, z=1.0, rotation=0.0, claw=1.0)
    return Observation(
        seq=1,
        frame_index=None,
        timestamp=0.0,
        top_image=top_image,
        bottom_image=bottom,
        robot_state=state,
    )


def test_high_quality_when_roi_filled_with_tool_color():
    img = np.zeros((100, 100, 3), dtype=np.uint8)  # black background, outside tool color range
    # ROI x=[0.4,0.6], y=[0.4,0.6] on a 100x100 image -> pixel bounds [40:60, 40:60].
    img[40:60, 40:60] = (150, 150, 150)  # inside (100,100,100)-(200,200,200)

    checker = ToolGripChecker(_make_config())
    result = checker.check(_make_observation(img))

    assert result.quality == pytest.approx(1.0)
    assert result.ok is True
    assert result.roi_used == (40, 60, 40, 60)


def test_low_quality_when_roi_has_no_tool_color():
    img = np.zeros((100, 100, 3), dtype=np.uint8)  # all black -- outside the configured range
    checker = ToolGripChecker(_make_config())
    result = checker.check(_make_observation(img))

    assert result.quality == pytest.approx(0.0)
    assert result.ok is False


def test_partial_grip_quality_between_zero_and_one():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Fill half of the 20x20 ROI (400px) with tool color -> quality ~0.5.
    img[40:60, 40:50] = (150, 150, 150)

    checker = ToolGripChecker(_make_config(threshold=0.9))
    result = checker.check(_make_observation(img))

    assert result.quality == pytest.approx(0.5, abs=0.01)
    assert result.ok is False  # below the (deliberately high) 0.9 threshold


def test_empty_roi_returns_zero_quality_not_a_crash():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Zero-width ROI (x collapses to a single fraction) -> empty slice.
    checker = ToolGripChecker(_make_config(x=(0.5, 0.5), y=(0.4, 0.6)))
    result = checker.check(_make_observation(img))

    assert result.quality == 0.0
    assert result.ok is False


def test_ok_flag_follows_detection_threshold():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60] = (150, 150, 150)  # full ROI -> quality 1.0

    strict = ToolGripChecker(_make_config(threshold=1.1))  # impossible to reach
    lenient = ToolGripChecker(_make_config(threshold=0.1))

    obs = _make_observation(img)
    assert strict.check(obs).ok is False
    assert lenient.check(obs).ok is True
