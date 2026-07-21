"""Tests for `autograsper.perception.yolo_segmenter` (Wave 3a): the construction-is-lazy
guarantee, plus an optional real-model smoke test.

Per CONVENTIONS.md: "no GPU or network" by default; the real-model test only runs when
`RUN_MODEL_TESTS=1` is set AND the configured weights file exists.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest


def test_importing_module_does_not_import_ultralytics_or_segmenter():
    """Importing `autograsper.perception.yolo_segmenter` must not require `ultralytics`/`torch`
    to be installed -- only constructing a `YoloOccupancyProvider` does (the import is inside
    `YoloOccupancyProvider.__init__`, not at module scope)."""
    for name in list(sys.modules):
        if name == "ultralytics" or name.startswith("ultralytics.") or name == "image_collector.chickpea_segmenter":
            del sys.modules[name]
    sys.modules.pop("autograsper.perception.yolo_segmenter", None)

    import autograsper.perception.yolo_segmenter  # noqa: F401

    assert "ultralytics" not in sys.modules
    assert "image_collector.chickpea_segmenter" not in sys.modules
    # sanity: the classes this module is supposed to provide are actually there
    assert hasattr(autograsper.perception.yolo_segmenter, "YoloOccupancyProvider")
    assert hasattr(autograsper.perception.yolo_segmenter, "SegmentationWorker")


_WEIGHTS_PATH = "image_collector/chickpeas_segmentation_best.pt"


@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="set RUN_MODEL_TESTS=1 (and ensure weights exist) to run the real-model test",
)
def test_real_model_predicts_on_synthetic_crop():
    if not os.path.exists(_WEIGHTS_PATH):
        pytest.skip(f"weights not found at {_WEIGHTS_PATH}")

    from autograsper.config_schema import CropConfig, GridConfig, YoloConfig
    from autograsper.observation.types import Observation, RobotState
    from autograsper.perception.frames import CoordinateFrames
    from autograsper.perception.yolo_segmenter import YoloOccupancyProvider

    crop = CropConfig(center_px=(320, 240), size=(360, 360))
    grid = GridConfig(height=128, width=128)
    frames = CoordinateFrames(crop, grid, H_crop_to_robot=np.eye(3))

    yolo_config = YoloConfig(
        weights_path=_WEIGHTS_PATH, conf_threshold=0.25, iou_threshold=0.45, imgsz=640
    )
    provider = YoloOccupancyProvider(frames, yolo_config)

    bottom = np.zeros((480, 640, 3), dtype=np.uint8)
    top = np.zeros((10, 10, 3), dtype=np.uint8)
    state = RobotState(x=0.5, y=0.5, z=1.0, rotation=0.0, claw=1.0)
    obs = Observation(
        seq=1, frame_index=None, timestamp=0.0, top_image=top, bottom_image=bottom, robot_state=state
    )

    result = provider.compute(obs)
    assert result.grid_mask.shape == (128, 128)
