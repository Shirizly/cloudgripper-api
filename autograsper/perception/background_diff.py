"""`BackgroundDiffProvider` — reference-image occupancy provider (Wave 3a).

Design reference: `autograsper/design/02_proposed_architecture.md` §3.3 (`perception ->
background_diff.py # reference-image provider (from granular_utils)`); this is the "cautious
pipeline" provider design 03 §1 contrasts with the segmenter-native one in `yolo_segmenter.py`.

**Validity note (design 03 §1, §3.2): this provider's masks are only meaningful when the robot arm
and tool are OUT of the bottom camera's field of view** -- it works by diffing against a reference
image of the empty plate, so any robot/tool pixels that aren't rejected by the color filters below
show up as spurious "occupied" region. The legacy caller (`custom_graspers/granular_pusher.py
::update_mask_and_process`) enforced this by physically driving the robot to a corner before every
mask capture. That move-aside choreography is a *planner/session* concern (design 03 §1's
`RefreshMask` primitive), not this provider's -- `BackgroundDiffProvider.compute()` will happily
compute a mask from any `Observation` it's given; whether that observation was taken with the robot
clear of the frame is the caller's responsibility to arrange and check.

Porting notes (from `object_tracker/granular_utils.py`, copied and adapted, not imported):
- `create_occupancy_mask` -> `_create_occupancy_mask` below: pixel-difference threshold (0.11 of
  255, i.e. ~28), primary-color robot-part rejection (red/green/blue/yellow ranges via
  `cv2.inRange`), and background gray/black rejection. Ported faithfully including the legacy
  color-range constants. One intentional simplification: legacy's own signature
  (`create_occupancy_mask(image, reference_empty, threshold: int = 30)`) accepted a `threshold`
  parameter that the function body never actually used (the 0.11 factor is hardcoded regardless of
  the argument) -- this is dead legacy code, so the parameter is dropped here rather than carried
  forward as a non-functional knob. Logged in `design/IMPLEMENTATION_LOG.md`.
- `clean_occupancy_mask` -> `perception.occupancy.clean_mask` (shared with the YOLO provider's
  option to reuse cleanup, though `yolo_segmenter.py` uses a lighter cleanup -- see that module).
- `process_image`'s crop step -> `perception.frames.CoordinateFrames.crop` (the one canonical crop,
  replacing `crop_center_region`; see `docs/perception.md` / `design/IMPLEMENTATION_LOG.md`'s
  "Wave 2: crop center axis order" entry).
- `process_image`'s unconditional `cv2.imwrite("occupancy_mask_for_debug.png", clean_mask)` ->
  `DebugSink.save(...)`, no-op unless a session directory has been wired in (CONVENTIONS.md: "No
  `cv2.imwrite`/`cv2.imshow` side effects in library code").

Threading: stateless after construction (the reference crop is computed once); `compute()` is a
pure function of its argument and safe to call from any thread (though in practice only the
`SegmentationWorker` thread calls it for a given provider instance).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from autograsper.observation.types import Observation
from autograsper.perception.occupancy import (
    ClumpStats,
    OccupancyResult,
    PerceptionDegraded,
    clean_mask,
    clump_stats_from_mask,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from autograsper.config_schema import BackgroundDiffConfig
    from autograsper.observation.debug import DebugSink
    from autograsper.perception.frames import CoordinateFrames

logger = logging.getLogger(__name__)

# Legacy `create_occupancy_mask`'s hardcoded diff threshold: 0.11 of the 255 pixel-value range.
_DIFF_THRESHOLD_FRACTION = 0.11

# Primary-color (robot part) rejection ranges, BGR, ported verbatim from
# `object_tracker/granular_utils.py::create_occupancy_mask`.
_LOWER_RED = np.array([0, 0, 100], dtype=np.uint8)
_UPPER_RED = np.array([80, 80, 255], dtype=np.uint8)
_LOWER_GREEN = np.array([0, 100, 0], dtype=np.uint8)
_UPPER_GREEN = np.array([80, 255, 80], dtype=np.uint8)
_LOWER_BLUE = np.array([100, 0, 0], dtype=np.uint8)
_UPPER_BLUE = np.array([255, 80, 80], dtype=np.uint8)
_LOWER_YELLOW = np.array([0, 150, 150], dtype=np.uint8)
_UPPER_YELLOW = np.array([120, 255, 255], dtype=np.uint8)

# Background (empty plate) gray/black rejection ranges, BGR, ported verbatim.
_LOWER_GRAY = np.array([170, 170, 170], dtype=np.uint8)
_UPPER_GRAY = np.array([240, 240, 240], dtype=np.uint8)
_LOWER_BLACK = np.array([0, 0, 0], dtype=np.uint8)
_UPPER_BLACK = np.array([40, 40, 40], dtype=np.uint8)


def _create_occupancy_mask(image: np.ndarray, reference_empty: np.ndarray) -> np.ndarray:
    """Binary occupancy mask: pixels that differ from `reference_empty` by more than 11% of the
    full pixel range, minus pixels in robot primary colors or plate background colors.

    Ported from `object_tracker/granular_utils.py::create_occupancy_mask`. `image` and
    `reference_empty` must be same-shape BGR crops.
    """
    color_diff = np.abs(image.astype(np.float32) - reference_empty.astype(np.float32))
    max_diff = 255 * _DIFF_THRESHOLD_FRACTION
    diff = np.max(color_diff, axis=2) if color_diff.ndim == 3 else color_diff
    mask = (diff > max_diff).astype(np.uint8) * 255

    red_mask = cv2.inRange(image, _LOWER_RED, _UPPER_RED)
    green_mask = cv2.inRange(image, _LOWER_GREEN, _UPPER_GREEN)
    blue_mask = cv2.inRange(image, _LOWER_BLUE, _UPPER_BLUE)
    yellow_mask = cv2.inRange(image, _LOWER_YELLOW, _UPPER_YELLOW)
    primary_color_mask = cv2.bitwise_or(
        cv2.bitwise_or(red_mask, green_mask), cv2.bitwise_or(blue_mask, yellow_mask)
    )
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(primary_color_mask))

    gray_mask = cv2.inRange(image, _LOWER_GRAY, _UPPER_GRAY)
    black_mask = cv2.inRange(image, _LOWER_BLACK, _UPPER_BLACK)
    gray_mask = cv2.bitwise_or(gray_mask, black_mask)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(gray_mask))

    return mask


class BackgroundDiffProvider:
    """Reference-image occupancy provider (design 02 §3.3, design 03 §1's "cautious pipeline").

    See module docstring: masks are only valid when the robot/tool are out of frame -- that policy
    is enforced by planner/session (Wave 4+), not here.
    """

    def __init__(
        self,
        frames: "CoordinateFrames",
        config: "BackgroundDiffConfig",
        debug: Optional["DebugSink"] = None,
    ):
        self._frames = frames
        self._min_size = config.min_granule_size
        self._debug = debug

        reference = cv2.imread(config.reference_image_path)
        if reference is None:
            raise FileNotFoundError(
                "BackgroundDiffProvider: could not read reference image at "
                f"{config.reference_image_path!r} "
                "(perception.background_diff.reference_image_path). Capture one with the robot "
                "and tool clear of the workspace before constructing this provider."
            )
        # Cropped once at construction: the reference image is assumed to already be a full_px-
        # frame capture (same pipeline stage as Observation.bottom_image -- i.e. saved from the
        # live undistorted+rectified feed, not a raw camera frame). Logged in
        # design/IMPLEMENTATION_LOG.md.
        self._reference_crop = frames.crop(reference)

    def compute(self, obs: Observation) -> OccupancyResult:
        crop_img = self._frames.crop(obs.bottom_image)
        if crop_img.shape != self._reference_crop.shape:
            raise PerceptionDegraded(
                "BackgroundDiffProvider: crop shape mismatch between observation "
                f"({crop_img.shape}) and reference image ({self._reference_crop.shape}) -- cannot "
                "compare pixel-for-pixel."
            )

        raw_mask = _create_occupancy_mask(crop_img, self._reference_crop)
        cleaned = clean_mask(raw_mask, min_size=self._min_size)
        stats: ClumpStats = clump_stats_from_mask(cleaned)
        grid_mask = self._frames.crop_to_grid(cleaned)

        if self._debug is not None and self._debug.enabled:
            self._debug.save("background_diff_raw_mask", raw_mask)
            self._debug.save("background_diff_clean_mask", cleaned)

        return OccupancyResult(
            source_seq=obs.seq,
            frame_index=obs.frame_index,
            timestamp=obs.timestamp,
            grid_mask=grid_mask,
            crop_mask=cleaned,
            instances=(),
            stats=stats,
        )
