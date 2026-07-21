"""`ToolGripChecker` — top-camera tool-in-hand color check (Wave 3a).

Design reference: `autograsper/design/03_segmenter_native_design.md` §3.4 ("The top-camera
color-ROI grip check ... ports as-is (`perception/tool_grip.py`, ...)"); `02_proposed_architecture
.md` §3.3 (`ToolStatus` in `WorldState`) and §2 layout (`perception/tool_grip.py # tool-in-hand
check (from tool_user_utils)`).

Orthogonal to the bottom-camera occupancy providers in this package: this reads the *top* image
(`Observation.top_image`) and answers a completely different question ("is the tool correctly
gripped, based on a fixed ROI's color composition"), unrelated to granule occupancy.

Porting notes (copied and adapted, not imported -- `object_tracker/` is legacy and will be
retired):
- `custom_graspers/granular_pusher.py::RandomPushGrasper.check_tool_grip` -- the fractional-ROI
  extraction from the top image (`roi.x`/`roi.y` as fractions of image width/height, legacy
  defaults `x=[0.4, 0.7]`, `y=[0.1, 0.8]`) and the legacy hardcoded tool color range
  (`lower_bgr=(160, 160, 120)`, `upper_bgr=(190, 210, 190)` -- now sourced from
  `config_schema.ToolCheckConfig.color_lower_bgr`/`.color_upper_bgr`, never hardcoded here).
- `object_tracker/tool_user_utils.py::analyze_tool_grip` -- the color-mask-fraction computation
  itself, copied into `_analyze_tool_grip` below verbatim except: (1) the two unconditional
  `cv2.imwrite("tool_grip_analysis.png", ...)` / `cv2.imwrite("tool_grip_original.png", ...)` calls
  to the current working directory are removed (CONVENTIONS.md: no `cv2.imwrite` in library code) --
  `ToolGripChecker` accepts an optional `DebugSink` and writes the same two artifacts through it,
  no-op unless a session directory is wired in; (2) a zero-area ROI (possible if `roi.x`/`roi.y`
  collapse to an empty range, or the ROI falls entirely outside the image) now returns `0.0`
  instead of raising `ZeroDivisionError`.

Threading: stateless after construction; `check()` is a pure function of its argument (plus the
optional `DebugSink` write, which is itself thread-safe) and safe to call from any thread.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import cv2
import numpy as np

from autograsper.observation.types import Observation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from autograsper.config_schema import ToolCheckConfig
    from autograsper.observation.debug import DebugSink

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GripCheckResult:
    """Result of one `ToolGripChecker.check()` call.

    - `quality`: fraction (`[0, 1]`) of the ROI's pixels within the configured tool color range --
      ported verbatim from `object_tracker/tool_user_utils.py::analyze_tool_grip`'s return value.
    - `ok`: `quality >= config.detection_threshold`.
    - `roi_used`: the actual pixel bounds sliced out of `Observation.top_image`, `(x_min, x_max,
      y_min, y_max)`, for debugging/visualization.
    """

    quality: float
    ok: bool
    roi_used: Tuple[int, int, int, int]


def _analyze_tool_grip(
    image: np.ndarray,
    lower_bgr: Tuple[int, int, int],
    upper_bgr: Tuple[int, int, int],
) -> float:
    """Ported from `object_tracker/tool_user_utils.py::analyze_tool_grip` (legacy `object_tracker`
    package, to be retired). Debug `cv2.imwrite` calls removed -- see module docstring; pass a
    `DebugSink` to `ToolGripChecker` for equivalent artifacts. Returns `0.0` for an empty ROI
    instead of raising (legacy would `ZeroDivisionError`)."""
    if image.size == 0:
        return 0.0
    lower = np.array(lower_bgr, dtype=np.uint8)
    upper = np.array(upper_bgr, dtype=np.uint8)
    mask = cv2.inRange(image, lower, upper)
    tool_region_pixels = cv2.countNonZero(mask)
    total_pixels = image.shape[0] * image.shape[1]
    return tool_region_pixels / total_pixels


class ToolGripChecker:
    """Top-camera, fixed-ROI, color-threshold tool-grip check (design 03 §3.4)."""

    def __init__(self, config: "ToolCheckConfig", debug: Optional["DebugSink"] = None):
        self._config = config
        self._debug = debug

    def check(self, obs: Observation) -> GripCheckResult:
        top_img = obs.top_image
        h, w = top_img.shape[:2]
        x_min = int(self._config.roi.x[0] * w)
        x_max = int(self._config.roi.x[1] * w)
        y_min = int(self._config.roi.y[0] * h)
        y_max = int(self._config.roi.y[1] * h)

        roi_img = top_img[y_min:y_max, x_min:x_max]
        quality = _analyze_tool_grip(
            roi_img, self._config.color_lower_bgr, self._config.color_upper_bgr
        )
        ok = quality >= self._config.detection_threshold

        if self._debug is not None and self._debug.enabled:
            lower = np.array(self._config.color_lower_bgr, dtype=np.uint8)
            upper = np.array(self._config.color_upper_bgr, dtype=np.uint8)
            mask = cv2.inRange(roi_img, lower, upper) if roi_img.size else np.zeros((0, 0), dtype=np.uint8)
            self._debug.save("tool_grip_analysis", mask)
            self._debug.save("tool_grip_original", roi_img)

        return GripCheckResult(quality=quality, ok=ok, roi_used=(x_min, x_max, y_min, y_max))
