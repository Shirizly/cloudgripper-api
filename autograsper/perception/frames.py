"""`CameraPipeline` (fisheye undistort + homography rectify) and `CoordinateFrames` (the one
authority for `full_px` / `crop_px` / `grid` / `robot` conversions).

Design reference: `autograsper/design/02_proposed_architecture.md` §3.2 ("applies the camera
pipeline (undistort + homography rectify from `perception.frames`)") and
`autograsper/design/03_segmenter_native_design.md` §2.1, §7.2 ("Mask <-> homography consistency
... the design forces it through one `CoordinateFrames` object").

Porting notes (legacy sources copied/adapted, not imported):
- `autograsper/library/calibration.py::undistort` — fisheye branch (`cv2.fisheye
  .initUndistortRectifyMap` + `cv2.remap`), the post-undistort vertical flip
  (`cv2.flip(img, 0)`), and the 90-degree rotation. The non-fisheye branch
  (`cv2.getOptimalNewCameraMatrix` / `cv2.undistort` + ROI crop) is intentionally NOT ported: every
  checked-in distortion-coefficient array (`autograsper/config.yaml`, `granular-config.yaml`) has
  exactly 4 coefficients, i.e. always takes the fisheye branch; see
  `design/IMPLEMENTATION_LOG.md`.
- `autograsper/library/bottom_image_preprocessing.py::rotate` — the general rotate-with
  -bounding-box-expansion helper; ported verbatim and always called with `angle=90`, matching
  `calibration.py::undistort`'s `rotate(undistorted_img, 90)` call site.
- `autograsper/library/utils.py::get_undistorted_bottom_image` — the `cv2.warpPerspective(
  fisheye_undistorted, H, (w, h))` homography-rectification step applied after undistortion, same
  output size as the undistorted image.
- `object_tracker/granular_utils.py::crop_center_region` — the canonical crop. Legacy's own call
  signature is `crop_center_region(image, crop_size=(360, 360), crop_center=(275, 200))` and
  unpacks it as `crop_h, crop_w = crop_size` / `center_y, center_x = crop_center` — i.e. legacy's
  *tuple order* for `crop_center` is `(y, x)`. Wave 1's `config_schema.CropConfig` (see
  `docs/configuration.md`'s table: "`perception.crop.center_px` / `.size` | `[x,y]` / `[w,h]`
  ints") already commits to `(x, y)` / `(w, h)` order — the *opposite* axis order from legacy's
  argument names for the same numeric values (`[275, 200]` in `granular-config.yaml`, copied
  verbatim from legacy's default). This module follows `config_schema`'s documented `(x, y)`/
  `(w, h)` convention (`CoordinateFrames.crop`), which is an independent, logged decision — see
  `design/IMPLEMENTATION_LOG.md` ("2026-07-20 -- Wave 2: crop center axis order"). This is
  precisely the "current 360x360-at-(275,200) vs 360x360-at-center discrepancy" design 03 §2.1
  flags as needing exactly one resolved definition; this class is that definition.
- `object_tracker/granular_utils.py::downscale_mask` — `cv2.resize(..., interpolation=
  cv2.INTER_NEAREST)` for binary-mask-safe downscaling; ported into `crop_to_grid`.
- `custom_graspers/fence_utils.py::PixelRobotTransform` — `pix_to_robot`/`robot_to_pix`
  homogeneous-homography math; ported into `crop_px_to_robot`/`robot_to_crop_px`. VERIFIED (see
  class docstring below) that legacy applies this homography to CROP-frame pixels, not full-image
  pixels.

Threading: both classes are stateless after construction (pure functions of their fields); safe
to share and call from any thread.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a hard import-time dependency
    from autograsper.config_schema import CameraConfig, CropConfig, GridConfig, PerceptionConfig, WorkspaceConfig

logger = logging.getLogger(__name__)


class FramesError(Exception):
    """Raised when `CoordinateFrames` cannot be constructed (missing/malformed homography file,
    wrong matrix shape)."""


def _rotate90(image: np.ndarray) -> np.ndarray:
    """Rotate `image` 90 degrees, expanding the canvas to fit (no cropping of content).

    Ported verbatim from `autograsper/library/bottom_image_preprocessing.py::rotate`, specialized
    to the single angle (90) that `calibration.py::undistort` always calls it with.
    """
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)
    angle = 90
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    M[0, 2] += (nw / 2) - cx
    M[1, 2] += (nh / 2) - cy
    return cv2.warpAffine(image, M, (nw, nh))


class CameraPipeline:
    """Bottom-camera undistort + homography rectification. Top image is pass-through (design 02
    §3.2: "applies the camera pipeline ... to the raw bottom image" only).
    """

    def __init__(self, m: np.ndarray, d: np.ndarray, H: "np.ndarray | None"):
        self._m = np.asarray(m, dtype=np.float64)
        self._d = np.asarray(d, dtype=np.float64)
        self._H = np.asarray(H, dtype=np.float64) if H is not None else None

    @classmethod
    def from_config(cls, camera: "CameraConfig") -> "CameraPipeline":
        """Build from `config_schema.CameraConfig` (`camera.m`, `camera.d`, `camera.H`)."""
        return cls(m=np.array(camera.m), d=np.array(camera.d), H=np.array(camera.H))

    def process_bottom(self, raw_bottom: "np.ndarray | None") -> "np.ndarray | None":
        """Undistort + rectify a raw bottom-camera frame. Returns `None` unchanged (no image
        available this cycle) rather than raising."""
        if raw_bottom is None:
            return None
        undistorted = self._undistort(raw_bottom)
        if self._H is not None:
            h, w = undistorted.shape[:2]
            undistorted = cv2.warpPerspective(undistorted, self._H, (w, h))
        return undistorted

    def process_top(self, raw_top: "np.ndarray | None") -> "np.ndarray | None":
        """Pass-through (documented for symmetry with `process_bottom`; no processing applied)."""
        return raw_top

    def _undistort(self, img: np.ndarray) -> np.ndarray:
        K, D = self._m, self._d
        if D.shape[0] == 4:
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), K, img.shape[:2][::-1], cv2.CV_16SC2
            )
            out = cv2.remap(
                img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            )
        else:
            h, w = img.shape[:2]
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
            out = cv2.undistort(img, K, D, None, new_K)
            x, y, w2, h2 = roi
            out = out[y : y + h2, x : x + w2]
        out = cv2.flip(out, 0)
        out = _rotate90(out)
        return out


class CoordinateFrames:
    """The one authority for `full_px` <-> `crop_px` <-> `grid` <-> `robot` conversions.

    Frames (see also `docs/perception.md`):
    - `full_px`: pixel coordinates in the rectified bottom image (`CameraPipeline.process_bottom`
      output). `(x, y)` = `(column, row)`, origin top-left, x right, y down — OpenCV's native
      image-array indexing convention (row-major `image[y, x]`).
    - `crop_px`: pixel coordinates within the canonical crop (`perception.crop.center_px`,
      `perception.crop.size`), same axis directions as `full_px`, origin at the crop's top-left
      corner.
    - `grid`: the canonical dataset grid (`perception.grid.height`/`.width`), obtained by
      resizing `crop_px` — same axis directions, scaled by `grid_dim / crop_dim`.
    - `robot`: the robot's normalized workspace xy in `[0, 1]^2`, exactly the values sent to
      `RobotInterface.move_xy`.

    Y-axis convention (**verified**, not guessed, against
    `custom_graspers/fence_utils.py::check_wall_reset_needed`, which maps a robot-normalized band
    center into mask/crop pixel coordinates via
    `center_px = [center[0] * w, (1 - center[1]) * h]` — i.e. it explicitly flips y before scaling
    by the image height): **robot `y=1` is the TOP of the image (row 0); robot `y=0` is the
    BOTTOM (row `h-1`)**. The crop-frame row axis therefore runs opposite to robot y.
    `crop_px_to_robot`/`robot_to_crop_px` do not re-apply this flip manually — the homography
    matrix (`workspace.homography_npz_path`, key `arr_0`) is fit directly from calibration
    correspondences and already encodes it (along with any x-axis/rotation quirks of the specific
    camera mount), exactly like legacy `custom_graspers/fence_utils.py::PixelRobotTransform`.

    Homography frame (**verified** against `custom_graspers/granular_pusher.py::perform_task`):
    `self.latest_mask` there comes from `object_tracker/granular_utils.py::process_image`, which
    crops both the live image and the reference image to the canonical crop *before* computing
    the mask — so `self.latest_mask` is in crop-frame pixels. `tool_placement = find_tool_placements
    (self.latest_mask, ...)` operates on that crop-frame mask, and `pos_world =
    self.pix2robtrans.pix_to_robot(*tool_placement["pos_px"])` feeds the resulting crop-frame pixel
    straight into the homography with no full-image offset applied anywhere in between. So `H`
    maps **crop_px -> robot**, matching `crop_px_to_robot`/`robot_to_crop_px` (NOT
    `full_to_crop_px` composed with anything else — the homography was never calibrated against
    full-image pixels).
    """

    def __init__(self, crop: "CropConfig", grid: "GridConfig", H_crop_to_robot: np.ndarray):
        self._crop = crop
        self._grid = grid
        H = np.asarray(H_crop_to_robot, dtype=np.float64)
        if H.shape != (3, 3):
            raise FramesError(f"homography matrix must be 3x3, got shape {H.shape}")
        self._H = H
        try:
            self._H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError as exc:
            raise FramesError(f"homography matrix is singular, cannot invert: {H!r}") from exc

    @classmethod
    def from_config(
        cls, workspace: "WorkspaceConfig", perception: "PerceptionConfig"
    ) -> "CoordinateFrames":
        """Build from `config_schema.WorkspaceConfig` + `PerceptionConfig`.

        Loads `workspace.homography_npz_path` with `np.load(...)['arr_0']`, matching the format
        legacy `custom_graspers/granular_pusher.py` and the `image_collector/
        create_homography_calibration.py` / `calibrate_from_dataset.py` tooling already produce
        and consume. Raises `FramesError` with a clear, actionable message if the file is missing,
        unreadable, lacks the `arr_0` key, or isn't a 3x3 matrix.
        """
        path = workspace.homography_npz_path
        if not os.path.exists(path):
            raise FramesError(
                f"homography file not found: {path!r} (workspace.homography_npz_path). Generate "
                "it with image_collector/create_homography_calibration.py or "
                "image_collector/calibrate_from_dataset.py before constructing CoordinateFrames."
            )
        try:
            data = np.load(path)
            H = data["arr_0"]
        except Exception as exc:
            raise FramesError(
                f"failed to load homography from {path!r} (expected an .npz with key 'arr_0', "
                f"matching custom_graspers/granular_pusher.py's loader): {exc}"
            ) from exc
        return cls(crop=perception.crop, grid=perception.grid, H_crop_to_robot=H)

    # -- crop / grid geometry -------------------------------------------------

    def _crop_origin(self) -> Tuple[int, int]:
        cx, cy = self._crop.center_px
        crop_w, crop_h = self._crop.size
        return cx - crop_w // 2, cy - crop_h // 2

    def crop(self, image_full: np.ndarray) -> np.ndarray:
        """Extract the canonical crop from a full-resolution `full_px`-frame image.

        Ported from `object_tracker/granular_utils.py::crop_center_region`, using
        `config_schema.CropConfig`'s `(x, y)`/`(w, h)` axis order (see module docstring for the
        legacy axis-order discrepancy this resolves).
        """
        h, w = image_full.shape[:2]
        x1, y1 = self._crop_origin()
        crop_w, crop_h = self._crop.size
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x1 + crop_w), min(h, y1 + crop_h)
        return image_full[y1c:y2c, x1c:x2c]

    def crop_to_grid(self, mask_crop: np.ndarray) -> np.ndarray:
        """Resize a crop-frame mask to the canonical grid.

        Uses nearest-neighbor interpolation so binary masks stay binary (0/1 or 0/255), matching
        `object_tracker/granular_utils.py::downscale_mask`'s use of `cv2.INTER_NEAREST` for the
        same reason.
        """
        return cv2.resize(
            mask_crop, (self._grid.width, self._grid.height), interpolation=cv2.INTER_NEAREST
        )

    # -- point conversions -----------------------------------------------------

    def full_to_crop_px(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        """`full_px` -> `crop_px`. Assumes the point lies within the (uncropped-at-the-image
        -boundary) canonical crop; does not account for `crop()`'s edge-clipping."""
        x, y = pt
        x1, y1 = self._crop_origin()
        return (x - x1, y - y1)

    def crop_to_full_px(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        """`crop_px` -> `full_px`. Inverse of `full_to_crop_px`."""
        x, y = pt
        x1, y1 = self._crop_origin()
        return (x + x1, y + y1)

    def crop_px_to_robot(self, u: float, v: float) -> Tuple[float, float]:
        """`crop_px` -> `robot` xy via the homography. Ported from
        `custom_graspers/fence_utils.py::PixelRobotTransform.pix_to_robot`."""
        p = np.array([u, v, 1.0])
        x, y, w = self._H @ p
        return (x / w, y / w)

    def robot_to_crop_px(self, x: float, y: float) -> Tuple[int, int]:
        """`robot` xy -> `crop_px` via the inverse homography. Ported from
        `custom_graspers/fence_utils.py::PixelRobotTransform.robot_to_pix` (int-truncated, same as
        legacy)."""
        p = np.array([x, y, 1.0])
        u, v, w = self._H_inv @ p
        return (int(u / w), int(v / w))

    def grid_to_crop_px(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        """`grid` -> `crop_px` (inverse of the resize in `crop_to_grid`)."""
        gx, gy = pt
        scale_x = self._crop.size[0] / self._grid.width
        scale_y = self._crop.size[1] / self._grid.height
        return (gx * scale_x, gy * scale_y)

    def crop_px_to_grid(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        """`crop_px` -> `grid` (forward direction of the resize in `crop_to_grid`)."""
        x, y = pt
        scale_x = self._grid.width / self._crop.size[0]
        scale_y = self._grid.height / self._crop.size[1]
        return (x * scale_x, y * scale_y)
