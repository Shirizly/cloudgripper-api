"""Occupancy contract shared by every perception provider (Wave 3a).

Design reference: `autograsper/design/03_segmenter_native_design.md` §2.1 (the `OccupancyResult`
shape, `latest()`/`await_result()` API mirrored by `perception.yolo_segmenter.SegmentationWorker`)
and §7.3/§7.4 (confidence policy, degraded-fallback risk). Also
`autograsper/design/02_proposed_architecture.md` §3.3 (`WorldState.occupancy: OccupancyResult |
None`).

This module defines only the *contract* — the frozen value types every provider produces and the
`OccupancyProvider` protocol they satisfy by duck typing (planning code, Wave 4+, is written
against this shape, not against any concrete provider class). Concrete providers live in
`background_diff.py` and `yolo_segmenter.py`; the tool-in-hand check is a separate, unrelated
contract in `tool_grip.py` (different camera, different question).

Porting notes:
- `clean_mask()` ports the connected-components area/elongation/sparsity filters from
  `object_tracker/granular_utils.py::clean_occupancy_mask` (morphological open/close is unchanged;
  the legacy function's own recomputed `num_labels/stats/centroids` return values after filtering
  are dropped here — callers that want stats call `clump_stats_from_mask` on the result, which is
  the one place connected-components analysis for *reporting* purposes lives, decoupling "clean the
  mask" from "describe the mask").
- `clump_stats_from_mask()` is a new helper (no direct legacy equivalent as a public function --
  `granular_utils.find_clumps`/`generate_clump_hierarchy` did a similar job with size-window
  filtering for a hierarchical scheme that Wave 3a does not port; connected-components analysis
  for `ClumpStats` is unfiltered by design, so consumers see the true clump count/areas of
  whatever mask they pass in).

Threading: all functions here are pure (no shared state); the dataclasses are immutable value
types. Array fields have `flags.writeable = False` set in `__post_init__` as a best-effort
read-only guard (see `observation/types.py`'s module docstring for why this can't be a true deep
freeze) -- copy (`arr.copy()`) before any in-place edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import cv2
import numpy as np

from autograsper.observation.types import Observation


class PerceptionDegraded(Exception):
    """Raised by an `OccupancyProvider.compute()` implementation (or observed by
    `SegmentationWorker`, which sets `.degraded` instead of raising -- see that module) when the
    provider's own output is not trustworthy enough to act on: design 03 §7.4 ("if the segmenter
    output degenerates (0 detections while the previous frames showed many, or CUDA failure) ...
    session layer falls back to the background-diff provider ... or pauses for intervention").

    Providers are free to raise this from `compute()` for a single-frame failure (e.g. a shape
    mismatch that makes the comparison meaningless); `SegmentationWorker` itself never raises it --
    it turns repeated failures/zero-detection streaks into a `.degraded` flag so the worker thread
    keeps running and the session layer (Wave 5+) decides what to do.
    """


@dataclass(frozen=True)
class Instance:
    """One per-detection result (YOLO instance segmentation). Empty for background-diff, which
    has no notion of individual instances -- see `OccupancyResult.instances`.

    - `mask`: binary uint8 {0, 255} in `crop_px` frame, or `None` if the provider does not produce
      per-instance masks.
    - `box`: `(x1, y1, x2, y2)` in `crop_px` frame.
    - `confidence`: the detector's confidence score, `[0, 1]`.
    """

    mask: Optional[np.ndarray]
    box: Tuple[float, float, float, float]
    confidence: float

    def __post_init__(self) -> None:
        if self.mask is not None:
            if not isinstance(self.mask, np.ndarray):
                raise TypeError(f"Instance.mask must be an ndarray or None, got {type(self.mask)!r}")
            self.mask.flags.writeable = False


@dataclass(frozen=True)
class ClumpStats:
    """Connected-components summary of a binary mask (crop_px frame).

    - `num_clumps`: number of connected components (background label excluded).
    - `total_area_px`: sum of `areas`.
    - `areas`: per-clump pixel area, `crop_px`.
    - `centroids`: per-clump `(x, y)` centroid, `crop_px` (OpenCV's native column,row order --
      matches `cv2.connectedComponentsWithStats`'s own centroid convention).

    Index `i` of `areas`/`centroids` refers to the same clump throughout (both built from the same
    `cv2.connectedComponentsWithStats` call, in label order); there is no guaranteed correspondence
    between clump `i` here and `instances[i]` on the same `OccupancyResult` -- the two are computed
    independently (`ClumpStats` from a mask, `instances` from the detector's own per-object output).
    """

    num_clumps: int
    total_area_px: int
    areas: Tuple[int, ...]
    centroids: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class OccupancyResult:
    """Immutable occupancy snapshot published by an `OccupancyProvider` / `SegmentationWorker`.

    - `source_seq`: the `Observation.seq` this result was computed from -- `staleness = obs.seq -
      source_seq` is how planning (Wave 4+) checks freshness (design 02 §3.3, design 03 §3.2).
    - `frame_index`: passed through from the source `Observation` (recorder frame number, or
      `None`).
    - `timestamp`: passed through from the source `Observation`.
    - `grid_mask`: binary uint8 {0, 255}, canonical grid frame (`perception.frames.CoordinateFrames
      .crop_to_grid`'s output shape, e.g. 128x128 -- the exact size is a config choice, see
      `docs/perception.md`).
    - `crop_mask`: binary uint8 {0, 255}, full-resolution crop frame (`CoordinateFrames.crop`'s
      output shape) -- used for pixel-precise placement search.
    - `instances`: per-detection results; always `()` for background-diff (see `Instance`'s
      docstring).
    - `stats`: connected-components summary of `crop_mask` (NOT `grid_mask` -- crop-frame stats are
      the ones placement search / wall checks need pixel precision for).
    """

    source_seq: int
    frame_index: Optional[int]
    timestamp: float
    grid_mask: np.ndarray
    crop_mask: np.ndarray
    instances: Tuple[Instance, ...]
    stats: ClumpStats

    def __post_init__(self) -> None:
        if not isinstance(self.grid_mask, np.ndarray):
            raise TypeError(f"OccupancyResult.grid_mask must be an ndarray, got {type(self.grid_mask)!r}")
        if not isinstance(self.crop_mask, np.ndarray):
            raise TypeError(f"OccupancyResult.crop_mask must be an ndarray, got {type(self.crop_mask)!r}")
        self.grid_mask.flags.writeable = False
        self.crop_mask.flags.writeable = False


class OccupancyProvider(Protocol):
    """Duck-typed contract every occupancy provider satisfies (planning code, Wave 4+, is written
    against this shape -- see design 03 §7 preamble)."""

    def compute(self, obs: Observation) -> OccupancyResult:
        """Synchronous, pure: no I/O beyond what was set up at construction (e.g. a reference image
        loaded once), no robot calls, no mutation of `obs`. May raise `PerceptionDegraded` (or let
        an unexpected exception propagate) if the result would not be trustworthy; callers running
        this inside `SegmentationWorker` never see the worker thread die because of it."""
        ...


# ---------------------------------------------------------------------------------------------
# Mask cleanup / stats helpers
# ---------------------------------------------------------------------------------------------


def clean_mask(mask: np.ndarray, kernel_size: int = 3, min_size: int = 300) -> np.ndarray:
    """Morphological open/close + area/elongation/sparsity filtering of a raw binary mask.

    Ported from `object_tracker/granular_utils.py::clean_occupancy_mask` (legacy default
    `min_size=300`, kept here). Legacy's connected-components filtering loop applies four
    independent conditions per component without short-circuiting (a component zeroed by the area
    check is checked again by the elongation/sparsity checks against its *original* stats -- since
    the pixels are already zero this is a no-op, not a bug, and is preserved verbatim rather than
    "simplified" with an early `continue`, in case a future min_size tuning ever makes the order
    observable). Returns only the cleaned mask; connected-components *statistics* for reporting are
    a separate concern -- call `clump_stats_from_mask` on the result if needed.

    Args:
        mask: input binary mask (any non-zero treated as foreground).
        kernel_size: morphological structuring element size (legacy default 3).
        min_size: minimum component area to survive filtering (legacy default 300).

    Returns:
        Cleaned binary mask, same dtype/shape as `mask`.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    for i in range(1, num_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        if area < min_size:
            closed[labels == i] = 0
        # very elongated small regions are probably wall/robot edges, not granule clumps
        if (width / height > 3 or height / width > 3) and area < min_size * 2:
            closed[labels == i] = 0
        # too thin in either dimension to plausibly be a granule clump
        if width < np.sqrt(min_size) or height < np.sqrt(min_size):
            closed[labels == i] = 0
        # sparse relative to bounding box (area much smaller than width*height)
        if area < (width * height) * 0.25 and area < min_size * 2:
            closed[labels == i] = 0

    return closed


def clump_stats_from_mask(crop_mask: np.ndarray) -> ClumpStats:
    """Connected-components summary of `crop_mask` (unfiltered -- pass an already-`clean_mask`ed
    mask in if filtering is wanted; this function reports whatever is in the mask it's given).
    """
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(crop_mask, connectivity=8)
    areas = tuple(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels))
    cents = tuple((float(centroids[i][0]), float(centroids[i][1])) for i in range(1, num_labels))
    return ClumpStats(
        num_clumps=num_labels - 1,
        total_area_px=int(sum(areas)),
        areas=areas,
        centroids=cents,
    )
