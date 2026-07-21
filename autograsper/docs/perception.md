# `autograsper/perception/` — CoordinateFrames, CameraPipeline, occupancy providers, tool grip

Status: **implemented** (Wave 2: `frames.py`; Wave 3a: `occupancy.py`, `background_diff.py`,
`yolo_segmenter.py`, `tool_grip.py`). Design reference:
[`../design/02_proposed_architecture.md`](../design/02_proposed_architecture.md) §3.2, §3.3
(layout), [`../design/03_segmenter_native_design.md`](../design/03_segmenter_native_design.md) §2
(occupancy contract, `SegmentationWorker`), §3.4 (tool grip), §7 (risk register: confidence policy,
degraded fallback).

Not yet implemented: the `LowerTool` real-time safety guard, freshness-policy enforcement
(`staleness == 0` checks), and the mask-aware executor hooks that *consume* `OccupancyResult` —
those are planning/execution concerns (Wave 4+), out of scope here. This module only produces
`OccupancyResult`/`GripCheckResult`; it does not decide what to do with them.

## Coordinate frames & camera pipeline

### Purpose

One authority for every full-image-pixel / crop-pixel / dataset-grid / robot-xy conversion in the
system, replacing the "two competing crops with different coordinate frames" problem design 01
§5 (defect #10) documents: the legacy grasper cropped around `(275, 200)` while
`SegmentationThread` used the true image center, and both fed a homography that assumed one fixed
frame. `CoordinateFrames` is now the one place that definition lives; every consumer (occupancy
providers, planner geometry, transition-dataset pixel math — Waves 3-5) is expected to go through
it rather than recomputing crop offsets or homography math locally.

`CameraPipeline` is the bottom-camera undistort + homography-rectify step
`observation.source.ObservationSource` applies to every raw frame before publishing an
`Observation` (see [observation.md](observation.md)) — it is a dependency of the observation
layer, not of the occupancy providers that will be added here in Wave 3.

### Public API

`autograsper/perception/frames.py`:

- `CameraPipeline(m, d, H)` — fisheye undistort (`cv2.fisheye.initUndistortRectifyMap` +
  `cv2.remap`) + the legacy vertical-flip-then-90-degree-rotate step + optional homography
  rectification (`cv2.warpPerspective`).
  - `CameraPipeline.from_config(camera: config_schema.CameraConfig)` — builds from `camera.m`,
    `camera.d`, `camera.H`.
  - `process_bottom(raw_bottom: Optional[np.ndarray]) -> Optional[np.ndarray]` — the full
    pipeline; `None` in, `None` out (no exception) when no image is available this cycle.
  - `process_top(raw_top) -> raw_top` — documented pass-through (top image is never processed).
- `FramesError(Exception)` — raised by `CoordinateFrames` construction problems (missing/
  unreadable homography file, wrong `arr_0` shape).
- `CoordinateFrames(crop, grid, H_crop_to_robot)` — the frame authority.
  - `CoordinateFrames.from_config(workspace: config_schema.WorkspaceConfig, perception:
    config_schema.PerceptionConfig)` — loads `workspace.homography_npz_path` via
    `np.load(path)['arr_0']` (same format legacy `custom_graspers/granular_pusher.py` and the
    `image_collector/create_homography_calibration.py` / `calibrate_from_dataset.py` tools
    produce/consume); raises `FramesError` with an actionable message if the file is missing,
    unreadable, or not a 3x3 matrix.
  - `crop(image_full) -> image_crop` — extract the canonical crop from a `full_px`-frame image.
  - `crop_to_grid(mask_crop) -> mask_grid` — nearest-neighbor resize (binary-mask-safe) to the
    canonical dataset grid.
  - `full_to_crop_px(pt) -> pt` / `crop_to_full_px(pt) -> pt` — pure pixel-offset conversions.
  - `crop_px_to_robot(u, v) -> (x, y)` / `robot_to_crop_px(x, y) -> (int u, int v)` — homography
    conversions (the latter int-truncates, matching legacy).
  - `grid_to_crop_px(pt) -> pt` / `crop_px_to_grid(pt) -> pt` — scale conversions between the
    dataset grid and the crop.

### Frames (definitions)

| Frame | Meaning | Axis convention |
|---|---|---|
| `full_px` | Pixels in the rectified bottom image (`CameraPipeline.process_bottom` output) | `(x, y)` = `(column, row)`, origin top-left, x right, y down |
| `crop_px` | Pixels within the canonical crop (`perception.crop.center_px`/`.size`) | same axis directions as `full_px`, origin at the crop's top-left corner |
| `grid` | The canonical dataset grid (`perception.grid.height`/`.width`) | same axis directions as `crop_px`, scaled by `grid_dim / crop_dim` |
| `robot` | Robot-normalized workspace xy, exactly what's sent to `move_xy` | `[0, 1]^2`; **see y-axis note below** |

**Y-axis convention** (verified, not guessed — see `custom_graspers/fence_utils.py
::check_wall_reset_needed`, which computes `center_px = [center[0] * w, (1 - center[1]) * h]`
when mapping a robot-normalized point into mask/crop pixels, i.e. it explicitly flips y before
scaling by image height): **robot `y=1` is the TOP of the image (row 0); robot `y=0` is the
BOTTOM (row `h-1`)**. `crop_px_to_robot`/`robot_to_crop_px` do not re-apply this flip manually —
the homography matrix (`workspace.homography_npz_path`, key `arr_0`) is fit directly from
calibration correspondences and already encodes it, exactly like legacy
`fence_utils.py::PixelRobotTransform`.

**Homography frame** (verified against `custom_graspers/granular_pusher.py::perform_task`): the
mask fed to `find_tool_placements` there (`self.latest_mask`, from
`object_tracker/granular_utils.py::process_image`) is already crop-cropped before the mask is
computed, and the resulting `pos_px` is passed straight into `pix2robtrans.pix_to_robot(*pos_px)`
with no full-image offset applied anywhere in between. So the homography maps **`crop_px` ->
`robot`**, matching `CoordinateFrames.crop_px_to_robot`/`.robot_to_crop_px` — never `full_px`
directly.

### Data flow

```
config_schema.CameraConfig(m, d, H)
    │
    ▼
CameraPipeline.from_config(...)
    │  .process_bottom(raw_bottom)   -- used by observation.source.ObservationSource
    ▼
Observation.bottom_image  (full_px frame, rectified)

config_schema.WorkspaceConfig.homography_npz_path + PerceptionConfig.crop/.grid
    │
    ▼
CoordinateFrames.from_config(...)
    │  .crop(image_full) -> crop_px image        (Wave 3 occupancy providers)
    │  .crop_to_grid(mask_crop) -> grid mask      (Wave 3 occupancy providers, dataset writers)
    │  .crop_px_to_robot / .robot_to_crop_px      (Wave 3-4 planner geometry: placement search,
    │                                               wall checks — ports fence_utils.py math)
```

### Threading

Both `CameraPipeline` and `CoordinateFrames` are stateless after construction (pure functions of
their constructor arguments); safe to call from any thread, including concurrently.

### Config keys consumed

- `camera.m`, `camera.d`, `camera.H` (`CameraPipeline.from_config`).
- `workspace.homography_npz_path` (`CoordinateFrames.from_config`, loads `arr_0`).
- `perception.crop.center_px`/`.size`, `perception.grid.height`/`.width`
  (`CoordinateFrames.from_config`).

See [configuration.md](configuration.md) for the full schema.

### Porting notes

- `autograsper/legacy/library/calibration.py::undistort` — fisheye branch only
  (`cv2.fisheye.initUndistortRectifyMap` + `cv2.remap`), plus the post-undistort
  `cv2.flip(img, 0)` and 90-degree rotation. The non-fisheye branch
  (`cv2.getOptimalNewCameraMatrix`/`cv2.undistort` + ROI crop) is intentionally **not** ported:
  every checked-in distortion-coefficient array has exactly 4 entries (always the fisheye case) —
  logged in `design/IMPLEMENTATION_LOG.md`.
- `autograsper/legacy/library/bottom_image_preprocessing.py::rotate` — ported verbatim as a private
  helper, always called with `angle=90` (the only angle `calibration.py::undistort` ever used it
  with).
- `autograsper/legacy/library/utils.py::get_undistorted_bottom_image` — the
  `cv2.warpPerspective(fisheye_undistorted, H, same_size)` rectification step.
- `object_tracker/granular_utils.py::crop_center_region` — the canonical crop, with an
  **intentional axis-order change**: legacy's own call signature unpacks `crop_center` as
  `center_y, center_x` (i.e. `(y, x)` order) while `config_schema.CropConfig` (Wave 1,
  `docs/configuration.md`'s table) already documents `center_px`/`size` as `(x, y)`/`(w, h)`.
  `CoordinateFrames.crop` follows `config_schema`'s documented order — this is precisely the "one
  definition, calibrated once" design 03 §2.1 calls for to resolve the legacy
  360x360-at-(275,200)-vs-center discrepancy. Logged in `design/IMPLEMENTATION_LOG.md`.
- `object_tracker/granular_utils.py::downscale_mask` — `cv2.INTER_NEAREST` resize, ported into
  `crop_to_grid` for the same binary-mask-safety reason.
- `custom_graspers/fence_utils.py::PixelRobotTransform` — `pix_to_robot`/`robot_to_pix`
  homogeneous-homography math, ported into `crop_px_to_robot`/`robot_to_crop_px`.

### Testing

`autograsper/tests/test_frames.py` — `CameraPipeline` on a synthetic image with the template's
real `m`/`d` (shape preserved modulo the documented 90-degree swap, no exception, `None`
pass-through); `CoordinateFrames` construction errors (`FramesError` on a non-3x3 matrix, on a
missing homography file via `from_config`); a synthetic affine-like homography (crop 360x360 ->
unit square) round-tripping `crop_px -> robot -> crop_px`; the documented y-axis convention;
`crop`/`crop_to_grid` geometry; `grid_to_crop_px`/`crop_px_to_grid` and
`full_to_crop_px`/`crop_to_full_px` round trips. Run:

```
cd /home/alon/Code/cloudgripper-api
/home/alon/anaconda3/envs/cge/bin/python -m pytest autograsper/tests/test_frames.py -q
```

## Occupancy contract (`occupancy.py`)

### Purpose

The shape every occupancy provider (`background_diff.py`, `yolo_segmenter.py`) produces, and the
`OccupancyProvider` protocol planning code (Wave 4+) is written against by duck typing — no
concrete provider import required at that layer (design 02 §3.3: `WorldState.occupancy:
OccupancyResult | None`).

### Public API

`autograsper/perception/occupancy.py`:

- `Instance(mask, box, confidence)` — one per-detection result. `mask: Optional[np.ndarray]`
  (binary uint8 `{0, 255}`, `crop_px`, read-only), `box: (x1, y1, x2, y2)` in `crop_px`,
  `confidence: float`. Always `()` on `OccupancyResult.instances` for background-diff (no
  per-instance notion there).
- `ClumpStats(num_clumps, total_area_px, areas, centroids)` — connected-components summary of a
  `crop_px`-frame binary mask. `areas`/`centroids` are per-clump, same label order, `centroids` in
  `(x, y)` `crop_px` (OpenCV's own centroid convention). Unfiltered: reports whatever is in the mask
  it was built from — call it on an already-`clean_mask`ed mask for a filtered count.
- `OccupancyResult(source_seq, frame_index, timestamp, grid_mask, crop_mask, instances, stats)` —
  frozen; `grid_mask`/`crop_mask` are binary uint8 `{0, 255}`, read-only (`flags.writeable = False`
  set in `__post_init__`, same best-effort guard as `Observation`'s image fields —
  `docs/observation.md`). `stats` is always computed from `crop_mask` (pixel precision), never
  `grid_mask`. `source_seq` is the `Observation.seq` the result was computed from —
  `staleness = obs.seq - source_seq` is how a consumer checks freshness (design 02 §3.3, design 03
  §3.2's freshness rule; enforcing a required staleness is a planning/execution concern, not
  perception's).
- `OccupancyProvider` (Protocol) — `compute(obs: Observation) -> OccupancyResult`. Synchronous,
  pure (no robot calls, no I/O beyond what was set up at construction, e.g. a reference image
  loaded once). May raise `PerceptionDegraded` (or let an unexpected exception propagate) for a
  single-frame failure.
- `PerceptionDegraded(Exception)` — design 03 §7.4. `SegmentationWorker` never raises this itself —
  see its "Degraded detection" section below — but a provider's own `compute()` may raise it (or
  anything else); the worker catches all exceptions equally.
- `clean_mask(mask, kernel_size=3, min_size=300) -> np.ndarray` — morphological open/close +
  area/elongation/sparsity connected-components filtering. Used by `background_diff.py`; NOT used
  by `yolo_segmenter.py` (which applies a lighter, model-appropriate cleanup — see below).
- `clump_stats_from_mask(crop_mask) -> ClumpStats` — connected components on whatever mask it's
  given, unfiltered.

### Threading

Pure functions / immutable value types; safe to call/share from any thread.

### Porting notes

- `clean_mask` ports `object_tracker/granular_utils.py::clean_occupancy_mask`'s morphological
  open/close and its four per-component filters (area < `min_size`; elongation ratio > 3 with area
  < `2*min_size`; either dimension < `sqrt(min_size)`; sparse relative to bounding box with area <
  `2*min_size`), legacy default `min_size=300` kept. Dropped: legacy's own recomputed
  `num_labels/stats/centroids` return values after filtering (this function returns only the
  cleaned mask) — that's now `clump_stats_from_mask`'s job, called separately by whichever provider
  wants stats, decoupling "clean" from "describe".

## `BackgroundDiffProvider` (`background_diff.py`)

### Purpose

The reference-image occupancy provider — design 03 §1's "cautious pipeline" provider, ported from
`object_tracker/granular_utils.py`. **Its masks are only valid when the robot arm and tool are out
of the bottom camera's field of view** (it diffs against an empty-plate reference image); enforcing
that — e.g. via the `RefreshMask`/move-aside primitive design 03 §1 keeps around for exactly this
path — is a planner/session concern (Wave 4+), not this provider's. `compute()` will compute
*a* mask from any `Observation` it's given regardless of what's in frame.

### Public API

`autograsper/perception/background_diff.py`:

- `BackgroundDiffProvider(frames: CoordinateFrames, config: config_schema.BackgroundDiffConfig,
  debug: Optional[DebugSink] = None)` — loads `config.reference_image_path` once via `cv2.imread`
  and crops it once via `frames.crop`; raises `FileNotFoundError` immediately (at construction, not
  first `compute()`) with an actionable message if the file is missing/unreadable.
  - `compute(obs: Observation) -> OccupancyResult` — crops `obs.bottom_image`, diffs against the
    cached reference crop, cleans, and returns a result with `instances=()`. Raises
    `PerceptionDegraded` if the observation's crop shape doesn't match the reference crop's shape
    (can't diff pixel-for-pixel).

### Data flow

```
config_schema.BackgroundDiffConfig.reference_image_path
    │  cv2.imread (once, at construction) -> frames.crop (once, cached)
    ▼
Observation.bottom_image
    │  frames.crop
    ▼
_create_occupancy_mask(crop, reference_crop)   -- pixel-diff > 11% of 255, minus color rejections
    │
    ▼
clean_mask(raw_mask, min_size=config.min_granule_size)
    │
    ├─► clump_stats_from_mask(cleaned) -> ClumpStats
    ├─► frames.crop_to_grid(cleaned)   -> grid_mask
    └─► OccupancyResult(instances=(), ...)
```

### Config keys consumed

`perception.background_diff.reference_image_path`, `perception.background_diff.min_granule_size`
(`config_schema.BackgroundDiffConfig`); `perception.crop`/`perception.grid` indirectly, via the
`CoordinateFrames` passed in.

### Porting notes

- `object_tracker/granular_utils.py::create_occupancy_mask` — pixel-difference threshold (11% of
  255, i.e. `diff > 28.05`), primary-color (robot part) rejection via `cv2.inRange` for
  red/green/blue/yellow BGR ranges, and background gray/black rejection — all four color ranges
  ported verbatim as module constants. Legacy's own signature accepted a `threshold: int = 30`
  parameter that the function body never actually used (the `0.11` factor is hardcoded regardless)
  — dead legacy code, so the parameter is **not** carried into `_create_occupancy_mask`. Logged in
  `design/IMPLEMENTATION_LOG.md`.
- `object_tracker/granular_utils.py::clean_occupancy_mask` — via `perception.occupancy.clean_mask`
  (shared helper; see above).
- `object_tracker/granular_utils.py::process_image`'s crop step — via `CoordinateFrames.crop` (the
  one canonical crop; see `frames.py`'s porting notes on the axis-order resolution).
- `process_image`'s unconditional `cv2.imwrite("occupancy_mask_for_debug.png", clean_mask)` — via
  an optional `DebugSink.save(...)`, no-op unless a session directory is wired in.
- **Assumption logged, not verified against a real capture**: the reference image is treated as
  already being in the `full_px` frame (i.e. captured from the same undistort+homography-rectified
  pipeline stage as `Observation.bottom_image`, not a raw uncorrected camera frame) — this matches
  how `object_tracker/granular_utils.py::process_image` was actually called in
  `custom_graspers/granular_pusher.py` (against `shared_state.latest_bottom_image`, which
  `recording_seg.py::Recorder._update` already ran through undistort+homography before storing).
  Logged in `design/IMPLEMENTATION_LOG.md`.

## `YoloOccupancyProvider` + `SegmentationWorker` (`yolo_segmenter.py`)

### Purpose

The segmenter-native occupancy provider (design 03 §2.1) — successor of
`recording_seg.py::SegmentationThread`, promoted from a recorder detail to a standalone perception
worker. No move-aside choreography needed: the model is assumed accurate with the robot/tool in
frame (design 03 §1's core premise).

### Public API

`autograsper/perception/yolo_segmenter.py`:

- `YoloOccupancyProvider(frames: CoordinateFrames, yolo_config: config_schema.YoloConfig, debug:
  Optional[DebugSink] = None)` — **lazily** imports `image_collector.chickpea_segmenter
  .ChickpeaSegmenter` inside `__init__`; importing this module never requires `ultralytics`/`torch`
  to be installed, only constructing this class does.
  - `compute(obs: Observation) -> OccupancyResult` — crops, calls
    `ChickpeaSegmenter.predict(crop, return_format='dict')`, and builds the result per the
    confidence policy below.
- `occupied_mask_at_conf(result: OccupancyResult, min_conf: float) -> np.ndarray` — module-level
  helper: recomputes a crop-frame mask from `result.instances` at an arbitrary confidence cutoff,
  for consumers that want something other than the two thresholds baked into the result (all
  instances / `stats_conf_threshold`).
- `SegmentationWorker(provider: OccupancyProvider, source: ObservationSource, shutdown_event:
  threading.Event, *, max_consecutive_failures: int = 3, zero_detection_streak_for_degraded:
  Optional[int] = 5)` — background thread, name `"SegmentationWorker"`.
  - `start()` — subscribes to `source` (maxsize-1 `LatestWinsQueue`) and spawns the thread. Raises
    `RuntimeError` if already running.
  - `stop(timeout=None)` — signals the thread to stop, joins it, unsubscribes from `source`. Never
    touches `shutdown_event`.
  - `is_running() -> bool`.
  - `latest() -> Optional[OccupancyResult]` — non-blocking; `None` before the first successful
    `compute()`.
  - `await_result(min_source_seq: int, timeout: Optional[float] = None) -> Optional[OccupancyResult]`
    — condition-variable wait for the first result whose `source_seq >= min_source_seq` (`>=`, not
    `>` — see docstring: callers pass "the seq right after my last motion completed" and want that
    one or a later one). Returns `None` on timeout; `timeout=None` blocks indefinitely.
  - `.degraded: bool` — see "Degraded detection" below. Sticky once `True` (the worker itself never
    clears it; a session that wants to retry constructs a fresh worker).

### Confidence policy (design 03 §7.3)

The model call itself excludes detections below `yolo_config.conf_threshold` (default 0.25) — that
filtering is `ChickpeaSegmenter.predict`'s job, not this provider's. On top of that:

| Field | Built from | Rationale |
|---|---|---|
| `crop_mask` / `instances` | **every** instance the model returned, after the `min_instance_area_px` sanity filter | safety-conservative: "occupied for safety" (design 03 §7.3) — a safety guard (e.g. `LowerTool`) should see everything, even a borderline-confidence detection |
| `stats` (`ClumpStats`) | only instances with `confidence >= stats_conf_threshold` (default 0.5) | "counted for statistics" (design 03 §7.3) — reset/counting heuristics shouldn't be swayed by borderline detections |

`min_instance_area_px` (default 0, i.e. off) and `stats_conf_threshold` (default 0.5) are **not**
fields on `config_schema.YoloConfig` (Wave 1 schema doesn't carry them) — both are read via
`getattr(yolo_config, name, default)`, so a future schema addition is picked up with no code change
here. Logged in `design/IMPLEMENTATION_LOG.md`.

### Degraded detection (design 03 §7.4)

`SegmentationWorker` never raises out of its own thread and never touches `shutdown_event` — it
only sets `.degraded = True` and keeps running, for the session layer (Wave 5+) to observe and act
on (e.g. falling back to `BackgroundDiffProvider`, or pausing for intervention). Two independent,
one-way (never auto-reset) triggers:

1. `provider.compute()` raises on `max_consecutive_failures` (default 3) observations in a row.
2. `OccupancyResult.stats.num_clumps` (chosen over `len(instances)` so this works identically for
   `BackgroundDiffProvider`, whose `instances` is always `()`) drops to 0 for
   `zero_detection_streak_for_degraded` (default 5, i.e. ~2s at the template's 2.5 FPS) consecutive
   *successful* results — but only once at least one prior result showed `num_clumps > 0`. A
   workspace that starts empty and stays empty (e.g. before setup) is not "degraded", it's just
   empty; design 03 §7.4's trigger is specifically "0 detections while the previous frames showed
   many". Pass `zero_detection_streak_for_degraded=None` to disable this trigger (the
   consecutive-failure trigger still applies).

Both numeric defaults (3, 5) are this task's choice, not dictated by the design docs — logged in
`design/IMPLEMENTATION_LOG.md`.

### Threading

- `YoloOccupancyProvider.compute()` is synchronous; only `SegmentationWorker`'s single thread calls
  it for a given instance (design 03 §2.1: "GPU note ... one worker thread suffices").
- `SegmentationWorker` owns exactly one background thread. `latest()`, `await_result()`,
  `.degraded`, `start()`/`stop()`/`is_running()` are safe from any thread — same model as
  `ObservationSource`.

### Config keys consumed

`perception.yolo.weights_path`, `.conf_threshold`, `.iou_threshold`, `.imgsz`
(`config_schema.YoloConfig`); `perception.crop`/`.grid` indirectly via `CoordinateFrames`.

### Porting notes

- `recording_seg.py::SegmentationThread` — the loop shape (poll for a new frame, skip if
  unchanged, segment, publish) is `SegmentationWorker._run`'s direct ancestor. Differences: (1)
  subscribes to `ObservationSource`'s pub/sub instead of polling a shared-state timestamp under a
  lock; (2) publishes a structured `OccupancyResult` instead of a bare mask array into mutable
  shared state; (3) explicit `latest()`/`await_result()` API instead of callers reaching into
  `shared_state.latest_mask`; (4) tracks `.degraded` (no legacy equivalent — `SegmentationThread`
  only logged exceptions and looped forever).
- `image_collector/chickpea_segmenter.py::ChickpeaSegmenter` — reused as-is via a lazy import, not
  copied.

### Testing

`autograsper/tests/test_perception_yolo_segmenter.py` — construction-is-lazy (`ultralytics`/
`image_collector.chickpea_segmenter` not in `sys.modules` after importing the module) + a real-model
smoke test gated on `RUN_MODEL_TESTS=1` and the weights file existing.
`autograsper/tests/test_perception_worker.py` — `SegmentationWorker` against a real
`ObservationSource` (`DryRunRobot`) + stub providers: `latest()`/`await_result()` semantics
(including the `>=` boundary and timeout-to-`None`), latest-wins backlog skipping under a slow
provider, degraded-on-repeated-failures, degraded-on-zero-detection-streak-after-occupancy, and
never-degraded when occupancy was never seen. Run:

```
cd /home/alon/Code/cloudgripper-api
/home/alon/anaconda3/envs/cge/bin/python -m pytest autograsper/tests/test_perception_worker.py autograsper/tests/test_perception_yolo_segmenter.py -q
```

## `ToolGripChecker` (`tool_grip.py`)

### Purpose

Top-camera, fixed-ROI, color-threshold check for "is the tool correctly gripped" — orthogonal to
the bottom-camera occupancy providers above (different camera, different question). Ports as-is
per design 03 §3.4 ("The top-camera color-ROI grip check ... [is] orthogonal to the bottom-camera
segmenter and port as-is").

### Public API

`autograsper/perception/tool_grip.py`:

- `GripCheckResult(quality: float, ok: bool, roi_used: Tuple[int, int, int, int])` — frozen.
  `quality` is the fraction of the ROI's pixels within the configured tool color range (`[0, 1]`);
  `ok = quality >= config.detection_threshold`; `roi_used` is `(x_min, x_max, y_min, y_max)` in
  `Observation.top_image` pixels, for debugging/visualization.
- `ToolGripChecker(config: config_schema.ToolCheckConfig, debug: Optional[DebugSink] = None)`.
  - `check(obs: Observation) -> GripCheckResult` — extracts the fractional ROI (`config.roi.x`/
    `.y`, as fractions of `top_image` width/height) and analyzes its color composition. An
    empty/degenerate ROI returns `quality=0.0` rather than raising.

### Config keys consumed

`tool_check.enabled` (read by the caller deciding whether to call this at all — this class doesn't
check it itself), `tool_check.roi.x`/`.y`, `.detection_threshold`, `.color_lower_bgr`,
`.color_upper_bgr` (`config_schema.ToolCheckConfig`).

### Porting notes

- `custom_graspers/granular_pusher.py::RandomPushGrasper.check_tool_grip` — the fractional-ROI
  extraction from the top image (legacy defaults `x=[0.4, 0.7]`, `y=[0.1, 0.8]`, now sourced from
  config, never hardcoded) and the legacy hardcoded tool color range (`lower_bgr=(160, 160, 120)`,
  `upper_bgr=(190, 210, 190)`, now `config_schema.ToolCheckConfig.color_lower_bgr`/
  `.color_upper_bgr`).
- `object_tracker/tool_user_utils.py::analyze_tool_grip` — the color-mask-fraction computation,
  copied into `_analyze_tool_grip` verbatim except: (1) the two unconditional
  `cv2.imwrite("tool_grip_analysis.png", ...)` / `cv2.imwrite("tool_grip_original.png", ...)` calls
  to the current working directory are removed (CONVENTIONS.md) — an optional `DebugSink` writes
  the same two artifacts instead, no-op unless enabled; (2) a zero-area ROI now returns `0.0`
  instead of raising `ZeroDivisionError`.

### Threading

Stateless after construction; `check()` is a pure function of its argument (plus the optional
thread-safe `DebugSink` write) and safe to call from any thread.

### Testing

`autograsper/tests/test_tool_grip.py` — full-ROI tool color -> quality 1.0; no tool color in ROI ->
quality 0.0; partial fill -> quality between 0 and 1; zero-width ROI -> quality 0.0 without
raising; `ok` flag follows `detection_threshold`. Run:

```
cd /home/alon/Code/cloudgripper-api
/home/alon/anaconda3/envs/cge/bin/python -m pytest autograsper/tests/test_tool_grip.py -q
```

## Testing (background-diff provider)

`autograsper/tests/test_perception_background_diff.py` — synthetic reference (solid gray) + scene
(reference + a granule-colored blob + a robot-primary-color blob) images: the granule blob is
detected and survives `clean_mask`, the robot-colored blob is fully rejected by the color filters;
`ClumpStats` centroid falls within the blob; `grid_mask` has the configured shape and stays binary;
an identical-to-reference scene yields zero clumps; a missing reference image raises
`FileNotFoundError` at construction; result arrays are read-only. Run:

```
cd /home/alon/Code/cloudgripper-api
/home/alon/anaconda3/envs/cge/bin/python -m pytest autograsper/tests/test_perception_background_diff.py -q
```
