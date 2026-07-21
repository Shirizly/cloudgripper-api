# `autograsper/config_schema.py` — typed config schema

Status: **implemented** (Wave 1). Design reference:
[`../design/02_proposed_architecture.md`](../design/02_proposed_architecture.md) §4.

## Purpose

Replaces scattered `config["section"][key]` reads (and their silent defaults +
`KeyError`s three calls deep — see
[`../design/01_current_architecture.md`](../design/01_current_architecture.md) §3.1 defect #4:
"`config.yaml` lacks most keys the granular code reads") with one fail-fast loader that reports
**every** problem in the document at once, not just the first one hit.

## Public API

`autograsper/config_schema.py`:

- `load_config(path: str) -> Config` — reads a YAML file, validates it, returns an immutable
  `Config`, or raises `ConfigError`.
- `load_config_from_env(env_var: str = "AUTOGRASPER_CONFIG") -> Config` — convenience wrapper for
  the future `main_granular.py` composition root.
- `ConfigError(Exception)` — `.errors: List[str]`, each entry formatted `"section.key: problem"`.
  `str(exc)` renders all of them as a bulleted list.
- `Config` and its section dataclasses (all `@dataclass(frozen=True)`): `CameraConfig`,
  `RobotConfig`, `ExperimentConfig`, `WorkspaceConfig` (+ `ManipulationBoundaryRobot`),
  `PerceptionConfig` (+ `CropConfig`, `GridConfig`, `BackgroundDiffConfig`, `YoloConfig`),
  `ToolCheckConfig` (+ `RoiConfig`), `StorageConfig`, `UiConfig`.

## Data flow

```
YAML file ──yaml.safe_load──► dict ──per-section builder──► errors: List[str] (accumulated)
                                                          └► Config (only if errors == [])
```

Each section builder (`_build_camera`, `_build_robot`, ...) is defensive: a missing section or
key never raises a Python `KeyError`/`TypeError` mid-build — it appends a human-readable message
to `errors` and substitutes a placeholder so the *rest* of the document still gets checked. If
`errors` is non-empty after all sections are built, `load_config` raises `ConfigError(errors)`
and the (placeholder-containing) `Config` object it built internally is discarded — callers never
see a partially-valid `Config`.

## Config keys / schema

See `autograsper/granular-config.yaml` for a runnable template with real legacy values filled in
and `# PLACEHOLDER — calibrate` comments on anything with no recoverable legacy value.

| Section | Key | Type | Required | Notes |
|---|---|---|---|---|
| `camera` | `m` | 3x3 float matrix | yes | intrinsics |
| | `d` | list[float] | yes | fisheye distortion coeffs |
| | `H` | 3x3 float matrix | yes | homography (bottom-camera rectification) |
| | `fps` | float | yes | |
| | `record` | bool | yes | |
| | `record_only_after_action` | bool | yes | |
| | `save_images_individually` | bool | yes | |
| | `save_bottom_raw` | bool | no (default `False`) | |
| | `clip_length` | int \| null | no (default `None`) | |
| `robot` | `idx` | str | yes | e.g. `"robot23"` |
| | `token_env_var` | str | no (default `"CLOUDGRIPPER_TOKEN"`) | |
| | `rotation_bias` | float | no (default `0`) | see `hardware.md` |
| `experiment` | `name` | str | yes | |
| | `episode_budget` | int \| null | no (default `None`) | |
| | `n_pushes` | int | yes | |
| | `time_between_orders` | float | yes | |
| | `timeout_between_experiments` | float | yes | |
| `workspace` | `fence_center` | [x, y] | yes | robot-normalized |
| | `fence_size` | [w, h] | yes | robot-normalized |
| | `manipulation_boundary_robot.x/.y` | [lo, hi] | yes | robot-normalized |
| | `manipulation_boundary_px` | [xmin,ymin,xmax,ymax] \| null | no | full-px frame |
| | `tool_length_robot` / `tool_width_robot` | float | yes | robot-normalized |
| | `tool_dims_px` | [w, h] ints | yes | full-px frame |
| | `homography_npz_path` | str | yes | |
| | `grasp_height` / `sweep_height` / `clearance_height` | float | yes | robot-normalized z |
| | `safety_margin` | float | yes | |
| `perception` | `provider` | `"background_diff"` \| `"yolo"` \| `"none"` | yes | |
| | `crop.center_px` / `crop.size` | [x,y] / [w,h] ints | yes | full-px frame |
| | `grid.height` / `grid.width` | int | yes | occupancy grid resolution |
| | `freshness_require_zero_for` | list[str] | no (default `[]`) | see note below |
| | `background_diff.reference_image_path` | str | required iff `provider == "background_diff"` | |
| | `background_diff.min_granule_size` | int | required iff `provider == "background_diff"` | |
| | `yolo.weights_path` | str | required iff `provider == "yolo"` | |
| | `yolo.conf_threshold` / `.iou_threshold` | float | required iff `provider == "yolo"` | |
| | `yolo.imgsz` | int | required iff `provider == "yolo"` | |
| `tool_check` | `enabled` | bool | yes | |
| | `roi.x` / `roi.y` | [lo, hi] | yes | |
| | `detection_threshold` | float | yes | |
| | `color_lower_bgr` / `color_upper_bgr` | [b,g,r] ints | yes | |
| `storage` | `base_dir` | str | no (default `"autograsper/recorded_data"`) | |
| | `emit_transitions_online` | bool | no (default `False`) | |
| | `tool_size_px_for_transitions` | [w,h] ints | no (default `(8, 120)`) | |
| `ui` | `enabled` | bool | no (default `True`) | |
| | `port` | int | no (default `3000`) | |

Type coercion: any field typed `float` also accepts a YAML `int` literal (coerced to `float`);
`bool` is never accepted where `int`/`float` is expected (Python's `bool <: int` is explicitly
guarded against, otherwise e.g. `save_bottom_raw: true` could silently satisfy an `int` field
elsewhere).

## Threading

None — `load_config` is a pure function, called once at process startup.

## Porting notes

- Calibration matrices (`camera.m`, `camera.d`, `camera.H`, `workspace.homography_npz_path`) have
  **no defaults** — required, per the task spec, since a wrong or interpolated calibration value
  would silently corrupt every downstream measurement.
- `camera.m` / `camera.d` values in `granular-config.yaml` are copied verbatim from
  `autograsper/legacy/config.yaml` (the existing, partial config fragment).
- `workspace.fence_center` (`[0.5, 0.49]`), `.fence_size` (`[0.94, 0.955]`),
  `.grasp_height`/`.sweep_height`/`.clearance_height` (`0.34`/`0.57`/`0.8`),
  `.tool_length_robot`/`.tool_width_robot` (`0.36`/`0.017`),
  `perception.crop.center_px`/`.size` (`[275, 200]`/`[360, 360]`), `workspace.tool_dims_px`
  (`[8, 120]`), `robot.idx` (`"robot23"`), `camera.fps` (`2.5`), and
  `experiment.time_between_orders` (`2.5`) are recovered from
  `../design/01_current_architecture.md` §3.5-3.6 and `autograsper/legacy/config.yaml` — see that
  design doc for which legacy call sites they came from.
- Everything else the design doc says "lived outside version control" (design 01 §5 defect #4:
  neither `chickpeas-config.yaml` nor `chickpea-config.yaml` exists in the repo) is filled with a
  best-effort placeholder and a `# PLACEHOLDER — calibrate` comment in the template — see that
  file for the exact list (camera.H, robot.rotation_bias, workspace.manipulation_boundary_*,
  workspace.safety_margin, perception.grid, perception.background_diff.*, perception.yolo.weights_path,
  tool_check.*).
- `perception.background_diff` / `perception.yolo` are **conditionally required**: only the
  section matching the active `provider` must be fully populated; the other, if present, is
  parsed leniently (defaults substituted for any of its own missing sub-keys) so a config can
  carry both without one blocking validation of the other (e.g. switching providers without
  re-editing the file). This is not dictated verbatim by design 02 §4 (which lists both
  sub-sections without specifying which are conditional) — logged in
  [`../design/IMPLEMENTATION_LOG.md`](../design/IMPLEMENTATION_LOG.md).
- `perception.freshness_require_zero_for` defaulting to `["MOVE_XY", "ROTATE"]` in the template
  (an independent choice, not a schema default — the schema itself defaults to `[]`) reflects the
  legacy `interaction_since_last_mask` flag semantics (design 01 §3.5: mask refresh is lazy,
  triggered only after the tool has touched the scene via a planar move or rotate) — see
  `../design/IMPLEMENTATION_LOG.md`.

## Testing

`autograsper/tests/test_config_schema.py` — the checked-in template loads successfully; deleting
several required keys at once reports all of them in one `ConfigError`; type errors (wrong
Python type for a scalar, wrong matrix shape, invalid `perception.provider` enum value) are
reported; optional-key defaults apply when absent. Run:

```
cd /home/alon/Code/cloudgripper-api
/home/alon/anaconda3/envs/cge/bin/python -m pytest autograsper/tests/test_config_schema.py -q
```
