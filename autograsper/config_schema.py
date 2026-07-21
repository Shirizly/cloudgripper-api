"""Typed configuration schema for the granular-manipulation stack.

Design reference: `autograsper/design/02_proposed_architecture.md` §4 ("Config schema (typed)").
Legacy reference: scattered `config["section"][...]` reads in `autograsper/recording.py`,
`autograsper/grasper.py`, and `autograsper/custom_graspers/*` — this module replaces all of that
with one fail-fast, fully-typed loader. None of those legacy modules are imported here.

Responsibility
--------------
`load_config(path)` reads a YAML document and returns an immutable `Config` object, or raises
`ConfigError` listing *every* missing/invalid key found (not just the first). There is no
"silent default + KeyError three calls later" path left: every field either has an explicit,
documented default, or is required and reported by name (`section.key`) if absent/malformed.

Threading: pure, stateless, called once at process startup. No shared mutable state.

Coordinate frames / units of note (see `autograsper/docs/configuration.md` for the full list):
- `workspace.*_robot` fields are in the robot's normalized [0, 1]^2 xy plane.
- `workspace.*_px` / `perception.crop.*` fields are in the *full* bottom-camera pixel frame
  unless named otherwise.
- Heights (`grasp_height`, `sweep_height`, `clearance_height`) are robot-normalized z in [0, 1].
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml


class ConfigError(Exception):
    """Raised by `load_config` when the config document has missing or invalid keys.

    Carries the full list of problems (see `.errors`); the message renders all of them so a
    user can fix the config in one pass instead of one `KeyError` at a time.
    """

    def __init__(self, errors: List[str]):
        self.errors = list(errors)
        message = "Invalid configuration (%d problem%s):\n%s" % (
            len(self.errors),
            "" if len(self.errors) == 1 else "s",
            "\n".join(f"  - {e}" for e in self.errors),
        )
        super().__init__(message)


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraConfig:
    m: List[List[float]]  # 3x3 camera intrinsic matrix
    d: List[float]  # fisheye distortion coefficients
    H: List[List[float]]  # 3x3 homography (bottom-camera rectification)
    fps: float
    record: bool
    record_only_after_action: bool
    save_images_individually: bool
    save_bottom_raw: bool = False
    clip_length: Optional[int] = None


@dataclass(frozen=True)
class RobotConfig:
    idx: str
    token_env_var: str = "CLOUDGRIPPER_TOKEN"
    rotation_bias: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    episode_budget: Optional[int]
    n_pushes: int
    time_between_orders: float
    timeout_between_experiments: float


@dataclass(frozen=True)
class ManipulationBoundaryRobot:
    x: Tuple[float, float]
    y: Tuple[float, float]


@dataclass(frozen=True)
class WorkspaceConfig:
    fence_center: Tuple[float, float]
    fence_size: Tuple[float, float]
    manipulation_boundary_robot: ManipulationBoundaryRobot
    tool_length_robot: float
    tool_width_robot: float
    tool_dims_px: Tuple[int, int]
    homography_npz_path: str
    grasp_height: float
    sweep_height: float
    clearance_height: float
    safety_margin: float
    manipulation_boundary_px: Optional[Tuple[int, int, int, int]] = None


@dataclass(frozen=True)
class CropConfig:
    center_px: Tuple[int, int]
    size: Tuple[int, int]


@dataclass(frozen=True)
class GridConfig:
    height: int
    width: int


@dataclass(frozen=True)
class BackgroundDiffConfig:
    reference_image_path: str
    min_granule_size: int


@dataclass(frozen=True)
class YoloConfig:
    weights_path: str
    conf_threshold: float
    iou_threshold: float
    imgsz: int


@dataclass(frozen=True)
class PerceptionConfig:
    provider: str  # "background_diff" | "yolo" | "none"
    crop: CropConfig
    grid: GridConfig
    freshness_require_zero_for: List[str] = field(default_factory=list)
    background_diff: Optional[BackgroundDiffConfig] = None
    yolo: Optional[YoloConfig] = None


@dataclass(frozen=True)
class RoiConfig:
    x: Tuple[float, float]
    y: Tuple[float, float]


@dataclass(frozen=True)
class ToolCheckConfig:
    enabled: bool
    roi: RoiConfig
    detection_threshold: float
    color_lower_bgr: Tuple[int, int, int]
    color_upper_bgr: Tuple[int, int, int]


@dataclass(frozen=True)
class StorageConfig:
    base_dir: str = "autograsper/recorded_data"
    emit_transitions_online: bool = False
    tool_size_px_for_transitions: Tuple[int, int] = (8, 120)


@dataclass(frozen=True)
class UiConfig:
    enabled: bool = True
    port: int = 3000


@dataclass(frozen=True)
class Config:
    camera: CameraConfig
    robot: RobotConfig
    experiment: ExperimentConfig
    workspace: WorkspaceConfig
    perception: PerceptionConfig
    tool_check: ToolCheckConfig
    storage: StorageConfig
    ui: UiConfig


_PROVIDERS = ("background_diff", "yolo", "none")

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
#
# These collect *all* problems into an `errors: List[str]` list rather than raising on the first
# one, per CONVENTIONS.md / design 02 §4. Every helper is defensive: on failure it records an
# error and returns a placeholder/default so the caller can keep validating the rest of the
# document (the placeholder is never surfaced — if `errors` is non-empty at the end,
# `load_config` raises before constructing the final `Config`).

_NUMBER = (int, float)


def _is_number(v: Any) -> bool:
    return isinstance(v, _NUMBER) and not isinstance(v, bool)


def _section(doc: Dict[str, Any], name: str, errors: List[str]) -> Dict[str, Any]:
    """Return `doc[name]` as a dict, recording an error (and returning {}) if absent/wrong type."""
    value = doc.get(name)
    if value is None:
        errors.append(f"{name}: missing required section")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{name}: expected a mapping, got {type(value).__name__}")
        return {}
    return value


def _req(d: Dict[str, Any], path: str, key: str, kind: type, errors: List[str]) -> Any:
    """Required scalar field. `kind` is one of float/int/bool/str."""
    if key not in d:
        errors.append(f"{path}.{key}: missing required key")
        return None
    v = d[key]
    if kind is float:
        if _is_number(v):
            return float(v)
        errors.append(f"{path}.{key}: expected a number, got {type(v).__name__}")
        return None
    if kind is int:
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        errors.append(f"{path}.{key}: expected an int, got {type(v).__name__}")
        return None
    if kind is bool:
        if isinstance(v, bool):
            return v
        errors.append(f"{path}.{key}: expected a bool, got {type(v).__name__}")
        return None
    if kind is str:
        if isinstance(v, str):
            return v
        errors.append(f"{path}.{key}: expected a string, got {type(v).__name__}")
        return None
    raise AssertionError(f"unsupported kind {kind!r}")


def _opt(d: Dict[str, Any], path: str, key: str, kind: type, default: Any, errors: List[str]) -> Any:
    """Optional scalar field with a default; type-checked only if present."""
    if key not in d or d[key] is None:
        return default
    v = d[key]
    if kind is float:
        if _is_number(v):
            return float(v)
        errors.append(f"{path}.{key}: expected a number, got {type(v).__name__}")
        return default
    if kind is int:
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        errors.append(f"{path}.{key}: expected an int, got {type(v).__name__}")
        return default
    if kind is bool:
        if isinstance(v, bool):
            return v
        errors.append(f"{path}.{key}: expected a bool, got {type(v).__name__}")
        return default
    if kind is str:
        if isinstance(v, str):
            return v
        errors.append(f"{path}.{key}: expected a string, got {type(v).__name__}")
        return default
    raise AssertionError(f"unsupported kind {kind!r}")


def _num_list(
    d: Dict[str, Any],
    path: str,
    key: str,
    errors: List[str],
    *,
    length: Optional[int] = None,
    required: bool = True,
    default: Any = None,
) -> Any:
    if key not in d or d[key] is None:
        if required:
            errors.append(f"{path}.{key}: missing required key")
        return default
    v = d[key]
    ok = isinstance(v, list) and all(_is_number(x) for x in v)
    if ok and length is not None:
        ok = len(v) == length
    if not ok:
        expected = f"a list of {length} numbers" if length is not None else "a list of numbers"
        errors.append(f"{path}.{key}: expected {expected}, got {v!r}")
        return default
    return [float(x) for x in v]


def _int_list(
    d: Dict[str, Any],
    path: str,
    key: str,
    errors: List[str],
    *,
    length: int,
    required: bool = True,
    default: Any = None,
) -> Any:
    if key not in d or d[key] is None:
        if required:
            errors.append(f"{path}.{key}: missing required key")
        return default
    v = d[key]
    ok = isinstance(v, list) and len(v) == length and all(
        isinstance(x, int) and not isinstance(x, bool) for x in v
    )
    if not ok:
        errors.append(f"{path}.{key}: expected a list of {length} ints, got {v!r}")
        return default
    return tuple(v)


def _matrix(
    d: Dict[str, Any],
    path: str,
    key: str,
    rows: int,
    cols: int,
    errors: List[str],
    *,
    required: bool = True,
    default: Any = None,
) -> Any:
    if key not in d or d[key] is None:
        if required:
            errors.append(f"{path}.{key}: missing required key")
        return default
    v = d[key]
    ok = isinstance(v, list) and len(v) == rows and all(
        isinstance(r, list) and len(r) == cols and all(_is_number(x) for x in r) for r in v
    )
    if not ok:
        errors.append(f"{path}.{key}: expected a {rows}x{cols} matrix of numbers")
        return default
    return [[float(x) for x in row] for row in v]


def _range_pair(
    d: Dict[str, Any], path: str, key: str, errors: List[str], *, default: Any = None
) -> Optional[Tuple[float, float]]:
    if key not in d or d[key] is None:
        errors.append(f"{path}.{key}: missing required key")
        return default
    v = d[key]
    ok = isinstance(v, list) and len(v) == 2 and all(_is_number(x) for x in v)
    if not ok:
        errors.append(f"{path}.{key}: expected a [lo, hi] pair of numbers, got {v!r}")
        return default
    lo, hi = float(v[0]), float(v[1])
    if lo > hi:
        errors.append(f"{path}.{key}: lo ({lo}) must be <= hi ({hi})")
        return default
    return (lo, hi)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_camera(doc: Dict[str, Any], errors: List[str]) -> CameraConfig:
    d = _section(doc, "camera", errors)
    return CameraConfig(
        m=_matrix(d, "camera", "m", 3, 3, errors),
        d=_num_list(d, "camera", "d", errors),
        H=_matrix(d, "camera", "H", 3, 3, errors),
        fps=_req(d, "camera", "fps", float, errors),
        record=_req(d, "camera", "record", bool, errors),
        record_only_after_action=_req(d, "camera", "record_only_after_action", bool, errors),
        save_images_individually=_req(d, "camera", "save_images_individually", bool, errors),
        save_bottom_raw=_opt(d, "camera", "save_bottom_raw", bool, False, errors),
        clip_length=_opt(d, "camera", "clip_length", int, None, errors),
    )


def _build_robot(doc: Dict[str, Any], errors: List[str]) -> RobotConfig:
    d = _section(doc, "robot", errors)
    return RobotConfig(
        idx=_req(d, "robot", "idx", str, errors),
        token_env_var=_opt(d, "robot", "token_env_var", str, "CLOUDGRIPPER_TOKEN", errors),
        rotation_bias=_opt(d, "robot", "rotation_bias", float, 0.0, errors),
    )


def _build_experiment(doc: Dict[str, Any], errors: List[str]) -> ExperimentConfig:
    d = _section(doc, "experiment", errors)
    return ExperimentConfig(
        name=_req(d, "experiment", "name", str, errors),
        episode_budget=_opt(d, "experiment", "episode_budget", int, None, errors),
        n_pushes=_req(d, "experiment", "n_pushes", int, errors),
        time_between_orders=_req(d, "experiment", "time_between_orders", float, errors),
        timeout_between_experiments=_req(
            d, "experiment", "timeout_between_experiments", float, errors
        ),
    )


def _build_workspace(doc: Dict[str, Any], errors: List[str]) -> WorkspaceConfig:
    d = _section(doc, "workspace", errors)

    fence_center = _num_list(d, "workspace", "fence_center", errors, length=2)
    fence_size = _num_list(d, "workspace", "fence_size", errors, length=2)

    mb = d.get("manipulation_boundary_robot")
    if mb is None:
        errors.append("workspace.manipulation_boundary_robot: missing required key")
        mb = {}
    elif not isinstance(mb, dict):
        errors.append(
            "workspace.manipulation_boundary_robot: expected a mapping, got "
            f"{type(mb).__name__}"
        )
        mb = {}
    boundary_robot = ManipulationBoundaryRobot(
        x=_range_pair(mb, "workspace.manipulation_boundary_robot", "x", errors, default=(0.0, 0.0)),
        y=_range_pair(mb, "workspace.manipulation_boundary_robot", "y", errors, default=(0.0, 0.0)),
    )

    boundary_px = _int_list(
        d, "workspace", "manipulation_boundary_px", errors, length=4, required=False, default=None
    )

    return WorkspaceConfig(
        fence_center=tuple(fence_center) if fence_center is not None else (0.0, 0.0),
        fence_size=tuple(fence_size) if fence_size is not None else (0.0, 0.0),
        manipulation_boundary_robot=boundary_robot,
        manipulation_boundary_px=boundary_px,
        tool_length_robot=_req(d, "workspace", "tool_length_robot", float, errors),
        tool_width_robot=_req(d, "workspace", "tool_width_robot", float, errors),
        tool_dims_px=_int_list(d, "workspace", "tool_dims_px", errors, length=2) or (0, 0),
        homography_npz_path=_req(d, "workspace", "homography_npz_path", str, errors),
        grasp_height=_req(d, "workspace", "grasp_height", float, errors),
        sweep_height=_req(d, "workspace", "sweep_height", float, errors),
        clearance_height=_req(d, "workspace", "clearance_height", float, errors),
        safety_margin=_req(d, "workspace", "safety_margin", float, errors),
    )


def _build_perception(doc: Dict[str, Any], errors: List[str]) -> PerceptionConfig:
    d = _section(doc, "perception", errors)

    provider = _req(d, "perception", "provider", str, errors)
    if provider is not None and provider not in _PROVIDERS:
        errors.append(
            f"perception.provider: must be one of {_PROVIDERS}, got {provider!r}"
        )
        provider = None

    crop_d = d.get("crop")
    if not isinstance(crop_d, dict):
        errors.append("perception.crop: missing required section")
        crop_d = {}
    crop = CropConfig(
        center_px=_int_list(crop_d, "perception.crop", "center_px", errors, length=2) or (0, 0),
        size=_int_list(crop_d, "perception.crop", "size", errors, length=2) or (0, 0),
    )

    grid_d = d.get("grid")
    if not isinstance(grid_d, dict):
        errors.append("perception.grid: missing required section")
        grid_d = {}
    grid = GridConfig(
        height=_req(grid_d, "perception.grid", "height", int, errors) or 0,
        width=_req(grid_d, "perception.grid", "width", int, errors) or 0,
    )

    freshness = d.get("freshness_require_zero_for", [])
    if not isinstance(freshness, list) or not all(isinstance(x, str) for x in freshness):
        errors.append(
            "perception.freshness_require_zero_for: expected a list of strings, got "
            f"{freshness!r}"
        )
        freshness = []

    background_diff = None
    yolo = None

    if provider == "background_diff":
        bd = d.get("background_diff")
        if not isinstance(bd, dict):
            errors.append(
                "perception.background_diff: required when perception.provider == "
                "'background_diff'"
            )
            bd = {}
        background_diff = BackgroundDiffConfig(
            reference_image_path=_req(
                bd, "perception.background_diff", "reference_image_path", str, errors
            ),
            min_granule_size=_req(
                bd, "perception.background_diff", "min_granule_size", int, errors
            ),
        )
    elif isinstance(d.get("background_diff"), dict):
        # Present but not the active provider: parse leniently, still validated if given.
        bd = d["background_diff"]
        background_diff = BackgroundDiffConfig(
            reference_image_path=_opt(
                bd, "perception.background_diff", "reference_image_path", str, "", errors
            ),
            min_granule_size=_opt(
                bd, "perception.background_diff", "min_granule_size", int, 0, errors
            ),
        )

    if provider == "yolo":
        y = d.get("yolo")
        if not isinstance(y, dict):
            errors.append("perception.yolo: required when perception.provider == 'yolo'")
            y = {}
        yolo = YoloConfig(
            weights_path=_req(y, "perception.yolo", "weights_path", str, errors),
            conf_threshold=_req(y, "perception.yolo", "conf_threshold", float, errors),
            iou_threshold=_req(y, "perception.yolo", "iou_threshold", float, errors),
            imgsz=_req(y, "perception.yolo", "imgsz", int, errors),
        )
    elif isinstance(d.get("yolo"), dict):
        y = d["yolo"]
        yolo = YoloConfig(
            weights_path=_opt(y, "perception.yolo", "weights_path", str, "", errors),
            conf_threshold=_opt(y, "perception.yolo", "conf_threshold", float, 0.25, errors),
            iou_threshold=_opt(y, "perception.yolo", "iou_threshold", float, 0.45, errors),
            imgsz=_opt(y, "perception.yolo", "imgsz", int, 640, errors),
        )

    return PerceptionConfig(
        provider=provider or "none",
        crop=crop,
        grid=grid,
        freshness_require_zero_for=freshness,
        background_diff=background_diff,
        yolo=yolo,
    )


def _build_tool_check(doc: Dict[str, Any], errors: List[str]) -> ToolCheckConfig:
    d = _section(doc, "tool_check", errors)
    roi_d = d.get("roi")
    if not isinstance(roi_d, dict):
        errors.append("tool_check.roi: missing required section")
        roi_d = {}
    roi = RoiConfig(
        x=_range_pair(roi_d, "tool_check.roi", "x", errors, default=(0.0, 0.0)),
        y=_range_pair(roi_d, "tool_check.roi", "y", errors, default=(0.0, 0.0)),
    )
    return ToolCheckConfig(
        enabled=_req(d, "tool_check", "enabled", bool, errors),
        roi=roi,
        detection_threshold=_req(d, "tool_check", "detection_threshold", float, errors),
        color_lower_bgr=_int_list(d, "tool_check", "color_lower_bgr", errors, length=3) or (0, 0, 0),
        color_upper_bgr=_int_list(d, "tool_check", "color_upper_bgr", errors, length=3) or (0, 0, 0),
    )


def _build_storage(doc: Dict[str, Any], errors: List[str]) -> StorageConfig:
    d = doc.get("storage", {})
    if not isinstance(d, dict):
        errors.append(f"storage: expected a mapping, got {type(d).__name__}")
        d = {}
    return StorageConfig(
        base_dir=_opt(d, "storage", "base_dir", str, "autograsper/recorded_data", errors),
        emit_transitions_online=_opt(d, "storage", "emit_transitions_online", bool, False, errors),
        tool_size_px_for_transitions=_int_list(
            d, "storage", "tool_size_px_for_transitions", errors, length=2, required=False, default=(8, 120)
        ),
    )


def _build_ui(doc: Dict[str, Any], errors: List[str]) -> UiConfig:
    d = doc.get("ui", {})
    if not isinstance(d, dict):
        errors.append(f"ui: expected a mapping, got {type(d).__name__}")
        d = {}
    return UiConfig(
        enabled=_opt(d, "ui", "enabled", bool, True, errors),
        port=_opt(d, "ui", "port", int, 3000, errors),
    )


def load_config(path: str) -> Config:
    """Load and validate a YAML config file into a `Config`.

    Raises `ConfigError` (with `.errors` listing every problem found) if any required key is
    missing or has the wrong type. Unlike the legacy `config["section"][key]` reads this never
    raises a bare `KeyError`/`TypeError` — every problem is collected and reported together.
    """
    with open(path, "r") as f:
        doc = yaml.safe_load(f) or {}

    if not isinstance(doc, dict):
        raise ConfigError([f"top-level document must be a mapping, got {type(doc).__name__}"])

    errors: List[str] = []
    camera = _build_camera(doc, errors)
    robot = _build_robot(doc, errors)
    experiment = _build_experiment(doc, errors)
    workspace = _build_workspace(doc, errors)
    perception = _build_perception(doc, errors)
    tool_check = _build_tool_check(doc, errors)
    storage = _build_storage(doc, errors)
    ui = _build_ui(doc, errors)

    if errors:
        raise ConfigError(errors)

    return Config(
        camera=camera,
        robot=robot,
        experiment=experiment,
        workspace=workspace,
        perception=perception,
        tool_check=tool_check,
        storage=storage,
        ui=ui,
    )


def load_config_from_env(env_var: str = "AUTOGRASPER_CONFIG") -> Config:
    """Convenience wrapper: load the config path from an environment variable."""
    path = os.environ.get(env_var)
    if not path:
        raise ConfigError([f"environment variable {env_var} is not set"])
    return load_config(path)
