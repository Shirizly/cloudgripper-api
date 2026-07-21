"""Dataset writers — session directory numbering + append-mode `.jsonl`/legacy `.json` sinks +
online transition-dataset emission (Wave 5).

Design reference: `autograsper/design/02_proposed_architecture.md` §3.5/§3.6/§6 ("Keep the dataset
contract ... `states.json`/`orders.json` gain JSONL siblings first, with the array files produced
at episode end instead of per frame"); `autograsper/design/03_segmenter_native_design.md` §4
(`masks_meta.jsonl`), §5 (online transition emission — `TransitionWriter`); `autograsper/MD
files/transition_dataset_design.md` (the `_{id}_data.pt`/`_{id}_config.yaml` format, ported
exactly).

Porting notes (copied and adapted, not imported):
- `autograsper/file_manager.py::FileManager.get_session_dirs` — the "next integer directory under
  base_dir" numbering scheme and `task/`/`restore/` subdirectory creation, ported into
  `create_session_dirs`. `Images/`/`Bottom_Images/`/`Masks/` are NOT created here (legacy's
  `create_image_dirs`) — the recorder creates those lazily on first frame (`recording/recorder.py`).
- `autograsper/recording.py::Recorder.save_state` — the `states.json` row shape (flattened robot
  state dict + `"time"` + `"frame_index"` + an optional `"action"` block with `action_id`/
  `action_type`/`phase`/`start_frame`/`is_planar_2d`/`action_details`/`description`), ported
  verbatim into `StatesWriter.record`. Legacy rewrote the whole `states.json` array on every frame
  (design 01 §5 defect #13, "O(n²) rewrite"); this module fixes that by appending to `states.jsonl`
  (flush per row) during recording and producing the legacy `states.json` array only once, at
  `finalize()` — both files are kept (`.jsonl` for streaming/crash-safety, `.json` for existing
  tooling).
- `autograsper/library/utils.py::write_order`'s `orders.json` row shape (`order_type`/
  `order_value`/`time`), as already extended by `execution.executor.Executor`'s `order_sink`
  contract (`robot_reported_time`/`frame_index` additive) — `OrdersWriter` is that sink, same
  jsonl-then-finalize pattern as `StatesWriter`.
- `autograsper/recording.py::Recorder.save_action_summary` — `actions.json`'s
  `{"total_actions": N, "actions": [...]}` shape, including legacy's "skip writing an empty file"
  behavior (`if len(actions) == 0: return`), ported into `ActionsWriter.finalize`.
- `autograsper/coordinator.py::DataCollectionCoordinator._on_resetting_state` — `status.txt`
  (`"success"`/`"fail"`) at the session (not task/restore) level, ported into `write_status`.

`TransitionWriter` is new (no direct legacy equivalent — the offline `extract_transitions.py`/
`segment_transitions.py` pipeline this replaces for online use predates this refactor and remains
as a validation/backfill tool per design 03 §5's closing note). See its class docstring for the
mask-lookup contract it depends on (supplied by `recording.recorder.Recorder.mask_for_frame`).

Threading: every writer here is meant to be driven by a single thread at a time (the recorder's
consumer thread for `StatesWriter`/`MasksMetaWriter`, the session runner's control thread for
`OrdersWriter`/`TransitionWriter`/`ActionsWriter`/`write_status`) — none do their own locking.
`OrderSinkRouter` is the one exception: its `set()`/`__call__` are safe to call from different
threads (a plain attribute swap), matching `Executor`'s "single control thread drives it, but the
sink itself may be reassigned between episodes by the session runner" usage pattern.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

from autograsper.execution.actions import Action, ActionType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session directory numbering (ports file_manager.py::FileManager.get_session_dirs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionDirs:
    """One numbered session's directory triple."""

    session_dir: str
    task_dir: str
    restore_dir: str


def create_session_dirs(base_dir: str, experiment_name: str) -> SessionDirs:
    """Create the next numbered session directory under `base_dir/experiment_name`.

    Ported from `file_manager.py::FileManager.get_session_dirs` (same "max existing all-digit
    subdirectory name + 1" numbering), with `experiment_name` folded in as an extra path segment
    (legacy's caller, `coordinator.py::_create_new_data_point`, already did this join before
    calling `get_session_dirs` — `os.path.join(..., "recorded_data", self.experiment_name)`).
    `task/`/`restore/` are created eagerly (matching legacy); `Images/`/`Bottom_Images/`/`Masks/`
    are NOT — the recorder creates those lazily on first frame.
    """
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    session_ids = [int(x) for x in os.listdir(experiment_dir) if x.isdigit()]
    new_id = max(session_ids, default=0) + 1
    session_dir = os.path.join(experiment_dir, str(new_id))
    task_dir = os.path.join(session_dir, "task")
    restore_dir = os.path.join(session_dir, "restore")
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(restore_dir, exist_ok=True)
    return SessionDirs(session_dir=session_dir, task_dir=task_dir, restore_dir=restore_dir)


def write_status(session_dir: str, status: str) -> str:
    """Write `<session_dir>/status.txt` (`"success"` or `"fail"`). Ported from
    `coordinator.py::_on_resetting_state`."""
    if status not in ("success", "fail"):
        raise ValueError(f"write_status: status must be 'success' or 'fail', got {status!r}")
    path = os.path.join(session_dir, "status.txt")
    with open(path, "w") as f:
        f.write(status)
    return path


# ---------------------------------------------------------------------------
# Append-mode JSONL sink
# ---------------------------------------------------------------------------


class JsonlWriter:
    """Append-only `.jsonl` file (one JSON object per line, flushed per row), plus an in-memory
    copy of every row appended so far (used by callers that also need to produce a legacy `.json`
    array at `finalize()` time without re-reading the file back)."""

    def __init__(self, path: str) -> None:
        self._path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._rows: List[Dict[str, Any]] = []

    def append(self, row: Dict[str, Any]) -> None:
        self._rows.append(row)
        with open(self._path, "a") as f:
            f.write(json.dumps(row, default=_json_default) + "\n")
            f.flush()

    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows)

    @property
    def path(self) -> str:
        return self._path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return str(value)


# ---------------------------------------------------------------------------
# states.jsonl / states.json
# ---------------------------------------------------------------------------


class StatesWriter:
    """`states.jsonl` (append, crash-safe) + `states.json` (legacy array, written once at
    `finalize()`). Row shape ported verbatim from `recording.py::Recorder.save_state`."""

    def __init__(self, dir_path: str) -> None:
        self._dir = dir_path
        self._jsonl = JsonlWriter(os.path.join(dir_path, "states.jsonl"))

    def record(
        self,
        robot_state: Optional[Dict[str, Any]],
        time_: float,
        frame_index: int,
        action: Optional[Action] = None,
    ) -> None:
        row: Dict[str, Any] = dict(robot_state) if robot_state else {}
        row["time"] = time_
        row["frame_index"] = frame_index
        if action is not None:
            row["action"] = {
                "action_id": action.action_id,
                "action_type": action.action_type.value,
                "phase": action.phase.value,
                "start_frame": action.start_frame,
                "is_planar_2d": action.is_planar_2d,
                "action_details": action.action_details,
                "description": action.description,
            }
        self._jsonl.append(row)

    def finalize(self) -> str:
        path = os.path.join(self._dir, "states.json")
        with open(path, "w") as f:
            json.dump(self._jsonl.rows(), f, indent=4, default=_json_default)
        return path


# ---------------------------------------------------------------------------
# orders.jsonl / orders.json
# ---------------------------------------------------------------------------


class OrdersWriter:
    """`orders.jsonl` (append) + `orders.json` (legacy array at `finalize()`). Row shape matches
    `execution.executor.Executor`'s `order_sink` contract verbatim: legacy keys `order_type`/
    `order_value`/`time` plus additive `robot_reported_time`/`frame_index`."""

    def __init__(self, dir_path: str) -> None:
        self._dir = dir_path
        self._jsonl = JsonlWriter(os.path.join(dir_path, "orders.jsonl"))

    def record(self, order_record: Dict[str, Any]) -> None:
        """Matches `Executor`'s `order_sink=callable(record_dict)` signature exactly — pass this
        bound method (or an `OrderSinkRouter` wrapping it) straight to `Executor(order_sink=...)`.
        """
        self._jsonl.append(order_record)

    def finalize(self) -> str:
        path = os.path.join(self._dir, "orders.json")
        with open(path, "w") as f:
            json.dump(self._jsonl.rows(), f, indent=4, default=_json_default)
        return path


class OrderSinkRouter:
    """Mutable indirection so one `Executor` (constructed once, for the whole run) can have its
    `order_sink` redirected to a fresh `OrdersWriter` every episode/phase without reconstructing the
    `Executor`. Pass an instance of this as `Executor(order_sink=router)`; the session runner calls
    `router.set(writer.record)` per episode and `router.set(None)` between episodes.

    Safe to call `set()`/`__call__` from different threads (a single attribute swap/read); no
    locking needed since `Executor` is itself only ever driven by one thread at a time and the
    session runner never calls `set()` concurrently with an in-flight `run_plan()`.
    """

    def __init__(self) -> None:
        self._sink: Optional[Callable[[Dict[str, Any]], None]] = None

    def set(self, sink: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        self._sink = sink

    def __call__(self, record: Dict[str, Any]) -> None:
        sink = self._sink
        if sink is not None:
            sink(record)


# ---------------------------------------------------------------------------
# actions.json (finalize-only, no jsonl sibling — matches legacy's single save point)
# ---------------------------------------------------------------------------


class ActionsWriter:
    """`actions.json` — written only at `finalize()` (no incremental jsonl sibling; actions are
    already durably recorded via `execution.actions.ActionTracker`'s in-memory list and this
    module's `TransitionWriter`/other completion callbacks as they complete). Ported from
    `recording.py::Recorder.save_action_summary`, including its "skip writing an empty file"
    behavior."""

    @staticmethod
    def finalize(dir_path: str, actions: List[Action]) -> Optional[str]:
        if not actions:
            return None
        path = os.path.join(dir_path, "actions.json")
        data = {"total_actions": len(actions), "actions": [a.to_dict() for a in actions]}
        with open(path, "w") as f:
            json.dump(data, f, indent=4, default=_json_default)
        return path


# ---------------------------------------------------------------------------
# masks_meta.jsonl (design 03 §4)
# ---------------------------------------------------------------------------


class MasksMetaWriter:
    """`masks_meta.jsonl` — one row per saved mask, linking `frame_index -> source_seq,
    num_instances, mask_area` (design 03 §4) so downstream tools know mask provenance without
    recomputing. No legacy equivalent; append-only, no finalize step needed."""

    def __init__(self, dir_path: str) -> None:
        self._jsonl = JsonlWriter(os.path.join(dir_path, "masks_meta.jsonl"))

    def record(
        self,
        frame_index: int,
        source_seq: int,
        num_instances: Optional[int],
        mask_area: Optional[int],
    ) -> None:
        self._jsonl.append(
            {
                "frame_index": frame_index,
                "source_seq": source_seq,
                "num_instances": num_instances,
                "mask_area": mask_area,
            }
        )


# ---------------------------------------------------------------------------
# TransitionWriter — online emission of the RealData transition-dataset format (design 03 §5)
# ---------------------------------------------------------------------------


_REQUIRED_PUSH_DETAILS = ("start_x", "start_y", "angle", "end_x", "end_y", "height")


class TransitionWriter:
    """Subscribes to completed top-level `Push` actions (via
    `executor.register_completion_callback(writer.on_action_completed)`) and, once per episode, at
    `finalize()`, writes the `_{id}_data.pt` + `_{id}_config.yaml` pair described in `MD
    files/transition_dataset_design.md` — implemented exactly against that spec.

    `mask_for_frame(frame_index, mode)` is supplied by the caller (in practice
    `recording.recorder.Recorder.mask_for_frame`, bound): `mode="before"` must return the latest
    saved grid mask with frame `<= frame_index`; `mode="after"` the earliest with frame
    `>= frame_index`. Passing the *same* recorder instance across consecutive pushes within one
    episode automatically satisfies design 03 §5's "edge rule" (share the boundary mask between
    consecutive pushes when push *i*'s "after" frame and push *i+1*'s "before" frame coincide) —
    both queries resolve to the same underlying ring-buffer entry with no special-casing needed
    here.

    One `TransitionWriter` is meant to be constructed once per run (registered once as a completion
    callback) and `finalize()`d once per episode (task phase only) — `finalize()` clears its
    pending-push buffer so the next episode starts clean; if `mask_for_frame` cannot resolve masks
    for a given push (e.g. the recorder's ring buffer was reset before `finalize()` ran), that push
    is skipped with a warning rather than aborting the whole write.
    """

    def __init__(
        self,
        frames,
        grid_height: int,
        grid_width: int,
        tool_size_px: Tuple[int, int],
        mask_for_frame: Callable[[int, str], Optional[np.ndarray]],
        experiment_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._frames = frames
        self._grid_h = int(grid_height)
        self._grid_w = int(grid_width)
        self._tool_size_px = tuple(tool_size_px)
        self._mask_for_frame = mask_for_frame
        self._experiment_meta = dict(experiment_meta) if experiment_meta else {}
        self._pending: List[Dict[str, Any]] = []

    def on_action_completed(self, action: Action) -> None:
        """Register with `executor.register_completion_callback`. Filters for completed top-level
        `Push` actions per `docs/execution.md`'s Wave 5 contract: `action.parent_id is None and
        action.action_type is ActionType.MOVE_XY and action.is_planar_2d`."""
        if action.parent_id is not None:
            return
        if action.action_type is not ActionType.MOVE_XY or not action.is_planar_2d:
            return
        details = action.action_details or {}
        if not all(k in details for k in _REQUIRED_PUSH_DETAILS):
            return
        if action.start_frame is None or action.end_frame is None:
            return
        self._pending.append(
            {
                "start_frame": action.start_frame,
                "end_frame": action.end_frame,
                "start_x": details["start_x"],
                "start_y": details["start_y"],
                "end_x": details["end_x"],
                "end_y": details["end_y"],
                "angle_deg": details["angle"],
            }
        )

    def pending_count(self) -> int:
        return len(self._pending)

    def finalize(self, transitions_dir: str, episode_id: Any) -> Optional[Tuple[str, str]]:
        """Write `_{episode_id}_data.pt` + `_{episode_id}_config.yaml` under `transitions_dir` for
        every push collected since the last `finalize()` call, then clear the pending buffer.
        Returns `(data_path, config_path)`, or `None` if there was nothing to write (no pushes
        collected, or none had resolvable masks)."""
        pending = self._pending
        self._pending = []
        if not pending:
            return None

        masks_before: List[np.ndarray] = []
        masks_after: List[np.ndarray] = []
        p_starts: List[Tuple[float, float]] = []
        p_stops: List[Tuple[float, float]] = []
        angles: List[float] = []

        for push in pending:
            before = self._mask_for_frame(push["start_frame"], "before")
            after = self._mask_for_frame(push["end_frame"], "after")
            if before is None or after is None:
                logger.warning(
                    "TransitionWriter: no resolvable mask for push frames %s-%s; skipping",
                    push["start_frame"],
                    push["end_frame"],
                )
                continue
            masks_before.append(_mask_to_float01(before, self._grid_h, self._grid_w))
            masks_after.append(_mask_to_float01(after, self._grid_h, self._grid_w))
            p_starts.append(self._robot_to_grid_px(push["start_x"], push["start_y"]))
            p_stops.append(self._robot_to_grid_px(push["end_x"], push["end_y"]))
            angles.append(float(np.deg2rad(push["angle_deg"])))

        if not masks_before:
            return None

        os.makedirs(transitions_dir, exist_ok=True)
        data = {
            "masks_before": torch.tensor(np.stack(masks_before), dtype=torch.float32),
            "masks_after": torch.tensor(np.stack(masks_after), dtype=torch.float32),
            "p_starts_px": torch.tensor(np.asarray(p_starts, dtype=np.float32)),
            "p_stops_px": torch.tensor(np.asarray(p_stops, dtype=np.float32)),
            "angles": torch.tensor(np.asarray(angles, dtype=np.float32)),
        }
        data_path = os.path.join(transitions_dir, f"_{episode_id}_data.pt")
        torch.save(data, data_path)

        cfg = {
            "grid": {"height": self._grid_h, "width": self._grid_w},
            "tool": {"size_px": [int(self._tool_size_px[0]), int(self._tool_size_px[1])]},
            "physics": {"friction": None, "density": None, "box_friction": None},
            "experiment": {
                "material": self._experiment_meta.get("material", "chickpeas"),
                "surface": self._experiment_meta.get("surface", "glass"),
                "date": time.strftime("%Y-%m-%d"),
                **{k: v for k, v in self._experiment_meta.items() if k not in ("material", "surface")},
            },
        }
        config_path = os.path.join(transitions_dir, f"_{episode_id}_config.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        return data_path, config_path

    def _robot_to_grid_px(self, x: float, y: float) -> Tuple[float, float]:
        """Robot xy -> GRID pixel `(x_col, y_row)`, top-left origin, per `MD
        files/transition_dataset_design.md`'s coordinate note, via `CoordinateFrames`
        (`robot -> crop_px -> grid`)."""
        u, v = self._frames.robot_to_crop_px(x, y)
        gx, gy = self._frames.crop_px_to_grid((u, v))
        return (float(gx), float(gy))


def _mask_to_float01(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Binary uint8 {0, 255} (or already-float) mask -> float32 in [0, 1], resized to
    `(target_h, target_w)` with nearest-neighbor if it doesn't already match (binary-mask-safe,
    matching `perception.frames.CoordinateFrames.crop_to_grid`'s choice of interpolation)."""
    m = mask
    if m.shape[:2] != (target_h, target_w):
        m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    m = m.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    return m
