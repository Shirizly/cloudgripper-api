"""`ActionType`/`ActionPhase`/`Action`/`ActionTracker` — robot action tracking (Wave 4).

Design reference: `autograsper/design/02_proposed_architecture.md` §3.4 ("Brackets every primitive
with `ActionTracker` start/end ... the executor is the only writer of actions") and §3.6 migration
map ("`action_tracker.py` | kept; gains composite (parent/child) actions").

Porting notes (source: `autograsper/action_tracker.py`, copied and adapted per CONVENTIONS.md —
not imported): `ActionType`, `ActionPhase`, `Action.to_dict`/`.from_dict`, and every
`ActionTracker` method (`start_action`/`end_action`/`get_action_for_frame`/`get_all_actions`/
`get_actions_by_type`/`get_actions_by_phase`/`get_planar_2d_actions`/`to_dict`/`to_json`/`clear`)
carry over with their original semantics and JSON key names unchanged (dataset compatibility —
existing `actions.json` consumers keep working). Extensions made in this wave, all additive:

- `Action.parent_id: Optional[int]` — `None` for a top-level (primitive-level) action, or the
  `action_id` of the in-flight top-level action a child (per-order) action belongs to. Included in
  `to_dict()`/`from_dict()`; a missing `parent_id` key in old JSON defaults to `None` via the
  dataclass field default, so previously recorded `actions.json` files still load.
- `ActionTracker` now tracks **two** simultaneously in-flight actions instead of one: the top-level
  `current_action` (a `Plan` primitive, per `execution.executor.Executor.execute_primitive`) and
  `current_child_action` (one expanded `RobotInterface` order within it, per
  `Executor._send_order`). `start_action(..., parent_id=None)` opens a top-level action;
  `start_action(..., parent_id=<id>)` opens a child. `end_action(action_id, ...)` matches whichever
  of the two is currently open with that id.
- `get_action_for_frame` now prefers the deepest (child) match over a top-level match when a frame
  index falls within both ranges (a child action's frame range is always nested inside its
  parent's) — checked in order: in-flight child, in-flight top-level, then completed actions
  (child match preferred over top-level match there too).
- `register_completion_callback(fn)` — a registry of `Callable[[Action], None]` invoked once per
  completed action (child or top-level), each call made **outside** the tracker's lock (after the
  completed action has already been appended to `self.actions` and both in-flight slots updated),
  so a slow or reentrant callback (e.g. Wave 5's `TransitionWriter` subscribing to completed `Push`
  actions per design 03 §5) can never deadlock against a concurrent `start_action`/`end_action`
  call. A callback that raises is logged and does not affect any other callback or the tracker's
  own state.

Threading: same model as legacy — one `threading.RLock` guards all mutable state
(`current_action`, `current_child_action`, `actions`, `next_action_id`, the callback list);
`Executor` is the only writer (design 02 §3.4), but reads (`get_action_for_frame`, etc.) are safe
from any thread.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of robot actions that can be tracked. Ported verbatim from `action_tracker.py`."""

    GRIPPER_OPEN = "gripper_open"
    GRIPPER_CLOSE = "gripper_close"
    MOVE_Z = "move_z"
    MOVE_XY = "move_xy"
    ROTATE = "rotate"
    SWEEP = "sweep"
    OTHER = "other"


class ActionPhase(Enum):
    """Whether the action is part of task execution, reset, startup, or other. Ported verbatim
    from `action_tracker.py`."""

    TASK = "task"
    RESET = "reset"
    STARTUP = "startup"
    OTHER = "other"


@dataclass
class Action:
    """A single tracked robot action (top-level primitive, or a child order within one).

    Attributes:
        action_id: Unique identifier for the action.
        action_type: Type of action (from `ActionType`).
        phase: Whether this action is part of task/reset/startup/other (from `ActionPhase`).
        start_frame: Frame index where the action begins.
        end_frame: Frame index where the action ends (`None` if still open).
        is_planar_2d: `True` if this action is a 2D planar motion at grasp height (dataset
            transition candidate).
        parent_id: `None` for a top-level (primitive) action; the top-level action's `action_id`
            for a child (per-order) action. New in this wave (see module docstring).
        start_robot_state / end_robot_state: robot state snapshots (legacy-shaped dict, see
            `observation.types.RobotState`) at the action's start/end.
        action_details: action-specific fields (e.g. `{"x":..., "y":...}` for `MOVE_XY`, or a
            primitive's own dataclass fields for a top-level action — see
            `execution.executor.Executor.execute_primitive`).
        description: optional human-readable summary (`Primitive.describe()` for top-level
            actions).
        extra_metadata: free-form additional metadata.
    """

    action_id: int
    action_type: ActionType
    phase: ActionPhase
    start_frame: int
    end_frame: Optional[int] = None
    is_planar_2d: bool = False
    parent_id: Optional[int] = None
    start_robot_state: Optional[Dict[str, Any]] = None
    end_robot_state: Optional[Dict[str, Any]] = None
    action_details: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary for JSON serialization. Legacy keys unchanged; `parent_id`
        added (new in this wave)."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "phase": self.phase.value,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "is_planar_2d": self.is_planar_2d,
            "parent_id": self.parent_id,
            "start_robot_state": self.start_robot_state,
            "end_robot_state": self.end_robot_state,
            "action_details": self.action_details,
            "description": self.description,
            "extra_metadata": self.extra_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Create an `Action` from a dictionary (e.g., loaded from JSON). Tolerant of a missing
        `parent_id` key (older recordings predating this wave) — defaults to `None`."""
        data = dict(data)
        if isinstance(data.get("action_type"), str):
            data["action_type"] = ActionType(data["action_type"])
        if isinstance(data.get("phase"), str):
            data["phase"] = ActionPhase(data["phase"])
        return cls(**data)


def _frame_in_range(action: Action, frame_index: int) -> bool:
    return action.start_frame <= frame_index and (
        action.end_frame is None or frame_index <= action.end_frame
    )


class ActionTracker:
    """Thread-safe tracker for robot actions, supporting one in-flight top-level (primitive)
    action and one in-flight child (per-order) action simultaneously. See module docstring."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.actions: List[Action] = []
        self.current_action: Optional[Action] = None
        self.current_child_action: Optional[Action] = None
        self.next_action_id: int = 0
        self._completion_callbacks: List[Callable[[Action], None]] = []

    # -- completion callbacks ---------------------------------------------------------

    def register_completion_callback(self, fn: Callable[[Action], None]) -> None:
        """Register `fn` to be called with each completed `Action` (child or top-level), once per
        `end_action()` call, always outside the tracker's lock (see module docstring)."""
        with self.lock:
            self._completion_callbacks.append(fn)

    # -- start/end ----------------------------------------------------------------------

    def start_action(
        self,
        action_type: ActionType,
        phase: ActionPhase,
        start_frame: int,
        start_robot_state: Optional[Dict[str, Any]] = None,
        is_planar_2d: bool = False,
        action_details: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[int] = None,
    ) -> int:
        """Start tracking a new action. `parent_id=None` (the default) opens a top-level action;
        passing the currently open top-level action's id opens a child action alongside it.

        Returns the new action's `action_id`.
        """
        with self.lock:
            action_id = self.next_action_id
            self.next_action_id += 1

            action = Action(
                action_id=action_id,
                action_type=action_type,
                phase=phase,
                start_frame=start_frame,
                is_planar_2d=is_planar_2d,
                parent_id=parent_id,
                start_robot_state=start_robot_state,
                action_details=action_details or {},
                description=description,
                extra_metadata=extra_metadata or {},
            )

            if parent_id is None:
                if self.current_action is not None:
                    logger.warning(
                        "ActionTracker: starting top-level action %d while %d is still open "
                        "(overwriting reference; the previous action will never be end_action()ed)",
                        action_id,
                        self.current_action.action_id,
                    )
                self.current_action = action
            else:
                if self.current_child_action is not None:
                    logger.warning(
                        "ActionTracker: starting child action %d while %d is still open "
                        "(overwriting reference; the previous action will never be end_action()ed)",
                        action_id,
                        self.current_child_action.action_id,
                    )
                self.current_child_action = action

            logger.debug(
                "Started action %d: %s (frame %d, phase=%s, parent_id=%s)",
                action_id,
                action_type.value,
                start_frame,
                phase.value,
                parent_id,
            )
            return action_id

    def end_action(
        self,
        action_id: int,
        end_frame: int,
        end_robot_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Action]:
        """End the action matching `action_id` (checked against the in-flight child first, then
        the in-flight top-level action). Returns the completed `Action`, or `None` if `action_id`
        matches neither (logged as a warning, matching legacy behavior)."""
        completed: Optional[Action] = None
        callbacks: List[Callable[[Action], None]] = []
        with self.lock:
            if self.current_child_action is not None and self.current_child_action.action_id == action_id:
                self.current_child_action.end_frame = end_frame
                self.current_child_action.end_robot_state = end_robot_state
                completed = self.current_child_action
                self.current_child_action = None
            elif self.current_action is not None and self.current_action.action_id == action_id:
                self.current_action.end_frame = end_frame
                self.current_action.end_robot_state = end_robot_state
                completed = self.current_action
                self.current_action = None
            else:
                logger.warning(
                    "Attempt to end action %d but it matches neither the current top-level "
                    "action (%s) nor the current child action (%s)",
                    action_id,
                    self.current_action.action_id if self.current_action else None,
                    self.current_child_action.action_id if self.current_child_action else None,
                )
                return None

            self.actions.append(completed)
            callbacks = list(self._completion_callbacks)
            logger.debug(
                "Ended action %d (%s), frames %d-%s",
                action_id,
                completed.action_type.value,
                completed.start_frame,
                end_frame,
            )

        for cb in callbacks:
            try:
                cb(completed)
            except Exception:
                logger.exception("ActionTracker: completion callback raised for action %d", action_id)

        return completed

    # -- reads ------------------------------------------------------------------------------

    def get_current_action(self) -> Optional[Action]:
        """Get the currently active top-level action (if any)."""
        with self.lock:
            return self.current_action

    def get_current_child_action(self) -> Optional[Action]:
        """Get the currently active child action (if any). New in this wave."""
        with self.lock:
            return self.current_child_action

    def get_action_for_frame(self, frame_index: int) -> Optional[Action]:
        """Get the (deepest, i.e. child-preferred) action that encompasses `frame_index`.

        Checked in order: in-flight child action, in-flight top-level action, then completed
        actions (a completed child match is preferred over a completed top-level match, since a
        child's frame range nests inside its parent's — see module docstring)."""
        with self.lock:
            if self.current_child_action is not None and _frame_in_range(
                self.current_child_action, frame_index
            ):
                return self.current_child_action
            if self.current_action is not None and _frame_in_range(self.current_action, frame_index):
                return self.current_action

            matches = [a for a in self.actions if _frame_in_range(a, frame_index)]
            if not matches:
                return None
            children = [a for a in matches if a.parent_id is not None]
            return children[0] if children else matches[0]

    def get_all_actions(self) -> List[Action]:
        """Get all completed actions (top-level and child, in completion order)."""
        with self.lock:
            return self.actions.copy()

    def get_actions_by_type(self, action_type: ActionType) -> List[Action]:
        with self.lock:
            return [a for a in self.actions if a.action_type == action_type]

    def get_actions_by_phase(self, phase: ActionPhase) -> List[Action]:
        with self.lock:
            return [a for a in self.actions if a.phase == phase]

    def get_planar_2d_actions(self) -> List[Action]:
        """Get all planar 2D actions (useful for planar pushing datasets)."""
        with self.lock:
            return [a for a in self.actions if a.is_planar_2d]

    def to_dict(self) -> Dict[str, Any]:
        """Convert all actions (plus in-flight state) to a dict for JSON serialization."""
        with self.lock:
            return {
                "actions": [a.to_dict() for a in self.actions],
                "current_action": (
                    self.current_action.to_dict() if self.current_action is not None else None
                ),
                "current_child_action": (
                    self.current_child_action.to_dict()
                    if self.current_child_action is not None
                    else None
                ),
            }

    def to_json(self, filepath: str) -> None:
        """Save all actions to a JSON file."""
        with self.lock:
            data = self.to_dict()
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("Saved %d actions to %s", len(self.actions), filepath)

    def clear(self) -> None:
        """Clear all tracked actions and in-flight state (completion callbacks are kept)."""
        with self.lock:
            self.actions.clear()
            self.current_action = None
            self.current_child_action = None
            self.next_action_id = 0
