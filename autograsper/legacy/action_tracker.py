"""
Action Tracking Module

Provides classes and utilities for tracking and recording robot actions (movements, grasps, etc.)
along with associated frame indices and metadata. Actions are intended to support:
- Planar pushing datasets (2D motions at grasp_height)
- Task vs. reset motion differentiation
- Association of actions with specific frame ranges
"""

import threading
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of robot actions that can be tracked."""
    GRIPPER_OPEN = "gripper_open"
    GRIPPER_CLOSE = "gripper_close"
    MOVE_Z = "move_z"
    MOVE_XY = "move_xy"
    ROTATE = "rotate"
    SWEEP = "sweep"
    OTHER = "other"


class ActionPhase(Enum):
    """Whether the action is part of task execution or reset."""
    TASK = "task"
    RESET = "reset"
    STARTUP = "startup"
    OTHER = "other"


@dataclass
class Action:
    """
    Represents a single robot action with associated metadata.
    
    Attributes:
        action_id: Unique identifier for the action
        action_type: Type of action (from ActionType enum)
        phase: Whether this action is part of task or reset (from ActionPhase enum)
        start_frame: Frame index where action begins
        end_frame: Frame index where action ends (None if ongoing)
        is_planar_2d: True if action is a 2D planar motion at grasp_height
        start_robot_state: Robot state at action start
        end_robot_state: Robot state at action end
        action_details: Dictionary with action-specific details:
            - For MOVE_XY: {"x": float, "y": float}
            - For MOVE_Z: {"z": float}
            - For ROTATE: {"angle": float}
            - For SWEEP: {"wall_label": str, "direction": str, ...}
            - For GRIPPER: {"position": float}
        description: Optional human-readable description
        extra_metadata: Any additional metadata as dict
    """
    action_id: int
    action_type: ActionType
    phase: ActionPhase
    start_frame: int
    end_frame: Optional[int] = None
    is_planar_2d: bool = False
    start_robot_state: Optional[Dict[str, Any]] = None
    end_robot_state: Optional[Dict[str, Any]] = None
    action_details: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary for JSON serialization."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "phase": self.phase.value,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "is_planar_2d": self.is_planar_2d,
            "start_robot_state": self.start_robot_state,
            "end_robot_state": self.end_robot_state,
            "action_details": self.action_details,
            "description": self.description,
            "extra_metadata": self.extra_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Action':
        """Create Action from dictionary (e.g., from JSON)."""
        data = data.copy()
        # Convert string enums back to enum instances
        if isinstance(data.get("action_type"), str):
            data["action_type"] = ActionType(data["action_type"])
        if isinstance(data.get("phase"), str):
            data["phase"] = ActionPhase(data["phase"])
        return cls(**data)


class ActionTracker:
    """
    Thread-safe tracker for robot actions.
    
    Maintains a list of completed actions and tracks the currently active action.
    Provides methods to mark action start/end and retrieve action information.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.actions: List[Action] = []
        self.current_action: Optional[Action] = None
        self.next_action_id: int = 0

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
    ) -> int:
        """
        Start tracking a new action.
        
        Returns:
            action_id: ID of the started action
        """
        with self.lock:
            action_id = self.next_action_id
            self.next_action_id += 1

            self.current_action = Action(
                action_id=action_id,
                action_type=action_type,
                phase=phase,
                start_frame=start_frame,
                is_planar_2d=is_planar_2d,
                start_robot_state=start_robot_state,
                action_details=action_details or {},
                description=description,
                extra_metadata=extra_metadata or {},
            )
            logger.debug(
                f"Started action {action_id}: {action_type.value} "
                f"(frame {start_frame}, phase={phase.value})"
            )
            return action_id

    def end_action(
        self,
        action_id: int,
        end_frame: int,
        end_robot_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Action]:
        """
        End the current action.
        
        Returns:
            The completed Action if it matches action_id, otherwise None
        """
        with self.lock:
            if self.current_action is None:
                logger.warning(
                    f"Attempt to end action {action_id} but no action is active"
                )
                return None

            if self.current_action.action_id != action_id:
                logger.warning(
                    f"Action ID mismatch: expected {self.current_action.action_id}, "
                    f"got {action_id}"
                )
                return None

            self.current_action.end_frame = end_frame
            self.current_action.end_robot_state = end_robot_state
            completed_action = self.current_action
            self.actions.append(completed_action)
            self.current_action = None

            logger.debug(
                f"Ended action {action_id} ({completed_action.action_type.value}), "
                f"frames {completed_action.start_frame}-{end_frame}"
            )
            return completed_action

    def get_current_action(self) -> Optional[Action]:
        """Get the currently active action (if any)."""
        with self.lock:
            return self.current_action

    def get_action_for_frame(self, frame_index: int) -> Optional[Action]:
        """
        Get the action that encompasses the given frame.
        
        Returns:
            Action if frame is within an action's range, None otherwise
        """
        with self.lock:
            # Check current active action
            if self.current_action is not None:
                if (
                    self.current_action.start_frame <= frame_index
                    and (
                        self.current_action.end_frame is None
                        or frame_index <= self.current_action.end_frame
                    )
                ):
                    return self.current_action

            # Check completed actions
            for action in self.actions:
                if (
                    action.start_frame <= frame_index
                    and (
                        action.end_frame is None or frame_index <= action.end_frame
                    )
                ):
                    return action

            return None

    def get_all_actions(self) -> List[Action]:
        """Get all completed actions."""
        with self.lock:
            return self.actions.copy()

    def get_actions_by_type(self, action_type: ActionType) -> List[Action]:
        """Get all completed actions of a specific type."""
        with self.lock:
            return [a for a in self.actions if a.action_type == action_type]

    def get_actions_by_phase(self, phase: ActionPhase) -> List[Action]:
        """Get all completed actions from a specific phase (task/reset/etc.)."""
        with self.lock:
            return [a for a in self.actions if a.phase == phase]

    def get_planar_2d_actions(self) -> List[Action]:
        """Get all planar 2D actions (useful for planar pushing datasets)."""
        with self.lock:
            return [a for a in self.actions if a.is_planar_2d]

    def to_dict(self) -> Dict[str, Any]:
        """Convert all actions to dictionary for JSON serialization."""
        with self.lock:
            return {
                "actions": [a.to_dict() for a in self.actions],
                "current_action": (
                    self.current_action.to_dict()
                    if self.current_action is not None
                    else None
                ),
            }

    def to_json(self, filepath: str) -> None:
        """Save all actions to a JSON file."""
        with self.lock:
            data = self.to_dict()
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.actions)} actions to {filepath}")

    def clear(self) -> None:
        """Clear all tracked actions."""
        with self.lock:
            self.actions.clear()
            self.current_action = None
            self.next_action_id = 0
