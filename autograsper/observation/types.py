"""`RobotState` and `Observation` — immutable snapshot value types (Wave 2).

Design reference: `autograsper/design/02_proposed_architecture.md` §3.2.

Porting notes (legacy sources copied/adapted, not imported):
- `autograsper/library/utils.py::manual_control` — the reference shape of the robot state dict
  read off `GripperRobot.get_state()`: keys `x_norm`, `y_norm`, `z_norm`, `rotation`, `claw_norm`.
  `RobotState.from_legacy_dict` consumes exactly this key set (also the shape of Wave 1's
  `RawObservation.robot_state_dict` / `DryRunRobot`'s simulated state, see `hardware/dryrun.py`).
- `autograsper/recording.py::Recorder._update` — applied the rotation-bias correction to
  `state["rotation"]` at this layer. In this refactor the correction moved one layer down: Wave 1's
  `CloudGripperRobot.get_all_states()` already returns bias-corrected `rotation` (hardware.md,
  "Rotation bias lives here and only here"). `RobotState.rotation` is therefore *already*
  unbiased degrees for both `CloudGripperRobot` and `DryRunRobot` (which has no bias concept) —
  this module does not re-apply any correction.

Threading: both types are `@dataclass(frozen=True)` value types with no shared mutable state.
`np.ndarray` fields cannot be made truly immutable by `frozen=True` (only *rebinding* the field
name is prevented, not in-place mutation of the array it points to). `Observation.__post_init__`
sets `arr.flags.writeable = False` on both image arrays as a best-effort guard: any attempt to
write into them after construction raises `ValueError: assignment destination is read-only`
rather than silently corrupting data another consumer may be holding a reference to. Treat both
image fields as read-only; copy (`arr.copy()`) before any in-place edit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Legacy robot_state_dict key -> RobotState field name (see module docstring).
_LEGACY_KEYS = {
    "x": "x_norm",
    "y": "y_norm",
    "z": "z_norm",
    "rotation": "rotation",
    "claw": "claw_norm",
}


@dataclass(frozen=True)
class RobotState:
    """Typed robot pose/gripper snapshot.

    - `x`, `y`: robot-normalized workspace xy, `[0, 1]`.
    - `z`: robot-normalized height, `[0, 1]`.
    - `rotation`: bias-corrected degrees (see module docstring — correction applied at the
      hardware layer, Wave 1).
    - `claw`: robot-normalized gripper opening, `[0, 1]` (0 = closed, 1 = open).

    Any field is `None` if the source dict was missing (or had a `None`/non-numeric value for)
    the corresponding legacy key — see `from_legacy_dict`. This is a deliberate tolerance
    decision (logged in `design/IMPLEMENTATION_LOG.md`): a single malformed/partial state read
    should not crash the observation poll loop, since `ObservationSource` already has its own
    consecutive-failure policy for cycles that raise outright.
    """

    x: Optional[float]
    y: Optional[float]
    z: Optional[float]
    rotation: Optional[float]
    claw: Optional[float]

    @classmethod
    def from_legacy_dict(cls, d: Optional[dict]) -> "RobotState":
        """Build a `RobotState` from a legacy-shaped state dict (or `None`).

        Tolerant: any missing key, or a value that is `None`/not numeric, becomes `None` in the
        corresponding field (never raises). Every such gap is logged once per call at WARNING so
        silent data loss is still visible in logs.
        """
        if d is None:
            logger.warning("RobotState.from_legacy_dict: state dict is None; all fields None")
            return cls(x=None, y=None, z=None, rotation=None, claw=None)

        values: dict = {}
        missing = []
        for field_name, legacy_key in _LEGACY_KEYS.items():
            raw = d.get(legacy_key)
            if raw is None or isinstance(raw, bool) or not isinstance(raw, (int, float)):
                values[field_name] = None
                if legacy_key not in d or raw is None:
                    missing.append(legacy_key)
                else:
                    missing.append(f"{legacy_key} (non-numeric: {raw!r})")
            else:
                values[field_name] = float(raw)
        if missing:
            logger.warning(
                "RobotState.from_legacy_dict: missing/invalid keys %s in state dict %r",
                missing,
                d,
            )
        return cls(**values)


@dataclass(frozen=True)
class Observation:
    """One immutable snapshot from `ObservationSource` (design 02 §3.2).

    - `seq`: monotonically increasing, source-assigned (starts at 1 for the first published
      observation of a given `ObservationSource` instance).
    - `frame_index`: recorder-assigned frame number, or `None` when nothing has registered a
      frame-index provider (`ObservationSource.set_frame_index_provider`, wired by the recorder
      in Wave 5).
    - `timestamp`: local wall-clock float (seconds), passed through from
      `hardware.RawObservation.timestamp`.
    - `top_image` / `bottom_image`: `np.ndarray` (BGR, uint8). `bottom_image` has ALREADY had the
      camera pipeline applied (fisheye undistort + homography rectify, `perception.frames`) —
      this is never the raw camera frame. Both arrays are read-only (see module docstring).
    - `robot_state`: typed pose/gripper snapshot (see `RobotState`).
    """

    seq: int
    frame_index: Optional[int]
    timestamp: float
    top_image: np.ndarray
    bottom_image: np.ndarray
    robot_state: RobotState

    def __post_init__(self) -> None:
        if not isinstance(self.top_image, np.ndarray):
            raise TypeError(f"Observation.top_image must be an ndarray, got {type(self.top_image)!r}")
        if not isinstance(self.bottom_image, np.ndarray):
            raise TypeError(
                f"Observation.bottom_image must be an ndarray, got {type(self.bottom_image)!r}"
            )
        # Best-effort read-only guard (see module docstring) — does not deep-freeze, just flips
        # the writeable flag so accidental in-place mutation raises instead of corrupting data.
        self.top_image.flags.writeable = False
        self.bottom_image.flags.writeable = False
