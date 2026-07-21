"""`RobotInterface` protocol and the `Order`/`CommandReceipt`/`RawObservation` message types.

Design reference: `autograsper/design/02_proposed_architecture.md` §3.1.
Porting notes (legacy sources, copied and adapted — not imported):
- `autograsper/library/utils.py::OrderType` — legacy had 5 members (`MOVE_XY`, `MOVE_Z`,
  `GRIPPER_CLOSE`, `GRIPPER_OPEN`, `ROTATE`). This module collapses `GRIPPER_CLOSE`/`GRIPPER_OPEN`
  into a single `GRIPPER` order carrying an `opening` value in [0, 1] (0 = closed, 1 = open),
  matching the `RobotInterface.set_gripper(opening: float)` signature already specified by design
  02 §3.1. See `autograsper/design/IMPLEMENTATION_LOG.md` (2026-07-20 — "canonical OrderType") for
  the rationale; this is not dictated verbatim by the design doc, which only shows the
  `set_gripper` method signature, not an order enum.
- `autograsper/library/utils.py::execute_order` — value clipping (`np.clip(order_value, 0, 1)`)
  and int-cast-then-send for rotation. Ported into `Order.validate()`: xy/z/gripper values are
  clipped to [0, 1] (silently, like legacy, but with a `logging.warning` — legacy was silent),
  rotation is int-cast and normalized to [0, 360).

Threading: these are plain value types (frozen dataclasses) and pure functions; no shared state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class OrderValidationError(ValueError):
    """Raised by `Order.validate()` when an order has the wrong arity or a non-numeric /
    non-int-castable value. Range violations for xy/z/gripper are NOT errors — they are clipped
    (with a logged warning), matching legacy `execute_order` behavior.
    """


class OrderType(Enum):
    """Canonical set of robot commands (design 02 §3.1). One member per `RobotInterface` method."""

    MOVE_XY = "MOVE_XY"
    MOVE_Z = "MOVE_Z"
    ROTATE = "ROTATE"
    GRIPPER = "GRIPPER"


_ARITY = {
    OrderType.MOVE_XY: 2,
    OrderType.MOVE_Z: 1,
    OrderType.ROTATE: 1,
    OrderType.GRIPPER: 1,
}

_UNIT_RANGE_TYPES = (OrderType.MOVE_XY, OrderType.MOVE_Z, OrderType.GRIPPER)


@dataclass(frozen=True)
class Order:
    """A single robot command: `type` + positional `values`.

    - `MOVE_XY`: `(x, y)`, robot-normalized [0, 1].
    - `MOVE_Z`: `(z,)`, robot-normalized [0, 1].
    - `ROTATE`: `(angle_deg,)`, int-castable degrees.
    - `GRIPPER`: `(opening,)`, robot-normalized [0, 1] (0 = closed, 1 = open).
    """

    type: OrderType
    values: Tuple[float, ...]

    def validate(self) -> "Order":
        """Return a normalized/clipped copy of this order, or raise `OrderValidationError`.

        Raises on: wrong number of values, non-numeric values, a ROTATE value that cannot be
        cast to `int`. Does NOT raise on out-of-range xy/z/gripper values — those are clipped to
        [0, 1] and a warning is logged (legacy `execute_order` clipped silently via `np.clip`;
        the refactor keeps the clip but adds visibility). Idempotent: validating an already-valid
        order returns it unchanged (no duplicate warnings).
        """
        expected_arity = _ARITY[self.type]
        if len(self.values) != expected_arity:
            raise OrderValidationError(
                f"{self.type.name} expects {expected_arity} value(s), got {len(self.values)}: "
                f"{self.values!r}"
            )
        for v in self.values:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise OrderValidationError(
                    f"{self.type.name} values must be numeric, got {v!r} in {self.values!r}"
                )

        if self.type == OrderType.ROTATE:
            raw = self.values[0]
            try:
                angle = int(raw)
            except (TypeError, ValueError) as exc:
                raise OrderValidationError(
                    f"ROTATE value must be int-castable, got {raw!r}"
                ) from exc
            normalized = angle % 360
            return Order(self.type, (normalized,))

        assert self.type in _UNIT_RANGE_TYPES
        original = tuple(float(v) for v in self.values)
        clipped = tuple(min(max(v, 0.0), 1.0) for v in original)
        if clipped != original:
            logger.warning(
                "Order %s values %r clipped to %r (out of [0, 1] range)",
                self.type.name,
                self.values,
                clipped,
            )
        return Order(self.type, clipped)


@dataclass(frozen=True)
class CommandReceipt:
    """Result of sending one `Order`.

    - `order`: the (validated) order that was sent.
    - `send_time`: local clock reading (float, seconds) at send time.
    - `robot_reported_time`: the raw `"time"` field the CloudGripper API returns for the command
      (e.g. `GripperRobot.move_xy(...)`), pass-through and un-parsed; `None` for `DryRunRobot` or
      if the API omitted it.
    """

    order: Order
    send_time: float
    robot_reported_time: Optional[str]


@dataclass(frozen=True)
class RawObservation:
    """One `get_all_states()` snapshot, un-processed (no undistortion/homography — that pipeline
    lives in `observation/source.py`, Wave 2). This is the hardware layer's raw truth.

    - `top_image` / `bottom_image_raw`: `np.ndarray` (BGR, uint8) or `None` if unavailable.
    - `robot_state_dict`: legacy-format dict with keys `x_norm`, `y_norm`, `z_norm`, `rotation`,
      `claw_norm` (see `autograsper/library/utils.py::manual_control` for the reference key set).
      For `CloudGripperRobot` the `rotation` value has already had the rotation-bias correction
      applied (see `cloudgripper.py` docstring) — every layer above the hardware layer operates in
      unbiased degrees.
    - `timestamp`: local wall-clock float (seconds) at the moment this snapshot was assembled.
      (Not the API's own `time_state` string — that raw value is not currently surfaced on this
      type; see IMPLEMENTATION_LOG for rationale.)
    """

    top_image: Optional[np.ndarray]
    bottom_image_raw: Optional[np.ndarray]
    robot_state_dict: Optional[dict]
    timestamp: float


class RobotInterface(Protocol):
    """Structural interface every hardware backend implements (design 02 §3.1)."""

    def move_xy(self, x: float, y: float) -> CommandReceipt: ...

    def move_z(self, z: float) -> CommandReceipt: ...

    def rotate(self, angle_deg: int) -> CommandReceipt: ...

    def set_gripper(self, opening: float) -> CommandReceipt: ...

    def get_all_states(self) -> RawObservation: ...


class OrderDispatchMixin:
    """Shared `execute(order) -> CommandReceipt` dispatcher.

    Mixed into `CloudGripperRobot` and `DryRunRobot` so both get the same
    validate-then-dispatch convenience method without duplicating the if/elif ladder. Requires
    the concrete class to provide `move_xy`/`move_z`/`rotate`/`set_gripper` (i.e. satisfy
    `RobotInterface`).
    """

    def execute(self, order: Order) -> CommandReceipt:
        validated = order.validate()
        if validated.type == OrderType.MOVE_XY:
            return self.move_xy(validated.values[0], validated.values[1])  # type: ignore[attr-defined]
        if validated.type == OrderType.MOVE_Z:
            return self.move_z(validated.values[0])  # type: ignore[attr-defined]
        if validated.type == OrderType.ROTATE:
            return self.rotate(int(validated.values[0]))  # type: ignore[attr-defined]
        if validated.type == OrderType.GRIPPER:
            return self.set_gripper(validated.values[0])  # type: ignore[attr-defined]
        raise OrderValidationError(f"Unknown order type: {validated.type!r}")
