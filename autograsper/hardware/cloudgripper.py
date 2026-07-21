"""`CloudGripperRobot` — real `RobotInterface` implementation wrapping
`client.cloudgripper_client.GripperRobot`.

Design reference: `autograsper/design/02_proposed_architecture.md` §3.1.
Porting notes (copied and adapted, not imported):
- `autograsper/recording.py` lines ~30-45 (config reads: `camera.m/d/H`, `robot_idx`,
  `rotation_bias`, `CLOUDGRIPPER_TOKEN` env var) and lines ~186-188 (the rotation-bias
  correction: `state['rotation'] -= angle_bias; state['rotation'] %= 180`).
- `autograsper/grasper.py` lines ~302-306 (`execute_order`: `local_order[1][0] += self.rotation_bias`
  applied to the OUTGOING rotate value, before `library/utils.py::execute_order` int-casts it).
- `autograsper/library/utils.py::execute_order` (the `GRIPPER_CLOSE` empty-value branch): the
  staged gripper-close sequence (0.30 -> 0.24 in -0.04 steps, 0.1s wait between steps).

**Safety rule** (see `autograsper/docs/CONVENTIONS.md`): this class performs real HTTP calls
against a physical robot. It MUST NOT be instantiated by any test, example, or default config.
`DryRunRobot` is the only backend used in tests.

Rotation bias — single site (fixes design 01 §5 defect #16, "dual-site rotation bias"): legacy
applied `+bias` in `grasper.py::execute_order` and `-bias` in `recording.py::Recorder._update`,
two different files that had to be kept in sync. Here both corrections live in this one class:
`rotate()` adds `+bias` (then int-casts) before sending to hardware; `get_all_states()` subtracts
`bias` from the reported rotation and takes it mod 180 (exact legacy formula). Every layer above
`CloudGripperRobot` sees/sends unbiased degrees.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from client.cloudgripper_client import GripperRobot

from autograsper.config_schema import RobotConfig
from autograsper.hardware.robot_interface import (
    CommandReceipt,
    Order,
    OrderDispatchMixin,
    OrderType,
    RawObservation,
)

logger = logging.getLogger(__name__)

# Legacy staged gripper-close sequence, ported verbatim from
# `autograsper/library/utils.py::execute_order` (OrderType.GRIPPER_CLOSE, empty-value branch).
_STEP_CLOSE_START = 0.3
_STEP_CLOSE_END = 0.24
_STEP_CLOSE_STEP = 0.04
_STEP_CLOSE_WAIT_S = 0.1


class RobotTokenError(RuntimeError):
    """Raised at construction time when the configured token env var is unset/empty."""


class CloudGripperRobot(OrderDispatchMixin):
    """Real `RobotInterface` backend. One instance is meant to be shared (design 02 §3.1: "the
    HTTP client is stateless, so sharing is safe; the point is one place for token/robot-idx/
    retry logic") by the executor and the observation source in later waves.

    Threading: not internally synchronized. The underlying `GripperRobot`/`requests` calls are
    blocking HTTP; callers needing concurrent access should serialize their own calls (Wave 2's
    `ObservationSource` and `execution/executor.py` are expected to coordinate this).
    """

    def __init__(self, config: RobotConfig, *, client: Optional[GripperRobot] = None):
        token = os.environ.get(config.token_env_var)
        if not token:
            raise RobotTokenError(
                f"Environment variable {config.token_env_var!r} is not set (required for "
                f"robot {config.idx!r})"
            )
        self._robot_idx = config.idx
        self._rotation_bias = config.rotation_bias
        self._client = client if client is not None else GripperRobot(config.idx, token)

    # -- RobotInterface -----------------------------------------------------

    def move_xy(self, x: float, y: float) -> CommandReceipt:
        order = Order(OrderType.MOVE_XY, (x, y)).validate()
        send_time = time.time()
        reported = self._client.move_xy(order.values[0], order.values[1])
        return CommandReceipt(order=order, send_time=send_time, robot_reported_time=reported)

    def move_z(self, z: float) -> CommandReceipt:
        order = Order(OrderType.MOVE_Z, (z,)).validate()
        send_time = time.time()
        reported = self._client.move_z(order.values[0])
        return CommandReceipt(order=order, send_time=send_time, robot_reported_time=reported)

    def rotate(self, angle_deg: int) -> CommandReceipt:
        """Send a ROTATE command. `angle_deg` is the UNBIASED angle (as validated/normalized by
        `Order.validate()`, range [0, 360)); the rotation bias is added here, then the biased
        value is int-cast and sent — exactly mirroring legacy
        `grasper.py::execute_order` (`+= self.rotation_bias`) followed by
        `library/utils.py::execute_order` (`int(order_value[0])`). No modulo is applied to the
        outgoing biased value (legacy did not apply one either).
        """
        order = Order(OrderType.ROTATE, (angle_deg,)).validate()
        biased = int(order.values[0] + self._rotation_bias)
        send_time = time.time()
        reported = self._client.rotate(biased)
        return CommandReceipt(order=order, send_time=send_time, robot_reported_time=reported)

    def set_gripper(self, opening: float, *, stepped_close: bool = False) -> CommandReceipt:
        """Set the gripper opening (0 = closed, 1 = open).

        `stepped_close=True` reproduces the legacy "cautious close" sequence from
        `library/utils.py::execute_order` (`OrderType.GRIPPER_CLOSE` with no explicit value):
        step the opening down from 0.30 to 0.24 in -0.04 increments with a 0.1s wait between
        steps, regardless of the requested `opening` value (legacy behavior — the sequence
        target was hardcoded, not parameterized). The returned receipt reflects the *final*
        step's send time/reported time.
        """
        order = Order(OrderType.GRIPPER, (opening,)).validate()
        if stepped_close:
            send_time = time.time()
            reported = self._client.move_gripper(_STEP_CLOSE_START)
            current = _STEP_CLOSE_START
            while current >= _STEP_CLOSE_END:
                reported = self._client.move_gripper(current)
                time.sleep(_STEP_CLOSE_WAIT_S)
                current -= _STEP_CLOSE_STEP
            return CommandReceipt(order=order, send_time=send_time, robot_reported_time=reported)

        send_time = time.time()
        reported = self._client.move_gripper(order.values[0])
        return CommandReceipt(order=order, send_time=send_time, robot_reported_time=reported)

    def get_all_states(self) -> RawObservation:
        """Fetch images + robot state in one HTTP round trip.

        Applies the rotation-bias correction to `state['rotation']` (subtract bias, mod 180 —
        ported verbatim from `recording.py::Recorder._update`). Does NOT undistort/rectify the
        bottom image — that is the observation layer's job (Wave 2, `perception/frames.py`).
        """
        image_top, image_base, state, _time_state = self._client.get_all_states()
        if isinstance(state, dict) and "rotation" in state:
            state = dict(state)
            state["rotation"] = (state["rotation"] - self._rotation_bias) % 180
        return RawObservation(
            top_image=image_top,
            bottom_image_raw=image_base,
            robot_state_dict=state,
            timestamp=time.time(),
        )
