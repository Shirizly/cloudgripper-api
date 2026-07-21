"""`DryRunRobot` — hardware-free `RobotInterface` implementation for tests and offline dev.

Design reference: `autograsper/design/02_proposed_architecture.md` §3.1 ("`DryRunRobot` validates
types and ranges ..., logs each command, advances a simulated state ..., and can replay a
directory of recorded frames as its camera. This is the permanent home for robot-free testing.").

Per `autograsper/docs/CONVENTIONS.md`'s hard safety rule, this is the ONLY backend allowed in
tests, examples, and default configs — never `CloudGripperRobot`.

State-dict keys (`x_norm`, `y_norm`, `z_norm`, `rotation`, `claw_norm`) match the legacy format
read in `autograsper/library/utils.py::manual_control` (`state["x_norm"]`, etc.), so downstream
code written against the legacy dict shape keeps working against `DryRunRobot.get_all_states()
.robot_state_dict`. `rotation` here is always the unbiased value (`DryRunRobot` has no rotation
bias concept — bias is a hardware-layer-only concern, see `cloudgripper.py`).

Threading: not synchronized; intended for single-threaded test/dev use. `command_log` is a plain
list appended to in call order.
"""

from __future__ import annotations

import glob
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from autograsper.hardware.robot_interface import (
    CommandReceipt,
    Order,
    OrderDispatchMixin,
    OrderType,
    RawObservation,
)

logger = logging.getLogger(__name__)

DEFAULT_TOP_SHAPE: Tuple[int, int, int] = (480, 640, 3)
DEFAULT_BOTTOM_SHAPE: Tuple[int, int, int] = (480, 640, 3)

_DEFAULT_STATE = {
    "x_norm": 0.5,
    "y_norm": 0.5,
    "z_norm": 1.0,
    "rotation": 0,
    "claw_norm": 1.0,
}


@dataclass
class _FrameReplay:
    """Cycles through image files in a directory, sorted by filename."""

    paths: List[str]
    index: int = 0

    @classmethod
    def from_dir(cls, directory: str) -> "_FrameReplay":
        patterns = ("*.jpeg", "*.jpg", "*.png")
        paths: List[str] = []
        for pattern in patterns:
            paths.extend(glob.glob(os.path.join(directory, pattern)))
        paths.sort()
        if not paths:
            raise ValueError(f"No image files found in frame_source_dir={directory!r}")
        return cls(paths=paths)

    def next_image(self) -> np.ndarray:
        path = self.paths[self.index % len(self.paths)]
        self.index += 1
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to read replay frame: {path}")
        return image


class DryRunRobot(OrderDispatchMixin):
    """Simulated `RobotInterface`. Validates every order via `Order.validate()` (same path as
    `CloudGripperRobot`), logs it, and updates an in-memory state dict assuming immediate,
    unconditional success (`assume_success=True` semantics — there is no notion of a command
    failing or taking real time unless `latency` is set).
    """

    def __init__(
        self,
        *,
        robot_idx: str = "dryrun",
        initial_state: Optional[dict] = None,
        top_image_shape: Tuple[int, int, int] = DEFAULT_TOP_SHAPE,
        bottom_image_shape: Tuple[int, int, int] = DEFAULT_BOTTOM_SHAPE,
        frame_source_dir: Optional[str] = None,
        latency: float = 0.0,
        assume_success: bool = True,
    ):
        self.robot_idx = robot_idx
        self._state = dict(_DEFAULT_STATE)
        if initial_state:
            self._state.update(initial_state)
        self._top_image_shape = top_image_shape
        self._bottom_image_shape = bottom_image_shape
        self._replay = _FrameReplay.from_dir(frame_source_dir) if frame_source_dir else None
        self.latency = latency
        self.assume_success = assume_success

        self.command_log: List[CommandReceipt] = []
        # Monotonic fake clock: deterministic, independent of wall-clock scheduling jitter, so
        # tests can assert strict ordering without real sleeps. Advances by a fixed step per
        # command (see `_tick`), plus `latency` if configured.
        self._fake_clock = 0.0
        self._clock_step = 0.1

    # -- internal helpers -----------------------------------------------------

    def _tick(self) -> float:
        """Advance and return the fake monotonic clock, applying configured latency."""
        if self.latency:
            time.sleep(self.latency)
        self._fake_clock += self._clock_step + self.latency
        return self._fake_clock

    def _log_and_record(self, order: Order) -> CommandReceipt:
        send_time = self._tick()
        logger.info("DryRunRobot: %s %r (t=%.3f)", order.type.name, order.values, send_time)
        receipt = CommandReceipt(order=order, send_time=send_time, robot_reported_time=None)
        self.command_log.append(receipt)
        return receipt

    # -- RobotInterface ---------------------------------------------------

    def move_xy(self, x: float, y: float) -> CommandReceipt:
        order = Order(OrderType.MOVE_XY, (x, y)).validate()
        if self.assume_success:
            self._state["x_norm"], self._state["y_norm"] = order.values
        return self._log_and_record(order)

    def move_z(self, z: float) -> CommandReceipt:
        order = Order(OrderType.MOVE_Z, (z,)).validate()
        if self.assume_success:
            self._state["z_norm"] = order.values[0]
        return self._log_and_record(order)

    def rotate(self, angle_deg: int) -> CommandReceipt:
        order = Order(OrderType.ROTATE, (angle_deg,)).validate()
        if self.assume_success:
            self._state["rotation"] = order.values[0]
        return self._log_and_record(order)

    def set_gripper(self, opening: float) -> CommandReceipt:
        order = Order(OrderType.GRIPPER, (opening,)).validate()
        if self.assume_success:
            self._state["claw_norm"] = order.values[0]
        return self._log_and_record(order)

    def get_all_states(self) -> RawObservation:
        timestamp = self._tick()
        if self._replay is not None:
            top_image = self._replay.next_image()
            bottom_image = self._replay.next_image()
        else:
            top_image = np.zeros(self._top_image_shape, dtype=np.uint8)
            bottom_image = np.zeros(self._bottom_image_shape, dtype=np.uint8)
        return RawObservation(
            top_image=top_image,
            bottom_image_raw=bottom_image,
            robot_state_dict=dict(self._state),
            timestamp=timestamp,
        )
