"""Tests for autograsper.hardware.dryrun / robot_interface (design 02 §3.1).

Never instantiates CloudGripperRobot (hard safety rule, CONVENTIONS.md) — DryRunRobot only.
"""

import numpy as np
import pytest

from autograsper.hardware.dryrun import DEFAULT_BOTTOM_SHAPE, DEFAULT_TOP_SHAPE, DryRunRobot
from autograsper.hardware.robot_interface import Order, OrderType, OrderValidationError


# --- Order.validate(): arity -------------------------------------------------


def test_move_xy_wrong_arity_raises():
    with pytest.raises(OrderValidationError):
        Order(OrderType.MOVE_XY, (0.5,)).validate()


def test_move_z_wrong_arity_raises():
    with pytest.raises(OrderValidationError):
        Order(OrderType.MOVE_Z, (0.5, 0.6)).validate()


def test_rotate_wrong_arity_raises():
    with pytest.raises(OrderValidationError):
        Order(OrderType.ROTATE, ()).validate()


def test_gripper_wrong_arity_raises():
    with pytest.raises(OrderValidationError):
        Order(OrderType.GRIPPER, (0.1, 0.2)).validate()


# --- Order.validate(): types --------------------------------------------------


def test_non_numeric_value_raises():
    with pytest.raises(OrderValidationError):
        Order(OrderType.MOVE_XY, (0.5, "oops")).validate()


def test_bool_value_rejected():
    # bool is technically an int subclass in Python; must not be silently accepted as numeric.
    with pytest.raises(OrderValidationError):
        Order(OrderType.MOVE_Z, (True,)).validate()


def test_rotation_non_int_castable_raises():
    with pytest.raises(OrderValidationError):
        Order(OrderType.ROTATE, ("abc",)).validate()


# --- Order.validate(): range clipping (xy/z/gripper) --------------------------


def test_out_of_range_clips_with_warning(caplog):
    caplog.set_level("WARNING")
    order = Order(OrderType.MOVE_XY, (1.5, -0.5)).validate()
    assert order.values == (1.0, 0.0)
    assert any("clipped" in rec.message for rec in caplog.records)


def test_z_out_of_range_clips():
    order = Order(OrderType.MOVE_Z, (2.0,)).validate()
    assert order.values == (1.0,)


def test_gripper_in_range_does_not_warn(caplog):
    caplog.set_level("WARNING")
    order = Order(OrderType.GRIPPER, (0.42,)).validate()
    assert order.values == (0.42,)
    assert not any("clipped" in rec.message for rec in caplog.records)


# --- Order.validate(): rotation normalization ---------------------------------


def test_rotation_normalizes_over_360():
    order = Order(OrderType.ROTATE, (400,)).validate()
    assert order.values == (40,)


def test_rotation_normalizes_negative():
    order = Order(OrderType.ROTATE, (-30,)).validate()
    assert order.values == (330,)


def test_rotation_int_casts_float_by_truncation():
    # Matches legacy `library/utils.py::execute_order`: `order_value[0] = int(order_value[0])`.
    order = Order(OrderType.ROTATE, (45.9,)).validate()
    assert order.values == (45,)


# --- DryRunRobot: state evolution ---------------------------------------------


def test_state_evolves_across_command_sequence():
    robot = DryRunRobot()
    robot.move_xy(0.2, 0.8)
    robot.move_z(0.5)
    robot.rotate(370)
    robot.set_gripper(0.0)

    state = robot.get_all_states().robot_state_dict
    assert state["x_norm"] == 0.2
    assert state["y_norm"] == 0.8
    assert state["z_norm"] == 0.5
    assert state["rotation"] == 10
    assert state["claw_norm"] == 0.0


def test_state_dict_has_legacy_key_set():
    robot = DryRunRobot()
    state = robot.get_all_states().robot_state_dict
    assert set(state.keys()) == {"x_norm", "y_norm", "z_norm", "rotation", "claw_norm"}


def test_initial_state_overridable():
    robot = DryRunRobot(initial_state={"x_norm": 0.1, "claw_norm": 0.3})
    state = robot.get_all_states().robot_state_dict
    assert state["x_norm"] == 0.1
    assert state["claw_norm"] == 0.3
    assert state["y_norm"] == 0.5  # untouched default


def test_out_of_range_command_clips_state():
    robot = DryRunRobot()
    robot.move_xy(5.0, -5.0)
    state = robot.get_all_states().robot_state_dict
    assert state["x_norm"] == 1.0
    assert state["y_norm"] == 0.0


# --- DryRunRobot: command_log --------------------------------------------------


def test_command_log_contents_and_order():
    robot = DryRunRobot()
    robot.move_xy(0.3, 0.4)
    robot.rotate(10)

    assert len(robot.command_log) == 2
    receipt_xy, receipt_rot = robot.command_log
    assert receipt_xy.order.type is OrderType.MOVE_XY
    assert receipt_xy.order.values == (0.3, 0.4)
    assert receipt_xy.robot_reported_time is None
    assert receipt_rot.order.type is OrderType.ROTATE
    assert receipt_rot.send_time > receipt_xy.send_time  # monotonic fake clock


def test_execute_dispatcher_uses_same_validation_path():
    robot = DryRunRobot()
    receipt = robot.execute(Order(OrderType.MOVE_XY, (1.5, 0.5)))
    assert receipt.order.values == (1.0, 0.5)  # clipped, same as calling move_xy directly
    assert robot.get_all_states().robot_state_dict["x_norm"] == 1.0


def test_execute_bad_order_raises_before_logging():
    robot = DryRunRobot()
    with pytest.raises(OrderValidationError):
        robot.execute(Order(OrderType.MOVE_XY, (0.5,)))
    assert robot.command_log == []


# --- DryRunRobot: get_all_states -----------------------------------------------


def test_get_all_states_default_shapes_and_dtype():
    robot = DryRunRobot()
    obs = robot.get_all_states()
    assert obs.top_image.shape == DEFAULT_TOP_SHAPE
    assert obs.bottom_image_raw.shape == DEFAULT_BOTTOM_SHAPE
    assert obs.top_image.dtype == np.uint8
    assert obs.bottom_image_raw.dtype == np.uint8


def test_get_all_states_configurable_shape():
    robot = DryRunRobot(top_image_shape=(64, 48, 3), bottom_image_shape=(32, 32, 3))
    obs = robot.get_all_states()
    assert obs.top_image.shape == (64, 48, 3)
    assert obs.bottom_image_raw.shape == (32, 32, 3)


def test_get_all_states_timestamps_monotonic():
    robot = DryRunRobot()
    obs1 = robot.get_all_states()
    obs2 = robot.get_all_states()
    assert obs2.timestamp > obs1.timestamp


def test_frame_replay_from_directory(tmp_path):
    import cv2

    for i in range(2):
        img = np.full((10, 12, 3), i * 10, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"frame_{i}.jpeg"), img)

    robot = DryRunRobot(frame_source_dir=str(tmp_path))
    obs = robot.get_all_states()
    assert obs.top_image.shape == (10, 12, 3)
    assert obs.bottom_image_raw.shape == (10, 12, 3)


def test_zero_latency_by_default_is_fast():
    robot = DryRunRobot()
    assert robot.latency == 0.0
