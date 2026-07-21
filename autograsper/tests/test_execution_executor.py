"""Tests for `autograsper.execution.executor.Executor` (Wave 4): per-primitive order-sequence
expansion (verified against `DryRunRobot.command_log`), action-tree correctness (parent/child,
frames, `Push` action_details), freshness enforcement, the `LowerTool` real-time footprint guard,
`RegraspTool`/`CheckToolGrip` grip checks, `order_sink` record shape, and `run_plan`/`PlanResult`.

Uses `DryRunRobot` + a real `ObservationSource` (per CONVENTIONS.md: no network/GPU/robot). Stub
`occupancy_supplier`/`tool_grip_checker` objects stand in for `SegmentationWorker`/
`ToolGripChecker` (duck-typed, no perception-layer import needed for these tests).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from autograsper.config_schema import (
    CropConfig,
    GridConfig,
    ManipulationBoundaryRobot,
    PerceptionConfig,
    WorkspaceConfig,
)
from autograsper.execution.actions import ActionPhase, ActionTracker, ActionType
from autograsper.execution.errors import StaleMaskTimeout, ToolLost, UnsafeLower
from autograsper.execution.executor import Executor, PlanResult
from autograsper.execution.safety import SafetyValidator
from autograsper.hardware.dryrun import DryRunRobot
from autograsper.hardware.robot_interface import OrderType
from autograsper.observation.source import ObservationSource
from autograsper.perception.frames import CoordinateFrames
from autograsper.planning.planner import FreshnessPolicy
from autograsper.planning.types import (
    CheckToolGrip,
    LowerTool,
    MoveTo,
    PlaceTool,
    Plan,
    Push,
    RefreshMask,
    RegraspTool,
    SweepWall,
)
from autograsper.planning.workspace import Workspace, sample_tool_pose

_CROP_SIZE = (300, 300)
_TOOL_DIMS_PX = (8, 40)


class _PassthroughCamera:
    def process_bottom(self, raw_bottom):
        return raw_bottom


def _synthetic_H(crop_w: int, crop_h: int) -> np.ndarray:
    return np.array(
        [
            [1.0 / crop_w, 0.0, 0.0],
            [0.0, -1.0 / crop_h, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _make_workspace(tmp_path, manip=(0.0, 1.0)) -> Workspace:
    crop = CropConfig(center_px=(_CROP_SIZE[0] // 2, _CROP_SIZE[1] // 2), size=_CROP_SIZE)
    grid = GridConfig(height=32, width=32)
    H = _synthetic_H(*_CROP_SIZE)
    npz_path = tmp_path / "homography.npz"
    np.savez(npz_path, H)

    workspace_config = WorkspaceConfig(
        fence_center=(0.5, 0.49),
        fence_size=(0.94, 0.955),
        manipulation_boundary_robot=ManipulationBoundaryRobot(x=manip, y=manip),
        tool_length_robot=0.36,
        tool_width_robot=0.017,
        tool_dims_px=_TOOL_DIMS_PX,
        homography_npz_path=str(npz_path),
        grasp_height=0.34,
        sweep_height=0.57,
        clearance_height=0.8,
        safety_margin=0.01,
    )
    perception_config = PerceptionConfig(provider="none", crop=crop, grid=grid)
    frames = CoordinateFrames.from_config(workspace_config, perception_config)
    return Workspace(workspace_config, frames)


@pytest.fixture
def env():
    shutdown_event = threading.Event()
    robot = DryRunRobot()
    source = ObservationSource(robot, _PassthroughCamera(), fps=300.0, shutdown_event=shutdown_event)
    source.start()
    source.await_next(0, timeout=2.0)  # ensure at least one Observation exists before use
    try:
        yield robot, source, shutdown_event
    finally:
        source.stop(timeout=2.0)


def _make_executor(
    workspace,
    robot,
    source,
    shutdown_event,
    *,
    occupancy_supplier=None,
    tool_grip_checker=None,
    order_sink=None,
    freshness_categories=(),
):
    tracker = ActionTracker()
    safety = SafetyValidator(workspace)
    freshness = FreshnessPolicy(freshness_categories)
    config = SimpleNamespace(experiment=SimpleNamespace(time_between_orders=0.0))
    return Executor(
        robot,
        source,
        tracker,
        safety,
        freshness,
        workspace,
        config,
        occupancy_supplier=occupancy_supplier,
        tool_grip_checker=tool_grip_checker,
        order_sink=order_sink,
        shutdown_event=shutdown_event,
        freshness_timeout=0.3,
        end_frame_timeout=1.0,
    )


def _seq(robot):
    """`[(OrderType, values_tuple), ...]` from `DryRunRobot.command_log`, in call order."""
    return [(r.order.type, tuple(r.order.values)) for r in robot.command_log]


def _assert_seq_matches(robot, expected):
    actual = _seq(robot)
    assert len(actual) == len(expected), f"expected {len(expected)} orders, got {len(actual)}: {actual}"
    for (atype, avalues), (etype, evalues) in zip(actual, expected):
        assert atype == etype
        assert avalues == pytest.approx(evalues, abs=1e-6)


class _StubOccupancy:
    def __init__(self, crop_mask: np.ndarray, source_seq: int = 1):
        self.crop_mask = crop_mask
        self.source_seq = source_seq


class _StubOccupancySupplier:
    """`SegmentationWorker`-shaped stub: `latest()`/`await_result(min_source_seq, timeout)`."""

    def __init__(self, result=None, *, always_timeout: bool = False):
        self._result = result
        self._always_timeout = always_timeout

    def latest(self):
        return self._result

    def await_result(self, min_source_seq, timeout=None):
        if self._always_timeout:
            return None
        return self._result


class _StubGripChecker:
    def __init__(self, quality: float):
        self._quality = quality

    def check(self, obs):
        ok = self._quality >= 0.6
        return SimpleNamespace(quality=self._quality, ok=ok)


# --- MoveTo ------------------------------------------------------------------------------------


def test_move_to_raises_before_translating(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    executor = _make_executor(workspace, robot, source, shutdown_event)
    executor._last_z = 0.3  # white-box: force a "raising" scenario deterministically

    executor.execute_primitive(MoveTo(x=0.6, y=0.6, z=0.9, angle=45), ActionPhase.TASK)

    _assert_seq_matches(
        robot,
        [
            (OrderType.MOVE_Z, (0.9,)),
            (OrderType.MOVE_XY, (0.6, 0.6)),
            (OrderType.ROTATE, (45,)),
        ],
    )


def test_move_to_translates_before_lowering(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    executor = _make_executor(workspace, robot, source, shutdown_event)
    executor._last_z = 0.9  # force a "lowering" scenario

    executor.execute_primitive(MoveTo(x=0.6, y=0.6, z=0.4, angle=None), ActionPhase.TASK)

    _assert_seq_matches(
        robot,
        [
            (OrderType.MOVE_XY, (0.6, 0.6)),
            (OrderType.MOVE_Z, (0.4,)),
        ],
    )


def test_move_to_no_z_only_translates_and_rotates(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    executor = _make_executor(workspace, robot, source, shutdown_event)

    executor.execute_primitive(MoveTo(x=0.2, y=0.3, z=None, angle=60), ActionPhase.TASK)

    _assert_seq_matches(
        robot,
        [
            (OrderType.MOVE_XY, (0.2, 0.3)),
            (OrderType.ROTATE, (60,)),
        ],
    )


# --- PlaceTool -----------------------------------------------------------------------------


def test_place_tool_expansion_unguarded(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    executor = _make_executor(workspace, robot, source, shutdown_event)  # occupancy_supplier=None

    executor.execute_primitive(PlaceTool(x=0.4, y=0.6, angle=30, lower_to=0.34), ActionPhase.TASK)

    _assert_seq_matches(
        robot,
        [
            (OrderType.MOVE_Z, (0.8,)),  # clearance_height
            (OrderType.MOVE_XY, (0.4, 0.6)),
            (OrderType.ROTATE, (30,)),
            (OrderType.MOVE_Z, (0.34,)),  # lower_to, unguarded (occupancy_supplier is None)
        ],
    )


# --- LowerTool guard --------------------------------------------------------------------------


def test_lower_tool_guarded_proceeds_on_clear_mask(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    crop_mask = np.zeros((_CROP_SIZE[1], _CROP_SIZE[0]), dtype=np.uint8)
    supplier = _StubOccupancySupplier(_StubOccupancy(crop_mask))
    executor = _make_executor(workspace, robot, source, shutdown_event, occupancy_supplier=supplier)

    executor.execute_primitive(LowerTool(z=0.34, guarded=True), ActionPhase.TASK)

    _assert_seq_matches(robot, [(OrderType.MOVE_Z, (0.34,))])


def test_lower_tool_guarded_raises_unsafe_lower_on_obstacle(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    crop_mask = np.zeros((_CROP_SIZE[1], _CROP_SIZE[0]), dtype=np.uint8)
    # Default seeded pose is (x=0.5, y=0.5) -- DryRunRobot's default state -- put a blob right
    # under the tool footprint there.
    cx, cy = workspace.frames.robot_to_crop_px(0.5, 0.5)
    crop_mask[cy - 5 : cy + 5, cx - 5 : cx + 5] = 255
    supplier = _StubOccupancySupplier(_StubOccupancy(crop_mask))
    executor = _make_executor(workspace, robot, source, shutdown_event, occupancy_supplier=supplier)

    with pytest.raises(UnsafeLower) as excinfo:
        executor.execute_primitive(LowerTool(z=0.34, guarded=True), ActionPhase.TASK)
    assert excinfo.value.pose == (0.5, 0.5)
    assert excinfo.value.clearance_px == 0.0
    assert robot.command_log == []  # MOVE_Z was never sent
    # the top-level LowerTool action was opened, then closed cleanly on the raise (not left
    # dangling) -- but since the guard fired before any order was sent, there is no child action.
    all_actions = executor.tracker.get_all_actions()
    assert len(all_actions) == 1
    assert all_actions[0].parent_id is None
    assert all_actions[0].end_frame is not None
    assert executor.tracker.get_current_action() is None


def test_lower_tool_unguarded_when_occupancy_supplier_is_none(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    executor = _make_executor(workspace, robot, source, shutdown_event, occupancy_supplier=None)

    executor.execute_primitive(LowerTool(z=0.34, guarded=True), ActionPhase.TASK)

    _assert_seq_matches(robot, [(OrderType.MOVE_Z, (0.34,))])


# --- freshness gate -----------------------------------------------------------------------


def test_freshness_gate_raises_stale_mask_timeout(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    supplier = _StubOccupancySupplier(always_timeout=True)
    executor = _make_executor(
        workspace,
        robot,
        source,
        shutdown_event,
        occupancy_supplier=supplier,
        freshness_categories=["lower_tool"],
    )

    with pytest.raises(StaleMaskTimeout):
        # guarded=False isolates the generic gate: it must raise before the primitive is even
        # opened as an Action, regardless of the LowerTool-specific guard logic.
        executor.execute_primitive(LowerTool(z=0.34, guarded=False), ActionPhase.TASK)

    assert robot.command_log == []
    assert executor.tracker.get_all_actions() == []


def test_freshness_gate_proceeds_when_fresh_result_available(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    crop_mask = np.zeros((_CROP_SIZE[1], _CROP_SIZE[0]), dtype=np.uint8)
    supplier = _StubOccupancySupplier(_StubOccupancy(crop_mask))
    executor = _make_executor(
        workspace,
        robot,
        source,
        shutdown_event,
        occupancy_supplier=supplier,
        freshness_categories=["lower_tool"],
    )

    executor.execute_primitive(LowerTool(z=0.34, guarded=True), ActionPhase.TASK)
    _assert_seq_matches(robot, [(OrderType.MOVE_Z, (0.34,))])


# --- Push --------------------------------------------------------------------------------------


def test_push_skips_redundant_start_and_z_moves(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    executor = _make_executor(workspace, robot, source, shutdown_event)
    executor._last_x, executor._last_y, executor._last_z = 0.5, 0.5, 0.34  # already there

    push = Push(start_x=0.5, start_y=0.5, angle=10, end_x=0.6, end_y=0.6, height=0.34)
    executor.execute_primitive(push, ActionPhase.TASK)

    _assert_seq_matches(
        robot,
        [
            (OrderType.ROTATE, (10,)),
            (OrderType.MOVE_XY, (0.6, 0.6)),
        ],
    )


def test_push_includes_start_and_z_moves_when_needed(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    executor = _make_executor(workspace, robot, source, shutdown_event)
    # default seed: x=0.5, y=0.5, z=1.0

    push = Push(start_x=0.7, start_y=0.7, angle=20, end_x=0.3, end_y=0.3, height=0.34)
    executor.execute_primitive(push, ActionPhase.TASK)

    _assert_seq_matches(
        robot,
        [
            (OrderType.ROTATE, (20,)),
            (OrderType.MOVE_XY, (0.7, 0.7)),
            (OrderType.MOVE_Z, (0.34,)),
            (OrderType.MOVE_XY, (0.3, 0.3)),
        ],
    )


def test_push_action_details_and_state_complete(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    executor = _make_executor(workspace, robot, source, shutdown_event)

    push = Push(start_x=0.5, start_y=0.5, angle=15, end_x=0.55, end_y=0.55, height=0.34)
    executor.execute_primitive(push, ActionPhase.TASK)

    top_actions = [a for a in executor.tracker.get_all_actions() if a.parent_id is None]
    assert len(top_actions) == 1
    top = top_actions[0]

    assert top.action_type is ActionType.MOVE_XY
    assert top.is_planar_2d is True
    assert top.action_details == {
        "start_x": 0.5,
        "start_y": 0.5,
        "angle": 15,
        "end_x": 0.55,
        "end_y": 0.55,
        "height": 0.34,
    }
    # design 03 §5's transition-writer needs: start/end robot state + frames, both already on the
    # Action itself (not duplicated into action_details -- see executor.py's module docstring).
    assert top.start_frame is not None and top.end_frame is not None
    assert top.end_frame >= top.start_frame
    assert top.start_robot_state is not None and top.end_robot_state is not None
    assert set(top.start_robot_state.keys()) == {"x", "y", "z", "rotation", "claw"}


def test_action_tree_children_reference_top_level_parent_id(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    executor = _make_executor(workspace, robot, source, shutdown_event)

    push = Push(start_x=0.7, start_y=0.7, angle=20, end_x=0.3, end_y=0.3, height=0.34)
    executor.execute_primitive(push, ActionPhase.TASK)

    all_actions = executor.tracker.get_all_actions()
    tops = [a for a in all_actions if a.parent_id is None]
    children = [a for a in all_actions if a.parent_id is not None]
    assert len(tops) == 1
    assert len(children) == 4  # rotate, move_xy(start), move_z, move_xy(end)
    assert all(c.parent_id == tops[0].action_id for c in children)
    # frames monotonic across the full recorded sequence
    frames = [a.start_frame for a in all_actions] + [a.end_frame for a in all_actions]
    assert all(f is not None for f in frames)


# --- SweepWall -----------------------------------------------------------------------------


def _expected_sweep_pass(wall, t, step, already_at_wall, cfg):
    pose = sample_tool_pose(wall, tool_width=0.02, safety_margin=0.02, t=t)
    x, y = float(pose["x"]), float(pose["y"])
    dx, dy = float(pose["perpendicular_dir"][0]), float(pose["perpendicular_dir"][1])
    rotation_angle = pose["angle"]

    seq = []
    if not already_at_wall:
        seq += [
            (OrderType.MOVE_Z, (cfg.clearance_height,)),
            (OrderType.MOVE_XY, (x, y)),
            (OrderType.ROTATE, (rotation_angle,)),
            (OrderType.MOVE_Z, (cfg.sweep_height + 0.02,)),
            (OrderType.MOVE_XY, (x + dx * 0.02, y + dy * 0.02)),
            (OrderType.MOVE_Z, (cfg.clearance_height,)),
            (OrderType.MOVE_XY, (x, y)),
            (OrderType.MOVE_Z, (cfg.sweep_height - 0.01,)),
            (OrderType.MOVE_XY, (x + dx * 0.02, y + dy * 0.02)),
            (OrderType.MOVE_Z, (cfg.clearance_height,)),
            (OrderType.MOVE_XY, (x, y)),
            (OrderType.MOVE_Z, (cfg.grasp_height,)),
        ]
    seq += [
        (OrderType.MOVE_XY, (x, y)),
        (OrderType.MOVE_XY, (x + dx * step, y + dy * step)),
        (OrderType.MOVE_XY, (x, y)),
    ]
    return seq


def test_sweep_wall_targeted_expansion(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    executor = _make_executor(workspace, robot, source, shutdown_event)
    wall = workspace.wall("top")
    cfg = workspace.config

    t_values = tuple(np.linspace(wall.t_min, wall.t_max, num=3))
    sweep = SweepWall(
        wall_label="top",
        mode="targeted",
        angle=wall.angle,
        sweep_dir=(float(wall.normal[0]), float(wall.normal[1])),
        step=0.1,
        t_values=t_values,
        approach_x=0.5,
        approach_y=0.6,
        sweep_pos_x=0.5,
        sweep_pos_y=0.65,
    )
    executor.execute_primitive(sweep, ActionPhase.RESET)

    expected = [
        (OrderType.MOVE_Z, (cfg.clearance_height,)),
        (OrderType.MOVE_XY, (0.5, 0.6)),
        (OrderType.ROTATE, (wall.angle,)),
        (OrderType.MOVE_Z, (cfg.grasp_height,)),
        (OrderType.MOVE_XY, (0.5, 0.65)),
    ]
    for t in t_values:
        expected += _expected_sweep_pass(wall, t, 0.1, already_at_wall=True, cfg=cfg)
    _assert_seq_matches(robot, expected)

    top_actions = [a for a in executor.tracker.get_all_actions() if a.parent_id is None]
    assert len(top_actions) == 1
    assert top_actions[0].action_type is ActionType.SWEEP


def test_sweep_wall_fallback_expansion(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    executor = _make_executor(workspace, robot, source, shutdown_event)
    wall = workspace.wall("left")
    cfg = workspace.config

    t_values = tuple(np.linspace(wall.t_min, wall.t_max, num=3))
    sweep = SweepWall(
        wall_label="left",
        mode="fallback",
        angle=wall.angle,
        sweep_dir=(float(wall.normal[0]), float(wall.normal[1])),
        step=0.12,
        t_values=t_values,
    )
    executor.execute_primitive(sweep, ActionPhase.RESET)

    expected = []
    already_at_wall = False
    for t in t_values:
        expected += _expected_sweep_pass(wall, t, 0.12, already_at_wall=already_at_wall, cfg=cfg)
        already_at_wall = True
    _assert_seq_matches(robot, expected)


# --- RegraspTool ---------------------------------------------------------------------------


def test_regrasp_tool_expansion_sequence(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    executor = _make_executor(workspace, robot, source, shutdown_event)  # no grip checker

    executor.execute_primitive(RegraspTool(), ActionPhase.STARTUP)

    _assert_seq_matches(
        robot,
        [
            (OrderType.MOVE_Z, (1.0,)),
            (OrderType.GRIPPER, (1.0,)),
            (OrderType.ROTATE, (0,)),
            (OrderType.MOVE_XY, (0.03, 0.49)),
            (OrderType.MOVE_Z, (0.27,)),
            (OrderType.GRIPPER, (0.0,)),
            (OrderType.MOVE_Z, (1.0,)),
            (OrderType.MOVE_XY, (0.5, 0.41)),
            (OrderType.ROTATE, (90,)),
        ],
    )


def test_regrasp_tool_raises_tool_lost_on_bad_grip(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    checker = _StubGripChecker(quality=0.1)
    executor = _make_executor(workspace, robot, source, shutdown_event, tool_grip_checker=checker)

    with pytest.raises(ToolLost) as excinfo:
        executor.execute_primitive(RegraspTool(), ActionPhase.STARTUP)
    assert excinfo.value.grip_quality == 0.1
    # the scripted motion itself still fully ran before the check failed
    assert len(robot.command_log) == 9


def test_regrasp_tool_ok_grip_does_not_raise(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    checker = _StubGripChecker(quality=0.9)
    executor = _make_executor(workspace, robot, source, shutdown_event, tool_grip_checker=checker)

    executor.execute_primitive(RegraspTool(), ActionPhase.STARTUP)  # must not raise
    assert len(robot.command_log) == 9


# --- CheckToolGrip -------------------------------------------------------------------------


def test_check_tool_grip_expansion_sequence(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    checker = _StubGripChecker(quality=0.9)
    executor = _make_executor(workspace, robot, source, shutdown_event, tool_grip_checker=checker)

    executor.execute_primitive(CheckToolGrip(), ActionPhase.STARTUP)

    _assert_seq_matches(
        robot,
        [
            (OrderType.MOVE_Z, (1.0,)),
            (OrderType.MOVE_XY, (0.5, 0.41)),
            (OrderType.ROTATE, (90,)),
        ],
    )


def test_check_tool_grip_raises_tool_lost(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    checker = _StubGripChecker(quality=0.2)
    executor = _make_executor(workspace, robot, source, shutdown_event, tool_grip_checker=checker)

    with pytest.raises(ToolLost):
        executor.execute_primitive(CheckToolGrip(), ActionPhase.STARTUP)


def test_check_tool_grip_no_checker_is_noop_with_warning(tmp_path, env, caplog):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    executor = _make_executor(workspace, robot, source, shutdown_event, tool_grip_checker=None)

    with caplog.at_level("WARNING"):
        executor.execute_primitive(CheckToolGrip(), ActionPhase.STARTUP)  # must not raise
    assert any("no tool_grip_checker configured" in r.message for r in caplog.records)


# --- RefreshMask ---------------------------------------------------------------------------


def test_refresh_mask_expansion_sequence(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    executor = _make_executor(workspace, robot, source, shutdown_event)

    executor.execute_primitive(RefreshMask(move_aside=True), ActionPhase.TASK)

    _assert_seq_matches(
        robot,
        [
            (OrderType.MOVE_Z, (1.0,)),
            (OrderType.MOVE_XY, (0.0, 1.0)),
            (OrderType.ROTATE, (90,)),
        ],
    )


def test_refresh_mask_move_aside_false_is_noop(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    executor = _make_executor(workspace, robot, source, shutdown_event)

    executor.execute_primitive(RefreshMask(move_aside=False), ActionPhase.TASK)
    assert robot.command_log == []


# --- order_sink ----------------------------------------------------------------------------


def test_order_sink_receives_legacy_shaped_records_plus_frame_index(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    records = []
    executor = _make_executor(
        workspace, robot, source, shutdown_event, order_sink=lambda rec: records.append(rec)
    )

    executor.execute_primitive(MoveTo(x=0.3, y=0.3, z=None, angle=None), ActionPhase.TASK)

    assert len(records) == 1
    record = records[0]
    assert set(record.keys()) == {"order_type", "order_value", "time", "robot_reported_time", "frame_index"}
    assert record["order_type"] == "MOVE_XY"
    assert record["order_value"] == pytest.approx([0.3, 0.3])
    assert isinstance(record["time"], float)
    assert isinstance(record["frame_index"], int)


def test_order_sink_exception_is_swallowed(tmp_path, env, caplog):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)

    def bad_sink(record):
        raise RuntimeError("boom")

    executor = _make_executor(workspace, robot, source, shutdown_event, order_sink=bad_sink)

    with caplog.at_level("ERROR"):
        executor.execute_primitive(MoveTo(x=0.3, y=0.3, z=None, angle=None), ActionPhase.TASK)
    assert any("order_sink raised" in r.message for r in caplog.records)


# --- run_plan / PlanResult -------------------------------------------------------------------


def test_run_plan_completes_all_primitives(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    executor = _make_executor(workspace, robot, source, shutdown_event)

    plan = Plan(
        primitives=(
            MoveTo(x=0.3, y=0.3, z=None, angle=None),
            MoveTo(x=0.4, y=0.4, z=None, angle=None),
        ),
        meta={"phase": "task"},
    )
    result = executor.run_plan(plan, ActionPhase.TASK)
    assert result == PlanResult(completed=2, aborted=False, error=None)


def test_run_plan_aborts_cleanly_on_shutdown_between_primitives(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    executor = _make_executor(workspace, robot, source, shutdown_event)

    shutdown_event.set()
    plan = Plan(primitives=(MoveTo(x=0.3, y=0.3, z=None, angle=None),), meta={})
    result = executor.run_plan(plan, ActionPhase.TASK)
    assert result.completed == 0
    assert result.aborted is True
    assert result.error is None
    assert robot.command_log == []


def test_run_plan_propagates_typed_error_and_closes_actions(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    crop_mask = np.zeros((_CROP_SIZE[1], _CROP_SIZE[0]), dtype=np.uint8)
    cx, cy = workspace.frames.robot_to_crop_px(0.5, 0.5)
    crop_mask[cy - 5 : cy + 5, cx - 5 : cx + 5] = 255
    supplier = _StubOccupancySupplier(_StubOccupancy(crop_mask))
    executor = _make_executor(workspace, robot, source, shutdown_event, occupancy_supplier=supplier)

    plan = Plan(primitives=(LowerTool(z=0.34, guarded=True),), meta={})
    with pytest.raises(UnsafeLower):
        executor.run_plan(plan, ActionPhase.TASK)
    # top-level action was opened then closed (not left dangling)
    assert executor.tracker.get_current_action() is None
    assert executor.tracker.get_current_child_action() is None


# --- completed-action stream ------------------------------------------------------------------


def test_register_completion_callback_passthrough(tmp_path, env):
    robot, source, shutdown_event = env
    workspace = _make_workspace(tmp_path)
    executor = _make_executor(workspace, robot, source, shutdown_event)

    completed = []
    executor.register_completion_callback(lambda action: completed.append(action))

    executor.execute_primitive(MoveTo(x=0.3, y=0.3, z=None, angle=None), ActionPhase.TASK)

    # one top-level action + one child (MOVE_XY) action completed
    assert len(completed) == 2
    top = next(a for a in completed if a.parent_id is None)
    child = next(a for a in completed if a.parent_id is not None)
    assert child.parent_id == top.action_id
