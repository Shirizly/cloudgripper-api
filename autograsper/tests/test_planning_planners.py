"""Tests for `autograsper.planning` (Wave 3b): `WorldState.staleness`, `RandomPushPlanner`,
`SegPushPlanner`.

Synthetic data + stub `OccupancyLike` objects only (no network/GPU/robot, no perception-provider
imports — this wave code-generates against the `OccupancyLike`/`ClumpStatsLike` Protocols by duck
typing, per the shared contract with the parallel perception-layer agent). All planner randomness
is seeded via an explicit `np.random.Generator`, per CONVENTIONS.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import pytest

from autograsper.config_schema import (
    CropConfig,
    GridConfig,
    ManipulationBoundaryRobot,
    PerceptionConfig,
    WorkspaceConfig,
)
from autograsper.observation.types import Observation, RobotState
from autograsper.perception.frames import CoordinateFrames
from autograsper.planning.planner import FreshnessPolicy, NoGranulesDetected
from autograsper.planning.random_push_planner import RandomPushPlanner
from autograsper.planning.seg_push_planner import SegPushPlanner
from autograsper.planning.types import (
    CheckToolGrip,
    LowerTool,
    OccupancyLike,
    PlaceTool,
    Plan,
    Push,
    RefreshMask,
    SweepWall,
    ToolStatus,
    WorldState,
)
from autograsper.planning.workspace import Workspace

# --- Shared test fixtures --------------------------------------------------------------

_FENCE_CENTER = (0.5, 0.49)
_FENCE_SIZE = (0.94, 0.955)


def _synthetic_H(crop_w: int, crop_h: int) -> np.ndarray:
    return np.array(
        [
            [1.0 / crop_w, 0.0, 0.0],
            [0.0, -1.0 / crop_h, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _make_workspace(tmp_path, crop_size=(300, 300), tool_dims_px=(8, 40)) -> Workspace:
    crop = CropConfig(center_px=(crop_size[0] // 2, crop_size[1] // 2), size=crop_size)
    grid = GridConfig(height=32, width=32)
    H = _synthetic_H(crop_size[0], crop_size[1])
    npz_path = tmp_path / "homography.npz"
    np.savez(npz_path, H)

    workspace_config = WorkspaceConfig(
        fence_center=_FENCE_CENTER,
        fence_size=_FENCE_SIZE,
        manipulation_boundary_robot=ManipulationBoundaryRobot(x=(0.1, 0.9), y=(0.1, 0.9)),
        tool_length_robot=0.36,
        tool_width_robot=0.017,
        tool_dims_px=tool_dims_px,
        homography_npz_path=str(npz_path),
        grasp_height=0.34,
        sweep_height=0.57,
        clearance_height=0.8,
        safety_margin=0.01,
    )
    perception_config = PerceptionConfig(provider="none", crop=crop, grid=grid)
    frames = CoordinateFrames.from_config(workspace_config, perception_config)
    return Workspace(workspace_config, frames)


def _make_config(n_pushes: int = 3, min_granule_size: int = 300):
    """Duck-typed stand-in for `config_schema.Config` carrying only the fields the planners
    actually read (`experiment.n_pushes`, `workspace.grasp_height`,
    `perception.background_diff.min_granule_size`) — avoids constructing every unrelated required
    `Config` section (`camera`, `robot`, `tool_check`, `storage`, `ui`, ...) for a planning-only
    test."""
    return SimpleNamespace(
        experiment=SimpleNamespace(n_pushes=n_pushes),
        workspace=SimpleNamespace(grasp_height=0.34),
        perception=SimpleNamespace(
            background_diff=SimpleNamespace(min_granule_size=min_granule_size)
        ),
    )


@dataclass(frozen=True)
class StubClumpStats:
    num_clumps: int
    total_area_px: int
    areas: Tuple[int, ...]
    centroids: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class StubOccupancy:
    """Duck-typed stand-in for `perception.occupancy.OccupancyResult` — satisfies
    `planning.types.OccupancyLike` structurally without importing anything from
    `autograsper.perception` beyond the already-complete `frames.py`."""

    source_seq: int
    frame_index: Optional[int]
    timestamp: float
    grid_mask: np.ndarray
    crop_mask: np.ndarray
    instances: tuple
    stats: StubClumpStats


def _make_observation(seq: int = 5) -> Observation:
    top = np.zeros((10, 10, 3), dtype=np.uint8)
    bottom = np.zeros((10, 10, 3), dtype=np.uint8)
    state = RobotState(x=0.5, y=0.5, z=0.8, rotation=0.0, claw=1.0)
    return Observation(seq=seq, frame_index=None, timestamp=0.0, top_image=top, bottom_image=bottom, robot_state=state)


def _make_occupancy(mask: np.ndarray, source_seq: int = 5, num_clumps: int = 1) -> StubOccupancy:
    stats = StubClumpStats(num_clumps=num_clumps, total_area_px=int(mask.sum() // 255), areas=(1,), centroids=((0.0, 0.0),))
    return StubOccupancy(
        source_seq=source_seq,
        frame_index=None,
        timestamp=0.0,
        grid_mask=mask,
        crop_mask=mask,
        instances=(),
        stats=stats,
    )


def _make_world_state(workspace: Workspace, mask: Optional[np.ndarray], *, obs_seq=5, occ_seq=5, num_clumps=1) -> WorldState:
    obs = _make_observation(seq=obs_seq)
    occupancy = None if mask is None else _make_occupancy(mask, source_seq=occ_seq, num_clumps=num_clumps)
    tool = ToolStatus(held=True, grip_quality=0.9)
    return WorldState(obs=obs, occupancy=occupancy, tool=tool, workspace=workspace)


def _free_mask(shape=(300, 300)) -> np.ndarray:
    """Mostly-empty mask: a small sparse occupied patch far from the manipulation boundary so
    that `crop_mask` isn't entirely zero (some code paths treat a fully-zero mask as "no
    material") while leaving the whole manipulation region free for placement."""
    mask = np.zeros(shape, dtype=np.uint8)
    mask[0:5, 0:5] = 255
    return mask


def _fully_occupied_mask(shape=(300, 300)) -> np.ndarray:
    return np.full(shape, 255, dtype=np.uint8)


def _peripheral_mask(shape=(300, 300)) -> np.ndarray:
    """Material piled in a far corner, well outside the central 30-70% box -> needs_reset True."""
    mask = np.zeros(shape, dtype=np.uint8)
    mask[0:20, 0:20] = 255
    return mask


def _central_mask(shape=(300, 300)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    mask[int(h * 0.4) : int(h * 0.6), int(w * 0.4) : int(w * 0.6)] = 255
    return mask


def _top_wall_piled_mask(shape=(300, 300)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[0:40, :] = 255  # robot y near 1 (top wall) maps to low row indices
    return mask


# --- OccupancyLike protocol / WorldState.staleness --------------------------------------


def test_stub_occupancy_satisfies_occupancy_like_protocol():
    occupancy = _make_occupancy(_free_mask())
    assert isinstance(occupancy, OccupancyLike)


def test_world_state_staleness_computed_from_seqs(tmp_path):
    workspace = _make_workspace(tmp_path)
    w = _make_world_state(workspace, _free_mask(), obs_seq=10, occ_seq=7)
    assert w.staleness == 3


def test_world_state_staleness_none_without_occupancy(tmp_path):
    workspace = _make_workspace(tmp_path)
    w = _make_world_state(workspace, None)
    assert w.staleness is None


# --- FreshnessPolicy ---------------------------------------------------------------------


def test_freshness_policy_matches_configured_categories():
    policy = FreshnessPolicy(["lower_tool", "wall_check"])
    assert policy.requires_fresh(LowerTool(z=0.34)) is True
    assert policy.requires_fresh(SweepWall(wall_label="top", mode="fallback", angle=90.0, sweep_dir=(0.0, -1.0), step=0.1)) is True
    assert policy.requires_fresh(PlaceTool(x=0.5, y=0.5, angle=0.0, lower_to=0.34)) is False
    assert policy.requires_fresh(Push(start_x=0.5, start_y=0.5, angle=0.0, end_x=0.6, end_y=0.6, height=0.34)) is False


def test_freshness_policy_empty_config_requires_nothing():
    policy = FreshnessPolicy([])
    assert policy.requires_fresh(LowerTool(z=0.34)) is False


# --- RandomPushPlanner ---------------------------------------------------------------------


def test_random_push_planner_plan_startup_shape(tmp_path):
    workspace = _make_workspace(tmp_path)
    planner = RandomPushPlanner(_make_config(), workspace, np.random.default_rng(0))
    w = _make_world_state(workspace, None)
    plan = planner.plan_startup(w)
    assert isinstance(plan, Plan)
    assert plan.primitives == (CheckToolGrip(), RefreshMask(move_aside=True))
    assert plan.meta["phase"] == "startup"


def test_random_push_planner_needs_reset_true_when_peripheral(tmp_path):
    workspace = _make_workspace(tmp_path)
    planner = RandomPushPlanner(_make_config(), workspace, np.random.default_rng(0))
    w = _make_world_state(workspace, _peripheral_mask())
    assert planner.needs_reset(w) is True


def test_random_push_planner_needs_reset_false_when_central(tmp_path):
    workspace = _make_workspace(tmp_path)
    planner = RandomPushPlanner(_make_config(), workspace, np.random.default_rng(0))
    w = _make_world_state(workspace, _central_mask())
    assert planner.needs_reset(w) is False


def test_random_push_planner_needs_reset_false_without_occupancy(tmp_path):
    workspace = _make_workspace(tmp_path)
    planner = RandomPushPlanner(_make_config(), workspace, np.random.default_rng(0))
    w = _make_world_state(workspace, None)
    assert planner.needs_reset(w) is False


def test_random_push_planner_plan_task_places_tool_and_pushes_within_boundary(tmp_path):
    n_pushes = 4
    workspace = _make_workspace(tmp_path)
    planner = RandomPushPlanner(_make_config(n_pushes=n_pushes), workspace, np.random.default_rng(1))
    w = _make_world_state(workspace, _free_mask())

    plan = planner.plan_task(w)
    prims = plan.primitives

    assert isinstance(prims[0], PlaceTool)
    pushes = [p for p in prims if isinstance(p, Push)]
    assert len(pushes) == n_pushes
    assert isinstance(prims[-1], RefreshMask)

    manip_x = workspace.manip_x
    manip_y = workspace.manip_y
    for i, push in enumerate(pushes):
        assert manip_x[0] <= push.end_x <= manip_x[1]
        assert manip_y[0] <= push.end_y <= manip_y[1]
        assert 0.0 <= push.angle <= 180.0
        assert push.height == pytest.approx(0.34)
        if i > 0:
            assert push.start_x == pushes[i - 1].end_x
            assert push.start_y == pushes[i - 1].end_y

    assert plan.meta["phase"] == "task"
    assert plan.meta["reason"] == "placed"


def test_random_push_planner_plan_task_sweeps_when_no_placement(tmp_path):
    workspace = _make_workspace(tmp_path)
    planner = RandomPushPlanner(_make_config(), workspace, np.random.default_rng(2))
    w = _make_world_state(workspace, _fully_occupied_mask())

    plan = planner.plan_task(w)
    prims = plan.primitives
    assert len(prims) == 2
    assert isinstance(prims[0], SweepWall)
    assert prims[0].mode == "fallback"
    assert isinstance(prims[1], RefreshMask)
    assert plan.meta["reason"] == "no_placement"


def test_random_push_planner_plan_task_raises_without_occupancy(tmp_path):
    workspace = _make_workspace(tmp_path)
    planner = RandomPushPlanner(_make_config(), workspace, np.random.default_rng(0))
    w = _make_world_state(workspace, None)
    with pytest.raises(ValueError):
        planner.plan_task(w)


def test_random_push_planner_plan_reset_sweeps_the_needy_wall(tmp_path):
    workspace = _make_workspace(tmp_path)
    planner = RandomPushPlanner(_make_config(), workspace, np.random.default_rng(3))
    w = _make_world_state(workspace, _top_wall_piled_mask())

    plan = planner.plan_reset(w)
    sweeps = [p for p in plan.primitives if isinstance(p, SweepWall)]
    assert any(s.wall_label == "top" for s in sweeps)
    # Every SweepWall in a random-push reset plan is followed by a RefreshMask (cautious pipeline).
    for i, prim in enumerate(plan.primitives):
        if isinstance(prim, SweepWall):
            assert isinstance(plan.primitives[i + 1], RefreshMask)
    assert plan.meta["phase"] == "reset"


def test_random_push_planner_plan_reset_empty_without_occupancy(tmp_path):
    workspace = _make_workspace(tmp_path)
    planner = RandomPushPlanner(_make_config(), workspace, np.random.default_rng(0))
    w = _make_world_state(workspace, None)
    plan = planner.plan_reset(w)
    assert plan.primitives == ()


# --- SegPushPlanner ------------------------------------------------------------------------


def test_seg_push_planner_plan_startup_raises_when_no_granules(tmp_path):
    workspace = _make_workspace(tmp_path)
    planner = SegPushPlanner(_make_config(), workspace, np.random.default_rng(0))
    w = _make_world_state(workspace, _free_mask(), num_clumps=0)
    with pytest.raises(NoGranulesDetected):
        planner.plan_startup(w)


def test_seg_push_planner_plan_startup_raises_when_no_occupancy(tmp_path):
    workspace = _make_workspace(tmp_path)
    planner = SegPushPlanner(_make_config(), workspace, np.random.default_rng(0))
    w = _make_world_state(workspace, None)
    with pytest.raises(NoGranulesDetected):
        planner.plan_startup(w)


def test_seg_push_planner_plan_startup_ok_when_granules_present(tmp_path):
    workspace = _make_workspace(tmp_path)
    planner = SegPushPlanner(_make_config(), workspace, np.random.default_rng(0))
    w = _make_world_state(workspace, _free_mask(), num_clumps=2)
    plan = planner.plan_startup(w)
    assert plan.primitives == (CheckToolGrip(),)
    assert not any(isinstance(p, RefreshMask) for p in plan.primitives)


def test_seg_push_planner_plan_task_emits_no_refresh_mask(tmp_path):
    n_pushes = 3
    workspace = _make_workspace(tmp_path)
    planner = SegPushPlanner(_make_config(n_pushes=n_pushes), workspace, np.random.default_rng(1))
    w = _make_world_state(workspace, _free_mask())

    plan = planner.plan_task(w)
    assert not any(isinstance(p, RefreshMask) for p in plan.primitives)
    pushes = [p for p in plan.primitives if isinstance(p, Push)]
    assert len(pushes) == n_pushes


def test_seg_push_planner_plan_task_no_placement_emits_no_refresh_mask(tmp_path):
    workspace = _make_workspace(tmp_path)
    planner = SegPushPlanner(_make_config(), workspace, np.random.default_rng(2))
    w = _make_world_state(workspace, _fully_occupied_mask())

    plan = planner.plan_task(w)
    assert not any(isinstance(p, RefreshMask) for p in plan.primitives)
    assert isinstance(plan.primitives[0], SweepWall)


def test_seg_push_planner_plan_reset_emits_one_sweep_and_requests_replan(tmp_path):
    workspace = _make_workspace(tmp_path)
    planner = SegPushPlanner(_make_config(), workspace, np.random.default_rng(3))
    w = _make_world_state(workspace, _top_wall_piled_mask())

    plan = planner.plan_reset(w)
    sweeps = [p for p in plan.primitives if isinstance(p, SweepWall)]
    assert len(sweeps) <= 1
    assert not any(isinstance(p, RefreshMask) for p in plan.primitives)
    assert plan.meta["replan_after_each"] is True


def test_seg_push_planner_plan_reset_empty_without_occupancy(tmp_path):
    workspace = _make_workspace(tmp_path)
    planner = SegPushPlanner(_make_config(), workspace, np.random.default_rng(0))
    w = _make_world_state(workspace, None)
    plan = planner.plan_reset(w)
    assert plan.primitives == ()
    assert plan.meta["replan_after_each"] is True
