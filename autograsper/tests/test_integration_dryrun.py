"""End-to-end integration test for Wave 5 (session/recording/storage/composition).

Exercises the full stack — `DryRunRobot` -> `ObservationSource` -> `Executor` ->
`recording.recorder.Recorder` -> `session.coordinator.SessionRunner` -> `session.storage` writers —
against a deterministic synthetic scene (no network/GPU/robot, per CONVENTIONS.md). Components are
wired by hand (same pattern `test_execution_executor.py`/`test_planning_planners.py` use: a
synthetic homography + a pass-through camera) rather than through `main_granular.build_components`,
since the latter requires real calibration files (`camera.m/d/H`, `workspace.homography_npz_path`)
that intentionally don't exist in this repo (`granular-config.yaml`'s `# PLACEHOLDER — calibrate`
entries) — `main_granular.py`'s own composition logic is exercised structurally (imports, argparse,
dryrun-only default) rather than by a live end-to-end run.

Occupancy is a hand-built, unchanging "mostly-central blob" (`_make_synthetic_masks`): fully inside
the central 30-70% box `planning.workspace.check_center_reset_needed` checks, so `needs_reset` is
always `False` and no `RESETTING` phase is exercised (kept deterministic and fast); there is ample
free space around it for the placement search to find a granule-free tool pose every episode.

Two scenarios:
- `test_full_episode_lifecycle_dryrun`: 2 episodes end-to-end, asserting the full on-disk dataset
  contract (`docs/dataset_formats.md`) and the online transition-dataset emission.
- `test_tool_lost_intervention_recovers`: a scripted grip checker that fails once (raising
  `ToolLost` -> `INTERVENTION`) then succeeds, asserting the state machine actually visits
  `INTERVENTION` and recovers to a successful episode.
"""

from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace
from typing import List, Optional

import cv2
import numpy as np
import pytest
import torch
import yaml

from autograsper.config_schema import (
    CropConfig,
    GridConfig,
    ManipulationBoundaryRobot,
    PerceptionConfig,
    WorkspaceConfig,
)
from autograsper.execution.actions import ActionTracker
from autograsper.execution.executor import Executor
from autograsper.execution.safety import SafetyValidator
from autograsper.hardware.dryrun import DryRunRobot
from autograsper.observation.source import ObservationSource
from autograsper.perception.frames import CoordinateFrames
from autograsper.perception.occupancy import clump_stats_from_mask
from autograsper.planning.planner import FreshnessPolicy
from autograsper.planning.seg_push_planner import SegPushPlanner
from autograsper.planning.workspace import Workspace
from autograsper.recording.recorder import Recorder
from autograsper.session.coordinator import SessionRunner
from autograsper.session.episode import EpisodeState
from autograsper.session.storage import OrderSinkRouter, TransitionWriter

_CROP_SIZE = (300, 300)  # (w, h)
_GRID_SIZE = (32, 32)  # (w, h)
_TOOL_DIMS_PX = (8, 40)
_N_PUSHES = 3


class _PassthroughCamera:
    def process_bottom(self, raw_bottom):
        return raw_bottom


def _synthetic_H(crop_w: int, crop_h: int) -> np.ndarray:
    """Same synthetic homography used by `test_execution_executor.py`/`test_planning_planners.py`:
    `crop_px (u, v) -> robot (u/crop_w, 1 - v/crop_h)`."""
    return np.array(
        [
            [1.0 / crop_w, 0.0, 0.0],
            [0.0, -1.0 / crop_h, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _make_workspace(tmp_path, manip=(0.05, 0.95)) -> Workspace:
    crop = CropConfig(center_px=(_CROP_SIZE[0] // 2, _CROP_SIZE[1] // 2), size=_CROP_SIZE)
    grid = GridConfig(height=_GRID_SIZE[1], width=_GRID_SIZE[0])
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
        safety_margin=0.02,
    )
    perception_config = PerceptionConfig(provider="yolo", crop=crop, grid=grid)
    frames = CoordinateFrames.from_config(workspace_config, perception_config)
    return Workspace(workspace_config, frames)


def _make_synthetic_masks():
    """A filled circle centered in the crop -- fully inside the central 30-70% box (`needs_reset`
    stays `False` throughout), with ample free space around it for placement search."""
    w, h = _CROP_SIZE
    crop_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(crop_mask, (w // 2, h // 2), 30, 255, thickness=-1)
    grid_mask = cv2.resize(crop_mask, _GRID_SIZE, interpolation=cv2.INTER_NEAREST)
    stats = clump_stats_from_mask(crop_mask)
    return crop_mask, grid_mask, stats


class _StubOccupancySource:
    """Deterministic, unchanging occupancy, reachable via the `SegmentationWorker`-shaped API
    (`latest()`/`await_result()`) both `Executor`'s guards and `SessionRunner.world_state()` use.
    `source_seq` tracks the live `ObservationSource`'s current `seq` (not an independent counter)
    so it behaves like a real continuously-fresh segmenter: freshness/guard checks never spuriously
    time out, and the recorder's own "already saved this source_seq" dedup naturally advances frame
    to frame instead of getting stuck."""

    def __init__(self, source: ObservationSource, crop_mask: np.ndarray, grid_mask: np.ndarray, stats) -> None:
        self._source = source
        self._crop_mask = crop_mask
        self._grid_mask = grid_mask
        self._stats = stats

    def _make_result(self):
        obs = self._source.latest()
        seq = obs.seq if obs is not None else 0
        frame_index = obs.frame_index if obs is not None else None
        return SimpleNamespace(
            source_seq=seq,
            frame_index=frame_index,
            timestamp=time.time(),
            grid_mask=self._grid_mask,
            crop_mask=self._crop_mask,
            instances=(),
            stats=self._stats,
        )

    def latest(self):
        return self._make_result()

    def await_result(self, min_source_seq=0, timeout=None):
        return self._make_result()


class _StubGripChecker:
    """`ToolGripChecker`-shaped stub: `.check(obs) -> SimpleNamespace(quality, ok)`. `script`, if
    given, is consumed left-to-right (one entry per `check()` call); once exhausted, every further
    call returns `default_ok`."""

    def __init__(self, *, default_ok: bool = True, script: Optional[List[bool]] = None) -> None:
        self._default_ok = default_ok
        self._script = list(script) if script is not None else []
        self.calls = 0

    def check(self, obs):
        self.calls += 1
        ok = self._script.pop(0) if self._script else self._default_ok
        quality = 1.0 if ok else 0.0
        return SimpleNamespace(quality=quality, ok=ok)


def _build_environment(tmp_path, *, grip_checker, episode_budget, on_state_transition=None):
    """Wire every Wave 1-5 layer by hand (synthetic calibration, no config file needed). Returns
    `(runner, source, recorder, teardown)` -- call `teardown()` after the run to stop background
    threads."""
    shutdown_event = threading.Event()
    robot = DryRunRobot(top_image_shape=(32, 32, 3), bottom_image_shape=(32, 32, 3))
    source = ObservationSource(robot, _PassthroughCamera(), fps=20.0, shutdown_event=shutdown_event)
    source.start()
    source.await_next(0, timeout=2.0)

    workspace = _make_workspace(tmp_path)
    crop_mask, grid_mask, stats = _make_synthetic_masks()
    occupancy_source = _StubOccupancySource(source, crop_mask, grid_mask, stats)

    config = SimpleNamespace(
        experiment=SimpleNamespace(n_pushes=_N_PUSHES, time_between_orders=0.0, timeout_between_experiments=0.01),
        workspace=SimpleNamespace(grasp_height=0.34),
        perception=SimpleNamespace(background_diff=None),
    )

    tracker = ActionTracker()
    safety = SafetyValidator(workspace)
    freshness = FreshnessPolicy(())
    order_sink_router = OrderSinkRouter()

    executor = Executor(
        robot,
        source,
        tracker,
        safety,
        freshness,
        workspace,
        config,
        occupancy_supplier=occupancy_source,
        tool_grip_checker=grip_checker,
        order_sink=order_sink_router,
        shutdown_event=shutdown_event,
        freshness_timeout=1.0,
        end_frame_timeout=1.0,
    )

    camera_config = SimpleNamespace(
        save_images_individually=True, clip_length=None, record_only_after_action=False, fps=20.0
    )
    recorder = Recorder(
        source, camera_config, frames=workspace.frames, occupancy_supplier=occupancy_source, action_tracker=tracker
    )

    transition_writer = TransitionWriter(
        workspace.frames,
        _GRID_SIZE[1],
        _GRID_SIZE[0],
        _TOOL_DIMS_PX,
        recorder.mask_for_frame,
        experiment_meta={"material": "chickpeas", "surface": "glass"},
    )
    executor.register_completion_callback(transition_writer.on_action_completed)

    planner = SegPushPlanner(config, workspace, np.random.default_rng(42))

    storage_base = str(tmp_path / "recorded_data")
    runner = SessionRunner(
        robot=robot,
        source=source,
        occupancy_source=occupancy_source,
        planner=planner,
        executor=executor,
        recorder=recorder,
        tracker=tracker,
        workspace=workspace,
        tool_grip_checker=grip_checker,
        config=config,
        storage_base_dir=storage_base,
        experiment_name="integration_test",
        shutdown_event=shutdown_event,
        order_sink_router=order_sink_router,
        transition_writer=transition_writer,
        episode_budget=episode_budget,
        intervention_wait_seconds=0.05,
        max_reset_iterations=4,
        first_frame_timeout=3.0,
        on_state_transition=on_state_transition,
    )

    def teardown():
        recorder.stop_recording()
        source.stop(timeout=2.0)

    return runner, source, recorder, storage_base, teardown


def test_full_episode_lifecycle_dryrun(tmp_path):
    grip_checker = _StubGripChecker(default_ok=True)
    runner, source, recorder, storage_base, teardown = _build_environment(
        tmp_path, grip_checker=grip_checker, episode_budget=2
    )

    start = time.perf_counter()
    try:
        runner.run()
    finally:
        teardown()
    elapsed = time.perf_counter() - start
    assert elapsed < 55.0, f"integration run took too long: {elapsed:.1f}s"
    assert runner.state == EpisodeState.FINISHED
    assert runner.episodes_run == 2

    session_root = os.path.join(storage_base, "integration_test")
    assert set(os.listdir(session_root)) >= {"1", "2"}

    for episode_id in ("1", "2"):
        session_dir = os.path.join(session_root, episode_id)
        task_dir = os.path.join(session_dir, "task")

        images_dir = os.path.join(task_dir, "Images")
        bottom_images_dir = os.path.join(task_dir, "Bottom_Images")
        masks_dir = os.path.join(task_dir, "Masks")
        assert os.path.isdir(images_dir) and os.listdir(images_dir)
        assert os.path.isdir(bottom_images_dir) and os.listdir(bottom_images_dir)
        assert os.path.isdir(masks_dir) and os.listdir(masks_dir)

        states_json_path = os.path.join(task_dir, "states.json")
        states_jsonl_path = os.path.join(task_dir, "states.jsonl")
        assert os.path.exists(states_json_path)
        assert os.path.exists(states_jsonl_path)
        import json

        with open(states_json_path) as f:
            states_array = json.load(f)
        with open(states_jsonl_path) as f:
            jsonl_lines = [line for line in f if line.strip()]
        assert len(states_array) == len(jsonl_lines) > 0
        push_rows = [
            row
            for row in states_array
            if row.get("action") is not None
            and row["action"]["action_type"] == "move_xy"
            and row["action"]["is_planar_2d"]
        ]
        assert push_rows, "expected at least one states.json row tagged with a push action"

        orders_json_path = os.path.join(task_dir, "orders.json")
        assert os.path.exists(orders_json_path)
        with open(orders_json_path) as f:
            orders = json.load(f)
        assert orders
        for row in orders:
            assert {"order_type", "order_value", "time", "robot_reported_time", "frame_index"} <= set(row.keys())

        actions_json_path = os.path.join(task_dir, "actions.json")
        assert os.path.exists(actions_json_path)
        with open(actions_json_path) as f:
            actions_doc = json.load(f)
        actions = actions_doc["actions"]
        assert actions_doc["total_actions"] == len(actions)
        push_actions = [
            a for a in actions if a["action_type"] == "move_xy" and a["is_planar_2d"] and a["parent_id"] is None
        ]
        assert len(push_actions) == _N_PUSHES
        child_actions = [a for a in actions if a["parent_id"] is not None]
        assert child_actions, "expected at least one child (per-order) action"

        status_path = os.path.join(session_dir, "status.txt")
        with open(status_path) as f:
            assert f.read().strip() == "success"

        transitions_dir = os.path.join(session_dir, "transitions")
        data_path = os.path.join(transitions_dir, f"_{episode_id}_data.pt")
        config_path = os.path.join(transitions_dir, f"_{episode_id}_config.yaml")
        assert os.path.exists(data_path), f"missing {data_path}"
        assert os.path.exists(config_path), f"missing {config_path}"

        data = torch.load(data_path)
        for key in ("masks_before", "masks_after", "p_starts_px", "p_stops_px", "angles"):
            assert key in data
        n = _N_PUSHES
        assert data["masks_before"].shape == (n, _GRID_SIZE[1], _GRID_SIZE[0])
        assert data["masks_after"].shape == (n, _GRID_SIZE[1], _GRID_SIZE[0])
        assert data["p_starts_px"].shape == (n, 2)
        assert data["p_stops_px"].shape == (n, 2)
        assert data["angles"].shape == (n,)
        for key in ("masks_before", "masks_after", "p_starts_px", "p_stops_px", "angles"):
            assert data[key].dtype == torch.float32
        assert float(data["masks_before"].min()) >= 0.0 and float(data["masks_before"].max()) <= 1.0

        with open(config_path) as f:
            cfg_doc = yaml.safe_load(f)
        assert cfg_doc["grid"] == {"height": _GRID_SIZE[1], "width": _GRID_SIZE[0]}
        assert cfg_doc["tool"]["size_px"] == list(_TOOL_DIMS_PX)
        assert set(cfg_doc["physics"].keys()) == {"friction", "density", "box_friction"}
        assert cfg_doc["experiment"]["material"] == "chickpeas"
        assert "date" in cfg_doc["experiment"]


def test_tool_lost_intervention_recovers(tmp_path):
    grip_checker = _StubGripChecker(default_ok=True, script=[False])
    visited: List[EpisodeState] = []

    def _on_transition(_old, new):
        visited.append(new)

    runner, source, recorder, storage_base, teardown = _build_environment(
        tmp_path, grip_checker=grip_checker, episode_budget=1, on_state_transition=_on_transition
    )
    start = time.perf_counter()
    try:
        runner.run()
    finally:
        teardown()
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0, f"intervention-recovery run took too long: {elapsed:.1f}s"

    assert EpisodeState.INTERVENTION in visited
    assert runner.state == EpisodeState.FINISHED
    assert runner.episodes_run == 1
    assert grip_checker.calls >= 3  # first CheckToolGrip (fail) + RegraspTool re-check (ok) + next CheckToolGrip (ok)

    session_root = os.path.join(storage_base, "integration_test")
    session_dirs = sorted(os.listdir(session_root))
    assert session_dirs == ["1"]
    status_path = os.path.join(session_root, "1", "status.txt")
    with open(status_path) as f:
        assert f.read().strip() == "success"
