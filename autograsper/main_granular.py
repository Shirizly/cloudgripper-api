"""Composition root for the granular-manipulation data-collection stack (Wave 5).

Design reference: `autograsper/design/02_proposed_architecture.md` §2 (`main_granular.py #
composition root (one main; variant chosen by config)`) and §5's migration map
(`main_chickpeas*.py -> main_granular.py (variant via perception.provider config)`).

Usage (see `docs/testing.md` for more):
```
python -m autograsper.main_granular --config autograsper/granular-config.yaml --robot dryrun
python -m autograsper.main_granular --config <path> --robot real --episodes 5   # REAL ROBOT
python -m autograsper.main_granular --config <path> --no-ui
```

**Hard safety rule** (`docs/CONVENTIONS.md`): `--robot` defaults to `dryrun`. `CloudGripperRobot`
(the real-hardware backend) is constructed only when `--robot real` is passed explicitly — never by
a default, and never anywhere in this module's importable surface (`build_components`/`parse_args`
have no side effects; only `main()`, guarded by `if __name__ == "__main__":`, can reach the real
branch, and only when the caller opts in on the command line).

This module builds every layer (`hardware` -> `observation` -> `perception` -> `planning` ->
`execution` -> `session`/`recording`/`ui`) from one `config_schema.Config`, wires the failure/
lifecycle policy into a `session.coordinator.SessionRunner`, installs `SIGINT`/`SIGTERM` handlers
that set the shared `shutdown_event`, and joins everything cleanly on exit.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from typing import Any, Dict, Optional

import numpy as np

from autograsper.config_schema import Config, load_config
from autograsper.execution.actions import ActionTracker
from autograsper.execution.executor import Executor
from autograsper.execution.safety import SafetyValidator
from autograsper.hardware.dryrun import DryRunRobot
from autograsper.observation.debug import DebugSink
from autograsper.observation.source import ObservationSource
from autograsper.perception.frames import CameraPipeline, CoordinateFrames
from autograsper.perception.tool_grip import ToolGripChecker
from autograsper.planning.planner import FreshnessPolicy
from autograsper.planning.random_push_planner import RandomPushPlanner
from autograsper.planning.seg_push_planner import SegPushPlanner
from autograsper.planning.workspace import Workspace
from autograsper.recording.recorder import Recorder
from autograsper.session.coordinator import SessionRunner
from autograsper.session.storage import OrderSinkRouter, TransitionWriter

logger = logging.getLogger(__name__)


def _build_robot(config: Config, robot_mode: str):
    """`DryRunRobot` unless `robot_mode == "real"` — the ONLY place `CloudGripperRobot` may be
    constructed, and only reachable via an explicit `--robot real` CLI flag (never a default)."""
    if robot_mode == "real":
        from autograsper.hardware.cloudgripper import CloudGripperRobot

        logger.warning(
            "main_granular: --robot real -- constructing CloudGripperRobot. Commands WILL be "
            "sent to a physical robot (idx=%s).",
            config.robot.idx,
        )
        return CloudGripperRobot(config.robot)
    if robot_mode != "dryrun":
        raise ValueError(f"_build_robot: unknown robot_mode {robot_mode!r}")
    return DryRunRobot(robot_idx=config.robot.idx)


def _build_occupancy(
    config: Config,
    frames: CoordinateFrames,
    source: ObservationSource,
    shutdown_event: threading.Event,
    debug: DebugSink,
):
    """Returns `(occupancy_worker, occupancy_provider)` — exactly one is non-`None` (or both are
    `None` for `perception.provider == "none"`). `occupancy_worker` is a `SegmentationWorker`
    (constructed but NOT started -- the caller starts it once `source` is running);
    `occupancy_provider` is a directly-callable `OccupancyProvider` for the cautious pipeline."""
    provider_name = config.perception.provider
    if provider_name == "yolo":
        from autograsper.perception.yolo_segmenter import SegmentationWorker, YoloOccupancyProvider

        provider = YoloOccupancyProvider(frames, config.perception.yolo, debug=debug)
        worker = SegmentationWorker(provider, source, shutdown_event)
        return worker, None
    if provider_name == "background_diff":
        from autograsper.perception.background_diff import BackgroundDiffProvider

        provider = BackgroundDiffProvider(frames, config.perception.background_diff, debug=debug)
        return None, provider
    return None, None


def build_components(
    config: Config,
    *,
    robot_mode: str = "dryrun",
    enable_ui: bool = True,
    episode_budget_override: Optional[int] = None,
    shutdown_event: Optional[threading.Event] = None,
    debug: Optional[DebugSink] = None,
) -> Dict[str, Any]:
    """Wire every layer together from `config`. Pure composition: object construction only, no
    threads started and no I/O beyond reading calibration files `CoordinateFrames`/providers need
    at construction time. The caller (`main()`, or a test) starts `source`/`occupancy_worker`/
    `ui_server`/`runner` itself.

    Returns a dict of every component a caller might need to start/stop/inspect: `config`, `robot`,
    `source`, `workspace`, `planner`, `executor`, `recorder`, `runner`, `ui_server`,
    `occupancy_worker`, `shutdown_event`.
    """
    shutdown_event = shutdown_event if shutdown_event is not None else threading.Event()
    debug = debug if debug is not None else DebugSink()

    robot = _build_robot(config, robot_mode)
    camera = CameraPipeline.from_config(config.camera)
    source = ObservationSource(robot, camera, fps=config.camera.fps, shutdown_event=shutdown_event)

    frames = CoordinateFrames.from_config(config.workspace, config.perception)
    workspace = Workspace(config.workspace, frames)

    occupancy_worker, occupancy_provider = _build_occupancy(config, frames, source, shutdown_event, debug)
    occupancy_source = occupancy_worker if occupancy_worker is not None else occupancy_provider

    rng = np.random.default_rng()
    if config.perception.provider == "yolo":
        planner = SegPushPlanner(config, workspace, rng, debug=debug)
    else:
        planner = RandomPushPlanner(config, workspace, rng, debug=debug)

    tracker = ActionTracker()
    safety = SafetyValidator(workspace)
    freshness = FreshnessPolicy(config.perception.freshness_require_zero_for)
    tool_grip_checker = ToolGripChecker(config.tool_check, debug=debug) if config.tool_check.enabled else None
    order_sink_router = OrderSinkRouter()

    executor = Executor(
        robot,
        source,
        tracker,
        safety,
        freshness,
        workspace,
        config,
        occupancy_supplier=occupancy_worker,
        tool_grip_checker=tool_grip_checker,
        order_sink=order_sink_router,
        debug=debug,
        shutdown_event=shutdown_event,
    )

    recorder = Recorder(
        source, config.camera, frames=frames, occupancy_supplier=occupancy_worker, action_tracker=tracker
    )

    transition_writer = None
    if config.storage.emit_transitions_online:
        transition_writer = TransitionWriter(
            frames,
            config.perception.grid.height,
            config.perception.grid.width,
            config.storage.tool_size_px_for_transitions,
            recorder.mask_for_frame,
            experiment_meta={"name": config.experiment.name},
        )
        executor.register_completion_callback(transition_writer.on_action_completed)

    episode_budget = (
        episode_budget_override if episode_budget_override is not None else config.experiment.episode_budget
    )

    runner = SessionRunner(
        robot=robot,
        source=source,
        occupancy_source=occupancy_source,
        planner=planner,
        executor=executor,
        recorder=recorder,
        tracker=tracker,
        workspace=workspace,
        tool_grip_checker=tool_grip_checker,
        config=config,
        storage_base_dir=config.storage.base_dir,
        experiment_name=config.experiment.name,
        shutdown_event=shutdown_event,
        order_sink_router=order_sink_router,
        transition_writer=transition_writer,
        debug=debug,
        episode_budget=episode_budget,
    )

    ui_server = None
    if enable_ui and config.ui.enabled:
        from autograsper.ui.stream import MJPEGServer

        ui_server = MJPEGServer(source, config.ui.port, shutdown_event)

    return {
        "config": config,
        "robot": robot,
        "source": source,
        "workspace": workspace,
        "planner": planner,
        "executor": executor,
        "recorder": recorder,
        "runner": runner,
        "ui_server": ui_server,
        "occupancy_worker": occupancy_worker,
        "shutdown_event": shutdown_event,
    }


def _install_signal_handlers(shutdown_event: threading.Event) -> None:
    def _handler(signum, _frame):
        logger.info("main_granular: received signal %s, requesting shutdown", signum)
        shutdown_event.set()

    try:
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError):
        # Not the main thread, or a platform without signal support (e.g. under some test
        # harnesses) -- shutdown_event.set() remains reachable via SessionRunner.stop()/KeyboardInterrupt.
        logger.debug("main_granular: could not install signal handlers", exc_info=True)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Granular-manipulation data-collection runner")
    parser.add_argument(
        "--config", default="autograsper/granular-config.yaml", help="path to a config_schema.yaml document"
    )
    parser.add_argument(
        "--robot",
        choices=("dryrun", "real"),
        default="dryrun",
        help="'real' requires explicit opt-in and instantiates CloudGripperRobot (sends live "
        "commands to a physical robot). Default: dryrun.",
    )
    ui_group = parser.add_mutually_exclusive_group()
    ui_group.add_argument("--ui", dest="ui", action="store_true", default=None, help="force-enable the MJPEG UI")
    ui_group.add_argument("--no-ui", dest="ui", action="store_false", help="force-disable the MJPEG UI")
    parser.add_argument(
        "--episodes", type=int, default=None, help="override experiment.episode_budget for this run"
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    config = load_config(args.config)
    shutdown_event = threading.Event()
    enable_ui = args.ui if args.ui is not None else config.ui.enabled

    components = build_components(
        config,
        robot_mode=args.robot,
        enable_ui=enable_ui,
        episode_budget_override=args.episodes,
        shutdown_event=shutdown_event,
    )
    _install_signal_handlers(shutdown_event)

    source = components["source"]
    runner = components["runner"]
    ui_server = components["ui_server"]
    occupancy_worker = components["occupancy_worker"]
    recorder = components["recorder"]

    source.start()
    if occupancy_worker is not None:
        occupancy_worker.start()
    if ui_server is not None:
        ui_server.start()

    runner.start()
    try:
        while not shutdown_event.is_set() and runner.is_alive():
            runner.join(timeout=0.5)
    except KeyboardInterrupt:
        logger.info("main_granular: KeyboardInterrupt, shutting down")
        shutdown_event.set()
    finally:
        shutdown_event.set()
        runner.join(timeout=15.0)
        if ui_server is not None:
            ui_server.stop(timeout=5.0)
        if occupancy_worker is not None:
            occupancy_worker.stop(timeout=5.0)
        recorder.stop_recording()
        source.stop(timeout=5.0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
