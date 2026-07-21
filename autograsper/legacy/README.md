# Legacy code (unmaintained)

Everything under `autograsper/legacy/` is **pre-refactor** code, moved here as-is during Wave 6
cleanup. It targets the **old** `AutograsperBase`/`DataCollectionCoordinator` API, not the new
layered architecture (`autograsper/{hardware,observation,perception,planning,execution,session,
recording,ui}/`).

**Status: unmaintained.** Nothing here is covered by `autograsper/tests/`, nothing here is
imported by the new architecture, and no further development is expected against this API. It is
kept only for reference and in case some of its task-specific behavior (backgammon, stacking,
evaluation, tool-use, calibration, ...) is worth re-porting onto the new architecture later, the
way `custom_graspers/granular_pusher.py`'s random-push behavior already was (see
`autograsper/design/02_proposed_architecture.md` §5's migration map for the granular-pipeline
side of that story).

For context on why this split happened and what the old system looked like, see
`autograsper/design/01_current_architecture.md`. For the current, maintained architecture, see
`autograsper/docs/` (start at `autograsper/docs/README.md`).

## What's here

- `custom_graspers/` — non-granular task graspers (`backgammon_grasper.py`,
  `calibrate_grasper.py`, `evaluation_grasper.py`, `example_grasper.py`, `manual_grasper.py`,
  `random_grasping_task.py`, `stacking_autograsper.py`, `subtask1Grasper.py`, `tool_user.py`).
  Each subclasses `AutograsperBase` (`legacy/grasper.py`) and none of them depend on the deleted
  granular-pipeline files (`coordinator.py`, `recording.py`, `recording_seg.py`,
  `custom_graspers/granular_pusher.py`, `custom_graspers/fence_utils.py`).
- `grasper.py` — `AutograsperBase`/`RobotActivity`, the base class every grasper above needs.
- `action_tracker.py` — the original (unextended) action tracker `grasper.py` uses. Independent
  from, and NOT the same module as, the new `autograsper/execution/actions.py` (a ported and
  extended copy) — both exist and are maintained/used separately.
- `file_manager.py` — `FileManager`, legacy on-disk directory layout helper.
- `utils.py` — `load_config` (YAML loader used by the legacy entry point below).
- `library/` — the legacy camera/calibration/color-tracking helper package
  (`utils.py`, `calibration.py`, `Camera2Robot.py`, `object_tracking.py`, `rgb_object_tracker.py`,
  `rgb_picker.py`, `bottom_image_preprocessing.py`, `color_config.ini`) that `grasper.py` and the
  graspers above import.
- `main.py` — the legacy Flask/MJPEG entry point that wired `DataCollectionCoordinator` to
  `backgammon_grasper`/`calibrate_grasper`. **Known broken import**: it imports
  `from coordinator import DataCollectionCoordinator`, and `coordinator.py` was deleted in this
  same cleanup wave (fully superseded by `autograsper/session/coordinator.py` + `session/
  episode.py`, which are not API-compatible with the old queue-based coordinator). Kept for
  reference only; would need a legacy `coordinator.py` restored from git history
  (`git log -- autograsper/coordinator.py`) to run again.
- `config.yaml`, `config.ini`, `backgammon-config.yaml` — configs for the graspers/entry point
  above (see `autograsper/design/IMPLEMENTATION_LOG.md`'s Wave 6 entry for which configures what).

## Running any of this

These files still use the pre-refactor style: bare, non-package-absolute imports
(`from grasper import ...`, `from library.utils import ...`) that resolve only when
`autograsper/legacy/` itself is on `sys.path` (for `grasper`, `action_tracker`, `library.*`) *and*
the repo root is also on `sys.path` (for `client.cloudgripper_client` and, for `tool_user.py`,
`object_tracker.ShapeAwarePoseEstimator`) — exactly the same two-roots requirement these files had
before the move (when `autograsper/` played the role `autograsper/legacy/` plays now). Verified
directly: `PYTHONPATH=<repo_root>:<repo_root>/autograsper/legacy python -c "import grasper"` (and
each grasper module) succeeds unmodified. They are **not** wired into `conftest.py`'s
package-absolute import setup and are not expected to work via `python -m autograsper...`.
