# Codebase Documentation — Granular-Manipulation Stack

Living documentation of the refactored data-collection codebase. Design rationale lives in
[`../design/`](../design/README.md); this directory documents the code **as implemented**.
Conventions for contributors/agents: [CONVENTIONS.md](CONVENTIONS.md).

## Module index

| Doc | Package | Status | One-liner |
|---|---|---|---|
| [configuration.md](configuration.md) | `autograsper/config_schema.py` | implemented | Typed config schema, YAML loading, validation |
| [hardware.md](hardware.md) | `autograsper/hardware/` | implemented | RobotInterface protocol; CloudGripper + DryRun implementations |
| [observation.md](observation.md) | `autograsper/observation/` | implemented | Immutable Observation snapshots; single polling source; pub/sub |
| [perception.md](perception.md) | `autograsper/perception/` | implemented | Coordinate frames (Wave 2); occupancy contract + background-diff/YOLO providers + SegmentationWorker + tool-grip check (Wave 3a) |
| [planning.md](planning.md) | `autograsper/planning/` | implemented | WorldState/Plan/Primitive types; Workspace (fence walls, placement search, reset checks); RandomPushPlanner + SegPushPlanner |
| [execution.md](execution.md) | `autograsper/execution/` | implemented | Primitive→order expansion; safety guards; action tracking |
| [session.md](session.md) | `autograsper/session/` | implemented | Episode state machine; session runner; storage/dataset writers |
| [recording.md](recording.md) | `autograsper/recording/`, `autograsper/ui/` | implemented | Frame sink; MJPEG stream |
| [dataset_formats.md](dataset_formats.md) | (cross-cutting) | implemented | On-disk session layout, states/actions/orders formats, transition dataset |
| [testing.md](testing.md) | `autograsper/tests/` | implemented | How to run and extend the test suite |

Status values: `pending` → `implemented` → `verified` (integration-tested).

## Migration status

**Wave 6 (2026-07-20) relocated/removed the pre-refactor code.** The table above now describes
the only maintained runtime. Superseded granular-pipeline files (`coordinator.py`, `recording.py`,
`recording_seg.py`, `custom_graspers/granular_pusher.py`, `custom_graspers/fence_utils.py`,
`custom_graspers/random_push_grasper.py` (a byte-identical duplicate of `granular_pusher.py`),
`main_chickpeas*.py`, `object_tracker/granular_utils.py`, `object_tracker/tool_user_utils.py`,
the empty `frame_holder.py` stub, and the empty `granular_manipulation/` skeleton) were deleted
via `git rm` — full content is still in git history. Non-granular legacy task graspers and the
minimal legacy core they depend on were moved as-is to `autograsper/legacy/` — see
[`../legacy/README.md`](../legacy/README.md) for exactly what's there and how to run it, and
[`../design/02_proposed_architecture.md`](../design/02_proposed_architecture.md) §5's migration
map for the old→new correspondence that drove these decisions.
[`../design/IMPLEMENTATION_LOG.md`](../design/IMPLEMENTATION_LOG.md)'s final entry has the exact
before/after file lists.

### Legacy code and standalone tools with pre-move import paths

- `autograsper/legacy/` — unmaintained, pre-refactor graspers/core (see its own README). Not
  imported by anything under `autograsper/{hardware,observation,perception,planning,execution,
  session,recording,ui}/` or by `autograsper/tests/`.
- A few standalone, not-test-covered scripts/tools reference paths that moved or were deleted in
  this wave and were **deliberately left as-is** rather than "fixed" (they're offline
  tools/one-off scripts, not part of the tested runtime — see the Wave 6 log entry for the
  rationale):
  - `autograsper/recorder_profiler.py` — its `_measure_...` helper does `from library.utils import
    get_undistorted_bottom_image` inside a function body; `library/` moved to
    `autograsper/legacy/library/`, so this lazy import now raises `ModuleNotFoundError` if that
    code path is actually exercised.
  - `autograsper/legacy/main.py` (moved from `autograsper/main.py`) imports `from coordinator
    import DataCollectionCoordinator`; `coordinator.py` was deleted (superseded by
    `session/coordinator.py` + `session/episode.py`, which are not API-compatible), so this
    entry point cannot run without restoring a legacy `coordinator.py` from git history.
  - `image_collector/manual_control.py` imports `object_tracker.granular_utils` (deleted),
    `object_tracker.base_tool_tracker` (itself importing the same deleted module),
    `autograsper.custom_graspers.fence_utils` (deleted), and `autograsper.library.utils` (moved to
    `autograsper.legacy.library.utils`) — all now broken.
  - `image_collector/grab_tool.py` imports `object_tracker.tool_user_utils` (deleted) — broken.
  - `object_tracker/base_tool_tracker.py` / `base_tool_tracker2.py` import
    `object_tracker.granular_utils` (deleted) — broken if imported (nothing in the maintained
    runtime imports either module; only `image_collector/manual_control.py` did).
  - `image_collector/camera_calibration.py` / `image_collector/collect_data.py` use bare
    (non-package-absolute) `from library....` imports that already depended on an
    externally-arranged `sys.path`/cwd predating this wave; that arrangement's target moved along
    with everything else in `autograsper/library/` → `autograsper/legacy/library/`.

## Related documents
- [`../design/01_current_architecture.md`](../design/01_current_architecture.md) — legacy system (historical reference)
- [`../design/02_proposed_architecture.md`](../design/02_proposed_architecture.md) — target architecture
- [`../design/03_segmenter_native_design.md`](../design/03_segmenter_native_design.md) — segmenter-native deltas
- [`../design/IMPLEMENTATION_LOG.md`](../design/IMPLEMENTATION_LOG.md) — decisions & uncertainties log
