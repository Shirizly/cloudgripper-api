# Implementation Conventions

Binding conventions for all work on the granular-manipulation refactor
(designs: [`../design/`](../design/README.md)). Every implementation task must follow these.

## Environment
- Python: `/home/alon/anaconda3/envs/cge/bin/python` (conda env `cge`; has cv2, numpy, yaml,
  flask, ultralytics/torch, pytest).
- Run tests from repo root:
  `cd /home/alon/Code/cloudgripper-api && /home/alon/anaconda3/envs/cge/bin/python -m pytest autograsper/tests -q`
- Entry point (when it exists): `python -m autograsper.main_granular` from repo root.

## Hard safety rule
**Never send commands to a real robot.** No test, example, `__main__` block, or default config
may instantiate `CloudGripperRobot` against the network. All testing uses `DryRunRobot`
(validates + simulates). The real implementation is exercised only by the user, deliberately.
Tests must also not require GPU or network (YOLO model-loading tests: skip unless weights file
exists AND an env var `RUN_MODEL_TESTS=1` is set).

## Code style
- Package-absolute imports only: `from autograsper.hardware.dryrun import DryRunRobot`,
  `from client.cloudgripper_client import GripperRobot`. **No `sys.path` manipulation** in
  library code (a root `conftest.py` handles test paths).
- New code must not import the legacy modules (`grasper.py`, `coordinator.py`, `recording.py`,
  `recording_seg.py`, `custom_graspers/*`, `library/utils.py`) — port logic by copying and
  adapting, citing the source file in the module docstring.
- Type hints everywhere; `@dataclass(frozen=True)` for value/message types.
- `logging.getLogger(__name__)`, never `print`, in library code.
- No `cv2.imwrite`/`cv2.imshow` side effects in library code. Debug artifacts go through the
  `DebugSink` (`autograsper/observation/debug.py`) which is a no-op unless enabled and writes
  only under the session directory.
- Docstrings state responsibility, threading expectations, and units/coordinate frame of every
  geometric quantity (`robot` normalized [0,1]², `full_px`, `crop_px`, `grid` — see
  `docs/perception.md` once written).

## Documentation duty (part of every task's definition of done)
- Update the layer doc `autograsper/docs/<layer>.md`: purpose, public API, data flow in/out,
  file/data formats, threading model, config keys consumed, porting notes (source legacy file,
  intentional behavior changes).
- Keep the index table in `autograsper/docs/README.md` current.
- Docs describe the code as it IS, not as planned.

## Uncertainty log
When a decision isn't dictated by the design docs or legacy code, decide independently and
append an entry to `autograsper/design/IMPLEMENTATION_LOG.md`:
`## YYYY-MM-DD — <topic>` + what was ambiguous, decision, rationale. Never block on it.

## Testing
- pytest, tests in `autograsper/tests/test_<layer>_*.py`; fast (<~10 s each), deterministic
  (seed numpy RNG), no network/GPU/robot.
- Synthetic images/masks generated in-test or tiny fixtures under `autograsper/tests/fixtures/`.
- Every layer ships with tests exercising its public API through `DryRunRobot` or synthetic data.
