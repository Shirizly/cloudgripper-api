# Testing — running and extending the suite

Status: **implemented** (cross-cutting; formalizes what every wave's tests already followed).
See [CONVENTIONS.md](CONVENTIONS.md) for the binding rules this doc elaborates on.

## Running the suite

```
cd /home/alon/Code/cloudgripper-api
/home/alon/anaconda3/envs/cge/bin/python -m pytest autograsper/tests -q
```

As of Wave 5: **170 passed, 1 skipped** in ~8 seconds. The one skip is a YOLO model-loading smoke
test (`test_perception_yolo_segmenter.py`), gated on `RUN_MODEL_TESTS=1` **and** the weights file
existing (see "Model-gated tests" below) — neither is true in this repo by default.

Run a single layer's tests, e.g.:

```
/home/alon/anaconda3/envs/cge/bin/python -m pytest autograsper/tests/test_integration_dryrun.py -q
/home/alon/anaconda3/envs/cge/bin/python -m pytest autograsper/tests/test_execution_executor.py -q
```

Every layer doc ([hardware.md](hardware.md), [observation.md](observation.md),
[perception.md](perception.md), [planning.md](planning.md), [execution.md](execution.md),
[session.md](session.md), [recording.md](recording.md)) lists the exact `pytest` invocation for
its own test file(s) at the bottom of its "Testing" section.

## Environment

`/home/alon/anaconda3/envs/cge/bin/python` (conda env `cge`) is the only environment with the full
runtime stack: `cv2`, `numpy`, `yaml`, `ultralytics`/`torch`, `pytest`. The system Python lacks
`cv2` and cannot run anything in this package.

**Known gap** (logged in `design/IMPLEMENTATION_LOG.md`): `flask`/`werkzeug` are listed in this
doc's own environment description (above, and historically in `CONVENTIONS.md`) but are **not
actually installed** in the `cge` env as of Wave 5 (`import flask` raises `ModuleNotFoundError`).
This only affects `ui/stream.py::MJPEGServer`, which imports both lazily and is never constructed
by any test or by `main_granular.py --no-ui` (the tested/default path). Installing them would be
required before `--ui`/`config.ui.enabled: true` could actually serve a stream.

## Model-gated tests

`RUN_MODEL_TESTS=1` (an environment variable, unset/`0` by default) gates any test that would load
real YOLO weights (`image_collector/chickpeas_segmentation_best.pt`, not committed to this repo —
see `granular-config.yaml`'s `# PLACEHOLDER — calibrate path` comment). Such a test is skipped
unless **both** `RUN_MODEL_TESTS=1` is set **and** the weights file exists on disk; this keeps the
default suite runnable with no GPU and no proprietary model artifact, per CONVENTIONS.md's hard
safety rule ("YOLO model-loading tests: skip unless weights file exists AND `RUN_MODEL_TESTS=1`").

To run a model-gated test (requires the weights file, works on CPU but is slower):

```
RUN_MODEL_TESTS=1 /home/alon/anaconda3/envs/cge/bin/python -m pytest autograsper/tests/test_perception_yolo_segmenter.py -q
```

## Writing new tests

- One file per layer: `autograsper/tests/test_<layer>_<topic>.py`.
- Fast (<~10s each individually; the one exception is `test_integration_dryrun.py`, which is
  allowed up to ~60s total since it drives several real background threads end-to-end — in
  practice it runs in ~2.5s).
- Deterministic: seed `numpy` RNGs explicitly (`np.random.default_rng(<seed>)`, never the global
  RNG — planners/tests alike).
- **Never instantiate `hardware.cloudgripper.CloudGripperRobot`** — always `hardware.dryrun.
  DryRunRobot`. No test may require network, GPU (outside the model-gated exception above), or a
  physical robot.
- Prefer hand-wiring components with a synthetic homography (see any of
  `test_execution_executor.py`/`test_planning_planners.py`/`test_integration_dryrun.py`'s
  `_synthetic_H`/`_make_workspace` helpers) over loading a real `config_schema.Config` from YAML —
  this repo's checked-in `granular-config.yaml` has several `# PLACEHOLDER — calibrate` values
  (missing homography, reference image, YOLO weights) that a real calibration-dependent test would
  otherwise need to work around.
- Synthetic images/masks: generate in-test (e.g. `cv2.circle` on a zeroed array) or use tiny
  fixtures under `autograsper/tests/fixtures/` — never depend on a real camera capture.

## Running `main_granular.py` in dryrun

```
cd /home/alon/Code/cloudgripper-api
/home/alon/anaconda3/envs/cge/bin/python -m autograsper.main_granular --help
/home/alon/anaconda3/envs/cge/bin/python -m autograsper.main_granular \
    --config autograsper/granular-config.yaml --robot dryrun --no-ui --episodes 1
```

`--robot` defaults to `dryrun` (the hard safety rule: `--robot real` is required, explicitly, to
ever construct `hardware.cloudgripper.CloudGripperRobot`). Against the checked-in
`granular-config.yaml` template, the above command builds every layer successfully (config load,
`DryRunRobot`, `ObservationSource`, `CoordinateFrames` — a real `homography.npz` already exists at
the repo root and loads fine) and then raises a clear, actionable `FileNotFoundError` from
`BackgroundDiffProvider` (the default `perception.provider`) because
`reference_empty_plate.jpg` — one of the template's documented `# PLACEHOLDER — calibrate`
entries — doesn't exist. This is expected until that reference image (or, for
`perception.provider: yolo`, the YOLO weights file) is actually captured/calibrated; it is not a
bug in the composition root, which is exercised end-to-end up to exactly that point. `main_granular
.build_components(...)` is importable and callable with no side effects (no threads started, no
robot commands sent) until the caller explicitly starts the returned components — see
[session.md](session.md)/`main_granular.py`'s own docstring for the full wiring.

To smoke-test the composition root without needing real calibration files at all, construct
components by hand the way `test_integration_dryrun.py` does (synthetic homography, no reference
image/YOLO weights needed) rather than through `build_components`.
