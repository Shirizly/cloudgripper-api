# Implementation Log — decisions & uncertainties

Running log of decisions made during implementation that were not dictated by the design docs
(`design/01..03`) or recoverable intent from legacy code. Newest entries at the bottom.
Format: date — topic; ambiguity; decision; rationale.

## 2026-07-20 — Python environment
Repo has no committed lockfile/venv; system python lacks cv2. Conda env `cge` contains
cv2/numpy/yaml/flask/ultralytics/torch — evidently the working env. **Decision:** standardize on
`/home/alon/anaconda3/envs/cge/bin/python`; installed `pytest` (9.1.1) into it as a dev
dependency. Rationale: only env with the full runtime stack; pytest addition is additive.

## 2026-07-20 — Fate of non-granular legacy graspers
User chose "replace old code" (delete superseded files). But `custom_graspers/` contains
non-granular behaviors (backgammon, stacking, evaluation, …) that import the legacy core
(`grasper.py`, `coordinator.py`) being deleted. **Decision:** superseded granular-pipeline files
are deleted; remaining legacy graspers + the minimal legacy core they need are moved to
`autograsper/legacy/` with a README stating they are unmaintained and target the old API.
Rationale: honors "replace" for the granular stack without silently leaving broken imports
scattered in the main tree; preserves reference code the user may still want.

## 2026-07-20 — Wave 1 (hardware + config schema): canonical OrderType
Design 02 §3.1 specifies `RobotInterface.set_gripper(opening: float)` but does not spell out an
order enum for the new `execution`/`hardware` boundary. Legacy `library/utils.py::OrderType` has
5 members (`MOVE_XY`, `MOVE_Z`, `GRIPPER_CLOSE`, `GRIPPER_OPEN`, `ROTATE`). **Decision:** collapse
to 4 canonical members (`MOVE_XY`, `MOVE_Z`, `ROTATE`, `GRIPPER`), with `GRIPPER` carrying a single
`opening` value in [0, 1] (0 = closed, 1 = open), matching `set_gripper`'s signature exactly.
Legacy's "stepped cautious close" (0.30→0.24 sequence) is preserved as an opt-in keyword
(`CloudGripperRobot.set_gripper(..., stepped_close=True)`) rather than a distinct order type,
since it is a hardware-layer implementation detail, not a distinct semantic command. Rationale:
one order type per `RobotInterface` method keeps `Order.validate()` and the dispatch mixin a
simple 1:1 mapping; nothing upstream needs to distinguish "open" from "close" as different verbs
when both are just `set_gripper(x)`.

## 2026-07-20 — Wave 1: `RawObservation.timestamp` is wall-clock, not the API's raw string
The CloudGripper API's `getAllStates` response includes a `time_state` field that legacy recorded
verbatim (`recording.py`: `self.timestamp = data[3]`) — its exact semantics (server timestamp?
request latency?) are undocumented and it's a string. Design 02's `Observation.timestamp` (Wave 2)
is specified as `float`, implying numeric wall-clock semantics for staleness/pacing math.
**Decision:** `RawObservation.timestamp` is a local `time.time()` float captured when the
hardware-layer read completes, not a pass-through of the API's raw field. The API's per-command
`"time"` string is still preserved, unparsed, on `CommandReceipt.robot_reported_time` (that value
has clearer per-command semantics and legacy already treated it as an opaque pass-through, e.g.
`write_order`'s `order_time` field). Rationale: gives Wave 2's `ObservationSource` a numeric clock
it can actually do arithmetic on, without inventing a parser for an undocumented server field.

## 2026-07-20 — Wave 1: config validation — clip range violations never raise
Task spec text for `Order.validate()` is self-contradictory on its face ("raising
OrderValidationError on wrong arity/types/ranges" vs. immediately describing range violations as
"clip silently... but log a warning when clipped"). **Decision:** only arity/type/rotation-cast
problems raise `OrderValidationError`; xy/z/gripper range violations are always clipped-with-warning,
never raised, matching legacy `execute_order`'s `np.clip` (which never raised either) and the
explicit clipping-policy sentence in the task. Rotation has no "range" to violate — it wraps via
`% 360` instead of clipping.

## 2026-07-20 — Wave 1: perception.background_diff / perception.yolo are conditionally required
Design 02 §4 lists both sub-sections under `perception` without saying whether one is optional
when the other provider is selected. **Decision:** only the sub-section matching the active
`perception.provider` is strictly required (missing/invalid keys reported); the inactive one, if
present in the document, is parsed leniently with its own defaults substituted for any missing
sub-key, so a config can carry settings for both providers without one blocking validation of the
other. Rationale: makes switching `provider` in an existing config a one-line edit instead of
requiring the unused section to be fully valid too.

## 2026-07-20 — Wave 2: `RobotState.from_legacy_dict` — tolerant, not raising
Task spec text was explicitly undecided ("tolerant of missing keys -> None fields or raise?
decide and log"). **Decision:** tolerant — any missing/non-numeric legacy key becomes `None` in
the corresponding `RobotState` field, never raises, with a `logging.warning` naming exactly which
key(s) were affected. Rationale: `ObservationSource` already has its own consecutive-failure
policy for poll cycles that raise outright (design 02 §3.5's "separated failure policy"); a
partially-populated state dict (e.g. a transient API hiccup that fills images but drops one state
key) shouldn't itself count as a fatal cycle failure when the rest of the `Observation` is usable —
downstream consumers that need a specific field present can check for `None` explicitly.

## 2026-07-20 — Wave 2: `perception.frames.CoordinateFrames.crop` axis order
`object_tracker/granular_utils.py::crop_center_region`'s own call signature
(`crop_center_region(image, crop_size=(360, 360), crop_center=(275, 200))`) unpacks as `crop_h,
crop_w = crop_size` / `center_y, center_x = crop_center` — i.e. legacy's *argument names* imply
`(h, w)`/`(y, x)` order for the exact same numeric values `granular-config.yaml` copies verbatim
into `perception.crop.size`/`.center_px`. But Wave 1's `config_schema.py` docstring/
`docs/configuration.md` table already documents `perception.crop.center_px`/`.size` as `(x, y)`/
`(w, h)` order without discussing this mismatch. **Decision:** `CoordinateFrames.crop` follows
`config_schema`'s already-committed `(x, y)`/`(w, h)` order (i.e. treats the checked-in `[275,
200]` as `x=275, y=200`, not legacy's intended `y=275, x=200`). Rationale: this is exactly the
"one definition, calibrated once against the homography" design 03 §2.1 asks for to resolve
design 01 defect #10 (two competing, differently-conventioned crops); picking a side and
documenting it beats trying to reverse-engineer which of two mutually-inconsistent legacy call
sites was "more correct". Whoever redoes the homography/crop calibration (design 03 §7.2, risk
register item 2) should calibrate against this axis order, not legacy's.

## 2026-07-20 — Wave 2: `CameraPipeline` — fisheye-only undistort branch
`library/calibration.py::undistort` has two branches selected by `len(D)`: a fisheye branch
(`D` has 4 coefficients) and a pinhole/`cv2.undistort` branch (otherwise). Every checked-in
distortion-coefficient array (`autograsper/config.yaml`, `autograsper/granular-config.yaml`) has
exactly 4 entries. **Decision:** port only the fisheye branch into
`perception.frames.CameraPipeline._undistort`; the pinhole branch is dead code for this camera and
is not carried over. Rationale: avoids maintaining and testing a code path with no reachable
config in this repo; if a future camera needs the pinhole branch, port it then with a real
distortion array to test against.

## 2026-07-20 — Wave 2: `ObservationSource` imports `perception.frames.CameraPipeline`
Design 02 §2's informal package-layout arrow list reads `... -> planning -> perception ->
observation` ("arrows = may import"), i.e. `perception` may import `observation`, not the reverse.
But design 02 §3.2's prose is explicit and specific: "`ObservationSource` runs the single FPS-paced
polling loop ..., applies the camera pipeline (undistort + homography rectify from
`perception.frames`)" — describing exactly the composition this task's deliverable spec also
mandates (`ObservationSource.__init__(robot, camera: CameraPipeline, ...)`). **Decision:** treat
§3.2's explicit prose as binding over the general layout diagram for this one pair of modules;
`observation/source.py` imports `perception/frames.py`. Rationale: the layout diagram's arrows are
a summary, and the one place the design doc is unambiguous about a *hard* forbidden direction is
"the planner never imports `hardware` or `execution`" — it does not make an equivalent explicit
claim about observation/perception. Noted in `docs/observation.md`'s porting notes so a later wave
doesn't "fix" this into a circular-import problem by trying to invert it.

## 2026-07-20 — Wave 2: `ObservationSource` failure policy — thread exits, does not retry forever
Task spec said "after a configurable number of consecutive failures (default 5) raise/flag a
source-failed state ... do NOT call `shutdown_event.set()` yourself", leaving open whether the
poll loop should keep retrying after flagging failure. **Decision:** once
`max_consecutive_failures` is reached, set `.failed`/`.failed_event` and **return from the poll
thread** (it exits; `is_running()` becomes `False`), rather than continuing to hammer a robot that
has already failed 5 times in a row. Rationale: an exited thread plus a `True` `.failed` flag gives
the session layer (Wave 5+) an unambiguous signal ("this source is dead, decide what to do") rather
than a thread silently retrying forever in the background at the configured fps; the session layer
remains free to construct a fresh `ObservationSource` and call `start()` on it if a retry is
warranted.

## 2026-07-20 — Wave 3a: `background_diff.py` — dropped `create_occupancy_mask`'s unused `threshold` param
`object_tracker/granular_utils.py::create_occupancy_mask(image, reference_empty, threshold: int =
30)` accepts a `threshold` parameter that the function body never reads — the diff cutoff is
hardcoded to `255 * 0.11` regardless of what's passed. **Decision:** the ported
`_create_occupancy_mask(image, reference_empty)` in `perception/background_diff.py` drops the dead
parameter entirely rather than carrying forward a non-functional knob that would look configurable
but silently do nothing. Rationale: CONVENTIONS.md's "port by copying and adapting" allows fixing
dead/misleading legacy signatures; keeping a parameter that lies about what it does would be worse
than a clean break, and the 0.11 fraction itself is preserved verbatim as `_DIFF_THRESHOLD_FRACTION`
so the actual behavior is unchanged.

## 2026-07-20 — Wave 3a: `BackgroundDiffProvider` — reference image assumed already full_px-frame
Design 03 §6's config delta and legacy `RandomPushGrasper.__init__` both just `cv2.imread` the
reference path with no mention of what pipeline stage it was captured at. Tracing the legacy call
site (`custom_graspers/granular_pusher.py::update_mask_and_process` calls `process_image(bottom_img,
self.reference_image)` where `bottom_img` comes from `shared_state.latest_bottom_image`, which
`recording_seg.py::Recorder._update` already ran through `get_undistorted_bottom_image` — undistort
+ homography rectify — before storing) shows the *live* image `create_occupancy_mask` diffs against
is always post-pipeline. **Decision:** treat `perception.background_diff.reference_image_path` as
already being a `full_px`-frame capture (same pipeline stage as `Observation.bottom_image`), so
`BackgroundDiffProvider.__init__` crops it with the same `CoordinateFrames.crop` and diffs
crop-to-crop with no separate undistort step. Rationale: matches how the reference image would
actually have been produced if captured from the same live feed the rest of the system uses;
whoever recaptures `reference_empty_plate.jpg` for the new pipeline should save it post-pipeline,
not as a raw fisheye frame. If this assumption proves wrong in practice (shapes will visibly
mismatch, which now raises `PerceptionDegraded` rather than silently misbehaving), it's a one-line
fix at the reference capture step, not an architecture change.

## 2026-07-20 — Wave 3a: `YoloOccupancyProvider` — `min_instance_area_px`/`stats_conf_threshold` not in config schema
Design 03 §6 says "a `min_instance_area_px` safety filter remains under `perception.yolo`" and §7.3
proposes a `stats_conf_threshold` of 0.5, but Wave 1's `config_schema.YoloConfig` (already
implemented, out of Wave 3a's scope to modify) only carries `weights_path`/`conf_threshold`/
`iou_threshold`/`imgsz`. **Decision:** read both values via `getattr(yolo_config, name, default)`
(`min_instance_area_px` default 0 i.e. off, `stats_conf_threshold` default 0.5 per design 03 §7.3)
rather than blocking on a config-schema change outside this task's file boundary. Rationale: keeps
Wave 3a self-contained; a future config-schema update that adds these fields to `YoloConfig` is
picked up automatically with no code change in `yolo_segmenter.py`, and in the meantime the
documented defaults match the design doc's stated proposal exactly.

## 2026-07-20 — Wave 3a: `SegmentationWorker` — degraded thresholds (3 consecutive failures, 5 consecutive zero-detection results)
Design 03 §7.4 specifies the *existence* of a degraded signal ("if the segmenter output
degenerates ... raise `PerceptionDegraded`") but not numeric thresholds; the task spec explicitly
left the zero-detection-streak default open ("default from design: flag only, configurable").
**Decision:** `max_consecutive_failures=3` (matches the task spec's explicit default for the
raising case) and `zero_detection_streak_for_degraded=5` (chosen to be ~2s at the template's 2.5
FPS camera rate — long enough to not false-trigger on a single bad frame, short enough to catch a
real collapse within one push cycle). Both are keyword arguments on `SegmentationWorker.__init__`,
overridable per deployment; `zero_detection_streak_for_degraded=None` disables that trigger
entirely. Rationale: needed *some* default to ship a working worker; picked values proportionate to
the one committed timing parameter (camera FPS) rather than an arbitrary round number, and made
both configurable so a real deployment can tune them against observed false-positive/negative rates
without code changes.

## 2026-07-20 — Wave 3a: `SegmentationWorker` uses `ObservationSource.subscribe()`, not polling `latest()`
`docs/observation.md` explicitly lists "segmentation worker" as an intended `subscribe()` consumer
alongside the recorder and UI stream, but design 03 §2.1's prose only says "Subscribes to
`ObservationSource` (latest-wins ...)" without naming the specific API method. **Decision:**
`SegmentationWorker.start()` calls `source.subscribe(name, maxsize=1)` and blocks on the returned
`LatestWinsQueue.get(timeout=...)`, rather than busy-polling `source.latest()` in a loop. Rationale:
this is the API `observation.md` was written expecting perception workers to use, avoids inventing
a second latest-wins mechanism when one already exists and is tested, and blocks efficiently instead
of spin-waiting. A defensive `obs.seq <= last_processed_seq: skip` check is kept even though it
should be unreachable with a single-producer maxsize-1 subscription, per the task spec's explicit
"skips if seq already processed" requirement.
# Implementation Log — Wave 3b (planning layer)

Same format as `design/IMPLEMENTATION_LOG.md`; kept in a separate file for this wave since
`IMPLEMENTATION_LOG.md` was being written concurrently by another wave's work and the task
instructed not to edit shared files directly. Meant to be merged into `IMPLEMENTATION_LOG.md` by
the orchestrator. Newest entries at the bottom.

## 2026-07-20 — Wave 3b: `build_fence_walls` angle pairing — verified, not assumed

The task brief describing this wave suggested a test expectation of "top/bottom walls angle 0,
left/right 90" and asked to verify it "mentally or in scratch" before writing the test. Directly
executing `build_fence_walls`'s exact formula (`angle = (atan2(tangent.y, tangent.x) + pi/2) %
pi`, scaled to degrees, then wrapped into `[0, 180)`) against the template fence values
(`fence_center=(0.5, 0.49)`, `fence_size=(0.94, 0.955)`) gives **`top=90`, `right=0`, `bottom=90`,
`left=0`** — the opposite pairing from the brief's suggested expectation. **Decision:** the test
(`test_planning_workspace.py::test_build_fence_walls_angle_values_match_verified_legacy_math`)
asserts the numerically verified values, not the brief's suggested ones, since the task's own
explicit priority is "port EXACTLY — same math, same angle conventions" over any paraphrase of
what that math should produce. Rationale: verifying by direct computation (not informal reasoning
about "tangent along x-axis => angle should be 0") is exactly the kind of check CONVENTIONS.md's
port-faithfully rule exists to force — an intuitive-but-wrong expectation would have quietly
flipped which walls get which tool orientation.

## 2026-07-20 — Wave 3b: `Planner` implementations are structural, not nominal

Design 02 §3.3 defines `Planner` as a `typing.Protocol`. `RandomPushPlanner`/`SegPushPlanner` do
not inherit from it (no `class RandomPushPlanner(Planner):`) — they satisfy it purely by
implementing the same four methods with the same signatures. **Decision:** keep this structural;
do not add explicit inheritance. Rationale: matches the "duck typing over `OccupancyLike`" pattern
mandated for the cross-team perception contract in this same wave, and the shared contract's
Protocols throughout Waves 2-3 (`RobotInterface`, `CameraPipeline` in `ObservationSource`,
`OccupancyLike` here) are consistently structural — `Planner` should be no different, and Wave 5's
session layer is expected to type-hint against `Planner` without caring which concrete class it
receives.

## 2026-07-20 — Wave 3b: `FreshnessPolicy`'s primitive-to-category mapping

The task specified `FreshnessPolicy.requires_fresh(primitive)` should match "primitive class names
lowercased: `lower_tool`, `placement`, `wall_check`" against
`config.perception.freshness_require_zero_for`. Taken literally this is impossible: no primitive
class, lowercased (even snake-cased), produces the string `"placement"` (`PlaceTool` ->
`place_tool`) or `"wall_check"` (`SweepWall` -> `sweep_wall`) — these are semantic categories
("this primitive's parameters came from a placement search over the mask" / "... a wall-band
check"), not literal class-name transforms. **Decision:** implemented an explicit
`_FRESHNESS_CATEGORY` dict in `planning/planner.py`: `LowerTool -> "lower_tool"`, `PlaceTool ->
"placement"`, `SweepWall -> "wall_check"`; every other primitive (`Push`, `RefreshMask`,
`CheckToolGrip`, `RegraspTool`, `MoveTo`) has no category and `requires_fresh` returns `False` for
them unconditionally. Rationale: this is the natural one-to-one reading of design 03 §3.2's three
named mask-reading decisions ("placement search, wall band check, lower-tool guard") onto the
primitive set actually defined this wave; documented here rather than left as an undiscoverable
mismatch between the task text and the config's example values (`granular-config.yaml`'s
`freshness_require_zero_for: ["MOVE_XY", "ROTATE"]` is itself stale relative to design 03 §6's own
example `[lower_tool, placement, wall_check]` — a config template inconsistency out of this
wave's scope to fix, noted for whichever wave next edits `granular-config.yaml`).

## 2026-07-20 — Wave 3b: `check_wall_reset_needed`'s two dead legacy parameters

Legacy `custom_graspers/fence_utils.py::check_wall_reset_needed(mask, wall,
image_space_tool_dimensions, min_granule_size, margin_of_safety=0.03)` has **two** parameters
whose values are never referenced anywhere in the function body: `margin_of_safety` (accepted,
never read) and `min_granule_size` (accepted, never read — the actual size filtering happens
earlier, in `object_tracker/granular_utils.py::clean_occupancy_mask`, on a different mask
entirely). **Decision:** drop `margin_of_safety` from this port's signature (CONVENTIONS.md: "no
dead code" — carrying forward an unused parameter with no discoverable purpose is exactly what
that rule targets); keep `min_granule_size` (unused in the ported math, exactly like legacy)
because the Wave 3b task brief's own specified signature
(`check_wall_reset_needed(mask_crop, wall, tool_dims_px, min_granule_size, frames, debug=None)`)
explicitly includes it. Rationale: the task brief is itself the authority for the second
parameter's presence (perhaps reserved for a future size-filtering step at this layer); the first
has no such directive and is unambiguously legacy debris. Both are called out in the module
docstring and the `planning.md` porting notes so nobody "fixes" `min_granule_size` into doing
something by accident without re-reading this note.

## 2026-07-20 — Wave 3b: `check_wall_reset_needed`'s `frames` parameter and `details["pos_robot"]`

Legacy's `check_wall_reset_needed` returns only pixel-frame results (`details["pos_px"]`); the
caller (`RandomPushGrasper.sweep_wall`) does the `pix2robtrans.pix_to_robot(*pos_px)` conversion
itself using a `PixelRobotTransform` built in `__init__`. The Wave 3b task brief's specified
signature for the port adds a `frames` parameter with no further explanation of what it's for.
**Decision:** use `frames.crop_px_to_robot` inside `check_wall_reset_needed` to add
`details["pos_robot"]` (robot-frame xy) alongside the legacy `details["pos_px"]`, sparing both
planners a duplicate homography call. Rationale: `frames: CoordinateFrames` is only useful as a
parameter here if it's actually invoked; converting the one pixel result this function already
computes is the only sensible use, and it directly serves `_push_policy.py::build_sweep_primitive`
(which needs a robot-frame position to compute `get_pos_sweep_from_optimal`).

## 2026-07-20 — Wave 3b: `find_tool_placements` returns `None`, not legacy's `{}`

Legacy returns `{}` (empty dict) when no placement is found, but every call site
(`RandomPushGrasper.perform_task`'s `if tool_placement is None:` check!) actually tests for
`None`, which an empty dict never equals — legacy's own caller code is already inconsistent with
its own function's contract (an empty dict is falsy but `is None` is `False` for `{}`, so that
specific `if tool_placement is None:` check in legacy would never have been `True` for a "not
found" result; a `bool({})`-style truthiness check elsewhere in the same file
(`check_wall_reset_needed`'s `if isinstance(free_region, list) or not free_region:`) uses
truthiness instead, working by accident). **Decision:** `find_tool_placements` returns `None` for
"not found", not `{}`, resolving the inconsistency in the ported callers'
(`_push_policy.py::find_placement_pose`, `workspace.py::check_wall_reset_needed`'s internal call)
favor of the check that was actually semantically intended.

## 2026-07-20 — Wave 3b: `Push`'s explicit `(start, angle, end)` re-expression of legacy's chaining

See `planning/_push_policy.py::sample_pushes`'s docstring for the full reasoning: legacy interleaves
`MOVE_XY(x, y)` (push at the *previous* orientation) then `ROTATE(orientation)` (preparing the
*next* push's angle) in its per-iteration loop, so "push *i*'s contact angle" and "iteration *i*'s
sampled orientation" are off by one index. **Decision:** re-express each push as a self-contained
`Push(start_x, start_y, angle, end_x, end_y, height)` value where the sampled angle for iteration
*i* is stored as *that* push's angle (not the next one's), with `start = previous push's end`
chaining consecutive pushes. Rationale: the design 02 §3.3 primitive table specifies `Push` as
one atomic value with a single `angle` field (needed for the transition dataset's `tool_start_px,
tool_stop_px, angle` triple per design 03 §5) — there is no legacy-faithful way to encode "this
push's angle is actually last iteration's sample" as a single immutable value without leaking
planner-internal iteration state into the primitive, which the design explicitly wants primitives
to be free of. The physical command sequence sent to the robot is unaffected in aggregate (same
multiset of `(x, y, orientation)` triples, same order), and rotation happening in-place at a fixed
`grasp_height` means the safety/hazard profile is identical either way.

## 2026-07-20 — Wave 3b: `SweepWall.step` sampled once per wall, not once per pass

Legacy `custom_graspers/granular_pusher.py::sweep()` recomputes `reset_step_size * (1 + 0.5 *
np.random.rand())` on every call — i.e. a fresh jittered step for each of the 3 `t`-value passes
per wall. `SweepWall` is a single immutable dataclass value (one per wall-sweep decision, per the
task's specified primitive shape with a single `step: float` field), so there is no way to encode
"a different jittered step per pass" without either (a) making `step` a tuple (deviating from the
task's specified field type) or (b) pushing the jitter into the executor (which would mean the
executor owns a source of randomness, contradicting "planners own randomness, execution is a pure
re-actor of planner decisions"). **Decision:** sample one jittered `step` per `SweepWall` (from
the planner's own `rng`), reused across all of that wall's `t`-value passes. Rationale: the
practical effect is a slightly narrower variance in push distance across a wall's 3 passes
(instead of 3 independent samples, 1 sample used 3 times) — the sampled range and its safety
implications (never larger than legacy's own jittered range) are unchanged; documented as a
logged, deliberate simplification per the task's own instruction to flag such differences rather
than silently resolve them.

## 2026-07-20 — Wave 3b: `RandomPushPlanner.plan_reset` batches all walls against one snapshot

Legacy `reset_task()` calls `update_mask_and_process()` (a real mask recapture) **between** wall
checks within the same loop, so each wall's `check_wall_reset_needed` call sees a mask reflecting
every prior wall's just-completed sweep in that same reset pass. A pure `plan_reset(w:
WorldState) -> Plan` function cannot replicate this — it has exactly one `WorldState` snapshot to
decide from. **Decision:** `RandomPushPlanner.plan_reset` evaluates every wall's
`check_wall_reset_needed` against the *same* `w.occupancy.crop_mask`, in `rng`-shuffled order,
still emitting a `RefreshMask` after each `SweepWall` (as a hint for whatever comes *next*, not to
feed back into this call). **Contrast:** `SegPushPlanner.plan_reset` does NOT do this — it emits
only one wall's `SweepWall` per call, tagged `meta["replan_after_each"] = True`, explicitly asking
the session layer to re-derive `WorldState` and call `plan_reset` again before deciding the next
wall (see `docs/planning.md`'s "Replanning contract" section). Rationale for the asymmetry: for
the background-diff pipeline, actually refreshing a mask mid-plan would cost another ~10s robot
motion per wall — legacy already accepted a same-episode staleness compromise for exactly this
reason (this is why the design 03 rewrite exists at all); for the segmenter-native pipeline, a
fresh mask is nearly free, so there is no efficiency reason to ever plan more than one wall against
a snapshot that's about to go stale. Any residual over/under-sweep from the batched
`RandomPushPlanner` path self-corrects on the next `needs_reset` check, matching legacy's own
per-episode (not per-wall) convergence granularity.

## 2026-07-20 — Wave 3b: shared push/reset policy math factored into `planning/_push_policy.py`

`RandomPushPlanner` and `SegPushPlanner` are nearly identical policies (design 03 §3: "mostly
identical... minus the caution choreography"); the task's file list (section F) only named the two
planner modules plus `types.py`/`workspace.py`/`planner.py`, but implementing the shared placement
search / push sampling / sweep-primitive construction logic twice would have meant every future
fix to that math needing two edits. **Decision:** added one small internal module,
`planning/_push_policy.py` (leading underscore: not part of the public planning API, an
implementation-sharing detail between the two planner modules), holding
`find_placement_pose`/`sample_pushes`/`build_sweep_primitive` plus the shared numeric defaults
(`DEFAULT_RESET_STEP_SIZE`, `DEFAULT_MIN_GRANULE_SIZE`, `DEFAULT_MIN_CLEARANCE_PX`). Rationale:
minimal, well-justified addition beyond the literal deliverable list — reduces duplication risk
without changing either planner's public shape or behavior.

## 2026-07-20 — Wave 4 (execution layer): `execution` imports `planning.types`/`planning.workspace`

Design 02 §2's informal dependency-arrow summary (`hardware ← execution ← session → planning →
perception → observation`) reads as "execution may not import planning". Taken literally this is
unworkable: the executor's entire job (design 02 §3.4) is to consume `planning.types.Primitive`
subclasses and expand them into orders using the exact geometry the planner used to build them
(`planning.workspace.make_tool_mask`/`check_placement` for the `LowerTool` guard,
`planning.workspace.sample_tool_pose` for `SweepWall`'s per-pass poses). **Decision:**
`execution/executor.py` and `execution/safety.py` import `autograsper.planning.types` (primitive
dataclasses, pure value types) and `autograsper.planning.workspace` (pure functions/`Workspace` —
no planner classes, no `random_push_planner`/`seg_push_planner` imports). Rationale: duplicating
this geometry in `execution/` would violate "one definition, calibrated once" for exactly the
reason design 03 §2.1 gives for `CoordinateFrames`, and two precedents already exist for treating
specific prose over the general diagram: Wave 2's `observation/source.py` importing
`perception/frames.py`, and Wave 3b's `planning/workspace.py` importing
`perception/frames.py::CoordinateFrames`. The one arrow design 02 explicitly calls a hard rule
("the planner never imports `hardware` or `execution`") is respected — nothing in `planning/`
imports anything from `execution/`.

## 2026-07-20 — Wave 4: `PlanResult.error` is always `None`; typed failures raise instead

The task spec asks for `PlanResult(completed, aborted, error: Exception|None)` from `run_plan`, but
also says "raise on typed failures after closing actions cleanly" — self-tensioned if `error` were
meant to carry a captured exception on the same call that returns normally. **Decision:**
`run_plan`/`execute_primitive` always raise (never return) on any failure — a typed
`ExecutionError` subclass or an unexpected exception — after closing the in-flight `Action`(s)
cleanly via `end_action()`. `PlanResult` is only returned for the two non-failure outcomes: full
completion (`aborted=False`) and a clean stop because `shutdown_event` was observed set *between*
primitives (`aborted=True`) — `error` is `None` in both. Rationale: keeps the typed-exception
vocabulary (design 02 §3.5) meaningful for the session layer's retry/replan/intervention dispatch
(`except UnsafeLower: ...` is a much better session-layer ergonomic than
`if result.error is not None: isinstance(result.error, UnsafeLower)`); `error` is kept on
`PlanResult` anyway for forward compatibility and so "no error" is self-documenting at call sites
that only check the return value.

## 2026-07-20 — Wave 4: every order becomes a tracked child action, including gripper orders

Legacy `grasper.py::queue_orders` explicitly skips action tracking for gripper orders (`if
action_type is not ActionType.GRIPPER_CLOSE: start_action(...)`). **Decision:** every order sent
by `Executor._send_order` becomes a child `Action`, with no gripper exception. Rationale:
`RegraspTool`'s ported `perform_grab_tool` sequence includes two gripper orders (open, close) that
are just as much a part of that primitive's choreography as its moves/rotates; skipping them would
leave gaps in the child-action timeline for exactly the primitive whose "did the grasp actually
happen, and when" is most interesting to a dataset consumer. `OrderType.GRIPPER` (Wave 1's
collapsed single gripper order type) is mapped back to legacy's two-member split
(`ActionType.GRIPPER_OPEN`/`GRIPPER_CLOSE`) by opening value (`>= 0.5` → open) so the distinction
survives despite the order type itself no longer carrying it.

## 2026-07-20 — Wave 4: primitive → top-level `ActionType` mapping

Legacy/dataset tooling expects `ActionType` values from the existing enum; the task flagged this
tension directly ("Legacy dataset expects action_type values from the enum; add new ActionType
members PUSH and PLACE_TOOL etc. would break dataset compat — DECISION: keep legacy ActionType
values"). **Decision:** `Push` → `ActionType.MOVE_XY` with `is_planar_2d=True` (matching design 02
§3.3's "`Push` ... tracked as one `is_planar_2d` action"); `SweepWall` → `ActionType.SWEEP`
(already a legacy member for exactly this composite case); every other primitive (`MoveTo`,
`PlaceTool`, `LowerTool`, `RegraspTool`, `RefreshMask`, `CheckToolGrip`) → `ActionType.OTHER`.
`description` is `primitive.describe()`; `action_details` is `dataclasses.asdict(primitive)`.
Rationale: preserves every existing dataset consumer's ability to filter `action_type ==
"move_xy" and is_planar_2d` for pushes (unchanged query) while giving every other primitive a
recognizable, inspectable `action_details` blob without inventing new enum members.

## 2026-07-20 — Wave 4: `SafetyValidator` — boundary check scope and unknown-`current_z` default

Two related decisions for `execution/safety.py::SafetyValidator.validate_order`, neither dictated
verbatim by the design docs:
1. The manipulation-boundary check applies uniformly to every "low" `MOVE_XY` order, with no
   built-in exception for `SweepWall`'s approach/sweep passes near the fence walls (which sit
   outside a *tightly* configured manipulation boundary, e.g. the `(0.1, 0.9)` boundary borrowed
   from `test_planning_planners.py`'s fixtures). **Decision:** do not special-case `SweepWall`;
   instead, document that a real deployment's `workspace.manipulation_boundary_robot` must be sized
   to include the fence-wall reach (legacy's own comment calls `manip_x`/`manip_y` "hardcoded safe
   X,Y ranges, adjust to your setup" — i.e. this was always meant to be tuned per-deployment, not a
   fixed placement-search-only region). Rationale: `SweepWall` positions are already geometrically
   bounded by `build_fence_walls`'s `t_min`/`t_max` (fence extent minus half the tool length minus
   a safety margin), so a correctly configured manipulation boundary that covers the actual
   workspace should contain them without needing a bypass flag whose name (`allow_rack`) would be a
   semantic mismatch for a wall-sweep context. This wave's own executor tests use a wide `(0.0,
   1.0)` boundary for choreography tests (isolating "does the order sequence match" from "is this
   deployment's boundary sized correctly", which `test_execution_safety.py` already covers with a
   dedicated tighter-boundary fixture).
2. `current_z=None` (the caller didn't supply the z the robot is/will be at) is treated as `0.0` —
   i.e. "unknown, assume low, apply the boundary check." Rationale: conservative-by-default; `
   Executor` itself always tracks and passes the real last-commanded z, so `None` should only
   arise if a future caller invokes `SafetyValidator` directly without that context, in which case
   erring toward the stricter check is the safer failure mode.

## 2026-07-20 — Wave 4: `LowerTool` guard runs its own occupancy fetch independent of the generic freshness gate

Design 03 §3.2's freshness rule and §3.3's real-time `LowerTool` guard are related but distinct:
the former is a generic "don't act on a stale mask" policy across `LowerTool`/`PlaceTool`/
`SweepWall`; the latter is specifically "check the footprint is clear immediately before lowering."
**Decision:** `Executor._enforce_freshness_gate` runs once per primitive (generic, raises
`StaleMaskTimeout` on policy-required timeout, no-op otherwise); `Executor._expand_lower_tool`
*additionally* always performs its own occupancy fetch + `tool_footprint_clear` check whenever
`guarded=True` and `occupancy_supplier` is configured — regardless of whether the generic gate
already ran for this exact primitive. For a standalone guarded `LowerTool` with `"lower_tool"`
configured as freshness-required, this means two occupancy fetches per primitive execution (the
generic gate's, then the guard's own). Rationale: this is a deliberate, harmless duplication that
keeps `_expand_lower_tool` a single, always-correct, self-contained guard implementation — usable
identically whether reached as a standalone `LowerTool` or nested inside `PlaceTool`'s expansion
(where the approach motion has changed the pose since the generic gate ran, making a *second*,
pose-fresh check the actually-correct behavior, not redundant at all in that case). Conditionally
skipping the guard's own fetch when the generic gate "already handled it" would require threading
extra state between the two call sites for a marginal cost saving (one extra `latest()`/
`await_result()` call, not a robot motion) and was judged not worth the complexity.

## 2026-07-20 — Wave 4: `NeedsHumanHelp` defined but not raised by the executor

`execution/errors.py` defines `NeedsHumanHelp` per design 02 §3.5's typed-exception vocabulary, but
no `Executor` code path raises it in this wave. **Decision:** leave it unraised here; document it
as reserved for the session layer (Wave 5), which is expected to escalate a *pattern* of failures
(e.g. `RegraspTool` retried N times with `ToolLost` each time) into `NeedsHumanHelp` as part of its
own retry/intervention policy — a decision that requires state across multiple `Executor` calls
(a retry counter) that the executor itself, being a stateless-between-primitives expander, has no
natural place to hold. Rationale: matches design 02 §3.5's own framing ("only the session layer
decides between retry, human intervention, or abort") — `NeedsHumanHelp` is precisely a
session-level *decision*, not an execution-level *observation* (unlike `ToolLost`, which is a
direct, single-check observation the executor is well-positioned to raise itself).

## 2026-07-20 — Wave 5 (session/recording/storage/composition): `occupancy_source` is one
duck-typed constructor argument for both perception pipelines

The task brief described `SessionRunner`'s occupancy plumbing informally ("seg pipeline:
`worker.latest()`; cautious pipeline: last computed provider result held by runner") without
specifying whether this is one argument or two. **Decision:** a single constructor argument
(`occupancy_source`), distinguished by duck typing (`hasattr(x, "latest")` -> segmenter-native,
otherwise `hasattr(x, "compute")` -> cautious/background-diff, otherwise `None`). Rationale:
mirrors the "worker_or_provider" phrasing directly, and every other cross-layer seam in this
refactor (`RobotInterface`, `OccupancyProvider`, `Planner`) is already structural/duck-typed, not
nominal — an explicit `Union`-style pair of constructor kwargs would be the odd one out. The same
object is also what a caller passes to `Executor(occupancy_supplier=...)` for the seg-native case
(the executor's own freshness gate/`LowerTool` guard need the `SegmentationWorker`-shaped API);
for the cautious pipeline `Executor.occupancy_supplier` stays `None` (per `docs/execution.md`:
"this is the cautious/background-diff pipeline's mode... unguarded"), and only `SessionRunner`
holds the directly-callable provider.

## 2026-07-20 — Wave 5: `Recorder`'s mask-save "ahead" gate has a one-time bootstrap exception

Design 03 §4 says a mask is saved "if occupancy_supplier has a result not yet saved and its
source_seq <= current obs seq" — i.e. skip saving if the supplier is temporally *ahead* of the
frame currently being processed (avoids mislabeling a future mask under an earlier frame's
filename). Taken literally and applied uniformly, this stalls indefinitely whenever a fast
occupancy supplier is paired with a `Recorder` whose consumer thread hasn't had a chance to
process anything yet in a freshly (re)targeted directory (observed directly: `TransitionWriter`
reported an unresolvable "before" mask for the very first push of an episode in
`test_integration_dryrun.py` until this fix). **Decision:** the "ahead" check is only enforced
*after* at least one mask has already been saved in the current directory; the very first save is
unconditional. Rationale: guarantees `mask_for_frame`'s "before" lookup has *something* to resolve
against as early as possible in an episode (frame 0 or shortly after), while the steady-state
dedup/ahead semantics — which do matter for a real, possibly-lagging real segmenter — are
unchanged from frame 1 onward. Logged as a deliberate refinement of the literal design-doc
sentence, not a deviation from its intent.

## 2026-07-20 — Wave 5: `Recorder.wait_until_frame_processed` — closing a real finalize-time race

`Executor.execute_primitive`'s own end-of-primitive `source.await_next(...)` (per `docs/
execution.md`) only guarantees the last `Observation` of a plan *exists* on the shared
`ObservationSource` — it says nothing about whether `Recorder`'s independently-scheduled consumer
thread has already processed (written images/states/mask for) that specific observation by the
time `SessionRunner` calls `TransitionWriter.finalize()` right after `run_plan()` returns. This is
not a hypothetical: `test_integration_dryrun.py` reproduced it directly (the *last* push's "after"
mask was intermittently unresolvable because `_maybe_save_mask` for that frame hadn't run yet).
**Decision:** added `Recorder.wait_until_frame_processed(frame_index, timeout)` — a condition
variable the consumer thread notifies after every `_capture()` — and had
`SessionRunner._finalize_episode_writers` call it (bounded by `first_frame_timeout`) before
`TransitionWriter.finalize()`. Rationale: a small, targeted addition (not in the original per-wave
deliverable list) directly forced by an actually-observed race in this wave's own integration
test, per the task's explicit allowance to make minimal fixes to unblock genuine integration bugs
— logged here since the affected method (`Recorder`) is this same wave's own file, not a
cross-wave fix.

## 2026-07-20 — Wave 5: one `TransitionWriter` per run, `finalize()`d once per episode

`execution.actions.ActionTracker.register_completion_callback` has no unregister API (`docs/
execution.md`). Constructing a fresh `TransitionWriter` per episode and registering each one would
leave every earlier episode's writer permanently subscribed, silently accumulating later episodes'
completed `Push` actions into an already-`finalize()`d (and therefore never-flushed-again) pending
buffer — a slow, silent memory leak across a long run, not a correctness bug given the buffer is
only ever read at that instance's own `finalize()` call, but wasteful and easy to mistake for a
real per-episode isolation guarantee. **Decision:** the composition root
(`main_granular.build_components`) constructs exactly one `TransitionWriter` for the whole run and
registers it exactly once; `SessionRunner._handle_active`/`_finalize_episode_writers` calls its
`finalize(transitions_dir, episode_id)` once per completed task phase, which both writes that
episode's `_{id}_data.pt`/`_{id}_config.yaml` pair and clears the pending-push buffer for the next
episode. Rationale: matches "TransitionWriter finalized per episode (task phase only)" from the
task brief exactly, while keeping exactly one registered callback for the run's entire lifetime.

## 2026-07-20 — Wave 5: failure-policy specifics not fully dictated by the task brief

Three numeric/behavioral choices the task brief left to this wave's judgment (each logged here per
CONVENTIONS.md's uncertainty-log rule):
1. **`UnsafeLower`/`StaleMaskTimeout` retry count**: "replan once"/"retry once" is read as *exactly*
   one retry (a second occurrence of the same exception on the immediate retry ends the episode as
   `fail`, not a third attempt) — `SessionRunner._handle_active` tracks this with a single boolean
   flag per exception type, reset each new `ACTIVE` entry.
2. **`max_reset_iterations` default = 8**: caps `SegPushPlanner`'s `replan_after_each` loop in
   `RESETTING` (design 03 §3.3's adaptive-sweep contract). No design doc gives a number; 8 is
   generously above the 4-wall case (each wall needs at most one sweep to clear per iteration in
   the common case) while still bounding worst-case reset time. Configurable per `SessionRunner`
   instance.
3. **The between-episode pause (`recorder.pause()` + `timeout_between_experiments`) fires only on
   `EVALUATING -> STARTUP`**, not `INTERVENTION -> STARTUP`. Traced from legacy: `coordinator.py::
   _on_state_transition`'s pause fires on any transition *into* `STARTUP` from a different state,
   but legacy's only actually-reachable such transition (given `RandomPushGrasper.run_grasping`'s
   loop shape) is `ACTIVE -> STARTUP` ("recheck before next episode") — legacy's tool-grip
   intervention retry loop lived *inside* `startup()` with no visible `RobotActivity` state change
   at all, so there was never an "intervention -> back to startup" transition for the pause to
   fire on. Promoting `INTERVENTION` to a first-class `EpisodeState` (this wave) doesn't change
   that legacy fact; the 60s intervention wait itself already provides ample pause, so this wave
   does not add a second one on top of it.

## 2026-07-20 — Wave 5: `docs/CONVENTIONS.md` overstates the `cge` env — `flask`/`werkzeug` are not
installed

`CONVENTIONS.md`'s "Environment" section lists the `cge` conda env as having "cv2, numpy, yaml,
flask, ultralytics/torch, pytest". Verified directly: `import flask` raises `ModuleNotFoundError`
in that exact interpreter (`werkzeug` likewise absent). **Decision:** left the environment as-is
(did not `pip install` flask/werkzeug into `cge`) since no test in this wave needs it —
`ui/stream.py::MJPEGServer` imports both lazily (inside `start()`/`_build_app()`, never at module
import time) specifically so this module stays importable regardless, and `main_granular.py`'s
tested/default path is `--no-ui`. Logged here (and in `docs/recording.md`/`docs/testing.md`) so a
future wave that wants to actually exercise the MJPEG stream knows to `pip install flask werkzeug`
first rather than assuming CONVENTIONS.md's claim is current.

## 2026-07-20 — Wave 5: `main_granular.py` composition root — `build_components` has no side
effects; `--robot real` is the only path to `CloudGripperRobot`

Per the hard safety rule, `build_components(config, ...)` only constructs objects (including
`CoordinateFrames.from_config`/occupancy-provider construction, which do real file I/O — reading a
homography `.npz`/reference image — but send no robot commands and start no threads); `main()`
(guarded by `if __name__ == "__main__":`) is the only function that calls `.start()`/`.run()` on
anything, and `hardware.cloudgripper.CloudGripperRobot` is constructed only inside
`_build_robot(config, robot_mode)`'s `robot_mode == "real"` branch, reachable only via the
explicit `--robot real` CLI flag (default: `"dryrun"`, never inferred from config). Verified
end-to-end: `python -m autograsper.main_granular --config autograsper/granular-config.yaml
--robot dryrun --no-ui --episodes 1` builds every layer successfully against the checked-in
template (a real `homography.npz` already exists at the repo root, so `CoordinateFrames`
construction succeeds) and fails only at the expected, actionable point — `BackgroundDiffProvider`
raising `FileNotFoundError` for the template's placeholder `reference_empty_plate.jpg` — confirming
the composition root's wiring is correct up to exactly the boundary this repo's own
`# PLACEHOLDER — calibrate` comments already document as needing real calibration data.

## 2026-07-20 — Wave 6 (final wave): legacy relocation/deletion + documentation consolidation

Executed the "Fate of non-granular legacy graspers" decision logged above. All moves used `git
mv`, all deletions used `git rm` (full history preserved either way); verified by import-testing
every moved module and by running the full test suite (170 passed, 1 skipped, unchanged from
before this wave) both before and after.

**Deleted (`git rm`) — superseded granular-pipeline files, per design 02 §5's migration map:**
`autograsper/main_chickpeas.py`, `autograsper/main_chickpeas_segmenter.py`,
`autograsper/main_chickpeas_server.py` (read first as instructed: confirmed a granular-pipeline
main — already had a broken import of a nonexistent `autograsper.custom_graspers.
segmenting_granular_pusher` module and referenced an undefined `GranularPusher` name, i.e. it was
already non-functional before this wave), `autograsper/coordinator.py`, `autograsper/recording.py`,
`autograsper/recording_seg.py`, `autograsper/custom_graspers/granular_pusher.py`,
`autograsper/custom_graspers/fence_utils.py`, `object_tracker/granular_utils.py`,
`object_tracker/tool_user_utils.py`, `autograsper/frame_holder.py` (empty stub, confirmed
zero importers), `autograsper/granular_manipulation/utils.py` (empty file) plus the now-empty
`autograsper/granular_manipulation/{manipulation,perception}/` directories (`rmdir`, not
git-tracked). **Uncertainty resolved by direct verification:**
`autograsper/custom_graspers/random_push_grasper.py` was byte-identical (`md5sum` match) to
`granular_pusher.py` — a stale duplicate, not a distinct grasper — so it was deleted too,
alongside its twin.

**Moved (`git mv`) to `autograsper/legacy/`, preserved as-is, with a new `autograsper/legacy/
README.md`:** non-granular graspers `custom_graspers/{backgammon_grasper,calibrate_grasper,
evaluation_grasper,example_grasper,manual_grasper,stacking_autograsper,subtask1Grasper,
tool_user}.py` -> `legacy/custom_graspers/`, plus **one addition not in the original task list**:
`custom_graspers/random_grasping_task.py` — grep showed it imports the same legacy core
(`grasper`, `library.utils`, `library.rgb_object_tracker`) as the explicitly-named graspers, and
diffing it against `subtask1Grasper.py` (the closest sibling) showed a genuinely distinct
`RandomGrasper` class, not a duplicate — moved to `legacy/custom_graspers/` under the same "if
unsure, don't delete, keep in legacy" rule. Minimal legacy core: `grasper.py`, `action_tracker.py`,
`file_manager.py` (verified via grep to have zero importers among the kept graspers — only
`coordinator.py`/`recording.py`/`recording_seg.py`, all deleted — moved anyway per the task's
explicit list, harmless, kept for reference alongside `grasper.py`), `utils.py` (`load_config`),
and the whole `library/` package -> `legacy/library/`. Configs moved alongside what they configure:
`backgammon-config.yaml` -> `legacy/` (explicit); `config.yaml` -> `legacy/` (only consumer is
`main.py`, itself moved); `config.ini` -> `legacy/` (no code path actually loads it by name via
`configparser`/`load_config` — only referenced in `example_grasper.py` code comments — moved
alongside `config.yaml` as the same kind of legacy template rather than left ambiguously at the
repo root). **One addition beyond the task's explicit list:** `autograsper/main.py` (the
Flask/MJPEG entry point wiring `coordinator.DataCollectionCoordinator` to
`BackgammonGrasper`/`CalibrateGrasper`) -> `legacy/main.py`. It wasn't named in the task brief, but
it's the only runner for the two graspers moved above and depends on `coordinator.py`, which is
deleted per the migration map — moving it (rather than leaving it at the top level pointing at
nothing) keeps every one of its dependencies' relocations consistent with each other; its now-dead
`coordinator` import is documented as a known-broken caveat in `legacy/README.md`, matching how
this same log's earlier entries treat other standalone-tool broken imports.
**Import resolution verified, not assumed:** after moving,
`PYTHONPATH=<repo_root>:<repo_root>/autograsper/legacy python -c "import grasper; import
action_tracker; import library.utils; import custom_graspers.<each>"` succeeds unmodified for
every moved grasper — the same two path roots (repo root, for
`client.cloudgripper_client`/`object_tracker.ShapeAwarePoseEstimator`; the directory containing
`grasper.py`, for the bare `grasper`/`action_tracker`/`library.*` imports) that were required
before the move, now rooted at `autograsper/legacy/` instead of `autograsper/`. No import
statements needed changing.

**Left in place with documented broken-import caveats (standalone tools, not covered by
`autograsper/tests/`, per the task's explicit "leave broken + log, don't fix" option (a)):**
`autograsper/recorder_profiler.py` (lazy `from library.utils import
get_undistorted_bottom_image`), `image_collector/manual_control.py` (imports
`object_tracker.granular_utils`, `autograsper.custom_graspers.fence_utils`, and
`autograsper.library.utils` — all deleted or moved; also transitively broken via
`object_tracker.base_tool_tracker`), `image_collector/grab_tool.py` (imports
`object_tracker.tool_user_utils`, deleted), `object_tracker/base_tool_tracker.py` and
`base_tool_tracker2.py` (import `object_tracker.granular_utils`, deleted — these two files
themselves aren't in this wave's move/delete scope, so they were left in `object_tracker/` as-is,
now broken if imported). `autograsper/extract_transitions.py`, `segment_transitions.py`,
`run_transition_pipeline.py`, `reorganize_dataset.py`, `validate_action_tracking.py` were checked
and have no dependency on anything moved/deleted — untouched, unaffected. `temp.py` is empty and
untouched. `image_collector/chickpea_segmenter.py` was verified (per the task's explicit flag) to
be a live lazy-import dependency of `autograsper/perception/yolo_segmenter.py` and was
correspondingly **not touched**.

**Documentation:** updated `autograsper/docs/README.md` with a "Migration status" section
(pointing to `legacy/README.md` and design 02 §5) and an explicit list of the standalone-tool
broken-import caveats above. Fixed stale literal path citations across `docs/hardware.md`,
`docs/perception.md`, `docs/observation.md`, `docs/execution.md`, `docs/configuration.md` (each
had `autograsper/library/...`, `autograsper/grasper.py`, `autograsper/action_tracker.py`, or
`autograsper/config.yaml` citations that pointed at now-moved files; repointed to
`autograsper/legacy/...`). Left bare, non-path-prefixed historical citations (e.g. plain
`grasper.py`, `coordinator.py`, `file_manager.py` used as module-attribution prose in
`docs/session.md`/`docs/dataset_formats.md`/`docs/CONVENTIONS.md`) as-is — these read as "ported
from module X", not as literal current repo paths, and `coordinator.py`/`recording.py`/
`recording_seg.py` have no relocated path to redirect to anyway (deleted, not moved). No content
of any Wave 1-5 file was changed; only this log, `docs/README.md`, and the five doc files' path
citations above were edited.

Full test suite re-run after all moves/deletions: `170 passed, 1 skipped` (identical to the
pre-wave baseline). `python -m autograsper.main_granular --help` re-verified working.
