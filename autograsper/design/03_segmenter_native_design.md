# Segmenter-Native Design — assuming an accurate, occlusion-tolerant segmenter

*Specialization of [02_proposed_architecture.md](02_proposed_architecture.md) for the case the
abandoned refactor was heading toward: a trained YOLOv11 `ChickpeaSegmenter` that is accurate
**with the robot and tool in the field of view**, so the robot never needs to move aside to get a
valid mask. This document is the design for the never-written `SegGranularPusher` — expressed in
the new architecture rather than as another grasper subclass.*

## 1. What the assumption buys

The single premise — *masks are trustworthy anytime, robot visible or not* — removes the three
most expensive behaviors of the cautious pipeline:

| Cautious pipeline (`RandomPushGrasper`) | Segmenter-native |
|---|---|
| `update_mask_and_process()`: raise Z, drive to corner (0,1), wait 1.5 s, capture, drive back — **~10 s of robot motion per mask** | No motion. A perception worker segments every observation; a fresh mask is always available. |
| `interaction_since_last_mask` lazy-refresh flag threaded through every behavior | Replaced by a uniform **freshness rule** on the observation stream (§3.2). |
| Reference empty-plate image, color-threshold robot rejection, morphology tuning (`granular_utils.create_occupancy_mask`) | Model handles appearance; robot/tool pixels are simply "not chickpea". |
| Reset/placement decisions made on a mask that is stale by one whole approach motion | Decisions and **safety guards** run on the mask captured frames before/while the action happens. |
| One mask per episode phase → transitions extracted offline by diffing sparse masks | Per-frame masks aligned with action boundaries → **transitions emitted online** (§5). |

Everything else from doc 02 (layers, primitives, session machine, dry-run robot) carries over
unchanged. Below are only the deltas.

## 2. Perception layer changes

### 2.1 `perception/yolo_segmenter.py` — `SegmentationWorker`

Successor of `recording_seg.SegmentationThread`, promoted from a recorder detail to the standard
perception provider:

- Subscribes to `ObservationSource` (latest-wins; segmenting every frame is not required — it
  processes the newest unprocessed observation, tagged with that observation's `seq` and
  `frame_index`).
- Crops using the **single canonical crop from `perception/frames.py`** (this is where the
  current 360×360-at-(275,200) vs. 360×360-at-center discrepancy gets resolved — one definition,
  calibrated once against the homography; the `create_homography_calibration.py` /
  `calibrate_from_dataset.py` scripts already in `image_collector/` are the tooling for this).
- Runs `ChickpeaSegmenter.predict(return_format='dict')` and publishes an immutable result:

```python
@dataclass(frozen=True)
class OccupancyResult:
    source_seq: int             # Observation.seq this mask was computed from
    frame_index: int | None
    timestamp: float
    grid_mask: np.ndarray       # binary, canonical grid frame (crop → resize, e.g. 128×128)
    crop_mask: np.ndarray       # full-resolution crop mask (for placement search)
    instances: list[Instance]   # per-chickpea masks/boxes/confidences (from 'individual')
    stats: ClumpStats           # connected components on grid_mask (area, centroids)
```

- `latest()` / `await_result(min_seq)` mirror the `ObservationSource` API, so consumers can
  demand "a mask computed from an observation taken after my last motion completed".

GPU note: inference latency (~tens of ms on CUDA, ~hundreds on CPU) is far below the 2.5 FPS
observation rate, so one worker thread suffices; if it ever lags, latest-wins semantics keep it
from queueing stale work.

### 2.2 `WorldState.occupancy` is now always present

The planner can assume `occupancy is not None` after startup; `staleness = obs.seq -
occupancy.source_seq` is exposed so policies can require `staleness == 0` where it matters.

## 3. Planner changes (`planning/seg_push_planner.py`)

The policy logic is *mostly identical* to `random_push_planner` — same walls, same placement
search, same random pushes — minus the caution choreography:

### 3.1 Removed
- `RefreshMask` primitive and every move-aside round trip (startup, between wall sweeps,
  post-task). `plan_startup` shrinks to: tool-grip check pose → grip check → verify granules
  present in `occupancy.stats`.
- `interaction_since_last_mask` bookkeeping.
- The reference-image config (`Granuler_detection.reference_image_path`) and its "no reference →
  abort" startup path.

### 3.2 Replaced by a freshness rule
One policy, enforced in the executor rather than the planner:

> A primitive that *reads* the mask (placement search, wall band check, lower-tool guard) must
> use an `OccupancyResult` whose `source_seq` is newer than the completion of the last
> scene-touching primitive. If not yet available, `await_result()` — worst case one observation
> period (400 ms at 2.5 FPS), vs. ~10 s for a move-aside refresh.

### 3.3 Upgraded behaviors
- **`needs_reset` / wall checks after every episode become cheap**, so they run every cycle
  (the current code comments show this was desired but disabled in `reset_task` because each
  check cost a mask round trip).
- **`LowerTool` gains a real-time guard** (doc 02 §3.4): immediately before MOVE_Z down, verify
  the tool footprint region of the *current* mask is clear. With continuous masks this check is
  finally meaningful — in the cautious pipeline the mask predated the whole approach motion.
- **Mid-sweep verification**: after each wall sweep pass, the next mask shows whether the band
  cleared; the planner can stop sweeping a wall early or add a pass, instead of the fixed
  3-pass `np.linspace(t_min, t_max, 3)` schedule.
- **Push-effect awareness (optional, later)**: `instances`/`stats` before and after each push
  allow rejecting no-op pushes (tool moved through empty space) and resampling — increases the
  fraction of informative transitions in the dataset at zero robot-time cost.

### 3.4 Tool-grip check stays vision-based but unchanged
The top-camera color-ROI grip check and the human-assisted regrasp loop are orthogonal to the
bottom-camera segmenter and port as-is (`perception/tool_grip.py`, `RegraspTool` primitive,
`Intervention` session state). A future improvement (out of scope): train the segmenter or a
second head to detect the tool, replacing the color threshold.

## 4. Recording changes

- The recorder saves the `OccupancyResult` associated with each frame (`Masks/mask_<f>.npy`,
  binary `grid_mask`, as `recording_seg.py` already does) **plus** a `masks_meta.jsonl` row
  linking `frame_index → source_seq, num_instances, mask_area` so downstream tools know mask
  provenance without recomputing.
- `latest_mask_saved` flag disappears — masks are keyed by `frame_index`, saved when their
  source frame is saved, once.
- `states.json` rows keep the action metadata; no change needed for the transition tooling.

## 5. Online transition emission (the payoff)

With per-frame masks and executor-owned action boundaries, the pipeline currently done offline
(`segment_transitions.py` → `extract_transitions.py` → `run_transition_pipeline.py`) can be
emitted during collection. The executor already knows, for every `Push` primitive:

- `start_frame` / `end_frame` (ActionTracker),
- start/end robot state → tool center in robot coords → **pixels** via
  `perception/frames.py` (top-left-origin convention per `transition_dataset_design.md`),
- tool yaw angle (radians).

A `TransitionWriter` in `session/storage.py` subscribes to completed `Push` actions and writes,
per episode:

```
masks_before[i] = grid_mask at latest frame ≤ start_frame with staleness 0
masks_after[i]  = grid_mask at earliest frame ≥ end_frame with staleness 0
p_starts_px, p_stops_px, angles from the action record
```

accumulated into the `_{id}_data.pt` + `_{id}_config.yaml` pair of the RealData format. The
offline scripts remain as validation/backfill tools for previously recorded sessions.

**Edge rule:** if the "after" mask of push *i* and the "before" mask of push *i+1* would be the
same frame, share it — matching how the sim dataset chains sweeps.

## 6. Configuration delta

```yaml
perception:
  provider: yolo                 # was: background_diff
  yolo:
    weights: image_collector/chickpeas_segmentation_best.pt
    conf_threshold: 0.25
    iou_threshold: 0.5
    imgsz: 640
  grid: {height: 128, width: 128}     # canonical dataset grid
  crop: {center_px: [275, 200], size: [360, 360]}   # ONE definition, calibrated
  freshness: {require_staleness_0_for: [lower_tool, placement, wall_check]}
transitions:
  emit_online: true
  tool_size_px: [40, 2]
```

Removed keys: `Granuler_detection.reference_image_path`, `Granuler_detection.min_granule_size`
(instance filtering is the model's job; a `min_instance_area_px` safety filter remains under
`perception.yolo`).

## 7. Risk register / open questions

1. **Model blind spots under the tool.** Chickpeas directly beneath the flat tool are occluded
   from the bottom camera regardless of model quality. The `LowerTool` guard therefore checks
   the mask *before* the tool covers the spot (approach at clearance height → guard → lower).
   Residual risk accepted, same as today.
2. **Mask ↔ homography consistency** is the highest-value calibration item; the design forces it
   through one `CoordinateFrames` object, but the actual calibration (redo `homography.npz`
   against the canonical crop) is a prerequisite task before any behavior work.
3. **Confidence handling**: `predict` gives per-instance confidences; policy for low-confidence
   detections (treat as occupied for safety guards, ignore for reset statistics?) — proposed
   default: **occupied for safety, threshold 0.25; counted for statistics at 0.5**.
4. **Failure fallback**: if the segmenter output degenerates (0 detections while the previous
   frames showed many, or CUDA failure), raise `PerceptionDegraded` → session layer falls back
   to the background-diff provider (move-aside behavior re-enabled via the `RefreshMask`
   primitive, which stays implemented for exactly this path) or pauses for intervention.
5. **Throughput**: episode time becomes dominated by pushes themselves (~2.5 s/order). Expected
   saving: ~10 s × (1 startup + N wall checks) per episode — roughly a 30–50 % increase in
   transitions/hour at current settings.
