# Dataset formats — on-disk session layout (cross-cutting)

Status: **implemented** (Wave 5, `session/storage.py` + `recording/recorder.py`). Design
reference: [`../design/02_proposed_architecture.md`](../design/02_proposed_architecture.md) §6
("Keep the dataset contract... same on-disk layout... with additive extensions only");
[`../design/01_current_architecture.md`](../design/01_current_architecture.md) §3.2 (legacy
baseline layout); [`../design/03_segmenter_native_design.md`](../design/03_segmenter_native_design.md)
§4-5 (`masks_meta.jsonl`, online transitions); [`../MD files/transition_dataset_design.md`](../MD%20files/transition_dataset_design.md)
(the `RealData` transition format, implemented exactly).

This is the doc to consult for "what's in a recorded session directory and what does each file
mean" — module docs ([session.md](session.md), [recording.md](recording.md),
[execution.md](execution.md)) describe the *code* that produces these files; this doc describes
the *files*.

## Directory tree

```
<storage.base_dir>/<experiment.name>/<n>/            # n = 1, 2, 3, ... (session.storage.create_session_dirs)
├── status.txt                                       # "success" | "fail"
├── task/                                             # recorded during the ACTIVE episode phase
│   ├── Images/image_top_<f>.jpeg                     # f = 0, 1, 2, ... per-directory frame index
│   ├── Bottom_Images/image_bottom_<f>.jpeg
│   ├── Bottom_Images/image_bottom_raw_<f>.jpeg       # only if camera.save_bottom_raw (not yet wired
│   │                                                 # by Recorder in this wave -- see recording.md)
│   ├── Masks/mask_<f>.npy                            # binary grid_mask, uint8 {0, 255}
│   ├── Video/video_<n>.mp4                           # only if camera.save_images_individually: false
│   ├── Bottom_Video/video_<n>.mp4                    # (see recording.md's "Known limitations")
│   ├── states.jsonl                                  # one JSON object per line, appended live
│   ├── states.json                                   # same rows, as a JSON array (written once,
│   │                                                  # at episode end -- both files always present)
│   ├── orders.jsonl / orders.json                    # same jsonl-then-array pattern
│   ├── actions.json                                  # only written if any actions were recorded
│   └── masks_meta.jsonl                               # provenance for every saved mask
├── restore/                                          # same structure as task/, recorded during
│   │                                                  # the RESETTING episode phase
│   └── ... (Images/, Bottom_Images/, Masks/, states.*, orders.*, actions.json, masks_meta.jsonl)
└── transitions/                                      # only if storage.emit_transitions_online
    ├── _<n>_data.pt                                  # torch.save'd dict of tensors (task phase only)
    └── _<n>_config.yaml                              # grid/tool/physics/experiment metadata
```

`<n>` in `transitions/_<n>_data.pt` is the same session number as the enclosing directory (e.g.
session `3`'s transitions are `.../3/transitions/_3_data.pt`), not a separate counter.

## `states.json` / `states.jsonl`

One row per recorded frame. `states.jsonl` is append-only (flushed per row, crash-safe);
`states.json` is the same rows as a single JSON array, written once when the directory is
finalized (`session.coordinator.SessionRunner` retargeting/pausing the recorder, or the whole run
stopping) — **both files are always produced together**, `len(states.json) ==
number of states.jsonl lines`.

Row shape (`session.storage.StatesWriter.record`, ported verbatim from
`recording.py::Recorder.save_state`):

```json
{
  "x": 0.5, "y": 0.41, "z": 0.34, "rotation": 90.0, "claw": 1.0,
  "time": 1737382391.0,
  "frame_index": 12,
  "action": {
    "action_id": 5,
    "action_type": "move_xy",
    "phase": "task",
    "start_frame": 11,
    "is_planar_2d": true,
    "action_details": {"start_x": 0.4, "start_y": 0.6, "angle": 37.2, "end_x": 0.55, "end_y": 0.3, "height": 0.34},
    "description": "Push(start=(0.400,0.600), end=(0.550,0.300), angle=37.2, height=0.340)"
  }
}
```

- The flattened robot-state keys (`x`/`y`/`z`/`rotation`/`claw`) are `observation.types.RobotState`'s
  fields (`dataclasses.asdict`) — any may be `null` if the underlying `get_all_states()` read was
  missing/invalid for that field (see [observation.md](observation.md)).
- `"action"` is present only if `execution.actions.ActionTracker.get_action_for_frame(frame_index)`
  found a match (deepest/child-preferred — see [execution.md](execution.md)); its shape is a
  **subset** of a full `Action.to_dict()` (no `action_id`'s parent linkage, `end_frame`,
  robot-state snapshots, or `extra_metadata` — those live in `actions.json`, not repeated per
  frame).

## `orders.json` / `orders.jsonl`

One row per order sent to the robot (`execution.executor.Executor`'s `order_sink` contract, wired
through `session.storage.OrderSinkRouter` to a fresh `OrdersWriter` every episode). Same
jsonl-then-array pattern as `states.*`.

```json
{
  "order_type": "MOVE_XY",
  "order_value": [0.55, 0.3],
  "time": 1737382391.842,
  "robot_reported_time": null,
  "frame_index": 12
}
```

- `order_type`/`order_value`/`time` are the **legacy** keys (`library/utils.py::write_order`),
  unchanged — existing tooling reading only these three keeps working.
- `robot_reported_time`/`frame_index` are additive: `robot_reported_time` is the CloudGripper API's
  own per-command `"time"` field, unparsed (`null` for `DryRunRobot`, always); `frame_index` is the
  recorder frame index in effect when the order was sent (falls back to `Observation.seq` if no
  recorder has registered a frame-index provider yet — see [execution.md](execution.md)).

## `actions.json`

Written once, when a directory is finalized; **not written at all if no actions were recorded**
(legacy behavior, preserved).

```json
{
  "total_actions": 14,
  "actions": [
    {
      "action_id": 0, "action_type": "other", "phase": "startup", "start_frame": 0, "end_frame": 3,
      "is_planar_2d": false, "parent_id": null,
      "start_robot_state": {"x": 0.5, "y": 0.41, "z": 1.0, "rotation": 0.0, "claw": 1.0},
      "end_robot_state": {"x": 0.5, "y": 0.41, "z": 1.0, "rotation": 90.0, "claw": 1.0},
      "action_details": {},
      "description": "CheckToolGrip()",
      "extra_metadata": {}
    },
    {
      "action_id": 5, "action_type": "move_xy", "phase": "task", "start_frame": 11, "end_frame": 12,
      "is_planar_2d": true, "parent_id": null,
      "start_robot_state": {...}, "end_robot_state": {...},
      "action_details": {"start_x": 0.4, "start_y": 0.6, "angle": 37.2, "end_x": 0.55, "end_y": 0.3, "height": 0.34},
      "description": "Push(start=(0.400,0.600), end=(0.550,0.300), angle=37.2, height=0.340)",
      "extra_metadata": {}
    },
    {
      "action_id": 6, "action_type": "rotate", "phase": "task", "start_frame": 11, "end_frame": 11,
      "is_planar_2d": true, "parent_id": 5,
      "action_details": {"angle": 37.2}, "description": null, "extra_metadata": {},
      "start_robot_state": {...}, "end_robot_state": {...}
    }
  ]
}
```

Every completed `Action` (top-level primitive **and** per-order child) appears in `"actions"`, in
completion order (`execution.actions.Action.to_dict()` verbatim). Filter for **pushes** with
`action_type == "move_xy" and is_planar_2d and parent_id is None`; filter for **child (per-order)**
actions with `parent_id is not None` (`parent_id` equals the enclosing top-level action's
`action_id`).

## `masks_meta.jsonl`

One row per saved mask (design 03 §4), append-only:

```json
{"frame_index": 12, "source_seq": 47, "num_instances": 3, "mask_area": 1842}
```

`num_instances`/`mask_area` are `null` for providers with no per-instance notion (e.g.
`BackgroundDiffProvider`, whose `instances` is always `()`, still reports `mask_area` from
`stats.total_area_px`).

## `Masks/mask_<f>.npy`

One binary occupancy mask per saved frame (`np.save`), uint8 `{0, 255}`, in the canonical dataset
`grid` frame (`perception.grid.height`/`.width` — see [perception.md](perception.md)). Not saved
for every frame — only when a new, distinct `source_seq` result becomes available from the
configured occupancy supplier (see [recording.md](recording.md)'s "Mask saving" section for the
exact save/dedup/bootstrap rule).

## `transitions/_<n>_data.pt` + `_<n>_config.yaml` — the `RealData` transition format

Implemented **exactly** per `MD files/transition_dataset_design.md` (the contract
`RealPileSweepData`/`PileSweepData` share), emitted **online** (design 03 §5) rather than by the
offline `extract_transitions.py`/`segment_transitions.py` pipeline (which remains a
validation/backfill tool for previously recorded sessions using the same file pair format).

### `_<n>_data.pt` (`torch.save`'d dict)

| Key | Shape | dtype | Description |
|---|---|---|---|
| `masks_before` | `(N, H, W)` | `float32` | Binary occupancy mask **before** each push, values in `[0, 1]` |
| `masks_after` | `(N, H, W)` | `float32` | Binary occupancy mask **after** each push |
| `p_starts_px` | `(N, 2)` | `float32` | Tool centre at push start, `(x_col, y_row)`, GRID pixels, top-left origin |
| `p_stops_px` | `(N, 2)` | `float32` | Tool centre at push end, same frame |
| `angles` | `(N,)` | `float32` | Tool yaw angle in **radians** (converted from the planner's degrees) |

`N` = the number of completed top-level `Push` actions in that episode's **task** phase whose
before/after masks were resolvable (`session.storage.TransitionWriter.on_action_completed`
filters for `parent_id is None and action_type == MOVE_XY and is_planar_2d`; a push whose masks
can't be resolved — e.g. the recorder's ring was reset before the masks were saved — is skipped
with a logged warning rather than aborting the write, so `N` can be `< n_pushes` in that edge
case; the normal case is `N == experiment.n_pushes`).

- `masks_before[i]`/`masks_after[i]`: the grid mask most recently saved at/before the push's
  `start_frame`, and earliest saved at/after its `end_frame`, respectively
  (`recording.recorder.Recorder.mask_for_frame`). Per design 03 §5's edge rule, if push *i*'s
  "after" frame and push *i+1*'s "before" frame resolve to the same underlying saved frame, they
  are (by construction, not a special case) the exact same array.
- `p_starts_px`/`p_stops_px`: the push's robot-frame `(start_x, start_y)`/`(end_x, end_y)`
  converted to GRID pixel coordinates via `perception.frames.CoordinateFrames`
  (`robot_to_crop_px` then `crop_px_to_grid`) — `(x_col, y_row)` order, top-left origin, matching
  the coordinate note in `transition_dataset_design.md`.
- `angles`: the push's `angle` field (planner degrees) converted to radians (`np.deg2rad`).

### `_<n>_config.yaml`

```yaml
grid:
  height: 128        # perception.grid.height -- must match masks_before/after's H
  width: 128          # perception.grid.width -- must match masks_before/after's W
tool:
  size_px: [8, 120]   # storage.tool_size_px_for_transitions, [width_px, height_px]
physics:
  friction: null
  density: null
  box_friction: null
experiment:
  material: chickpeas
  surface: glass
  date: "2026-07-20"  # time.strftime('%Y-%m-%d') at finalize() time -- NOT the recording date
```

`physics.*` is always `null` (unknown/unmeasured — `RealPileSweepData` falls back to
`default_physics` for these). `experiment.*` beyond `material`/`surface`/`date` is whatever extra
keys were passed as `TransitionWriter`'s `experiment_meta` (free-form, not read by the dataset
loader).

## Legacy compatibility

- `states.json`/`orders.json`/`actions.json`/`status.txt`/`Masks/mask_<f>.npy` keep their legacy
  key names and file names; every extension (`.jsonl` siblings, `parent_id`, `robot_reported_time`,
  `frame_index` on order rows, `masks_meta.jsonl`) is additive — existing tooling reading only the
  legacy keys/files keeps working unmodified.
- The offline transition-extraction scripts (`extract_transitions.py`, `segment_transitions.py`,
  `run_transition_pipeline.py`) still work against previously-recorded sessions (or sessions
  recorded with `storage.emit_transitions_online: false`) — they produce the exact same
  `_<id>_data.pt`/`_<id>_config.yaml` pair format this module now also emits online.
