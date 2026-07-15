Real Experiment Data Format (`RealData/`)

Designed to be as close as possible to the simulation format so that both `PileSweepData`
and `RealPileSweepData` return identical `((input_grid, physics), output_grid)` batches.

### Pre-processing contract (before saving)
The caller is responsible for:
1. **Segmenting** each camera frame into a binary occupancy mask and **resizing** it to the
   target grid resolution (e.g. 128 × 128).
2. **Calibrating** robot/tool positions to image-pixel coordinates in that cropped grid,
   using a homography or known camera intrinsics + extrinsics.
3. Storing tool positions as pixel coordinates with the **top-left corner as origin**,
   consistent with the convention used by the sim dataset's cv2 drawing calls.

### File pairs per run (under `real_data/{experiment_id}/`)
```
_{id}_data.pt      ← torch.save'd dict of tensors
_{id}_config.yaml  ← grid, tool, and physics metadata
```

### `_data.pt` dict keys

| Key | Shape | dtype | Description |
|---|---|---|---|
| `masks_before` | `(N, H, W)` | float32 | Binary occupancy mask **before** the action, values in [0,1] |
| `masks_after` | `(N, H, W)` | float32 | Binary occupancy mask **after** the action |
| `p_starts_px` | `(N, 2)` | float32 | Tool centre at action start `(x_col, y_row)` in pixels |
| `p_stops_px` | `(N, 2)` | float32 | Tool centre at action end `(x_col, y_row)` in pixels |
| `angles` | `(N,)` | float32 | Tool yaw angle in radians |

> **Coordinate note:** `x_col` = horizontal (column) axis, `y_row` = vertical (row) axis,
> matching cv2's `(cx, cy)` convention so that drawing functions are identical to the sim path.

### Config YAML structure
```yaml
grid:
  height: 128        # pixels — must match mask tensor H
  width:  128        # pixels — must match mask tensor W
tool:
  size_px: [40, 2]   # [width_px, height_px] of the tool rectangle
physics:             # known/estimated values; null = unknown (dataset returns 0.0)
  friction:     null
  density:      null
  box_friction: null
experiment:          # free-form metadata (not used by dataset)
  material: chickpeas
  surface:  glass
  date:     "2025-01-01"
```

### Split strategy
Deterministic by hashing the **run file path** (not physics, which may be unknown).
`val_pct=10`, `test_pct=10` by default (same API as `PileSweepData`).

---

## For reference: `RealPileSweepData` Dataset (`RealData/dataset.py`)

Drop-in replacement for `PileSweepData` when working with real data.

**Construction:**
```python
RealPileSweepData(
    data_root: str | Path,   # root directory; paths are resolved relative to this
    paths: list[str] | str,  # subdirectories under data_root
    split: "train"|"val"|"test"|None,
    default_physics: [f, d, bf] | None,  # fallback when config has null values
)
```

**`__getitem__` output** — identical to `PileSweepData`:
```
((input_grid: Tensor[2, H, W], physics: Tensor[3]), output_grid: Tensor[H, W])
```

| Channel | Source | Rendering |
|---|---|---|
| `input_grid[0]` | `masks_before[i]` | copied directly — no rendering needed |
| `input_grid[1]` | `p_starts_px`, `p_stops_px`, `angles` | same cv2 rotated-rect draw as sim (start→0.5, end→1.0) |
| `output_grid` | `masks_after[i]` | copied directly |
| `physics` | config `physics.*` or `default_physics` | `[friction, density, box_friction]` |
