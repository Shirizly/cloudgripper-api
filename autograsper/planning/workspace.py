"""`Workspace` — fence walls, placement search, wall/center reset checks (Wave 3b).

Design reference: `autograsper/design/02_proposed_architecture.md` §3.3 ("All geometry helpers
(`fence_utils`, placement search) move to `planning/workspace.py` and become pure functions of
`WorldState` — no robot calls, no `cv2.imwrite`, no prints").

Porting notes (legacy sources copied/adapted, not imported, per CONVENTIONS.md):
- `custom_graspers/fence_utils.py::Wall`, `build_fence_walls`, `sample_tool_pose`,
  `get_pos_sweep_from_optimal` — ported **exactly** (same math, same angle conventions). Verified
  numerically (not just read): for the template `granular-config.yaml` fence
  (`center=(0.5, 0.49)`, `size=(0.94, 0.955)`), `build_fence_walls` yields `top=90°`,
  `right=0°`, `bottom=90°`, `left=0°` — see `tests/test_planning_workspace.py`. (Anyone reasoning
  informally from the tangent vectors alone might expect the opposite pairing; the `%pi`
  wrap-then-scale-to-degrees formula is what actually produces this, confirmed by direct
  computation, not by assumption.)
- `custom_graspers/fence_utils.py::check_placement`, `make_tool_mask`, `in_region_pix`,
  `find_tool_placements` — ported, with cleanups:
  - No `cv2.imwrite`/`cv2.imshow` (`make_tool_mask` wrote `tool_mask_*.png` to the CWD on every
    call; `visualize_mask_overlap`, a commented-out-by-default `cv2.imshow`/`waitKey` GUI helper,
    is dropped entirely — no library code may open a GUI window). Debug artifacts go through an
    optional `autograsper.observation.debug.DebugSink` instead.
  - `find_tool_placements` returns `None` when no placement was found, instead of legacy's
    inconsistent `{}` (an empty dict is falsy but not `None`-like/typed; every call site had to
    special-case it — `None` is the one honest "no result" value).
  - No `print` — `logging.debug`/`logging.warning`.
- `custom_graspers/fence_utils.py::check_wall_reset_needed` — ported with its band-occupancy math
  and y-flip conventions **verbatim** (`center_px = [center[0]*w, (1-center[1])*h]` — see
  `docs/perception.md`'s "Y-axis convention" note, confirmed against this exact line), plus:
  - No `cv2.imwrite` (`'band mask and mask at {wall.label}.png'`) / `print`.
  - Two **dead legacy parameters** are surfaced here, not silently dropped: `margin_of_safety`
    (accepted a default but was never referenced in the function body) is removed entirely (kept
    would be dead code, forbidden by CONVENTIONS.md); `min_granule_size` **is** kept in this
    port's signature (matching the Wave 3b task's specified signature) even though — like
    legacy — nothing in the ported band-occupancy math actually uses it. Both are logged in
    `design/IMPLEMENTATION_LOG_planning.md`.
  - This port additionally accepts a `frames: CoordinateFrames` parameter (absent from legacy,
    which relied on a hand-rolled `PixelRobotTransform` the caller (`sweep_wall`) invoked
    separately) so the returned `details` dict already carries a robot-frame position
    (`"pos_robot"`) alongside the legacy `"pos_px"`, sparing planners a second homography call.
  - `custom_graspers/fence_utils.py::PixelRobotTransform` is NOT re-implemented here — Wave 2's
    `perception.frames.CoordinateFrames.crop_px_to_robot`/`.robot_to_crop_px` already ported that
    math (see `docs/perception.md`'s porting notes); this module takes a `CoordinateFrames`
    instance rather than duplicating homography code.
- `object_tracker/granular_utils.py::check_reset_needed` -> `check_center_reset_needed`: ported
  with its 0.3 threshold and central-30%-70% workspace box verbatim, `logging.debug` instead of
  `print`. **Known quirk preserved, not fixed** (task explicitly asks for this): the function
  computes `occupancy_ratio = occupied_area / mask_area` where `mask_area` is the **total**
  nonzero area of the whole mask (`cv2.countNonZero(mask)`), not the area of the central
  `workspace_mask` region. So the "ratio" is really "fraction of all detected granule mass that
  currently sits in the central box", not "how full is the central box" — a mask with very little
  total material but almost none of it central would still report a low ratio and (correctly, by
  legacy's own logic) trigger a reset, but a mask where 90% of the (small) total mass sits
  centrally and 10% sits elsewhere would report `ratio=0.9` and skip reset even if the central box
  is nearly empty in absolute terms. This is the exact behavior `RandomPushGrasper.startup`/
  `RandomPushPlanner.needs_reset` relies on; changing it would change what triggers `RESETTING` in
  a way not requested by this task.

Threading: pure functions of their arguments (no shared mutable state); `Wall`/`Workspace` are
safe to share and call from any thread. `Workspace` is stateless after construction (built once
from a `WorkspaceConfig` + `CoordinateFrames`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from autograsper.observation.debug import DebugSink

if TYPE_CHECKING:  # pragma: no cover - typing only
    from autograsper.config_schema import WorkspaceConfig
    from autograsper.perception.frames import CoordinateFrames

logger = logging.getLogger(__name__)

PxBox = Tuple[Tuple[int, int], Tuple[int, int]]  # ((xmin, xmax), (ymin, ymax)) in crop_px


# ---------------------------------------------------------------------------
# Wall / fence geometry — ported from custom_graspers/fence_utils.py
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Wall:
    """One fence wall. Robot-frame geometry (fence coordinates are in the same `[0, 1]^2` xy
    plane as `RobotInterface.move_xy`).

    - `origin`: reference point on the wall (its center), robot xy.
    - `tangent`: unit vector along the wall, robot frame.
    - `normal`: unit vector perpendicular to the wall, pointing INTO the workspace (e.g. the top
      wall's normal is `(0, -1)`, pointing down/inward — see `build_fence_walls`).
    - `t_min`/`t_max`: valid range for the tangent-direction slide parameter `t` used by
      `sample_tool_pose` (keeps the tool's placement footprint fully on the wall segment, with
      `safety_margin` clearance from the corners).
    - `angle`: tool orientation in degrees for this wall (used for `ROTATE` commands when
      approaching/sweeping this wall).
    """

    label: str
    origin: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    t_min: float
    t_max: float
    angle: float


def build_fence_walls(
    fence_center: Sequence[float],
    fence_size: Sequence[float],
    tool_length: float,
    tool_width: float,
    safety_margin: float,
) -> List[Wall]:
    """Build the four fence walls (top/right/bottom/left, CCW) around `fence_center`/
    `fence_size` (robot-frame xy). Ported verbatim from
    `custom_graspers/fence_utils.py::build_fence_walls` (same math, same angle formula)."""
    fx, fy = fence_center
    h = [fs / 2 for fs in fence_size]
    hx, hy = h

    half_tool_len = tool_length / 2

    walls: List[Wall] = []

    # Define walls in CCW order (verbatim from legacy).
    wall_defs = [
        ("top", np.array([fx, fy + hy]), np.array([1, 0]), np.array([0, -1])),
        ("right", np.array([fx + hx, fy]), np.array([0, -1]), np.array([-1, 0])),
        ("bottom", np.array([fx, fy - hy]), np.array([-1, 0]), np.array([0, 1])),
        ("left", np.array([fx - hx, fy]), np.array([0, 1]), np.array([1, 0])),
    ]

    for label, origin, tangent, normal in wall_defs:
        tangent = tangent / np.linalg.norm(tangent)
        normal = normal / np.linalg.norm(normal)

        t_min = -abs(np.inner(h, tangent)) + half_tool_len + safety_margin
        t_max = abs(np.inner(h, tangent)) - half_tool_len - safety_margin

        angle = np.arctan2(tangent[1], tangent[0])
        angle = (angle + np.pi / 2) % np.pi
        angle = angle / np.pi * 180  # convert to degrees
        angle = angle + 180 if angle < 0 else angle
        angle = 0 if angle == 180 else angle
        angle = int(angle)

        walls.append(
            Wall(
                label=label,
                origin=origin,
                tangent=tangent,
                normal=normal,
                t_min=t_min,
                t_max=t_max,
                angle=angle,
            )
        )

    return walls


def sample_tool_pose(
    wall: Wall, tool_width: float, safety_margin: float, t: Optional[float] = None
) -> Dict[str, object]:
    """Sample a tool pose sliding along `wall` at parameter `t` (or a uniform-random `t` in
    `[wall.t_min, wall.t_max]` if `t` is `None`). Ported verbatim from
    `custom_graspers/fence_utils.py::sample_tool_pose`.

    Callers that need reproducibility (planners) should always pass an explicit `t` — this
    function's `t=None` branch uses the numpy global RNG (`np.random.uniform`), matching legacy
    behavior exactly, but is therefore not seed-controlled by a planner's own
    `np.random.Generator`.

    Returns a dict with `x`/`y` (robot frame), `angle` (degrees, `wall.angle`),
    `parallel_dir`/`perpendicular_dir` (`wall.tangent`/`wall.normal`).
    """
    if t is None:
        t = np.random.uniform(wall.t_min, wall.t_max)
    else:
        t = np.clip(t, wall.t_min, wall.t_max)

    offset = tool_width / 2 + safety_margin
    center = wall.origin + wall.tangent * t + wall.normal * offset

    return {
        "x": center[0],
        "y": center[1],
        "angle": wall.angle,
        "parallel_dir": wall.tangent,
        "perpendicular_dir": wall.normal,
    }


def get_pos_sweep_from_optimal(
    pos_optimal: np.ndarray, wall: Wall, margin: float = 0.02
) -> np.ndarray:
    """Move `pos_optimal` toward `wall` until it is `margin` away (robot frame). Ported verbatim
    from `custom_graspers/fence_utils.py::get_pos_sweep_from_optimal`."""
    optimal_dist = np.dot(pos_optimal - wall.origin, wall.normal)
    delta_dist = optimal_dist - margin
    return pos_optimal - wall.normal * delta_dist


# ---------------------------------------------------------------------------
# Placement search — ported from custom_graspers/fence_utils.py
# ---------------------------------------------------------------------------


def check_placement(
    dist: np.ndarray, tool_mask: np.ndarray, cx: int, cy: int, r: int
) -> Tuple[bool, float]:
    """Does `tool_mask` (centered at `(cx, cy)`, radius `r`) fit inside the free region of `dist`
    (a distance transform of the free space) without overlapping any obstacle pixel? Returns
    `(fits, clearance_px)`. Ported verbatim from `custom_graspers/fence_utils.py::check_placement`.
    """
    ys = slice(cy - r, cy + r + 1)
    xs = slice(cx - r, cx + r + 1)

    y_start = max(0, ys.start)
    y_stop = min(dist.shape[0], ys.stop)
    x_start = max(0, xs.start)
    x_stop = min(dist.shape[1], xs.stop)

    y_offset = y_start - (cy - r)
    x_offset = x_start - (cx - r)

    local_dist = dist[y_start:y_stop, x_start:x_stop]
    local_mask = tool_mask[
        y_offset : y_offset + (y_stop - y_start), x_offset : x_offset + (x_stop - x_start)
    ]

    overlap = np.any((local_mask == 1) & (local_dist == 0))
    if overlap:
        return False, 0.0

    mask_pixels = local_dist[local_mask == 1]
    if mask_pixels.size == 0:
        return False, 0.0

    clearance = float(np.min(mask_pixels))
    return True, clearance


def make_tool_mask(
    w_px: int, h_px: int, angle_deg: float, debug: Optional[DebugSink] = None
) -> Tuple[np.ndarray, int]:
    """Binary mask of the tool footprint (`w_px` x `h_px`) rotated `angle_deg` (tool is vertical,
    along height, at `angle_deg=0`). Ported from `custom_graspers/fence_utils.py::make_tool_mask`,
    with the `cv2.imwrite(f'tool_mask_{w_px}x{h_px}_angle{angle_deg}.png', ...)` debug dump
    replaced by an optional `DebugSink` (no-op unless enabled)."""
    r = int(np.ceil(0.5 * np.hypot(h_px, w_px)))
    pad = 2 * r + 1

    mask = np.zeros((pad, pad), dtype=np.uint8)
    center = (r, r)

    # negative angle for cv2 convention to match robot coordinate system (verbatim from legacy).
    rect = (center, (w_px, h_px), float(-angle_deg))
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(mask, box, 1)

    if debug is not None:
        debug.save(f"tool_mask_{w_px}x{h_px}_angle{angle_deg}", mask * 255)

    return mask, r


def in_region_pix(x: float, y: float, region: PxBox) -> bool:
    """Is `(x, y)` inside `region = ((xmin, xmax), (ymin, ymax))`? Ported verbatim from
    `custom_graspers/fence_utils.py::in_region_pix`."""
    (xmin, xmax), (ymin, ymax) = region
    return xmin <= x <= xmax and ymin <= y <= ymax


def find_tool_placements(
    obstacle_mask_crop: np.ndarray,
    tool_dims_px: Tuple[int, int],
    angles_deg: Sequence[float],
    search_region_px: PxBox,
    min_clearance_px: int = 10,
    debug: Optional[DebugSink] = None,
) -> Optional[Dict[str, object]]:
    """Search `obstacle_mask_crop` (crop-frame binary mask, nonzero = occupied) for a granule-free
    spot to place the tool (`tool_dims_px = (w_px, h_px)`), trying each angle in `angles_deg`,
    restricted to `search_region_px`. Returns the best (highest-clearance) placement found —
    `{"pos_px": (x, y), "angle": angle_deg, "clearance_px": float}` — or `None` if none exists.

    Ported from `custom_graspers/fence_utils.py::find_tool_placements`: same distance-transform +
    rotated-rectangle-footprint search, same "stop early once `min_clearance_px` is reached, else
    keep the best-clearance candidate seen" logic. Differs from legacy only in: returns `None`
    instead of `{}` when nothing is found (legacy's return value was falsy either way, but typed
    inconsistently — an empty dict has no way to `.get("pos_px")` meaningfully, forcing every call
    site to special-case truthiness); the `visualize_mask_overlap` GUI debug call is dropped (no
    `cv2.imshow` in library code, per CONVENTIONS.md).
    """
    free = ~obstacle_mask_crop.astype(bool)
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)

    placements: List[Dict[str, object]] = []

    (xmin, xmax), (ymin, ymax) = search_region_px

    for angle in angles_deg:
        tool_mask, r = make_tool_mask(tool_dims_px[0], tool_dims_px[1], angle, debug=debug)

        search_dist = np.zeros_like(dist)
        search_dist[ymin : ymax + 1, xmin : xmax + 1] = dist[ymin : ymax + 1, xmin : xmax + 1]

        candidates = np.argwhere(search_dist > int(np.min(tool_dims_px) / 2 + 1))
        if candidates.size == 0:
            continue

        for cy, cx in candidates:
            ok, clearance = check_placement(dist, tool_mask, int(cx), int(cy), r)
            if not ok:
                continue

            placements.append({"pos_px": (int(cx), int(cy)), "angle": angle, "clearance_px": clearance})

            if clearance >= min_clearance_px:
                break  # good enough for this angle; move on to the next candidate angle

    if not placements:
        logger.debug("find_tool_placements: no valid placement found in search region")
        return None

    placements.sort(key=lambda p: -p["clearance_px"])
    return placements[0]


# ---------------------------------------------------------------------------
# Reset checks — ported from custom_graspers/fence_utils.py + object_tracker/granular_utils.py
# ---------------------------------------------------------------------------


def check_wall_reset_needed(
    mask_crop: np.ndarray,
    wall: Wall,
    tool_dims_px: Optional[Tuple[int, int]],
    min_granule_size: int,
    frames: "CoordinateFrames",
    debug: Optional[DebugSink] = None,
) -> Tuple[bool, Dict[str, object]]:
    """Is there enough granule mass piled against `wall` to warrant a reset sweep, and if so,
    where's a good spot to place the tool to do it?

    Returns `(reset_needed, details)`:
    - `reset_needed=False`, `details={}` — wall is fine (or mask is empty/`None`).
    - `reset_needed=True`, `details={"use_fallback": True, "wall": wall}` — reset needed but no
      granule-free spot was found near the wall; the caller should use the cautious fallback
      choreography (see `SweepWall`'s `mode="fallback"`).
    - `reset_needed=True`, `details={"use_fallback": False, "wall": wall, "pos_px": (x, y),
      "pos_robot": (rx, ry), "angle": tool_angle_deg, "min_distance": clearance_px}` — a
      granule-free spot was found; `pos_robot` is `pos_px` converted via
      `frames.crop_px_to_robot` (added relative to legacy, which left this conversion to the
      caller). NOTE: `details["angle"]` is the *search* angle (`atan2` of the wall normal, used
      only to orient the tool-footprint mask for this free-space search) — legacy's actual caller
      (`sweep_wall`) rotates to `wall.angle` instead, not this value; see `SweepWall`'s docstring.

    Ported from `custom_graspers/fence_utils.py::check_wall_reset_needed`: same band-occupancy
    math and y-flip conventions (`center_px = [center[0]*w, (1-center[1])*h]`), same 0.2
    band-width / 0.1 occupancy-ratio thresholds, same fallback search-region construction. No
    `cv2.imwrite`/`print` (see module docstring); the dead `margin_of_safety` parameter is
    dropped, `min_granule_size` is kept (also unused in the ported math, matching legacy) per
    module docstring.
    """
    if mask_crop is None:
        return False, {}

    h, w = mask_crop.shape
    wall_vec = wall.tangent
    wall_normal = wall.normal

    tool_angle = np.rad2deg(np.arctan2(wall_normal[1], wall_normal[0]))
    tool_angle = tool_angle % 360

    mask_area = cv2.countNonZero(mask_crop)
    if mask_area == 0:
        return False, {}

    band_width = 0.2  # normalized units

    center = wall.origin + wall_normal * (band_width / 2)
    center_px = np.array([int(center[0] * w), int((1 - center[1]) * h)])  # y-flip: robot -> image
    tangent_px = wall_vec * np.array([w, -h])
    normal_px = wall_normal * np.array([w, -h])

    half_length = 0.5
    half_width = band_width / 2
    corners = np.array(
        [
            center_px - tangent_px * half_length + normal_px * half_width,
            center_px + tangent_px * half_length + normal_px * half_width,
            center_px + tangent_px * half_length - normal_px * half_width,
            center_px - tangent_px * half_length - normal_px * half_width,
        ],
        dtype=np.int32,
    )

    band_mask = np.zeros_like(mask_crop, dtype=np.uint8)
    cv2.rectangle(
        band_mask,
        tuple(np.clip(corners[0], 0, [w - 1, h - 1])),
        tuple(np.clip(corners[2], 0, [w - 1, h - 1])),
        255,
        thickness=-1,
    )

    band_area = cv2.countNonZero(band_mask)
    if band_area == 0:
        return False, {}

    occupied_area = cv2.countNonZero(cv2.bitwise_and(mask_crop, band_mask))
    if debug is not None:
        debug.save(f"band_mask_and_mask_at_{wall.label}", cv2.bitwise_and(mask_crop, band_mask))

    threshold = 0.1
    occupancy_ratio = occupied_area / mask_area
    logger.debug("Wall %s: occupancy ratio in band = %.2f", wall.label, occupancy_ratio)
    if occupancy_ratio <= threshold:
        return False, {}

    logger.debug("Wall %s needs reset: occupancy ratio %.2f", wall.label, occupancy_ratio)

    if tool_dims_px is None:
        logger.debug("Wall %s: no tool dimensions available, using fallback", wall.label)
        return True, {"use_fallback": True, "wall": wall}

    tool_width, tool_length = tool_dims_px
    min_distance_threshold = 5

    half_width_px = tool_width * 1.5
    center = wall.origin + wall_normal * (half_width_px / np.max([w, h]))
    center_px = np.array([int(center[0] * w), int((1 - center[1]) * h)])
    x_range = (
        center_px[0]
        + np.array([-half_width_px, half_width_px]) * wall_normal[0]
        + np.array([-0.5, 0.5]) * w * wall_vec[0]
    )
    y_range = (
        center_px[1]
        + np.array([-half_width_px, half_width_px]) * wall_normal[1]
        + np.array([-0.5, 0.5]) * h * wall_vec[1]
    )
    if wall.label in ("left", "right"):
        x_range = np.clip(x_range, tool_width, w - tool_width)
        y_range = np.clip(y_range, tool_length * 0.6, h - tool_length * 0.6)
    else:
        y_range = np.clip(y_range, tool_width, h - tool_width)
        x_range = np.clip(x_range, tool_length * 0.6, w - tool_length * 0.6)
    search_region_px = (
        tuple(map(int, np.sort(x_range))),
        tuple(map(int, np.sort(y_range))),
    )

    free_region = find_tool_placements(
        mask_crop,
        (tool_width, tool_length),
        [tool_angle],
        search_region_px,
        min_clearance_px=min_distance_threshold,
        debug=debug,
    )

    if free_region is None:
        logger.debug("Wall %s: no sufficient free space found, using fallback", wall.label)
        return True, {"use_fallback": True, "wall": wall}

    pos_px = free_region["pos_px"]
    pos_robot = frames.crop_px_to_robot(*pos_px)
    logger.debug("Wall %s: found free region at pos_px=%s", wall.label, pos_px)
    return True, {
        "use_fallback": False,
        "wall": wall,
        "pos_px": pos_px,
        "pos_robot": pos_robot,
        "angle": free_region["angle"],
        "min_distance": free_region["clearance_px"],
    }


def check_center_reset_needed(mask_crop: np.ndarray) -> bool:
    """Is less than 30% of the granule mass inside the central workspace region? Ported from
    `object_tracker/granular_utils.py::check_reset_needed` (see module docstring for the
    `occupancy_ratio = occupied_area / mask_area` quirk this preserves verbatim)."""
    try:
        mask_area = cv2.countNonZero(mask_crop) if mask_crop is not None else 0
        if mask_area == 0:
            return False
        h, w = mask_crop.shape
        x_start = int(w * 0.3)
        x_end = int(w * 0.7)
        y_start = int(h * 0.3)
        y_end = int(h * 0.7)
        workspace_mask = np.zeros_like(mask_crop, dtype=np.uint8)
        cv2.rectangle(workspace_mask, (x_start, y_start), (x_end, y_end), 255, thickness=-1)
        workspace_area = cv2.countNonZero(workspace_mask)
        if workspace_area == 0:
            return False
        occupied_area = cv2.countNonZero(cv2.bitwise_and(mask_crop, workspace_mask))

        occupancy_ratio = occupied_area / mask_area
        threshold = 0.3
        if occupancy_ratio < threshold:
            logger.debug("Main workspace needs reset: occupancy ratio %.2f", occupancy_ratio)
            return True
        logger.debug("Main workspace doesn't need reset: occupancy ratio %.2f", occupancy_ratio)
        return False
    except Exception as exc:  # noqa: BLE001 - legacy was equally broad; log instead of print
        logger.warning("check_center_reset_needed: error checking reset need: %r", exc)
        return False


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


class Workspace:
    """Static workspace geometry: fence walls, manipulation boundary, tool dims — built once from
    `config_schema.WorkspaceConfig` + `perception.frames.CoordinateFrames` and shared read-only by
    every planner call (design 02 §3.3: "workspace: Workspace # fence walls, manipulation
    boundary, tool dims (static)")."""

    def __init__(self, config: "WorkspaceConfig", frames: "CoordinateFrames") -> None:
        self.config = config
        self.frames = frames
        self.walls: List[Wall] = build_fence_walls(
            fence_center=config.fence_center,
            fence_size=config.fence_size,
            tool_length=config.tool_length_robot,
            tool_width=config.tool_width_robot,
            safety_margin=config.safety_margin,
        )
        self._walls_by_label: Dict[str, Wall] = {w.label: w for w in self.walls}

    def wall(self, label: str) -> Wall:
        """Look up a wall by label (`"top"`/`"right"`/`"bottom"`/`"left"`)."""
        return self._walls_by_label[label]

    @property
    def manip_x(self) -> Tuple[float, float]:
        """Manipulation boundary x range, robot frame."""
        return self.config.manipulation_boundary_robot.x

    @property
    def manip_y(self) -> Tuple[float, float]:
        """Manipulation boundary y range, robot frame."""
        return self.config.manipulation_boundary_robot.y

    @property
    def manipulation_boundary_robot(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """`(x_range, y_range)`, robot frame."""
        return (self.manip_x, self.manip_y)

    @property
    def manipulation_boundary_px(self) -> PxBox:
        """`((xmin, xmax), (ymin, ymax))`, crop-pixel frame. Uses
        `config.manipulation_boundary_px` verbatim if given; otherwise derives it from
        `manipulation_boundary_robot` via `frames.robot_to_crop_px` (legacy always required this
        as an explicit config value — `image_space_manipulation_boundary` — with no derivation
        path; this fallback is new but a straightforward consequence of having `CoordinateFrames`
        available)."""
        if self.config.manipulation_boundary_px is not None:
            xmin, xmax, ymin, ymax = self.config.manipulation_boundary_px
            return ((xmin, xmax), (ymin, ymax))
        (x0, x1) = self.manip_x
        (y0, y1) = self.manip_y
        corners = [
            self.frames.robot_to_crop_px(x, y) for x in (x0, x1) for y in (y0, y1)
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return ((min(xs), max(xs)), (min(ys), max(ys)))

    @property
    def tool_dims_robot(self) -> Tuple[float, float]:
        """`(tool_length, tool_width)`, robot-normalized units."""
        return (self.config.tool_length_robot, self.config.tool_width_robot)

    @property
    def tool_dims_px(self) -> Tuple[int, int]:
        """`(w_px, h_px)`, crop-pixel frame — matches `fence_utils.make_tool_mask`'s
        `(w_px, h_px)` argument order."""
        return self.config.tool_dims_px
