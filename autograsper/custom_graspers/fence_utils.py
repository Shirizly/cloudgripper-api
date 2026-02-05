from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class Wall:
    label: str             # wall label (e.g., 'top', 'right', etc.)
    origin: np.ndarray      # reference point on wall (center of wall)
    tangent: np.ndarray     # unit vector along wall
    normal: np.ndarray      # unit vector perpendicular to wall
    t_min: float            # minimum slide parameter
    t_max: float            # maximum slide parameter
    angle: float            # tool orientation angle (radians)

def build_fence_walls(
    fence_center,
    fence_size,
    tool_length,
    tool_width,
    safety_margin
):
    fx, fy = fence_center
    h = [fence_size / 2 for fence_size in fence_size]
    hx,hy = h

    half_tool_len = tool_length / 2
    half_tool_wid = tool_width / 2

    walls = []

    # Define walls in CCW order
    wall_defs = [
        # top
        ('top', np.array([fx, fy + hy]), np.array([1, 0]), np.array([0, -1])),
        # right
        ('right', np.array([fx + hx, fy]), np.array([0, -1]), np.array([-1, 0])),
        # bottom
        ('bottom', np.array([fx, fy - hy]), np.array([-1, 0]), np.array([0, 1])),
        # left
        ('left', np.array([fx - hx, fy]), np.array([0, 1]), np.array([1, 0])),
    ]

    for label, origin, tangent, normal in wall_defs:
        tangent = tangent / np.linalg.norm(tangent)
        normal = normal / np.linalg.norm(normal)

        # valid sliding range
        t_min = -abs(np.inner(h, tangent)) + half_tool_len + safety_margin
        t_max =  abs(np.inner(h, tangent)) - half_tool_len - safety_margin

        angle = np.arctan2(tangent[1], tangent[0])
        angle = (angle+np.pi/2) % np.pi
        angle = angle/np.pi*180  # convert to degrees
        angle = angle + 180 if angle < 0 else angle
        angle = 0 if angle==180 else angle
        angle = int(angle)

        walls.append(
            Wall(
                label=label,
                origin=origin,
                tangent=tangent,
                normal=normal,
                t_min=t_min,
                t_max=t_max,
                angle=angle
            )
        )

    return walls

def sample_tool_pose(wall: Wall, tool_width, safety_margin, t=None):
    # slide along wall
    if t is None:
        t = np.random.uniform(wall.t_min, wall.t_max)
    else:
        t = np.clip(t, wall.t_min, wall.t_max)

    # offset from wall (exact contact + margin)
    offset = tool_width / 2 + safety_margin

    center = (
        wall.origin
        + wall.tangent * t
        + wall.normal * offset
    )

    return {
        "x": center[0],
        "y": center[1],
        "angle": wall.angle,
        "parallel_dir": wall.tangent,
        "perpendicular_dir": wall.normal,
    }




def get_pos_sweep_from_optimal(pos_optimal, wall, margin=0.02):
    # Move towards wall to contact + margin
    optimal_dist = np.dot(pos_optimal - wall.origin, wall.normal)
    desired_dist = margin
    delta_dist = optimal_dist - desired_dist

    pos_sweep = pos_optimal - wall.normal * delta_dist
    return pos_sweep

############################################## New versions of checking empty placements #######################################################
class PixelRobotTransform:
    def __init__(self, H_pix_to_robot):
        self.H_p2r = H_pix_to_robot
        self.H_r2p = np.linalg.inv(H_pix_to_robot)

    def pix_to_robot(self, u, v):
        p = np.array([u, v, 1.0])
        x, y, w = self.H_p2r @ p
        return x / w, y / w

    def robot_to_pix(self, x, y):
        p = np.array([x, y, 1.0])
        u, v, w = self.H_r2p @ p
        return int(u / w), int(v / w)


def check_placement(dist, tool_mask, cx, cy, r):
    ys = slice(cy - r, cy + r + 1)
    xs = slice(cx - r, cx + r + 1)

    # Clip slices to valid bounds
    y_start = max(0, ys.start)
    y_stop = min(dist.shape[0], ys.stop)
    x_start = max(0, xs.start)
    x_stop = min(dist.shape[1], xs.stop)

    # Calculate offsets in the tool mask for clipped region
    y_offset = y_start - (cy - r)
    x_offset = x_start - (cx - r)
    
    local_dist = dist[y_start:y_stop, x_start:x_stop]
    local_mask = tool_mask[y_offset:y_offset + (y_stop - y_start), 
                           x_offset:x_offset + (x_stop - x_start)]

    overlap = np.any((local_mask == 1) & (local_dist == 0))
    if overlap:
        return False, 0.0

    clearance = np.min(local_dist[local_mask == 1])
    return True, clearance

def make_tool_mask(w_px, h_px, angle_deg):
    # Create a binary mask of the tool at the given angle
    # tool is vertical (along height) in angle_deg=0
    r = int(np.ceil(0.5 * np.hypot(h_px, w_px)))
    pad = 2 * r + 1

    mask = np.zeros((pad, pad), dtype=np.uint8)

    center = (r, r)

    rect = (center, (w_px, h_px), float(-angle_deg)) # negative angle for cv2 convention to match robot coordinate system
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(mask, box, 1)
    cv2.imwrite(f'tool_mask_{w_px}x{h_px}_angle{angle_deg}.png', mask*255)
    return mask, r

def in_region_pix(x, y, region):
    (xmin, xmax), (ymin, ymax) = region
    return xmin <= x <= xmax and ymin <= y <= ymax

def find_tool_placements(
    obstacle_mask,
    tool_dims_px,
    angles_deg,
    search_region_pix,
    MIN_CLEARANCE_PX = 10
):
    free = ~obstacle_mask
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)

    h_img, w_img = dist.shape
    placements = []

    for angle in angles_deg:
        tool_mask, r = make_tool_mask(tool_dims_px[0], tool_dims_px[1], angle)

        # Apply search region mask before finding candidates
        (xmin, xmax), (ymin, ymax) = search_region_pix
        search_dist = np.zeros_like(dist)
        search_dist[ymin:ymax+1, xmin:xmax+1] = dist[ymin:ymax+1, xmin:xmax+1]
        
        # prune: tool must fit at center (within search region only)
        candidates = np.argwhere(search_dist > int(np.min(tool_dims_px)/2 + 1))
        if candidates.size == 0:
            continue

        for cy, cx in candidates:

            ok, clearance = check_placement(dist, tool_mask, cx, cy, r)
            if not ok:
                continue

            placements.append({
                "pos_px": (cx, cy),
                "angle": angle,
                "clearance_px": clearance
            })

            if clearance >= MIN_CLEARANCE_PX:
                break

            
    if not placements:
        return {}
    placements.sort(key=lambda p: -p["clearance_px"])
    return placements[0]


def check_wall_reset_needed(mask, wall: Wall, stats, image_space_tool_dimensions, min_granule_size, margin_of_safety = 0.03):
    """
    Identify regions along the wall where mask 1s are sufficiently far from the wall.
    Returns (reset_needed, details) where details contains optimal position info.
    """
    if mask is None or stats is None:
        return False, {}
    
    h, w = mask.shape
    wall_vec = wall.tangent
    wall_normal = wall.normal

    tool_angle = np.rad2deg(np.arctan2(wall_normal[1], wall_normal[0]))  # degrees
    tool_angle = tool_angle % 360 # tool is vertical at 0 degrees, but mask generation corrects for this



    mask_area = cv2.countNonZero(mask)
    if mask_area == 0:
        return False, {}

    # Define a band along the wall
    band_width = 0.2  # in normalized units 
    
    # Translate band's rectangle into image coordinates
    center = wall.origin + wall_normal * (band_width / 2)
    center_px = np.array([int(center[0] * w), int((1 - center[1]) * h)])  # flip y for image coords
    tangent_px = wall_vec * np.array([w, -h])  # flip y for image coords
    normal_px = wall_normal * np.array([w, -h])  # flip y for image coords

    half_length = 0.5  # half of normalized wall extent
    half_width = band_width / 2
    corners = np.array([
        center_px - tangent_px * half_length + normal_px * half_width,
        center_px + tangent_px * half_length + normal_px * half_width,
        center_px + tangent_px * half_length - normal_px * half_width,
        center_px - tangent_px * half_length - normal_px * half_width,
    ], dtype=np.int32)
    
    band_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.rectangle(band_mask, tuple(np.clip(corners[0], 0, [w-1, h-1])), tuple(np.clip(corners[2], 0, [w-1, h-1])), 255, thickness=-1)
    
    # Calculate occupancy in the band
    band_area = cv2.countNonZero(band_mask)
    if band_area == 0:
        return False, {}
    
    occupied_area = cv2.countNonZero(cv2.bitwise_and(mask, band_mask))
    cv2.imwrite(f'band mask and mask at {wall.label}.png',cv2.bitwise_and(mask, band_mask))
    # If occupancy ratio exceeds threshold, reset is needed
    threshold = 0.1  # if more than 10% of mask is in the band, needs sweeping
    occupancy_ratio = occupied_area / mask_area
    if occupancy_ratio <= threshold:
        return False, {}
    
    print(f"Wall {wall.label} needs reset: occupancy ratio {occupancy_ratio:.2f}")
    
    # Find free regions along the wall where mask 1s are sufficiently far
    if image_space_tool_dimensions is None:
        # Fallback: no tool information exist
        print(f"Wall {wall.label}: No tool dimensions available, using fallback")
        return True, {"use_fallback": True, "wall": wall}
    
    tool_width, tool_length = image_space_tool_dimensions
    min_distance_threshold = 5  # margin of safety
    
    # bring search region closer to wall:
    half_width = tool_width * 1.5
    center = wall.origin + wall_normal * (half_width / np.max([w, h]))  # move center closer to wall, scaled to image size)
    center_px = np.array([int(center[0] * w), int((1 - center[1]) * h)])  # flip y for image coords
    x_range = center_px[0] + np.array([-half_width, half_width])*wall_normal[0] + np.array([-0.5, 0.5])*w*wall_vec[0]
    y_range = center_px[1] + np.array([-half_width, half_width])*wall_normal[1] + np.array([-0.5, 0.5])*h*wall_vec[1]
    if wall.label in ['left', 'right']:
        x_range = np.clip(x_range, tool_width, w-tool_width)  # ensure search region is within image bounds considering tool width
        y_range = np.clip(y_range, tool_length*0.6, h-tool_length*0.6)  # ensure search region is within image bounds considering tool length
    else:
        y_range = np.clip(y_range, tool_width, h-tool_width) # ensure search region is within image bounds considering tool width
        x_range = np.clip(x_range, tool_length*0.6, w-tool_length*0.6)  # ensure search region is within image bounds considering tool length
    search_region_pix = (tuple(map(int, np.sort(x_range))), tuple(map(int, np.sort(y_range))))
    


    # Scan along wall and find regions far enough from masked obstacles
    free_region = find_tool_placements(mask,[tool_length,tool_width],[tool_angle],
                                       search_region_pix,
                                       MIN_CLEARANCE_PX = min_distance_threshold)
    
    
    
    if isinstance(free_region, list) or not free_region:
        # No sufficient free space exists
        print(f"Wall {wall.label}: No sufficient free space found, using fallback")
        return True, {"use_fallback": True, "wall": wall}
    
    # Find the largest contiguous free region and return the t-value that maximizes minimal distance
    # best_region = max(free_regions, key=lambda r: r["min_distance"])
    
    print(f"Wall {wall.label}: Found free region at pos_px={free_region['pos_px']}")


    return True, {
        "use_fallback": False,
        "wall": wall,
        "pos_px": free_region["pos_px"],
        "angle": free_region["angle"],
        "min_distance": free_region["clearance_px"]
    }


if __name__ == "__main__":
    # create and save matrices H and Hinv based on some manual data for robot to pix homography
    points_rob = np.array([[0.21,0.95],
                  [0.83,0.94],
                  [0.2,0.04],
                  [0.83,0.04] 
                  ])
    points_pix = np.array([[62,4],
                  [301,4],
                  [62,350],
                  [301,350] 
                  ])
    H,_ = cv2.findHomography(points_pix, points_rob)
    np.savez("homography.npz", H)

