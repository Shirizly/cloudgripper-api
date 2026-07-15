"""
create_homography_calibration.py

Offline tool to (re)create the ``homography.npz`` file used by
``PixelRobotTransform`` in the robot pipeline.

It displays a reference bottom-camera image and asks you to click on four
known workspace positions.  For each click the corresponding robot
normalised coordinates (x_norm ∈ [0,1], y_norm ∈ [0,1]) are provided either:

  (a) automatically, from a ``fence_calibration_<robot>.npz`` file, or
  (b) by typing them interactively in the terminal.

The resulting 3 × 3 homography  H_pix_to_robot  is saved to
``homography.npz`` as ``arr_0``, matching the format expected by

    pixTransH = np.load("homography.npz")["arr_0"]
    PixelRobotTransform(pixTransH)

Usage examples
--------------
# Using a fence-calibration file (recommended):
python create_homography_calibration.py \\
    --image      latest_base.jpg \\
    --fence_cal  fence_calibration_robot24.npz \\
    --output     homography.npz

# Manual entry (you will be prompted in the terminal):
python create_homography_calibration.py \\
    --image      latest_base.jpg \\
    --output     homography.npz

Click order
-----------
The script prompts for exactly 4 clicks in this order:

    1. TOP-LEFT     of the workspace  →  robot corner 1
    2. TOP-RIGHT    of the workspace  →  robot corner 2
    3. BOTTOM-RIGHT of the workspace  →  robot corner 3
    4. BOTTOM-LEFT  of the workspace  →  robot corner 4

When using a fence-calibration file the robot positions for the four
corners are taken from ``horizontal_corners[0..3]`` (the four positions
recorded during horizontal-tool calibration, in TL/TR/BR/BL order).
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# Default output path
DEFAULT_OUTPUT = "homography.npz"

# Labels and description of the 4 clicks shown on the image
CORNER_LABELS = [
    "1 – TOP-LEFT",
    "2 – TOP-RIGHT",
    "3 – BOTTOM-RIGHT",
    "4 – BOTTOM-LEFT",
]

# Overlay colours
CLICK_COLOUR  = (0, 255, 0)    # BGR green for confirmed clicks
HOVER_COLOUR  = (0, 200, 255)  # BGR amber for the crosshair
TEXT_COLOUR   = (255, 255, 255)
FONT          = cv2.FONT_HERSHEY_SIMPLEX

# ============================================================


def draw_overlay(canvas: np.ndarray, clicks: list, next_label: str) -> np.ndarray:
    """Return a copy of canvas with recorded clicks and the current prompt."""
    out = canvas.copy()
    for i, (u, v) in enumerate(clicks):
        cv2.circle(out, (u, v), 8, CLICK_COLOUR, -1)
        cv2.putText(out, CORNER_LABELS[i], (u + 10, v - 10),
                    FONT, 0.55, CLICK_COLOUR, 1, cv2.LINE_AA)
    if next_label:
        cv2.putText(out, f"Click: {next_label}", (10, 30),
                    FONT, 0.7, TEXT_COLOUR, 2, cv2.LINE_AA)
        cv2.putText(out, "Press 'u' to undo last click  |  'q' to quit",
                    (10, 60), FONT, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    else:
        cv2.putText(out, "All 4 corners recorded – press ENTER to confirm or 'u' to undo",
                    (10, 30), FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def collect_pixel_clicks(image: np.ndarray) -> list[tuple[int, int]]:
    """
    Interactive window: collect exactly 4 pixel positions by left-clicking.
    Returns list of (u, v) tuples (column, row).
    """
    clicks: list[tuple[int, int]] = []
    mouse_pos = [0, 0]

    def on_mouse(event, x, y, flags, param):
        mouse_pos[0], mouse_pos[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            print(f"  Recorded pixel ({x}, {y}) for {CORNER_LABELS[len(clicks)-1]}")

    win = "Homography Calibration  [click 4 corners]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        remaining = CORNER_LABELS[len(clicks)] if len(clicks) < 4 else ""
        frame = draw_overlay(image, clicks, remaining)

        # Draw crosshair at current mouse position
        u, v = mouse_pos
        cv2.line(frame, (u - 15, v), (u + 15, v), HOVER_COLOUR, 1)
        cv2.line(frame, (u, v - 15), (u, v + 15), HOVER_COLOUR, 1)

        cv2.imshow(win, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('u') and clicks:
            removed = clicks.pop()
            print(f"  Undid click at {removed}")
        elif key == ord('q'):
            cv2.destroyWindow(win)
            sys.exit("Aborted by user.")
        elif key == 13 and len(clicks) == 4:  # Enter
            break

    cv2.destroyWindow(win)
    return clicks


def get_robot_coords_from_fence(fence_path: str) -> list[tuple[float, float]]:
    """
    Load the four workspace-corner robot positions from a fence calibration file.
    Returns [(x0,y0), (x1,y1), (x2,y2), (x3,y3)] in TL/TR/BR/BL order.
    """
    data = np.load(fence_path, allow_pickle=True)
    corners = data["horizontal_corners"]  # shape (4, 2): TL, TR, BR, BL
    result = [(float(c[0]), float(c[1])) for c in corners]
    print("Robot corner positions loaded from fence calibration:")
    for label, rc in zip(CORNER_LABELS, result):
        print(f"  {label}  →  robot ({rc[0]:.4f}, {rc[1]:.4f})")
    return result


def get_robot_coords_interactive() -> list[tuple[float, float]]:
    """Ask the user to type the robot (x_norm, y_norm) for each of the 4 corners."""
    print("\nEnter robot normalised coordinates (0.0 – 1.0) for each corner.")
    print("If the workspace corners map to (0,0),(1,0),(1,1),(0,1) just press Enter.\n")

    defaults = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    coords = []
    for i, label in enumerate(CORNER_LABELS):
        dx, dy = defaults[i]
        raw = input(f"  {label}  x_norm, y_norm  [default {dx},{dy}]: ").strip()
        if raw == "":
            coords.append((dx, dy))
        else:
            try:
                parts = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]
                x, y = float(parts[0]), float(parts[1])
                coords.append((x, y))
            except (ValueError, IndexError):
                sys.exit(f"Could not parse '{raw}' as two floats.")
    return coords


def compute_homography(
    pixel_points: list[tuple[int, int]],
    robot_points: list[tuple[float, float]],
) -> np.ndarray:
    """
    Compute 3×3 homography  H  such that:
        [rx, ry, 1]^T  ∝  H @ [px, py, 1]^T
    i.e. pixel coordinates → robot normalised coordinates.
    """
    src = np.array(pixel_points, dtype=np.float64)   # (4, 2) pixel (u, v)
    dst = np.array(robot_points, dtype=np.float64)   # (4, 2) robot (x, y)
    H, mask = cv2.findHomography(src, dst)
    if H is None:
        sys.exit("findHomography failed – check that the 4 points are not collinear.")
    inliers = int(mask.sum()) if mask is not None else "?"
    print(f"\nHomography computed ({inliers}/4 inliers):\n{H}")
    return H


def verify_homography(
    H: np.ndarray,
    pixel_points: list[tuple[int, int]],
    robot_points: list[tuple[float, float]],
) -> None:
    """Print reprojection error for the 4 calibration points."""
    print("\nReprojection check (pixel → robot):")
    for i, ((px, py), (rx_gt, ry_gt)) in enumerate(zip(pixel_points, robot_points)):
        p_h = np.array([px, py, 1.0])
        r_h = H @ p_h
        rx, ry = r_h[0] / r_h[2], r_h[1] / r_h[2]
        err = np.hypot(rx - rx_gt, ry - ry_gt)
        print(f"  {CORNER_LABELS[i]}  pixel({px},{py})  "
              f"→ robot({rx:.4f},{ry:.4f})  gt({rx_gt:.4f},{ry_gt:.4f})  "
              f"err={err:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline tool to recreate homography.npz for PixelRobotTransform."
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to a reference bottom-camera image (e.g. latest_base.jpg).",
    )
    parser.add_argument(
        "--fence_cal", default=None,
        help="Optional path to fence_calibration_<robot>.npz.  "
             "If supplied, robot corner positions are read from it; "
             "otherwise you will be prompted to type them.",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output path for homography.npz (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    # --- load reference image ---
    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"Could not load image: {args.image}")
    print(f"Loaded image {args.image}  ({image.shape[1]}×{image.shape[0]} px)")

    # --- collect pixel clicks ---
    print("\nA window will open.  Click the 4 workspace corners in order:")
    for lbl in CORNER_LABELS:
        print(f"  {lbl}")
    print("Press 'u' to undo, Enter when all 4 are placed.\n")
    pixel_clicks = collect_pixel_clicks(image)

    # --- get robot positions ---
    if args.fence_cal:
        robot_coords = get_robot_coords_from_fence(args.fence_cal)
    else:
        robot_coords = get_robot_coords_interactive()

    # --- compute and save ---
    H = compute_homography(pixel_clicks, robot_coords)
    verify_homography(H, pixel_clicks, robot_coords)

    out_path = Path(args.output)
    np.savez(str(out_path), H)   # saved as arr_0, matching np.load(...)["arr_0"]
    print(f"\nSaved homography to {out_path.resolve()}")
    print(f"Load with:  np.load('{out_path}')['arr_0']")


if __name__ == "__main__":
    main()
