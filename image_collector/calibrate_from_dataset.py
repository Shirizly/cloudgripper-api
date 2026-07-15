"""
calibrate_from_dataset.py

Reconstructs ``homography.npz`` from existing recorded data, without
requiring a live robot.

How it works
------------
1.  Scans all states.json files under a dataset root and collects frames
    where the robot is at push height (z_norm ≤ Z_THRESHOLD) and a
    Bottom_Image exists.
2.  Snaps each (x_norm, y_norm) to the nearest cell of a configurable
    grid and picks the closest candidate per cell.  This gives a well-
    distributed set of calibration frames.
3.  Shows each frame in a GUI (the same crop used by segment_transitions).
    You click the pixel where the tool centre is visible.
    Controls:
        Left-click          – place / move the marker for this frame
        Enter or n          – accept marker and go to next frame
        s                   – skip this frame (no marker)
        u                   – remove marker from the last accepted frame
        q                   – stop collecting early (if ≥ MIN_POINTS accepted)
4.  Fits an overdetermined affine transform (appropriate for the linear
    pixel↔robot assumption) from all accepted clicks and saves the result
    as a 3×3 homography matrix in ``homography.npz`` (key ``arr_0``),
    compatible with ``PixelRobotTransform``.

Usage
-----
    python image_collector/calibrate_from_dataset.py \\
        --dataset_root autograsper/recorded_data/push_chickpeas50

    # Custom options:
    python image_collector/calibrate_from_dataset.py \\
        --dataset_root autograsper/recorded_data/push_chickpeas50 \\
        --output       homography.npz \\
        --grid_rows    5  --grid_cols  5 \\
        --modes        restore
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# CONFIGURATION  (override with CLI flags)
# ============================================================

Z_THRESHOLD  = 0.4       # frames with z_norm ≤ this are "at push height"

GRID_ROWS    = 5         # rows of the sampling grid
GRID_COLS    = 5         # columns of the sampling grid
MIN_POINTS   = 4         # minimum accepted clicks to compute calibration
MODES        = ("task", "restore")   # which sub-dirs to scan

# Crop window – must match segment_transitions.py / crop_center_region defaults
CROP_X1, CROP_Y1 = 20,  95
CROP_X2, CROP_Y2 = 380, 455

ZOOM_FACTOR  = 2        # magnifier zoom level
ZOOM_SIZE    = 40        # side length of the magnifier inset (pre-zoom pixels)

# ============================================================
# Colours / fonts
MARKER_COL  = (0, 255,   0)
PENDING_COL = (0, 200, 255)
SKIP_COL    = (100, 100, 100)
TEXT_COL    = (255, 255, 255)
MAP_OK_COL  = (0, 220,   0)
MAP_PEND_COL= (0, 160, 255)
FONT        = cv2.FONT_HERSHEY_SIMPLEX


# ------------------------------------------------------------------
# Step 1 – scan dataset
# ------------------------------------------------------------------

def scan_dataset(dataset_root: Path, modes: tuple) -> list[dict]:
    """Return a flat list of candidate frame records."""
    records = []
    for exp_dir in sorted(dataset_root.iterdir()):
        if not (exp_dir.is_dir() and exp_dir.name.isdigit()):
            continue
        for mode in modes:
            mode_path = exp_dir / mode
            states_file = mode_path / "states.json"
            if not states_file.exists():
                continue
            with open(states_file) as f:
                states = json.load(f)
            for s in states:
                z = s.get("z_norm", 1.0)
                if z > Z_THRESHOLD:
                    continue
                fi = s.get("frame_index")
                if fi is None:
                    continue
                img_path = mode_path / "Bottom_Images" / f"image_bottom_{fi}.jpeg"
                if not img_path.exists():
                    continue
                records.append({
                    "img_path":  img_path,
                    "x_norm":    s["x_norm"],
                    "y_norm":    s["y_norm"],
                    "z_norm":    z,
                    "exp_id":    exp_dir.name,
                    "mode":      mode,
                    "frame_idx": fi,
                })
    return records


def select_grid_frames(
    records: list[dict],
    grid_rows: int,
    grid_cols: int,
) -> list[dict]:
    """
    For each cell of a grid_rows × grid_cols grid, pick the frame whose
    (x_norm, y_norm) is closest to the cell centre.
    Returns one record per occupied cell, sorted by (row, col).
    """
    best: dict[tuple, dict] = {}
    for rec in records:
        x, y = rec["x_norm"], rec["y_norm"]
        col = min(int(x * grid_cols), grid_cols - 1)
        row = min(int(y * grid_rows), grid_rows - 1)
        cx  = (col + 0.5) / grid_cols
        cy  = (row + 0.5) / grid_rows
        dist = (x - cx) ** 2 + (y - cy) ** 2
        key = (row, col)
        if key not in best or dist < best[key]["_dist"]:
            rec["_dist"]   = dist
            rec["_grid_rc"] = key
            best[key] = rec

    return [v for _, v in sorted(best.items())]


# ------------------------------------------------------------------
# Step 2 – GUI helpers
# ------------------------------------------------------------------

def crop_image(img: np.ndarray) -> np.ndarray:
    return img[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]


def draw_mini_map(
    canvas: np.ndarray,
    frames: list[dict],
    current_idx: int,
    accepted: list[int],
    map_size: int = 120,
) -> None:
    """Draw a small workspace coverage map in the top-right corner."""
    top_right_x = canvas.shape[1] - map_size - 10
    top_right_y = 10
    m = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    cv2.rectangle(m, (0, 0), (map_size - 1, map_size - 1), (60, 60, 60), 1)

    for i, fr in enumerate(frames):
        px = int(fr["x_norm"] * (map_size - 1))
        py = int(fr["y_norm"] * (map_size - 1))
        if i in accepted:
            col = MAP_OK_COL
            r = 4
        elif i == current_idx:
            col = MAP_PEND_COL
            r = 5
        else:
            col = (50, 50, 50)
            r = 3
        cv2.circle(m, (px, py), r, col, -1)

    canvas[top_right_y:top_right_y + map_size,
           top_right_x:top_right_x + map_size] = m
    cv2.putText(canvas, "Workspace", (top_right_x, top_right_y - 4),
                FONT, 0.4, (160, 160, 160), 1, cv2.LINE_AA)


def draw_magnifier(
    canvas: np.ndarray,
    src: np.ndarray,
    mx: int,
    my: int,
    zoom_factor: int,
    zoom_size: int,
) -> None:
    """Draw a zoomed inset of `src` near (mx, my) onto canvas."""
    h, w = src.shape[:2]
    half = int(zoom_size) // 2
    x1 = max(0, mx - half);    x2 = min(w, mx + half)
    y1 = max(0, my - half);    y2 = min(h, my + half)
    patch = src[y1:y2, x1:x2]
    if patch.size == 0:
        return
    zf  = int(zoom_factor)
    zh  = int(y2 - y1) * zf
    zw  = int(x2 - x1) * zf
    if zh <= 0 or zw <= 0:
        return
    zoomed = cv2.resize(patch, (zw, zh), interpolation=cv2.INTER_NEAREST)
    # crosshair in zoomed patch
    cxz = int(mx - x1) * zf
    cyz = int(my - y1) * zf
    cv2.line(zoomed, (cxz - 15, cyz), (cxz + 15, cyz), PENDING_COL, 1)
    cv2.line(zoomed, (cxz, cyz - 15), (cxz, cyz + 15), PENDING_COL, 1)
    # place in bottom-left corner
    bx = 10
    by = max(0, canvas.shape[0] - zh - 10)
    ch = min(zh, canvas.shape[0] - by)
    cw = min(zw, canvas.shape[1] - bx)
    canvas[by:by + ch, bx:bx + cw] = zoomed[:ch, :cw]
    cv2.rectangle(canvas, (bx - 1, by - 1), (bx + cw, by + ch), PENDING_COL, 1)


def run_gui(frames: list[dict]) -> list[dict]:
    """
    Present frames one at a time.  Click the two endpoints of the tool;
    the midpoint is used as the calibration pixel.  Enter to accept.
    Returns a list of accepted dicts each with added keys 'pix_x', 'pix_y'.
    """
    accepted: list[dict] = []   # fully confirmed records
    accepted_idx: list[int] = []

    win = "Calibration  |  click tool endpoints (2 clicks)  |  Enter=accept  s=skip  u=undo  q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 800, 650)

    mouse = {"x": 0, "y": 0, "p1": None, "p2": None}

    def on_mouse(event, x, y, flags, _):
        mouse["x"], mouse["y"] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            if mouse["p1"] is None:
                mouse["p1"] = (x, y)
            elif mouse["p2"] is None:
                mouse["p2"] = (x, y)
            else:
                # Third click: restart endpoint selection
                mouse["p1"] = (x, y)
                mouse["p2"] = None

    cv2.setMouseCallback(win, on_mouse)

    idx = 0
    while idx < len(frames):
        fr = frames[idx]
        raw = cv2.imread(str(fr["img_path"]))
        if raw is None:
            idx += 1
            continue
        img = crop_image(raw)

        while True:
            canvas = img.copy()
            mx, my = mouse["x"], mouse["y"]
            n_acc = len(accepted)
            n_tot = len(frames)

            # --- top-left status text ---
            lines = [
                f"Frame {idx+1}/{n_tot}  |  accepted: {n_acc}",
                f"robot  x={fr['x_norm']:.3f}  y={fr['y_norm']:.3f}  z={fr['z_norm']:.3f}",
                f"exp={fr['exp_id']}  mode={fr['mode']}  frame={fr['frame_idx']}",
                "Click 2 endpoints of tool  |  Enter/n=accept   s=skip   u=undo   q=quit",
            ]
            for i, line in enumerate(lines):
                cv2.putText(canvas, line, (10, 18 + i * 20),
                            FONT, 0.48, TEXT_COL, 1, cv2.LINE_AA)

            # --- endpoint markers and midpoint indicator ---
            p1 = mouse["p1"]
            p2 = mouse["p2"]
            if p1 is not None and p2 is not None:
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                cv2.line(canvas, p1, p2, PENDING_COL, 1)
                cv2.drawMarker(canvas, p1, PENDING_COL, cv2.MARKER_CROSS, 10, 1)
                cv2.drawMarker(canvas, p2, PENDING_COL, cv2.MARKER_CROSS, 10, 1)
                cv2.drawMarker(canvas, mid, MARKER_COL, cv2.MARKER_CROSS, 16, 2)
                cv2.putText(canvas, f"mid=({mid[0]},{mid[1]})", (mid[0] + 8, mid[1] - 8),
                            FONT, 0.4, MARKER_COL, 1)
            elif p1 is not None:
                cv2.drawMarker(canvas, p1, PENDING_COL, cv2.MARKER_CROSS, 10, 1)
                cv2.line(canvas, p1, (mx, my), PENDING_COL, 1)
            else:
                # crosshair following cursor
                cv2.line(canvas, (mx - 12, my), (mx + 12, my), PENDING_COL, 1)
                cv2.line(canvas, (mx, my - 12), (mx, my + 12), PENDING_COL, 1)

            draw_magnifier(canvas, img, mx, my, ZOOM_FACTOR, ZOOM_SIZE)
            draw_mini_map(canvas, frames, idx, accepted_idx)

            cv2.imshow(win, canvas)

            key = cv2.waitKey(20) & 0xFF

            if key in (13, ord('n')):   # Enter or n → accept
                if p1 is not None and p2 is not None:
                    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    rec = dict(fr)
                    rec["pix_x"] = mid[0]
                    rec["pix_y"] = mid[1]
                    accepted.append(rec)
                    accepted_idx.append(idx)
                    print(f"  [accepted]  frame {idx+1}  "
                          f"endpoints=({p1[0]},{p1[1]})->({p2[0]},{p2[1]})  "
                          f"mid=({mid[0]},{mid[1]})  "
                          f"robot=({fr['x_norm']:.3f},{fr['y_norm']:.3f})")
                    mouse["p1"] = mouse["p2"] = None
                    idx += 1
                    break
                elif p1 is not None:
                    print("  Click the second endpoint of the tool first.")
                else:
                    print("  Click both endpoints of the tool first.")

            elif key == ord('s'):       # skip
                print(f"  [skip]  frame {idx+1}")
                mouse["p1"] = mouse["p2"] = None
                idx += 1
                break

            elif key == ord('u'):       # undo last accepted
                if accepted:
                    removed = accepted.pop()
                    accepted_idx.pop()
                    print(f"  [undo]  removed mid=({removed['pix_x']},{removed['pix_y']})")
                    # go back to that frame
                    idx = frames.index(removed)
                    mouse["p1"] = mouse["p2"] = None
                    break

            elif key == ord('q'):       # quit early
                if len(accepted) >= MIN_POINTS:
                    print(f"\nStopping early with {len(accepted)} accepted points.")
                    idx = len(frames)  # exit outer loop
                    break
                else:
                    print(f"  Need at least {MIN_POINTS} points before quitting "
                          f"(have {len(accepted)}).")

    cv2.destroyWindow(win)
    return accepted


# ------------------------------------------------------------------
# Step 3 – fit calibration
# ------------------------------------------------------------------

def fit_affine_homography(accepted: list[dict]) -> np.ndarray:
    """
    Fit an overdetermined affine transform:
        [x_robot]   [a  b  c] [u_pix]
        [y_robot] = [d  e  f] [v_pix]
        [   1   ]   [0  0  1] [  1  ]

    Uses numpy least-squares for robustness with many points.
    Returns the 3×3 matrix.
    """
    n = len(accepted)
    # Build system A·t = b  where  t = [a,b,c,d,e,f]
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    for i, rec in enumerate(accepted):
        u, v = rec["pix_x"], rec["pix_y"]
        rx, ry = rec["x_norm"], rec["y_norm"]
        A[2*i]     = [u, v, 1, 0, 0, 0]
        A[2*i + 1] = [0, 0, 0, u, v, 1]
        b[2*i]     = rx
        b[2*i + 1] = ry

    t, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    a, bv, c, d, e, f = t

    H = np.array([
        [a,  bv, c],
        [d,  e,  f],
        [0., 0., 1.],
    ])
    return H


def report_errors(H: np.ndarray, accepted: list[dict]) -> None:
    errors = []
    for rec in accepted:
        u, v = rec["pix_x"], rec["pix_y"]
        p = np.array([u, v, 1.0])
        r = H @ p
        rx_hat, ry_hat = r[0] / r[2], r[1] / r[2]
        err = np.hypot(rx_hat - rec["x_norm"], ry_hat - rec["y_norm"])
        errors.append(err)
        print(f"  pixel({u:3d},{v:3d}) → robot_hat({rx_hat:.4f},{ry_hat:.4f})  "
              f"gt({rec['x_norm']:.4f},{rec['y_norm']:.4f})  err={err:.5f}")
    print(f"\n  Mean error: {np.mean(errors):.5f}  Max: {np.max(errors):.5f}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct homography.npz from existing recorded dataset images."
    )
    parser.add_argument("--dataset_root", required=True,
                        help="Root containing numbered experiment directories.")
    parser.add_argument("--output", default="homography.npz",
                        help="Output path for homography.npz (default: homography.npz).")
    parser.add_argument("--grid_rows", type=int, default=GRID_ROWS)
    parser.add_argument("--grid_cols", type=int, default=GRID_COLS)
    parser.add_argument("--modes", nargs="+", default=list(MODES),
                        help="Sub-dirs to scan (default: task restore).")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        sys.exit(f"dataset_root not found: {dataset_root}")

    print(f"Scanning {dataset_root} …")
    records = scan_dataset(dataset_root, tuple(args.modes))
    print(f"Found {len(records)} low-height frames with images.")

    frames = select_grid_frames(records, args.grid_rows, args.grid_cols)
    print(f"Selected {len(frames)} frames covering a "
          f"{args.grid_rows}×{args.grid_cols} grid.\n")

    if len(frames) < MIN_POINTS:
        sys.exit(f"Not enough frames ({len(frames)}) to calibrate.")

    print("Controls: Left-click=place marker | Enter/n=accept | s=skip | u=undo | q=quit early\n")
    accepted = run_gui(frames)

    if len(accepted) < MIN_POINTS:
        sys.exit(f"Only {len(accepted)} point(s) accepted – need at least {MIN_POINTS}.")

    print(f"\nFitting affine transform to {len(accepted)} points …")
    H = fit_affine_homography(accepted)
    print("\nReprojection errors (in robot normalised units):")
    report_errors(H, accepted)

    out_path = Path(args.output)
    np.savez(str(out_path), H)   # arr_0 = H, matching np.load(...)["arr_0"]
    print(f"\nSaved to {out_path.resolve()}")
    print(f"Load with:  np.load('{out_path.name}')['arr_0']")


if __name__ == "__main__":
    main()
