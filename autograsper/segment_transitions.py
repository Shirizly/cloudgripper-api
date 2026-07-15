"""
segment_transitions.py

Applies the chickpea segmentation model to the transition images produced by
extract_transitions.py, and saves the resulting binary masks into:

    <transitions_dir>/masks/start/   – masks for start-frame images
    <transitions_dir>/masks/end/     – masks for end-frame images

Mask filenames mirror the source image stems (e.g., the mask for
``images/206_restore_act3_start_bottom.jpeg`` is saved as
``masks/start/206_restore_act3_start_bottom.png``).

transitions.json is updated in-place to add ``start_mask`` and ``end_mask``
fields (relative to the transitions directory) for each transition that has
a corresponding image.

Usage:
    python segment_transitions.py --transitions_dir autograsper/transitions_data/push_chickpeas50
    python segment_transitions.py --transitions_dir ...  --weights path/to/best.pt
"""

import cv2
import json
import argparse
import logging
import sys
from pathlib import Path

import torch

# ============================================================
# CONFIGURATION
# ============================================================

# Pixel crop window applied to every image before segmentation.
# Set to None to use the full image dimension.
# Coordinates are (column/row) in the original image space.
CROP_X1: int | None = 20   # left edge
CROP_Y1: int | None = 95   # top edge
CROP_X2: int | None = 380   # right edge  (exclusive)
CROP_Y2: int | None = 455   # bottom edge (exclusive)

# Default weights path relative to this file's location
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = _REPO_ROOT / "image_collector" / "chickpeas_segmentation_best.pt"

# ============================================================

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Lazy import so the script is importable even without ultralytics installed
# (the runner only imports it if segmentation is actually run).
def _load_segmenter(weights_path: Path):
    sys.path.insert(0, str(_REPO_ROOT / "image_collector"))
    from chickpea_segmenter import ChickpeaSegmenter  # noqa: PLC0415
    return ChickpeaSegmenter(weights_path, imgsz=360)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _crop(image: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
    """Apply the configured crop window, or return image unchanged if all None."""
    y1 = CROP_Y1 if CROP_Y1 is not None else 0
    x1 = CROP_X1 if CROP_X1 is not None else 0
    y2 = CROP_Y2 if CROP_Y2 is not None else image.shape[0]
    x2 = CROP_X2 if CROP_X2 is not None else image.shape[1]
    return image[y1:y2, x1:x2]


def _mask_path(transitions_dir: Path, endpoint: str, image_stem: str) -> Path:
    return transitions_dir / "masks" / endpoint / f"{image_stem}.png"


# ------------------------------------------------------------------
# Core
# ------------------------------------------------------------------

def segment_transitions(
    transitions_dir: str | Path,
    weights_path: str | Path | None = None,
) -> None:
    transitions_dir = Path(transitions_dir)
    transitions_json = transitions_dir / "transitions.json"

    print(torch.cuda.is_available())

    if not transitions_json.exists():
        raise FileNotFoundError(f"transitions.json not found in {transitions_dir}")

    weights = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
    if not weights.exists():
        raise FileNotFoundError(f"Segmentation weights not found: {weights}")

    # Create output dirs
    for endpoint in ("start", "end"):
        (transitions_dir / "masks" / endpoint).mkdir(parents=True, exist_ok=True)

    log.info("Loading segmentation model from %s", weights.name)
    segmenter = _load_segmenter(weights)

    with open(transitions_json) as f:
        data = json.load(f)

    # Record crop config in the manifest
    data["crop"] = {
        "x1": CROP_X1, "y1": CROP_Y1,
        "x2": CROP_X2, "y2": CROP_Y2,
    }

    total = len(data["transitions"])
    processed = 0
    skipped = 0

    for i, transition in enumerate(data["transitions"]):
        for endpoint in ("start", "end"):
            img_rel = transition.get(f"{endpoint}_bottom_image")
            if img_rel is None:
                transition[f"{endpoint}_mask"] = None
                continue

            img_path = transitions_dir / img_rel
            if not img_path.exists():
                log.warning("[%d/%d] Image not found, skipping: %s", i + 1, total, img_path)
                transition[f"{endpoint}_mask"] = None
                skipped += 1
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                log.warning("[%d/%d] Could not read image: %s", i + 1, total, img_path)
                transition[f"{endpoint}_mask"] = None
                skipped += 1
                continue

            cropped = _crop(image)
            mask = segmenter.predict(cropped, return_format="combined")

            out_path = _mask_path(transitions_dir, endpoint, img_path.stem)
            cv2.imwrite(str(out_path), mask)

            transition[f"{endpoint}_mask"] = f"masks/{endpoint}/{out_path.name}"
            processed += 1

        if (i + 1) % 20 == 0 or (i + 1) == total:
            log.info("  Progress: %d / %d transitions", i + 1, total)

    # Write updated transitions.json
    with open(transitions_json, "w") as f:
        json.dump(data, f, indent=2)

    log.info(
        "Done. %d masks saved, %d skipped → %s",
        processed, skipped, transitions_dir / "masks",
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment transition images and save binary masks."
    )
    parser.add_argument(
        "--transitions_dir",
        required=True,
        help="Path to a transitions output directory (must contain transitions.json "
             "and an images/ subdirectory, as produced by extract_transitions.py).",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help=f"Path to segmentation model weights. "
             f"Defaults to {DEFAULT_WEIGHTS}",
    )
    args = parser.parse_args()
    segment_transitions(args.transitions_dir, args.weights)
