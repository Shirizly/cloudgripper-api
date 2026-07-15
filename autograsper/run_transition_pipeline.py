"""
run_transition_pipeline.py

Convenience runner that executes the full transition-dataset pipeline:

    Step 1 – extract_transitions.py
        Scans the raw experiment data, identifies planar push (move_xy)
        actions at the right height, and copies the relevant bottom-camera
        images into a structured output directory.

    Step 2 – segment_transitions.py
        Crops each image according to the configured bounding box, runs the
        chickpea segmentation model, and saves binary masks alongside the
        transition images.

Usage:
    python run_transition_pipeline.py \\
        --dataset_root autograsper/recorded_data/push_chickpeas50 \\
        --output_root  autograsper/transitions_data

    # With a custom weights path:
    python run_transition_pipeline.py \\
        --dataset_root autograsper/recorded_data/push_chickpeas50 \\
        --output_root  autograsper/transitions_data \\
        --weights      image_collector/chickpeas_segmentation_best.pt

Final output layout:
    <output_root>/<dataset_name>/
        transitions.json          ← updated with mask paths and crop config
        images/
            {exp}_{mode}_act{id}_{start|end}_bottom.jpeg
        masks/
            start/
                {exp}_{mode}_act{id}_start_bottom.png
            end/
                {exp}_{mode}_act{id}_end_bottom.png
"""

import argparse
import logging
from pathlib import Path

from extract_transitions import process_dataset
from segment_transitions import segment_transitions

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def run_pipeline(
    dataset_root: str,
    output_root: str,
    weights: str | None = None,
) -> None:
    dataset_name = Path(dataset_root).name
    transitions_dir = Path(output_root) / dataset_name

    # ---- Step 1: extract transitions ----
    log.info("=" * 60)
    log.info("STEP 1 – Extracting transitions from %s", dataset_root)
    log.info("=" * 60)
    process_dataset(dataset_root, output_root)

    # ---- Step 2: segment images ----
    log.info("=" * 60)
    log.info("STEP 2 – Segmenting transition images in %s", transitions_dir)
    log.info("=" * 60)
    segment_transitions(transitions_dir, weights_path=weights)

    log.info("=" * 60)
    log.info("Pipeline complete. Dataset at: %s", transitions_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full transition-dataset extraction and segmentation pipeline."
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the directory containing numbered experiment subdirs.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root directory under which the output folder will be created.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to segmentation model weights (.pt). Uses the default repo "
             "path (image_collector/chickpeas_segmentation_best.pt) if omitted.",
    )
    args = parser.parse_args()
    run_pipeline(args.dataset_root, args.output_root, args.weights)
