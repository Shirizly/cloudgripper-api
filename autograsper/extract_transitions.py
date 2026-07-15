"""
extract_transitions.py

Scans a collected dataset directory and extracts planar push (move_xy) transitions
for training transition models. For each qualifying action it:
  - Records transition metadata (start/end robot state, action details) in a
    flat transitions.json file.
  - Copies the bottom-camera images at the start frame and end frame of each
    action into an images/ subdirectory.

Output layout:
    <output_root>/<dataset_name>/
        transitions.json
        images/
            {exp_id}_{mode}_act{action_id}_start_bottom.jpeg
            {exp_id}_{mode}_act{action_id}_end_bottom.jpeg

Usage:
    python extract_transitions.py --dataset_root path/to/experiments \
                                  --output_root  path/to/transitions_data
"""

import os
import json
import shutil
import argparse
import logging
from collections import defaultdict
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Only include move_xy actions whose start z_norm is at or below this value.
# At z_norm=0.8 the arm is at home height; push actions happen at ~0.339.
Z_THRESHOLD = 0.4

# Which action phases to include.  None or an empty set means "all phases".
# Example: {"task"} to include only task-phase pushes.
INCLUDED_PHASES = None  # None → include all phases (task, reset, …)

# ============================================================

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Action loading helpers
# ------------------------------------------------------------------

def _state_to_dict(state_record: dict) -> dict:
    """Extract robot-state fields from a states.json frame record."""
    return {
        "x_norm": state_record.get("x_norm"),
        "y_norm": state_record.get("y_norm"),
        "z_norm": state_record.get("z_norm"),
        "rotation": state_record.get("rotation"),
        "claw_norm": state_record.get("claw_norm"),
    }


def load_actions_from_actions_json(path: Path) -> list[dict]:
    """Load action list directly from a pre-built actions.json file."""
    with open(path) as f:
        data = json.load(f)
    return data.get("actions", [])


def load_actions_from_states_json(path: Path) -> list[dict]:
    """
    Reconstruct actions by grouping per-frame action annotations in states.json.
    Frames that share an action_id belong to the same action; the first frame
    gives start state and the last gives end state.
    """
    with open(path) as f:
        states = json.load(f)

    groups: dict[int, list[dict]] = defaultdict(list)
    for s in states:
        if "action" in s:
            groups[s["action"]["action_id"]].append(s)

    actions = []
    for action_id in sorted(groups):
        frames = sorted(groups[action_id], key=lambda s: s["frame_index"])
        first, last = frames[0], frames[-1]
        ann = first["action"]
        actions.append({
            "action_id":        action_id,
            "action_type":      ann.get("action_type"),
            "phase":            ann.get("phase"),
            "start_frame":      first["frame_index"],
            "end_frame":        last["frame_index"],
            "is_planar_2d":     ann.get("is_planar_2d", False),
            "start_robot_state": _state_to_dict(first),
            "end_robot_state":  _state_to_dict(last),
            "action_details":   ann.get("action_details", {}),
            "description":      ann.get("description"),
            "extra_metadata":   ann.get("extra_metadata", {}),
        })

    return actions


def load_actions(mode_path: Path) -> list[dict]:
    """Load actions from actions.json if present, otherwise from states.json."""
    actions_json = mode_path / "actions.json"
    states_json  = mode_path / "states.json"

    if actions_json.exists():
        log.debug("  Using actions.json: %s", actions_json)
        return load_actions_from_actions_json(actions_json)
    elif states_json.exists():
        log.debug("  Reconstructing actions from states.json: %s", states_json)
        return load_actions_from_states_json(states_json)
    else:
        log.warning("  No actions.json or states.json found in %s", mode_path)
        return []


# ------------------------------------------------------------------
# Image helpers
# ------------------------------------------------------------------

BOTTOM_IMAGE_TEMPLATE = "image_bottom_{frame_index}.jpeg"


def bottom_image_path(mode_path: Path, frame_index: int) -> Path | None:
    """Return the path to the bottom-camera image for a given frame, or None."""
    p = mode_path / "Bottom_Images" / BOTTOM_IMAGE_TEMPLATE.format(frame_index=frame_index)
    return p if p.exists() else None


def copy_image(src: Path | None, dst: Path) -> bool:
    """Copy src → dst. Returns True on success, False if src is None or missing."""
    if src is None:
        return False
    shutil.copy2(src, dst)
    return True


# ------------------------------------------------------------------
# Core processing
# ------------------------------------------------------------------

def process_mode(
    experiment_id: str,
    mode: str,
    mode_path: Path,
    images_out: Path,
    included_phases: set | None,
) -> list[dict]:
    """
    Process one mode subdir (task or restore) of one experiment.
    Returns a list of transition dicts.
    """
    actions = load_actions(mode_path)
    if not actions:
        return []

    transitions = []

    for action in actions:
        # --- filter by action type ---
        if action.get("action_type") != "move_xy":
            continue

        # --- filter by phase ---
        if included_phases:
            if action.get("phase") not in included_phases:
                continue

        # --- filter by height ---
        start_state = action.get("start_robot_state") or {}
        z_start = start_state.get("z_norm")
        if z_start is None or z_start > Z_THRESHOLD:
            continue

        start_frame = action["start_frame"]
        end_frame   = action["end_frame"]
        action_id   = action["action_id"]

        # Build unique image names
        stem  = f"{experiment_id}_{mode}_act{action_id}"
        start_dst = images_out / f"{stem}_start_bottom.jpeg"
        end_dst   = images_out / f"{stem}_end_bottom.jpeg"

        start_src = bottom_image_path(mode_path, start_frame)
        end_src   = bottom_image_path(mode_path, end_frame)

        if not copy_image(start_src, start_dst):
            log.warning(
                "  Start image missing for %s (frame %d): %s",
                stem, start_frame, mode_path / "Bottom_Images"
            )
        if not copy_image(end_src, end_dst):
            log.warning(
                "  End image missing for %s (frame %d): %s",
                stem, end_frame, mode_path / "Bottom_Images"
            )

        # Action delta (end position – start position)
        end_state = action.get("end_robot_state") or {}
        dx = (end_state.get("x_norm") or 0.0) - (start_state.get("x_norm") or 0.0)
        dy = (end_state.get("y_norm") or 0.0) - (start_state.get("y_norm") or 0.0)

        transitions.append({
            "transition_id":     f"{experiment_id}_{mode}_act{action_id}",
            "experiment_id":     experiment_id,
            "mode":              mode,
            "action_id":         action_id,
            "action_type":       action["action_type"],
            "phase":             action.get("phase"),
            "start_frame":       start_frame,
            "end_frame":         end_frame,
            "start_state":       start_state,
            "end_state":         end_state,
            "action_target":     action.get("action_details", {}),
            "action_delta":      {"dx": dx, "dy": dy},
            "start_bottom_image": f"images/{start_dst.name}" if start_src else None,
            "end_bottom_image":   f"images/{end_dst.name}"   if end_src   else None,
        })

    return transitions


def process_dataset(dataset_root: str, output_root: str) -> None:
    dataset_root = Path(dataset_root)
    output_root  = Path(output_root)

    dataset_name = dataset_root.name
    out_dir      = output_root / dataset_name
    images_out   = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    included_phases: set | None = (
        set(INCLUDED_PHASES) if INCLUDED_PHASES else None
    )

    # Collect numbered experiment subdirs
    experiment_dirs = sorted(
        [d for d in dataset_root.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not experiment_dirs:
        log.warning("No numbered experiment directories found in %s", dataset_root)
        return

    all_transitions: list[dict] = []

    for exp_dir in experiment_dirs:
        exp_id = exp_dir.name
        log.info("Processing experiment %s", exp_id)

        for mode in ("task", "restore"):
            mode_path = exp_dir / mode
            if not mode_path.is_dir():
                continue
            log.info("  Mode: %s", mode)
            transitions = process_mode(
                exp_id, mode, mode_path, images_out, included_phases
            )
            log.info("  → %d qualifying transition(s)", len(transitions))
            all_transitions.extend(transitions)

    # Write summary JSON
    output_json = out_dir / "transitions.json"
    with open(output_json, "w") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "z_threshold": Z_THRESHOLD,
                "included_phases": list(included_phases) if included_phases else "all",
                "total_transitions": len(all_transitions),
                "transitions": all_transitions,
            },
            f,
            indent=2,
        )

    log.info(
        "Done. %d transitions written to %s", len(all_transitions), output_json
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract planar push transitions from a collected robot dataset."
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the directory containing numbered experiment subdirs.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root directory under which the output folder will be created "
             "(a subdirectory named after the dataset will be created inside).",
    )
    args = parser.parse_args()
    process_dataset(args.dataset_root, args.output_root)
