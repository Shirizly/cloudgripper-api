import os
import json
import numpy as np
import cv2
import h5py
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

z_threshold = 0.4

xy_epsilon = 1e-3
z_epsilon = 1e-3
angle_epsilon = 1e-3

max_direction_deviation_deg = 20
min_frames_per_action = 3
min_segment_length = 3
desired_frames_per_video = 16
max_idle_gap = 10  # max idle frames between segments to still consider them consecutive

fps = 15

camera_poses = {
    "front": np.array([0, 0, 1, 0, 0, 0]),
    "bottom": np.array([0, 0, -1, 0, 0, 0]),
}

# ============================================================
# LOADING
# ============================================================

def load_states(path):
    with open(path, "r") as f:
        data = json.load(f)
    states = []

    for s in data:
        states.append([
            s["x_norm"],
            s["y_norm"],
            s["z_norm"],
            0.0,
            0.0,
            s["rotation"]
        ])

    return np.array(states), data


# ============================================================
# ACTION SEGMENTATION (multi-frame)
# ============================================================

def segment_actions(states):
    T = len(states)
    deltas = np.diff(states, axis=0)

    segments = []
    cos_thresh = np.cos(np.deg2rad(max_direction_deviation_deg))

    i = 1
    while i < T:

        dx, dy, dz, _, _, dangle = deltas[i - 1]
        xy_mag = np.linalg.norm([dx, dy])
        z_mag = abs(dz)
        rot_mag = abs(dangle)

        # ---------------------------
        # XY MOTION
        # ---------------------------
        if xy_mag > xy_epsilon:

            start = i - 1
            direction_sum = np.array([dx, dy])
            j = i

            while j < T:
                dx2, dy2, dz2, _, _, dangle2 = deltas[j - 1]
                v = np.array([dx2, dy2])
                mag = np.linalg.norm(v)

                if mag <= xy_epsilon:
                    break

                dir_norm = direction_sum / (np.linalg.norm(direction_sum) + 1e-8)
                alignment = np.dot(v / (mag + 1e-8), dir_norm)

                if alignment < cos_thresh:
                    break

                direction_sum += v
                j += 1

            if j - start >= min_frames_per_action:
                delta_total = states[j - 1] - states[start]

                segments.append({
                    "type": "xy",
                    "start": start,
                    "end": j,
                    "delta": delta_total.tolist()
                })

                i = j
                continue

        # ---------------------------
        # Z MOTION
        # ---------------------------
        if z_mag > z_epsilon:
            start = i - 1
            j = i

            while j < T and abs(deltas[j - 1][2]) > z_epsilon:
                j += 1

            if j - start >= min_frames_per_action:
                delta_total = states[j - 1] - states[start]

                segments.append({
                    "type": "z",
                    "start": start,
                    "end": j,
                    "delta": delta_total.tolist()
                })

                i = j
                continue

        # ---------------------------
        # ROTATION
        # ---------------------------
        if rot_mag > angle_epsilon:
            start = i - 1
            j = i

            while j < T and abs(deltas[j - 1][5]) > angle_epsilon:
                j += 1

            if j - start >= min_frames_per_action:
                delta_total = states[j - 1] - states[start]

                segments.append({
                    "type": "rotation",
                    "start": start,
                    "end": j,
                    "delta": delta_total.tolist()
                })

                i = j
                continue

        i += 1

    return segments


# ============================================================
# PER-FRAME ACTIONS (for v-JEPA2-AC)
# ============================================================

def compute_per_frame_actions(states):
    actions = np.zeros_like(states)
    actions[:-1] = states[1:] - states[:-1]
    return actions


# ============================================================
# TRAJECTORY EXTRACTION
# ============================================================

def merge_consecutive_segments(segments):
    """Merge segments that are adjacent or overlapping into continuous frame ranges."""
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s["start"])

    merged = []
    current_start = sorted_segs[0]["start"]
    current_end = sorted_segs[0]["end"]

    for seg in sorted_segs[1:]:
        if seg["start"] <= current_end + max_idle_gap:
            current_end = max(current_end, seg["end"])
        else:
            merged.append((current_start, current_end))
            current_start = seg["start"]
            current_end = seg["end"]

    merged.append((current_start, current_end))
    return merged


def split_into_chunks(merged_ranges, desired_frames=desired_frames_per_video):
    """Split merged frame ranges into chunks of desired_frames length."""
    chunks = []
    for start, end in merged_ranges:
        length = end - start
        if length <= desired_frames:
            chunks.append((start, end))
        else:
            for chunk_start in range(start, end, desired_frames):
                chunk_end = min(chunk_start + desired_frames, end)
                if chunk_end - chunk_start >= min_segment_length:
                    chunks.append((chunk_start, chunk_end))
    return chunks


# ============================================================
# VIDEO WRITING
# ============================================================

def save_video_from_images(image_dir, start_idx, end_idx, output_path, frame_rate=None):
    if frame_rate is None:
        frame_rate = fps
    
    image_files = sorted(os.listdir(image_dir))

    first_img = cv2.imread(str(Path(image_dir) / image_files[start_idx]))
    h, w, _ = first_img.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, frame_rate, (w, h))

    for frame_idx in range(start_idx, end_idx):
        if frame_idx < len(image_files):
            frame = cv2.imread(str(Path(image_dir) / image_files[frame_idx]))
            if frame is not None:
                writer.write(frame)

    writer.release()


# ============================================================
# SESSION PROCESSING
# ============================================================

def process_session(session_path, output_root):

    session_path = Path(session_path)
    session_id = session_path.name

    for mode in ["task", "restore"]:

        mode_path = session_path / mode
        states_path = mode_path / "states.json"
        print(f"states_path: {states_path}")

        if not states_path.exists():
            print(f"States file not found for session {session_id} mode {mode}, skipping.")
            continue

        states, keys = load_states(states_path)
        print(f"Loaded states for session {session_id} mode {mode}, shape: {states.shape}")
        
        # Calculate FPS from actual timestamps
        calculated_fps = fps
        if len(keys) > 1:
            time_diffs = [keys[i]["time"] - keys[i-1]["time"] for i in range(1, len(keys))]
            avg_time_diff = np.mean(time_diffs)
            calculated_fps = 1.0 / avg_time_diff if avg_time_diff > 0 else fps
            print(f"Calculated FPS from timestamps: {calculated_fps:.2f}")
        
        # ---- segmentation for summary
        segments = segment_actions(states)
        print(f"Identified {len(segments)} segments for session {session_id} mode {mode}")
        summary_path = session_path / f"{mode}_action_summary.json"
        with open(summary_path, "w") as f:
            json.dump({"actions": segments}, f, indent=2)

        # ---- per-frame actions for v-JEPA
        per_frame_actions = compute_per_frame_actions(states)
        print(f"Computed per-frame actions for session {session_id} mode {mode}, shape: {per_frame_actions.shape}")
        # ---- group consecutive actions and split into chunks
        merged_ranges = merge_consecutive_segments(segments)
        print(f"Merged {len(segments)} segments into {len(merged_ranges)} consecutive action groups for session {session_id} mode {mode}")
        traj_ranges = split_into_chunks(merged_ranges)
        print(f"Split into {len(traj_ranges)} trajectory chunks (desired_frames={desired_frames_per_video}) for session {session_id} mode {mode}")

        for idx, (start, end) in enumerate(traj_ranges):
            print(f"Extracting trajectory {idx} for session {session_id} mode {mode}, frames {start} to {end}")
            traj_dir = Path(output_root) / f"{session_id}_{mode}_{idx}"
            recordings_dir = traj_dir / "recordings"
            recordings_dir.mkdir(parents=True, exist_ok=True)

            seg_states = states[start:end]
            seg_actions = per_frame_actions[start:end]

            T = len(seg_states)

            # save videos
            save_video_from_images(
                mode_path / "Images",
                start,
                end,
                recordings_dir / "front.mp4",
                frame_rate=calculated_fps
            )

            save_video_from_images(
                mode_path / "Bottom_Images",
                start,
                end,
                recordings_dir / "bottom.mp4",
                frame_rate=calculated_fps
            )

            # save h5
            with h5py.File(traj_dir / "trajectory.h5", "w") as f:
                f.create_dataset(
                    "videos",
                    data=np.array([
                        b"recordings/front.mp4",
                        b"recordings/bottom.mp4"
                    ])
                )

            # save v-JEPA compatible json
            traj_json = {
                "states": seg_states.tolist(),
                "actions": seg_actions.tolist(),
                "camera_front": np.tile(camera_poses["front"], (T, 1)).tolist(),
                "camera_bottom": np.tile(camera_poses["bottom"], (T, 1)).tolist()
            }

            with open(traj_dir / "states.json", "w") as f:
                json.dump(traj_json, f, indent=2)


# ============================================================
# DATASET PROCESSING
# ============================================================

def process_dataset(dataset_root, output_root):
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    output_root.mkdir(exist_ok=True)
    session_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    session_dirs = sorted(session_dirs, key=lambda x: int(x.name))
    for session in session_dirs:
        print(f"Processing session: {session.name}")
        process_session(session, output_root)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_root", required=True)

    args = parser.parse_args()

    process_dataset(args.dataset_root, args.output_root)
