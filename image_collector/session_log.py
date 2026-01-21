import os
from datetime import datetime
import pickle
from typing import Any, Dict, Optional, Tuple

def session_initializer(bounds,base_resolution,N) -> Tuple[list, str]:
    # Define the directory structure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Get the script's directory
    recordings_dir = os.path.join(base_dir, "recordings")
    os.makedirs(recordings_dir, exist_ok=True)    

    # Find the most recent sequence directory
    if os.path.exists(recordings_dir):
        dataset_dirs = [d for d in os.listdir(recordings_dir) if d.startswith("dataset_0_sequence_")]
        if dataset_dirs:
            latest_sequence = sorted(dataset_dirs, reverse=True)[0]  # Get the most recent sequence
            sequence_dir = os.path.join(recordings_dir, latest_sequence)
            previous_log_file = os.path.join(sequence_dir, "image_log.txt")

    # If a previous session exists, check if it was completed
    if previous_log_file and os.path.exists(previous_log_file):
        with open(previous_log_file, "r") as f:
            collected_images = len(f.readlines())  # Count images collected
        
    sub_grids_path = "recordings/sub_grids.txt"
    if os.path.exists(sub_grids_path):
        with open(sub_grids_path, "rb") as f:
            sub_grids = pickle.load(f)
    else:
        print("No sub_grids.txt found. Starting fresh.")
        sub_grids = []

    # Resume last session if needed
    if collected_images > 0 and collected_images < len(sub_grids[0]):
        print(f"Resuming previous sub_grid from position {collected_images}.")
        sub_grid = sub_grids[0][collected_images:]  # Resume from where it stopped
        log_file = previous_log_file
    
    else: # Start a new session

        # Create a unique directory for the sequence
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sequence_dir = os.path.join(recordings_dir, f"dataset_0_sequence_{timestamp}")
        os.makedirs(sequence_dir)

        # File to store image metadata (image file name and robot configuration)
        log_file = os.path.join(sequence_dir, "image_log.txt")

        # generate grid for robot to move, or load from memory if exists

        file_path = "recordings/sub_grids.txt" # File path for sub_grids for operating the recording sessions
        file_path_for_later = "recordings/sub_grids_for_later.txt" # File path for sub_grids storage
        sub_grid = None
        # Check if the file exists
        if os.path.exists(file_path):
            # Load sub_grids from file
            with open(file_path, "rb") as f:
                sub_grids = pickle.load(f)
            
            if sub_grids:
                sub_grid = sub_grids[0]  # Take the first sub-grid
            
            else: # sometimes the file exists but is empty
                os.remove(file_path)  # Delete file if already empty
        if sub_grid is None: #################################################################################################################
            # Generate sub_grids and store them in the file
            from grid_gen import generate_modular_nd_grid_random_order
            sub_grids = generate_modular_nd_grid_random_order(bounds, base_resolution, N, seed=42) ###########################################
            with open(file_path_for_later, "wb") as f: # Save the sub_grids to a file for posterity
                    pickle.dump(sub_grids, f)  # Update the file
            sub_grid = sub_grids[0]  # Take the first sub-grid
            
            if sub_grids:
                with open(file_path, "wb") as f:
                    pickle.dump(sub_grids, f)  # Update the file

    return sub_grid, log_file, sequence_dir

def update_sub_grids():
    # Update the sub_grids file to remove the first sub-grid
    file_path = "recordings/sub_grids.txt"
    with open(file_path, "rb") as f:
        sub_grids = pickle.load(f)
    sub_grids.pop(0)
    if sub_grids:
        with open(file_path, "wb") as f:
            pickle.dump(sub_grids, f)
    else:
        os.remove(file_path)  # Delete file if already empty