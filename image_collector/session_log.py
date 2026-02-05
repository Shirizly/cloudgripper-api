import os
from datetime import datetime
import pickle
from typing import Any, Dict, Optional, Tuple

def session_initializer(bounds,base_resolution,N) -> Tuple[list, str]:
    # Define the directory structure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Get the script's directory
    recordings_dir = os.path.join(base_dir, "recordings")
    
    # Ensure recordings directory exists
    os.makedirs(recordings_dir, exist_ok=True)
    
    sub_grids_path = os.path.join(recordings_dir, "sub_grids.txt")
    sub_grids_backup_path = os.path.join(recordings_dir, "sub_grids_for_later.txt")
    
    # Check for incomplete previous session
    dataset_dirs = [d for d in os.listdir(recordings_dir) if d.startswith("dataset_0_sequence_")]
    if dataset_dirs:
        latest_sequence = sorted(dataset_dirs, reverse=True)[0]  # Get the most recent sequence
        sequence_dir = os.path.join(recordings_dir, latest_sequence)
        log_file = os.path.join(sequence_dir, "image_log.txt")
        
        # Check if previous session has incomplete collection
        if os.path.exists(log_file) and os.path.exists(sub_grids_path):
            with open(log_file, "r") as f:
                collected_images = len(f.readlines())  # Count images collected
            
            with open(sub_grids_path, "rb") as f:
                sub_grids = pickle.load(f)
            
            # Check if current sub_grid is incomplete
            if sub_grids and collected_images > 0 and collected_images < len(sub_grids[0]):
                print(f"Resuming previous sub_grid from position {collected_images}.")
                sub_grid = sub_grids[0][collected_images:]  # Resume from where it stopped
                return sub_grid, log_file, sequence_dir
            # Check if current sub_grid is complete but more sub_grids remain
            elif sub_grids and collected_images == len(sub_grids[0]):
                print("Current sub_grid completed. Moving to next sub_grid.")
                # Remove completed sub_grid and continue with next one
                update_sub_grids()
                with open(sub_grids_path, "rb") as f:
                    sub_grids = pickle.load(f)
                if sub_grids:
                    sub_grid = sub_grids[0]
                    return sub_grid, log_file, sequence_dir
    
    # Start a new data collection session
    print("Starting a new data collection session.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sequence_dir = os.path.join(recordings_dir, f"dataset_0_sequence_{timestamp}")
    os.makedirs(sequence_dir)
    
    # File to store image metadata
    log_file = os.path.join(sequence_dir, "image_log.txt")
    
    # Load or generate sub_grids
    sub_grid = None
    if os.path.exists(sub_grids_path):
        with open(sub_grids_path, "rb") as f:
            sub_grids = pickle.load(f)
        
        if sub_grids:
            sub_grid = sub_grids[0]  # Take the first sub-grid
        else:
            os.remove(sub_grids_path)  # Delete file if empty
    
    # Generate sub_grids if none exist
    if sub_grid is None:
        from grid_gen import generate_modular_nd_grid_random_order
        sub_grids = generate_modular_nd_grid_random_order(bounds, base_resolution, N, seed=42)
        
        # Save the sub_grids for backup
        with open(sub_grids_backup_path, "wb") as f:
            pickle.dump(sub_grids, f)
        
        # Save the sub_grids for current session
        with open(sub_grids_path, "wb") as f:
            pickle.dump(sub_grids, f)
        
        sub_grid = sub_grids[0]  # Take the first sub-grid

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