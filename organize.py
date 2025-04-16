import os
import shutil

def organize_images(recordings_dir):
    # Get all session directories inside recordings
    for session_folder in os.listdir(recordings_dir):
        session_path = os.path.join(recordings_dir, session_folder)
        
        if not os.path.isdir(session_path):
            continue  # Skip if it's not a directory
        
        # Create new folders inside the session folder
        top_dir = os.path.join(session_path, "top_images")
        base_dir = os.path.join(session_path, "base_images")
        os.makedirs(top_dir, exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)
        
        # Move images to respective folders
        for file in os.listdir(session_path):
            file_path = os.path.join(session_path, file)
            
            if file.startswith("image_top_") and file.endswith(".png"):
                shutil.move(file_path, os.path.join(top_dir, file))
            elif file.startswith("image_base_") and file.endswith(".png"):
                shutil.move(file_path, os.path.join(base_dir, file))

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    recordings_dir = os.path.join(project_root, "recordings")
    organize_images(recordings_dir)
    print("Images organized successfully.")