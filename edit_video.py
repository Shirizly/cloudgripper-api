import cv2
import os

def edit_video(video_path, time_start, time_end, output_path):
    """
    Load an MP4 video, remove frames between time_start and time_end, and save the edited video.
    
    Args:
        video_path (str): Path to the input MP4 video
        time_start (float): Start time in seconds
        time_end (float): End time in seconds
        output_path (str): Path to save the edited video
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_frame = int(time_start * fps)
    end_frame = int(time_end * fps)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Write frame if it's outside the removal range
        if frame_count < start_frame or frame_count > end_frame:
            out.write(frame)
        
        frame_count += 1
    
    cap.release()
    out.release()
    print(f"Edited video saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    edit_video("robot_cam_output.mp4", 0.0, 60.0, "edited_video.mp4")