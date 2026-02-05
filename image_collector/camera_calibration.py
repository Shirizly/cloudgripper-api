import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from library.calibration import calibrate_fisheye



def calibrate_camera_from_image(image_path):
    """
    Load an image and run fisheye calibration.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        Calibration parameters from calibrate_fisheye()
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    calibration_params = calibrate_fisheye([image], pattern_size=(7, 7), square_size=15, increase_contrast=True)
    return calibration_params


if __name__ == "__main__":
    image_path = "calibrate.jpg"
    params = calibrate_camera_from_image(image_path)
    print("Calibration parameters:", params)
    save_path = "calibration_params.npz"
    np.savez(save_path, *params)
    print(f"Calibration parameters saved to {save_path}")
    