import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class ToolIdentifier:
    def __init__(self, object_outline_path: str, gripper_position: Tuple[int, int], proximity_threshold: int = 50):
        """
        Initialize the Tool Identifier.
        
        Args:
            object_outline_path: Path to the object outline file (image template)
            gripper_position: (x, y) coordinates of the robot gripper in the image
            proximity_threshold: Maximum distance in pixels for a tool to be considered "held"
        """
        self.object_outline = cv2.imread(object_outline_path, cv2.IMREAD_GRAYSCALE)
        if self.object_outline is None:
            raise FileNotFoundError(f"Could not load object outline from {object_outline_path}")
        
        self.gripper_position = gripper_position
        self.proximity_threshold = proximity_threshold
    
    def identify_tool(self, image_path: str) -> Tuple[bool, Optional[dict]]:
        """
        Identify if a tool is held by the gripper.
        
        Args:
            image_path: Path to the robot image
            
        Returns:
            Tuple of (is_held, tool_info) where tool_info contains position and confidence
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")
        
        # Template matching
        result = cv2.matchTemplate(image, self.object_outline, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Get tool center position
        tool_height, tool_width = self.object_outline.shape
        tool_center_x = max_loc[0] + tool_width // 2
        tool_center_y = max_loc[1] + tool_height // 2
        tool_center = (tool_center_x, tool_center_y)
        
        # Calculate distance to gripper
        distance = np.sqrt(
            (tool_center_x - self.gripper_position[0])**2 + 
            (tool_center_y - self.gripper_position[1])**2
        )
        
        is_held = distance <= self.proximity_threshold
        
        tool_info = {
            "position": tool_center,
            "distance_to_gripper": distance,
            "confidence": max_val
        }
        
        return is_held, tool_info