import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

def analyze_tool_grip(image: np.ndarray, tool_color_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> float:
    """
    Analyze if a tool is held well by checking color presence in the expected tool region.
    Args:
        image: Input image (BGR format)
        tool_color_range: Tuple of (lower_bgr, upper_bgr) color bounds
    
    Returns:
        Float between 0 and 1 indicating grip quality (proportion of tool region with correct color)
    """
    lower_color, upper_color = tool_color_range
    lower = np.array(lower_color, dtype=np.uint8)
    upper = np.array(upper_color, dtype=np.uint8)
    
    mask = cv2.inRange(image, lower, upper)
    tool_region_pixels = cv2.countNonZero(mask)
    total_pixels = image.shape[0] * image.shape[1]
    # cv2.imshow("Tool Grip Analysis", mask)
    # cv2.waitKey(1)
    cv2.imwrite(str(Path("tool_grip_analysis.png")), mask)
    cv2.imwrite(str(Path("tool_grip_original.png")), image)
    
    grip_quality = tool_region_pixels / total_pixels
    return grip_quality