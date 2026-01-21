import cv2
import numpy as np
from typing import Tuple, List, Dict

def create_occupancy_mask(image: np.ndarray, reference_empty: np.ndarray, threshold: int = 30) -> np.ndarray:
    """
    Create a binary occupancy mask by comparing image to empty reference.
    
    Args:
        image: Current image from camera
        reference_empty: Reference image of empty glass plate
        threshold: Difference threshold to consider as occupied
        
    Returns:
        Binary occupancy mask (1 = occupied, 0 = empty)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    if len(reference_empty.shape) == 3:
        ref_gray = cv2.cvtColor(reference_empty, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = reference_empty
    
    # Compute absolute difference
    diff = cv2.absdiff(gray, ref_gray)
    
    # Create binary mask
    mask = (diff > threshold).astype(np.uint8) * 255
    
    return mask


def downscale_mask(mask: np.ndarray, scale_factor: int = 4) -> np.ndarray:
    """
    Downscale occupancy mask to lower resolution.
    
    Args:
        mask: Input binary mask
        scale_factor: Downscaling factor
        
    Returns:
        Downscaled binary mask
    """
    h, w = mask.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    
    downscaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    return downscaled


def find_clumps(mask: np.ndarray, min_size: int = 5, max_size: int = 500) -> List[Dict]:
    """
    Identify connected components (clumps) in occupancy mask.
    
    Args:
        mask: Binary occupancy mask
        min_size: Minimum clump size in pixels
        max_size: Maximum clump size in pixels
        
    Returns:
        List of clump dictionaries with properties
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    clumps = []
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        
        if min_size <= area <= max_size:
            clump = {
                'id': i,
                'area': area,
                'centroid': centroids[i],
                'bbox': {
                    'x': stats[i, cv2.CC_STAT_LEFT],
                    'y': stats[i, cv2.CC_STAT_TOP],
                    'width': stats[i, cv2.CC_STAT_WIDTH],
                    'height': stats[i, cv2.CC_STAT_HEIGHT]
                }
            }
            clumps.append(clump)
    
    return clumps


def generate_clump_hierarchy(mask: np.ndarray, scales: List[int] = None) -> List[List[Dict]]:
    """
    Generate clump detections at multiple resolution scales.
    
    Args:
        mask: Binary occupancy mask
        scales: List of scale factors for analysis
        
    Returns:
        List of clump lists, one per scale
    """
    if scales is None:
        scales = [1, 2, 4, 8]
    
    hierarchy = []
    for scale in scales:
        scaled_mask = downscale_mask(mask, scale)
        clumps = find_clumps(scaled_mask)
        hierarchy.append(clumps)
    
    return hierarchy


def process_image(image: np.ndarray, reference_empty: np.ndarray, 
                  scale_factor: int = 4) -> Tuple[np.ndarray, List[Dict]]:
    """
    Complete pipeline: mask, downscale, and detect clumps.
    
    Args:
        image: Current image
        reference_empty: Empty reference image
        scale_factor: Downscaling factor
        
    Returns:
        Tuple of (downscaled_mask, clumps_list)
    """
    mask = create_occupancy_mask(image, reference_empty)
    downscaled = downscale_mask(mask, scale_factor)
    clumps = find_clumps(downscaled)
    
    return downscaled, clumps