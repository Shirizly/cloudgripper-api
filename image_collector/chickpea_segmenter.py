"""
Chickpea Segmentation Model Wrapper for Robotics.

Provides a clean interface to the fine-tuned YOLOv11 segmentation model
for integration into robotic pipelines.

Example usage:
    from chickpea_segmenter import ChickpeaSegmenter
    
    # Initialize
    segmenter = ChickpeaSegmenter("runs/segment/chickpea/weights/best.pt")
    
    # Predict on single image
    image = cv2.imread("photo.jpg")
    masks_dict = segmenter.predict(image)
    # Returns: {
    #     'combined': numpy array (H, W) - all masks merged
    #     'individual': list of numpy arrays - one per chickpea
    #     'boxes': list of [x1, y1, x2, y2] bounding boxes
    #     'confidences': list of confidence scores
    #     'num_detections': int
    # }
    
    # Get individual masks
    individual_masks = segmenter.predict(image, return_format='individual')
    
    # Get combined mask
    combined_mask = segmenter.predict(image, return_format='combined')
    
    # Batch processing
    images = [cv2.imread(f) for f in image_paths]
    results = segmenter.predict_batch(images)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple

from ultralytics import YOLO


class ChickpeaSegmenter:
    """
    Fine-tuned YOLOv11 segmentation model for chickpea detection.
    
    Wraps the ultralytics YOLO model with a robotics-friendly interface.
    
    Attributes:
        model (YOLO): Loaded YOLO model
        device (str): Device used for inference ('cuda' or 'cpu')
        conf_threshold (float): Default confidence threshold
        iou_threshold (float): Default IoU threshold for NMS
    """
    
    def __init__(
        self,
        weights_path: Union[str, Path],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: Optional[str] = None,
        imgsz: int = 640,
    ):
        """
        Initialize the chickpea segmenter.
        
        Args:
            weights_path: Path to YOLO weights file (e.g., 'runs/segment/chickpea/weights/best.pt')
            conf_threshold: Confidence threshold for detections (default: 0.25)
            iou_threshold: IoU threshold for NMS (default: 0.5)
            device: Device to use ('cuda', 'cpu', or None for auto). Default: auto-detect
            imgsz: Input image size for inference (default: 640)
        
        Raises:
            FileNotFoundError: If weights file does not exist
            RuntimeError: If weights file format is not recognized
        """
        self.weights_path = Path(weights_path)
        
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found at {self.weights_path}")
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            print(f"Using device: {self.device}")
        
        # Load model (YOLO auto-detects architecture from weights)
        self.model = YOLO(str(self.weights_path))
        self.model.to(self.device)
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        
        print(f"✓ Loaded model from {self.weights_path.name}")
        print(f"  Device: {self.device} | Input size: {imgsz}x{imgsz}")
    
    def predict(
        self,
        image: np.ndarray,
        return_format: str = 'dict',
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> Union[Dict, np.ndarray, List[np.ndarray]]:
        """
        Predict segmentation masks for chickpeas in an image.
        
        Args:
            image: Input image (BGR numpy array from cv2.imread or similar)
            return_format: Format of returned masks:
                - 'dict': Full dictionary with all information (default)
                - 'combined': Single binary mask with all chickpeas (H, W)
                - 'individual': List of individual binary masks
                - 'confidences': List of confidence scores paired with masks
            conf: Confidence threshold (uses default if None)
            iou: IoU threshold (uses default if None)
        
        Returns:
            depends on return_format:
            
            'dict': {
                'combined': numpy.ndarray (H, W) - binary mask, 255=chickpea, 0=background
                'individual': List[numpy.ndarray] - individual binary masks
                'boxes': List[[x1, y1, x2, y2]] - detection bounding boxes
                'confidences': List[float] - detection confidence scores
                'num_detections': int - number of chickpeas detected
            }
            
            'combined': numpy.ndarray (H, W) - binary mask
            
            'individual': List[numpy.ndarray] - list of binary masks
            
            'confidences': List[Tuple[numpy.ndarray, float]] - (mask, confidence) pairs
        
        Raises:
            ValueError: If image is invalid or return_format is unrecognized
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array (e.g., from cv2.imread)")
        
        if image.size == 0:
            raise ValueError("Image is empty")
        
        conf_val = conf if conf is not None else self.conf_threshold
        iou_val = iou if iou is not None else self.iou_threshold
        
        # Run inference
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=conf_val,
            iou=iou_val,
            verbose=False,
        )
        result = results[0]
        
        h, w = image.shape[:2]
        
        # Extract data
        masks_individual = []
        boxes = []
        confidences = []
        
        if result.masks is not None and len(result.masks.data) > 0:
            for mask_tensor, box_tensor, conf_tensor in zip(
                result.masks.data,
                result.boxes.xyxy,
                result.boxes.conf,
            ):
                # Resize mask to original image size
                mask_np = mask_tensor.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_LINEAR)
                mask_binary = ((mask_resized > 0.5) * 255).astype(np.uint8)
                
                masks_individual.append(mask_binary)
                boxes.append(box_tensor.cpu().numpy().astype(np.float32))
                confidences.append(conf_tensor.cpu().item())
        
        # Build return value based on format
        if return_format == 'dict':
            # Create combined mask
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for mask in masks_individual:
                combined_mask = np.maximum(combined_mask, mask)
            
            return {
                'combined': combined_mask,
                'individual': masks_individual,
                'boxes': boxes,
                'confidences': confidences,
                'num_detections': len(masks_individual),
            }
        
        elif return_format == 'combined':
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for mask in masks_individual:
                combined_mask = np.maximum(combined_mask, mask)
            return combined_mask
        
        elif return_format == 'individual':
            return masks_individual
        
        elif return_format == 'confidences':
            return [(mask, conf) for mask, conf in zip(masks_individual, confidences)]
        
        else:
            raise ValueError(
                f"Unknown return_format: {return_format}. "
                "Choose from: 'dict', 'combined', 'individual', 'confidences'"
            )
    
    def predict_batch(
        self,
        images: List[np.ndarray],
        return_format: str = 'dict',
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[Union[Dict, np.ndarray, List[np.ndarray]]]:
        """
        Predict segmentation masks for multiple images.
        
        Args:
            images: List of input images (BGR numpy arrays)
            return_format: Same as predict()
            conf: Confidence threshold (uses default if None)
            iou: IoU threshold (uses default if None)
        
        Returns:
            List of results in the same format as predict() for each image
        """
        results = []
        for image in images:
            result = self.predict(image, return_format=return_format, conf=conf, iou=iou)
            results.append(result)
        return results
    
    def set_thresholds(self, conf: float, iou: float) -> None:
        """
        Update default confidence and IoU thresholds.
        
        Args:
            conf: New confidence threshold (0.0 to 1.0)
            iou: New IoU threshold (0.0 to 1.0)
        """
        if not (0.0 <= conf <= 1.0):
            raise ValueError("conf must be between 0.0 and 1.0")
        if not (0.0 <= iou <= 1.0):
            raise ValueError("iou must be between 0.0 and 1.0")
        
        self.conf_threshold = conf
        self.iou_threshold = iou
        print(f"Updated thresholds: conf={conf}, iou={iou}")
    
    def __repr__(self) -> str:
        """String representation of the segmenter."""
        return (
            f"ChickpeaSegmenter("
            f"device={self.device}, "
            f"conf={self.conf_threshold}, "
            f"iou={self.iou_threshold})"
        )


def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python chickpea_segmenter.py <image_path> [weights_path]")
        print("\nExample:")
        print("  python chickpea_segmenter.py test.jpg runs/segment/chickpea/weights/best.pt")
        sys.exit(1)
    
    image_path = sys.argv[1]
    weights_path = sys.argv[2] if len(sys.argv) > 2 else "runs/segment/chickpea/weights/best.pt"
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)
    
    # Initialize segmenter
    segmenter = ChickpeaSegmenter(weights_path)
    
    # Get predictions
    result = segmenter.predict(image)
    
    print(f"\nDetected {result['num_detections']} chickpea(s)")
    for i, (conf, box) in enumerate(zip(result['confidences'], result['boxes'])):
        print(f"  [{i+1}] confidence={conf:.3f}, box={box}")
    
    # Visualize
    combined = result['combined']
    colored = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    output_path = "output_segmentation.jpg"
    cv2.imwrite(output_path, colored)
    print(f"\nMask saved to {output_path}")


if __name__ == "__main__":
    main()
