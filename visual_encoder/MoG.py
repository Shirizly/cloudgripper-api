import numpy as np
import cv2

class AdaptiveMOG:
    def __init__(self, height, width, num_gaussians=3, alpha=0.01, bg_threshold=0.7, var_init=15, var_min=5, var_max=50):
        self.num_gaussians = num_gaussians  # Number of Gaussians per pixel
        self.alpha = alpha  # Learning rate
        self.bg_threshold = bg_threshold  # Background weight threshold
        self.var_min = var_min  # Minimum variance
        self.var_max = var_max  # Maximum variance
        
        # Initialize mixture model: means, variances, and weights per Gaussian
        self.means = np.zeros((height, width, num_gaussians), dtype=np.float32)
        self.variances = np.full((height, width, num_gaussians), var_init, dtype=np.float32)
        self.weights = np.full((height, width, num_gaussians), 1.0 / num_gaussians, dtype=np.float32)
    
    def initialize_with_median(self, median_image):
        """Initialize Gaussian means with the median background image."""
        self.means[:, :, 0] = median_image  # Set the first Gaussian to the median background
        self.weights[:, :, 0] = 0.8  # Assign higher weight to this background Gaussian
        self.weights[:, :, 1:] = 0.2 / (self.num_gaussians - 1)  # Spread small weights over others
    
    def update(self, frame):
        """Update the Gaussian mixture model with a new frame."""
        diff = np.abs(frame[:, :, None] - self.means)  # Compute pixel-wise difference
        match_mask = diff < (2.5 * np.sqrt(self.variances))  # Match if within 2.5 std-dev
        
        # Update weights and renormalize
        self.weights = (1 - self.alpha) * self.weights + self.alpha * match_mask
        self.weights /= np.sum(self.weights, axis=2, keepdims=True)  # Normalize weights
        
        # Update means and variances where matches occur
        rho = self.alpha * self.weights  # Adaptation rate
        self.means = (1 - rho) * self.means + rho * frame[:, :, None] * match_mask
        
        # Update variance, keeping it within valid bounds
        variance_update = (1 - rho) * self.variances + rho * (diff ** 2) * match_mask
        self.variances = np.clip(variance_update, self.var_min, self.var_max)
    
    def get_background_mask(self, frame):
        """Generate a binary mask where the background is 1 and foreground is 0."""
        diff = np.abs(frame[:, :, None] - self.means)
        match_mask = diff < (2.5 * np.sqrt(self.variances))  # Identify matching Gaussians
        
        # Compute cumulative weights of matching Gaussians
        bg_mask = np.sum(self.weights * match_mask, axis=2) > self.bg_threshold
        return bg_mask.astype(np.uint8) * 255

# Example usage:
if __name__ == "__main__":
    cap = cv2.VideoCapture("video.mp4")  # Replace with actual video or dataset
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    
    median_bg = cv2.imread("median_image.png", cv2.IMREAD_GRAYSCALE)  # Load median background image
    mog = AdaptiveMOG(height, width)
    mog.initialize_with_median(median_bg)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mog.update(gray_frame)
        bg_mask = mog.get_background_mask(gray_frame)
        
        cv2.imshow("Background Mask", bg_mask)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()