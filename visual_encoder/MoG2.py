import os
import numpy as np
import cv2
from tqdm import tqdm

def get_neighboring_pixel_values(image, x, y, window_size=3):
    """Extracts values of neighboring pixels within a given window size."""
    h, w, c = image.shape
    half_w = window_size // 2
    
    # Get valid neighbor coordinates
    x_min, x_max = max(0, x - half_w), min(h, x + half_w + 1)
    y_min, y_max = max(0, y - half_w), min(w, y + half_w + 1)
    
    neighbors = image[x_min:x_max, y_min:y_max].reshape(-1, c)
    return neighbors

class PixelwiseGMM:
    def __init__(self, height, width, num_components=3, alpha=0.01, var_threshold=2.5):
        """
        Initialize a per-pixel Gaussian Mixture Model (GMM).
        
        Parameters:
        - height, width: Dimensions of the image.
        - num_components: Number of Gaussian components per pixel.
        - alpha: Learning rate for updating the model.
        - var_threshold: Threshold for classifying foreground based on Mahalanobis distance.
        """
        self.height = height
        self.width = width
        self.num_components = num_components
        self.alpha = alpha
        self.var_threshold = var_threshold
        
        # Initialize GMM parameters
        self.means = np.zeros((height, width, num_components, 3))  # 3D means per Gaussian
        self.covariances = np.zeros((height, width, num_components, 3, 3))  # Full covariance matrices
        self.weights = np.full((height, width, num_components), 1 / num_components)  # Even weights initially

    def initialize_simple(self, background_median, initial_variance=400.0):
        """Initialize the means with the median image and set high initial variances."""
        for k in range(self.num_components):
            self.means[:, :, k, :] = background_median  # Initialize all means to median image
            self.covariances[:, :, k, :, :] = np.eye(3) * initial_variance  # Diagonal covariance


    def initialize(self, background_median, initial_variance=400.0, window_size=3):
        """Initialize the GMM with means sampled from neighboring pixels."""
        h, w, c = background_median.shape
        
        for x in range(h):
            for y in range(w):
                # Get neighboring pixel values
                neighbors = get_neighboring_pixel_values(background_median, x, y, window_size)

                # Randomly assign means from the neighborhood
                selected_means = neighbors[np.random.choice(neighbors.shape[0], self.num_components, replace=True)]
                self.means[x, y, :, :] = selected_means

                # Assign different variances to each Gaussian
                variance_scale = np.random.uniform(0.5, 1.5, size=(self.num_components, 1))  # Random variance scaling
                self.covariances[x, y, :, :, :] = np.eye(3) * (initial_variance * variance_scale[:, None])

                # Assign weights based on closeness to median
                distances = np.linalg.norm(selected_means - background_median[x, y], axis=-1)
                self.weights[x, y, :] = np.exp(-distances / 50)  # Closer colors get higher initial weights
                
        # Normalize weights so they sum to 1
        self.weights /= np.sum(self.weights, axis=2, keepdims=True)

    def update(self, image):
        """Update the GMM parameters using the current image frame."""
        diff = image[:, :, None, :] - self.means  # Shape: (H, W, K, 3)
        
        # Compute Mahalanobis distance for each component
        distances = np.zeros((self.height, self.width, self.num_components))
        for k in range(self.num_components):
            cov_inv = np.linalg.inv(self.covariances[:, :, k, :, :])
            distances[:, :, k] = np.einsum('...i,...ij,...j->...', diff[:, :, k, :], cov_inv, diff[:, :, k, :])
        
        # Find the best-matching Gaussian (smallest Mahalanobis distance)
        best_match = np.argmin(distances, axis=2)
        
        # Update the matched Gaussian using a running average
        for y in range(self.height):
            for x in range(self.width):
                k = best_match[y, x]
                self.weights[y, x, k] = (1 - self.alpha) * self.weights[y, x, k] + self.alpha
                
                # Update mean
                self.means[y, x, k, :] = (1 - self.alpha) * self.means[y, x, k, :] + self.alpha * image[y, x, :]
                
                # Update covariance using outer product
                diff_vec = image[y, x, :] - self.means[y, x, k, :]
                self.covariances[y, x, k, :, :] = (1 - self.alpha) * self.covariances[y, x, k, :, :] + self.alpha * np.outer(diff_vec, diff_vec)
        
        # Normalize weights
        self.weights /= np.sum(self.weights, axis=2, keepdims=True)

    def detect_foreground(self, image):
        """Classify pixels as foreground if they do not match any Gaussian."""
        foreground_mask = np.ones((self.height, self.width), dtype=np.uint8)  # Assume all foreground initially

        for k in range(self.num_components):
            diff = image - self.means[:, :, k, :]
            cov_inv = np.linalg.inv(self.covariances[:, :, k, :, :])
            distances = np.einsum('...i,...ij,...j->...', diff, cov_inv, diff)
            
            # If any Gaussian in the mixture classifies a pixel as background, mark it as such
            background_match = distances < self.var_threshold
            foreground_mask[background_match] = 0
        
        return foreground_mask

def load_images_from_folder(folder_path):
    """Load images from a folder into a NumPy array."""
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img.astype(np.float32))
            if len(images) >= 50:
                break
    return np.array(images)

def main(folder_path, num_components=3):
    images = load_images_from_folder(folder_path)
    height, width, _ = images[0].shape

    # Compute the median background image
    background_median = np.median(images, axis=0)

    # Initialize and train the GMM
    gmm = PixelwiseGMM(height, width, num_components)
    gmm.initialize(background_median)

    for image in tqdm(images, desc="Training GMM"):
        gmm.update(image)

    # Perform background subtraction on the last frame
    foreground_mask = gmm.detect_foreground(images[-1])

    # Display results
    cv2.imshow("Foreground Mask", foreground_mask * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    recordings_dir = os.path.join(project_root,"recordings")
    # generate list of data directories
    sequence_dirs = []
    log_file = []
    if os.path.exists(recordings_dir):
        dataset_dirs = [d for d in os.listdir(recordings_dir) if d.startswith("dataset_0_sequence_")]
        for sequence in dataset_dirs:
            sequence_dirs.append(os.path.join(os.path.join(recordings_dir, sequence)))
    main(os.path.join(sequence_dirs[0],"top_images"))

