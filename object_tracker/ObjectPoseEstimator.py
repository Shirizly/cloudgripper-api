import cv2
import numpy as np

class ObjectPoseEstimator:
    def __init__(self, outline_source, color_range_low, color_range_high, mm_to_pixel, camera_matrix, dist_coeffs):
        """
        outline_source: either a string filename containing the object outline or a list/array of points [[x, y], ...] in mm.
        color_range_low: lower bound of expected object color (R, G, B) tuple.
        color_range_high: upper bound of expected object color (R, G, B) tuple.
        mm_to_pixel: scaling constant to convert mm to pixel (nominal scale for the stored outline).
        camera_matrix: intrinsic camera matrix (numpy array).
        dist_coeffs: distortion coefficients (numpy array).
        """
        self.mm_to_pixel = mm_to_pixel
        self.color_range_low = np.array(color_range_low, dtype=np.uint8)
        self.color_range_high = np.array(color_range_high, dtype=np.uint8)
        self.camera_matrix = np.array(camera_matrix,dtype=np.float32)
        self.dist_coeffs = dist_coeffs

        # Load the object outline.
        if isinstance(outline_source, str):
            # Assumes a simple text file with two columns (x y) per line in mm.
            self.object_outline_mm = np.loadtxt(outline_source, dtype=np.float32)
        else:
            self.object_outline_mm = np.array(outline_source, dtype=np.float32)

        # For initial contour matching, convert the outline from mm to an approximate pixel scale.
        self.object_outline_px = self.object_outline_mm * self.mm_to_pixel
        
    @staticmethod
    def order_points(pts):
        """
        Order a set of 2D points in the order: top-left, top-right, bottom-right, bottom-left.
        Expects pts as an (N,2) numpy array with N=4.
        """
        # initialize a list of coordinates that will be ordered
        rect = np.zeros((4, 2), dtype="float32")
        # the top-left point will have the smallest sum,
        # whereas the bottom-right will have the largest sum.
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between points:
        # the top-right will have the smallest difference,
        # whereas the bottom-left will have the largest difference.
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def draw_contour(contours,image):
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        cv2.imshow("Pose Estimation", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def estimate_pose(self, image):
        """
        Processes the given image, locates the object, and estimates its pose.
        image: input image (numpy array). Assumes image is in BGR if loaded via cv2.imread.
        Returns: (success, rotation_vector, translation_vector). If unsuccessful, success is False.
        """
        # Convert to RGB (because the expected color ranges are in RGB)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a binary mask using the color thresholds.
        mask = cv2.inRange(img_rgb, self.color_range_low, self.color_range_high)

        # Optional: Clean up the mask.
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the mask.
        contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Compatibility with different OpenCV versions:
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

        if not contours:
            print("No contours found.")
            return False, None, None
        self.draw_contour(contours,image)
        # Choose the contour that best matches the saved outline.
        best_match = None
        best_score = float('inf')
        for cnt in contours:
            # Approximate contour to have fewer points; epsilon is a fraction of the arc length.
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 4:
                continue  # Skip if contour is too simple.
            # For matching, convert approx to a simple 2D array (flattened).
            approx_pts = approx.reshape( -1, 2).astype(np.float32)
            score = cv2.matchShapes(self.object_outline_px, approx_pts, cv2.CONTOURS_MATCH_I1, 0.0)
            if score < best_score:
                best_score = score
                best_match = approx_pts

        if best_match is None:
            print("No matching contour found.")
            return False, None, None

        # If best_match is not exactly 4 points, use a minAreaRect to obtain a quadrilateral.
        if best_match.shape[0] != 4:
            rect = cv2.minAreaRect(best_match)
            box = cv2.boxPoints(rect)
            best_match = np.array(box, dtype="float32")

        object_points = self.object_outline_mm
        if object_points.shape[0] !=4:
            rect = cv2.minAreaRect(best_match)
            box = cv2.boxPoints(rect)
            object_points = self.order_points(np.array(box, dtype="float32"))


        # Order the detected contour points.
        image_points = self.order_points(best_match)
        # For the pose estimation, use the original model outline points in mm.
        # (Assuming the object lies on a plane with z=0.)
        object_points = np.hstack([object_points, np.zeros((4, 1), dtype=np.float32)])
        # Now call solvePnP: note that the object_points are in mm and image_points in pixels.
        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs
        )
        if not success:
            print("solvePnP failed to find a valid pose.")
            return False, None, None

        return True, rotation_vector, translation_vector

# ----- Example Usage -----
if __name__ == "__main__":
    # Define parameters for the object.
    object_outline = 'object_outline.txt'  # Path to the outline file (or a list of points).
    
    # Define the expected color range in RGB (object's color).
    # Ours is black
    color_range_low = (0, 0, 0)
    brightness_threshold = 60
    color_range_high = (brightness_threshold, brightness_threshold, brightness_threshold)
    
    # Define a nominal mm-to-pixel conversion (this would be empirically determined).
    mm_to_pixel = 3.1  # Example: each mm corresponds to 0.5 pixels for the approximate scale

    # Example camera parameters (you must calibrate your camera to obtain these).
    camera_matrix = [ [505.24537524391866, 0.0, 324.5096286632362], [0.0, 505.6456651337437, 233.54118730278543], [0.0, 0.0, 1.0]]
    dist_coeffs = np.zeros((4, 1))  # assuming zero distortion for this example

    # Initialize the pose estimator.
    estimator = ObjectPoseEstimator(object_outline, color_range_low, color_range_high, mm_to_pixel, camera_matrix, dist_coeffs)

    # Load an example image (change the filename as appropriate).
    image = cv2.imread("example_image.jpg")
    if image is None:
        print("Error loading image.")
    else:
        # Estimate the object pose in the image.
        success, rvec, tvec = estimator.estimate_pose(image)
        if success:
            print("Pose estimation succeeded.")
            print("Rotation vector:\n", rvec)
            print("Translation vector (in mm):\n", tvec)
            # Optionally, draw the detected contour on the image for visualization.
            # Here we re-use the detected contour points from the last estimate (make a copy of image_points).
            pts = estimator.order_points(estimator.object_outline_px)  # This is the model shape in pixel scale.
            pts = pts.reshape((-1, 1, 2)).astype(np.int32)
            cv2.drawContours(image, [pts], -1, (0, 255, 0), 2)
            cv2.imshow("Pose Estimation", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Pose estimation failed.")