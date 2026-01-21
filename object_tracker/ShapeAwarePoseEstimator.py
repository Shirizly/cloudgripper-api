import cv2
import numpy as np

class ShapeAwarePoseEstimator:
    def __init__(self,
                 outline_source,
                 mm_to_pixel,
                 camera_matrix,
                 dist_coeffs,
                 canny_thresh1=10,
                 canny_thresh2=600,edges_source = None):
        """
        outline_source: filename or Nx2 array of (x,y) in mm defining the object's contour.
        mm_to_pixel:   scale factor to convert mm → pixels for your nominal distance.
        camera_matrix, dist_coeffs: calibration parameters for cv2.solvePnP.
        canny_thresh1/2: thresholds for Canny edge detector.
        """
           
        # Load 2D model outline in mm
        if isinstance(outline_source, str):
            self.model_mm = np.loadtxt(outline_source, dtype=np.float32)
        else:
            self.model_mm = np.array(outline_source, dtype=np.float32)
        # flip x-coordinates sign since bottom camera is upside down
        self.model_mm[:, 0] = -self.model_mm[:, 0]

        # Convert to pixel template
        self.mm_to_pixel = mm_to_pixel
        self.model_px = (self.model_mm * mm_to_pixel).astype(np.int32)
        if edges_source is not None:
                # Load edges from file
                self.template_edges = cv2.imread(edges_source, cv2.IMREAD_GRAYSCALE)
                if self.template_edges is None:
                    raise ValueError(f"Could not load edges from {edges_source}")
        else: 
            # Compute a minimal bounding image for the template
            xs, ys = self.model_px[:,0], self.model_px[:,1]
            w, h = xs.max() - xs.min() + 10, ys.max() - ys.min() + 10
            canvas = np.zeros((h, w), dtype=np.uint8)

            # Draw filled contour & extract its edges
            shifted = self.model_px - [xs.min()-5, ys.min()-5]
            cv2.drawContours(canvas, [shifted.reshape(-1,1,2)], -1, 255, thickness=-1)
            self.template_edges = cv2.Canny(canvas, canny_thresh1, canny_thresh2)

            # save the template edges for future use
            cv2.imwrite("template_edges.png", self.template_edges)
        
        # # Show the template edges for debugging
        # cv2.imshow("Template Edges", self.template_edges)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Instantiate GHT detector
        self.ght = cv2.createGeneralizedHoughBallard()
        self.ght.setTemplate(self.template_edges)
        self.ght.setLevels(360) # number of levels for the GHT 

        # Keep pose‐estimation params
        self.camera_matrix = camera_matrix
        self.dist_coeffs   = dist_coeffs
        self.canny_t1      = canny_thresh1
        self.canny_t2      = canny_thresh2

    def detect_shape(self, image):
        """
        Run GHT on the image’s edge map.
        Returns the best detection as (x, y, scale, angle_deg), or None if nothing found.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_t1, self.canny_t2)
        # detect() returns array of [x, y, scale, angle]
        dets, votes = self.ght.detect(edges, None)
        if dets is None or len(dets) == 0:
            return None
        # Ensure dets is Nx4 and votes is Nx1
        dets = dets.reshape(-1, 4)
        votes = votes.reshape(-1, 3)
        # pick detection with highest votes
        best_idx = int(np.argmax(votes[:,0]))
        
        
        # debugging tools:
        # visualize the query image's edges
        # cv2.imshow("Canny Edges", edges)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # visualize the top matching detections
        # visualize_top_detections = self.visualize_top_detections(image, dets, votes, top_n=10)

        # visualize vote heatmap for debugging
        # if dets is not None:
        #     vote_map = estimator.generate_vote_map(image.shape, dets, votes)
        #     heatmap = cv2.applyColorMap(vote_map, cv2.COLORMAP_JET)
        #     overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        #     cv2.imshow("Vote Heatmap", overlay)
        #     cv2.waitKey(0)
        
        return tuple(dets[best_idx])
    
    def generate_rotated_template(self, angle_deg, scale=1.0):
        """
        Returns a rotated and scaled binary template image of the model.
        """
        h, w = self.template_edges.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle_deg, scale)
        rotated = cv2.warpAffine(self.template_edges, rot_mat, (w, h), flags=cv2.INTER_NEAREST)
        return rotated
    
    def refine_pose_with_template_matching(
        self,
        image_gray,
        coarse_pose,
        scale_range=(-0.02, 0.02), scale_step=0.01,
        angle_range=(-5, 5), angle_step=0.5
    ):
        """
        Refines the pose using template matching over (scale, angle).
        Uses matchTemplate to find the best (x, y) automatically.
        Returns the refined (x, y, scale, angle).
        """
        _, _, coarse_scale, coarse_angle = coarse_pose

        best_score = -np.inf
        best_pose = None

        for dscale in np.arange(scale_range[0], scale_range[1] + scale_step, scale_step):
            for dangle in np.arange(angle_range[0], angle_range[1] + angle_step, angle_step):
                scale = coarse_scale + dscale
                angle = coarse_angle + dangle

                tmpl = self.generate_rotated_template(angle, scale)
                if tmpl is None or tmpl.shape[0] < 5 or tmpl.shape[1] < 5:
                    continue

                if image_gray.shape[0] < tmpl.shape[0] or image_gray.shape[1] < tmpl.shape[1]:
                    continue  # skip if template too big

                res = cv2.matchTemplate(image_gray, tmpl, cv2.TM_CCOEFF_NORMED)
                _, score, _, max_loc = cv2.minMaxLoc(res)

                if score > best_score:
                    x, y = max_loc[0] + tmpl.shape[1] // 2, max_loc[1] + tmpl.shape[0] // 2
                    best_score = score
                    best_pose = (x, y, scale, angle)

        return best_pose
    
    def estimate_pose(self, image):
        """
        Full pipeline: detects the shape, computes 2D→3D correspondences, and solves PnP.
        Returns (success, rvec, tvec), where tvec is in the same units as your model_mm (mm).
        """
        # step 1: detect shape using GHT
        det = self.detect_shape(image)
        if det is None:
            print("Shape not found.")
            return False, None, None
        # self.visualize_detection(image,det)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coarse_pose = det
        gray = cv2.Canny(gray, self.canny_t1, self.canny_t2)

        # step 2: refine pose using grid search and template matching
        refined_pose = self.refine_pose_with_template_matching(
        image_gray=gray,
        coarse_pose=coarse_pose,
        scale_range=(-0.01, 0.01),
        scale_step=0.01,
        angle_range=(-5, 5),
        angle_step=0.2
        )
        print("refined pose:", refined_pose)
        # Step 3: Visualize result
        vis, outline = self.visualize_detection(image.copy(), refined_pose)
        cv2.imshow("Refined Pose", vis)
        cv2.waitKey(0)

        return refined_pose,outline 

        # # unpack detection
        # x_img, y_img, scale, angle = det
        # # convert angle from degrees to radians
        # angle = np.deg2rad(angle)
        # # build rotation matrix to map template px → image px
        # M = cv2.getRotationMatrix2D(center=(0,0), angle=angle, scale=scale)
        # # apply to each model pixel point (shifted so template origin is at 0,0)
        # xs, ys = self.model_px[:,0], self.model_px[:,1]
        # shift = np.array([xs.min()-5, ys.min()-5])
        # pts = (self.model_px - shift).astype(np.float32)
        # pts_rot = cv2.transform(pts.reshape(-1,1,2), M).reshape(-1,2)
        # # translate to image coordinates
        # img_pts = pts_rot + np.array([x_img, y_img], dtype=np.float32)

        # # pick four corners in consistent order
        # # (assumes model_mm has exactly 4 points)
        # if img_pts.shape[0] != 4:
        #     rect = cv2.minAreaRect(img_pts)
        #     box = cv2.boxPoints(rect)
        #     img_pts = np.array(box, dtype="float32")
        # def order4(pts):
        #     s = pts.sum(axis=1)
        #     diff = np.diff(pts, axis=1).ravel()
        #     return np.array([
        #         pts[np.argmin(s)],      # top-left
        #         pts[np.argmin(diff)],   # top-right
        #         pts[np.argmax(s)],      # bottom-right
        #         pts[np.argmax(diff)]    # bottom-left
        #     ], dtype=np.float32)
        # img4 = order4(img_pts)
        # obj4 = order4(self.model_mm)

        # # solvePnP (object Z=0 plane)
        # obj4_3d = np.hstack([obj4, np.zeros((4,1),dtype=np.float32)])
        # success, rvec, tvec = cv2.solvePnP(obj4_3d, img4,
        #                                    self.camera_matrix,
        #                                    self.dist_coeffs,
        #                                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
        # if not success:
        #     print("PnP failed.")
        #     return False, None, None
        # return True, rvec, tvec

# --- Methods for visualization ---

    def visualize_detection(self, image, detection, color=(0, 255, 0)):
        """
        Draw the detected outline on the image.
        
        detection: (x, y, scale, angle_deg)
        color:     BGR color for overlay
        Returns a copy of the image with the overlay drawn.
        """
        x, y, scale, angle = detection
        # Build rotation matrix around (0,0) 
        M = cv2.getRotationMatrix2D(center=(0, 0), angle=angle, scale=scale)

        # Transform the model outline (pixel coordinates, relative to origin)
        xs, ys = self.model_px[:, 0], self.model_px[:, 1]
        # first shift to center of model
        shift = np.array([(xs.max() + xs.min())/2, (ys.max() + ys.min())/2])
        outline_shifted = self.model_px - shift
        # Apply rotation and scale
        transformed = cv2.transform(outline_shifted.reshape(-1, 1, 2), M).reshape(-1, 2).astype(np.float32)
        transformed += np.array([x, y])  # translate into image

        # Draw
        img_vis = image.copy()
        cv2.polylines(img_vis, [np.round(transformed).astype(np.int32)], isClosed=True, color=color, thickness=2)

        # Draw center point
        center = (int(round(x)), int(round(y)))
        cv2.circle(img_vis, center, 4, (0, 0, 255), -1)

        # Draw orientation arrow
        angle_rad = np.deg2rad(angle)
        arrow_length = 50  # pixels
        tip = (
            int(round(x + arrow_length * np.cos(angle_rad))),
            int(round(y + arrow_length * np.sin(angle_rad)))
        )
        cv2.arrowedLine(img_vis, center, tip, (255, 0, 0), 2, tipLength=0.2)
        cv2.imshow("Detected Shape", img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return img_vis, transformed
    
    def crop_around_detection(self, image, detection, box_size=200):
        """
        Crop a square region around the detected (x, y) center.
        Useful for checking what's being matched.
        """
        x, y = int(detection[0]), int(detection[1])
        h, w = image.shape[:2]
        half = box_size // 2

        x1, x2 = max(0, x - half), min(w, x + half)
        y1, y2 = max(0, y - half), min(h, y + half)

        return image[y1:y2, x1:x2].copy()

    def visualize_top_detections(self, image, dets, votes, top_n=5):
        """
        Visualize the top-N detections based on vote strength.
        """
        dets = dets.reshape(-1, 4)
        votes = votes.reshape(-1, votes.shape[-1])  # shape: (N, 3)

        scores = votes[:, 0]  # primary vote score
        idxs = np.argsort(-scores)[:top_n]

        img_overlay = image.copy()

        for i, idx in enumerate(idxs):
            x, y, scale, angle = dets[idx]
            color = tuple(int(c) for c in np.random.randint(100, 255, 3))  # unique color
            img_overlay = self.visualize_detection(img_overlay, (x, y, scale, angle), color=color)
            cv2.putText(img_overlay, f"#{i+1}", (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return img_overlay
    
    def generate_vote_map(self, image_shape, dets, votes, normalize=True):
        """
        Generate a 2D heatmap of votes over the image.
        Returns a grayscale image of the same size as the input image.
        """
        h, w = image_shape[:2]
        dets = dets.reshape(-1, 4)
        votes = votes.reshape(-1, votes.shape[-1])
        scores = votes[:, 0]

        vote_map = np.zeros((h, w), dtype=np.float32)

        for (x, y), score in zip(dets[:, :2], scores):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                vote_map[yi, xi] += score  # accumulate votes at that location

        if normalize and vote_map.max() > 0:
            vote_map = cv2.normalize(vote_map, None, 0, 255, cv2.NORM_MINMAX)

        return vote_map.astype(np.uint8)


# --- example usage ---
if __name__ == "__main__":
    # 1) load model outline
    object_outline = 'object_tracker/object_outline.txt'  # Path to the outline file (or a list of points).
    
    # Define the expected color range in RGB (object's color).
    # Ours is black
    color_range_low = (0, 0, 0)
    brightness_threshold = 80
    color_range_high = (brightness_threshold, brightness_threshold, brightness_threshold)
    
    # Define a nominal mm-to-pixel conversion (this would be empirically determined).
    mm_to_pixel = 3.255  # Example: each mm corresponds to 0.5 pixels for the approximate scale

    # Example camera parameters (you must calibrate your camera to obtain these).
    

    # 2) load camera intrinsics 
    camera_matrix = np.array([ [505.24537524391866, 0.0, 324.5096286632362], [0.0, 505.6456651337437, 233.54118730278543], [0.0, 0.0, 1.0]])
    dist_coeffs = np.zeros((4, 1))  # assuming zero distortion for this example


    # 3) instantiate
    estimator = ShapeAwarePoseEstimator(
        outline_source=object_outline,
        mm_to_pixel=mm_to_pixel,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        edges_source="object_tracker/template_edges.png",  # Optional: path to precomputed edges
    )

    # 4) run on an image
    image = cv2.imread("example_image.jpg")
    # image = image[100:800, 50:700]  # Crop to a smaller region for testing
    estimator.estimate_pose(image)
    # if ok:
    #     print("Rotation vector:\n", rvec)
    #     print("Translation (mm):\n", tvec)
    #     # pts = pts.reshape((-1, 1, 2)).astype(np.int32)
    #     # cv2.drawContours(image, [pts], -1, (0, 255, 0), 2)
    #     # cv2.imshow("Pose Estimation", image)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    # else:
    #     print("Detection/pose failed.")