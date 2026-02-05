import cv2
from matplotlib.pyplot import flag
import numpy as np
import threading
import time
import random
import sys
import os
from pynput import keyboard  # Replace 'keyboard' with 'pynput'
from dotenv import load_dotenv
from cv2displayer import update_images
from queue import Queue


calibration_flag = False  # Set to True to save one image for calibration
save_img = True  # Set to True to save one image from the base camera

print("Starting manual control...", flush=True)
state_queue = Queue()
# Ensure the project root is in the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from client.cloudgripper_client import GripperRobot
from autograsper.library.utils import get_undistorted_bottom_image
from object_tracker.granular_utils import process_image, crop_center_region
from object_tracker.base_tool_tracker import find_thin_tool_center
from autograsper.custom_graspers.fence_utils import PixelRobotTransform
from sklearn.mixture import GaussianMixture

# Load environment variables
load_dotenv()

# Get the CloudGripper API token from environment variables
token = os.getenv("CLOUDGRIPPER_TOKEN")

# Create a GripperRobot instance
robotName = "robot24"
robot = GripperRobot(robotName, token)

# Shared state
running = True
lock = threading.Lock()  # Ensure thread safety

# video recording setup
save_video = True        # turn on/off saving
video_path = "robot_cam_output.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = None                # will initialize once we know frame size
frame_times = []          # store timestamps to estimate frame rate later
# Load calibration parameters
calib = np.load("calibration_params.npz")
camera_matrix = calib["arr_1"]
distortion_coeffs = calib["arr_2"]
print("Camera matrix:", camera_matrix)
print("Distortion coefficients:", distortion_coeffs)
calibration_dict = {}

H = None
if os.path.exists("homography_matrix.npy"):
    H = np.load("homography_matrix.npy")

reference_empty = cv2.imread("reference_empty_base.jpg")
cropped_reference_empty = cv2.imread("cropped_reference.png")
# src_pts = [(33,162), (236,164), (236,372), (32,368)]  # Example source points
# scaling = 240
# dst_pts = [(0,0), (scaling,0), (scaling,scaling), (0,scaling)]  # Example destination points
# H = cv2.findHomography(np.array(src_pts), np.array(dst_pts))[0]
# print("Homography matrix loaded. H:", H)
# np.save("homography_matrix.npy", H)


# Function to update camera feed
def update_camera():
    global running, current_config, calibration_flag, save_img, reference_empty, calibration_dict, cropped_reference_empty
    out = None
    prev_center_np = np.array([1,1])
    pixTransH = np.load("homography.npz")['arr_0'] if os.path.exists("homography.npz") else None
    pixel_robot_transform = PixelRobotTransform(pixTransH) if pixTransH is not None else None
    while running:
        with lock:
            # Get images from both cameras
            data = robot.get_all_states()  # Get new images and robot configuration - if doesn't work, switch to separate calls
            time.sleep(0.15)
            image_top = data[0]
            image_base_raw = data[1]
            
            state = data[2]
            # image_top, _ = robot.get_image_top()
            # time.sleep(0.5)
            # image_base, _ = robot.get_image_base()
            # time.sleep(0.5) 

        if image_top is None or image_base_raw is None:
            print("Failed to retrieve images from robot.")
            continue
        if calibration_flag:
            cv2.imwrite("calibrate.jpg", np.array(image_base_raw))
            calibration_flag = False
        # Undistort bottom image
        image_base = get_undistorted_bottom_image(image_base_raw, camera_matrix, distortion_coeffs, H)
        if save_img:
            cv2.imwrite("latest_base.jpg", np.array(image_base))
            save_img = False

        # Convert images to NumPy arrays for OpenCV
        img_top = np.array(image_top)
        img_base = np.array(image_base)
        # img_base = np.transpose(img_base, (1, 0, 2))  # Rotate base image

        # combine images side by side
        def combine_side_by_side(img_left, img_right, common_height=480):
            """
            Resize both images to the same height and concatenate horizontally.
            """
            # Compute aspect-preserving scaling
            hL, wL = img_left.shape[:2]
            hR, wR = img_right.shape[:2]

            scaleL = common_height / hL
            scaleR = common_height / hR

            new_wL = int(wL * scaleL)
            new_wR = int(wR * scaleR)

            # Resize both
            left_resized = cv2.resize(img_left, (new_wL, common_height))
            right_resized = cv2.resize(img_right, (new_wR, common_height))

            # Ensure same height
            if left_resized.shape[0] != right_resized.shape[0]:
                common_height = min(left_resized.shape[0], right_resized.shape[0])
                left_resized = cv2.resize(left_resized, (new_wL, common_height))
                right_resized = cv2.resize(right_resized, (new_wR, common_height))

            combined = np.hstack((left_resized, right_resized))
            return combined

        frame = combine_side_by_side(img_top, img_base, common_height=480)
        now = time.time()
        frame_times.append(now)
        # Initialize video writer if needed
        if save_video and out is None:
            height, width, _ = frame.shape
            out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))  
        
        # Print current configuration
        prev_config = current_config
        current_config = np.array(list(state.values())[:5])  # x, y, z, rotation, gripper
        if np.any(np.abs(current_config - prev_config) > 1e-6):
            # state_queue.put(current_config)
            # sys.stdout.write(f"Current configuration: {[ '%.2f' % elem for elem in current_config]}\n")
            # sys.stdout.flush()
            print(f"Current configuration: {[ '%.2f' % elem for elem in current_config]}", flush=True)
        
        # Write frame
        if save_video and out is not None:
            out.write(frame)
        # 
        t_start = time.time()
        cropped_image = crop_center_region(img_base)
        mask,_,_,_ = process_image(img_base, reference_empty, scale_factor=4, min_size=400)  # for chickpeas
        masked_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
        # analyze masked image to find color gaussian peaks
        def get_color_gaussians(masked_image, num_gaussians=3):
            """
            Fit Gaussian distributions to the dominant colors in a masked image.
            Returns a list of Gaussians ordered by dominance (weight).
            """
            
            # Reshape image to list of pixels
            pixels = masked_image.reshape(-1, 3).astype(np.float32)
            
            # Remove black pixels (background)
            non_black = pixels[np.any(pixels > 10, axis=1)]
            
            if len(non_black) < num_gaussians:
                return []
            
            # Fit Gaussian Mixture Model
            gmm = GaussianMixture(n_components=num_gaussians, random_state=42)
            gmm.fit(non_black)
            
            # Sort by weight (dominance)
            sorted_indices = np.argsort(-gmm.weights_)
            
            gaussians = []
            for idx in sorted_indices:
                gaussians.append({
                    'mean': gmm.means_[idx],
                    'covariance': gmm.covariances_[idx],
                    'weight': gmm.weights_[idx]
                })
            
            return gaussians

        # color_gaussians = get_color_gaussians(masked_image, num_gaussians=4)
        # print("Detected color gaussians (most to least dominant):")
        # for i, g in enumerate(color_gaussians):
        #     mean_color = g['mean']
        #     weight = g['weight']
        #     covariance = g['covariance']
        #     print(f"  Gaussian {i+1}: Mean Color (BGR) = {mean_color}, covariance = {covariance}, Weight = {weight:.4f}") 
        #     # visualize gaussians on image
        #     cv2.circle(cropped_image, (i*10+20,i*10+20), 10, (int(mean_color[0]), int(mean_color[1]), int(mean_color[2])), -1)
        cv2.imwrite("latest_manual_base_cropped.png", cropped_image)
        # center,_,_ = find_thin_tool_center(cropped_image,current_config[3]-90,3,112,2)
        center = pixel_robot_transform.robot_to_pix(current_config[0], current_config[1]) if pixel_robot_transform is not None else (0,0)
        t_end = time.time()
        x,y = center
        center_np = np.array([x,y])
        # print(f"Image processed in {t_end - t_start:.2f} seconds.", flush=True)
        # if np.linalg.norm(center_np-prev_center_np)>1:
        #     print(center)
        #     prev_center_np = center_np
        
        # Display images
        # cv2.circle(masked_image, (x, y), 6, (0, 0, 255), -1)
        update_images([cropped_image, masked_image], window_name="Robot Cameras")  # Display images side by side

        
        # Check if window is closed
        if cv2.waitKey(1) == 27:  # Escape key to exit
            running = False
            break

# Function to handle keyboard input
def on_press(key):
    global running, current_config, step_size
    # step_size = 0.01  # Step size for robot movement
    # current_config = np.clip(current_config, [0,0,0,-180,0], [1,1,1,180,1])  # Ensure within bounds
    current_config[0] = max(0, min(1, current_config[0]))
    current_config[1] = max(0, min(1, current_config[1]))
    current_config[2] = max(0, min(1, current_config[2]))
    current_config[3] = max(-180, min(180, current_config[3]))
    current_config[4] = max(0, min(1, current_config[4]))
    try:
        if key.char == "a":  # Move left (X-)
            current_config[0] -= step_size
            current_config[0] = max(0, current_config[0])  # Prevent going below 0
            print(f"requesting move to {current_config[0]}, {current_config[1]}")
            robot.move_xy(current_config[0], current_config[1])
            # robot.step_left()
        elif key.char == "d":  # Move right (X+)
            current_config[0] += step_size
            current_config[0] = min(1, current_config[0])  # Prevent going above 1
            print(f"requesting move to {current_config[0]}, {current_config[1]}")
            robot.move_xy(current_config[0], current_config[1])
            # robot.step_right()
        elif key.char == "w":  # Move forward (Y+)
            current_config[1] += step_size
            current_config[1] = min(1, current_config[1])  # Prevent going above 1
            print(f"requesting move to {current_config[0]}, {current_config[1]}")
            robot.move_xy(current_config[0], current_config[1])
            # robot.step_forward()
        elif key.char == "s":  # Move backward (Y-)
            current_config[1] -= step_size
            current_config[1] = max(0, current_config[1])  # Prevent going below 0
            print(f"requesting move to {current_config[0]}, {current_config[1]}")
            robot.move_xy(current_config[0], current_config[1])
            # robot.step_backward()
        elif key.char == "z":  # Move down (Z-)
            current_config[2] -= step_size
            current_config[2] = max(0, current_config[2])  # Prevent going below 0
            robot.move_z(current_config[2])
        elif key.char == "c":  # Move up (Z+)
            current_config[2] += step_size
            current_config[2] = min(1, current_config[2])  # Prevent going above 1
            robot.move_z(current_config[2])
        elif key.char == "q":  # Rotate counterclockwise
            current_config[3] -= 10
            current_config[3] = max(-180, current_config[3])
            robot.rotate(int(current_config[3]))
        elif key.char == "e":  # Rotate clockwise
            current_config[3] += 10
            current_config[3] = min(360, current_config[3])
            robot.rotate(int(current_config[3]))
        elif key.char == "x":  # close gripper
            robot.gripper_close()
        elif key.char == "u":  # Open gripper fully
            robot.gripper_open()
        elif key.char == "j":  # Open gripper a little
            current_config[4] -= 0.1
            current_config[4] = max(0, current_config[4])
            robot.move_gripper(current_config[4])
        elif key.char == "k":  # Close gripper a little
            current_config[4] += 0.1
            current_config[4] = min(1, current_config[4])
            robot.move_gripper(current_config[4])
        elif key.char == "n": # increase step size
            step_size = step_size*2
        elif key.char == "m": # decrease step size
            step_size = step_size/2
        elif key.char == "p":  # Stop robot
            running = False
            return False  # Stop listener
        
        # print(f"Current configuration: {current_config}")
        time.sleep(0.1)  # Delay to prevent rapid key presses
    except AttributeError:
        pass  # Handle non-character keys safely

def on_release(key):
    global running
    if key == keyboard.Key.backspace:  # Quit on backspace
        running = False
        return False  # Stop listener

# Get initial robot state
# robot.calibrate()
state = robot.get_state()
# robot.move_xy(0.5, 0.5)  # Move to a safe starting position
time.sleep(1)
state = robot.get_state()
if state[0] is not None:
    current_config = np.array(list(state[0].values())[:5])  # x, y, z, rotation, gripper
    global step_size
    step_size = 0.01  # Step size for robot movement
    print(f"Current configuration: {[ '%.2f' % elem for elem in current_config]}\n", flush=True)
# Start camera thread and keyboard listener - new safety version
try:
    # Start camera + keyboard
    camera_thread = threading.Thread(target=update_camera, daemon=True)
    camera_thread.start()

    with keyboard.Listener(on_press=on_press) as listener: #, on_release=on_release
        listener.join()

except KeyboardInterrupt:
    print("Keyboard interrupt: stopping...")

finally:
    running = False
    if camera_thread.is_alive():
        camera_thread.join()
    cv2.destroyAllWindows()
    print("Cleanup done, exiting.")



# # Start camera thread - old version
# camera_thread = threading.Thread(target=update_camera)
# camera_thread.start()

# # Start keyboard listener
# with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
#     listener.join()  # Keep listening until backspace is pressed

# # Cleanup
# running = False
# camera_thread.join()
# cv2.destroyAllWindows()

