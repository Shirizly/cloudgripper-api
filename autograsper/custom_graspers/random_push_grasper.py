# custom_graspers/random_push_grasper.py
import logging
from matplotlib.pyplot import flag
from autograsper.coordinator import SharedState
from grasper import AutograsperBase, RobotActivity, sleep_with_shutdown
import numpy as np
import time
from library.utils import OrderType

import random
from object_tracker.tool_user_utils import analyze_tool_grip
from object_tracker.granular_utils import process_image, check_reset_needed
import cv2
from autograsper.custom_graspers.fence_utils import Wall, build_fence_walls, find_tool_placements, check_wall_reset_needed, sample_tool_pose, PixelRobotTransform, get_pos_sweep_from_optimal
##### Helper functions for wall definition and sampling #####




##### Random Push Grasper Class #####

class RandomPushGrasper(AutograsperBase):
    def __init__(self, config, shutdown_event, N_pushes=10):
        super().__init__(config, shutdown_event=shutdown_event)
        self.N_pushes = N_pushes
        self.grasp_height = 0.34
        self.sweep_height = 0.57
        self.clearance_height = 0.8
        self.config = config
        self.walls = build_fence_walls(
            fence_center=config.get('fence', {}).get('center', (0.5,0.5)),
            fence_size=config.get('fence', {}).get('size', (0.9, 0.9)),
            tool_length=0.36,
            tool_width=0.017,
            safety_margin=0.01
        )
        for wall in self.walls:
            print(wall.label)
            dict = sample_tool_pose(wall, tool_width=0.02, safety_margin=0.02,t=-1)
            print(dict)
            dict = sample_tool_pose(wall, tool_width=0.02, safety_margin=0.02,t=1)
            print(dict)
        self.reset_step_size = 0.15 # base step size for reset sweeps
        # Hardcoded safe X,Y ranges (example, adjust to your setup)
        self.manip_x = config.get('fence', {}).get('manipulation_boundary',{}).get('x', (0.3,0.7))
        self.manip_y = config.get('fence', {}).get('manipulation_boundary',{}).get('y', (0.3,0.7))
        self.image_space_manip_boundary = config.get('fence', {}).get('image_space_manipulation_boundary', None)
        self.image_space_tool_dimensions = config.get('fence', {}).get('image_space_tool_dimensions', None)
        self.tool_detection_threshold = self.config.get('tool_detection',{}).get('detection_threshold',0.6)
        self.Homography_matrix_path = self.config.get('fence',{}).get('Homography_matrix_path',"homography.npz")
        temp = np.load(self.Homography_matrix_path)
        self.Homography_matrix = temp['arr_0']
        self.pix2robtrans = PixelRobotTransform(self.Homography_matrix)
        self.interaction_since_last_mask = True
        self.reference_image_path = config.get('Granuler_detection', {}).get('reference_image_path', None)
        self.min_granule_size = config.get('Granuler_detection', {}).get('min_granule_size', 300)
        self.reference_image = None
        if self.reference_image_path is not None:
            self.reference_image = cv2.imread(self.reference_image_path)
            if self.reference_image is None:
                print(f"Warning: Could not load reference image from {self.reference_image_path}")
        self.latest_mask = None
        self.latest_num_clumps = None
        self.latest_stats = None
        self.latest_centroids = None
        self.robot_state = None  # Track robot state (position, orientation, gripper)
        self.shared_state = None  # to be set by coordinator
        self.state = RobotActivity.STARTUP

    def connect_shared_state(self, shared_state: 'SharedState'):
        self.shared_state = shared_state

    def update_robot_state(self):
        with self.shared_state.image_lock:
            self.robot_state = self.shared_state.latest_robot_state


    def update_mask_and_process(self):
        """
        Move robot out of the way, capture bottom image, and update latest_mask.
        Call this before any check_wall_reset_needed invocation.
        Returns True if successful, False otherwise.
        """
        # Move to safe position out of the way
        if not self.interaction_since_last_mask:
            print("No interaction since last mask, skipping update")
            return True

        orders = [
            (OrderType.MOVE_Z, [1.0]),
            # (OrderType.MOVE_XY, [0.5, 1.0]),  # Move to corner out of workspace
            (OrderType.MOVE_XY, [0.0, 1.0]),  # Move to corner out of workspace
            (OrderType.ROTATE, [90])
        ]
        self.queue_orders(orders)
        time.sleep(1.5)  # Wait for motion to complete
        self.update_robot_state()
        # Capture latest bottom image
        capture_flag = False
        while not capture_flag:
            with self.shared_state.image_lock:
                bottom_img = self.shared_state.latest_bottom_image
            
            if bottom_img is None:
                print("Warning: No bottom image available for mask processing")
                sleep_with_shutdown(1,)
            else:
                capture_flag = True
        cv2.imwrite("latest_bottom_image.png", bottom_img) # for debugging
        # Process image to extract mask
        try:
            mask, num_clumps, stats, centroids = process_image(bottom_img, self.reference_image)
            self.latest_mask = mask
            self.shared_state.latest_mask = mask
            self.shared_state.latest_mask_saved = False
            self.latest_num_clumps = num_clumps
            self.latest_stats = stats
            self.latest_centroids = centroids
            self.interaction_since_last_mask = False
            mask_area = cv2.countNonZero(mask) if mask is not None else 0
            print(f"Mask processed: {num_clumps} clumps, area: {mask_area}")
            
            if mask_area == 0:
                return False
            return True
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            return False


    def calibrate_walls(self):
        # Sample wall and tool pose
        orders = []
        for wall in self.walls:
            for t in [-1, 1]:
                pose = sample_tool_pose(wall, tool_width=0.02, safety_margin=0.02, t=t)
                x = pose['x']
                y = pose['y']
                orientation = pose['angle']
                orders.append((OrderType.ROTATE, [orientation]))
                orders.append((OrderType.MOVE_XY, [x, y]))
                print(orders[-1])
        print(f"Calibrating walls, moving to {len(orders)} positions")
        self.queue_orders(orders)

    def perform_task(self):        
        orders = []
        logging.info("start new task session")
        logging.info(self.robot_state)
        if self.latest_mask is None:
            self.update_mask_and_process()

        self.queue_orders([(OrderType.MOVE_Z, [self.clearance_height])])
        time.sleep(1)
        self.update_robot_state()
        logging.info(self.robot_state)
        try:
            if self.robot_state['z_norm']>self.grasp_height+0.02:
                # didn't need sweep, now need to find empty place to put tool
                tool_placement = find_tool_placements(self.latest_mask,self.image_space_tool_dimensions,[0,30,45,60,90,120,135,150],((self.image_space_manip_boundary[0], self.image_space_manip_boundary[1]), (self.image_space_manip_boundary[0], self.image_space_manip_boundary[1]))) 
                logging.info("Tool placement: %s", tool_placement)
                # print(tool_placement)
                if tool_placement is None:
                    self.sweep_wall(random.sample(self.walls, 1))
                else: # tool is already in position to start pushing
                    pos_pixel = tool_placement.get("pos_px")
                    pos_world = self.pix2robtrans.pix_to_robot(*pos_pixel)
                    orders = [(OrderType.MOVE_XY, pos_world),
                              (OrderType.ROTATE, [tool_placement.get("angle")]),
                            (OrderType.MOVE_Z, [self.grasp_height])]
                    self.queue_orders(orders)
                    self.update_robot_state()
        except Exception as e:
            print(f"exception in tool placement at task: {e}")
            self.shutdown_event.set()
            return

        for i in range(self.N_pushes):
            if self.shutdown_event.is_set():
                break

            # Sample random position
            x = np.random.uniform(self.manip_x[0], self.manip_x[1])
            y = np.random.uniform(self.manip_y[0], self.manip_y[1])
            orientation = np.random.uniform(0, 180)

            
            # For debugging, use fixed grid positions
            # x = i / self.N_pushes * (self.x_max - self.x_min) + self.x_min
            # y = 0.65  # Fixed Y for simplicity

            # Move XY at Z=grasp height
            orders.append((OrderType.MOVE_XY, [x, y]))
            orders.append((OrderType.ROTATE, [orientation]))
            # Optional short wait at push position

        self.queue_orders(orders, time_between_orders=1)
        self.update_robot_state()
        self.interaction_since_last_mask = True

    def check_tool_grip(self):
        with self.shared_state.image_lock:
            top_img = self.shared_state.latest_top_image
        # if top_img is None:
        #     top_img, _ = self.robot.get_image_top() # should hopefully only be used in first iteration
        roi_cfg = self.config.get('tool_detection',{}).get('region_of_interest',{})
        x_range = roi_cfg.get('x',[0.4,0.7])
        y_range = roi_cfg.get('y',[0.1,0.8])
        h, w, _ = top_img.shape
        x_min = int(x_range[0]*w)
        x_max = int(x_range[1]*w)
        y_min = int(y_range[0]*h)
        y_max = int(y_range[1]*h)
        
        roi_img = top_img[y_min:y_max, x_min:x_max]
        # np_img = np.array(top_img)
        
        # cv2.rectangle(np_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        # cv2.imshow("Tool Detection", np_img)
        # cv2.waitKey(1)
        # Define color range for tool detection (example: white tool in BGR)
        lower_bgr = (160, 160, 120)
        upper_bgr = (190, 210, 190)
        grip_quality = analyze_tool_grip(roi_img, (lower_bgr, upper_bgr))
        print(f"Tool grip quality: {grip_quality:.2f}")
        return grip_quality
    
    def perform_grab_tool(self, check_tool_grasp=False) -> float:
        self.robot.move_z(1.0)
        time.sleep(1.0)
        self.robot.gripper_open()
        time.sleep(1.0)
        self.robot.rotate(0)
        time.sleep(1.0)
        print("Moving to tool position...", flush=True)
        self.robot.move_xy(0.03, 0.49)
        time.sleep(1)
        self.robot.move_z(0.27)
        time.sleep(1)
        self.robot.gripper_close()

        try: 
            self.robot.move_z(1.0)
            time.sleep(1.0)
            self.robot.move_xy(0.5,0.41)
            time.sleep(1.0)
            self.robot.rotate(90)
            time.sleep(1.0)
            if check_tool_grasp:
                top_img, _ = self.robot.get_image_top()
                roi_cfg = {'x': [0.46, 0.58], 'y': [0.54, 0.63]}
                x_range = roi_cfg.get('x',[0.4,0.7])
                y_range = roi_cfg.get('y',[0.1,0.8])
                h, w, _ = top_img.shape
                x_min = int(x_range[0]*w)
                x_max = int(x_range[1]*w)
                y_min = int(y_range[0]*h)
                y_max = int(y_range[1]*h)
                roi_img = top_img[y_min:y_max, x_min:x_max]
                # Define color range for tool detection (example: white tool in BGR)
                lower_bgr = (160, 160, 120)
                upper_bgr = (190, 210, 190)
                grip_quality = self.check_tool_grip()
            else:
                grip_quality = 0.0
            return grip_quality
        except Exception as e:
            print(f"Error during tool grasping: {e}")
        return 0.0

    
    def startup(self):
        # check visually if tool is in hand:
        try: 
            orders = [
                (OrderType.MOVE_Z, [1.0]),
                (OrderType.MOVE_XY, [0.5, 0.41]),
                (OrderType.ROTATE, [90])
            ]            
            self.queue_orders(orders)
            time.sleep(1.5)
            self.update_robot_state()
            if self.config.get('tool_detection',{}).get('enabled',False):
                grip_quality = self.check_tool_grip()
            
            if grip_quality < self.tool_detection_threshold:
                print("Tool not detected properly in hand. Please adjust tool and restart.")
                self.robot.gripper_open()
                while grip_quality < self.tool_detection_threshold and not self.shutdown_event.is_set():
                    print(f"grip quality is: {grip_quality:.2f}, below threshold {self.tool_detection_threshold}, attempting to grab tool...")
                    sleep_with_shutdown(60,self.shutdown_event)
                    self.robot.gripper_close() # close and open gripper to warn user grasp attempt is coming
                    sleep_with_shutdown(2,self.shutdown_event)
                    self.robot.gripper_open()
                    sleep_with_shutdown(10,self.shutdown_event)
                    grip_quality = self.perform_grab_tool(check_tool_grasp=True)
                if self.shutdown_event.is_set():
                    return
        except Exception as e:
            print(f"Error during tool detection: {e}")
            self.shutdown_event.set()
            return
        
        # analyze initial bottom image to check for granular setup, if reset is needed, where to place tool, etc.
        if self.reference_image is not None:
            try:
                if not self.update_mask_and_process():
                    self.shutdown_event.set()
                    return
                num_clumps = self.latest_num_clumps
                if num_clumps is None or num_clumps == 0:
                    print("Granular material not detected in workspace. Please set up and restart.")
                    self.shutdown_event.set()
                    return
                else:
                    print("Moving to check reset necessity.")
                    try:
                        reset_needed = check_reset_needed(self.latest_mask)
                    except Exception as e:
                        print("error in checking reset necessity")
                        self.shutdown_event.set()
                        return
                    if reset_needed:
                        self.state = RobotActivity.RESETTING
                    else:
                        # self.sweep_wall(random.sample(self.walls, 1))
                        self.state = RobotActivity.ACTIVE                   
            except Exception as e:
                print(f"Error during granular detection: {e}, {repr(e)}, type: {type(e)}")
                self.shutdown_event.set()
                return
        else:
            print("No reference image for granule segmentation, please provide and restart")
            self.shutdown_event.set()
            return


    def sweep_wall(self, wall):
        """
        Sweep wall with proper reset handling using free space detection.
        Uses details from check_wall_reset_needed to plan motion.
        """
        
        reset_needed, details = check_wall_reset_needed(
            self.latest_mask, wall,
            self.image_space_tool_dimensions,
            self.min_granule_size
        )
        print(f"Reset needed for wall {wall.label}: {reset_needed}")
        print(f"Details: {details}")
        if not reset_needed:
            return False
        
        use_fallback = details.get("use_fallback", False)
        if use_fallback:
            # No sufficient free space exists - perform fallback reset
            print(f"Using fallback reset for wall {wall.label}")
            already_at_wall = False
            # Perform once the small pre-sweep at sweep_height
            for t in np.linspace(wall.t_min, wall.t_max, num=3):
                self.sweep(wall, t=t, already_at_wall=already_at_wall)
                already_at_wall = True
            return True

        else:
            # Use free space information for planning
            pos_px = details.get("pos_px")
            pos_optimal = self.pix2robtrans.pix_to_robot(*pos_px)
            
            print(f"Sweeping wall {wall.label} using free space at pos_px={pos_px}")
            pos_sweep = get_pos_sweep_from_optimal(pos_optimal, wall, margin=0.01)
            # Place tool in empty space, lower to grasp_height, come closer to wall
            orders = [
                (OrderType.MOVE_Z, [self.clearance_height]),
                (OrderType.MOVE_XY, [pos_optimal[0], pos_optimal[1]]),
                (OrderType.ROTATE, [wall.angle]),
                (OrderType.MOVE_Z, [self.grasp_height]),
                (OrderType.MOVE_XY, [pos_sweep[0], pos_sweep[1]]),
            ]
            self.queue_orders(orders, time_between_orders=1.5)
            self.update_robot_state()
            # Now sweep using existing logic with already_at_wall=True
            for t in np.linspace(wall.t_min, wall.t_max, num=3):
                self.sweep(wall, t=t, already_at_wall=True)
            return True


    def sweep(self, wall: Wall, t=None, already_at_wall=False):
        """
        Execute sweep motion along wall.
        If already_at_wall=True, skip the initial positioning logic.
        """
        sweep_grasp_pos_dict = sample_tool_pose(wall, tool_width=0.02, safety_margin=0.02, t=t)
        sweep_grasp_pos = [sweep_grasp_pos_dict['x'], sweep_grasp_pos_dict['y']]
        sweep_dir = sweep_grasp_pos_dict['perpendicular_dir']
        rotation_angle = sweep_grasp_pos_dict['angle']
        step = self.reset_step_size * (1 + 0.5 * np.random.rand())
        
        order = []
        
        if not already_at_wall:
            # Move to wall position first, then perform small sweep at higher clearance for safety
            order.extend([
                (OrderType.MOVE_Z, [self.clearance_height]),
                (OrderType.MOVE_XY, [sweep_grasp_pos[0], sweep_grasp_pos[1]]),
                (OrderType.ROTATE, [rotation_angle]),
                (OrderType.MOVE_Z, [self.sweep_height+0.02]),
                (OrderType.MOVE_XY, [sweep_grasp_pos[0] + sweep_dir[0] * 0.02, sweep_grasp_pos[1] + sweep_dir[1] * 0.02]),
                (OrderType.MOVE_Z, [self.clearance_height]),
                (OrderType.MOVE_XY, [sweep_grasp_pos[0], sweep_grasp_pos[1]]),
                (OrderType.MOVE_Z, [self.sweep_height-0.01]),
                (OrderType.MOVE_XY, [sweep_grasp_pos[0] + sweep_dir[0] * 0.02, sweep_grasp_pos[1] + sweep_dir[1] * 0.02]),
                (OrderType.MOVE_Z, [self.clearance_height]),
                (OrderType.MOVE_XY, [sweep_grasp_pos[0], sweep_grasp_pos[1]]),
                (OrderType.MOVE_Z, [self.grasp_height]),
            ])
        
        # Perform the actual sweep at grasp height
        order.extend([
            (OrderType.MOVE_XY, [sweep_grasp_pos[0], sweep_grasp_pos[1]]),
            (OrderType.MOVE_XY, [sweep_grasp_pos[0] + sweep_dir[0] * step, sweep_grasp_pos[1] + sweep_dir[1] * step]),
            (OrderType.MOVE_XY, [sweep_grasp_pos[0], sweep_grasp_pos[1]])
        ])
        self.interaction_since_last_mask = True
        
        self.queue_orders(order, time_between_orders=0.8)


    def reset_task(self):
        """
        Reset task with reorganized logic.
        Calls update_mask_and_process before checking walls.
        """
        # Update mask and process before checking walls (includes moving robot out of way) - currently commented out since we already do this in startup
        # while not self.update_mask_and_process():
        #     print("Retrying mask update...")
        #     time.sleep(0.2)
        
        # if not check_reset_needed(self.latest_mask):
        #     return
        
        print("Resetting")
        try:
            # Sample direction to sweep from
            for wall in random.sample(self.walls, len(self.walls)):
                print(f"Checking wall {wall.label} for reset")
                # print(wall.origin, wall.tangent, wall.normal)
                
                swept = self.sweep_wall(wall)
                if swept:
                    print(f"Swept wall {wall.label} for reset")
                    self.interaction_since_last_mask = True
                    self.update_mask_and_process()
        except Exception as e:
            print(f"Error during reset task: {e}, {repr(e)}, type: {type(e)}")
            self.shutdown_event.set()
            return
        self.state = RobotActivity.ACTIVE
        
        


    def run_grasping(self):
        print("Starting Random Push Grasper")
        while not self.shutdown_event.is_set():
            if self.state == RobotActivity.STARTUP:
                self.startup()
                # self.state = RobotActivity.ACTIVE # for testing without reset

            if self.state == RobotActivity.ACTIVE:
                try:
                    self.perform_task()
                except Exception as e:
                    print(f"Error during task performance: {e}, {repr(e)}, type: {type(e)}")
                    self.failed = True
                    self.shutdown_event.set()
                    raise

                # After performing task, go to startup again to check if reset is needed before next task
                
                self.state = RobotActivity.STARTUP

            if self.state == RobotActivity.RESETTING:
                # Optional: recovery/reset logic
                self.reset_task() 

                # sleep_with_shutdown(self.task_time_margin, self.shutdown_event)
                self.state = RobotActivity.ACTIVE
        self.state = RobotActivity.FINISHED        
        return