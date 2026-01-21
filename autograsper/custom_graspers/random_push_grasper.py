# custom_graspers/random_push_grasper.py
from grasper import AutograsperBase, RobotActivity, sleep_with_shutdown
import numpy as np
import time
from library.utils import OrderType
from dataclasses import dataclass
##### Helper functions for wall definition and sampling #####

@dataclass(frozen=True)
class Wall:
    label: str             # wall label (e.g., 'top', 'right', etc.)
    origin: np.ndarray      # reference point on wall (center of wall)
    tangent: np.ndarray     # unit vector along wall
    normal: np.ndarray      # unit vector perpendicular to wall
    t_min: float            # minimum slide parameter
    t_max: float            # maximum slide parameter
    angle: float            # tool orientation angle (radians)

def build_fence_walls(
    fence_center,
    fence_size,
    tool_length,
    tool_width,
    safety_margin
):
    fx, fy = fence_center
    h = [fence_size / 2 for fence_size in fence_size]
    hx,hy = h

    half_tool_len = tool_length / 2
    half_tool_wid = tool_width / 2

    walls = []

    # Define walls in CCW order
    wall_defs = [
        # top
        ('top', np.array([fx, fy + hy]), np.array([1, 0]), np.array([0, -1])),
        # right
        ('right', np.array([fx + hx, fy]), np.array([0, -1]), np.array([-1, 0])),
        # bottom
        ('bottom', np.array([fx, fy - hy]), np.array([-1, 0]), np.array([0, 1])),
        # left
        ('left', np.array([fx - hx, fy]), np.array([0, 1]), np.array([1, 0])),
    ]

    for label, origin, tangent, normal in wall_defs:
        tangent = tangent / np.linalg.norm(tangent)
        normal = normal / np.linalg.norm(normal)

        # valid sliding range
        t_min = -abs(np.inner(h, tangent)) + half_tool_len + safety_margin
        t_max =  abs(np.inner(h, tangent)) - half_tool_len - safety_margin

        angle = np.arctan2(tangent[1], tangent[0])
        angle = (angle+np.pi/2) % np.pi
        angle = angle/np.pi*180  # convert to degrees
        angle = angle + 180 if angle < 0 else angle
        angle = 0 if angle==180 else angle
        angle = int(angle)

        walls.append(
            Wall(
                label=label,
                origin=origin,
                tangent=tangent,
                normal=normal,
                t_min=t_min,
                t_max=t_max,
                angle=angle
            )
        )

    return walls

def sample_tool_pose(wall: Wall, tool_width, safety_margin, t=None):
    # slide along wall
    if t is None:
        t = np.random.uniform(wall.t_min, wall.t_max)
    else:
        t = np.clip(t, wall.t_min, wall.t_max)

    # offset from wall (exact contact + margin)
    offset = tool_width / 2 + safety_margin

    center = (
        wall.origin
        + wall.tangent * t
        + wall.normal * offset
    )

    return {
        "x": center[0],
        "y": center[1],
        "angle": wall.angle,
        "parallel_dir": wall.tangent,
        "perpendicular_dir": wall.normal,
    }

##### Random Push Grasper Class #####

class RandomPushGrasper(AutograsperBase):
    def __init__(self, config, shutdown_event, N_pushes=10):
        super().__init__(config, shutdown_event=shutdown_event)
        self.N_pushes = N_pushes
        self.grasp_height = 0.4
        self.sweep_height = 0.57
        self.clearance_height = 0.8
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
        self.reset_step_size = 0.1
        # Hardcoded safe X,Y ranges (example, adjust to your setup)
        self.manip_x = config.get('fence', {}).get('manipulation_boundary',{}).get('x', (0.3,0.7))
        self.manip_y = config.get('fence', {}).get('manipulation_boundary',{}).get('y', (0.3,0.7))

    def warmup(self):
        # Move to safe height
        self.robot.move_z(self.clearance_height)
        time.sleep(0.5)

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
        # Lower robot initially
        # self.robot.move_z(self.grasp_height)
        time.sleep(0.5)
        orders = []
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

        self.queue_orders(orders)

    def sweep(self,wall):
        print("sweeping")
        orders = []
        for t in np.linspace(wall.t_min, wall.t_max, num=3):
            sweep_grasp_pos_dict = sample_tool_pose(wall, tool_width=0.02, safety_margin=0.02,t=t)
            sweep_grasp_pos = (sweep_grasp_pos_dict['x'],sweep_grasp_pos_dict['y'])
            sweep_dir = sweep_grasp_pos_dict['perpendicular_dir']
            rotation_angle = sweep_grasp_pos_dict['angle']
            step = self.reset_step_size*(1+0.5*np.random.rand())
            order = [
                (OrderType.MOVE_Z, [self.clearance_height]),
                (OrderType.MOVE_XY, [sweep_grasp_pos[0], sweep_grasp_pos[1]]),
                (OrderType.ROTATE, [rotation_angle]),
                (OrderType.MOVE_Z, [self.sweep_height]),
                (OrderType.MOVE_XY, [sweep_grasp_pos[0]+sweep_dir[0]*0.02, sweep_grasp_pos[1]+sweep_dir[1]*0.02]),
                (OrderType.MOVE_XY, [sweep_grasp_pos[0], sweep_grasp_pos[1]]),
                (OrderType.MOVE_Z, [self.grasp_height]),
                (OrderType.MOVE_XY, [sweep_grasp_pos[0]+sweep_dir[0]*step, sweep_grasp_pos[1]+sweep_dir[1]*step]),
            ]
            orders.extend(order)
        self.queue_orders(orders)

                  
    def reset_task(self):
        print("Resetting")
        # Move to safe height
        self.robot.move_z(1)
        time.sleep(0.5)
        # sample direction to sweep from
        for wall in self.walls:
            self.sweep(wall)

    def run_grasping(self):
        print("Starting Random Push Grasper")
        while not self.shutdown_event.is_set():
            if self.state == RobotActivity.STARTUP:
                self.startup()
                self.state = RobotActivity.ACTIVE

            if self.state == RobotActivity.ACTIVE:
                self.wait_for_start_signal()
                # try:
                
                self.perform_task()
                # except Exception as e:
                #     self.failed = True
                #     self.shutdown_event.set()
                #     raise

                # After performing task, go to RESETTING
                
                self.state = RobotActivity.RESETTING

            if self.state == RobotActivity.RESETTING:
                # Optional: recovery/reset logic
                self.reset_task()
                # sleep_with_shutdown(self.task_time_margin, self.shutdown_event)
                self.state = RobotActivity.STARTUP  # Stop here
