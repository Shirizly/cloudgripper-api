# grasper.py
from abc import ABC, abstractmethod
import os
import sys
import time
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import threading
from dotenv import load_dotenv

# Ensure project root is in the system path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


from client.cloudgripper_client import GripperRobot
import library.utils as utils
from action_tracker import ActionType, ActionPhase, ActionTracker

load_dotenv()


class RobotActivity(Enum):
    ACTIVE = 1
    RESETTING = 2
    FINISHED = 3
    STARTUP = 4


def sleep_with_shutdown(duration: float, shutdown_event: threading.Event | None = None):
    """Sleep in small increments, checking for shutdown."""
    end_time = time.time() + duration
    while time.time() < end_time:
        if shutdown_event is not None and shutdown_event.is_set():
            break
        time.sleep(0.05)


class AutograsperBase(ABC):
    def __init__(
        self, config, output_dir: str = "", shutdown_event: threading.Event | None = None
    ):
        if shutdown_event is None:
            raise ValueError("shutdown_event must be provided")
        self.shutdown_event = shutdown_event
        self.shared_state = None  # to be set in coordinator

        self.token = os.getenv("CLOUDGRIPPER_TOKEN")
        if not self.token:
            raise ValueError("CLOUDGRIPPER_TOKEN environment variable not set")

        self.output_dir = output_dir
        self.start_time = time.time()
        self.failed = False

        self.state = RobotActivity.STARTUP
        self.start_event = threading.Event()
        self.state_recorded_event = threading.Event()

        self.request_state_record = False
        self.task_time_margin = 2
        self.robot_state = None

        try:
            camera_config = config["camera"]
            experiment_config = config["experiment"]
            self.record_only_after_action = bool(
                camera_config["record_only_after_action"]
            )
            self.robot_idx = experiment_config["robot_idx"]
            self.time_between_orders = experiment_config["time_between_orders"]
        except KeyError as e:
            raise ValueError(
                f"Missing configuration key in AutograsperBase: {e}"
            ) from e
        except TypeError as e:
            raise ValueError(
                f"Invalid configuration format in AutograsperBase: {e}"
            ) from e

        self.robot = self.initialize_robot()

    def connect_shared_state(self, shared_state):
        self.shared_state = shared_state

    # ========== Action Tracking Methods ==========
    # These methods provide an easy interface for marking robot actions
    # that will be recorded along with frames for dataset creation.

    def start_action(
        self,
        action_type: ActionType,
        phase: ActionPhase,
        frame_index: int,
        is_planar_2d: bool = False,
        action_details: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Mark the start of a robot action.
        
        Args:
            action_type: Type of action (MOVE_XY, ROTATE, GRIPPER_CLOSE, etc.)
            phase: Whether this is TASK, RESET, or STARTUP
            frame_index: Frame number where action begins
            is_planar_2d: True if this is a 2D planar motion at grasp_height
            action_details: Dict with action-specific details
            description: Optional human-readable description
            extra_metadata: Additional metadata dict
            
        Returns:
            action_id that can be used to end the action
        """
        if self.shared_state is None:
            return -1
        
        start_robot_state = self.robot_state.copy() if self.robot_state else None
        
        return self.shared_state.action_tracker.start_action(
            action_type=action_type,
            phase=phase,
            start_frame=frame_index,
            start_robot_state=start_robot_state,
            is_planar_2d=is_planar_2d,
            action_details=action_details,
            description=description,
            extra_metadata=extra_metadata,
        )

    def end_action(
        self,
        action_id: int,
        frame_index: int,
    ) -> None:
        """
        Mark the end of a robot action.
        
        Args:
            action_id: ID returned from start_action()
            frame_index: Frame number where action ends
        """
        if self.shared_state is None:
            return
        
        end_robot_state = self.robot_state.copy() if self.robot_state else None
        self.shared_state.action_tracker.end_action(
            action_id=action_id,
            end_frame=frame_index,
            end_robot_state=end_robot_state,
        )

    def get_current_frame_index(self) -> Optional[int]:
        """
        Get the current frame index being recorded.
        
        Returns:
            Frame index if available, None otherwise.
            
        Note: There may be a small delay (1-2 frames) due to threading,
        but this is acceptable for marking action boundaries.
        """
        if self.shared_state is None:
            return None
        
        try:
            with self.shared_state.frame_index_lock:
                return self.shared_state.frame_index
        except (AttributeError, ValueError):
            # Fallback if frame_index not available
            return None
    
    def action_phase_from_state(self, state: RobotActivity) -> ActionPhase:
        if state == RobotActivity.ACTIVE:
            return ActionPhase.TASK
        elif state == RobotActivity.RESETTING:
            return ActionPhase.RESET
        elif state == RobotActivity.STARTUP:
            return ActionPhase.STARTUP
        else:
            return ActionPhase.OTHER
        
    def action_type_from_order(self, order: Tuple) -> ActionType:
        order_type = order[0]
        if order_type == utils.OrderType.MOVE_XY:
            return ActionType.MOVE_XY
        elif order_type == utils.OrderType.MOVE_Z:
            return ActionType.MOVE_Z
        elif order_type == utils.OrderType.ROTATE:
            return ActionType.ROTATE
        elif order_type in (utils.OrderType.GRIPPER_OPEN, utils.OrderType.GRIPPER_CLOSE):
            return ActionType.GRIPPER_CLOSE
        else:
            raise ValueError(f"Unknown order type: {order_type}")
        
    def action_details_from_order(self, order: Tuple) -> Dict[str, Any]:
        order_type = order[0]
        if order_type == utils.OrderType.MOVE_XY:
            return {"x": order[1][0], "y": order[1][1]}
        elif order_type == utils.OrderType.MOVE_Z:
            return {"z": order[1][0]}
        elif order_type == utils.OrderType.ROTATE:
            return {"angle": order[1][0]}
        elif order_type in (utils.OrderType.GRIPPER_OPEN, utils.OrderType.GRIPPER_CLOSE):
            return {"position": order[1][0]}
        else:
            raise ValueError(f"Unknown order type: {order_type}")

    # ========== End Action Tracking Methods ==========


    def initialize_robot(self) -> GripperRobot:
        try:
            assert self.token is not None, "CLOUDGRIPPER_TOKEN environment variable must be set"
            return GripperRobot(self.robot_idx, self.token)
        except Exception as e:
            raise ValueError("Invalid robot ID or token: ", e) from e

    def record_current_state(self):
        """Request a state record and wait until processed or shutdown."""
        self.request_state_record = True
        self.state_recorded_event.clear()
        while not self.shutdown_event.is_set():
            if self.state_recorded_event.wait(timeout=0.1):
                return

    def wait_for_start_signal(self):
        """Wait for the start event, checking periodically for shutdown."""
        while not self.shutdown_event.is_set():
            if self.start_event.wait(timeout=0.05):
                self.start_event.clear()  # Clear for next cycle.
                return

    def run_grasping(self):
        """
        A simple state-machine loop:
          - STARTUP: Run startup logic, then move to ACTIVE.
          - ACTIVE: Wait for start signal, perform task.
          - RESETTING: After task, sleep and either recover (if failed) or reset.
        """
        while self.state != RobotActivity.FINISHED and not self.shutdown_event.is_set():
            if self.state == RobotActivity.STARTUP:
                self.startup()
                self.state = RobotActivity.ACTIVE
            elif self.state == RobotActivity.ACTIVE:
                self.wait_for_start_signal()
                try:
                    self.perform_task()
                except Exception as e:
                    print(f"Unexpected error during perform_task: {e}")
                    self.failed = True
                    self.shutdown_event.set()
                    raise Exception(e)
                if self.shutdown_event.is_set() or self.state == RobotActivity.FINISHED:
                    break
                self.state = RobotActivity.RESETTING
            elif self.state == RobotActivity.RESETTING:
                sleep_with_shutdown(self.task_time_margin, self.shutdown_event)
                if self.failed:
                    print("Experiment failed, recovering")
                    self.recover_after_fail()
                    self.failed = False
                else:
                    self.reset_task()
                self.state = RobotActivity.STARTUP
            else:
                break

    def recover_after_fail(self):
        """Override to implement recovery logic after failure."""
        pass

    @abstractmethod
    def perform_task(self):
        """
        Override to perform robot actions.
        Default implementation prints a message periodically.
        """
        while not self.shutdown_event.is_set():
            print(
                "GRASPER: No task defined. Override perform_task() to perform robot actions."
            )
            sleep_with_shutdown(0.5, self.shutdown_event)
        print("GRASPER: Exiting perform_task() due to shutdown signal.")

    def reset_task(self):
        """Override to implement logic for resetting between tasks."""
        pass

    def startup(self):
        """Override to implement initialization logic before a task."""
        pass

    def get_state(self):
        "Update state for robot"

        self.robot_state = self.robot.get_state()

        return self.robot_state

    def execute_order(self, order, output_dir, reverse_xy):
        local_order = (order[0], order[1].copy())
        if order[0] == utils.OrderType.ROTATE and hasattr(self, 'rotation_bias'): # apply rotation bias if defined
            local_order[1][0] = local_order[1][0] + self.rotation_bias
        utils.execute_order(self.robot, local_order, output_dir, reverse_xy)



    def queue_orders(
        self,
        order_list: List[Tuple],
        time_between_orders: None | float = None,
        output_dir: str = "",
        reverse_xy: bool = False,
        record=True,
    ):
        """
        Queue a list of orders for the robot to execute sequentially and save state after each order.
        """
        if time_between_orders is None:
            time_between_orders = self.time_between_orders
        assert time_between_orders is not None, "time_between_orders must be provided either as an argument or in the config"

        for order in order_list:
            # print(f"Executing order: {order}")
            if self.shutdown_event.is_set():
                break
            frame_start = self.get_current_frame_index()
            action_type = self.action_type_from_order(order)
            action_id = -1
            if action_type is not ActionType.GRIPPER_CLOSE:
                action_id = self.start_action(
                    action_type=action_type,
                    phase=self.action_phase_from_state(self.state),
                    frame_index = frame_start,
                    is_planar_2d = action_type == ActionType.MOVE_XY or action_type == ActionType.ROTATE,
                    action_details=self.action_details_from_order(order),
                )
            self.execute_order(order, output_dir, reverse_xy)
            sleep_with_shutdown(time_between_orders, self.shutdown_event)
            if action_id != -1 and action_type is not ActionType.GRIPPER_CLOSE:
                self.end_action(action_id, self.get_current_frame_index())
            if (
                record
                and self.record_only_after_action
                and (self.state in (RobotActivity.ACTIVE, RobotActivity.RESETTING))
            ):
                self.record_current_state()

    # # CG1Specific
    # def manual_control(self, step_size=0.1, state=None, time_between_orders: None | float = None):
    #     """
    #     Manually control the robot using keyboard inputs.
    #     """
    #     from pynput import keyboard

    #     if self.robot_state is None:
    #         self.robot_state, _ = self.get_state()
    #     if time_between_orders is None:
    #         time_between_orders = self.time_between_orders

    #     current_x = self.robot_state["x_norm"]
    #     current_y = self.robot_state["y_norm"]
    #     current_z = self.robot_state["z_norm"]
    #     current_rotation = self.robot_state["rotation"]
    #     current_angle = self.robot_state["claw_norm"]

    #     def on_press(key):
    #         nonlocal current_x, current_y, current_z, current_rotation, current_angle
    #         try:
    #             # == XY axis ==
    #             if key.char == "w":
    #                 current_y += step_size
    #                 current_y = min(max(current_y, 0), 1)
    #                 self.queue_orders(
    #                     [(OrderType.MOVE_XY, [current_x, current_y])],
    #                     time_between_orders,
    #                 )
    #             elif key.char == "a":
    #                 current_x -= step_size
    #                 current_x = min(max(current_x, 0), 1)
    #                 self.queue_orders(
    #                     [(OrderType.MOVE_XY, [current_x, current_y])],
    #                     time_between_orders,
    #                 )
    #             elif key.char == "s":
    #                 current_y -= step_size
    #                 current_y = min(max(current_y, 0), 1)
    #                 self.queue_orders(
    #                     [(OrderType.MOVE_XY, [current_x, current_y])],
    #                     time_between_orders,
    #                 )
    #             elif key.char == "d":
    #                 current_x += step_size
    #                 current_x = min(max(current_x, 0), 1)
    #                 self.queue_orders(
    #                     [(OrderType.MOVE_XY, [current_x, current_y])],
    #                     time_between_orders,
    #                 )

    #             # == Z axis ==
    #             elif key.char == "r":
    #                 current_z += step_size
    #                 current_z = min(max(current_z, 0), 1)
    #                 print(current_z)
    #                 self.queue_orders(
    #                     [(OrderType.MOVE_Z, [current_z])], time_between_orders
    #                 )
    #             elif key.char == "f":
    #                 current_z -= step_size
    #                 current_z = min(max(current_z, 0), 1)
    #                 print(current_z)
    #                 self.queue_orders(
    #                     [(OrderType.MOVE_Z, [current_z])], time_between_orders
    #                 )

    #             # == Gripper open ==
    #             elif key.char == "i":
    #                 current_angle += step_size / 100
    #                 current_angle = min(current_angle, 1)
    #                 print(current_angle)
    #                 self.queue_orders(
    #                     [(OrderType.GRIPPER_CLOSE, [current_angle])],
    #                     time_between_orders,
    #                 )
    #             # == Gripper open small steps==
    #             elif key.char == "o":
    #                 current_angle += step_size / 200
    #                 current_angle = min(current_angle, 1)
    #                 print(current_angle)
    #                 self.queue_orders(
    #                     [(OrderType.GRIPPER_CLOSE, [current_angle])],
    #                     time_between_orders,
    #                 )
    #             # == Gripper close ==
    #             elif key.char == "k":
    #                 current_angle -= step_size / 100
    #                 current_angle = max(current_angle, 0.2)
    #                 print(current_angle)
    #                 self.queue_orders(
    #                     [(OrderType.GRIPPER_CLOSE, [current_angle])],
    #                     time_between_orders,
    #                 )
    #             # == Gripper close small steps==
    #             elif key.char == "l":
    #                 current_angle -= step_size / 200
    #                 current_angle = max(current_angle, 0.2)
    #                 print(current_angle)
    #                 self.queue_orders(
    #                     [(OrderType.GRIPPER_CLOSE, [current_angle])],
    #                     time_between_orders,
    #                 )

    #             # == Rotate ==
    #             elif key.char == "z":
    #                 current_rotation -= int(step_size * 100)
    #                 current_rotation = np.clip(current_rotation, 0, 360)
    #                 print(current_rotation)
    #                 self.queue_orders(
    #                     [(OrderType.ROTATE, [current_rotation])], time_between_orders
    #                 )
    #             elif key.char == "x":
    #                 print(current_rotation)
    #                 current_rotation += int(step_size * 100)
    #                 current_rotation = np.clip(current_rotation, 0, 360)
    #                 self.queue_orders(
    #                     [(OrderType.ROTATE, [current_rotation])], time_between_orders
    #                 )

    #             # == Quit ==
    #             elif key.char == "q":
    #                 return False

    #         except Exception as e:
    #             print(e)
    #             print(
    #                 "Make sure that the runtime has access to an X server. If running in a container on Wayland, you might need to perform `xhost local:root` in the host terminal."
    #             )

    #     def on_release(key):
    #         if key == keyboard.Key.esc:
    #             # Stop listener
    #             return False

    #     with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    #         listener.join()
