"""
Action Tracking Usage Guide

This guide demonstrates how to use the new action tracking system integrated
with the recording pipeline to capture robot movements and create annotated datasets.

Key Features:
- Track robot actions (movements, grasps, etc.) alongside frame capture
- Associate actions with specific frame ranges
- Mark actions as task vs. reset phases
- Flag planar 2D motions for planar pushing datasets
- Automatic action metadata saving in JSON format
"""

from action_tracker import ActionType, ActionPhase


# ============================================================================
# EXAMPLE 1: Basic Action Tracking in perform_task()
# ============================================================================

class ExampleGrasper(AutograsperBase):
    """Example showing action tracking in perform_task()."""
    
    def perform_task(self):
        """
        Perform random pushes while tracking each movement as an action.
        """
        N_pushes = 10
        
        # Get current frame index when task starts
        task_start_frame = self.get_current_frame_index()
        
        for push_idx in range(N_pushes):
            # Sample random position
            x = np.random.uniform(0.3, 0.7)
            y = np.random.uniform(0.3, 0.7)
            orientation = np.random.uniform(0, 180)
            
            # Get frame index at action start
            action_start_frame = self.get_current_frame_index()
            
            # START TRACKING: Mark the movement as a planar 2D action
            action_id = self.start_action(
                action_type=ActionType.MOVE_XY,
                phase=ActionPhase.TASK,
                frame_index=action_start_frame,
                is_planar_2d=True,  # This is a planar push
                action_details={
                    "x": float(x),
                    "y": float(y),
                    "push_id": push_idx,
                },
                description=f"Push {push_idx}: moving to ({x:.3f}, {y:.3f})",
            )
            
            # Execute the movement
            orders = [
                (OrderType.MOVE_XY, [x, y]),
                (OrderType.ROTATE, [orientation]),
            ]
            self.queue_orders(orders, time_between_orders=1.2)
            self.update_robot_state()
            
            # END TRACKING: Mark when the movement completes
            action_end_frame = self.get_current_frame_index()
            self.end_action(action_id, action_end_frame)
        
        # Close gripper at end
        self.robot.gripper_close()


# ============================================================================
# EXAMPLE 2: Tracking Movements in reset_task()
# ============================================================================

class ExampleResetGrasper(AutograsperBase):
    """Example showing action tracking in reset_task()."""
    
    def reset_task(self):
        """
        Sweep walls while tracking each sweep as a reset action.
        """
        walls = self.walls  # Assume walls are defined
        
        for wall in walls:
            action_start_frame = self.get_current_frame_index()
            
            # START TRACKING: Mark reset sweep action
            action_id = self.start_action(
                action_type=ActionType.SWEEP,
                phase=ActionPhase.RESET,  # This is a reset action
                frame_index=action_start_frame,
                is_planar_2d=False,  # Sweeps may include vertical motion
                action_details={
                    "wall_label": wall.label,
                    "sweep_direction": "horizontal",
                },
                description=f"Sweeping wall: {wall.label}",
            )
            
            # Execute the sweep
            self.sweep_wall(wall)
            
            # END TRACKING
            action_end_frame = self.get_current_frame_index()
            self.end_action(action_id, action_end_frame)


# ============================================================================
# EXAMPLE 3: Tracking Gripper Actions
# ============================================================================

class ExampleGripperTracking(AutograsperBase):
    """Example showing tracking of gripper actions."""
    
    def perform_task(self):
        """Track gripper open/close as actions."""
        
        # Track gripper open
        frame_idx = self.get_current_frame_index()
        action_id = self.start_action(
            action_type=ActionType.GRIPPER_OPEN,
            phase=ActionPhase.TASK,
            frame_index=frame_idx,
            action_details={"position": 0.0},
            description="Opening gripper",
        )
        self.robot.gripper_open()
        time.sleep(1.0)
        self.end_action(action_id, self.get_current_frame_index())
        
        # ... perform task ...
        
        # Track gripper close
        frame_idx = self.get_current_frame_index()
        action_id = self.start_action(
            action_type=ActionType.GRIPPER_CLOSE,
            phase=ActionPhase.TASK,
            frame_index=frame_idx,
            action_details={"position": 1.0},
            description="Closing gripper",
        )
        self.robot.gripper_close()
        time.sleep(1.0)
        self.end_action(action_id, self.get_current_frame_index())


# ============================================================================
# EXAMPLE 4: Querying Actions for Dataset Creation
# ============================================================================

"""
After recording is complete, you can access action data in multiple ways:

From the recorded JSON files:
- states.json: Contains frame data with embedded action info
- actions.json: Summary of all actions with frame ranges

Example states.json entry:
{
    "state": {...robot state...},
    "time": 1234567890.123,
    "frame_index": 42,
    "action": {
        "action_id": 5,
        "action_type": "move_xy",
        "phase": "task",
        "start_frame": 40,
        "end_frame": 50,
        "is_planar_2d": true,
        "action_details": {
            "x": 0.45,
            "y": 0.55,
            "push_id": 2
        },
        "description": "Push 2: moving to (0.450, 0.550)"
    }
}

Example actions.json:
{
    "total_actions": 12,
    "actions": [
        {
            "action_id": 0,
            "action_type": "move_xy",
            "phase": "task",
            "start_frame": 10,
            "end_frame": 25,
            "is_planar_2d": true,
            ...
        },
        ...
    ]
}
"""


# ============================================================================
# EXAMPLE 5: Creating Datasets from Action Data
# ============================================================================

import json

def create_planar_pushing_dataset(states_file, output_dir):
    """
    Create a planar pushing dataset by extracting only planar 2D actions.
    
    This shows how to use the action metadata to create curated datasets
    for specific robot behaviors.
    """
    with open(states_file, 'r') as f:
        states = json.load(f)
    
    planar_actions = []
    
    for state_entry in states:
        if 'action' in state_entry:
            action = state_entry['action']
            
            # Only include planar 2D actions
            if action.get('is_planar_2d', False) and action['phase'] == 'task':
                planar_actions.append({
                    'frame_index': state_entry['frame_index'],
                    'action_id': action['action_id'],
                    'start_frame': action['start_frame'],
                    'end_frame': action['end_frame'],
                    'action_details': action['action_details'],
                    'robot_state': state_entry['state'],
                })
    
    # Save the curated dataset
    dataset_file = os.path.join(output_dir, 'planar_pushing_dataset.json')
    with open(dataset_file, 'w') as f:
        json.dump(planar_actions, f, indent=2)
    
    print(f"Created planar pushing dataset with {len(planar_actions)} entries")
    return planar_actions


def create_task_vs_reset_split(states_file, output_dir):
    """
    Create separate training splits for task and reset actions.
    """
    with open(states_file, 'r') as f:
        states = json.load(f)
    
    task_frames = []
    reset_frames = []
    
    for state_entry in states:
        if 'action' in state_entry:
            action = state_entry['action']
            if action['phase'] == 'task':
                task_frames.append(state_entry)
            elif action['phase'] == 'reset':
                reset_frames.append(state_entry)
    
    # Save splits
    with open(os.path.join(output_dir, 'task_frames.json'), 'w') as f:
        json.dump(task_frames, f, indent=2)
    
    with open(os.path.join(output_dir, 'reset_frames.json'), 'w') as f:
        json.dump(reset_frames, f, indent=2)
    
    print(f"Created splits: {len(task_frames)} task frames, {len(reset_frames)} reset frames")


# ============================================================================
# IMPLEMENTATION CHECKLIST FOR YOUR GRASPER
# ============================================================================

"""
To integrate action tracking into your RandomPushGrasper:

1. In perform_task():
   [ ] Add self.start_action() call before each movement
   [ ] Set is_planar_2d=True for MOVE_XY actions at grasp_height
   [ ] Set phase=ActionPhase.TASK
   [ ] Call self.end_action() after movement completes

2. In reset_task():
   [ ] Add self.start_action() call for each sweep
   [ ] Set phase=ActionPhase.RESET
   [ ] Include wall information in action_details
   [ ] Call self.end_action() after sweep completes

3. In startup():
   [ ] Optional: track preparation movements with ActionPhase.STARTUP

4. Testing:
   [ ] Verify states.json includes "action" field for tracking actions
   [ ] Verify actions.json is created at end of recording
   [ ] Check that frame_indices match expected ranges
   [ ] Verify is_planar_2d flags are set correctly

5. Dataset Creation:
   [ ] Write custom scripts in create_planar_pushing_dataset() style
   [ ] Extract specific action types for your use case
   [ ] Validate that frame ranges are consistent
"""
