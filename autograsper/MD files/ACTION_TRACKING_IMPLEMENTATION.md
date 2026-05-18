"""
Action Tracking Integration for RandomPushGrasper

This module provides practical examples and helper methods for integrating
action tracking into the RandomPushGrasper class.
"""

from action_tracker import ActionType, ActionPhase
import logging

logger = logging.getLogger(__name__)


class RandomPushGrasperWithActionTracking:
    """
    Practical integration guide for RandomPushGrasper with action tracking.
    
    Shows how to add action tracking to both perform_task() and reset_task()
    while maintaining the existing functionality.
    """

    # ========================================================================
    # ENHANCED perform_task() WITH ACTION TRACKING
    # ========================================================================

    def perform_task_with_tracking(self):
        """
        Enhanced perform_task() that tracks each push as a planar 2D action.
        
        Key features:
        - Each MOVE_XY at grasp_height is marked as a planar 2D action
        - Frame indices are captured at action boundaries
        - Action details include position and push ID
        - All actions are marked as TASK phase
        """
        orders = []
        logging.info("start new task session with action tracking")
        logging.info(self.robot_state)
        
        # Prepare tool (no action tracking needed, it's preparatory)
        while not self.prepare_tool_for_active_state():
            logging.error("Failed to prepare tool for active state, retrying...")
            sleep_with_shutdown(1, self.shutdown_event)

        # Track each push
        for i in range(self.N_pushes):
            if self.shutdown_event.is_set():
                break

            # Sample random position
            x = np.random.uniform(self.manip_x[0], self.manip_x[1])
            y = np.random.uniform(self.manip_y[0], self.manip_y[1])
            orientation = np.random.uniform(0, 180)

            # Get current frame index at movement start
            # Note: This requires the recorder to be accessible or frame counter to be set
            frame_start = self.get_current_frame_index()
            
            # START ACTION: Track the planar push
            action_id = self.start_action(
                action_type=ActionType.MOVE_XY,
                phase=ActionPhase.TASK,
                frame_index=frame_start,
                is_planar_2d=True,  # This is a planar push at grasp_height
                action_details={
                    "x": float(x),
                    "y": float(y),
                    "orientation": float(orientation),
                    "push_index": int(i),
                },
                description=f"Push {i}: target position ({x:.3f}, {y:.3f})",
                extra_metadata={
                    "total_pushes": self.N_pushes,
                },
            )
            
            # Execute the movement
            orders = [
                (OrderType.MOVE_XY, [x, y]),
                (OrderType.ROTATE, [orientation]),
            ]
            self.queue_orders(orders, time_between_orders=1.2)
            self.update_robot_state()
            
            # END ACTION: Record when movement is complete
            frame_end = self.get_current_frame_index()
            self.end_action(action_id, frame_end)
        
        # Track gripper close as final action
        frame_start = self.get_current_frame_index()
        action_id = self.start_action(
            action_type=ActionType.GRIPPER_CLOSE,
            phase=ActionPhase.TASK,
            frame_index=frame_start,
            action_details={"reason": "end_of_pushes"},
            description="Closing gripper at end of push sequence",
        )
        self.robot.gripper_close()
        time.sleep(0.5)
        frame_end = self.get_current_frame_index()
        self.end_action(action_id, frame_end)
        
        self.update_robot_state()
        self.interaction_since_last_mask = True

    # ========================================================================
    # ENHANCED reset_task() WITH ACTION TRACKING
    # ========================================================================

    def reset_task_with_tracking(self):
        """
        Enhanced reset_task() that tracks each wall sweep as a reset action.
        
        Key features:
        - Each wall sweep is tracked as a SWEEP action
        - All reset actions are marked with ActionPhase.RESET
        - Wall information is captured in action_details
        - Frame ranges show exact extent of each sweep
        """
        print("Resetting with action tracking")
        try:
            # Sample walls in random order
            wall_order = random.sample(self.walls, len(self.walls))
            
            for wall in wall_order:
                print(f"Checking wall {wall.label} for reset")
                
                # Get frame index at sweep start
                frame_start = self.get_current_frame_index()
                
                # START ACTION: Track the wall sweep as a reset action
                action_id = self.start_action(
                    action_type=ActionType.SWEEP,
                    phase=ActionPhase.RESET,  # This is a reset action
                    frame_index=frame_start,
                    is_planar_2d=False,  # Sweeps may include Z motion
                    action_details={
                        "wall_label": wall.label,
                        "wall_position": {
                            "x": float(wall.x),
                            "y": float(wall.y),
                        },
                        "normal_direction": wall.normal if hasattr(wall, 'normal') else None,
                    },
                    description=f"Sweep reset from wall: {wall.label}",
                    extra_metadata={
                        "wall_index": wall_order.index(wall),
                        "total_walls": len(wall_order),
                    },
                )
                
                # Execute the sweep
                swept = self.sweep_wall(wall)
                
                # END ACTION: Record when sweep is complete
                frame_end = self.get_current_frame_index()
                self.end_action(action_id, frame_end)
                
                if swept:
                    print(f"Swept wall {wall.label} for reset")
                    self.interaction_since_last_mask = True
                    # Update mask after each sweep (usually takes some frames)
                    self.update_mask_and_process()
                    
        except Exception as e:
            print(f"Error during reset task: {e}, {repr(e)}, type: {type(e)}")
            self.shutdown_event.set()
            return
        
        self.state = RobotActivity.ACTIVE

    # ========================================================================
    # HELPER METHODS FOR PRACTICAL USE
    # ========================================================================

    def get_action_summary(self):
        """
        Get summary of all actions performed in current session.
        
        Returns:
            dict with action statistics and categorization
        """
        if self.shared_state is None:
            return {}
        
        tracker = self.shared_state.action_tracker
        all_actions = tracker.get_all_actions()
        
        summary = {
            "total_actions": len(all_actions),
            "by_type": {},
            "by_phase": {},
            "planar_2d_count": 0,
        }
        
        for action in all_actions:
            # Count by type
            action_type = action.action_type.value
            summary["by_type"][action_type] = summary["by_type"].get(action_type, 0) + 1
            
            # Count by phase
            phase = action.phase.value
            summary["by_phase"][phase] = summary["by_phase"].get(phase, 0) + 1
            
            # Count planar 2D
            if action.is_planar_2d:
                summary["planar_2d_count"] += 1
        
        return summary

    def print_action_summary(self):
        """Print a human-readable summary of all tracked actions."""
        summary = self.get_action_summary()
        
        print("\n" + "="*60)
        print("ACTION TRACKING SUMMARY")
        print("="*60)
        print(f"Total Actions: {summary['total_actions']}")
        print(f"Planar 2D Actions: {summary['planar_2d_count']}")
        
        if summary['by_type']:
            print("\nActions by Type:")
            for action_type, count in summary['by_type'].items():
                print(f"  {action_type}: {count}")
        
        if summary['by_phase']:
            print("\nActions by Phase:")
            for phase, count in summary['by_phase'].items():
                print(f"  {phase}: {count}")
        print("="*60 + "\n")

    def verify_action_consistency(self):
        """
        Verify that tracked actions are consistent with recorded frames.
        
        Checks:
        - No overlapping actions (optional, can be relaxed)
        - All actions have positive frame ranges
        - Action frame indices match expectations
        
        Returns:
            tuple: (is_consistent: bool, issues: list of problem descriptions)
        """
        if self.shared_state is None:
            return True, []
        
        tracker = self.shared_state.action_tracker
        actions = tracker.get_all_actions()
        issues = []
        
        for action in actions:
            # Check frame range validity
            if action.start_frame < 0:
                issues.append(f"Action {action.action_id}: negative start_frame")
            
            if action.end_frame is not None and action.end_frame < action.start_frame:
                issues.append(f"Action {action.action_id}: end_frame before start_frame")
            
            if action.end_frame is not None and (action.end_frame - action.start_frame) < 0:
                issues.append(f"Action {action.action_id}: invalid duration")
        
        is_consistent = len(issues) == 0
        return is_consistent, issues


# ========================================================================
# INTEGRATION CHECKLIST
# ========================================================================

"""
To integrate action tracking into your RandomPushGrasper:

1. IMPORTS [Easy - 2 min]
   [ ] Add these imports to random_push_grasper.py:
       from action_tracker import ActionType, ActionPhase
       import logging
       logger = logging.getLogger(__name__)

2. PERFORM_TASK() [Moderate - 15-30 min]
   [ ] Replace the current perform_task() or update it with tracking:
       - Capture frame_start before each MOVE_XY
       - Call start_action(ActionType.MOVE_XY, ..., is_planar_2d=True)
       - Execute movement orders
       - Call end_action() after movement completes
   [ ] Test by running and checking states.json for "action" field

3. RESET_TASK() [Moderate - 15-30 min]
   [ ] Add action tracking to reset_task():
       - Capture frame_start before each sweep
       - Call start_action(ActionType.SWEEP, ..., phase=ActionPhase.RESET)
       - Execute wall sweep
       - Call end_action() after sweep completes
   [ ] Test by running reset and checking actions in states.json

4. FRAME INDEXING [Important - can vary]
   [ ] Figure out how to get current frame_index:
       Option A: Pass recorder reference to grasper (simple)
       Option B: Add callback mechanism (more elegant)
       Option C: Store frame_index in shared_state (dirty but works)
   
5. TESTING [Important - 10-20 min]
   [ ] Run a full task session
   [ ] Check that states.json has action field
   [ ] Check that actions.json is created
   [ ] Verify frame ranges are sensible
   [ ] Verify is_planar_2d flags are correct

6. DATASET CREATION [Useful for future]
   [ ] Write a script to extract planar pushing data
   [ ] Create train/test splits based on action types
   [ ] Verify dataset quality
"""
