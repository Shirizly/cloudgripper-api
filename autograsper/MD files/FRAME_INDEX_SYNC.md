"""
Frame Index Tracking for Action Boundary Marking

This module provides mechanisms to synchronize the recorder's frame counter
with the grasper so that actions can be marked with precise frame boundaries.

There are three approaches with different trade-offs:

1. SHARED STATE APPROACH (Simplest, recommended)
   - Add frame_index field to SharedState
   - Recorder updates it after each frame is processed
   - Grasper reads it when marking action boundaries
   Pros: Simple, thread-safe, no architectural changes needed
   Cons: Slight threading complexity, frame index may lag slightly

2. CALLBACK APPROACH (Most elegant)
   - Grasper registers callbacks to recorder
   - Recorder calls callbacks at frame milestones
   - Grasper can query frame index in real-time
   Pros: Clean architecture, decoupled, real-time
   Cons: More complex setup, requires architectural changes

3. DIRECT REFERENCE APPROACH (Quickest fix)
   - Pass recorder reference to grasper
   - Grasper directly accesses recorder.frame_counter
   Pros: Fastest to implement
   Cons: Tight coupling, not ideal architecture
"""


# ============================================================================
# APPROACH 1: SHARED STATE (RECOMMENDED)
# ============================================================================

# Step 1: Update SharedState in coordinator.py

from coordinator import SharedState

# ADD THIS to the @dataclass SharedState:
"""
    # Frame index synchronization
    frame_index: int = 0
    frame_index_lock: threading.RLock = field(default_factory=threading.RLock)
"""

# Step 2: Update Recorder to set frame_index in SharedState

# In recording.py, update the record() method or _capture_frame():

"""
    def _capture_frame(self) -> None:
        '''Capture and save the current frame as an image or add it to the video writer.'''
        try:
            # ... existing code ...
            
            # UPDATE SHARED STATE WITH CURRENT FRAME INDEX
            with self.shared_state.frame_index_lock:
                self.shared_state.frame_index = self.frame_counter
            
            # ... rest of existing code ...
"""

# Step 3: Update grasper to read from SharedState

# In grasper.py, update get_current_frame_index():

"""
    def get_current_frame_index(self) -> Optional[int]:
        '''Get the current frame index being recorded.'''
        if self.shared_state is None:
            return None
        
        with self.shared_state.frame_index_lock:
            return self.shared_state.frame_index
"""


# ============================================================================
# IMPLEMENTATION FOR APPROACH 1
# ============================================================================

# This is the code to add to coordinator.py:

COORDINATOR_SHARED_STATE_UPDATE = """
@dataclass
class SharedState:
    '''
    Holds shared references between threads.
    
    Includes action tracking for recording robot movements and actions
    alongside captured frames and states.
    '''
    state = RobotActivity.STARTUP
    latest_top_image: np.ndarray | None = None
    latest_bottom_image: np.ndarray | None = None
    latest_robot_state: dict | None = None
    latest_mask: np.ndarray | None = None
    latest_mask_saved: bool = False
    timestamp: float | None = None
    image_lock: threading.RLock = field(default_factory=threading.RLock)
    
    # Action tracking
    action_tracker: ActionTracker = field(default_factory=ActionTracker)
    
    # Frame index synchronization for precise action boundaries
    frame_index: int = 0
    frame_index_lock: threading.RLock = field(default_factory=threading.RLock)
"""

# This is the code to add to recording.py in the _capture_frame method:

RECORDER_FRAME_INDEX_UPDATE = """
    def _capture_frame(self) -> None:
        '''Capture and save the current frame as an image or add it to the video writer.'''
        try:
            if not self.ensure_images():
                return
            
            # Update shared state with current frame index BEFORE capture
            # This ensures action markers have accurate frame boundaries
            with self.shared_state.frame_index_lock:
                self.shared_state.frame_index = self.frame_counter
            
            # ... rest of existing code ...
"""

# This is the code to update in grasper.py:

GRASPER_FRAME_INDEX_READ = """
    def get_current_frame_index(self) -> Optional[int]:
        '''
        Get the current frame index being recorded.
        
        Returns:
            Frame index if available, None otherwise.
            
        Note: There may be a small delay (1-2 frames) due to threading,
        but this is acceptable for marking action boundaries.
        '''
        if self.shared_state is None:
            return None
        
        try:
            with self.shared_state.frame_index_lock:
                return self.shared_state.frame_index
        except (AttributeError, ValueError):
            # Fallback if frame_index not available
            return None
"""


# ============================================================================
# EXAMPLE: Using Frame Index in RandomPushGrasper
# ============================================================================

example_usage = """
def perform_task_with_frame_tracking(self):
    '''Example of using get_current_frame_index() for action tracking.'''
    
    for i in range(self.N_pushes):
        if self.shutdown_event.is_set():
            break
        
        x = np.random.uniform(self.manip_x[0], self.manip_x[1])
        y = np.random.uniform(self.manip_y[0], self.manip_y[1])
        
        # Get frame index BEFORE movement
        frame_start = self.get_current_frame_index()
        
        # Mark action start
        action_id = self.start_action(
            action_type=ActionType.MOVE_XY,
            phase=ActionPhase.TASK,
            frame_index=frame_start if frame_start is not None else -1,
            is_planar_2d=True,
            action_details={"x": float(x), "y": float(y)},
        )
        
        # Execute movement
        self.robot.move_xy(x, y)
        time.sleep(1.0)
        
        # Get frame index AFTER movement completes
        frame_end = self.get_current_frame_index()
        
        # Mark action end
        if frame_end is not None:
            self.end_action(action_id, frame_end)
"""


# ============================================================================
# QUICK IMPLEMENTATION GUIDE
# ============================================================================

implementation_steps = """
QUICK IMPLEMENTATION (5 minutes):

1. Add to coordinator.py SharedState:
   Copy the COORDINATOR_SHARED_STATE_UPDATE above 
   (adds 2 lines to the @dataclass)

2. Add to recording.py _capture_frame():
   Add these lines before self.ensure_images():
   
   with self.shared_state.frame_index_lock:
       self.shared_state.frame_index = self.frame_counter

3. Add to grasper.py:
   Replace the get_current_frame_index() stub with the full implementation
   (copy from GRASPER_FRAME_INDEX_READ above)

4. Use in your grasper:
   Call get_current_frame_index() when you need frame boundaries
   
   frame_start = self.get_current_frame_index()
   action_id = self.start_action(...)
   # ... do work ...
   frame_end = self.get_current_frame_index()
   self.end_action(action_id, frame_end)

5. Test:
   Run a task and verify that frame indices are reasonable
   Check outputs:
   - states.json should have action info
   - actions.json should have proper frame ranges
"""

print(implementation_steps)
