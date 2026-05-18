"""
USAGE INSTRUCTIONS FOR DIAGNOSING PERFORMANCE BOTTLENECKS

To use the recorder profiler with your existing code:

Option 1: MINIMAL CHANGES - Add profiling to recording.py
=============================================================

1. Import the profiler at the top of recording.py:
   
   from recorder_profiler import recorder_profiler, profile_update, profile_capture_frame, profile_save_state, profile_wait

2. In the _update method, replace:
   
   data = self.robot.get_all_states()
   # ... rest of update code
   
   With:
   
   # Profiled version
   update_dur, api_dur = profile_update(
       self.robot, self.camera_matrix, self.distortion_coeffs, self.H_matrix, self.shared_state
   )

3. In the record() method, wrap the capture and save calls:
   
   Before:
   if self.save_data and self.disk_enabled:
       self._capture_frame()
   # ...
   if self.save_data and self.disk_enabled:
       self.save_state()
   
   After (add timing tracking):
   capture_dur = 0
   save_state_dur = 0
   if self.save_data and self.disk_enabled:
       capture_dur = profile_capture_frame(self, self.image_lock)
   # ...
   if self.save_data and self.disk_enabled:
       save_state_dur = profile_save_state(self)
   
   # Record frame timing
   wait_dur = profile_wait(self.shutdown_event, self.FPS)
   
   frame_details = {
       'api_call': api_dur,
       'image_decoding': update_dur - api_dur,
       'capture_frame': capture_dur,
       'save_state': save_state_dur,
       'wait_delay': wait_dur,
       'total': api_dur + (update_dur - api_dur) + capture_dur + save_state_dur + wait_dur
   }
   recorder_profiler.record_frame(self.frame_counter, frame_details)

4. At the end of your script, print the report:
   
   finally:
       recorder_profiler.print_summary()
       recorder_profiler.print_timeline()


Option 2: STANDALONE DIAGNOSTIC SCRIPT
=========================================

Run the diagnostic script provided in autograsper/diagnose_recorder.py
This tests the profiling without modifying your production code.


WHAT THE PROFILER MEASURES
============================

- api_call: Time for robot.get_all_states() network call
- image_decoding: Time to decode base64 images to numpy/OpenCV
- capture_frame: Time to save images to disk
- save_state: Time to write state data to JSON file
- wait_delay: Time spent in timing.sleep() / Event.wait()
- total: Sum of all above

INTERPRETING RESULTS
====================

Your reported metrics:
  - Average timestep: 0.38s (want 0.2s)
  - Variance: 0.04s (standard deviation in timing)
  - Target: 5 FPS

The profiler will tell you:
  1. Which operation takes the most time
  2. If it's network (api_call) or I/O (capture_frame, save_state)
  3. Where variance is coming from (inconsistent operation times)

Likely bottleneck scenarios:

A) If api_call is 60%+ of frame time:
   → Network/server is slow
   → Solution: async API calls, image compression request, parallel requests

B) If capture_frame is 30%+ of frame time:
   → Disk I/O is slow (save_images_individually = True)
   → Solution: async disk writes, write batching, compress before save

C) If save_state is 20%+ of frame time:
   → JSON file writes are expensive
   → Solution: batch state writes, in-memory buffer, async persistence

D) If variance is high in api_call:
   → Inconsistent network latency
   → Solution: connection pooling, keep-alive, server optimization

E) If variance is high in capture_frame:
   → Disk I/O latency varies
   → Solution: write buffering, dedicated fast storage, reduce image size
"""

# Example output interpretation:

example_output = """
RECORDER PROFILING SUMMARY
===========================================================================

📊 FRAME TIMING
---------------------------------------------------------------------------
Total frames recorded: 50
Mean frame time:       0.3800s
Stdev:                 0.0400s
Min/Max:               0.3400s / 0.4500s
Achieved FPS:          2.63
Target FPS:            5.0 (0.2000s per frame)
Slowdown:              180.0ms over target

🌐 API CALL TIMING (robot.get_all_states)
---------------------------------------------------------------------------
Mean:                  0.2500s
Stdev:                 0.0350s
Min/Max:               0.2100s / 0.3200s
% of frame time:       65.8%

⏱️  OPERATION BREAKDOWN
---------------------------------------------------------------------------

api_call:
  Mean:   0.2500s (65.8% of frame)
  Total:  12.50s across 50 frames
  Stdev:  0.0350s

capture_frame:
  Mean:   0.0800s (21.1% of frame)
  Total:  4.00s across 50 frames
  Stdev:  0.0150s

save_state:
  Mean:   0.0150s (3.9% of frame)
  Total:  0.75s across 50 frames
  Stdev:  0.0030s

wait_delay:
  Mean:   0.0350s (9.2% of frame)
  Total:  1.75s across 50 frames
  Stdev:  0.0100s

🎯 BOTTLENECK ANALYSIS
---------------------------------------------------------------------------
Total frame time:      0.3800s (100%)
  API call:            0.2500s (65.8%) ← Network + server response
  Image saving:        0.0800s (21.1%) ← Disk I/O
  State persistence:   0.0150s (3.9%) ← JSON file I/O

💡 RECOMMENDATIONS
---------------------------------------------------------------------------
⚠️  API CALL is the main bottleneck (>70% of frame time)
   → Network latency or server processing is slow
   → Consider: async API calls, image compression, request batching
"""

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("EXAMPLE OUTPUT:")
    print("="*80)
    print(example_output)
