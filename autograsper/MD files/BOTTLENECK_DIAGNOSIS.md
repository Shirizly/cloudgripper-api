"""
DIAGNOSING FPS BOTTLENECKS IN autograsper/recording.py

Your Problem:
=============
- Target: 5 FPS (0.2 seconds per frame)
- Actual: 2.6 FPS (0.38 seconds per frame average)
- Variance: 0.04s stdev (inconsistent)

This is a ~1.9x slowdown from target.


The Recording Loop (simplified)
================================

Your recording loop does approximately:

    while recording:
        # 1. Get data from robot (API + decoding)
        _update()                    # ← Where is time spent?
        
        # 2. Save images to disk
        _capture_frame()             # ← Fast or slow?
        
        # 3. Write state to JSON
        save_state()                 # ← Fast or slow?
        
        # 4. Wait for remaining time
        wait(1/FPS)                  # ← How much time left?


The Key Issue:
===============
The wait(1/FPS) call doesn't account for time already spent in steps 1-3.

If FPS=5 (target 0.2s per frame):
  - Ideally: each of steps 1-3 takes ~0.06s, leaving 0.2s - 0.18s = 0.02s to wait
  - Actual: if steps 1-3 take 0.18s, you wait 0.2s more = 0.38s total

So the slowdown is NOT the wait, but something in steps 1-3.


Where Time Is Likely Spent:
===========================

Step 1: _update() does:
  a) robot.get_all_states()      ← Network call + JSON parse
  b) Image base64 decode        ← CPU intensive (cv2.imdecode)
  c) Image undistortion         ← CPU intensive (OpenCV warp)

Step 2: _capture_frame() does:
  a) Image copy (with locks)
  b) cv2.imwrite() to disk      ← Depends on: disk speed, image size, format

Step 3: save_state() does:
  a) Read existing JSON from disk
  b) Parse JSON
  c) Append new state
  d) Write JSON back to disk

Step 4: wait(1/FPS)
  a) Usually fast (just GPIO/Event.wait)


Diagnostic Strategy:
====================

To find the bottleneck, measure each step independently.

OPTION A: MINIMAL PATCH (Recommended, 5 minutes)
  1. Copy code from DIAGNOSTIC_PATCH.py into recording.py
  2. Run your recording for ~50 frames
  3. Review printed statistics
  4. Identify which step takes longest

OPTION B: DETAILED ANALYSIS (10 minutes)
  1. Import recorder_profiler module
  2. Run with profiling enabled
  3. Get detailed breakdown with recommendations


EXPECTED RESULTS AND WHAT THEY MEAN:
====================================

Scenario A: API call is 60%+ of time
  Example: 0.24s / 0.38s = 63%
  
  Interpretation:
    - robot.get_all_states() is slow
    - Could be: network latency, server processing, image encoding
    - Variance in this would explain variance in frames
  
  Solutions:
    1. Profile just the API call with timeouts:
       start = time.time()
       img1, img2, state, ts = robot.get_all_states()
       print(f"API took {time.time() - start:.4f}s")
    
    2. Check server response time:
       - Ask server team for processing time
       - Compare to your network delay
    
    3. Optimize API calls:
       - Request smaller images (compression before transmission)
       - Use async calls if possible
       - Batch multiple requests
       - Connection pooling / keep-alive

Scenario B: Image saving is 30%+ of time
  Example: 0.12s / 0.38s = 32%
  
  Interpretation:
    - cv2.imwrite() is slow
    - Could be: slow disk, large images, compression time
  
  Solutions:
    1. Check disk speed:
       time dd if=/dev/urandom of=test.img bs=1M count=100
       (should be >100 MB/s on decent hardware)
    
    2. Optimize image saving:
       - Use write buffering
       - Write asynchronously in separate thread
       - Reduce image size
       - Use faster format (MJPG vs PNG)
       - Write to faster storage (SSD vs spinning disk)

Scenario C: State persistence is 20%+ of time
  Example: 0.08s / 0.38s = 21%
  
  Interpretation:
    - JSON file I/O is expensive
    - Problem: reading full file, parsing, appending, writing on EVERY frame
  
  Solutions:
    1. Batch state writes:
       - Write every 10 frames instead of every frame
       - Or use in-memory buffer + periodic flush
    
    2. Use faster format:
       - Replace JSON with binary format (MessagePack, Protocol Buffers)
       - Use streaming JSON writer
    
    3. Async persistence:
       - Write to disk in background thread
       - Don't block frame capture


INSTRUMENTATION CODE (COPY-PASTE READY):
==========================================

Add this to recorder.py __init__:

    import time
    self.frame_timings = []

Replace _update() with this:

    def _update(self) -> None:
        start_api = time.perf_counter()
        data = self.robot.get_all_states()
        api_time = time.perf_counter() - start_api
        
        start_decode = time.perf_counter()
        with self.image_lock:
            self.image_top = data[0]
            if self.save_bottom_raw:
                self.bottom_image_raw = data[1]
            self.bottom_image = get_undistorted_bottom_image(
                data[1], self.camera_matrix, self.distortion_coeffs, self.H_matrix
            )
        decode_time = time.perf_counter() - start_decode
        
        self.state = data[2]
        self.timestamp = data[3]
        with self.shared_state.image_lock:
            self.shared_state.latest_top_image = self.image_top
            self.shared_state.latest_bottom_image = self.bottom_image
            self.shared_state.latest_robot_state = data[2]
            self.shared_state.timestamp = data[3]
        
        self._last_api_time = api_time
        self._last_decode_time = decode_time

In the record() loop, replace the timing section:

    if (not self.record_only_after_action) or (self.take_snapshot > 0):
        frame_start = time.perf_counter()
        
        capture_time = 0
        if self.save_data and self.disk_enabled:
            cap_start = time.perf_counter()
            self._capture_frame()
            capture_time = time.perf_counter() - cap_start
        
        save_time = 0
        if self.save_data and self.disk_enabled:
            sav_start = time.perf_counter()
            self.save_state()
            save_time = time.perf_counter() - sav_start
        
        self.shutdown_event.wait(1 / self.FPS)
        
        frame_time = time.perf_counter() - frame_start
        api_t = getattr(self, '_last_api_time', 0)
        decode_t = getattr(self, '_last_decode_time', 0)
        
        self.frame_timings.append({
            'frame': self.frame_counter,
            'total': frame_time,
            'api': api_t,
            'decode': decode_t,
            'capture': capture_time,
            'save': save_time,
        })
        
        if self.frame_counter % 50 == 0:
            print(f"Frame {self.frame_counter}: total={frame_time:.4f}s "
                  f"(api={api_t:.4f}s, decode={decode_t:.4f}s, "
                  f"capture={capture_time:.4f}s, save={save_time:.4f}s)")
        
        self.frame_counter += 1

At the end, add reporting:

    if hasattr(recorder, 'frame_timings'):
        import statistics
        timings = recorder.frame_timings
        totals = [t['total'] for t in timings]
        apis = [t['api'] for t in timings]
        
        print(f"\n{'='*70}")
        print(f"TIMING SUMMARY ({len(timings)} frames)")
        print(f"{'='*70}")
        print(f"Mean frame time: {statistics.mean(totals):.4f}s (target 0.2000s)")
        print(f"Variance: {statistics.stdev(totals):.4f}s")
        print(f"FPS: {1/statistics.mean(totals):.2f} (target 5.0)")
        print(f"\nAPI call: {statistics.mean(apis):.4f}s "
              f"({statistics.mean(apis)/statistics.mean(totals)*100:.0f}%)")
        
        cap = statistics.mean([t['capture'] for t in timings])
        print(f"Capture:  {cap:.4f}s ({cap/statistics.mean(totals)*100:.0f}%)")
        
        sav = statistics.mean([t['save'] for t in timings])
        print(f"Save:     {sav:.4f}s ({sav/statistics.mean(totals)*100:.0f}%)")


NEXT STEPS:
===========

1. Add minimal instrumentation to recording.py
2. Run a short recording session (50-100 frames)
3. Look at printed statistics
4. Identify which component is the bottleneck
5. Share the results for targeted optimization

The profiler will directly tell you:
  ✓ If it's network (robot.get_all_states)
  ✓ If it's image decoding/processing
  ✓ If it's disk I/O (image saving)
  ✓ If it's JSON persistence
"""

if __name__ == "__main__":
    print(__doc__)
