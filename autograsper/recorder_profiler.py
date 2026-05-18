"""
Profiling wrapper to diagnose performance bottlenecks in autograsper/recording.py
Measures timing of:
- API calls (robot.get_all_states)
- Image decoding
- Image saving
- State persistence
"""
import time
from collections import defaultdict
from typing import Dict, List, Tuple
import statistics


class RecorderProfiler:
    """Profiles the Recorder class to identify bottlenecks."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.frame_details: List[Dict[str, float]] = []
        
    def record_operation(self, op_name: str, duration: float):
        """Record an operation's duration."""
        self.timings[op_name].append(duration)
    
    def record_frame(self, frame_num: int, details: Dict[str, float]):
        """Record complete frame timing details."""
        self.frame_details.append({
            'frame': frame_num,
            **details
        })
    
    def print_summary(self):
        """Print comprehensive timing summary."""
        print("\n" + "="*90)
        print("RECORDER PROFILING SUMMARY")
        print("="*90)
        
        # Overall statistics
        if self.frame_details:
            total_times = [d['total'] for d in self.frame_details]
            api_times = [d['api_call'] for d in self.frame_details]
            
            print("\n📊 FRAME TIMING")
            print("-" * 90)
            print(f"Total frames recorded: {len(total_times)}")
            print(f"Mean frame time:       {statistics.mean(total_times):.4f}s")
            print(f"Stdev:                 {statistics.stdev(total_times):.4f}s" if len(total_times) > 1 else "")
            print(f"Min/Max:               {min(total_times):.4f}s / {max(total_times):.4f}s")
            print(f"Achieved FPS:          {1/statistics.mean(total_times):.2f}")
            print(f"Target FPS:            5.0 (0.2000s per frame)")
            print(f"Slowdown:              {(statistics.mean(total_times) - 0.2)*100:.1f}ms over target")
            
            print("\n🌐 API CALL TIMING (robot.get_all_states)")
            print("-" * 90)
            print(f"Mean:                  {statistics.mean(api_times):.4f}s")
            print(f"Stdev:                 {statistics.stdev(api_times):.4f}s" if len(api_times) > 1 else "")
            print(f"Min/Max:               {min(api_times):.4f}s / {max(api_times):.4f}s")
            print(f"% of frame time:       {(statistics.mean(api_times)/statistics.mean(total_times)*100):.1f}%")
        
        # Breakdown by operation
        print("\n⏱️  OPERATION BREAKDOWN")
        print("-" * 90)
        
        op_names = [
            'api_call',
            'image_decoding',
            'capture_frame',
            'save_state',
            'wait_delay'
        ]
        
        for op_name in op_names:
            if op_name in self.timings and self.timings[op_name]:
                times = self.timings[op_name]
                mean = statistics.mean(times)
                total = sum(times)
                pct = (mean / statistics.mean([d['total'] for d in self.frame_details]) * 100) if self.frame_details else 0
                
                print(f"\n{op_name}:")
                print(f"  Mean:   {mean:.4f}s ({pct:.1f}% of frame)")
                print(f"  Total:  {total:.2f}s across {len(times)} frames")
                print(f"  Stdev:  {statistics.stdev(times):.4f}s")
        
        # Identify bottleneck
        if self.frame_details:
            print("\n🎯 BOTTLENECK ANALYSIS")
            print("-" * 90)
            
            avg_frame = statistics.mean([d['total'] for d in self.frame_details])
            avg_api = statistics.mean([d['api_call'] for d in self.frame_details])
            avg_capture = statistics.mean([d['capture_frame'] for d in self.frame_details if 'capture_frame' in d])
            avg_save = statistics.mean([d['save_state'] for d in self.frame_details if 'save_state' in d])
            
            print(f"Total frame time:      {avg_frame:.4f}s (100%)")
            print(f"  API call:            {avg_api:.4f}s ({avg_api/avg_frame*100:.1f}%) ← Network + server response")
            if 'capture_frame' in self.frame_details[0]:
                print(f"  Image saving:        {avg_capture:.4f}s ({avg_capture/avg_frame*100:.1f}%) ← Disk I/O")
            if 'save_state' in self.frame_details[0]:
                print(f"  State persistence:   {avg_save:.4f}s ({avg_save/avg_frame*100:.1f}%) ← JSON file I/O")
            
            # Recommendations
            print("\n💡 RECOMMENDATIONS")
            print("-" * 90)
            
            if avg_api / avg_frame > 0.7:
                print("⚠️  API CALL is the main bottleneck (>70% of frame time)")
                print("   → Network latency or server processing is slow")
                print("   → Consider: async API calls, image compression, request batching")
            
            if 'capture_frame' in self.frame_details[0] and avg_capture / avg_frame > 0.3:
                print("⚠️  IMAGE SAVING is significant (>30% of frame time)")
                print("   → Disk I/O is limiting throughput")
                print("   → Consider: async disk writes, image compression, write buffering")
            
            if 'save_state' in self.frame_details[0] and avg_save / avg_frame > 0.2:
                print("⚠️  STATE PERSISTENCE is significant (>20% of frame time)")
                print("   → JSON file writes are expensive")
                print("   → Consider: batch writes, in-memory buffer, async persistence")
    
    def print_timeline(self, num_frames: int = 20):
        """Print timeline of last N frames."""
        print("\n" + "="*90)
        print(f"FRAME TIMELINE (last {min(num_frames, len(self.frame_details))} frames)")
        print("="*90)
        
        for frame_data in self.frame_details[-num_frames:]:
            frame_num = frame_data['frame']
            total = frame_data['total']
            api = frame_data.get('api_call', 0)
            capture = frame_data.get('capture_frame', 0)
            save = frame_data.get('save_state', 0)
            
            # Visual bar
            bar_len = int(total * 100)  # Scale: 0.01s = 1 char
            bar = "█" * min(bar_len, 50)  # Cap at 50 chars
            
            print(f"\nFrame {frame_num:3d}: {total:.4f}s {bar}")
            if api > 0:
                print(f"    API:   {api:.4f}s {' '*40}")
            if capture > 0:
                print(f"    Save:  {capture:.4f}s {' '*40}")
            if save > 0:
                print(f"    State: {save:.4f}s {' '*40}")


# Global profiler instance
recorder_profiler = RecorderProfiler()


def profile_update(robot_obj, camera_matrix, distortion_coeffs, h_matrix, shared_state):
    """
    Wrapper to profile the _update method.
    Measures: API call + image decoding time.
    """
    from library.utils import get_undistorted_bottom_image
    
    start = time.perf_counter()
    
    # API call
    api_start = time.perf_counter()
    data = robot_obj.get_all_states()
    api_duration = time.perf_counter() - api_start
    recorder_profiler.record_operation('api_call', api_duration)
    
    # Image processing
    decode_start = time.perf_counter()
    image_top = data[0]
    bottom_image_raw = data[1]
    bottom_image = get_undistorted_bottom_image(
        bottom_image_raw, camera_matrix, distortion_coeffs, h_matrix
    )
    decode_duration = time.perf_counter() - decode_start
    recorder_profiler.record_operation('image_decoding', decode_duration)
    
    # State update
    state = data[2]
    timestamp = data[3]
    
    with shared_state.image_lock:
        shared_state.latest_top_image = image_top
        shared_state.latest_bottom_image = bottom_image
        shared_state.latest_robot_state = state
        shared_state.timestamp = timestamp
    
    total_duration = time.perf_counter() - start
    return total_duration, api_duration


def profile_capture_frame(recorder, image_lock):
    """
    Wrapper to profile the _capture_frame method.
    Measures disk I/O time for image saving.
    """
    start = time.perf_counter()
    
    if not recorder.ensure_images():
        return 0
    
    bottom_image_raw = None
    mask = None
    with image_lock:
        top_image = recorder.image_top.copy()
        bottom_image = recorder.bottom_image.copy()
        if recorder.bottom_image_raw is not None:
            bottom_image_raw = recorder.bottom_image_raw.copy()
        if recorder.shared_state.latest_mask is not None and not recorder.shared_state.latest_mask_saved:
            mask = recorder.shared_state.latest_mask
            recorder.shared_state.latest_mask_saved = True
    
    save_start = time.perf_counter()
    
    if recorder.save_images_individually:
        recorder._save_individual_images(top_image, bottom_image, bottom_image_raw, mask)
    else:
        with recorder.writer_lock:
            if (recorder.video_writer_top is not None and 
                recorder.video_writer_bottom is not None):
                recorder.video_writer_top.write(top_image)
                recorder.video_writer_bottom.write(bottom_image)
    
    save_duration = time.perf_counter() - save_start
    recorder_profiler.record_operation('capture_frame', save_duration)
    
    with recorder.snapshot_cond:
        if recorder.take_snapshot > 0:
            recorder.take_snapshot -= 1
            if recorder.take_snapshot == 0:
                recorder.snapshot_cond.notify_all()
    
    return save_duration


def profile_save_state(recorder):
    """Wrapper to profile the save_state method."""
    start = time.perf_counter()
    recorder.save_state()
    duration = time.perf_counter() - start
    recorder_profiler.record_operation('save_state', duration)
    return duration


def profile_wait(shutdown_event, fps):
    """Measure actual wait time."""
    start = time.perf_counter()
    shutdown_event.wait(1 / fps)
    duration = time.perf_counter() - start
    recorder_profiler.record_operation('wait_delay', duration)
    return duration
