# live_test.py
"""
    
Controls:
    'q' or ESC  : Quit
    's'         : Save screenshot
    'r'         : Start/stop recording
    'c'         : Clear metrics
    'h'         : Toggle help text
    'f'         : Toggle FPS display
"""

import cv2
import numpy as np
import time
from collections import deque
from datetime import datetime
import csv
import os
from analysis import TemporalAnalyzer, DrowsinessState


# Note: You'll need to import P1, P2, P3 modules
# For now, we'll create mock versions - replace with actual imports
try:
    from segmentation import FaceSegmentor
    from eye_hog import EyeStateClassifier
    from dlib_ear import EARDetector
except ImportError:
    print("   Warning: P1-P3 modules not found. Using MOCK MODE for testing.")
    print("   In production, ensure P1, P2, P3 modules are imported correctly.\n")
    
    # MOCK MODE: Simulates P1, P2, P3 outputs for standalone testing
    class FaceSegmentor:
        def process(self, frame):
            return {
                "success": True,
                "face_crop": cv2.resize(frame, (224, 224)),
                "bbox": (50, 50, 200, 200),
                "skin_mask": np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8),
                "debug_frame": frame.copy(),
            }
    
    class EyeStateClassifier:
        def process(self, face_crop):
            return {
                "success": True,
                "eye_state": "open",
                "confidence": 0.9,
                "debug_frame": face_crop.copy(),
            }
    
    class EARDetector:
        def __init__(self):
            self.blink_count = 0
            
        def process(self, face_crop, bbox):
            # Simulate realistic EAR values
            import random
            self.blink_count += 1
            
            if self.blink_count % 30 == 15:
                ear = 0.10 + np.random.normal(0, 0.02)
            else:
                ear = 0.35 + np.random.normal(0, 0.03)
            
            return {
                "success": True,
                "ear": np.clip(ear, 0.0, 0.5),
                "ear_left": ear,
                "ear_right": ear,
                "blink_detected": self.blink_count % 30 == 15,
                "blink_count": self.blink_count // 30,
                "head_pose": (
                    np.random.normal(0, 5),
                    np.random.normal(0, 5),
                ),
                "left_eye_pts": np.array([[0, 0]] * 6),
                "right_eye_pts": np.array([[0, 0]] * 6),
            }


class LiveDrowsinessTest:
    
    def __init__(self, fps=30, enable_logging=True):
        
        self.fps = fps
        self.frame_count = 0
        self.enable_logging = enable_logging
        
        # Initialize components
        print("Initializing components...")
        self.segmentor = FaceSegmentor()
        self.hog_classifier = EyeStateClassifier()
        self.ear_detector = EARDetector()
        self.temporal_analyzer = TemporalAnalyzer(fps=fps)
        
        # Metrics tracking
        self.metrics_history = deque(maxlen=300)  # 10 seconds @ 30 FPS
        self.state_start_frame = 0
        self.state_durations = {"alert": 0, "drowsy": 0, "critical": 0}
        
        # Visualization
        self.show_help = False
        self.show_fps = True
        self.recording = False
        self.video_writer = None
        
        # Logging
        self.csv_file = None
        self.csv_writer = None
        if enable_logging:
            self._setup_logging()
        
        # Timing
        self.frame_times = deque(maxlen=30)
        self.start_time = time.time()
        
        print(" Components initialized. Starting camera...")
    
    def _setup_logging(self):
        """Setup CSV logging for metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"drowsiness_log_{timestamp}.csv"
        
        try:
            self.csv_file = open(self.log_filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header
            self.csv_writer.writerow([
                "frame", "timestamp", "drowsiness_score", "state",
                "ear", "blink_rate", "closure_duration",
                "ear_score", "blink_score", "eye_state_score",
                "closure_score", "head_pose_score"
            ])
            self.csv_file.flush()
            
            print(f"Logging to: {self.log_filename}")
        except Exception as e:
            print(f"Logging setup failed: {e}")
            self.enable_logging = False
    
    def _log_frame(self, result):
        """Log frame data to CSV"""
        if not self.csv_writer:
            return
        
        try:
            self.csv_writer.writerow([
                self.frame_count,
                datetime.now().isoformat(),
                result.get("drowsiness_score", 0),
                result.get("state", "unknown"),
                result.get("ear_mean", 0),
                result.get("blink_rate", 0),
                result.get("eye_closure_duration", 0),
                result.get("signals", {}).get("ear_score", 0),
                result.get("signals", {}).get("blink_score", 0),
                result.get("signals", {}).get("eye_state_score", 0),
                result.get("signals", {}).get("closure_score", 0),
                result.get("signals", {}).get("head_pose_score", 0),
            ])
            self.csv_file.flush()
        except Exception as e:
            print(f"Logging error: {e}")
    
    def run(self):
        """Main testing loop"""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("Could not open camera. Check if it's in use.")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        print("Camera opened. Press 'h' for help, 'q' to quit.")
        print()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            self.frame_count += 1
            frame_time = time.time()
            self.frame_times.append(frame_time)
            
            # Process frame through P1-P4 pipeline
            result = self._process_frame(frame)
            
            # Draw visualization
            output_frame = self._draw_visualization(frame, result)
            
            # Display
            cv2.imshow("P4 Drowsiness Detection - Live Test", output_frame)
            
            # Save video frame if recording
            if self.recording and self.video_writer:
                self.video_writer.write(output_frame)
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('s'):
                self._save_screenshot(output_frame)
            elif key == ord('r'):
                self._toggle_recording(output_frame)
            elif key == ord('c'):
                self.temporal_analyzer.reset()
                self.metrics_history.clear()
                print("Metrics cleared")
            elif key == ord('h'):
                self.show_help = not self.show_help
            elif key == ord('f'):
                self.show_fps = not self.show_fps
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if self.csv_file:
            self.csv_file.close()
        if self.video_writer:
            self.video_writer.release()
        
        print("\n✅ Test completed")
        self._print_summary()
    
    def _process_frame(self, frame):
        """Process a single frame through the P1-P4 pipeline"""
        # P1: Segmentation
        seg_result = self.segmentor.process(frame)
        if not seg_result["success"]:
            return {
                "success": False,
                "drowsiness_score": 0,
                "state": "unknown",
                "signals": {},
            }
        
        face_crop = seg_result["face_crop"]
        
        # P2: HOG Eye State Classification
        hog_result = self.hog_classifier.process(face_crop)
        
        # P3: dlib EAR Detection
        ear_result = self.ear_detector.process(face_crop, seg_result["bbox"])
        
        # P4: Temporal Analysis
        drowsiness_result = self.temporal_analyzer.process(ear_result, hog_result)
        
        # Log to CSV
        if drowsiness_result["success"]:
            self._log_frame(drowsiness_result)
            self.metrics_history.append(drowsiness_result)
        
        return drowsiness_result
    
    def _draw_visualization(self, frame, result):
        """Draw comprehensive visualization on frame."""
       
        h, w = frame.shape[:2]
        output = frame.copy()
        
        # Determine state color
        if not result.get("success"):
            state_color = (128, 128, 128)  # Gray - no detection
            state_text = "NO DETECTION"
        else:
            state = result.get("state", "alert")
            if state == "critical":
                state_color = (0, 0, 255)  # Red
            elif state == "drowsy":
                state_color = (0, 255, 255)  # Yellow
            else:
                state_color = (0, 255, 0)  # Green
            state_text = state.upper()
        
        # Draw background panel
        cv2.rectangle(output, (0, 0), (w, 120), (0, 0, 0), -1)
        
        
        # Main metrics
        if result.get("success"):
            score = result.get("drowsiness_score", 0)
            confidence = result.get("confidence", 0)
            
            # State indicator
            cv2.rectangle(output, (10, 10), (200, 100), state_color, -1)
            cv2.rectangle(output, (10, 10), (200, 100), state_color, 3)
            
            cv2.putText(output, state_text, (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(output, f"Score: {score:.2f}", (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Confidence and metrics
            cv2.putText(output, f"Conf: {confidence:.2f}", (220, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(output, f"Blinks: {result.get('blink_rate', 0):.0f}/min", 
                       (220, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(output, f"EAR: {result.get('ear_mean', 0):.3f}", 
                       (220, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Closure duration
            closure = result.get("eye_closure_duration", 0)
            if closure > 0.1:
                cv2.putText(output, f"Closure: {closure:.2f}s", (380, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            
            # Draw drowsiness gauge (horizontal bar)
            gauge_x, gauge_y = 10, 130
            gauge_w, gauge_h = 200, 25
            
            cv2.rectangle(output, (gauge_x, gauge_y), (gauge_x + gauge_w, gauge_y + gauge_h),
                         (200, 200, 200), 1)
            
            # Fill based on score
            fill_w = int(gauge_w * score)
            if fill_w > 0:
                cv2.rectangle(output, (gauge_x, gauge_y), (gauge_x + fill_w, gauge_y + gauge_h),
                             state_color, -1)
            
            cv2.putText(output, "Drowsiness Gauge", (gauge_x, gauge_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Signal breakdown
            signals = result.get("signals", {})
            signal_y = 170
            cv2.putText(output, "Signals:", (10, signal_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            signal_names = ["EAR", "Blink", "Eye", "Closure", "Head"]
            signal_keys = ["ear_score", "blink_score", "eye_state_score",
                          "closure_score", "head_pose_score"]
            
            for i, (name, key) in enumerate(zip(signal_names, signal_keys)):
                val = signals.get(key, 0)
                x = 10 + (i % 5) * 120
                y = signal_y + 25 + (i // 5) * 25
                
                # Signal bar
                bar_w = int(80 * val)
                cv2.rectangle(output, (x, y), (x + bar_w, y + 15),
                             (0, int(255 * val), int(255 * (1 - val))), -1)
                cv2.rectangle(output, (x, y), (x + 80, y + 15),
                             (100, 100, 100), 1)
                
                cv2.putText(output, f"{name}:{val:.2f}", (x, y - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        # FPS counter
        if self.show_fps and len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                fps = (len(self.frame_times) - 1) / time_diff
                cv2.putText(output, f"FPS: {fps:.1f}", (w - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Frame counter
        cv2.putText(output, f"Frame: {self.frame_count}", (w - 200, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Recording indicator
        if self.recording:
            cv2.circle(output, (w - 30, 30), 8, (0, 0, 255), -1)
            cv2.putText(output, "REC", (w - 70, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Help text
        if self.show_help:
            self._draw_help(output)
        
        return output
    
    def _draw_help(self, frame):
        """Draw help text overlay"""
        h, w = frame.shape[:2]
        help_text = [
            "CONTROLS:",
            "q/ESC - Quit",
            "s - Screenshot",
            "r - Record",
            "c - Clear metrics",
            "h - Toggle help",
            "f - Toggle FPS",
        ]
        
        y_start = h - len(help_text) * 25 - 10
        for i, text in enumerate(help_text):
            y = y_start + i * 25
            cv2.rectangle(frame, (w - 200, y - 5), (w - 5, y + 20),
                         (0, 0, 0), -1)
            
            cv2.putText(frame, text, (w - 190, y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _save_screenshot(self, frame):
        """Save screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drowsiness_screenshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f" Screenshot saved: {filename}")
    
    def _toggle_recording(self, frame):
        """Toggle video recording"""
        if self.recording:
            if self.video_writer:
                self.video_writer.release()
            self.recording = False
            print("  Recording stopped")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drowsiness_video_{timestamp}.mp4"
            
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, self.fps, (w, h))
            
            self.recording = True
            print(f"🔴 Recording started: {filename}")
    
    def _print_summary(self):
        """Print test summary"""
        if not self.metrics_history:
            print("  No metrics collected")
            return
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        scores = [m.get("drowsiness_score", 0) for m in self.metrics_history]
        states = [m.get("state", "unknown") for m in self.metrics_history]
        ears = [m.get("ear_mean", 0) for m in self.metrics_history]
        blinks = [m.get("blink_rate", 0) for m in self.metrics_history]
        closures = [m.get("eye_closure_duration", 0) for m in self.metrics_history]
        
        print(f"Total frames: {len(self.metrics_history)}")
        print(f"Duration: {len(self.metrics_history) / self.fps:.1f} seconds")
        print()
        
        print("Drowsiness Score:")
        print(f"  Min: {np.min(scores):.3f}")
        print(f"  Max: {np.max(scores):.3f}")
        print(f"  Mean: {np.mean(scores):.3f}")
        print(f"  Std: {np.std(scores):.3f}")
        print()
        
        print("State Distribution:")
        for state in ["alert", "drowsy", "critical"]:
            count = sum(1 for s in states if s == state)
            pct = 100 * count / len(states)
            print(f"  {state.upper()}: {count} frames ({pct:.1f}%)")
        print()
        
        print("Eye Metrics:")
        print(f"  EAR - Mean: {np.mean(ears):.3f}, Min: {np.min(ears):.3f}, Max: {np.max(ears):.3f}")
        print(f"  Blink Rate - Mean: {np.mean(blinks):.1f} blinks/min")
        print(f"  Max Closure: {np.max(closures):.2f} seconds")
        print()
        
        if self.enable_logging:
            print(f" Detailed logs saved to: {self.log_filename}")
        
        print("="*60)


def main():
    """Main entry point"""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     P4 Temporal Analysis - Live Webcam Testing             ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    # Create tester
    tester = LiveDrowsinessTest(fps=30, enable_logging=True)
    
    # Run
    tester.run()


if __name__ == "__main__":
    main()