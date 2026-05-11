"""
P4 live webcam test — run from project root: python tests/test_analysis_live.py
Uses real P1/P2/P3 if available, otherwise falls back to mock mode.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import cv2
import numpy as np
import time
from collections import deque
from datetime import datetime
import csv

from analysis import TemporalAnalyzer, DrowsinessState

try:
    from segmentation import FaceSegmentor
    from eye_hog import EyeStateClassifier
    from dlib_ear import EARDetector
except ImportError:
    print("Warning: P1-P3 modules not found. Using MOCK MODE.\n")

    class FaceSegmentor:
        def process(self, frame):
            return {"success": True, "face_crop": cv2.resize(frame, (224, 224)),
                    "bbox": (50, 50, 200, 200), "skin_mask": None, "debug_frame": frame.copy()}

    class EyeStateClassifier:
        def process(self, face_crop):
            return {"success": True, "eye_state": "open", "confidence": 0.9, "debug_frame": face_crop.copy()}

    class EARDetector:
        def __init__(self):
            self._n = 0
        def process(self, face_crop, bbox):
            self._n += 1
            ear = 0.10 if self._n % 30 == 15 else 0.35 + np.random.normal(0, 0.03)
            return {
                "success": True, "ear": float(np.clip(ear, 0, 0.5)),
                "ear_left": ear, "ear_right": ear,
                "blink_detected": self._n % 30 == 15,
                "blink_count": self._n // 30,
                "head_pose": (np.random.normal(0, 5), np.random.normal(0, 5)),
                "left_eye_pts": np.zeros((6, 2)), "right_eye_pts": np.zeros((6, 2)),
            }


class LiveDrowsinessTest:

    def __init__(self, fps=30):
        self.fps          = fps
        self.frame_count  = 0
        self.frame_times  = deque(maxlen=30)
        self.metrics      = deque(maxlen=300)

        self.segmentor  = FaceSegmentor()
        self.hog        = EyeStateClassifier()
        self.ear        = EARDetector()
        self.analyzer   = TemporalAnalyzer(fps=fps)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = f"drowsiness_log_{ts}.csv"
        self._csv = open(self.log_path, 'w', newline='')
        self._writer = csv.writer(self._csv)
        self._writer.writerow(["frame","timestamp","drowsiness_score","state",
                               "ear","blink_rate","closure_duration"])
        print(f"Logging to {self.log_path}")

    def _process(self, frame):
        r1 = self.segmentor.process(frame)
        if not r1["success"]:
            return None
        r2 = self.hog.process(r1["face_crop"])
        r3 = self.ear.process(r1["face_crop"], r1["bbox"])
        return self.analyzer.process(r3, r2)

    def _draw(self, frame, result):
        out = frame.copy()
        h, w = out.shape[:2]
        if result and result.get("success"):
            state = result["state"]
            color = {"alert": (0,200,0), "drowsy": (0,200,255), "critical": (0,0,255)}.get(state, (128,128,128))
            cv2.rectangle(out, (10,10), (200,90), color, -1)
            cv2.putText(out, state.upper(), (20,55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 2)
            cv2.putText(out, f"Score: {result['drowsiness_score']:.2f}", (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(out, f"EAR: {result['ear_mean']:.3f}", (210,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
            cv2.putText(out, f"Blinks: {result['blink_rate']:.0f}/min", (210,55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
        if len(self.frame_times) > 1:
            dt  = self.frame_times[-1] - self.frame_times[0]
            fps = (len(self.frame_times)-1)/dt if dt > 0 else 0
            cv2.putText(out, f"FPS: {fps:.1f}", (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 1)
        return out

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Cannot open camera.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        print("Running — q/ESC quit | c clear metrics")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_count += 1
            self.frame_times.append(time.time())
            result = self._process(frame)
            if result and result.get("success"):
                self.metrics.append(result)
                self._writer.writerow([
                    self.frame_count, datetime.now().isoformat(),
                    result["drowsiness_score"], result["state"],
                    result["ear_mean"], result["blink_rate"], result["eye_closure_duration"],
                ])
                self._csv.flush()
            out = self._draw(frame, result)
            cv2.imshow("P4 Live Test", out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('c'):
                self.analyzer.reset()
                self.metrics.clear()

        cap.release()
        cv2.destroyAllWindows()
        self._csv.close()
        self._print_summary()

    def _print_summary(self):
        if not self.metrics:
            return
        scores = [m["drowsiness_score"] for m in self.metrics]
        states = [m["state"] for m in self.metrics]
        print(f"\nFrames: {len(self.metrics)}  ({len(self.metrics)/self.fps:.1f}s)")
        print(f"Score — mean: {np.mean(scores):.3f}, max: {np.max(scores):.3f}")
        for s in ["alert","drowsy","critical"]:
            n = sum(1 for x in states if x == s)
            print(f"  {s.upper()}: {n} frames ({100*n/len(states):.1f}%)")
        print(f"Log: {self.log_path}")


if __name__ == "__main__":
    LiveDrowsinessTest().run()
