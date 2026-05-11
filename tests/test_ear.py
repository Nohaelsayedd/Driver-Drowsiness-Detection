"""
P1 + P3 EAR test on a video file — run from project root: python tests/test_ear.py
Place a video at data/test_blink_video.mov before running. Press Q to quit.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import cv2
import time
from segmentation import FaceSegmentor
from dlib_ear import EARDetector

VIDEO_PATH = "data/test_blink_video.mov"

p1 = FaceSegmentor()
p3 = EARDetector(ear_threshold=0.2)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Cannot open video: {VIDEO_PATH}")
    exit()

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    r1 = p1.process(frame)
    if r1["success"]:
        r3 = p3.process(r1["face_crop"], r1["bbox"])
        elapsed = (time.time() - t0) * 1000
        if r3["success"]:
            print(f"EAR: {r3['ear']:.3f}  Blinks: {r3['blink_count']}  "
                  f"Detected: {r3['blink_detected']}  Time: {elapsed:.1f}ms")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
