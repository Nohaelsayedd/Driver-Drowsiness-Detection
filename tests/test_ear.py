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

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(f"Cannot open video: {VIDEO_PATH}")
    exit()

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    r1 = p1.process(frame)
    if r1["success"]:
        r3 = p3.process(r1["face_crop"], r1["bbox"])
        elapsed = (time.time() - t0) * 1000
        if r3["success"]:
            print(f"EAR: {r3['ear']:.3f}  Blinks: {r3['blink_count']}  "
                  f"Detected: {r3['blink_detected']}  Time: {elapsed:.1f}ms")

            # draw face box
            x, y, w, h = r1["bbox"]
            color = (0, 0, 255) if r3["blink_detected"] else (0, 255, 0)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

            # draw eye landmarks
            for pt in list(r3["left_eye_pts"]) + list(r3["right_eye_pts"]):
                cv2.circle(display, (int(pt[0]) + x, int(pt[1]) + y), 2, (0, 255, 255), -1)

            # overlay EAR and blink count
            cv2.putText(display, f"EAR: {r3['ear']:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Blinks: {r3['blink_count']}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(display, "No landmarks", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(display, "No face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("P1+P3 EAR Test", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
