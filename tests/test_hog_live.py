"""
P1 + P2 live webcam test — run from project root: python tests/test_hog_live.py
Green box = eyes open, Red box = eyes closed. Press Q to quit.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import cv2
from segmentation import FaceSegmentor
from eye_hog import EyeStateClassifier

p1 = FaceSegmentor()
p2 = EyeStateClassifier()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    r1 = p1.process(frame)

    if not r1["success"]:
        cv2.putText(display, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        r2 = p2.process(r1["face_crop"])
        x, y, w, h = r1["bbox"]
        if r2["success"]:
            color = (0, 255, 0) if r2["eye_state"] == "open" else (0, 0, 255)
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
            cv2.putText(display, f"{r2['eye_state']} {r2['confidence']:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.rectangle(display, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.putText(display, "Eyes not detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("P1+P2 Test", cv2.resize(display, (800, 600)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
