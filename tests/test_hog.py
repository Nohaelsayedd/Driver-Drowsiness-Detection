"""
P1 + P2 static image test — run from project root: python tests/test_hog.py
Place a face image at data/test_closed.jpg before running.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import cv2
from segmentation import FaceSegmentor
from eye_hog import EyeStateClassifier

IMAGE_PATH = "data/closed.jpg"

frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print(f"Could not load image: {IMAGE_PATH}")
    exit()

p1 = FaceSegmentor()
p2 = EyeStateClassifier()

r1 = p1.process(frame)
if not r1["success"]:
    print("P1 failed — no face detected")
    exit()
print("P1: face detected")

r2 = p2.process(r1["face_crop"])
if not r2["success"]:
    print("P2 failed — eyes not detected")
    exit()
print(f"P2: eye_state={r2['eye_state']}  confidence={r2['confidence']:.3f}")


def show(title, img, width=400):
    h, w = img.shape[:2]
    cv2.imshow(title, cv2.resize(img, (width, int(h * width / w))))


show("P1 - Face", r1["debug_frame"], 600)
show("P2 - Eyes", r2["debug_frame"], 400)
cv2.waitKey(0)
cv2.destroyAllWindows()
