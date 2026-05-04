import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.segmentation import FaceSegmentor
from src.eye_hog import EyeStateClassifier

# Change this to any image of a face you have
IMAGE_PATH = "data/test_closed.jpg" 

frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print("Could not load image, check the path")
    exit()

p1 = FaceSegmentor()
p2 = EyeStateClassifier()

r1 = p1.process(frame)
if not r1["success"]:
    print("P1 failed to detect face")
    exit()

print("P1 succeeded, face detected")

r2 = p2.process(r1["face_crop"])
if not r2["success"]:
    print("P2 failed to detect eyes")
    exit()

print(f"Eye state: {r2['eye_state']}")
print(f"Confidence: {r2['confidence']}")

# Resize for display
def show_resized(title, img, width=400):
    h, w = img.shape[:2]
    scale = width / w
    resized = cv2.resize(img, (width, int(h * scale)))
    cv2.imshow(title, resized)

show_resized("P1 - Face", r1["debug_frame"], width=600)
show_resized("P2 - Eyes", r2["debug_frame"], width=400)
cv2.waitKey(0)
cv2.destroyAllWindows()