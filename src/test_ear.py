#src/test_ear.py
import cv2
import sys
sys.path.insert(0, '.')
from src.dlib_ear import EARDetector
import time
detector = EARDetector(ear_threshold=0.2)
video_path = "data/test_blink_video.mov"
video_path2="data/video_test2.mov"
cap = cv2.VideoCapture(video_path2)#to test with video
cap1=cv2.VideoCapture(0)#to test with webcam
from src.segmentation import FaceSegmentor
p1 = FaceSegmentor()

while True:
    start = time.time()
    

    ret, frame = cap.read()
    if not ret:
        break
    
    r1 = p1.process(frame)
    if r1["success"]:
        r3 = detector.process(r1["face_crop"], r1["bbox"])
        elapsed = (time.time() - start) * 1000
        
        if r3["success"]:
            print(f"EAR: {r3['ear']:.3f} | Blinks: {r3['blink_count']} | Detected: {r3['blink_detected']}| Time: {elapsed:.2f}ms")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
