# src/dlib_ear.py
import dlib
import numpy as np
import cv2
from scipy.spatial import distance

# Load dlib predictor once
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

class EARDetector:
    def __init__(self, ear_threshold=0.22, blink_frames=3):
        self.ear_threshold = ear_threshold
        self.blink_frames = blink_frames
        self.ear_history = []
        self.blink_count = 0
        self.in_blink = False
    
    def extract_eye_landmarks(self, face_crop):
        """Extract left and right eye landmarks from face crop"""
        dets = detector(face_crop, 1)
        if len(dets) == 0:
            return None, None
        
        shape = predictor(face_crop, dets[0])
        landmarks = np.array([(p.x, p.y) for p in shape.parts()])
        
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        return left_eye, right_eye
    
    def calculate_ear(self, eye_pts):
        """
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        A = distance.euclidean(eye_pts[1], eye_pts[5])
        B = distance.euclidean(eye_pts[2], eye_pts[4])
        C = distance.euclidean(eye_pts[0], eye_pts[3])
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def detect_blink(self, ear_current):
        """Track blink by monitoring EAR dips below threshold"""
        self.ear_history.append(ear_current)
        if len(self.ear_history) > 10:
            self.ear_history.pop(0)
        
        blink_detected = False
        
        # Only count transition from above to below threshold
        if len(self.ear_history) >= 2:
            prev_ear = self.ear_history[-2]
            if prev_ear >= self.ear_threshold and ear_current < self.ear_threshold:
                self.in_blink = True
                blink_detected = True
                self.blink_count += 1
            elif ear_current >= self.ear_threshold:
                self.in_blink = False
        
        return blink_detected
    
    def estimate_head_pose(self, landmarks):
        """Estimate yaw/pitch from landmark geometry (optional bonus)"""
        # Simplified: use nose to face center distance
        nose = landmarks[30]
        left_eye = landmarks[36]
        right_eye = landmarks[45]
        
        face_center_x = (left_eye[0] + right_eye[0]) / 2.0
        yaw = nose[0] - face_center_x
        
        # Simplified pitch from vertical distances
        pitch = nose[1] - ((left_eye[1] + right_eye[1]) / 2.0)
        
        return yaw, pitch
    
    def process(self, face_crop, frame_bbox):
        """Main processing function"""
        left_eye, right_eye = self.extract_eye_landmarks(face_crop)
        
        if left_eye is None:
            return {"success": False}
        
        ear_left = self.calculate_ear(left_eye)
        ear_right = self.calculate_ear(right_eye)
        ear_avg = (ear_left + ear_right) / 2.0
        
        blink_detected = self.detect_blink(ear_avg)
        
        landmarks_full = None
        yaw, pitch = 0, 0
        dets = detector(face_crop, 1)
        if len(dets) > 0:
            shape = predictor(face_crop, dets[0])
            landmarks_full = np.array([(p.x, p.y) for p in shape.parts()])
            yaw, pitch = self.estimate_head_pose(landmarks_full)
        
        return {
            "success": True,
            "ear": ear_avg,
            "ear_left": ear_left,
            "ear_right": ear_right,
            "blink_detected": blink_detected,
            "blink_count": self.blink_count,
            "head_pose": (yaw, pitch),
            "left_eye_pts": left_eye,
            "right_eye_pts": right_eye
        }