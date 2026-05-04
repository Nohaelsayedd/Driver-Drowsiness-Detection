import cv2
import numpy as np
import joblib
from skimage.feature import hog

MODEL_PATH = "models/eye_hog_svm.pkl"
EYE_SIZE = (64, 32)  # width, height

class EyeStateClassifier:
    """
    Loads the trained SVM model and classifies eye state from a face crop.
    Input: face_crop (numpy array) from P1
    Output: dict with eye_state and confidence
    """

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.left_eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml"
        )
        self.right_eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_righteye_2splits.xml"
        )

    def process(self, face_crop: np.ndarray) -> dict:
        result = {
            "success": False,
            "eye_state": None,  # "open" or "closed"
            "confidence": 0.0,
            "debug_frame": None,
        }

        if face_crop is None:
            return result

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # Detect eyes
        left_eyes = self.left_eye_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))
        right_eyes = self.right_eye_cascade.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))

        eyes = []
        for (x, y, w, h) in left_eyes[:1]:
            eyes.append(gray[y:y+h, x:x+w])
        for (x, y, w, h) in right_eyes[:1]:
            eyes.append(gray[y:y+h, x:x+w])

        if len(eyes) == 0:
            return result

        # Classify each eye and average the confidence
        predictions = []
        confidences = []
        for eye in eyes:
            features = self._extract_hog(eye)
            pred = self.model.predict([features])[0]
            proba = self.model.predict_proba([features])[0]
            predictions.append(pred)
            confidences.append(max(proba))

        # If any eye is closed, state is closed
        final_state = "closed" if "closed" in predictions else "open"
        final_confidence = float(np.mean(confidences))

        result["success"] = True
        result["eye_state"] = final_state
        result["confidence"] = round(final_confidence, 3)
        result["debug_frame"] = self._draw_debug(face_crop.copy(), left_eyes, right_eyes, final_state, final_confidence)

        return result

    def _extract_hog(self, eye_img):
        resized = cv2.resize(eye_img, EYE_SIZE)
        resized = cv2.equalizeHist(resized)
        features = hog(
            resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )
        return features

    def _draw_debug(self, frame, left_eyes, right_eyes, state, confidence):
        color = (0, 255, 0) if state == "open" else (0, 0, 255)
        for (x, y, w, h) in left_eyes[:1]:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, "L", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        for (x, y, w, h) in right_eyes[:1]:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, "R", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(frame, f"{state} ({confidence:.2f})", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame