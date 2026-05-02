import cv2
import numpy as np

# --- Skin Color Constants ---

# HSV: We define the skin color range using "Hue"
# Range 1: Covers standard and light skin tones
SKIN_LOWER_1 = np.array([0, 20, 70], dtype=np.uint8)
SKIN_UPPER_1 = np.array([20, 255, 255], dtype=np.uint8)

# Range 2: Covers reddish skin tones (since skin hue wraps around the HSV spectrum)
SKIN_LOWER_2 = np.array([170, 20, 70], dtype=np.uint8)
SKIN_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)

# YCrCb: A second color space that is much more stable when lighting changes
YCRCB_LOWER = np.array([0, 133, 77], dtype=np.uint8)
YCRCB_UPPER = np.array([255, 173, 127], dtype=np.uint8)

# Tools for cleaning up the binary mask (Morphology)
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
BBOX_PADDING = 0.15 
MIN_FACE_AREA_RATIO = 0.02

class FaceSegmentor:
    """
    This class handles the face extraction pipeline:
    1. Detects the face using Haar Cascades.
    2. Uses temporal smoothing to avoid flickering if detection fails for a frame.
    3. Validates that the detected area is actually "Skin" to avoid false positives.
    4. Crops and resizes the face for the next team members (P2/P3).
    """

    def __init__(self, scale_factor=1.05, min_neighbors=3, target_size=(224, 224)):
        # Load the pre-trained face detector model
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if self.face_cascade.empty():
            raise RuntimeError("Could not find the Haar Cascade XML file.")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.target_size = target_size
        self._last_bbox = None  # Stores the last known face position
        self._miss_counter = 0  # Counts how many consecutive frames the face was missing
        self.MAX_MISS_FRAMES = 10

    def process(self, frame: np.ndarray) -> dict:
        # Dictionary to store results for the rest of the team
        result = {
            "success": False,
            "face_crop": None,
            "bbox": None,
            "skin_mask": None,
            "debug_frame": None,
        }

        if frame is None or frame.size == 0:
            return result

        # Convert to Grayscale and normalize lighting for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # --- STEP 1: Detect Faces ---
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # --- STEP 2: Handle Detection Failures (Temporal Smoothing) ---
        if len(faces) == 0:
            # If we missed the face briefly, reuse the last known position
            if self._last_bbox is not None and self._miss_counter < self.MAX_MISS_FRAMES:
                self._miss_counter += 1
                x, y, w, h = self._last_bbox
            else:
                self._last_bbox = None
                return result
        else:
            # Face found: Reset the counter and pick the largest detected face
            self._miss_counter = 0
            face = self._largest_face(faces, frame.shape)
            if face is None:
                return result
            x, y, w, h = face

        # --- STEP 3: Generate Skin Mask ---
        skin_mask = self._skin_mask(frame)
        result["skin_mask"] = skin_mask

        # --- STEP 4: Skin Validation ---
        # If the detected box has less than 15% skin, it's likely a false detection
        if not self._validate_skin(skin_mask, x, y, w, h):
            return result

        # --- STEP 5: Apply Padding ---
        # Add 15% space around the face so P3 (dlib landmarks) has room to work
        x, y, w, h = self._pad_bbox(x, y, w, h, frame.shape)
        self._last_bbox = (x, y, w, h)
        result["bbox"] = (x, y, w, h)

        # --- STEP 6: Crop and Resize ---
        face_crop = frame[y:y+h, x:x+w]
        face_crop = cv2.resize(face_crop, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        result["face_crop"] = face_crop
        result["success"] = True

        # --- STEP 7: Debug Visualization ---
        result["debug_frame"] = self._draw_debug(frame.copy(), x, y, w, h, skin_mask)

        return result

    # --- Internal Helper Functions ---

    def _skin_mask(self, frame):
        # Apply blur to reduce pixel-level noise
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Build masks for both HSV and YCrCb color spaces
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, SKIN_LOWER_1, SKIN_UPPER_1)
        m2 = cv2.inRange(hsv, SKIN_LOWER_2, SKIN_UPPER_2)
        hsv_mask = cv2.bitwise_or(m1, m2)

        ycrcb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
        ycrcb_mask = cv2.inRange(ycrcb, YCRCB_LOWER, YCRCB_UPPER)

        # Pixels must pass both color space checks to be considered skin
        combined = cv2.bitwise_and(hsv_mask, ycrcb_mask)

        # Cleanup: Remove small noise and fill internal holes in the face mask
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=2)
        
        return combined

    def _validate_skin(self, mask, x, y, w, h, threshold=0.15):
        # Calculates the percentage of skin pixels within the bounding box
        roi = mask[y:y+h, x:x+w]
        skin_ratio = np.count_nonzero(roi) / (w * h + 1e-6)
        return skin_ratio >= threshold

    def _largest_face(self, faces, frame_shape):
        # Pick the largest bounding box (assumed to be the driver/main user)
        f_area = frame_shape[0] * frame_shape[1]
        best = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = best
        
        # Reject if the face is too small relative to the frame size
        if (w * h) / f_area < MIN_FACE_AREA_RATIO:
            return None
        return int(x), int(y), int(w), int(h)

    def _pad_bbox(self, x, y, w, h, frame_shape):
        # Expands the box by 15% and ensures it stays within frame boundaries
        fh, fw = frame_shape[:2]
        px, py = int(w * BBOX_PADDING), int(h * BBOX_PADDING)
        
        x_new = max(0, x - px)
        y_new = max(0, y - py)
        w_new = min(fw - x_new, w + 2 * px)
        h_new = min(fh - y_new, h + 2 * py)
        
        return x_new, y_new, w_new, h_new

    def _draw_debug(self, frame, x, y, w, h, mask):
        # Draws a green rectangle and the skin percentage for visual testing
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = mask[y:y+h, x:x+w]
        pct = int(100 * np.count_nonzero(roi) / (w * h + 1e-6))
        cv2.putText(frame, f"Face | Skin: {pct}%", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame