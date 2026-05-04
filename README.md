# Driver Drowsiness Detection
> Working README for team coordination. Will be replaced with proper documentation after full implementation.

---

## Setup (everyone do this once)
1. Clone the repo
   git clone https://github.com/Nohaelsayedd/Driver-Drowsiness-Detection.git

2. Go into the folder
   cd Driver-Drowsiness-Detection

3. Create a virtual environment
   python -m venv venv

4. Activate it
   Windows: venv\Scripts\activate
   Mac/Linux: source venv/bin/activate

5. Install dependencies
   pip install -r requirements.txt

---

## Rules
- Only work inside your own file in src/
- Never push directly to main, create a branch first: git checkout -b your-name-p3
- Put model files (.pkl, .dat) in models/ but check with the team before pushing large files
- Never push the data/ folder or your venv/
- Always run scripts from the project root, not from inside src/

---

## Project Status

| Part | File | Owner | Status |
|------|------|-------|--------|
| P1 | src/segmentation.py | [name] | Done |
| P2 | src/eye_hog.py | Noha | Done |
| P3 | src/eye_ear.py | [name] | Not started |
| P4 | src/analysis.py | [name] | Not started |
| P5 | main.py | [name] | Not started |

---

## What has been implemented

### P1 - Face Segmentation (src/segmentation.py)
Detects and crops the face from a frame using Haar Cascade.
Validates the detection using skin color in both HSV and YCrCb color spaces.
Uses temporal smoothing so if the face is missed for a few frames it reuses the last known position.

Output dictionary:
- success: True or False
- face_crop: cropped face image, resized to 224x224, numpy array
- bbox: (x, y, w, h) position of the face in the original frame
- skin_mask: binary mask of detected skin pixels
- debug_frame: original frame with green box drawn around the face

No model files needed, uses opencv built-in Haar Cascade.

---

### P2 - HOG Eye State Classifier (src/eye_hog.py)
Takes the face_crop from P1 and detects whether the eyes are open or closed.
Finds the eyes inside the face crop using Haar Cascade.
Extracts HOG (Histogram of Oriented Gradients) features from each eye.
Feeds the features into a trained SVM model to classify the eye state.
If either eye is detected as closed, the overall state is closed.

Output dictionary:
- success: True or False
- eye_state: "open" or "closed"
- confidence: float between 0.0 and 1.0
- debug_frame: face crop with colored boxes around detected eyes (green = open, red = closed)

Model file needed: models/eye_hog_svm.pkl (already in the repo, no retraining needed)
Trained on: https://www.kaggle.com/datasets/arindamxd/eyes-open-closed-dataset

Known limitation: accuracy drops in low lighting conditions. This is expected.
P3's EAR method is more reliable for lighting, P2 is one signal among many.

---

## How to use P1 and P2 in your code (P3, P4, P5 read this)

from src.segmentation import FaceSegmentor
from src.eye_hog import EyeStateClassifier

p1 = FaceSegmentor()
p2 = EyeStateClassifier()

# frame is a standard opencv BGR image from cv2.VideoCapture
r1 = p1.process(frame)

if r1["success"]:
    face_crop = r1["face_crop"]   # pass this to p2 and p3
    bbox = r1["bbox"]             # (x, y, w, h) in original frame

    r2 = p2.process(face_crop)
    if r2["success"]:
        print(r2["eye_state"])    # "open" or "closed"
        print(r2["confidence"])   # 0.0 to 1.0

Always check success before using the output. If success is False, skip that frame.

---

## Testing files (P2)

### Test on a static image (src/test_hog.py)
Tests P1 and P2 together on a single photo.
Put a face photo in data/ and name it test_face.jpg, then run:
python src/test_hog.py
Two windows will open showing the detected face and the eye state with confidence.

### Test on live webcam (src/test_hog_live.py)
Opens your webcam and runs P1 and P2 in real time.
Shows one window with a colored box around your face.
Green box = eyes open, Red box = eyes closed.
Run:
python src/test_hog_live.py
Press Q to quit.

### Retrain the model (src/train_hog.py)
Only needed if you want to retrain from scratch.
Download the dataset from https://www.kaggle.com/datasets/arindamxd/eyes-open-closed-dataset
Put it in data/dataset/ so the structure is data/dataset/train/open, data/dataset/train/closed, etc.
Then run:
python src/train_hog.py
The new model will be saved to models/eye_hog_svm.pkl