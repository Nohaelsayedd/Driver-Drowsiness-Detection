# Driver Drowsiness Detection

A real-time computer vision system that detects driver drowsiness using a webcam. The system monitors eye state, blink rate, eye closure duration, and head pose to produce a continuous drowsiness score and trigger alerts when the driver becomes drowsy or unresponsive.

---

## How It Works

The system is a 5-stage pipeline where each part feeds into the next:

| Part | Module | Description | Output |
|------|--------|-------------|--------|
| P1 | `src/segmentation.py` | Detects and crops the face using Haar Cascade. Validates using skin color (HSV + YCrCb). Applies temporal smoothing to handle missed frames. | `face_crop`, `bbox` |
| P2 | `src/eye_hog.py` | Extracts HOG features from each eye and classifies open/closed using a trained SVM. | `eye_state`, `confidence` |
| P3 | `src/dlib_ear.py` | Computes Eye Aspect Ratio (EAR) using dlib 68-point landmarks. Detects blinks and estimates head pose (yaw/pitch). | `ear`, `blink_count`, `head_pose` |
| P4 | `src/analysis.py` | Fuses all signals over time into a drowsiness score (0–1). Runs a state machine: Alert → Drowsy → Critical. | `drowsiness_score`, `state` |
| P5 | `src/pipeline.py` | Connects all parts in a real-time webcam loop. Draws the overlay, triggers audio alerts, and targets 30 FPS. | Live video window |

### Drowsiness Score Signals

The P4 score is a weighted combination of 5 signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| EAR score | 25% | Low EAR → high drowsiness |
| Blink rate | 25% | < 8 blinks/min is drowsy |
| Eye state (HOG) | 20% | HOG classifier agreement |
| Sustained closure | 20% | Eyes closed > 0.5s is drowsy, > 2s is critical |
| Head pose | 10% | Nodding/yawing detection |

---

## Setup

### 1. Clone the repo
```
git clone https://github.com/Nohaelsayedd/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection
```

### 2. Create and activate a virtual environment
```
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Set up model files

**P2 model** (HOG SVM) — already in the repo at `models/eye_hog_svm.pkl`.

**P3 model** (dlib landmarks) — the compressed file is included. Extract it before running:
```
python -c "import bz2; open('models/shape_predictor_68_face_landmarks.dat','wb').write(bz2.open('models/shape_predictor_68_face_landmarks.dat.bz2','rb').read())"
```

If the `.dat` file is missing and the `.bz2` is not available, download it from:
```
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```
Place it in the `models/` folder and run the extraction command above.

> **Note:** If the dlib model is not found, the system automatically falls back to using the HOG classifier (P2) alone for blink and EAR synthesis. Drowsiness detection still works but is less precise.

---

## Running the System

Always run from the project root directory.

### Full pipeline (main entry point)
```
python main.py
```

### Controls (while the window is open)

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `r` | Reset the temporal analyzer (clears history) |
| `s` | Save a screenshot |

### Alert behaviour

| State | Visual | Audio |
|-------|--------|-------|
| Alert | Green border + green badge | — |
| Drowsy | Orange border + yellow badge | 800 Hz beep every 3.5s |
| Critical | Red border + red badge | 1300 Hz beep every 1.2s |

---

## Project Structure

```
Driver-Drowsiness-Detection/
├── main.py                 ← Run this
├── requirements.txt
├── .gitignore
├── README.md
│
├── models/
│   ├── eye_hog_svm.pkl                            ← P2 trained SVM model
│   └── shape_predictor_68_face_landmarks.dat.bz2  ← P3 dlib model (extract first)
│
├── src/
│   ├── segmentation.py     ← P1: Face detection & skin validation
│   ├── eye_hog.py          ← P2: HOG eye state classifier
│   ├── dlib_ear.py         ← P3: EAR blink detection & head pose
│   ├── analysis.py         ← P4: Temporal drowsiness scoring
│   ├── pipeline.py         ← P5: Real-time integration
│   └── train_hog.py        ← Retrain the P2 SVM model (optional)
│
└── tests/
    ├── test_analysis.py     ← P4 unit tests (11 tests, run offline)
    ├── test_analysis_live.py← P4 live webcam test with CSV logging
    ├── test_hog.py          ← P1+P2 test on a static image
    ├── test_hog_live.py     ← P1+P2 live webcam test
    └── test_ear.py          ← P1+P3 test on a video file
```

---

## Running Tests

All tests are run from the project root:

```
# P4 unit tests (no camera needed)
python tests/test_analysis.py

# P1+P2 live webcam test
python tests/test_hog_live.py

# P1+P2 static image test (needs data/test_closed.jpg)
python tests/test_hog.py

# P4 live webcam test with CSV logging
python tests/test_analysis_live.py

# P1+P3 video test (needs data/test_blink_video.mov)
python tests/test_ear.py
```

---

## Retraining the HOG Model

If you want to retrain the P2 eye classifier from scratch:

1. Download the dataset: https://www.kaggle.com/datasets/arindamxd/eyes-open-closed-dataset
2. Place it at `data/dataset/` so the structure is:
   ```
   data/dataset/train/open/
   data/dataset/train/closed/
   data/dataset/test/open/
   data/dataset/test/closed/
   ```
3. Run:
   ```
   python src/train_hog.py
   ```
   The new model is saved to `models/eye_hog_svm.pkl`.

---

## Team

| Part | Owner |
|------|-------|
| P1 — Segmentation | Rawan |
| P2 — HOG Classifier | Noha |
| P3 — dlib EAR | Carol |
| P4 — Temporal Analysis | Hams |
| P5 — Integration | Jana |
