import cv2
import numpy as np
import joblib
import os
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import classification_report

TRAIN_DIR = "data/dataset/train"
TEST_DIR  = "data/dataset/test"
MODEL_PATH = "models/eye_hog_svm.pkl"
EYE_SIZE = (64, 32)

def load_dataset(folder):
    X, y = [], []
    for label in ["open", "closed"]:
        path = os.path.join(folder, label)
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, EYE_SIZE)
            img = cv2.equalizeHist(img)
            features = hog(
                img,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
            )
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

print("Loading training data...")
X_train, y_train = load_dataset(TRAIN_DIR)

print("Loading test data...")
X_test, y_test = load_dataset(TEST_DIR)

print(f"Train: {len(X_train)} images | Test: {len(X_test)} images")

print("Training SVM...")
model = SVC(kernel="rbf", probability=True, C=10, gamma="scale")
model.fit(X_train, y_train)

print("Evaluating...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")