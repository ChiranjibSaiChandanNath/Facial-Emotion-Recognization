import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import tensorflow as tf

# ─── Model & Image Path ──────────────────────────────────────────────────────
MODEL_PATH  = 'model_file.h5'
IMAGE_PATH  = sys.argv[1] if len(sys.argv) > 1 else 'img2.jpeg'

# ─── Load Model ──────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file '{MODEL_PATH}' not found.")
    print("        Please run main.py first to train the model.")
    sys.exit(1)

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"[INFO] Model loaded from '{MODEL_PATH}'")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

# ─── Load Image ──────────────────────────────────────────────────────────────
if not os.path.exists(IMAGE_PATH):
    print(f"[ERROR] Image '{IMAGE_PATH}' not found.")
    sys.exit(1)

frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print(f"[ERROR] Could not read image '{IMAGE_PATH}'. File may be corrupt.")
    sys.exit(1)

print(f"[INFO] Testing on image: {IMAGE_PATH}")

# ─── Load Face Detector ──────────────────────────────────────────────────────
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

if not os.path.exists(CASCADE_PATH):
    print(f"[ERROR] Haar cascade file '{CASCADE_PATH}' not found.")
    sys.exit(1)

faceDetect = cv2.CascadeClassifier(CASCADE_PATH)

# ─── Labels & Colors ─────────────────────────────────────────────────────────
labels_dict = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

color_dict = {
    0: (0,   0,   255),   # Angry    → Red
    1: (0,   140, 255),   # Disgust  → Orange
    2: (0,   255, 255),   # Fear     → Yellow
    3: (0,   255, 0),     # Happy    → Green
    4: (255, 255, 255),   # Neutral  → White
    5: (255, 0,   0),     # Sad      → Blue
    6: (255, 0,   255),   # Surprise → Magenta
}

# ─── Detection & Prediction ──────────────────────────────────────────────────
gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

if len(faces) == 0:
    print("[WARNING] No faces detected in the image.")
else:
    print(f"[INFO] Detected {len(faces)} face(s).")

for (x, y, w, h) in faces:
    try:
        sub_face_img = gray[y:y+h, x:x+w]
        resized      = cv2.resize(sub_face_img, (48, 48))
        normalized   = resized / 255.0
        reshaped     = np.reshape(normalized, (1, 48, 48, 1))

        result = model.predict(reshaped, verbose=0)
        label  = int(np.argmax(result, axis=1)[0])
        conf   = float(np.max(result) * 100)

        color = color_dict.get(label, (255, 255, 255))
        text  = f"{labels_dict[label]} ({conf:.1f}%)"

        print(f"[RESULT] Emotion: {labels_dict[label]} | Confidence: {conf:.1f}%")

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        # Draw label background
        cv2.rectangle(frame, (x, y-45), (x+w, y), color, -1)
        # Draw label text
        cv2.putText(frame, text, (x+4, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    except Exception as e:
        print(f"[WARNING] Prediction error for face at ({x},{y}): {e}")
        continue

# ─── Show Result ─────────────────────────────────────────────────────────────
window_title = f"Emotion Detection — {os.path.basename(IMAGE_PATH)}"
cv2.imshow(window_title, frame)
print("[INFO] Press any key to close the window.")
cv2.waitKey(0)
cv2.destroyAllWindows()